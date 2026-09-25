"""Integration tests for the one-click Auto-Analyze path.

The automation surface is deliberately thin: raw-IQ format detection, a clearly
flagged sample-rate hypothesis, sync-word discovery, and a burst-derived decode
bound all have to be honest.  These tests exercise the same ``analyze_file``
funnel used by the GUI and the headless entrypoint, rather than testing a
second implementation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from rf_analyzer.core.correlator import (
    FALSE_ALARM_TARGET,
    discover_sync_word,
    hex_to_bits,
)
from rf_analyzer.core.framing import build_frame, modulate
from rf_analyzer.core.io import load_sigmf_meta
from rf_analyzer.pipeline import analyze_file

SAMPLE_RATE = 100_000.0
SYNC_WORD = "0x1ACFFC1D"
SAMPLE_DATA = Path(__file__).resolve().parents[2] / "sample_data"
BPSK_IQ = SAMPLE_DATA / "bpsk.iq"


def _request(path: Path, **overrides) -> dict:
    """Build a deterministic, inexpensive analysis request."""

    request = {
        "file_path": str(path),
        "sample_rate": SAMPLE_RATE,
        "iq_format": "complex64",
        "modulation": "BPSK",
        "sync_word": SYNC_WORD,
        "decode": False,
    }
    request.update(overrides)
    return request


def _copy_bpsk(tmp_path: Path, name: str = "capture.iq") -> Path:
    if not BPSK_IQ.exists():
        pytest.skip(f"missing {BPSK_IQ}; run scripts/generate_test_data.py first")
    target = tmp_path / name
    target.write_bytes(BPSK_IQ.read_bytes())
    return target


def test_iq_auto_detection_reports_the_selected_format(tmp_path):
    path = _copy_bpsk(tmp_path)

    report = analyze_file(_request(path, iq_format="auto"))

    assert report["errors"] == []
    assert report["input"]["iq_format"] == "auto"
    assert report["input"]["iq_format_used"] in {
        "complex64",
        "int16",
        "uint8",
        "int8",
    }
    confidence = report["input"]["iq_format_confidence"]
    assert 0.0 <= float(confidence) <= 1.0


def test_iq_auto_detection_warns_when_confidence_is_low(tmp_path):
    # Repeated byte patterns leave the four raw interpretations nearly tied.
    # The detector is allowed to guess, but it must say that the guess is weak.
    path = tmp_path / "ambiguous.iq"
    path.write_bytes(bytes(range(256)) * 8)

    report = analyze_file(_request(path, iq_format="auto"))

    assert report["errors"] == []
    assert float(report["input"]["iq_format_confidence"]) < 0.6
    assert any(
        "IQ format auto-detection was inconclusive" in warning
        for warning in report["warnings"]
    ), report["warnings"]


def test_explicit_sample_rate_overrides_auto_sample_rate(tmp_path):
    path = _copy_bpsk(tmp_path)

    report = analyze_file(
        _request(
            path,
            sample_rate=73_000,
            auto_sample_rate=True,
        )
    )

    assert report["errors"] == []
    assert report["input"]["sample_rate"] == pytest.approx(73_000.0)
    assert report["input"]["sample_rate_assumed"] is False
    assert report["input"]["sample_rate_source"] == "explicit"
    assert not any(
        "sample_rate was not supplied" in warning for warning in report["warnings"]
    )


def test_auto_sample_rate_is_reported_as_an_hypothesis(tmp_path):
    path = _copy_bpsk(tmp_path)
    request = _request(path)
    request.pop("sample_rate")
    request["auto_sample_rate"] = True

    report = analyze_file(request)

    assert report["errors"] == []
    assert report["input"]["sample_rate_assumed"] is True
    assert report["input"]["sample_rate_source"] == "hypothesis"
    assert report["input"]["sample_rate"] > 0
    assert any(
        "sample_rate was not supplied" in warning and "assumed" in warning
        for warning in report["warnings"]
    ), report["warnings"]


def test_sigmf_metadata_supplies_the_auto_sample_rate(tmp_path):
    path = _copy_bpsk(tmp_path)
    (tmp_path / "capture.sigmf-meta").write_text(
        json.dumps(
            {
                "global": {
                    "core:sample_rate": 123_456,
                    "core:datatype": "cf32_le",
                },
                "captures": [{"core:frequency": 987_000}],
            }
        ),
        encoding="utf-8",
    )

    metadata = load_sigmf_meta(str(path))
    assert metadata is not None
    assert metadata["sample_rate"] == 123_456
    assert metadata["datatype"] == "cf32_le"
    assert metadata["center_frequency"] == 987_000

    request = _request(path)
    request.pop("sample_rate")
    request["auto_sample_rate"] = True
    report = analyze_file(request)

    assert report["errors"] == []
    assert report["input"]["sample_rate"] == pytest.approx(123_456.0)
    assert report["input"]["sample_rate_source"] == "sigmf"
    assert report["input"]["sample_rate_assumed"] is False
    assert not any(
        "sample_rate was not supplied" in warning for warning in report["warnings"]
    )


def test_sync_discovery_uses_cfar_and_finds_a_real_header():
    rng = np.random.default_rng(123)
    received = np.concatenate(
        [
            rng.integers(0, 2, size=400, dtype=np.uint8),
            hex_to_bits(SYNC_WORD),
            rng.integers(0, 2, size=400, dtype=np.uint8),
        ]
    )

    result = discover_sync_word(received)

    assert result["detected"] is True
    assert result["sync_word"] == SYNC_WORD
    assert result["offset"] == 400
    assert result["score"] == pytest.approx(1.0)
    assert result["min_score"] >= 28 / 32


def test_sync_discovery_rejects_a_chance_match_on_a_long_capture():
    rng = np.random.default_rng(321)
    received = rng.integers(0, 2, size=500_000, dtype=np.uint8)

    result = discover_sync_word(received)

    assert result["detected"] is False
    assert result["sync_word"] is None
    assert result["false_alarm_target"] == pytest.approx(FALSE_ALARM_TARGET)
    assert result["candidates_tried"]
    for candidate in result["candidates_tried"]:
        assert candidate["detected"] is False
        assert candidate["min_score"] >= 31 / 32


def test_discovered_sync_is_assumed_but_manual_sync_always_wins(tmp_path):
    # The first call discovers the common project header.  The second capture
    # uses a header outside the discovery candidate list, so discovery cannot
    # silently replace the caller's value.
    common_path = _copy_bpsk(tmp_path, "common.iq")
    common = analyze_file(_request(common_path, sync_word=None, auto_sync=True))
    assert common["errors"] == []
    assert common["correlation"]["sync_word"] == SYNC_WORD
    assert common["correlation"]["assumed"] is True
    assert common["correlation"]["detected"] is True
    assert any("auto-discovery assumed" in warning for warning in common["warnings"])

    custom = build_frame(
        b"manual header must remain authoritative",
        sync_word="0xDEADBEEF",
        fec="none",
        interleaver="none",
    )
    custom_path = tmp_path / "custom.iq"
    modulate(custom["bits"], "BPSK").astype(np.complex64).tofile(custom_path)
    manual = analyze_file(
        _request(
            custom_path,
            sync_word="0xDEADBEEF",
            auto_sync=True,
        )
    )
    assert manual["errors"] == []
    assert manual["correlation"]["sync_word"] == "0xDEADBEEF"
    assert manual["correlation"]["assumed"] is False
    assert manual["correlation"]["detected"] is True


def test_burst_derived_frame_bits_are_exposed_in_decode_search(tmp_path):
    message = b"auto-analyze burst frame hint"
    frame = build_frame(message, fec="conv", interleaver="block")
    core = modulate(frame["bits"], "BPSK")
    rng = np.random.default_rng(7)
    lead = (rng.standard_normal(800) + 1j * rng.standard_normal(800)) * 0.05
    tail = (rng.standard_normal(1_600) + 1j * rng.standard_normal(1_600)) * 0.05
    path = tmp_path / "burst.iq"
    np.concatenate([lead, core, tail]).astype(np.complex64).tofile(path)

    report = analyze_file(_request(path))

    assert report["errors"] == []
    assert report["signal"]["burst"]["found"] is True
    search = report["payload"]["decode_search"]
    assert search["frame_bits"] is not None
    assert search["frame_bits"] > 0
    assert search["frame_bits"] < report["demodulation"]["num_bits"]
    assert any(
        "burst region ends at sample" in warning for warning in report["warnings"]
    )


def test_headless_auto_analysis_smoke_and_strict_json(tmp_path):
    path = _copy_bpsk(tmp_path)
    request = _request(path, iq_format="auto")
    request.pop("sample_rate")
    request["auto_sample_rate"] = True
    request["sync_word"] = None
    request["auto_sync"] = True

    report = analyze_file(request)

    assert report["errors"] == []
    assert report["correlation"]["detected"] is True
    # json_safe/allow_nan=False is the same contract used by the dashboard;
    # this catches accidental NaN/Infinity in additive auto-analysis fields.
    json.dumps(report, allow_nan=False)


# --------------------------------------------------------------------------- #
# GUI wiring: keep this in the same integration file so the automation contract
# is tested at the user-facing boundary as well as in the headless funnel.
# --------------------------------------------------------------------------- #
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt6.QtWidgets", reason="PyQt6 not installed")


def test_gui_auto_button_calls_analyze_file_with_auto_flags(tmp_path, monkeypatch):
    import sys

    from PyQt6.QtWidgets import QApplication

    import rf_analyzer.pipeline as pipeline_module
    from rf_analyzer.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    try:
        path = _copy_bpsk(tmp_path, "gui.iq")
        window.open_file(str(path))
        app.processEvents()

        assert window.auto_analyze_btn.text()
        captured: list[dict] = []

        def fake_analyze(request: dict) -> dict:
            captured.append(dict(request))
            return {
                "input": {},
                "signal": {},
                "modulation": {},
                "demodulation": {"bits_preview": []},
                "correlation": {},
                "payload": {},
                "fec": {},
                "interleaving": {},
                "warnings": [],
                "errors": [],
            }

        # Keep this test focused on the request boundary, not on redrawing six
        # pyqtgraph panels with a deliberately tiny fake report.
        monkeypatch.setattr(pipeline_module, "analyze_file", fake_analyze)
        monkeypatch.setattr(window, "_update_results_table", lambda report: None)
        monkeypatch.setattr(window, "_update_plots", lambda report: None)
        monkeypatch.setattr(window, "_log_payload_summary", lambda report: None)
        monkeypatch.setattr(window, "_log_detection_summary", lambda report: None)
        monkeypatch.setattr(window, "_log_auto_summary", lambda report: None)

        window.auto_analyze_btn.click()
        app.processEvents()

        assert len(captured) == 1
        request = captured[0]
        assert request["iq_format"] == "auto"
        assert request["modulation"] == "auto"
        assert request["sync_word"] is None
        assert request["auto_sample_rate"] is True
        assert request["auto_sync"] is True
        assert request["decode"] is True
        assert "sample_rate" not in request

        # Manual controls remain untouched and a normal Run still sends them.
        assert window.iq_format_combo.currentText() == "complex64"
        assert window.sample_rate_spin.value() == pytest.approx(100_000.0)
        manual = window._build_request()
        assert manual["iq_format"] == "complex64"
        assert manual["sync_word"] == SYNC_WORD
        assert manual["sample_rate"] == pytest.approx(100_000.0)
        assert "auto_sample_rate" not in manual
    finally:
        window.close()
