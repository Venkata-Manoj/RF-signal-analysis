"""Strict-JSON guarantees for report serialisation.

``json.dump`` writes ``Infinity`` and ``NaN`` by default, but those tokens are
not valid JSON: a browser's ``JSON.parse`` rejects the whole document. These
tests pin the sanitising boundary so a non-finite measurement can never reach
a JavaScript consumer (the web dashboard) or an on-disk report.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from rf_analyzer.core.report import json_safe, save_report


def _strict_load(text: str):
    """Parse JSON with non-finite constants rejected, as a browser would."""

    def reject(token: str):
        raise ValueError(f"non-finite JSON token: {token}")

    return json.loads(text, parse_constant=reject)


# --------------------------------------------------------------------------- #
# json_safe
# --------------------------------------------------------------------------- #
def test_json_safe_replaces_non_finite_floats_with_none():
    assert json_safe(float("inf")) is None
    assert json_safe(float("-inf")) is None
    assert json_safe(float("nan")) is None


def test_json_safe_keeps_finite_floats():
    assert json_safe(1.5) == 1.5
    assert json_safe(0.0) == 0.0
    assert json_safe(-273.15) == -273.15


def test_json_safe_recurses_through_nested_structures():
    payload = {
        "a": [1.0, float("inf"), {"b": float("nan")}],
        "c": (2.0, float("-inf")),
        "d": "keep",
    }
    cleaned = json_safe(payload)
    assert cleaned == {"a": [1.0, None, {"b": None}], "c": [2.0, None], "d": "keep"}


def test_json_safe_preserves_booleans():
    """bool is a subclass of int, so it must not be coerced to a number."""
    assert json_safe(True) is True
    assert json_safe(False) is False
    assert json_safe({"ok": True}) == {"ok": True}


def test_json_safe_converts_numpy_scalars():
    assert json_safe(np.float64(2.5)) == 2.5
    assert isinstance(json_safe(np.int64(7)), int)
    assert json_safe(np.bool_(True)) is True
    assert json_safe(np.float32("nan")) is None
    assert json_safe(np.float64("inf")) is None


def test_json_safe_leaves_other_types_untouched():
    assert json_safe(None) is None
    assert json_safe("text") == "text"
    assert json_safe(3) == 3


def test_json_safe_output_is_strictly_serialisable():
    payload = {
        "finite": 1.0,
        "bad": float("inf"),
        "nested": [float("nan"), {"deep": np.float64("inf")}],
        "np_int": np.int32(4),
    }
    text = json.dumps(json_safe(payload), allow_nan=False)
    assert _strict_load(text) == {
        "finite": 1.0,
        "bad": None,
        "nested": [None, {"deep": None}],
        "np_int": 4,
    }


# --------------------------------------------------------------------------- #
# save_report
# --------------------------------------------------------------------------- #
def test_save_report_writes_strict_json(tmp_path):
    report = {
        "meta": {"tool_name": "t", "version": "1"},
        "signal": {"snr_db": float("inf"), "cfo_estimate_hz": float("nan")},
        "quality": {"mer_db": float("inf"), "evm_percent": 0.0},
        "warnings": [],
        "errors": [],
    }
    path = tmp_path / "report.json"
    save_report(report, str(path))

    text = path.read_text(encoding="utf-8")
    assert "Infinity" not in text
    assert "NaN" not in text

    loaded = _strict_load(text)
    assert loaded["signal"]["snr_db"] is None
    assert loaded["signal"]["cfo_estimate_hz"] is None
    assert loaded["quality"]["mer_db"] is None
    assert loaded["quality"]["evm_percent"] == 0.0


def test_save_report_creates_missing_parent_directories(tmp_path):
    path = tmp_path / "nested" / "deeper" / "report.json"
    save_report({"errors": []}, str(path))
    assert path.is_file()


def test_a_real_noiseless_capture_round_trips_through_disk(tmp_path, sample_data_dir):
    """The regression that motivated json_safe: coded_rs.iq has zero EVM."""
    from rf_analyzer.pipeline import analyze_file

    report = analyze_file(
        {
            "file_path": str(sample_data_dir / "coded_rs.iq"),
            "sample_rate": 100000,
            "iq_format": "complex64",
            "modulation": "auto",
        }
    )
    path = tmp_path / "coded_rs.json"
    save_report(report, str(path))

    text = path.read_text(encoding="utf-8")
    assert "Infinity" not in text
    loaded = _strict_load(text)
    assert loaded["quality"]["mer_db"] is None
    assert math.isfinite(loaded["quality"]["evm_percent"])
