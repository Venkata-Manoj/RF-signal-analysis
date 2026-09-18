"""Unit tests for the zero-install web dashboard payload builders."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from rf_analyzer.dashboard import (
    _decimate,
    _fmt,
    build_plots,
    constellation_payload,
    dashboard_payload,
    eye_payload,
    load_capture,
    psd_payload,
    summary_rows,
    time_payload,
    waterfall_payload,
)

SAMPLE_RATE = 100_000.0


def _bpsk(n: int = 1024) -> np.ndarray:
    """Deterministic BPSK-ish samples on the unit circle."""
    rng = np.random.default_rng(7)
    bits = rng.integers(0, 2, n)
    return (2.0 * bits - 1).astype(np.complex64)


def _noisy(n: int = 2048) -> np.ndarray:
    rng = np.random.default_rng(11)
    return (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def test_decimate_leaves_short_input_untouched():
    values = np.arange(10)
    assert _decimate(values, 100).size == 10


def test_decimate_caps_length():
    values = np.arange(10_000)
    out = _decimate(values, 512)
    assert out.size <= 512
    assert out[0] == 0


def test_fmt_handles_none_bool_and_scientific():
    assert _fmt(None) == "--"
    assert _fmt(True) == "yes"
    assert _fmt(False) == "no"
    assert _fmt(0) == "0"
    assert "e" in _fmt(1e7)
    assert _fmt(1234.5, 1, " Hz") == "1234.5 Hz"


# --------------------------------------------------------------------------- #
# constellation
# --------------------------------------------------------------------------- #
def test_constellation_returns_scatter_and_ideal_points():
    payload = constellation_payload(_bpsk(512), "BPSK", max_points=200)
    assert len(payload["i"]) == len(payload["q"])
    assert 0 < payload["shown"] <= 200
    assert len(payload["ideal_i"]) == 2  # BPSK has two ideal points
    assert payload["has_ideal"] is True


def test_constellation_respects_max_points():
    payload = constellation_payload(_noisy(5_000), "QPSK", max_points=100)
    assert payload["shown"] <= 100
    assert len(payload["i"]) == payload["shown"]


def test_constellation_has_no_ideal_points_for_fsk():
    payload = constellation_payload(_noisy(64), "2-FSK")
    assert "ideal_i" not in payload
    assert "ideal_q" not in payload
    # The UI needs to know this is expected, not a rendering failure: every
    # 2-FSK sample sits on the unit circle, so there is nothing to overlay.
    assert payload["has_ideal"] is False


def test_constellation_survives_non_finite_samples():
    samples = np.array([1 + 1j, np.nan + 0j, np.inf + 0j], dtype=np.complex64)
    payload = constellation_payload(samples, "BPSK")
    assert all(np.isfinite(v) for v in payload["i"])


# --------------------------------------------------------------------------- #
# psd / waterfall / eye / time
# --------------------------------------------------------------------------- #
def test_psd_returns_bounded_arrays_and_finite_peak():
    payload = psd_payload(_noisy(4096), SAMPLE_RATE, points=128)
    assert len(payload["freq_hz"]) == len(payload["power_db"]) <= 128
    assert np.isfinite(payload["peak_hz"])
    assert payload["freq_hz"][0] < payload["freq_hz"][-1]


def test_waterfall_encodes_exactly_rows_times_cols_bytes():
    payload = waterfall_payload(_noisy(8192), SAMPLE_RATE, shape=(32, 64))
    assert payload["available"] is True
    raw = base64.b64decode(payload["data_b64"])
    assert len(raw) == payload["rows"] * payload["cols"]
    assert payload["min_db"] <= payload["max_db"]


def test_waterfall_is_unavailable_for_tiny_input():
    payload = waterfall_payload(_noisy(8), SAMPLE_RATE)
    assert payload["available"] is False
    assert "reason" in payload


def test_eye_returns_traces_with_two_symbol_span():
    payload = eye_payload(_bpsk(2048), samples_per_symbol=8)
    assert payload["available"] is True
    assert payload["span"] == 16
    assert 0 < len(payload["traces"]) <= 200
    assert all(len(t) == 16 for t in payload["traces"])


def test_eye_is_unavailable_for_tiny_input():
    payload = eye_payload(_bpsk(4), samples_per_symbol=8)
    assert payload["available"] is False


def test_time_series_is_downsampled_and_aligned():
    payload = time_payload(_noisy(50_000), SAMPLE_RATE, points=200)
    assert payload["available"] is True
    n = len(payload["t_ms"])
    assert n <= 200
    assert len(payload["i"]) == len(payload["q"]) == len(payload["magnitude"]) == n
    assert payload["t_ms"][0] == 0


def test_time_series_rejects_non_positive_rate():
    payload = time_payload(_noisy(64), 0.0)
    assert payload["available"] is False


# --------------------------------------------------------------------------- #
# build_plots
# --------------------------------------------------------------------------- #
def test_build_plots_produces_every_plot():
    plots = build_plots(_noisy(8192), SAMPLE_RATE, mode="QPSK", samples_per_symbol=4)
    assert set(plots) == {"constellation", "psd", "waterfall", "eye", "time"}


def test_build_plots_truncates_long_captures():
    plots = build_plots(
        _noisy(200_000),
        SAMPLE_RATE,
        mode="BPSK",
        samples_per_symbol=1,
        max_samples=1_000,
    )
    assert plots["constellation"]["shown"] <= 1_000


# --------------------------------------------------------------------------- #
# load_capture
# --------------------------------------------------------------------------- #
def test_load_capture_reads_wav_with_its_own_rate(sample_data_dir):
    samples, rate = load_capture(str(sample_data_dir / "bpsk.wav"))
    assert samples.size > 0
    assert rate > 0


def test_load_capture_reads_iq_with_supplied_rate(sample_data_dir):
    samples, rate = load_capture(
        str(sample_data_dir / "bpsk.iq"), sample_rate=SAMPLE_RATE, iq_format="complex64"
    )
    assert samples.size > 0
    assert rate == pytest.approx(SAMPLE_RATE)


def test_load_capture_requires_rate_for_raw_iq(sample_data_dir):
    with pytest.raises(ValueError, match="sample rate is required"):
        load_capture(str(sample_data_dir / "bpsk.iq"), iq_format="complex64")


# --------------------------------------------------------------------------- #
# summary rows
# --------------------------------------------------------------------------- #
def test_summary_rows_covers_the_new_quality_block():
    report = {
        "input": {
            "file_name": "x.iq",
            "sample_rate": 1000.0,
            "iq_format_used": "complex64",
        },
        "signal": {"num_samples": 10, "snr_db": 12.5},
        "modulation": {"estimated_type": "BPSK", "confidence": 0.9},
        "demodulation": {"mode": "BPSK", "num_bits": 10},
        "correlation": {
            "sync_word": "0x1ACFFC1D",
            "score": 1.0,
            "detected": True,
            "header_offset": 0,
        },
        "quality": {
            "applicable": True,
            "evm_percent": 3.14,
            "mer_db": 30.0,
            "modulation_order": 1,
        },
        "payload": {"decoded": {"available": False}},
        "fec": {"candidate": "none", "validated": False},
        "interleaving": {"candidate": "none", "validated": False},
    }
    rows = {r["label"]: r["value"] for r in summary_rows(report)}
    assert rows["EVM / MER"] == "3.14 %"
    assert rows["MER"] == "30.00 dB"
    assert rows["Modulation order"] == "1 bits/sym"
    assert rows["Decoded"] == "no"


def test_summary_marks_a_probed_sync_word_as_assumed():
    report = {
        "correlation": {"sync_word": "0x1ACFFC1D", "assumed": True, "score": 1.0},
    }
    rows = {r["label"]: r["value"] for r in summary_rows(report)}
    assert rows["Sync word"] == "0x1ACFFC1D (assumed)"


def test_summary_reports_evm_as_not_applicable_for_fsk():
    rows = {
        r["label"]: r["value"] for r in summary_rows({"quality": {"applicable": False}})
    }
    assert "n/a" in rows["EVM / MER"]
    assert rows["MER"] == "--"


def test_summary_clamps_degenerate_noiseless_evm():
    rows = {
        r["label"]: r["value"]
        for r in summary_rows(
            {"quality": {"applicable": True, "evm_percent": 0.0, "mer_db": 316.0}}
        )
    }
    assert rows["EVM / MER"] == "<0.01 %"
    assert rows["MER"] == ">100 dB"


def test_summary_rows_tolerates_an_empty_report():
    rows = summary_rows({})
    assert rows
    assert all("label" in r and "value" in r for r in rows)


# --------------------------------------------------------------------------- #
# dashboard_payload
# --------------------------------------------------------------------------- #
def test_dashboard_payload_without_samples_skips_plots():
    payload = dashboard_payload({"input": {"sample_rate": 1000.0}}, samples=None)
    assert payload["plots"] is None
    assert payload["plot_error"] is None
    assert payload["summary"]


def test_dashboard_payload_builds_plots_when_samples_are_given():
    report = {
        "input": {"sample_rate": SAMPLE_RATE},
        "display": {"mode": "BPSK", "samples_per_symbol": 1},
    }
    payload = dashboard_payload(report, _noisy(4096))
    assert payload["plots"] is not None
    assert payload["plot_error"] is None


def test_dashboard_payload_reports_a_missing_sample_rate():
    report = {"input": {"sample_rate": 0}, "display": {"mode": "BPSK"}}
    payload = dashboard_payload(report, _noisy(512))
    assert payload["plots"] is None
    assert "sample rate" in payload["plot_error"]


def test_dashboard_payload_never_raises_on_broken_plot_data(monkeypatch):
    """A plotting failure must degrade to a message, not break the analysis."""
    import rf_analyzer.dashboard as dash

    def boom(*args, **kwargs):
        raise RuntimeError("plot exploded")

    monkeypatch.setattr(dash, "build_plots", boom)
    report = {"input": {"sample_rate": SAMPLE_RATE}, "display": {"mode": "BPSK"}}
    payload = dash.dashboard_payload(report, _noisy(512))
    assert payload["plots"] is None
    assert "plot exploded" in payload["plot_error"]
    assert payload["report"] is report
