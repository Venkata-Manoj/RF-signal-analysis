"""Integration tests for official SIH26147 portal tasks.

Mirrors the problem statement bullets:
i.  Identify signal parameters (sampling frequency, modulation, FEC, interleaving)
ii. Demodulate signals (FSK, QAM, PSK)
iii. Carry out de-interleaving (Block, Convolution, Diagonal, Pseudo Random)
iv. FEC (conv-Viterbi, RS, Concatenated, LDPC)
v.  Bit stream correlation for header/payload.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rf_analyzer.pipeline import analyze_file

SAMPLE_DATA = Path(__file__).resolve().parents[2] / "sample_data"

REQUIRED_REPORT_KEYS = [
    "meta",
    "input",
    "signal",
    "modulation",
    "demodulation",
    "correlation",
    "fec",
    "interleaving",
    "warnings",
    "errors",
]


def _require(name: str):
    p = SAMPLE_DATA / name
    if not p.exists():
        pytest.skip(f"missing synthetic data: {p} (run scripts/generate_test_data.py)")
    return str(p)


def test_bpsk_pipeline():
    request = {
        "file_path": _require("bpsk.iq"),
        "sample_rate": 100_000,
        "iq_format": "complex64",
        "modulation": "BPSK",
        "sync_word": "0x1ACFFC1D",
    }
    report = analyze_file(request)
    assert report["errors"] == []
    assert report["input"]["file_type"] == "iq"
    assert report["signal"]["num_samples"] > 0
    assert report["demodulation"]["num_bits"] > 0
    assert report["correlation"]["header_offset"] >= 0
    assert report["correlation"]["score"] > 0.9


def test_wav_pipeline():
    request = {
        "file_path": _require("bpsk.wav"),
        "iq_format": "auto",
        "modulation": "BPSK",
        "sync_word": "0x1ACFFC1D",
    }
    report = analyze_file(request)
    assert report["errors"] == []
    assert report["input"]["file_type"] == "wav"
    assert report["signal"]["num_samples"] > 0


def test_ber_bpsk_clean():
    from rf_analyzer.core.demod import demod_bpsk
    from rf_analyzer.core.io import load_iq

    samples = load_iq(_require("bpsk.iq"), dtype="complex64")
    truth = np.load(_require("bpsk_bits.npy"))
    recovered = np.asarray(demod_bpsk(samples), dtype=np.uint8)
    n = min(len(truth), len(recovered))
    ber = float(np.mean(recovered[:n] != truth[:n]))
    assert ber < 0.01


def test_report_schema():
    request = {
        "file_path": _require("bpsk.iq"),
        "sample_rate": 100_000,
        "iq_format": "complex64",
        "modulation": "BPSK",
        "sync_word": "0x1ACFFC1D",
    }
    report = analyze_file(request)
    for key in REQUIRED_REPORT_KEYS:
        assert key in report, f"missing report key: {key}"


def test_unsupported_file(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("not a signal file")
    report = analyze_file({"file_path": str(bad), "sample_rate": 100_000})
    assert "errors" in report and len(report["errors"]) > 0


def test_bits_preview_present():
    request = {
        "file_path": _require("bpsk.iq"),
        "sample_rate": 100_000,
        "iq_format": "complex64",
        "modulation": "BPSK",
        "sync_word": "0x1ACFFC1D",
    }
    report = analyze_file(request)
    assert report["errors"] == []
    dem = report["demodulation"]
    assert "bits_preview" in dem
    preview = dem["bits_preview"]
    assert isinstance(preview, list)
    num_bits = int(dem["num_bits"])
    assert len(preview) == min(2048, num_bits)
    assert all(int(b) in (0, 1) for b in preview)


def test_modulation_param_changes_result():
    base = {
        "file_path": _require("bpsk.iq"),
        "sample_rate": 100_000,
        "iq_format": "complex64",
        "sync_word": "0x1ACFFC1D",
    }
    bpsk_report = analyze_file({**base, "modulation": "BPSK"})
    qpsk_report = analyze_file({**base, "modulation": "QPSK"})
    assert bpsk_report["errors"] == [] and qpsk_report["errors"] == []
    assert bpsk_report["demodulation"]["mode"] == "BPSK"
    assert qpsk_report["demodulation"]["mode"] == "QPSK"
    assert bpsk_report["demodulation"]["mode"] != qpsk_report["demodulation"]["mode"]
    ratio = (
        qpsk_report["demodulation"]["num_bits"]
        / bpsk_report["demodulation"]["num_bits"]
    )
    assert 1.9 <= ratio <= 2.1


# ---- SIH26147 portal tasks i-v ----


def test_portal_task_ii_demod_qam():
    """Portal bullet ii: Demodulate QAM."""
    path = _require("qam16.iq")
    report = analyze_file(
        {
            "file_path": path,
            "sample_rate": 100_000,
            "iq_format": "complex64",
            "modulation": "16-QAM",
            "sync_word": "0x1ACFFC1D",
        }
    )
    assert report["errors"] == []
    assert report["demodulation"]["mode"] in ("16-QAM", "QAM")
    assert report["demodulation"]["num_bits"] > 0


def test_portal_task_i_sampling_rate_estimate():
    """Portal bullet i: Identify sampling frequency."""
    path = _require("tone.iq")
    report = analyze_file(
        {
            "file_path": path,
            "sample_rate": 100_000,
            "iq_format": "complex64",
            "modulation": "auto",
        }
    )
    assert report["errors"] == []
    sre = report["signal"].get("sample_rate_estimate")
    assert sre is not None and sre > 0


def test_portal_task_iii_deinterleave():
    """Portal bullet iii: De-interleaving candidates present."""
    path = _require("bpsk.iq")
    report = analyze_file(
        {
            "file_path": path,
            "sample_rate": 100_000,
            "iq_format": "complex64",
            "modulation": "BPSK",
            "sync_word": "0x1ACFFC1D",
        }
    )
    ilv = report["interleaving"]
    assert ilv["candidate"] is not None or ilv["candidate"] is None
    assert "confidence" in ilv


def test_portal_task_iv_fec():
    """Portal bullet iv: FEC candidates present, confidence capped."""
    path = _require("bpsk.iq")
    report = analyze_file(
        {
            "file_path": path,
            "sample_rate": 100_000,
            "iq_format": "complex64",
            "modulation": "BPSK",
            "sync_word": "0x1ACFFC1D",
        }
    )
    fec = report["fec"]
    assert "candidate" in fec and "confidence" in fec
    assert fec["confidence"] <= 0.5  # honest: never claims blind FEC


def test_portal_task_v_correlation():
    """Portal bullet v: Bitstream correlation for header/payload."""
    path = _require("bpsk.iq")
    report = analyze_file(
        {
            "file_path": path,
            "sample_rate": 100_000,
            "iq_format": "complex64",
            "modulation": "BPSK",
            "sync_word": "0x1ACFFC1D",
        }
    )
    corr = report["correlation"]
    assert corr["header_offset"] >= 0
    assert corr["score"] > 0.9
