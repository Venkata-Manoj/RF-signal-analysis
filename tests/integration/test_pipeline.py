"""Integration tests for the headless pipeline (info.md §18).

Sample-data files are owned by scripts/generate_test_data.py (another agent).
These tests skip gracefully when sample_data artifacts are absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rf_analyzer.pipeline import analyze_file

SAMPLE_DATA = Path(__file__).resolve().parents[2] / "sample_data"

# Report top-level keys per info.md §13.
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


def test_bpsk_pipeline():
    # Arrange
    iq_path = SAMPLE_DATA / "bpsk.iq"
    if not iq_path.exists():
        pytest.skip(f"missing synthetic data: {iq_path} (run generate_test_data.py)")
    request = {
        "file_path": str(iq_path),
        "sample_rate": 100000,
        "iq_format": "complex64",
        "modulation": "BPSK",
        "sync_word": "0x1ACFFC1D",
    }

    # Act
    report = analyze_file(request)

    # Assert
    assert report["errors"] == []
    assert report["input"]["file_type"] == "iq"
    assert report["signal"]["num_samples"] > 0
    assert report["demodulation"]["num_bits"] > 0
    assert report["correlation"]["header_offset"] >= 0
    assert report["correlation"]["score"] > 0.9


def test_wav_pipeline():
    # Arrange
    wav_path = SAMPLE_DATA / "bpsk.wav"
    if not wav_path.exists():
        pytest.skip(f"missing synthetic data: {wav_path} (run generate_test_data.py)")
    request = {
        "file_path": str(wav_path),
        "iq_format": "auto",
        "modulation": "BPSK",
        "sync_word": "0x1ACFFC1D",
    }

    # Act
    report = analyze_file(request)

    # Assert
    assert report["errors"] == []
    assert report["input"]["file_type"] == "wav"
    assert report["signal"]["num_samples"] > 0


def test_ber_bpsk_clean():
    # Arrange: clean synthetic BPSK ground truth.
    from rf_analyzer.core.demod import demod_bpsk
    from rf_analyzer.core.io import load_iq

    iq_path = SAMPLE_DATA / "bpsk.iq"
    bits_path = SAMPLE_DATA / "bpsk_bits.npy"
    if not iq_path.exists() or not bits_path.exists():
        pytest.skip("missing bpsk.iq / bpsk_bits.npy (run generate_test_data.py)")

    # Act
    samples = load_iq(str(iq_path), dtype="complex64")
    truth = np.load(str(bits_path))
    recovered = np.asarray(demod_bpsk(samples), dtype=np.uint8)
    n = min(len(truth), len(recovered))
    ber = float(np.mean(recovered[:n] != truth[:n]))

    # Assert: acceptance threshold per info.md §16/§22.
    assert ber < 0.01


def test_report_schema():
    # Arrange
    iq_path = SAMPLE_DATA / "bpsk.iq"
    if not iq_path.exists():
        pytest.skip(f"missing synthetic data: {iq_path} (run generate_test_data.py)")
    request = {
        "file_path": str(iq_path),
        "sample_rate": 100000,
        "iq_format": "complex64",
        "modulation": "BPSK",
        "sync_word": "0x1ACFFC1D",
    }

    # Act
    report = analyze_file(request)

    # Assert: all info.md §13 top-level keys present.
    for key in REQUIRED_REPORT_KEYS:
        assert key in report, f"missing report key: {key}"


def test_unsupported_file(tmp_path):
    # Arrange
    bad = tmp_path / "notes.txt"
    bad.write_text("not a signal file")
    request = {"file_path": str(bad), "sample_rate": 100000}

    # Act
    report = analyze_file(request)

    # Assert: graceful failure populates errors (never raises).
    assert "errors" in report
    assert len(report["errors"]) > 0
