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


def test_bits_preview_present():
    # Arrange: clean synthetic BPSK capture (seeded, see generate_test_data.py).
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

    # Assert: demodulation exposes a real bit preview, not just counts.
    assert report["errors"] == []
    dem = report["demodulation"]
    assert "bits_preview" in dem, "demodulation.bits_preview missing (regression)"
    preview = dem["bits_preview"]
    assert isinstance(preview, list), f"bits_preview must be list, got {type(preview)}"
    num_bits = int(dem["num_bits"])
    expected_len = min(2048, num_bits)
    assert (
        len(preview) == expected_len
    ), f"bits_preview length {len(preview)} != min(2048, num_bits)={expected_len}"
    assert all(int(b) in (0, 1) for b in preview), "bits_preview must contain only 0/1"

    # Assert: preview matches direct demod of the same file (first N exactly).
    from rf_analyzer.core.demod import demod_bpsk
    from rf_analyzer.core.io import load_iq

    samples = load_iq(str(iq_path), dtype="complex64")
    expected = np.asarray(demod_bpsk(samples), dtype=np.uint8)
    assert np.array_equal(
        np.asarray(preview, dtype=np.uint8), expected[: len(preview)]
    ), "bits_preview must equal demod_bpsk(load_iq(...)) first N exactly"


def test_modulation_param_changes_result():
    # Arrange: same capture analyzed under two modulation selections.
    iq_path = SAMPLE_DATA / "bpsk.iq"
    if not iq_path.exists():
        pytest.skip(f"missing synthetic data: {iq_path} (run generate_test_data.py)")
    base = {
        "file_path": str(iq_path),
        "sample_rate": 100000,
        "iq_format": "complex64",
        "sync_word": "0x1ACFFC1D",
    }

    # Act
    bpsk_report = analyze_file({**base, "modulation": "BPSK"})
    qpsk_report = analyze_file({**base, "modulation": "QPSK"})

    # Assert: modulation param is honored (no hardcoded demod path).
    assert bpsk_report["errors"] == []
    assert qpsk_report["errors"] == []
    bpsk_mode = bpsk_report["demodulation"]["mode"]
    qpsk_mode = qpsk_report["demodulation"]["mode"]
    assert bpsk_mode == "BPSK", f"expected BPSK mode, got {bpsk_mode!r}"
    assert qpsk_mode == "QPSK", f"expected QPSK mode, got {qpsk_mode!r}"
    assert bpsk_mode != qpsk_mode, "modulation param must change demodulation.mode"

    # Assert: QPSK yields ~2x bits of BPSK for the same samples.
    bpsk_n = int(bpsk_report["demodulation"]["num_bits"])
    qpsk_n = int(qpsk_report["demodulation"]["num_bits"])
    assert bpsk_n > 0 and qpsk_n > 0
    ratio = qpsk_n / float(bpsk_n)
    assert 1.9 <= ratio <= 2.1, f"QPSK/BPSK num_bits ratio {ratio:.3f} not ~2x"


def test_unmeasurable_bandwidth_is_warned_about_not_silently_zero():
    """A 0 Hz bandwidth must be explained, never presented as a measurement.

    For a flat-spectrum capture the naive median-floor estimator finds no
    usable span, which also zeroes the derived sampling-rate estimate. A user
    reading "0 Hz" would otherwise conclude the signal really is 0 Hz wide.
    """
    capture = SAMPLE_DATA / "qpsk.iq"
    if not capture.exists():
        pytest.skip("sample data not generated")

    report = analyze_file(
        {
            "file_path": str(capture),
            "sample_rate": 100000,
            "iq_format": "complex64",
            "modulation": "auto",
        }
    )
    assert report["errors"] == []
    if report["signal"]["bandwidth_estimate"] > 0.0:
        pytest.skip("this capture happens to yield a measurable bandwidth")

    assert any(
        "Occupied-bandwidth estimate unavailable" in w for w in report["warnings"]
    ), report["warnings"]


def test_measurable_bandwidth_is_not_warned_about():
    """The warning must be specific to the failure, not emitted unconditionally."""
    capture = SAMPLE_DATA / "bpsk.iq"
    if not capture.exists():
        pytest.skip("sample data not generated")

    report = analyze_file(
        {
            "file_path": str(capture),
            "sample_rate": 100000,
            "iq_format": "complex64",
            "modulation": "auto",
            "sync_word": "0x1ACFFC1D",
        }
    )
    assert report["signal"]["bandwidth_estimate"] > 0.0
    assert not any(
        "Occupied-bandwidth estimate unavailable" in w for w in report["warnings"]
    ), report["warnings"]


# --------------------------------------------------------------------------- #
# Modulation fit cross-check (EVM vs the chosen constellation)
# --------------------------------------------------------------------------- #
def _write_complex64(path, samples) -> None:
    np.asarray(samples, dtype=np.complex64).tofile(str(path))


def test_clean_synthetic_capture_reports_a_good_modulation_fit():
    capture = SAMPLE_DATA / "bpsk.iq"
    if not capture.exists():
        pytest.skip("sample data not generated")

    report = analyze_file(
        {
            "file_path": str(capture),
            "sample_rate": 100000,
            "iq_format": "complex64",
            "modulation": "BPSK",
        }
    )
    assert report["quality"]["fit_ok"] is True
    assert not any("Modulation fit is poor" in w for w in report["warnings"])


def test_poor_modulation_fit_is_warned_about_with_bias_caveat(tmp_path):
    """A Gaussian sample cloud does not fit a 16-QAM constellation.

    This mirrors a real GPS L1 capture (BPSK) that the classifier called
    16-QAM and which fitted at 39% EVM. The pipeline must flag the poor fit,
    and must *not* claim another modulation "fits better" -- EVM falls with
    constellation density, so the numbers are not a ranking.
    """
    rng = np.random.default_rng(21)
    cloud = (rng.normal(size=4096) + 1j * rng.normal(size=4096)).astype(np.complex64)
    path = tmp_path / "cloud.iq"
    _write_complex64(path, cloud)

    report = analyze_file(
        {
            "file_path": str(path),
            "sample_rate": 100000,
            "iq_format": "complex64",
            "modulation": "16-QAM",
        }
    )
    quality = report["quality"]
    assert quality["applicable"] is True
    assert quality["fit_ok"] is False

    fit_warnings = [w for w in report["warnings"] if "Modulation fit is poor" in w]
    assert fit_warnings, report["warnings"]
    message = fit_warnings[0]
    assert "16-QAM" in message
    assert "error-vector magnitude" in message
    # The caveat must be present so the numbers are not read as a ranking.
    assert "not a ranking" in message
    assert "explicitly" in message

    candidates = quality["candidates"]
    assert candidates, "expected the other constellations to be reported"
    assert all(row["mode"] != "16-QAM" for row in candidates)
    evms = [row["evm_percent"] for row in candidates]
    assert evms == sorted(evms), "candidates should be listed best-fit first"
    # The documented bias: the densest constellation fits best.
    assert candidates[0]["mode"] == "64-QAM"
    assert candidates[-1]["mode"] == "BPSK"


def test_fit_cross_check_is_bounded_for_long_captures(tmp_path, monkeypatch):
    """The extra candidate evaluations must not scale with capture length.

    ``quality.n_symbols`` describes the chosen constellation's fit over the
    whole capture; only the *cross-check* is bounded, so observe the sample
    counts the cross-check actually receives.
    """
    from rf_analyzer.config import MODULATION_FIT_MAX_SAMPLES
    from rf_analyzer.pipeline import compute_evm as pipeline_compute_evm

    rng = np.random.default_rng(22)
    big = (rng.normal(size=MODULATION_FIT_MAX_SAMPLES * 3) + 0j).astype(np.complex64)
    path = tmp_path / "big.iq"
    _write_complex64(path, big)

    seen: list[int] = []
    real = pipeline_compute_evm

    def spy(samples, mode="BPSK", samples_per_symbol=1):
        seen.append(int(np.asarray(samples).size))
        return real(samples, mode=mode, samples_per_symbol=samples_per_symbol)

    monkeypatch.setattr("rf_analyzer.pipeline.compute_evm", spy)

    report = analyze_file(
        {
            "file_path": str(path),
            "sample_rate": 100000,
            "iq_format": "complex64",
            "modulation": "16-QAM",
        }
    )
    assert report["quality"]["fit_ok"] is False
    assert len(seen) >= 2, "expected the chosen fit plus at least one candidate"
    assert max(seen[1:]) <= MODULATION_FIT_MAX_SAMPLES
