"""Tests for sampling-rate hypotheses, EVM/MER, and IQ-format auto-detection."""

from __future__ import annotations

import numpy as np
import pytest

from rf_analyzer.core.demod import bits_to_bpsk, bits_to_qam16
from rf_analyzer.core.dsp import (
    compute_evm,
    compute_psd,
    estimate_bandwidth,
    estimate_sampling_rate_candidates,
    ideal_constellation,
    modulation_order,
)
from rf_analyzer.core.io import IQ_FORMATS, detect_iq_format


def _awgn(samples: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    power = float(np.mean(np.abs(samples) ** 2))
    noise_power = power / (10 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(samples.size) + 1j * rng.standard_normal(samples.size)
    )
    return samples + noise


# --------------------------------------------------------------------------- #
# Sampling-rate hypotheses
# --------------------------------------------------------------------------- #


def test_sampling_rate_candidates_include_the_true_rate():
    sr = 100_000
    t = np.arange(sr) / sr
    samples = np.exp(1j * 2 * np.pi * 10_000 * t).astype(np.complex64)
    freqs, psd = compute_psd(samples, sr)
    bw = estimate_bandwidth(freqs, psd)

    candidates = estimate_sampling_rate_candidates(samples, sr, bandwidth_estimate=bw)

    assert candidates, "expected at least one sampling-rate hypothesis"
    rates = [c["rate_hz"] for c in candidates]
    # The true rate must appear among the top hypotheses for a 10 kHz tone.
    assert any(abs(r - sr) / sr < 0.25 for r in rates), rates


def test_sampling_rate_candidates_are_ranked_and_bounded():
    sr = 48_000
    t = np.arange(sr) / sr
    samples = np.exp(1j * 2 * np.pi * 3_000 * t).astype(np.complex64)

    candidates = estimate_sampling_rate_candidates(samples, sr, top_n=4)

    assert len(candidates) <= 4
    scores = [c["score"] for c in candidates]
    assert scores == sorted(scores, reverse=True), "must be ranked by score"
    for cand in candidates:
        assert 0.0 <= cand["score"] <= 1.0
        assert cand["rate_hz"] > 0
        assert cand["rationale"]


def test_sampling_rate_candidates_reject_rates_below_nyquist():
    """A candidate below 2x bandwidth must not outrank a feasible one."""
    sr = 200_000
    t = np.arange(sr) / sr
    samples = np.exp(1j * 2 * np.pi * 20_000 * t).astype(np.complex64)
    freqs, psd = compute_psd(samples, sr)
    bw = estimate_bandwidth(freqs, psd)

    candidates = estimate_sampling_rate_candidates(samples, sr, bandwidth_estimate=bw)

    feasible = [c for c in candidates if c["rate_hz"] >= 2.0 * bw]
    assert feasible, "expected feasible Nyquist-satisfying hypotheses"
    assert candidates[0]["rate_hz"] >= 2.0 * bw


def test_sampling_rate_candidates_empty_without_bandwidth():
    silent = np.zeros(1024, dtype=np.complex64)
    assert estimate_sampling_rate_candidates(silent, 100_000) == []


# --------------------------------------------------------------------------- #
# EVM / MER
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", ["BPSK", "QPSK", "16-QAM"])
def test_ideal_constellations_have_unit_average_power(mode):
    points = ideal_constellation(mode)
    assert points is not None and points.size >= 2
    assert float(np.mean(np.abs(points) ** 2)) == pytest.approx(1.0, abs=1e-9)


def test_modulation_order_values():
    assert modulation_order("BPSK") == 1
    assert modulation_order("QPSK") == 2
    assert modulation_order("16-QAM") == 4
    assert modulation_order("2-FSK") is None


def test_evm_is_zero_for_an_ideal_constellation():
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, size=400, dtype=np.uint8)
    symbols = bits_to_bpsk(bits).astype(np.complex128)

    result = compute_evm(symbols, "BPSK")

    assert result["applicable"] is True
    assert result["evm_percent"] < 1e-6
    assert result["modulation_order"] == 1


@pytest.mark.parametrize(
    ("mode", "maker", "n_bits"),
    [
        ("BPSK", bits_to_bpsk, 400),
        ("16-QAM", bits_to_qam16, 400),
    ],
)
def test_evm_grows_as_snr_drops(mode, maker, n_bits):
    rng = np.random.default_rng(2)
    bits = rng.integers(0, 2, size=n_bits, dtype=np.uint8)
    clean = np.asarray(maker(bits), dtype=np.complex128)

    evm_clean = compute_evm(_awgn(clean, 30.0, seed=3), mode)["evm_percent"]
    evm_noisy = compute_evm(_awgn(clean, 8.0, seed=3), mode)["evm_percent"]

    assert evm_clean < evm_noisy


def test_evm_derived_snr_tracks_injected_snr():
    """MER should land within a few dB of the injected SNR for clean BPSK."""
    rng = np.random.default_rng(4)
    bits = rng.integers(0, 2, size=4000, dtype=np.uint8)
    symbols = np.asarray(bits_to_bpsk(bits), dtype=np.complex128)

    for target_snr in (10.0, 20.0):
        noisy = _awgn(symbols, target_snr, seed=5)
        mer = compute_evm(noisy, "BPSK")["mer_db"]
        assert abs(mer - target_snr) < 3.0, f"target={target_snr} mer={mer:.2f}"


def test_evm_not_applicable_for_fsk():
    samples = np.exp(1j * np.pi * np.arange(256) / 8.0)
    result = compute_evm(samples, "2-FSK")
    assert result["applicable"] is False
    assert result["evm_percent"] is None


def test_evm_wrong_mode_costs_quality():
    """A deliberate mode mismatch must show up as much worse EVM."""
    rng = np.random.default_rng(6)
    bits = rng.integers(0, 2, size=800, dtype=np.uint8)
    qam = np.asarray(bits_to_qam16(bits), dtype=np.complex128)

    correct = compute_evm(qam, "16-QAM")["evm_percent"]
    wrong = compute_evm(qam, "BPSK")["evm_percent"]

    assert correct < wrong


def test_evm_handles_samples_per_symbol():
    rng = np.random.default_rng(7)
    bits = rng.integers(0, 2, size=200, dtype=np.uint8)
    symbols = np.asarray(bits_to_bpsk(bits), dtype=np.complex128)
    oversampled = np.repeat(symbols, 8)

    result = compute_evm(oversampled, "BPSK", samples_per_symbol=8)
    assert result["n_symbols"] == 200
    assert result["evm_percent"] < 1e-6


# --------------------------------------------------------------------------- #
# IQ-format auto-detection
# --------------------------------------------------------------------------- #


def _write_format(path, samples: np.ndarray, fmt: str) -> None:
    """Write ``samples`` to ``path`` using the requested raw format."""
    if fmt == "complex64":
        samples.astype(np.complex64).tofile(path)
    elif fmt == "int16":
        interleaved = np.empty(samples.size * 2, dtype=np.int16)
        interleaved[0::2] = np.clip(np.real(samples) * 32768.0, -32768, 32767).astype(
            np.int16
        )
        interleaved[1::2] = np.clip(np.imag(samples) * 32768.0, -32768, 32767).astype(
            np.int16
        )
        interleaved.tofile(path)
    elif fmt == "int8":
        interleaved = np.empty(samples.size * 2, dtype=np.int8)
        interleaved[0::2] = np.clip(np.real(samples) * 128.0, -128, 127).astype(np.int8)
        interleaved[1::2] = np.clip(np.imag(samples) * 128.0, -128, 127).astype(np.int8)
        interleaved.tofile(path)
    elif fmt == "uint8":
        interleaved = np.empty(samples.size * 2, dtype=np.uint8)
        interleaved[0::2] = np.clip(np.real(samples) * 128.0 + 128.0, 0, 255).astype(
            np.uint8
        )
        interleaved[1::2] = np.clip(np.imag(samples) * 128.0 + 128.0, 0, 255).astype(
            np.uint8
        )
        interleaved.tofile(path)
    else:  # pragma: no cover - guarded by the parametrisation
        raise ValueError(fmt)


def _qpsk_like(n: int = 4000, seed: int = 11) -> np.ndarray:
    """White QPSK at one sample per symbol (no oversampling)."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=n, dtype=np.uint8)
    i_bits, q_bits = bits[0::2], bits[1::2]
    i = 2.0 * i_bits - 1.0
    q = 2.0 * q_bits - 1.0
    return ((i + 1j * q) / np.sqrt(2.0)).astype(np.complex64)


def _oversampled_signal(n_symbols: int = 1500, sps: int = 8, seed: int = 11):
    """Band-limited, oversampled capture — what a real SDR produces.

    Format detection depends on oversampling (consecutive samples of a real
    capture are correlated), so this is the representative test signal.
    """
    from scipy.signal import firwin, lfilter

    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=n_symbols * 2, dtype=np.uint8)
    i = 2.0 * bits[0::2] - 1.0
    q = 2.0 * bits[1::2] - 1.0
    symbols = np.repeat((i + 1j * q) / np.sqrt(2.0), sps)

    taps = firwin(33, 0.15)
    shaped = lfilter(taps, 1.0, symbols)
    shaped = shaped / np.max(np.abs(shaped)) * 0.85
    return shaped.astype(np.complex64)


@pytest.mark.parametrize("fmt", IQ_FORMATS)
def test_detect_iq_format_identifies_each_real_format(tmp_path, fmt):
    """On a realistic oversampled capture all four formats are identified."""
    samples = _oversampled_signal()
    path = tmp_path / f"capture_{fmt}.iq"
    _write_format(path, samples, fmt)

    result = detect_iq_format(str(path))

    assert result["format"] == fmt, f"got {result}"
    assert result["confidence"] > 0.5
    assert set(result["scores"]) == set(IQ_FORMATS)


@pytest.mark.parametrize("fmt", IQ_FORMATS)
def test_detect_iq_format_survives_noise(tmp_path, fmt):
    """A noisy oversampled capture must still be identified correctly."""
    rng = np.random.default_rng(21)
    samples = _oversampled_signal()
    noise = rng.standard_normal(samples.size) + 1j * rng.standard_normal(samples.size)
    noisy = (samples + 0.25 * noise / np.sqrt(2)).astype(np.complex64)

    path = tmp_path / f"noisy_{fmt}.iq"
    _write_format(path, noisy, fmt)

    assert detect_iq_format(str(path))["format"] == fmt


def test_detect_iq_format_separates_float_from_integer(tmp_path):
    """Raw int16 data must not be mistaken for complex64."""
    samples = _oversampled_signal()
    path = tmp_path / "ints.iq"
    _write_format(path, samples, "int16")

    result = detect_iq_format(str(path))
    assert result["format"] != "complex64"
    assert result["scores"]["complex64"] == 0.0


def test_detect_iq_format_float_capture_is_never_read_as_integer(tmp_path):
    """A complex64 capture is detected regardless of oversampling."""
    path = tmp_path / "floats.iq"
    _write_format(path, _qpsk_like(), "complex64")

    result = detect_iq_format(str(path))
    assert result["format"] == "complex64"
    assert result["confidence"] > 0.9


def test_detect_iq_format_reports_low_confidence_when_ambiguous(tmp_path):
    """A white 1-sps integer capture is genuinely ambiguous — say so.

    Without oversampling there is no smoothness to exploit, so the integer
    formats are indistinguishable. The detector must report low confidence
    rather than a confident wrong answer.
    """
    samples = _qpsk_like()
    confidences = []
    for fmt in ("int16", "int8", "uint8"):
        path = tmp_path / f"white_{fmt}.iq"
        _write_format(path, samples, fmt)
        confidences.append(detect_iq_format(str(path))["confidence"])

    assert all(c <= 0.6 for c in confidences), confidences


def test_detect_iq_format_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        detect_iq_format(str(tmp_path / "nope.iq"))


def test_detect_iq_format_reports_confidence_and_detail(tmp_path):
    path = tmp_path / "c.iq"
    _write_format(path, _oversampled_signal(), "complex64")
    result = detect_iq_format(str(path))
    assert 0.0 <= result["confidence"] <= 1.0
    assert "complex64" in result["detail"]


def test_detect_iq_format_handles_garbage_without_crashing(tmp_path):
    path = tmp_path / "garbage.iq"
    path.write_bytes(bytes(range(256)) * 8)
    result = detect_iq_format(str(path))
    assert result["format"] in IQ_FORMATS
    assert 0.0 <= result["confidence"] <= 1.0


def test_detect_iq_format_handles_tiny_file(tmp_path):
    path = tmp_path / "tiny.iq"
    path.write_bytes(b"\x01\x02\x03")
    result = detect_iq_format(str(path))
    assert result["format"] in IQ_FORMATS


def test_evm_of_a_noiseless_capture_reports_unbounded_mer_as_none():
    """A zero error vector must not produce `inf`.

    `Infinity` is not valid JSON and `JSON.parse` rejects it, so the absence
    of a measurable error is reported as `None` instead.
    """
    import math

    rng = np.random.default_rng(9)
    bits = rng.integers(0, 2, size=512, dtype=np.uint8)
    ideal = np.asarray(bits_to_bpsk(bits), dtype=np.complex128)

    result = compute_evm(ideal, "BPSK")
    assert result["applicable"] is True
    assert result["evm_percent"] == pytest.approx(0.0, abs=1e-12)
    assert result["mer_db"] is None
    assert result["snr_db_from_evm"] is None
    assert not math.isinf(result["evm_percent"])
