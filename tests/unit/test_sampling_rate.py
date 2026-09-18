"""Unit tests for sampling-rate estimation and QAM classifier (SIH26147)."""

from __future__ import annotations

import numpy as np
import pytest

from rf_analyzer.core.demod import bits_to_bpsk, bits_to_qam16
from rf_analyzer.core.dsp import (
    compute_psd,
    estimate_and_correct_cfo,
    estimate_bandwidth,
    estimate_cfo,
    estimate_sampling_rate,
    estimate_sampling_rate_wideband,
)
from rf_analyzer.pipeline import _classify_modulation


def test_sampling_rate_from_tone():
    sr = 100_000
    t = np.arange(int(sr * 0.1)) / sr
    samples = np.exp(1j * 2 * np.pi * 10_000 * t).astype(np.complex64)
    freqs, psd = compute_psd(samples, sr)
    bw = estimate_bandwidth(freqs, psd)
    est = estimate_sampling_rate(bw)
    assert est > 0
    assert 0.5 * sr < est < 3.0 * sr


def test_qam_classifier():
    rng = np.random.default_rng(42)
    bits = rng.integers(0, 2, size=400, dtype=np.uint8)
    symbols = bits_to_qam16(bits)
    est_type, est_conf, _ = _classify_modulation(symbols)
    assert est_type == "16-QAM", f"Expected 16-QAM, got {est_type}"
    assert est_conf >= 0.5


def test_bpsk_not_qam():
    rng = np.random.default_rng(42)
    bits = rng.integers(0, 2, size=200, dtype=np.uint8)
    symbols = bits_to_bpsk(bits).astype(np.complex64)
    est_type, est_conf, _ = _classify_modulation(symbols)
    assert est_type == "BPSK", f"Expected BPSK, got {est_type}"


def test_wideband_falls_back_to_bw_only_when_no_symbol_rate():
    # No reliable symbol-rate line -> identical to the BW-only hint.
    assert estimate_sampling_rate_wideband(20_000.0, 0.0) == estimate_sampling_rate(
        20_000.0
    )
    assert estimate_sampling_rate_wideband(20_000.0, None) == estimate_sampling_rate(
        20_000.0
    )


def test_wideband_takes_tighter_of_bw_and_symbol_rate():
    # Symbol rate above bandwidth must win (it is the binding Nyquist constraint).
    fused = estimate_sampling_rate_wideband(20_000.0, 40_000.0, factor=2.0)
    assert fused == pytest.approx(80_000.0)
    # Bandwidth above symbol rate: bandwidth wins.
    fused_bw = estimate_sampling_rate_wideband(40_000.0, 20_000.0, factor=2.0)
    assert fused_bw == pytest.approx(80_000.0)


def test_wideband_invalid_bandwidth_returns_zero():
    assert estimate_sampling_rate_wideband(0.0, 12_000.0) == 0.0


def test_estimate_and_correct_cfo_removes_injected_offset():
    # Arrange: a pure tone with a known 2 kHz carrier offset.
    sr = 100_000
    t = np.arange(sr) / sr
    injected = 2_000.0
    samples = np.exp(1j * 2 * np.pi * injected * t).astype(np.complex64)

    # Act
    corrected, cfo = estimate_and_correct_cfo(samples, sr)

    # Assert: estimate is close to the injected offset, and the corrected
    # signal's residual offset is far smaller than the original.
    assert abs(cfo - injected) < 50.0
    residual = abs(estimate_cfo(corrected, sr))
    assert residual < abs(injected) * 0.25
