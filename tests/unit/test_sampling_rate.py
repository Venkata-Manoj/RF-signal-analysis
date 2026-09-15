"""Unit tests for sampling-rate estimation and QAM classifier (SIH26147)."""

from __future__ import annotations

import numpy as np

from rf_analyzer.core.demod import bits_to_bpsk, bits_to_qam16
from rf_analyzer.core.dsp import (
    compute_psd,
    estimate_bandwidth,
    estimate_sampling_rate,
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
