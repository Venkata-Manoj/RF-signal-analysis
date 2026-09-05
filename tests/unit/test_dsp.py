"""Unit tests for DSP estimators (info.md §17.2)."""

from __future__ import annotations

import numpy as np

from rf_analyzer.core.dsp import (
    compute_psd,
    estimate_bandwidth,
    estimate_center_frequency,
    estimate_snr,
)


def generate_tone(freq=10000, sample_rate=100000, duration=0.1):
    t = np.arange(int(sample_rate * duration)) / sample_rate
    return np.exp(1j * 2 * np.pi * freq * t).astype(np.complex64)


def test_psd_peak_center_frequency():
    # Arrange
    sample_rate = 100000
    target_freq = 10000
    samples = generate_tone(freq=target_freq, sample_rate=sample_rate)

    # Act
    freqs, psd_db = compute_psd(samples, sample_rate, nfft=4096)
    estimated_freq = estimate_center_frequency(freqs, psd_db)

    # Assert
    assert abs(estimated_freq - target_freq) < 1000


def test_bandwidth_positive():
    # Arrange
    sample_rate = 100000
    samples = generate_tone(freq=10000, sample_rate=sample_rate)

    # Act
    freqs, psd_db = compute_psd(samples, sample_rate, nfft=4096)
    bw = estimate_bandwidth(freqs, psd_db)

    # Assert
    assert bw >= 0.0


def test_snr_positive_for_tone():
    # Arrange
    sample_rate = 100000
    samples = generate_tone(freq=10000, sample_rate=sample_rate)

    # Act
    freqs, psd_db = compute_psd(samples, sample_rate, nfft=4096)
    snr = estimate_snr(psd_db)

    # Assert
    assert snr > 0
