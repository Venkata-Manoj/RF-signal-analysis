"""Unit tests for DSP estimators (info.md §17.2)."""

from __future__ import annotations

import numpy as np
import pytest

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


def test_bandwidth_is_zero_when_no_bin_clears_the_floor():
    freqs = np.linspace(-50_000, 50_000, 256)
    psd_db = np.full(256, -80.0)  # perfectly flat: nothing stands out
    assert estimate_bandwidth(freqs, psd_db) == 0.0


def test_bandwidth_is_zero_when_only_one_bin_clears_the_floor():
    """A single bin is not a bandwidth.

    Before this guard the span was ``max - min`` over one element, i.e. exactly
    0.0 -- indistinguishable from "no estimate", which then silently zeroed the
    derived sampling-rate estimate as well.
    """
    freqs = np.linspace(-50_000, 50_000, 256)
    psd_db = np.full(256, -80.0)
    psd_db[128] = -60.0  # exactly one bin, 20 dB above the floor
    assert estimate_bandwidth(freqs, psd_db) == 0.0


def test_bandwidth_spans_two_qualifying_bins():
    freqs = np.linspace(-50_000, 50_000, 256)
    psd_db = np.full(256, -80.0)
    psd_db[120] = -60.0
    psd_db[136] = -60.0
    bw = estimate_bandwidth(freqs, psd_db)
    assert bw > 0.0
    assert bw == pytest.approx(freqs[136] - freqs[120])


def test_flat_spectrum_modulation_cannot_be_measured_by_the_floor_rule():
    """Document the MVP limitation that motivates the pipeline warning.

    A randomly-modulated QPSK stream has an essentially flat spectrum, so its
    peak-above-median is no larger than that of pure noise. Relaxing the 10 dB
    threshold therefore does not separate the two -- it only starts calling
    noise a wideband signal. (A pure *tone* is the opposite case: narrowband,
    huge headroom, easily measured.)
    """
    rng = np.random.default_rng(2)

    noise = (rng.normal(size=4096) + 1j * rng.normal(size=4096)).astype(np.complex64)
    _, noise_psd = compute_psd(noise, 100_000)
    noise_headroom = float(np.max(noise_psd) - np.median(noise_psd))

    symbols = (2 * rng.integers(0, 2, 4096) - 1) + 1j * (
        2 * rng.integers(0, 2, 4096) - 1
    )
    qpsk = (symbols / np.sqrt(2.0)).astype(np.complex64)
    _, qpsk_psd = compute_psd(qpsk, 100_000)
    qpsk_headroom = float(np.max(qpsk_psd) - np.median(qpsk_psd))

    # Noise is at least as "peaky" as the QPSK signal, so the 10 dB rule is not
    # discriminative for flat-spectrum modulations.
    assert noise_headroom >= qpsk_headroom * 0.9


def test_narrowband_tone_is_still_measurable():
    """The estimator must keep returning a usable value for a strong tone.

    Note the number is only loosely related to the true tone width: the Hann
    window's leakage skirt crosses the 10 dB threshold over tens of kHz, so a
    10 kHz tone measures ~60 kHz wide. The estimator is deliberately crude
    (§12); this test only pins that it produces a usable non-zero span rather
    than the degenerate 0.0.
    """
    sample_rate = 100_000
    samples = generate_tone(freq=10_000, sample_rate=sample_rate, duration=0.1)
    freqs, psd_db = compute_psd(samples, sample_rate, nfft=4096)
    bw = estimate_bandwidth(freqs, psd_db)
    assert bw > 0.0
