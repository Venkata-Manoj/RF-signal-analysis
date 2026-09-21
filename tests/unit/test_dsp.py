"""Unit tests for DSP estimators (info.md §17.2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rf_analyzer.core.dsp import (
    compute_psd,
    estimate_bandwidth,
    estimate_burst_region,
    estimate_center_frequency,
    estimate_fsk_symbol_period,
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


# ---- 2-FSK symbol-period recovery ------------------------------------------
#
# The MVP demodulator works one bit per sample. That is only correct when the
# capture already has one sample per symbol; a real 2-FSK burst does not. These
# tests pin the estimator that recovers the period, including the cases where it
# must decline rather than guess.


def _fsk(bits, samples_per_symbol, sample_rate=100_000, dev=5000.0):
    """Continuous-phase 2-FSK, info.md §15.1 phase convention."""
    tone = np.where(np.asarray(bits, dtype=np.uint8) == 1, dev, -dev)
    advance = tone * ((samples_per_symbol - 1) * 2 * np.pi / sample_rate)
    carried = np.concatenate([[0.0], np.cumsum(advance)[:-1]])
    within = (
        np.tile(np.arange(samples_per_symbol), tone.size)
        * np.repeat(tone, samples_per_symbol)
        * (2 * np.pi / sample_rate)
    )
    return np.exp(1j * (np.repeat(carried, samples_per_symbol) + within)).astype(
        np.complex64
    )


@pytest.mark.parametrize("period", [8, 13, 20, 50, 100, 200, 500, 1000])
def test_fsk_symbol_period_is_recovered(period):
    """Every supported period must be recovered exactly, not approximately.

    A period that is off by even one sample accumulates a drift of one symbol
    per `period` symbols, so the recovered bit stream develops insertions and
    the sync word stops matching. "Close" is not good enough here.
    """
    rng = np.random.default_rng(11)
    bits = rng.integers(0, 2, size=400, dtype=np.uint8)
    assert estimate_fsk_symbol_period(_fsk(bits, period)) == period


def test_fsk_period_of_real_capture_is_exact():
    """The project's own sample must recover the period the generator used."""
    from rf_analyzer.core.io import load_iq

    path = Path(__file__).resolve().parents[2] / "sample_data" / "fsk2.iq"
    if not path.exists():
        pytest.skip("run scripts/generate_test_data.py first")
    samples = load_iq(str(path), dtype="complex64")
    # 1 ms symbols at the generator's 100 kHz rate.
    assert estimate_fsk_symbol_period(samples) == 100


def test_fsk_period_declines_on_non_fsk_signals():
    """A wrong period is worse than no period, so non-FSK must return 1.

    Constant-modulus FSK is the only case the piecewise-constant model
    describes. BPSK/QPSK/noise/tone must not be assigned a period, because the
    pipeline would then decimate a signal that has no symbol structure to
    recover.
    """
    rng = np.random.default_rng(5)
    bits = rng.integers(0, 2, size=4096, dtype=np.uint8)
    bpsk = (2.0 * bits - 1.0).astype(np.complex64)
    qpsk = ((2.0 * bits[0::2] - 1.0) + 1j * (2.0 * bits[1::2] - 1.0)).astype(
        np.complex64
    ) / np.sqrt(2.0)
    noise = (rng.standard_normal(40_000) + 1j * rng.standard_normal(40_000)).astype(
        np.complex64
    )
    tone = generate_tone(freq=10_000, sample_rate=100_000, duration=0.4)

    for name, samples in (
        ("BPSK", bpsk),
        ("QPSK", qpsk),
        ("noise", noise),
        ("tone", tone),
    ):
        assert estimate_fsk_symbol_period(samples) == 1, name


def test_fsk_period_declines_on_short_capture():
    """Too few samples to establish any period: decline, do not guess."""
    assert estimate_fsk_symbol_period(np.array([], dtype=np.complex64)) == 1
    assert estimate_fsk_symbol_period(np.ones(8, dtype=np.complex64)) == 1


def test_fsk_period_declines_on_constant_signal():
    """A DC signal has no instantaneous frequency to model at all."""
    assert estimate_fsk_symbol_period(np.ones(4096, dtype=np.complex64)) == 1


# --------------------------------------------------------------------------- #
# Burst region
# --------------------------------------------------------------------------- #


def _burst_in_noise(pad: int = 800, core_len: int = 2_000, seed: int = 7):
    """``noise | burst | noise`` with a constant-envelope burst."""
    rng = np.random.default_rng(seed)
    core = np.ones(core_len, dtype=np.complex64)
    lead = (rng.standard_normal(pad) + 1j * rng.standard_normal(pad)) * 0.05
    trail = (rng.standard_normal(pad) + 1j * rng.standard_normal(pad)) * 0.05
    return np.concatenate([lead, core, trail]).astype(np.complex64), pad + core_len


def test_burst_region_finds_the_burst_and_never_lands_early():
    """The edge error must be one-sided: late is safe, early is not.

    A block counts as burst when its mean power clears the threshold, so a block
    holding even a little burst is included. That makes the estimate land at most
    one block early and up to one block late -- and late is the direction the
    decoder can absorb, because appending bits leaves the earlier
    de-interleaver blocks intact while truncating corrupts the last one.
    """
    samples, true_end = _burst_in_noise(pad=800)
    res = estimate_burst_region(samples)

    assert res["found"] is True
    assert res["method"] == "envelope-threshold"
    assert res["snr_db"] > 20.0
    assert res["start_sample"] <= 800
    assert true_end <= res["end_sample"] <= true_end + res["window"]


@pytest.mark.parametrize("pad", [200, 400, 800, 1_600, 4_000])
def test_burst_region_edge_error_is_bounded_by_one_window(pad):
    samples, true_end = _burst_in_noise(pad=pad)
    res = estimate_burst_region(samples)
    assert res["found"] is True
    assert true_end <= res["end_sample"] <= true_end + res["window"]


def test_burst_region_declines_on_a_capture_that_is_all_signal():
    """No noise gap means no burst, so the estimate must decline rather than
    report the whole capture as a burst and change how a bare frame decodes."""
    res = estimate_burst_region(np.ones(4_096, dtype=np.complex64))
    assert res["found"] is False
    assert res["end_sample"] is None
    assert res["reason"], "a decline must say why"


def test_burst_region_declines_on_pure_noise():
    rng = np.random.default_rng(5)
    noise = (rng.standard_normal(8_000) + 1j * rng.standard_normal(8_000)) * 0.05
    res = estimate_burst_region(noise.astype(np.complex64))
    assert res["found"] is False
    assert res["end_sample"] is None
    assert "no burst" in res["reason"]


def test_burst_region_declines_on_short_and_empty_captures():
    """Degenerate input must be answered, not raised on."""
    assert estimate_burst_region(np.array([], dtype=np.complex64))["found"] is False
    short = estimate_burst_region(np.ones(16, dtype=np.complex64))
    assert short["found"] is False
    assert "shorter than" in short["reason"]
    assert (
        estimate_burst_region(np.ones(512, dtype=np.complex64), window=0)["found"]
        is False
    )
