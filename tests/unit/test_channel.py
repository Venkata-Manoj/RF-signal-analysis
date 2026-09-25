"""Unit tests for the real-channel model (``core/channel.py``).

Each impairment is pinned to its definition, and each is shown to be
removable by the matching correction -- a channel the receiver cannot undo is
a channel the robustness numbers cannot be trusted on. The CMA test asserts a
*benefit* (lower modulus error), not a threshold, because equaliser
convergence is approximate; the CFO/IQ tests assert near-exact inversion
because those stages are closed-form.
"""

from __future__ import annotations

import numpy as np
import pytest

from rf_analyzer.core import channel as ch
from rf_analyzer.core import dsp
from rf_analyzer.core import receiver as rx
from rf_analyzer.core import waveform as wf


def _qpsk_symbols(n_bits: int = 4000, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=n_bits, dtype=np.uint8)
    return wf.bits_to_symbols(bits, "QPSK"), bits


def _modulus_error(samples: np.ndarray) -> float:
    s = np.asarray(samples, dtype=np.complex128).ravel()
    s = s / np.sqrt(float(np.mean(np.abs(s) ** 2)))
    return float(np.mean(np.abs(np.abs(s) ** 2 - 1.0)))


def test_defaults_are_identity():
    x = (
        np.random.default_rng(0).standard_normal(512)
        + 1j * np.random.default_rng(1).standard_normal(512)
    ).astype(np.complex64)
    y = ch.apply_channel(x, 1_000_000)
    assert y.shape == x.shape
    np.testing.assert_allclose(y, x, rtol=0, atol=0)


def test_empty_capture_stays_empty():
    y = ch.apply_channel(
        np.zeros(0, dtype=np.complex64), 1_000_000, cfo_hz=100.0, snr_db=10.0
    )
    assert y.size == 0


def test_bad_sample_rate_raises():
    with pytest.raises(ValueError):
        ch.apply_channel(np.ones(16, dtype=np.complex64), 0.0)


def test_cfo_applies_the_documented_phase_ramp():
    """A DC input under a CFO must equal the analytic complex exponential."""
    fs, cfo, n = 100_000.0, 5_000.0, 4_096
    y = ch.apply_channel(np.ones(n, dtype=np.complex64), fs, cfo_hz=cfo)
    t = np.arange(n, dtype=np.float64) / fs
    expected = np.exp(1j * 2.0 * np.pi * cfo * t)
    np.testing.assert_allclose(
        np.asarray(y, dtype=np.complex128), expected, rtol=0, atol=1e-5
    )


def test_cfo_correction_restores_a_tone():
    """The naive phase-slope estimator inverts a pure CFO on a tone."""
    fs, cfo, n = 100_000.0, 5_000.0, 10_000
    impaired = ch.apply_channel(np.ones(n, dtype=np.complex64), fs, cfo_hz=cfo)
    estimated = float(dsp.estimate_cfo(impaired, fs))
    assert estimated == pytest.approx(cfo, rel=0.02)
    corrected = dsp.correct_cfo(impaired, fs, estimated)
    residual = np.asarray(corrected, dtype=np.complex128)
    residual = residual / np.mean(residual)
    assert float(np.mean(np.abs(residual - 1.0))) < 1e-3


def test_cfo_and_doppler_invert_with_negated_parameters():
    """CFO + linear drift is a pure phase ramp, so negating it must undo it."""
    fs, n = 100_000.0, 8_000
    rng = np.random.default_rng(9)
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    impaired = ch.apply_channel(x, fs, cfo_hz=1_000.0, doppler_rate_hz_s=500.0)
    restored = ch.apply_channel(impaired, fs, cfo_hz=-1_000.0, doppler_rate_hz_s=-500.0)
    np.testing.assert_allclose(
        np.asarray(restored, dtype=np.complex128),
        np.asarray(x, dtype=np.complex128),
        rtol=0,
        atol=1e-4,
    )


def test_doppler_matches_the_quadratic_phase():
    fs, n = 100_000.0, 10_000
    y = ch.apply_channel(
        np.ones(n, dtype=np.complex64), fs, cfo_hz=1_000.0, doppler_rate_hz_s=500.0
    )
    t = np.arange(n, dtype=np.float64) / fs
    expected = np.exp(1j * 2.0 * np.pi * (1_000.0 * t + 0.5 * 500.0 * t * t))
    np.testing.assert_allclose(
        np.asarray(y, dtype=np.complex128), expected, rtol=0, atol=1e-5
    )


def test_multipath_fir_places_each_tap_at_its_delay():
    """An impulse through the channel must read back the tap gains."""
    impulse = np.zeros(16, dtype=np.complex64)
    impulse[0] = 1.0
    y = ch.apply_channel(
        impulse, 1_000_000, multipath=[(0, 1.0), (3, 0.5j), (5, -0.25)]
    )
    assert y[0] == pytest.approx(1.0)
    assert y[3] == pytest.approx(0.5j)
    assert y[5] == pytest.approx(-0.25)
    assert y[1] == pytest.approx(0.0)
    assert y[2] == pytest.approx(0.0)


def test_single_tap_multipath_is_identity():
    x = (
        np.random.default_rng(2).standard_normal(256)
        + 1j * np.random.default_rng(4).standard_normal(256)
    ).astype(np.complex64)
    y = ch.apply_channel(x, 1_000_000, multipath=[(0, 1.0)])
    np.testing.assert_allclose(y, x, rtol=0, atol=0)


def test_cma_equaliser_benefits_a_multipath_capture():
    """CMA must tighten the modulus of a 3-tap ISI channel (the docstring case).

    Asserts a *reduction* in modulus error, not an absolute value: the
    equaliser is approximate and its exact landing point depends on the data,
    but on this channel it improves on every seed tried (0.38 -> ~0.23).
    """
    syms, _ = _qpsk_symbols()
    impaired = ch.apply_channel(
        syms, 1_000_000, multipath=[(0, 1.0), (1, 0.3), (2, 0.2)]
    )
    before = _modulus_error(impaired)
    equalised, info = rx.cma_equalize(impaired)
    after = _modulus_error(equalised)
    assert after < before, f"CMA did not help: {before:.3f} -> {after:.3f}"
    assert (
        after < 0.9 * before
    ), f"CMA benefit too small to matter: {before:.3f} -> {after:.3f}"


def test_iq_metrics_detect_imbalance_and_correction_removes_it():
    """Blind metrics must fire on the impaired capture and clear after fix."""
    syms, _ = _qpsk_symbols()
    clean = ch.iq_imbalance_metrics(syms)
    assert clean["magnitude"] < 0.10, clean

    impaired = ch.apply_channel(syms, 1_000_000, iq_gain_imb=0.3, iq_phase_imb_deg=10.0)
    bad = ch.iq_imbalance_metrics(impaired)
    assert bad["magnitude"] > 0.20, bad
    assert abs(bad["gain_metric"]) > abs(clean["gain_metric"]) + 0.10

    # The blind estimator recovers the settings well enough to matter ...
    estimate = ch.estimate_iq_imbalance(impaired)
    assert estimate["iq_gain_imb"] == pytest.approx(0.3, abs=0.08)
    # ... and the exact inverse with the known settings restores the waveform.
    restored = ch.correct_iq_imbalance(impaired, 0.3, 10.0)
    np.testing.assert_allclose(
        np.asarray(restored, dtype=np.complex128),
        np.asarray(syms, dtype=np.complex128),
        rtol=0,
        atol=1e-4,
    )
    fixed = ch.iq_imbalance_metrics(restored)
    assert fixed["magnitude"] < 0.10, fixed


def test_iq_correction_rejects_non_invertible_parameters():
    x = np.ones(64, dtype=np.complex64)
    with pytest.raises(ValueError):
        ch.correct_iq_imbalance(x, 2.0, 0.0)
    with pytest.raises(ValueError):
        ch.correct_iq_imbalance(x, 0.0, 90.0)


def test_awgn_is_seeded_and_last():
    """Same seed replays, a new seed differs, and None means no noise."""
    syms, _ = _qpsk_symbols()
    noiseless = ch.apply_channel(syms, 1_000_000, snr_db=None, seed=7)
    np.testing.assert_allclose(noiseless, syms, rtol=0, atol=1e-5)

    first = ch.apply_channel(syms, 1_000_000, snr_db=10.0, seed=7)
    replay = ch.apply_channel(syms, 1_000_000, snr_db=10.0, seed=7)
    other = ch.apply_channel(syms, 1_000_000, snr_db=10.0, seed=8)
    np.testing.assert_array_equal(first, replay)
    assert not np.array_equal(first, other)
    assert (
        float(
            np.mean(
                np.abs(
                    np.asarray(first, dtype=np.complex128)
                    - np.asarray(syms, dtype=np.complex128)
                )
                ** 2
            )
        )
        > 0.0
    )


def test_full_channel_is_deterministic():
    syms, _ = _qpsk_symbols()
    kwargs = {
        "cfo_hz": 800.0,
        "doppler_rate_hz_s": 200.0,
        "multipath": [(0, 1.0), (2, 0.4)],
        "iq_gain_imb": 0.15,
        "iq_phase_imb_deg": 5.0,
        "snr_db": 15.0,
        "seed": 11,
    }
    first = ch.apply_channel(syms, 1_000_000, **kwargs)
    second = ch.apply_channel(syms, 1_000_000, **kwargs)
    np.testing.assert_array_equal(first, second)
