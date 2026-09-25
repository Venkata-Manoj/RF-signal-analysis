"""Unit pins for the coherent receive chain building blocks."""

from __future__ import annotations

import numpy as np

from rf_analyzer.core import receiver as rx
from rf_analyzer.core import waveform as wf


def test_slice_linear_matches_waveform_round_trip():
    for scheme in ("BPSK", "QPSK", "8PSK", "16-QAM", "64-QAM"):
        bits = np.random.default_rng(0).integers(0, 2, size=240, dtype=np.uint8)
        # Truncate to a whole number of symbols.
        bps = wf.bits_per_symbol(scheme)
        bits = bits[: (bits.size // bps) * bps]
        symbols = wf.bits_to_symbols(bits, scheme)
        recovered = rx._slice_linear(symbols, scheme)
        assert np.array_equal(recovered, bits), scheme


def test_estimate_sps_finds_oversampling_and_abstains_on_noise():
    samples, _ = wf.synth("QPSK", 2000, sps=8, seed=0)
    found = rx.estimate_sps(samples, "QPSK")
    assert found["found"] is True
    assert found["sps"] == 8

    noise = (
        np.random.default_rng(1).standard_normal(4096)
        + 1j * np.random.default_rng(2).standard_normal(4096)
    ).astype(np.complex64)
    missed = rx.estimate_sps(noise, "QPSK")
    assert missed["found"] is False
    assert missed["sps"] is None


def test_ber_best_rotation_reads_half_on_uncorrelated():
    a = np.random.default_rng(0).integers(0, 2, size=2000, dtype=np.uint8)
    b = np.random.default_rng(1).integers(0, 2, size=2000, dtype=np.uint8)
    ber = rx.ber_best_rotation(a, b, "BPSK")
    assert ber is not None
    assert 0.4 < ber < 0.6
