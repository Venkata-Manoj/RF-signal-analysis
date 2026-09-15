"""Unit tests for naive demodulators (info.md §17.3).

2-FSK helper mirrors info.md §15.1 generate_2fsk exactly
(100 samples/symbol at default rate/duration, continuous phase).
"""

from __future__ import annotations

import numpy as np

from rf_analyzer.core.demod import (
    bits_to_bpsk,
    bits_to_qam16,
    demod_2fsk,
    demod_bpsk,
    demod_qam16,
    demod_qpsk,
)


def test_bpsk_demod_clean():
    # Arrange
    bits = np.array([0, 1, 0, 1, 1], dtype=np.uint8)
    symbols = bits_to_bpsk(bits).astype(np.complex64)
    # Act
    demod_bits = demod_bpsk(symbols)
    # Assert
    assert np.array_equal(demod_bits, bits)


def test_qpsk_demod_clean():
    # Arrange: bits as I,Q pairs.
    bits = np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8)
    i_bits = bits[0::2]
    q_bits = bits[1::2]
    i_sym = bits_to_bpsk(i_bits)
    q_sym = bits_to_bpsk(q_bits)
    samples = (i_sym + 1j * q_sym) / np.sqrt(2.0)
    samples = samples.astype(np.complex64)
    # Act
    demod_bits = demod_qpsk(samples)
    # Assert
    assert np.array_equal(demod_bits, bits)


def test_qam16_roundtrip_clean():
    # Arrange
    rng = np.random.default_rng(42)
    bits = rng.integers(0, 2, size=400, dtype=np.uint8)  # multiple of 4
    symbols = bits_to_qam16(bits)
    # Act
    recovered = demod_qam16(symbols)
    # Assert: naive phase-aligned must be bit-exact on clean symbols
    assert np.array_equal(recovered[: len(bits)], bits)


def test_qam16_alias_exists():
    from rf_analyzer.core import demod as m

    assert hasattr(m, "demod_qam16")
    assert hasattr(m, "demod_16qam")
    assert hasattr(m, "demod_qam")


def _generate_2fsk(
    bits: np.ndarray,
    sample_rate: float = 100_000,
    symbol_duration: float = 0.001,
    freq_low: float = -5000,
    freq_high: float = 5000,
) -> tuple[np.ndarray, int]:
    """Same logic as info.md §15.1 generate_2fsk (returns samples + sps)."""
    samples_per_symbol = int(sample_rate * symbol_duration)
    phase = 0.0
    samples = []
    for bit in bits:
        freq = freq_high if bit == 1 else freq_low
        t = np.arange(samples_per_symbol) / sample_rate
        segment = np.exp(1j * (2 * np.pi * freq * t + phase))
        samples.append(segment)
        phase = np.angle(segment[-1])
    return np.concatenate(samples).astype(np.complex64), samples_per_symbol


def test_2fsk_demod_roundtrip_clean():
    # Arrange: deterministic bits, 100 sps, clean (high-SNR) signal.
    rng = np.random.default_rng(42)
    bits = rng.integers(0, 2, size=64).astype(np.uint8)
    samples, sps = _generate_2fsk(bits)
    assert sps == 100

    # Act: per-symbol instantaneous-frequency demod (1 bit per symbol).
    raw_bits = np.asarray(demod_2fsk(samples, samples_per_symbol=sps), dtype=np.uint8)
    # Assert: clean roundtrip recovers every bit.
    assert np.array_equal(raw_bits, bits)
