"""Unit tests for naive demodulators (info.md §17.3).

2-FSK helper mirrors info.md §15.1 generate_2fsk exactly
(100 samples/symbol at default rate/duration, continuous phase).
"""

from __future__ import annotations

import numpy as np

from rf_analyzer.core.demod import demod_2fsk, demod_bpsk, demod_qpsk


def test_bpsk_demod_clean():
    # Arrange
    bits = np.array([0, 1, 0, 1, 1], dtype=np.uint8)
    symbols = 2.0 * bits.astype(np.float32) - 1.0
    samples = symbols.astype(np.complex64)

    # Act
    demod_bits = demod_bpsk(samples)

    # Assert
    assert np.array_equal(demod_bits, bits)


def test_qpsk_demod_clean():
    # Arrange: bits as I,Q pairs.
    bits = np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8)

    i_bits = bits[0::2]
    q_bits = bits[1::2]

    i_sym = 2.0 * i_bits.astype(np.float32) - 1.0
    q_sym = 2.0 * q_bits.astype(np.float32) - 1.0

    samples = (i_sym + 1j * q_sym) / np.sqrt(2.0)
    samples = samples.astype(np.complex64)

    # Act
    demod_bits = demod_qpsk(samples)

    # Assert
    assert np.array_equal(demod_bits, bits)


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

    # Act: per-sample FM demod, then majority vote per symbol window.
    raw_bits = np.asarray(demod_2fsk(samples), dtype=np.uint8)
    voted = np.zeros(len(bits), dtype=np.uint8)
    for i in range(len(bits)):
        window = raw_bits[i * sps : (i + 1) * sps]
        voted[i] = 1 if float(np.mean(window)) > 0.5 else 0

    # Assert: clean roundtrip recovers every bit.
    assert np.array_equal(voted, bits)
