"""BPSK/QPSK/2-FSK demodulation. Full spec: info.md §12.3.

MVP is intentionally naive (phase-aligned, no timing recovery).
Robust sync is V2 — do not add Costas loops here.
"""

from __future__ import annotations

import numpy as np


def normalize_signal(samples: np.ndarray) -> np.ndarray:
    """
    Normalize signal amplitude.
    """
    samples = np.asarray(samples)
    if samples.size == 0:
        return samples
    max_val = np.max(np.abs(samples))
    if max_val == 0:
        return samples
    return samples / max_val


def demod_bpsk(samples: np.ndarray) -> np.ndarray:
    """
    Simple BPSK demodulation.
    Assumes phase-aligned signal.
    """
    samples = normalize_signal(samples)
    bits = (np.real(samples) > 0).astype(np.uint8)
    return bits


def demod_qpsk(samples: np.ndarray) -> np.ndarray:
    """
    Simple QPSK demodulation.
    Assumes phase-aligned signal.
    Vectorized: bit_i = real > 0, bit_q = imag > 0, interleaved I,Q.
    """
    samples = normalize_signal(samples)
    bit_i = (np.real(samples) > 0).astype(np.uint8)
    bit_q = (np.imag(samples) > 0).astype(np.uint8)
    bits = np.empty(bit_i.size * 2, dtype=np.uint8)
    bits[0::2] = bit_i
    bits[1::2] = bit_q
    return bits


def demod_2fsk(samples: np.ndarray) -> np.ndarray:
    """
    Simple 2-FSK demodulation using instantaneous frequency.
    """
    samples = normalize_signal(samples)

    phase = np.angle(samples)
    phase_unwrapped = np.unwrap(phase)
    inst_freq = np.diff(phase_unwrapped)

    threshold = 0.0
    bits = (inst_freq > threshold).astype(np.uint8)

    return bits
