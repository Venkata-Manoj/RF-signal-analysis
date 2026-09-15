"""Sync-word correlation. Full spec: info.md §12.4."""

from __future__ import annotations

import numpy as np


def hex_to_bits(hex_string: str, bit_length: int | None = None) -> np.ndarray:
    """
    Convert hex string to bit array.

    Example:
      "0x1A" -> [0,0,0,1,1,0,1,0]
    """
    hex_string = hex_string.lower().strip()

    hex_string = hex_string.removeprefix("0x")

    value = int(hex_string, 16)

    if bit_length is None:
        bit_length = len(hex_string) * 4

    bits = [(value >> (bit_length - 1 - i)) & 1 for i in range(bit_length)]
    return np.array(bits, dtype=np.uint8)


def sliding_correlate(
    received_bits: np.ndarray, known_bits: np.ndarray
) -> tuple[int, float]:
    """Sliding hard-bit correlation (vectorized).

    Uses bipolar cross-correlation via ``np.correlate`` for O(n·m)
    performance instead of the original O(n²) pure-Python loop.

    Returns:
      best_offset, best_score (score in [0, 1])
    """
    received_bits = np.asarray(received_bits, dtype=np.uint8)
    known_bits = np.asarray(known_bits, dtype=np.uint8)

    if len(known_bits) == 0:
        raise ValueError("Known bits must not be empty.")

    if len(received_bits) < len(known_bits):
        return -1, 0.0

    known_len = len(known_bits)

    # Convert to bipolar (0 → −1, 1 → +1) for cross-correlation
    rx = 2.0 * received_bits.astype(np.float32) - 1.0
    kn = 2.0 * known_bits.astype(np.float32) - 1.0

    corr = np.correlate(rx, kn, mode="valid")
    # Normalize to [0, 1]: perfect match → +known_len → score 1.0
    scores = (corr / known_len + 1.0) / 2.0
    best_offset = int(np.argmax(scores))
    best_score = float(scores[best_offset])

    return best_offset, best_score


def find_header(
    received_bits: np.ndarray, sync_bits: np.ndarray, threshold: float = 0.85
) -> dict:
    """
    Find header using sync bits.
    """
    received_bits = np.asarray(received_bits, dtype=np.uint8)
    sync_bits = np.asarray(sync_bits, dtype=np.uint8)

    offset, score = sliding_correlate(received_bits, sync_bits)

    return {
        "offset": offset,
        "score": score,
        "detected": score >= threshold,
        "sync_length": len(sync_bits),
    }
