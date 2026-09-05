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

    if hex_string.startswith("0x"):
        hex_string = hex_string[2:]

    value = int(hex_string, 16)

    if bit_length is None:
        bit_length = len(hex_string) * 4

    bits = [(value >> (bit_length - 1 - i)) & 1 for i in range(bit_length)]
    return np.array(bits, dtype=np.uint8)


def sliding_correlate(
    received_bits: np.ndarray, known_bits: np.ndarray
) -> tuple[int, float]:
    """
    Simple sliding hard-bit correlation.

    Returns:
      best_offset, best_score
    """
    received_bits = np.asarray(received_bits, dtype=np.uint8)
    known_bits = np.asarray(known_bits, dtype=np.uint8)

    if len(known_bits) == 0:
        raise ValueError("Known bits must not be empty.")

    if len(received_bits) < len(known_bits):
        return -1, 0.0

    best_offset = -1
    best_score = -1.0

    known_len = len(known_bits)

    for offset in range(len(received_bits) - known_len + 1):
        segment = received_bits[offset : offset + known_len]
        matches = np.sum(segment == known_bits)
        score = float(matches) / known_len

        if score > best_score:
            best_score = score
            best_offset = offset

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
        "sync_length": int(len(sync_bits)),
    }
