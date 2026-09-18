"""Sync-word correlation. Full spec: info.md §12.4."""

from __future__ import annotations

from math import comb

import numpy as np

# Target expected number of false alarms for one sync search.
#
# A fixed score threshold does *not* control false alarms, because the number
# of positions the sync word slides over is part of the statistics: at 0.85
# (>= 28 of 32 bits) a 500k-bit capture expects ~4.8 chance matches, so an
# analog or noise-only recording will "find" a header that is not there. The
# minimum match count is therefore derived from the search length.
FALSE_ALARM_TARGET = 0.01


def min_matches_for_length(
    sync_length: int, n_positions: int, target: float = FALSE_ALARM_TARGET
) -> int:
    """Smallest bit-match count whose expected false alarms stay <= ``target``.

    For ``sync_length`` bits compared at ``n_positions`` offsets, a random
    match of at least ``k`` bits happens with probability
    ``P(X >= k), X ~ Binomial(sync_length, 1/2)``. This returns the smallest
    ``k`` for which ``P * n_positions <= target``.

    Returns ``sync_length`` (an exact match required) when even that cannot
    meet the target, which is the honest answer for a very short sync word on
    a long capture.
    """
    if sync_length <= 0:
        return 0
    if n_positions <= 0:
        return sync_length
    # P(>= k matches) shrinks as k grows, so the *smallest* k that meets the
    # target is the most permissive threshold that still controls false
    # alarms. Iterate upward and return the first hit.
    for matches in range(1, sync_length + 1):
        tail = sum(comb(sync_length, j) for j in range(matches, sync_length + 1))
        if tail / 2.0**sync_length * n_positions <= target:
            return matches
    return sync_length


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
    """Find header using sync bits.

    ``detected`` requires *both* the caller's score threshold and statistical
    significance. The effective threshold is raised on long captures so the
    expected number of chance matches stays at or below
    :data:`FALSE_ALARM_TARGET`; the returned ``min_score`` / ``min_matches`` /
    ``positions_searched`` fields make that adjustment inspectable instead of
    hidden.
    """
    received_bits = np.asarray(received_bits, dtype=np.uint8)
    sync_bits = np.asarray(sync_bits, dtype=np.uint8)

    offset, score = sliding_correlate(received_bits, sync_bits)

    sync_length = len(sync_bits)
    n_positions = max(0, int(received_bits.size) - sync_length + 1)
    min_matches = min_matches_for_length(sync_length, n_positions)
    min_score = (min_matches / sync_length) if sync_length else 0.0
    effective = max(float(threshold), min_score)

    return {
        "offset": offset,
        "score": score,
        "detected": score >= effective,
        "sync_length": sync_length,
        # Statistical significance context (see FALSE_ALARM_TARGET).
        "min_score": float(min_score),
        "min_matches": int(min_matches),
        "positions_searched": int(n_positions),
        "false_alarm_target": FALSE_ALARM_TARGET,
    }
