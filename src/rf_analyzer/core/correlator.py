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


# --------------------------------------------------------------------------- #
# Sync-word discovery (auto-sync for unknown captures)
# --------------------------------------------------------------------------- #

#: 32-bit sync words scanned by :func:`discover_sync_word`. All candidates are
#: 32 bits long on purpose: a shorter sync cannot meet
#: :data:`FALSE_ALARM_TARGET` on a realistic capture (even an exact 7-bit
#: Barker match is expected ~60 times on a 500k-bit stream), so scanning one
#: would manufacture headers out of noise. Barker codes are therefore
#: deliberately *not* in this list -- see the docstring of
#: :func:`discover_sync_word`.
SYNC_DISCOVERY_CANDIDATES: tuple[str, ...] = (
    "0x1ACFFC1D",  # project default (CCSDS-like)
    "0x55555555",  # alternating 0101... preamble
    "0xAAAAAAAA",  # alternating 1010... preamble (inverted)
    "0x7E7E7E7E",  # HDLC flag (01111110) repeated
)


def discover_sync_word(
    received_bits: np.ndarray,
    candidates: tuple[str, ...] | list[str] | None = None,
    threshold: float = 0.85,
) -> dict:
    """Scan common sync words and return the one that correlates significantly.

    Each candidate is tested with :func:`find_header`, which already applies
    the CFAR gate (``min_score`` from ``min_matches_for_length`` with
    :data:`FALSE_ALARM_TARGET`). A candidate only counts when ``detected`` is
    True -- a high raw ``score`` below its length-aware ``min_score`` is a
    chance match on a long capture, not a header.

    Among the candidates that pass, the winner is the one with the largest
    ``score - min_score`` margin (significance beyond chance), then the
    largest raw score. The margin is what makes different candidates
    comparable: a fixed 0.85 threshold is weaker evidence on a long capture
    than a short one.

    Honesty contract: short sync words (Barker-7/11/13) are excluded by
    default because they *cannot* meet the false-alarm target -- an exact
    13-bit match is still expected ~61 times on a 500k-bit capture. Scanning
    them would guarantee false headers. Callers that know their capture is
    short may pass an explicit ``candidates`` list, but the CFAR gate still
    applies.

    Returns:
        ``{"sync_word", "sync_bits", "offset", "score", "min_score",
        "min_matches", "positions_searched", "sync_length", "detected",
        "margin", "candidates_tried"}``. ``sync_word`` is None and
        ``detected`` False when nothing passed the gate; ``candidates_tried``
        records every candidate's score/min_score/detected triple so the
        decision is inspectable rather than silent.
    """
    received = np.asarray(received_bits, dtype=np.uint8).ravel()
    names: tuple[str, ...] | list[str] = (
        candidates if candidates is not None else SYNC_DISCOVERY_CANDIDATES
    )

    tried: list[dict] = []
    best: dict | None = None
    best_margin = float("-inf")

    for name in names:
        label = str(name).strip()
        try:
            sync_bits = hex_to_bits(label)
        except Exception:
            tried.append(
                {
                    "sync_word": label,
                    "score": 0.0,
                    "min_score": 0.0,
                    "detected": False,
                    "error": "unparseable hex sync word",
                }
            )
            continue
        if received.size < sync_bits.size:
            tried.append(
                {
                    "sync_word": label,
                    "score": 0.0,
                    "min_score": 1.0,
                    "detected": False,
                    "error": "bit stream shorter than sync word",
                }
            )
            continue
        found = find_header(received, sync_bits, threshold=threshold)
        detected = bool(found.get("detected", False))
        score = float(found.get("score", 0.0))
        min_score = float(found.get("min_score", 0.0))
        margin = score - min_score
        tried.append(
            {
                "sync_word": label,
                "score": score,
                "min_score": min_score,
                "min_matches": int(found.get("min_matches", 0)),
                "positions_searched": int(found.get("positions_searched", 0)),
                "offset": int(found.get("offset", -1)),
                "detected": detected,
                "margin": float(margin),
            }
        )
        if detected and margin > best_margin:
            best_margin = margin
            best = {
                "sync_word": label,
                "sync_bits": np.asarray(sync_bits, dtype=np.uint8),
                "offset": int(found.get("offset", -1)),
                "score": score,
                "min_score": min_score,
                "min_matches": int(found.get("min_matches", 0)),
                "positions_searched": int(found.get("positions_searched", 0)),
                "sync_length": int(sync_bits.size),
                "detected": True,
                "margin": float(margin),
            }
        elif detected and margin == best_margin and best is not None:
            # Tie-break deterministically: higher raw score, then longer word.
            if score > best["score"] or (
                score == best["score"]
                and int(sync_bits.size) > int(best["sync_length"])
            ):
                best = {
                    "sync_word": label,
                    "sync_bits": np.asarray(sync_bits, dtype=np.uint8),
                    "offset": int(found.get("offset", -1)),
                    "score": score,
                    "min_score": min_score,
                    "min_matches": int(found.get("min_matches", 0)),
                    "positions_searched": int(found.get("positions_searched", 0)),
                    "sync_length": int(sync_bits.size),
                    "detected": True,
                    "margin": float(margin),
                }

    if best is None:
        # Nothing passed the CFAR gate. Report the closest miss as context so
        # "no header" does not read as "never looked".
        closest: dict | None = None
        for entry in tried:
            if entry.get("error"):
                continue
            if closest is None or float(entry.get("margin", -9.0)) > float(
                closest.get("margin", -9.0)
            ):
                closest = entry
        return {
            "sync_word": None,
            "sync_bits": None,
            "offset": int(closest.get("offset", -1)) if closest else -1,
            "score": float(closest.get("score", 0.0)) if closest else 0.0,
            "min_score": float(closest.get("min_score", 0.0)) if closest else 0.0,
            "min_matches": int(closest.get("min_matches", 0)) if closest else 0,
            "positions_searched": (
                int(closest.get("positions_searched", 0)) if closest else 0
            ),
            "sync_length": 0,
            "detected": False,
            "false_alarm_target": FALSE_ALARM_TARGET,
            "margin": float(closest.get("margin", 0.0)) if closest else 0.0,
            "candidates_tried": tried,
        }

    return {
        **best,
        "false_alarm_target": FALSE_ALARM_TARGET,
        "candidates_tried": tried,
    }
