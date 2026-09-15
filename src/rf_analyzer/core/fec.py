"""FEC candidate scoring — NOT blind detection. See info.md §30/§32 V2."""

from __future__ import annotations

import numpy as np

CANDIDATES = ["Convolutional r=1/2 K=7", "RS(255,223)", "LDPC", "Concatenated"]


def _parity_score(bits: np.ndarray) -> float:
    if len(bits) < 16:
        return 0.0
    # Even-parity drift proxy for conv code.
    fails = int(np.sum(bits[1:] != bits[:-1]))
    return float(max(0.0, min(0.5, 0.35 - fails / len(bits) * 0.2)))


def score_fec_candidates(bits: np.ndarray) -> dict:
    """Return candidate FEC scores, confidence capped at 0.5 (never blind)."""
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    if bits.size == 0:
        return {
            "candidate": None,
            "confidence": 0.0,
            "crc_pass": None,
            "candidates": [],
        }
    scores = [
        {
            "candidate": "Convolutional r=1/2 K=7",
            "confidence": _parity_score(bits),
            "crc_pass": None,
        },
        {
            "candidate": "RS(255,223)",
            "confidence": 0.25 if len(bits) % 8 == 0 else 0.15,
            "crc_pass": None,
        },
        {"candidate": "LDPC", "confidence": 0.10, "crc_pass": None},
        {"candidate": "Concatenated", "confidence": 0.12, "crc_pass": None},
    ]
    best = max(scores, key=lambda x: x["confidence"])
    return {
        "candidate": best["candidate"],
        "confidence": float(best["confidence"]),
        "crc_pass": best["crc_pass"],
        "candidates": scores,
    }
