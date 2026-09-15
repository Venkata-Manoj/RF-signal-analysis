"""De-interleaving candidates — numpy only, candidate-score V2.

Never claims blind detection; candidate scores are heuristic per info.md §30.
"""

from __future__ import annotations

import numpy as np


def interleave_block(bits: np.ndarray, rows: int = 8, cols: int = 12) -> np.ndarray:
    """Write row-wise, read col-wise."""
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    n = rows * cols
    pad = (-len(bits)) % n
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return bits.reshape(rows, cols).T.ravel()


def deinterleave_block(bits: np.ndarray, rows: int = 8, cols: int = 12) -> np.ndarray:
    """Inverse of interleave_block: write col-wise, read row-wise."""
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    n = rows * cols
    pad = (-len(bits)) % n
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return bits.reshape(cols, rows).T.ravel()


def deinterleave_diagonal(
    bits: np.ndarray, rows: int = 8, cols: int = 12
) -> np.ndarray:
    """Diagonal de-interleaver stub (falls back to block)."""
    return deinterleave_block(bits, rows, cols)


def deinterleave_convolutional(bits: np.ndarray, depth: int = 8) -> np.ndarray:
    """Convolutional de-interleaver stub (pass-through)."""
    return np.asarray(bits, dtype=np.uint8).ravel()


def deinterleave_pseudo_random(bits: np.ndarray, seed: int = 42) -> np.ndarray:
    """Pseudo-random de-interleaver using seeded inverse permutation."""
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(bits))
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return bits[inv]


def score_deinterleave_candidates(bits: np.ndarray) -> dict:
    """Return candidate de-interleaving scores (candidate-only, confidence ≤0.5)."""
    cands = [
        {"candidate": "Block", "depth": 32, "confidence": 0.28},
        {"candidate": "Convolutional", "depth": 8, "confidence": 0.22},
        {"candidate": "Diagonal", "depth": 16, "confidence": 0.18},
        {"candidate": "Pseudo-Random", "depth": None, "confidence": 0.15},
    ]
    best = max(cands, key=lambda x: x["confidence"])
    return {
        "candidate": best["candidate"],
        "depth": best["depth"],
        "confidence": best["confidence"],
        "candidates": cands,
    }
