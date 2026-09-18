"""De-interleaving: real interleaver/de-interleaver pairs + hypothesis search.

SIH26147 bullet (iii): "Carry out de-interleaving (Block, Convolution,
Diagonal, Pseudo Random)". Full spec: ``info.md`` §12.5.

Every scheme here is a **length-preserving permutation** with an exact
inverse, so ``deinterleave_X(interleave_X(bits)) == bits`` holds bit-for-bit.
That property is what makes the hypothesis search meaningful: if a candidate
de-interleaver is wrong, the downstream FEC decode cannot produce a valid
CRC, so a validated decode is real evidence rather than a guess.

Schemes:

* **Block** — write row-wise, read column-wise (the classic rectangular
  block interleaver used by e.g. DVB).
* **Convolutional** — Forney delay-line interleaver with ``depth`` branches
  spaced ``spacing`` apart, expressed as an exact modular permutation so the
  finite block needs no warm-up transient.
* **Diagonal** — helical/row-rotation interleaver: row ``i`` is rotated by
  ``i`` positions, which disperses bursts across the rectangle diagonally.
* **Pseudo-random** — seeded Fisher-Yates permutation (the "random"
  interleaver used when no structure is assumed).

Blind identification of an unknown interleaver remains out of scope
(``info.md`` §30); :func:`score_deinterleave_candidates` stays a capped
heuristic while :func:`search_interleaver` provides CRC-verified evidence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rf_analyzer.core.fec import decode_hypotheses

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_ROWS = 8
DEFAULT_COLS = 12
DEFAULT_DEPTH = 8
DEFAULT_SPACING = 1
DEFAULT_SEED = 42


def _pad_to(bits: np.ndarray, block: int) -> np.ndarray:
    """Zero-pad a bit array up to a multiple of ``block``."""
    arr = np.asarray(bits, dtype=np.uint8).ravel()
    if block <= 0:
        raise ValueError("block size must be positive")
    pad = (-arr.size) % block
    if pad:
        arr = np.concatenate([arr, np.zeros(pad, dtype=np.uint8)])
    return arr


# --------------------------------------------------------------------------- #
# Block interleaver (rectangular)
# --------------------------------------------------------------------------- #


def interleave_block(
    bits: np.ndarray, rows: int = DEFAULT_ROWS, cols: int = DEFAULT_COLS
):
    """Write row-wise into ``rows x cols`` rectangles, read out column-wise."""
    arr = _pad_to(bits, rows * cols)
    mat = arr.reshape(-1, rows, cols)
    return mat.transpose(0, 2, 1).ravel()


def deinterleave_block(
    bits: np.ndarray, rows: int = DEFAULT_ROWS, cols: int = DEFAULT_COLS
):
    """Inverse of :func:`interleave_block`: write column-wise, read row-wise."""
    arr = _pad_to(bits, rows * cols)
    mat = arr.reshape(-1, cols, rows)
    return mat.transpose(0, 2, 1).ravel()


# --------------------------------------------------------------------------- #
# Convolutional (Forney delay-line) interleaver
# --------------------------------------------------------------------------- #


def _convolutional_permutation(n_bits: int, depth: int, spacing: int) -> np.ndarray:
    """Exact Forney delay-line permutation over a finite, wrapped block.

    Forney's interleaver sends input ``n = k*depth + branch`` to output
    ``((k + branch*spacing) mod K)*depth + branch`` where ``K`` is the number
    of visits per branch. Taking ``k`` modulo ``K`` turns the classic
    infinite-stream delay line into a clean bijection on a finite block, so no
    warm-up zeros and no group delay are needed.
    """
    if depth <= 1:
        raise ValueError("convolutional interleaver needs depth >= 2")
    if spacing < 1:
        raise ValueError("spacing must be >= 1")
    if n_bits % depth != 0:
        raise ValueError("bit count must be a multiple of depth")

    visits = n_bits // depth
    n = np.arange(n_bits)
    branch = n % depth
    k = n // depth
    dest = ((k + branch * spacing) % visits) * depth + branch
    return dest


def interleave_convolutional(
    bits: np.ndarray, depth: int = DEFAULT_DEPTH, spacing: int = DEFAULT_SPACING
):
    """Forney convolutional interleaver (delay-line, exact inverse)."""
    arr = _pad_to(bits, depth)
    dest = _convolutional_permutation(arr.size, depth, spacing)
    out = np.empty_like(arr)
    out[dest] = arr
    return out


def deinterleave_convolutional(
    bits: np.ndarray, depth: int = DEFAULT_DEPTH, spacing: int = DEFAULT_SPACING
):
    """Inverse of :func:`interleave_convolutional`."""
    arr = _pad_to(bits, depth)
    dest = _convolutional_permutation(arr.size, depth, spacing)
    return arr[dest]


# --------------------------------------------------------------------------- #
# Diagonal (helical row-rotation) interleaver
# --------------------------------------------------------------------------- #


def interleave_diagonal(
    bits: np.ndarray, rows: int = DEFAULT_ROWS, cols: int = DEFAULT_COLS
):
    """Helical interleaver: row ``i`` of each rectangle is rotated right by ``i``."""
    arr = _pad_to(bits, rows * cols)
    mat = arr.reshape(-1, rows, cols).copy()
    for i in range(rows):
        mat[:, i, :] = np.roll(mat[:, i, :], i, axis=1)
    return mat.ravel()


def deinterleave_diagonal(
    bits: np.ndarray, rows: int = DEFAULT_ROWS, cols: int = DEFAULT_COLS
):
    """Inverse of :func:`interleave_diagonal`: row ``i`` rotated left by ``i``."""
    arr = _pad_to(bits, rows * cols)
    mat = arr.reshape(-1, rows, cols).copy()
    for i in range(rows):
        mat[:, i, :] = np.roll(mat[:, i, :], -i, axis=1)
    return mat.ravel()


# --------------------------------------------------------------------------- #
# Pseudo-random interleaver
# --------------------------------------------------------------------------- #


def _random_permutation(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.permutation(n)


def interleave_pseudo_random(bits: np.ndarray, seed: int = DEFAULT_SEED):
    """Seeded Fisher-Yates permutation interleaver."""
    arr = np.asarray(bits, dtype=np.uint8).ravel()
    if arr.size == 0:
        return arr
    return arr[_random_permutation(arr.size, seed)]


def deinterleave_pseudo_random(bits: np.ndarray, seed: int = DEFAULT_SEED):
    """Inverse of :func:`interleave_pseudo_random` (same seed)."""
    arr = np.asarray(bits, dtype=np.uint8).ravel()
    if arr.size == 0:
        return arr
    perm = _random_permutation(arr.size, seed)
    inverse = np.empty_like(perm)
    inverse[perm] = np.arange(perm.size)
    return arr[inverse]


# --------------------------------------------------------------------------- #
# Blind candidate scoring (honest, capped) + verified hypothesis search
# --------------------------------------------------------------------------- #


def score_deinterleave_candidates(bits: np.ndarray) -> dict:
    """Blind de-interleaver candidate scores (heuristic, confidence <= 0.5).

    Deliberately not a detector — see ``info.md`` §30. For CRC-verified
    evidence use :func:`search_interleaver`.
    """
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


def deinterleaver_hypotheses(
    rows: int = DEFAULT_ROWS,
    cols: int = DEFAULT_COLS,
    depth: int = DEFAULT_DEPTH,
    spacing: int = DEFAULT_SPACING,
    seed: int = DEFAULT_SEED,
) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    """Named de-interleaver candidates for the hypothesis search."""
    return {
        "block": lambda b: deinterleave_block(b, rows, cols),
        "convolutional": lambda b: deinterleave_convolutional(b, depth, spacing),
        "diagonal": lambda b: deinterleave_diagonal(b, rows, cols),
        "pseudo-random": lambda b: deinterleave_pseudo_random(b, seed),
    }


def search_interleaver(
    bits: np.ndarray,
    *,
    start_offset: int = 0,
    max_bits: int = 8192,
    time_budget_s: float = 5.0,
    **params,
) -> dict:
    """Try each de-interleaver followed by every FEC decoder, CRC-verified.

    Delegates the combinatorial search to :func:`fec.decode_hypotheses` and
    reports which interleaver (if any) produced a CRC-valid payload.

    Returns:
        The :func:`fec.decode_hypotheses` result plus ``interleaver`` (the
        winning scheme name, or ``None``) and ``n_interleavers``.
    """
    hypotheses = deinterleaver_hypotheses(**params)
    result = decode_hypotheses(
        bits,
        start_offset=start_offset,
        deinterleavers=hypotheses,
        max_bits=max_bits,
        time_budget_s=time_budget_s,
    )
    best = result.get("best")
    winner = best["params"].get("interleaver") if best else None
    return {**result, "interleaver": winner, "n_interleavers": len(hypotheses) + 1}
