"""Unit tests for de-interleaving candidates (SIH26147)."""

from __future__ import annotations

import numpy as np

from rf_analyzer.core.deinterleave import (
    deinterleave_block,
    deinterleave_convolutional,
    deinterleave_diagonal,
    deinterleave_pseudo_random,
    interleave_block,
    score_deinterleave_candidates,
)


def test_block_roundtrip():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=96, dtype=np.uint8)
    inter = interleave_block(bits, rows=8, cols=12)
    rec = deinterleave_block(inter, rows=8, cols=12)
    assert np.array_equal(rec[: len(bits)], bits)


def test_diagonal_returns_same_shape():
    bits = np.array([0, 1] * 48, dtype=np.uint8)
    rec = deinterleave_diagonal(bits, rows=8, cols=12)
    assert len(rec) == len(bits)


def test_convolutional_pass_through():
    bits = np.array([0, 1] * 48, dtype=np.uint8)
    rec = deinterleave_convolutional(bits, depth=8)
    assert np.array_equal(rec, bits)


def test_pseudo_random_roundtrip():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=100, dtype=np.uint8)
    # Apply pseudo-random then undo with same seed
    rng2 = np.random.default_rng(42)
    perm = rng2.permutation(len(bits))
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    inter = bits[perm]
    rec = deinterleave_pseudo_random(inter, seed=42)
    assert np.array_equal(rec, bits)


def test_deinterleave_confidence_cap():
    res = score_deinterleave_candidates(np.array([0, 1] * 50, dtype=np.uint8))
    assert res["confidence"] <= 0.5
    assert "candidate" in res
    assert "depth" in res
