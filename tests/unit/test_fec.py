"""Unit tests for FEC candidate scoring (SIH26147)."""

from __future__ import annotations

import numpy as np

from rf_analyzer.core.fec import score_fec_candidates


def test_fec_scoring_never_claims_blind():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=200, dtype=np.uint8)
    res = score_fec_candidates(bits)
    assert "candidates" in res
    assert "candidate" in res
    assert "confidence" in res
    assert "crc_pass" in res
    for c in res["candidates"]:
        assert (
            0.0 <= c["confidence"] <= 0.5
        ), f"confidence {c['confidence']} exceeds stub cap"
        assert c["crc_pass"] is None
    # top candidate field matches report schema §13
    assert res["candidate"] is None or isinstance(res["candidate"], str)
    # confidence never exceeds 0.5 (never claims blind detection)
    assert res["confidence"] <= 0.5


def test_fec_empty():
    res = score_fec_candidates(np.array([], dtype=np.uint8))
    assert res["candidate"] is None
    assert res["confidence"] == 0.0
    assert res["crc_pass"] is None
