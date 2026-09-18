"""Unit tests for de-interleaving (SIH26147 bullet iii).

The old stub tests only asserted "returns the same length" / "passes through".
These tests assert the stronger, meaningful property: every scheme is an exact
inverse pair, and it disperses bursts rather than copying them.
"""

from __future__ import annotations

import numpy as np
import pytest

from rf_analyzer.core.deinterleave import (
    deinterleave_block,
    deinterleave_convolutional,
    deinterleave_diagonal,
    deinterleave_pseudo_random,
    interleave_block,
    interleave_convolutional,
    interleave_diagonal,
    interleave_pseudo_random,
    score_deinterleave_candidates,
    search_interleaver,
)
from rf_analyzer.core.fec import concatenated_encode, crc16_append, rs_encode

ROUNDTRIP_CASES = [
    ("block", interleave_block, deinterleave_block),
    ("convolutional", interleave_convolutional, deinterleave_convolutional),
    ("diagonal", interleave_diagonal, deinterleave_diagonal),
    ("pseudo-random", interleave_pseudo_random, deinterleave_pseudo_random),
]


@pytest.mark.parametrize(("name", "inter", "deinter"), ROUNDTRIP_CASES)
def test_each_scheme_is_an_exact_inverse(name, inter, deinter):
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=96, dtype=np.uint8)

    restored = deinter(inter(bits))

    assert np.array_equal(restored[: bits.size], bits), f"{name} roundtrip failed"


@pytest.mark.parametrize(("name", "inter", "deinter"), ROUNDTRIP_CASES)
def test_roundtrip_survives_non_multiple_lengths(name, inter, deinter):
    """Padding must not corrupt the payload for awkward input sizes."""
    rng = np.random.default_rng(5)
    bits = rng.integers(0, 2, size=101, dtype=np.uint8)  # not a nice multiple
    restored = deinter(inter(bits))
    assert np.array_equal(restored[: bits.size], bits)


def test_block_interleaver_disperses_a_burst():
    # A contiguous burst must be spread out by block interleaving.
    bits = np.zeros(96, dtype=np.uint8)
    bits[0:12] = 1
    interleaved = interleave_block(bits, rows=8, cols=12)
    # Column-wise read spreads the first row across every 8th position.
    ones_positions = np.flatnonzero(interleaved)
    gaps = np.diff(ones_positions)
    assert gaps.min() > 1, "burst should no longer be contiguous"


def test_diagonal_interleaver_is_not_the_block_interleaver():
    """Diagonal must be a genuinely different mapping from Block."""
    bits = np.arange(96, dtype=np.uint8) % 2
    assert not np.array_equal(
        interleave_diagonal(bits, 8, 12), interleave_block(bits, 8, 12)
    )


def test_convolutional_interleaver_is_not_a_pass_through():
    """Regression guard: the old stub returned its input unchanged.

    A burst is used rather than an alternating pattern, because an
    even-depth convolutional permutation happens to preserve the parity of
    the bit index and would leave ``1010...`` looking unchanged.
    """
    bits = np.zeros(96, dtype=np.uint8)
    bits[0:16] = 1
    assert not np.array_equal(interleave_convolutional(bits, depth=8), bits)
    assert not np.array_equal(deinterleave_convolutional(bits, depth=8), bits)


def test_convolutional_permutation_is_a_bijection():
    """The delay-line map must be a true permutation, not a lossy copy."""
    bits = np.arange(96, dtype=np.uint8)
    interleaved = interleave_convolutional(bits, depth=8, spacing=1)
    assert sorted(interleaved.tolist()) == sorted(bits.tolist())


def test_pseudo_random_different_seeds_differ():
    bits = np.arange(64, dtype=np.uint8) % 2
    assert not np.array_equal(
        interleave_pseudo_random(bits, seed=1), interleave_pseudo_random(bits, seed=2)
    )


def test_convolutional_rejects_bad_parameters():
    bits = np.zeros(16, dtype=np.uint8)
    with pytest.raises(ValueError):
        interleave_convolutional(bits, depth=1)
    with pytest.raises(ValueError):
        interleave_convolutional(bits, depth=4, spacing=0)


def test_deinterleave_confidence_cap():
    res = score_deinterleave_candidates(np.array([0, 1] * 50, dtype=np.uint8))
    assert res["confidence"] <= 0.5
    assert "candidate" in res
    assert "depth" in res


# --------------------------------------------------------------------------- #
# Verified hypothesis search (the part that proves de-interleaving works)
# --------------------------------------------------------------------------- #


def _build_interleaved_coded_stream(payload: bytes, scheme: str, prepend: int = 32):
    """CRC-16 -> RS(255,223) -> convolutional, optionally interleaved."""
    framed = crc16_append(payload)
    word = framed + rs_encode(framed, 32)
    from rf_analyzer.core.fec import bytes_to_bits, conv_encode

    coded = conv_encode(bytes_to_bits(word))
    if scheme == "block":
        coded = interleave_block(coded, rows=8, cols=12)
    elif scheme == "convolutional":
        coded = interleave_convolutional(coded, depth=8)
    elif scheme == "diagonal":
        coded = interleave_diagonal(coded, rows=8, cols=12)
    elif scheme == "pseudo-random":
        coded = interleave_pseudo_random(coded, seed=42)
    header = np.ones(prepend, dtype=np.uint8)
    return np.concatenate([header, coded]), prepend


@pytest.mark.parametrize(
    "scheme", ["none", "block", "convolutional", "diagonal", "pseudo-random"]
)
def test_search_interleaver_recovers_payload_for_each_scheme(scheme):
    """A CRC-valid payload must be recovered for every interleaver scheme."""
    payload = b"de-interleaving verification payload"
    stream, offset = _build_interleaved_coded_stream(payload, scheme)

    # Generous budget: this test asserts correctness, not the timing guard.
    result = search_interleaver(
        stream, start_offset=offset, max_bits=100_000, time_budget_s=120.0
    )

    assert result["validated"] is True, f"{scheme}: no CRC-valid decode"
    assert result["interleaver"] == scheme
    assert bytes.fromhex(result["best"]["payload_hex"]) == payload


def test_search_interleaver_reports_no_winner_on_random_bits():
    rng = np.random.default_rng(3)
    noise = rng.integers(0, 2, size=6000, dtype=np.uint8)
    result = search_interleaver(noise, max_bits=100_000, time_budget_s=120.0)
    assert result["validated"] is False
    assert result["interleaver"] is None


def test_concatenated_encode_output_is_still_decodable_after_interleaving():
    """Sanity: interleaving a real coded stream preserves its information."""
    from rf_analyzer.core.fec import concatenated_decode

    payload = b"roundtrip through the diagonal interleaver"
    coded = concatenated_encode(payload, nsym=32)
    interleaved = interleave_diagonal(coded, rows=8, cols=12)
    restored = deinterleave_diagonal(interleaved, rows=8, cols=12)

    result = concatenated_decode(restored[: coded.size], nsym=32)
    assert result["crc_pass"] is True
    assert result["payload"] == payload
