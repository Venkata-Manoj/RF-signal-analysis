"""Acceptance tests for the extended blind-FEC hypothesis search.

These tests deliberately exercise the public codecs and the bounded,
CRC-verified search rather than duplicating the receive pipeline.  A candidate
is accepted only when a real CRC-16 check passes; the blind score remains a
separate, capped heuristic.
"""

from __future__ import annotations

import numpy as np
import pytest

from rf_analyzer.core import fec
from rf_analyzer.core.deinterleave import (
    deinterleave_block,
    deinterleave_diagonal,
    deinterleave_pseudo_random,
)
from rf_analyzer.core.fec import (
    CONV_CODE_BY_NAME,
    CONV_CODES,
    CONV_GENERATORS,
    LDPC_FAMILIES,
    ConvCode,
    bits_to_bytes,
    build_ldpc_matrix,
    bytes_to_bits,
    concatenated_decode_general,
    concatenated_encode_general,
    conv_encode_general,
    crc16_append,
    decode_hypotheses,
    hard_bits_to_llr,
    ldpc_decode_family,
    ldpc_encode_family,
    rs_decode,
    rs_encode,
    scan_crc16_strict,
    score_fec_candidates,
    viterbi_decode_general,
    viterbi_decode_soft,
)
from rf_analyzer.core.framing import apply_interleaver, encode_fec

# The error loads are deliberately well below the measured edge of the
# shipped (3,4)-regular LDPC code.  LDPC correction is probabilistic, so the
# matrix test records all five seeds rather than claiming that one lucky seed
# proves a capability.
MATRIX_ERRORS = {"scattered": 2, "burst": 1}
MATRIX_SEEDS = tuple(range(5))
MATRIX_INTERLEAVERS = ("none", "block", "diagonal", "pseudo-random")


def _flip_scattered(bits: np.ndarray, count: int, seed: int) -> np.ndarray:
    out = np.asarray(bits, dtype=np.uint8).copy()
    positions = np.random.default_rng(seed).choice(out.size, count, replace=False)
    out[positions] ^= 1
    return out


def _flip_burst(bits: np.ndarray, count: int, seed: int) -> np.ndarray:
    out = np.asarray(bits, dtype=np.uint8).copy()
    start = int(np.random.default_rng(seed).integers(0, out.size - count + 1))
    out[start : start + count] ^= 1
    return out


def _rs_codeword_bits(payload: bytes, nsym: int) -> np.ndarray:
    framed = crc16_append(payload)
    return bytes_to_bits(framed + rs_encode(framed, nsym))


def _find_ldpc_family(dv: int, dc: int) -> dict:
    return next(
        family
        for family in LDPC_FAMILIES
        if int(family["dv"]) == dv and int(family["dc"]) == dc
    )


@pytest.mark.parametrize(
    "k,rate",
    [
        (5, "1/2"),
        (5, "1/3"),
        (7, "1/2"),
        (7, "1/3"),
        (9, "1/2"),
        (9, "1/3"),
    ],
)
def test_convolutional_family_has_parameter_table_and_codec(k, rate):
    """Every required K/rate row is a real, round-tripping codec."""
    assert (k, rate) in CONV_GENERATORS
    generators = CONV_GENERATORS[(k, rate)]
    rng = np.random.default_rng(1000 + k * 10 + int(rate[-1]))
    information = rng.integers(0, 2, size=96, dtype=np.uint8)

    coded = conv_encode_general(information, k=k, generators=generators)
    hard = viterbi_decode_general(coded, k=k, generators=generators)
    assert np.array_equal(hard[: information.size], information)
    assert np.all(hard[-max(0, k - 1) :] == 0)

    # The public soft API accepts LLRs directly, including non-binary values.
    llr = hard_bits_to_llr(coded, n_out=len(generators), llr_mag=1.7)
    # Make a few bits ambiguous without changing their sign.
    llr[:: max(1, llr.size // 9)] *= 0.05
    soft = viterbi_decode_soft(llr, ConvCode(f"K={k} r={rate}", k, generators))
    assert np.array_equal(soft[: information.size], information)


def test_convolutional_tables_have_no_duplicate_code_rows():
    """The search table and the parameter table must not drift apart."""
    expected = {(code.k, f"1/{code.rate_den}"): code.gens for code in CONV_CODES}
    assert expected == dict(CONV_GENERATORS)
    assert set(CONV_CODE_BY_NAME) == {code.name for code in CONV_CODES}


@pytest.mark.parametrize("nsym", [8, 16, 32, 64])
@pytest.mark.parametrize("payload_length", [12, 40, 80])
def test_rs_shortened_lengths_are_searched_and_verified(nsym, payload_length):
    """The search covers all requested parity sizes and shortened word sizes."""
    payload = bytes((i * 17 + nsym) & 0xFF for i in range(payload_length))
    coded = _rs_codeword_bits(payload, nsym)
    # A small payload error makes the uncoded CRC hypothesis fail while the
    # RS hypothesis still has a wide correction margin.
    # Corrupt the protected message itself so the raw CRC framing cannot win
    # before the RS hypothesis gets an independent syndrome check.
    coded[0] ^= 1

    result = decode_hypotheses(
        coded,
        max_bits=200_000,
        time_budget_s=20.0,
        families=True,
    )
    assert result["validated"] is True, result
    assert result["best"] is not None
    assert result["best"]["params"]["nsym"] == nsym
    assert result["best"]["candidate"].startswith("RS(")
    assert bytes.fromhex(result["best"]["payload_hex"]) == payload


@pytest.mark.parametrize(("dv", "dc"), [(2, 6), (4, 4)])
def test_ldpc_parametric_families_are_valid_and_decodable(dv, dc):
    """The non-(3,4) families really are regular, systematic LDPC codes."""
    family = _find_ldpc_family(dv, dc)
    h = build_ldpc_matrix(
        int(family["k"]), int(family["m"]), int(family["dv"]), int(family["dc"])
    )
    k, m = int(family["k"]), int(family["m"])
    assert h.shape == (m, k + m)
    assert np.all(h[:, k:] == np.eye(m, dtype=np.uint8))
    assert np.all(h[:, :k].sum(axis=1) == dc)
    assert np.all(h[:, :k].sum(axis=0) == dv)
    assert len({tuple(np.flatnonzero(h[:, j])) for j in range(k)}) == k

    rng = np.random.default_rng(700 + dv * 10 + dc)
    information = rng.integers(0, 2, size=k, dtype=np.uint8)
    codeword = ldpc_encode_family(information, family)
    decoded, syndrome_ok = ldpc_decode_family(codeword, family)
    assert syndrome_ok is True
    assert np.array_equal(decoded[:k], information)

    # One error is well inside the capability of both newly required families.
    noisy = _flip_scattered(codeword, 1, seed=dv * 100 + dc)
    decoded, syndrome_ok = ldpc_decode_family(noisy, family)
    assert syndrome_ok is True
    assert np.array_equal(decoded[:k], information)


@pytest.mark.parametrize("nsym", [8, 32, 64])
@pytest.mark.parametrize("code_name", ["K=5 r=1/2", "K=7 r=1/2", "K=9 r=1/3"])
def test_concatenated_variants_round_trip_with_crc(nsym, code_name):
    """RS-outer/convolutional-inner is available beyond the historical pair."""
    code = CONV_CODE_BY_NAME[code_name]
    payload = b"concatenated family acceptance: " + code_name.encode()
    coded = fec.concatenated_encode_general(payload, nsym=nsym, code=code)
    noisy = _flip_scattered(coded, 1, seed=nsym + len(code_name))

    result = fec.concatenated_decode_general(noisy, nsym=nsym, code=code)
    assert result["crc_pass"] is True
    assert result["payload"] == payload


def test_blind_scores_never_claim_detection():
    for seed in range(5):
        rng = np.random.default_rng(seed)
        bits = rng.integers(0, 2, size=4000, dtype=np.uint8)
        scores = score_fec_candidates(bits)
        assert scores["confidence"] <= 0.5
        assert scores["crc_pass"] is None
        assert all(row["confidence"] <= 0.5 for row in scores["candidates"])
        assert all(row["crc_pass"] is None for row in scores["candidates"])


@pytest.mark.parametrize("seed", range(5))
def test_noise_is_never_reported_as_a_verified_decode(seed):
    rng = np.random.default_rng(1000 + seed)
    noise = rng.integers(0, 2, size=4000, dtype=np.uint8)
    result = decode_hypotheses(
        noise,
        max_bits=8192,
        time_budget_s=0.5,
        families=True,
    )
    assert result["validated"] is False
    assert result["best"] is None
    assert result["confidence"] == 0.0
    assert result["crc_trials"] >= 0


def test_crc_window_keeps_the_true_shortest_end_at_the_one_byte_boundary():
    message = b"shortest-first CRC framing with a nonzero low CRC byte"
    framed = crc16_append(message)
    # With a 32-byte window, the true end is exactly one byte below the
    # nominal lower edge.  The one-byte lookback must make it visible, and
    # sorting candidates must put it before the deterministic one-byte-longer
    # collision.
    tail_length = 32
    ok, payload = scan_crc16_strict(
        framed + b"\x00" * tail_length, min_payload=4, max_pad=32
    )
    assert ok is True
    assert payload == message


@pytest.mark.parametrize("interleaver", MATRIX_INTERLEAVERS)
@pytest.mark.parametrize("error_kind", tuple(MATRIX_ERRORS))
def test_interleaver_matrix_ldpc_tolerance_is_safe_over_five_seeds(
    interleaver, error_kind
):
    """Exercise the real interleaver inverses with safe scattered/burst loads."""
    payload = b"LDPC interleaver matrix acceptance payload: " + interleaver.encode()
    encoded = encode_fec(payload, scheme="ldpc")
    transmitted = apply_interleaver(encoded, interleaver)
    count = MATRIX_ERRORS[error_kind]

    if interleaver == "none":

        def deinterleave(bits):
            return bits

    elif interleaver == "block":
        deinterleave = deinterleave_block
    elif interleaver == "diagonal":
        deinterleave = deinterleave_diagonal
    else:
        deinterleave = deinterleave_pseudo_random

    recovered_all = []
    for seed in MATRIX_SEEDS:
        damaged = (
            _flip_scattered(transmitted, count, seed)
            if error_kind == "scattered"
            else _flip_burst(transmitted, count, seed)
        )
        recovered = np.asarray(deinterleave(damaged), dtype=np.uint8)[: encoded.size]
        assert recovered.size == encoded.size
        for offset in range(0, encoded.size, fec.LDPC_N):
            block = recovered[offset : offset + fec.LDPC_N]
            decoded = fec.ldpc_decode(block)
            assert fec.ldpc_syndrome_ok(decoded)
            assert np.array_equal(
                decoded[: fec.LDPC_K], encoded[offset : offset + fec.LDPC_K]
            )
        recovered_all.append(recovered)
    assert len(recovered_all) == 5


def test_rs_codec_returns_parity_only_and_corrects_shortened_word():
    payload = b"shortened Reed-Solomon word"
    framed = crc16_append(payload)
    parity = rs_encode(framed, 16)
    assert len(parity) == 16
    word = bytearray(framed + parity)
    word[2] ^= 0xA5
    corrected, count = rs_decode(bytes(word), 16)
    assert count == 1
    assert corrected == framed + parity


def test_search_exposes_its_bit_and_time_bounds():
    result = decode_hypotheses(
        np.zeros(200, dtype=np.uint8),
        max_bits=64,
        time_budget_s=0.0,
        families=True,
    )
    assert result["max_bits"] == 64
    assert result["time_budget_s"] == 0.0
    assert result["budget_exhausted"] is True
    # The budget marker is retained for diagnostics but no hypothesis ran.
    assert result["attempts"][-1]["candidate"] == "(budget)"
    assert all(row["candidate"] != "(budget)" for row in result["attempts"][:-1])
