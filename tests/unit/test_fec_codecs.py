"""Unit tests for the real FEC codecs (SIH26147 bullet iv).

These tests prove the decoders actually *work*, not merely that they exist:
every case encodes, injects channel errors, and checks that the decoder
recovers the original payload.
"""

from __future__ import annotations

import numpy as np
import pytest

from rf_analyzer.core.fec import (
    CONV_TAIL,
    GF_EXP,
    LDPC_K,
    LDPC_N,
    ReedSolomonError,
    bits_to_bytes,
    bytes_to_bits,
    concatenated_decode,
    concatenated_encode,
    conv_encode,
    crc16_append,
    crc16_ccitt,
    crc16_check,
    crc32_ieee,
    decode_hypotheses,
    gf_div,
    gf_inverse,
    gf_mul,
    gf_pow,
    ldpc_decode,
    ldpc_encode,
    ldpc_syndrome_ok,
    rs_check,
    rs_decode,
    rs_encode,
    scan_crc16_strict,
    viterbi_decode,
)


def _bsc(bits: np.ndarray, n_errors: int, seed: int = 0) -> np.ndarray:
    """Flip exactly ``n_errors`` distinct bits (binary symmetric channel)."""
    out = np.asarray(bits, dtype=np.uint8).copy()
    rng = np.random.default_rng(seed)
    if n_errors <= 0:
        return out
    n_errors = min(n_errors, out.size)
    positions = rng.choice(out.size, size=n_errors, replace=False)
    out[positions] ^= 1
    return out


# --------------------------------------------------------------------------- #
# Bit packing + CRC
# --------------------------------------------------------------------------- #


def test_bits_bytes_roundtrip():
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, size=64, dtype=np.uint8)
    assert np.array_equal(bytes_to_bits(bits_to_bytes(bits)), bits)


def test_crc16_ccitt_known_vector():
    # Standard CRC-16/CCITT-FALSE check value for "123456789".
    assert crc16_ccitt(b"123456789") == 0x29B1


def test_crc32_ieee_known_vector():
    # Standard CRC-32/IEEE check value for "123456789".
    assert crc32_ieee(b"123456789") == 0xCBF43926


def test_crc16_append_check_roundtrip_and_detection():
    payload = b"RF signal analysis"
    framed = crc16_append(payload)
    assert len(framed) == len(payload) + 2

    ok, recovered = crc16_check(framed)
    assert ok and recovered == payload

    # Any single-bit corruption must be detected.
    corrupted = bytearray(framed)
    corrupted[3] ^= 0x01
    bad_ok, _ = crc16_check(bytes(corrupted))
    assert not bad_ok


def test_crc16_check_rejects_short_input():
    ok, payload = crc16_check(b"\x01")
    assert ok is False and payload == b"\x01"


def test_crc_framing_prefers_the_shorter_payload_over_a_one_byte_collision():
    """The documented deterministic false positive must never win.

    For CRC-16/CCITT-FALSE the identity ``crc16(M || C_hi) == C_lo << 8`` holds,
    so whenever the byte following a real frame is ``0x00`` -- which is exactly
    what block-interleaver zero fill supplies -- the framing one byte *longer*
    validates too. Scanning shortest-first is what rejects it, and that only
    works while the true end stays inside the candidate window.

    Regression: with the frame followed by 33 zero bytes the window's lower edge
    landed precisely on the collision, so the truth was invisible and the scan
    returned a CRC-verified payload carrying one extra trailing byte (the real
    CRC's high byte). A wrong payload reported as verified is the worst possible
    failure mode here, hence the one-byte lookback in ``_candidate_ends``.
    """
    message = b"SIH26147 framing regression payload for the CRC lookback."
    framed = crc16_append(message)
    assert crc16_check(framed) == (True, message)

    # The identity that creates the collision, asserted rather than assumed.
    crc = crc16_ccitt(message)
    c_hi, c_lo = crc >> 8, crc & 0xFF
    assert framed == message + bytes([c_hi, c_lo])
    assert crc16_ccitt(message + bytes([c_hi])) == (c_lo << 8)

    # 33 zero bytes put the window's lower edge one byte past the true end.
    buf = framed + b"\x00" * 33
    assert len(buf) - 32 == len(framed) + 1

    ok, payload = scan_crc16_strict(buf)
    assert ok is True
    assert payload == message, f"extra trailing bytes: {payload[len(message):]!r}"


def test_crc_framing_still_scans_normally_inside_the_window():
    """The lookback must not disturb the ordinary case."""
    message = b"short"
    framed = crc16_append(message)
    ok, payload = scan_crc16_strict(framed + b"\x00" * 4)
    assert (ok, payload) == (True, message)


# --------------------------------------------------------------------------- #
# Convolutional code + Viterbi
# --------------------------------------------------------------------------- #


def test_conv_encode_is_rate_one_half_with_tail():
    rng = np.random.default_rng(2)
    bits = rng.integers(0, 2, size=100, dtype=np.uint8)
    coded = conv_encode(bits)
    assert coded.size == (100 + CONV_TAIL) * 2


def test_viterbi_roundtrip_clean():
    rng = np.random.default_rng(3)
    bits = rng.integers(0, 2, size=300, dtype=np.uint8)
    coded = conv_encode(bits)

    decoded = viterbi_decode(coded)
    info = decoded[: bits.size]
    assert np.array_equal(info, bits), "clean channel must decode bit-exactly"


def test_viterbi_corrects_channel_errors():
    rng = np.random.default_rng(4)
    bits = rng.integers(0, 2, size=600, dtype=np.uint8)
    coded = conv_encode(bits)
    noisy = _bsc(coded, n_errors=18, seed=4)  # ~1.5% raw BER

    raw_ber = float(np.mean(noisy[: bits.size * 2] != coded[: bits.size * 2]))
    decoded = viterbi_decode(noisy)[: bits.size]
    coded_ber = float(np.mean(decoded != bits))

    assert raw_ber > 0.0, "test must actually inject errors"
    assert coded_ber < raw_ber / 5, (
        f"Viterbi should improve BER markedly (raw={raw_ber:.4f}, "
        f"decoded={coded_ber:.4f})"
    )
    assert coded_ber < 0.02


def test_viterbi_rejects_odd_length():
    with pytest.raises(ValueError):
        viterbi_decode(np.array([1, 0, 1], dtype=np.uint8))


# --------------------------------------------------------------------------- #
# GF(256) + Reed-Solomon
# --------------------------------------------------------------------------- #


def test_gf_field_axioms():
    # a * a^-1 == 1 for every non-zero element; exp/log are consistent.
    for a in range(1, 256):
        assert gf_mul(a, gf_inverse(a)) == 1
    assert gf_mul(0, 123) == 0
    assert gf_div(0, 7) == 0
    assert gf_pow(2, 8) == gf_mul(2, GF_EXP[7])


def test_rs_encode_length_and_validity():
    payload = bytes(range(200))
    parity = rs_encode(payload, 32)
    assert len(parity) == 32
    assert rs_check(payload + parity, 32)


def test_rs_roundtrip_clean():
    payload = bytes(range(100))
    codeword = payload + rs_encode(payload, 16)
    corrected, n_err = rs_decode(codeword, 16)
    assert n_err == 0
    assert corrected[:100] == payload


@pytest.mark.parametrize("nsym,n_errors", [(32, 16), (16, 8), (8, 4)])
def test_rs_corrects_up_to_half_nsym_symbol_errors(nsym, n_errors):
    """RS(n,k) corrects exactly t = nsym/2 symbol errors."""
    rng = np.random.default_rng(nsym)
    payload = rng.integers(0, 256, size=80, dtype=np.uint8).tobytes()
    codeword = bytearray(payload + rs_encode(payload, nsym))

    positions = rng.choice(len(codeword), size=n_errors, replace=False)
    for pos in positions:
        codeword[pos] ^= int(rng.integers(1, 256))

    corrected, n_err = rs_decode(bytes(codeword), nsym)
    assert n_err == n_errors
    assert corrected[: len(payload)] == payload


def test_rs_raises_beyond_correction_radius():
    rng = np.random.default_rng(99)
    payload = rng.integers(0, 256, size=60, dtype=np.uint8).tobytes()
    codeword = bytearray(payload + rs_encode(payload, 8))  # t = 4

    # 8 symbol errors > t -> must refuse rather than silently return junk.
    positions = rng.choice(len(codeword), size=8, replace=False)
    for pos in positions:
        codeword[pos] ^= int(rng.integers(1, 256))

    with pytest.raises(ReedSolomonError):
        rs_decode(bytes(codeword), 8)


def test_rs_full_length_255_223():
    """Full-length RS(255,223): 223 data symbols, 32 parity, t=16."""
    rng = np.random.default_rng(255)
    payload = rng.integers(0, 256, size=223, dtype=np.uint8).tobytes()
    codeword = bytearray(payload + rs_encode(payload, 32))
    assert len(codeword) == 255

    positions = rng.choice(255, size=16, replace=False)
    for pos in positions:
        codeword[pos] ^= int(rng.integers(1, 256))

    corrected, n_err = rs_decode(bytes(codeword), 32)
    assert n_err == 16
    assert corrected == payload + rs_encode(payload, 32)


# --------------------------------------------------------------------------- #
# LDPC
# --------------------------------------------------------------------------- #


def test_ldpc_encode_shape_and_validity():
    rng = np.random.default_rng(7)
    info = rng.integers(0, 2, size=LDPC_K, dtype=np.uint8)
    codeword = ldpc_encode(info)
    assert codeword.size == LDPC_N
    assert np.array_equal(codeword[:LDPC_K], info)
    assert ldpc_syndrome_ok(codeword)


def test_ldpc_rejects_wrong_info_length():
    with pytest.raises(ValueError):
        ldpc_encode(np.zeros(LDPC_K - 1, dtype=np.uint8))


def test_ldpc_decode_clean():
    rng = np.random.default_rng(8)
    info = rng.integers(0, 2, size=LDPC_K, dtype=np.uint8)
    codeword = ldpc_encode(info)

    decoded = ldpc_decode(codeword)
    assert ldpc_syndrome_ok(decoded)
    assert np.array_equal(decoded[:LDPC_K], info)


def test_ldpc_corrects_channel_errors():
    rng = np.random.default_rng(9)
    info = rng.integers(0, 2, size=LDPC_K, dtype=np.uint8)
    codeword = ldpc_encode(info)
    noisy = _bsc(codeword, n_errors=3, seed=9)

    decoded = ldpc_decode(noisy, p_err=0.05)
    assert ldpc_syndrome_ok(decoded), "min-sum BP should converge to a codeword"
    assert np.array_equal(decoded[:LDPC_K], info)


# --------------------------------------------------------------------------- #
# Concatenated code
# --------------------------------------------------------------------------- #


def test_concatenated_roundtrip_clean():
    payload = b"SIH26147 payload: header + telemetry block 001"
    coded = concatenated_encode(payload, nsym=32)
    assert coded.dtype == np.uint8
    assert set(np.unique(coded)).issubset({0, 1})

    result = concatenated_decode(coded, nsym=32)
    assert result["crc_pass"] is True
    assert result["payload"] == payload


def test_concatenated_survives_channel_errors():
    payload = bytes(range(120))
    coded = concatenated_encode(payload, nsym=32)
    noisy = _bsc(coded, n_errors=int(coded.size * 0.01), seed=11)

    result = concatenated_decode(noisy, nsym=32)
    assert result["crc_pass"] is True, "concatenated code should recover ~1% errors"
    assert result["payload"] == payload


# --------------------------------------------------------------------------- #
# Hypothesis verification
# --------------------------------------------------------------------------- #


def test_decode_hypotheses_validates_real_coded_stream():
    payload = b"payload for hypothesis verification"
    coded = concatenated_encode(payload, nsym=32)
    noisy = _bsc(coded, n_errors=12, seed=13)

    result = decode_hypotheses(noisy, max_bits=100_000)
    assert result["validated"] is True
    best = result["best"]
    assert best is not None
    assert best["crc_pass"] is True
    assert best["confidence"] > 0.9
    assert bytes.fromhex(best["payload_hex"]) == payload
    assert result["n_attempts"] >= 4


def test_decode_hypotheses_respects_start_offset():
    payload = b"offset test payload"
    coded = concatenated_encode(payload, nsym=16)
    # Prepend a 32-bit sync word that the caller will skip.
    stream = np.concatenate([np.ones(32, dtype=np.uint8), coded])

    miss = decode_hypotheses(stream, start_offset=0, max_bits=100_000)
    hit = decode_hypotheses(stream, start_offset=32, max_bits=100_000)

    assert hit["validated"] is True
    assert bytes.fromhex(hit["best"]["payload_hex"]) == payload
    # Without skipping the header the alignment is wrong, so it must not validate.
    assert miss["validated"] is False


def test_decode_hypotheses_rejects_random_bits():
    rng = np.random.default_rng(21)
    noise = rng.integers(0, 2, size=4000, dtype=np.uint8)
    result = decode_hypotheses(noise, max_bits=100_000)
    assert result["validated"] is False
    assert result["best"] is None
    assert result["confidence"] == 0.0


def test_decode_hypotheses_short_stream_is_safe():
    result = decode_hypotheses(np.array([1, 0, 1], dtype=np.uint8))
    assert result["validated"] is False
    assert result["attempts"] == []
