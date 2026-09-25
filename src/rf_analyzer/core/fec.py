"""Forward-error-correction codecs + hypothesis validation.

SIH26147 bullet (iv) asks for "FEC (short-constrained convolution codes with
Viterbi decoding, RS block codes, Concatenated codes, LDPC)". Full spec:
``info.md`` §12.5, with the honesty rules in §30.

Two clearly separated responsibilities live in this module:

1. **Real codecs** — from-scratch, pure-numpy implementations:
   * :func:`crc16_ccitt` / :func:`crc32_ieee` integrity checks;
   * convolutional encoders (:func:`conv_encode_general`) with hard
     (:func:`viterbi_decode_general`) and soft-decision
     (:func:`viterbi_decode_soft`) Viterbi decoders over the K=5..9,
     r=1/2 and r=1/3 family in :data:`CONV_GENERATORS`. The historical
     K=7, r=1/2 pair (:func:`conv_encode` / :func:`viterbi_decode`,
     ``G1=0o171, G2=0o133``) is preserved exactly and stays the default;
   * GF(256) arithmetic plus a systematic Reed-Solomon codec
     (:func:`rs_encode` / :func:`rs_decode`) supporting RS(255,223),
     RS(255,239), RS(255,247) and shortened/truncated variants;
   * a systematic LDPC family (:data:`LDPC_FAMILY`) with a normalized
     min-sum belief-propagation decoder (:func:`ldpc_encode` /
     :func:`ldpc_decode`), built by the parametric
     :func:`build_ldpc_matrix` (the original (3,4)-regular code is the
     default member);
   * concatenated RS-outer / convolutional-inner schemes
     (:func:`concatenated_encode_general` /
     :func:`concatenated_decode_general`) over several (nsym, K) variants.

2. **Blind identification** — :func:`score_fec_candidates`, which ranks
   candidates by structural metrics (parity drift, RS syndrome zeros, LDPC
   syndrome weight) but is deliberately capped at 0.5 confidence and never
   claims detection. Fully blind FEC identification is research-grade
   (``info.md`` §30) and stays out of scope.

:func:`decode_hypotheses` bridges the two: it *tries* the real decoders on a
hypothesis basis and reports high confidence only when an independent
integrity check (CRC) actually passes. We never guess — we verify.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from rf_analyzer.config import DECODE_MAX_BITS, DECODE_TIME_BUDGET_S

# --------------------------------------------------------------------------- #
# Bit/byte helpers
# --------------------------------------------------------------------------- #


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Pack a 0/1 bit array MSB-first into bytes (trailing bits zero-padded)."""
    arr = np.asarray(bits, dtype=np.uint8).ravel()
    if arr.size == 0:
        return b""
    return np.packbits(arr).tobytes()


def bytes_to_bits(data: bytes) -> np.ndarray:
    """Unpack bytes into a 0/1 uint8 bit array (MSB-first, length = 8 * len)."""
    if not data:
        return np.array([], dtype=np.uint8)
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


# --------------------------------------------------------------------------- #
# CRC integrity checks
# --------------------------------------------------------------------------- #


def _crc16_update(crc: int, byte: int) -> int:
    """Advance a CRC-16/CCITT-FALSE register by one byte."""
    crc ^= (byte & 0xFF) << 8
    for _ in range(8):
        crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def crc16_ccitt(data: bytes, init: int = 0xFFFF) -> int:
    """CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, no reflection/xorout)."""
    crc = init & 0xFFFF
    for byte in data:
        crc = _crc16_update(crc, byte)
    return crc


def crc32_ieee(data: bytes) -> int:
    """CRC-32/IEEE (poly 0xEDB88320, init/xorout 0xFFFFFFFF), table-free."""
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte & 0xFF
        for _ in range(8):
            crc = (crc >> 1) ^ 0xEDB88320 if crc & 1 else crc >> 1
    return crc ^ 0xFFFFFFFF


def crc16_append(data: bytes) -> bytes:
    """Return ``data`` with a big-endian CRC-16/CCITT appended."""
    return bytes(data) + crc16_ccitt(data).to_bytes(2, "big")


def crc16_check(framed: bytes) -> tuple[bool, bytes]:
    """Split ``framed`` into (payload, trailing CRC-16) and verify.

    Returns ``(ok, payload)``. ``ok`` is False when fewer than 2 bytes are
    present or the trailing CRC does not match.
    """
    framed = bytes(framed)
    if len(framed) < 3:
        return False, framed
    payload, crc_bytes = framed[:-2], framed[-2:]
    expected = int.from_bytes(crc_bytes, "big")
    return crc16_ccitt(payload) == expected, payload


def _crc16_prefix(buf: bytes) -> list[int]:
    """``out[i] == crc16_ccitt(buf[:i])`` — one O(n) pass shared by the scanners."""
    running = [0xFFFF] * (len(buf) + 1)
    crc = 0xFFFF
    for i, byte in enumerate(buf):
        crc = _crc16_update(crc, byte)
        running[i + 1] = crc
    return running


def _candidate_ends(buf: bytes, min_payload: int, window: int | None) -> list[int]:
    """Candidate CRC framing end positions, shortest payload first.

    Why shortest-first: the only *deterministic* false positive in this framing
    scheme is a framing one byte too long. For CRC-16/CCITT-FALSE the identity
    ``crc16(M || C_hi) == C_lo << 8`` always holds (``C_lo`` is one byte, so the
    eight shift steps never reach bit 15 and never reduce). Whenever the byte
    following a real frame is ``0x00``, the longer framing ``[M][C_hi]`` with
    ``[C_lo][0x00]`` as its CRC therefore validates too — and it is *longer*
    than the truth, so a longest-first scan picks it every time. Scanning from
    the shortest payload up removes that systematic error and replaces it with
    the bounded 1/65536-per-position chance of a random coincidence.

    ``window`` restricts the search to positions near the end of the buffer,
    which is where a frame ends once the zero fill a block interleaver appends
    has been accounted for. ``None`` scans the whole buffer.

    The window is widened by one byte at the bottom so the shortest-first rule
    above can always do its job. The deterministic false positive is exactly one
    byte *longer* than the truth, so the false positive is reachable only when
    the true end is at ``lo - 1``: leave that position out and the false positive
    becomes the first (and only) match, which is how a capture whose frame is
    followed by a non-zero tail once returned a CRC-verified payload with one
    extra trailing byte -- the real CRC's high byte. One byte of lookback is
    exactly enough (``true_end >= lo - 1`` is precisely the condition under which
    the collision can be seen at all), and it removes the whole class. The cost
    is a single extra candidate position (~1/65536 of false-positive risk), which
    is nothing next to reporting a wrong payload as verified.
    """
    n = len(buf)
    lo = min_payload + 2
    if window is not None:
        lo = max(lo, n - window - 1)
    ends = list(range(lo, n + 1))

    # Safety net: the padding-stripped end is the structurally cleanest anchor,
    # so keep it as a candidate even when it falls outside a narrow window.
    zero_end = n
    while zero_end > 0 and buf[zero_end - 1] == 0:
        zero_end -= 1
    if zero_end >= min_payload + 2 and zero_end not in ends:
        ends.append(zero_end)
    # ``zero_end`` is a safety-net candidate and can sit just below the
    # window's normal lower edge.  Keep the ordering global rather than merely
    # ordering the contiguous range: appending the safety candidate last would
    # let the deterministic one-byte-longer CRC collision win over the true
    # shortest frame at the lookback boundary.
    return sorted(set(ends))


def scan_crc16(buf: bytes, min_payload: int = 1) -> tuple[bool, bytes]:
    """Find a valid CRC-16 framing *anywhere* in ``buf``.

    Needed when the payload length is unknown (e.g. a block-coded stream that
    was zero-padded to a fixed block size): every candidate split point is
    tested and the shortest payload with a matching trailing CRC-16 wins (see
    :func:`_candidate_ends` for why shortest is the right tie-break).

    Runs in a single O(n) CRC pass over every prefix, so the whole hypothesis
    search stays fast.

    A 16-bit check has a 1/65536 false-positive rate per position, so callers
    should combine this with an independent structural check (e.g. LDPC parity
    or a valid RS codeword) before trusting it.
    """
    buf = bytes(buf)
    n = len(buf)
    if n < min_payload + 2:
        return False, b""

    running = _crc16_prefix(buf)
    for end in _candidate_ends(buf, min_payload, None):
        expected = (buf[end - 2] << 8) | buf[end - 1]
        if running[end - 2] == expected:
            return True, buf[: end - 2]
    return False, b""


def scan_crc16_strict(
    buf: bytes, min_payload: int = 4, max_pad: int = 32
) -> tuple[bool, bytes]:
    """CRC-16 framing anchored near the *end* of the buffer.

    A free scan (:func:`scan_crc16`) has a ~n/65536 chance of matching by luck,
    which is far too high to report as a decoded payload on its own. This
    variant only considers positions within ``max_pad`` bytes of the buffer end,
    which is where the frame really ends once the zero fill a block interleaver
    appends has been accounted for. Use it whenever no independent structural
    check (valid RS codeword, LDPC syndrome) backs the CRC up.

    Keeping the search inside a short window holds the false-positive rate near
    ``(max_pad + 1) / 65536`` -- the ``+1`` is the one-byte lookback
    :func:`_candidate_ends` needs to keep the shortest-first rule effective --
    while still tolerating padding whose last byte happens to be non-zero.
    """
    buf = bytes(buf)
    n = len(buf)
    if n < min_payload + 2:
        return False, b""

    running = _crc16_prefix(buf)
    for end in _candidate_ends(buf, min_payload, max_pad):
        expected = (buf[end - 2] << 8) | buf[end - 1]
        if running[end - 2] == expected:
            return True, buf[: end - 2]
    return False, b""


# --------------------------------------------------------------------------- #
# Convolutional codes: K=5..9, r=1/2 and r=1/3, hard + soft-decision Viterbi
# --------------------------------------------------------------------------- #

CONV_K = 7
CONV_G1 = 0o171  # 1111001 -> 121
CONV_G2 = 0o133  # 1011011 -> 91
CONV_STATES = 1 << (CONV_K - 1)  # 64 trellis states
CONV_TAIL = CONV_K - 1  # 6 zero tail bits flush the encoder

#: Generator tables (octal) for the blind-search code family. ``(7, "1/2")``
#: is the CCSDS / 802.11 pair the original hard-decision path uses; the rest
#: are the standard published polynomials for each constraint length, so a
#: transmitter built from any of them decodes here. There is exactly one
#: definition of each code: the encoder and every decoder read this table.
CONV_GENERATORS: dict[tuple[int, str], tuple[int, ...]] = {
    (5, "1/2"): (0o23, 0o35),
    (6, "1/2"): (0o53, 0o75),
    (7, "1/2"): (0o171, 0o133),
    (8, "1/2"): (0o247, 0o371),
    (9, "1/2"): (0o561, 0o753),
    (5, "1/3"): (0o25, 0o33, 0o37),
    (6, "1/3"): (0o51, 0o53, 0o75),
    (7, "1/3"): (0o171, 0o133, 0o165),
    (8, "1/3"): (0o247, 0o371, 0o225),
    (9, "1/3"): (0o557, 0o663, 0o711),
}

#: Trellis tables per ``(K, generators)``, built once and shared by every
#: decode. A K=9 table is 256 states; rebuilding it per block would dominate
#: the search budget.
_CONV_CODE_CACHE: dict[tuple[int, tuple[int, ...]], dict] = {}


def _get_conv_code(k: int, generators: tuple[int, ...]) -> dict:
    """Trellis tables for constraint length ``k`` and ``generators``.

    Returns ``outputs`` (the ``2**k`` output symbols MSB-first), the
    predecessor tables (``in_bit``/``pred_a``/``pred_b``/``out_a``/``out_b``)
    and the soft-decision sign tables (``sign_a``/``sign_b``, +1 for an
    expected 0 and -1 for an expected 1).
    """
    key = (int(k), tuple(int(g) for g in generators))
    hit = _CONV_CODE_CACHE.get(key)
    if hit is not None:
        return hit
    k, gens = key[0], key[1]
    n_out = len(gens)
    if k < 3 or k > 12:
        raise ValueError(f"constraint length K={k} out of supported range 3..12")
    if n_out not in (2, 3):
        raise ValueError(f"only r=1/2 and r=1/3 are supported, got {n_out} outputs")
    mask = (1 << k) - 1
    for g in gens:
        if g <= 0 or g > mask or (g & 1) == 0:
            raise ValueError(f"generator {g:#o} is not a valid K={k} polynomial")

    def _parity(x: int) -> int:
        return bin(x).count("1") & 1

    outputs = np.empty(1 << k, dtype=np.uint8)
    for reg in range(1 << k):
        sym = 0
        for g in gens:
            sym = (sym << 1) | _parity(reg & g)
        outputs[reg] = sym

    n_states = 1 << (k - 1)
    half = 1 << (k - 2)
    states = np.arange(n_states, dtype=np.int32)
    # Next state n = (input << (K-2)) | (prev >> 1): the top bit is the input
    # bit and each predecessor pair shares the low K-2 bits shifted left.
    in_bit = states >> (k - 2)
    low = states & (half - 1)
    pred_a = (low << 1).astype(np.int32)
    pred_b = (pred_a | 1).astype(np.int32)
    out_a = np.empty(n_states, dtype=np.int16)
    out_b = np.empty(n_states, dtype=np.int16)
    for n in range(n_states):
        b = int(in_bit[n])
        out_a[n] = outputs[(b << (k - 1)) | int(pred_a[n])]
        out_b[n] = outputs[(b << (k - 1)) | int(pred_b[n])]

    # Soft-decision signs: expected bit 0 contributes +LLR, 1 contributes -LLR.
    sign_a = np.empty((n_states, n_out), dtype=np.float64)
    sign_b = np.empty((n_states, n_out), dtype=np.float64)
    for table, signs in ((out_a, sign_a), (out_b, sign_b)):
        for n in range(n_states):
            sym = int(table[n])
            for j in range(n_out):
                bit = (sym >> (n_out - 1 - j)) & 1
                signs[n, j] = 1.0 if bit == 0 else -1.0

    # Hamming distances between n_out-bit symbols, via a popcount table.
    popcount = np.array([bin(i).count("1") for i in range(1 << n_out)], dtype=np.int16)

    code = {
        "k": k,
        "n_out": n_out,
        "n_states": n_states,
        "outputs": outputs,
        "in_bit": in_bit,
        "pred_a": pred_a,
        "pred_b": pred_b,
        "out_a": out_a,
        "out_b": out_b,
        "sign_a": sign_a,
        "sign_b": sign_b,
        "popcount": popcount,
    }
    _CONV_CODE_CACHE[key] = code
    return code


def conv_encode_general(
    bits: np.ndarray,
    k: int = CONV_K,
    generators: tuple[int, ...] | None = None,
    terminate: bool = True,
) -> np.ndarray:
    """Convolutional encoder for constraint length ``k`` and ``generators``.

    Emits ``len(generators)`` bits per input bit (r=1/2 or r=1/3). With
    ``terminate=True`` (default) ``k - 1`` zero bits are appended so the
    trellis ends in state 0, which is what the Viterbi decoders expect.
    """
    gens = (
        (CONV_G1, CONV_G2) if generators is None else tuple(int(g) for g in generators)
    )
    code = _get_conv_code(int(k), gens)
    k = int(code["k"])
    n_out = int(code["n_out"])
    outputs = code["outputs"]
    arr = np.asarray(bits, dtype=np.uint8).ravel()
    if terminate:
        arr = np.concatenate([arr, np.zeros(k - 1, dtype=np.uint8)])
    if arr.size == 0:
        return np.array([], dtype=np.uint8)

    # state = last K-1 input bits; reg = (new_bit << (K-1)) | state
    state = 0
    out = np.empty(arr.size * n_out, dtype=np.uint8)
    for i, bit in enumerate(arr):
        reg = (int(bit) << (k - 1)) | state
        sym = int(outputs[reg])
        for j in range(n_out):
            out[i * n_out + j] = (sym >> (n_out - 1 - j)) & 1
        state = reg >> 1
    return out


def conv_encode(bits: np.ndarray, terminate: bool = True) -> np.ndarray:
    """Rate-1/2, K=7 convolutional encoder (hard bits in, 2 bits per input).

    With ``terminate=True`` (default) ``CONV_TAIL`` zero bits are appended so
    the trellis ends in state 0, which is what the Viterbi decoder expects.
    """
    return conv_encode_general(
        bits, k=CONV_K, generators=(CONV_G1, CONV_G2), terminate=terminate
    )


def viterbi_decode_general(
    rx_bits: np.ndarray,
    k: int = CONV_K,
    generators: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Hard-decision Viterbi decoder for constraint length ``k``.

    Fully vectorized across the trellis states. Returns the decoded
    information bits (including any tail bits the encoder appended).

    Raises:
        ValueError: when the received stream is not a multiple of the code
            rate's output width.
    """
    gens = (
        (CONV_G1, CONV_G2) if generators is None else tuple(int(g) for g in generators)
    )
    code = _get_conv_code(int(k), gens)
    k = int(code["k"])
    n_out = int(code["n_out"])
    n_states = int(code["n_states"])
    rx = np.asarray(rx_bits, dtype=np.uint8).ravel()
    if rx.size == 0:
        return np.array([], dtype=np.uint8)
    if rx.size % n_out != 0:
        if n_out == 2:
            raise ValueError("Viterbi input must have an even number of bits (r=1/2).")
        raise ValueError(
            f"Viterbi input length {rx.size} is not a multiple of {n_out} (r=1/{n_out})."
        )

    n_steps = rx.size // n_out
    # Symbol values 0..2**n_out - 1 for each received group.
    rx_sym = np.zeros(n_steps, dtype=np.int16)
    for j in range(n_out):
        rx_sym = (rx_sym << 1) | rx[j::n_out].astype(np.int16)

    pred_a = code["pred_a"]
    pred_b = code["pred_b"]
    out_a = code["out_a"]
    out_b = code["out_b"]
    popcount = code["popcount"]
    in_bit = code["in_bit"]

    INF = np.int32(1 << 20)
    metric = np.full(n_states, INF, dtype=np.int32)
    metric[0] = 0
    trace = np.empty((n_steps, n_states), dtype=np.int32)

    for t in range(n_steps):
        sym = int(rx_sym[t])
        branch_a = popcount[sym ^ out_a].astype(np.int32)
        branch_b = popcount[sym ^ out_b].astype(np.int32)
        cand_a = metric[pred_a] + branch_a
        cand_b = metric[pred_b] + branch_b
        take_a = cand_a <= cand_b
        metric = np.where(take_a, cand_a, cand_b).astype(np.int32)
        trace[t] = np.where(take_a, pred_a, pred_b).astype(np.int32)

    # Terminated code -> end in state 0; otherwise pick the best final state.
    state = 0
    decoded = np.empty(n_steps, dtype=np.uint8)
    for t in range(n_steps - 1, -1, -1):
        decoded[t] = (in_bit[state]) & 1
        state = int(trace[t, state])
    return decoded


def viterbi_decode(rx_bits: np.ndarray) -> np.ndarray:
    """Hard-decision Viterbi decoder for the K=7, r=1/2 code above.

    Fully vectorized across the 64 trellis states. Returns the decoded
    information bits (including any tail bits the encoder appended).

    Raises:
        ValueError: when the received stream length is odd.
    """
    return viterbi_decode_general(rx_bits, k=CONV_K, generators=(CONV_G1, CONV_G2))


def hard_bits_to_llr(
    bits: np.ndarray, n_out: int = 2, llr_mag: float = 2.0
) -> np.ndarray:
    """Map hard bits to LLRs of shape ``(n_steps, n_out)``.

    Bit 0 maps to ``+llr_mag`` and bit 1 to ``-llr_mag`` (the LLR convention
    ``log(P0/P1)`` that :func:`viterbi_decode_soft` expects). This is what the
    blind search feeds the soft decoder when only hard decisions are
    available; a receiver with real channel state would supply measured LLRs
    instead and see the full soft gain.
    """
    arr = np.asarray(bits, dtype=np.uint8).ravel()
    n_out = int(n_out)
    if arr.size == 0:
        return np.zeros((0, n_out), dtype=np.float64)
    if arr.size % n_out != 0:
        raise ValueError(
            f"hard-bit stream length {arr.size} is not a multiple of {n_out}."
        )
    grouped = arr.reshape(-1, n_out)
    return np.where(grouped == 0, float(llr_mag), -float(llr_mag)).astype(np.float64)


# --------------------------------------------------------------------------- #
# GF(256) arithmetic + Reed-Solomon
# --------------------------------------------------------------------------- #

GF_PRIM_POLY = 0x11D  # x^8 + x^4 + x^3 + x^2 + 1
GF_SIZE = 256
GF_ORDER = 255


def _build_gf_tables() -> tuple[list[int], list[int]]:
    exp = [0] * (GF_ORDER * 2)
    log = [0] * GF_SIZE
    x = 1
    for i in range(GF_ORDER):
        exp[i] = x
        log[x] = i
        x <<= 1
        if x & 0x100:
            x ^= GF_PRIM_POLY
    for i in range(GF_ORDER, GF_ORDER * 2):
        exp[i] = exp[i - GF_ORDER]
    return exp, log


GF_EXP, GF_LOG = _build_gf_tables()


def gf_mul(a: int, b: int) -> int:
    """Multiply two GF(256) elements."""
    if a == 0 or b == 0:
        return 0
    return GF_EXP[GF_LOG[a] + GF_LOG[b]]


def gf_div(a: int, b: int) -> int:
    """Divide a by b in GF(256)."""
    if b == 0:
        raise ZeroDivisionError("GF(256) division by zero")
    if a == 0:
        return 0
    return GF_EXP[(GF_LOG[a] - GF_LOG[b]) % GF_ORDER]


def gf_pow(a: int, n: int) -> int:
    """Raise a GF(256) element to an integer power."""
    if a == 0:
        return 0
    return GF_EXP[(GF_LOG[a] * n) % GF_ORDER]


def gf_inverse(a: int) -> int:
    """Multiplicative inverse in GF(256)."""
    if a == 0:
        raise ZeroDivisionError("GF(256) inverse of zero")
    return GF_EXP[GF_ORDER - GF_LOG[a]]


def gf_poly_scale(p: list[int], x: int) -> list[int]:
    return [gf_mul(c, x) for c in p]


def gf_poly_add(p: list[int], q: list[int]) -> list[int]:
    r = [0] * max(len(p), len(q))
    for i, c in enumerate(p):
        r[i + len(r) - len(p)] = c
    for i, c in enumerate(q):
        r[i + len(r) - len(q)] ^= c
    return r


def gf_poly_mul(p: list[int], q: list[int]) -> list[int]:
    r = [0] * (len(p) + len(q) - 1)
    for j, qj in enumerate(q):
        if qj == 0:
            continue
        lq = GF_LOG[qj]
        for i, pi in enumerate(p):
            if pi != 0:
                r[i + j] ^= GF_EXP[GF_LOG[pi] + lq]
    return r


def gf_poly_eval(poly: list[int], x: int) -> int:
    """Horner evaluation of a polynomial at x in GF(256)."""
    if not poly:
        return 0
    y = poly[0]
    for c in poly[1:]:
        y = gf_mul(y, x) ^ c
    return y


def gf_poly_div(dividend: list[int], divisor: list[int]) -> tuple[list[int], list[int]]:
    """Polynomial long division; returns (quotient, remainder)."""
    msg = list(dividend)
    for i in range(len(dividend) - (len(divisor) - 1)):
        coef = msg[i]
        if coef != 0:
            lc = GF_LOG[coef]
            for j in range(1, len(divisor)):
                if divisor[j] != 0:
                    msg[i + j] ^= GF_EXP[GF_LOG[divisor[j]] + lc]
    sep = -(len(divisor) - 1)
    return msg[:sep], msg[sep:]


class ReedSolomonError(ValueError):
    """Raised when a Reed-Solomon codeword cannot be corrected."""


def rs_generator_poly(nsym: int) -> list[int]:
    """Generator polynomial g(x) = prod_{i=0}^{nsym-1} (x - a^i)."""
    g = [1]
    for i in range(nsym):
        g = gf_poly_mul(g, [1, GF_EXP[i]])
    return g


def rs_encode(msg: bytes, nsym: int) -> bytes:
    """Systematically encode ``msg`` with ``nsym`` RS parity symbols.

    Supports the full-length RS(255,223) (``nsym=32``) and shortened codes
    (any message shorter than 223 bytes).
    """
    data = list(bytes(msg))
    if len(data) + nsym > GF_ORDER:
        raise ValueError(
            f"message too long for RS(255,{255 - nsym}): "
            f"{len(data)} + {nsym} > {GF_ORDER}"
        )
    gen = rs_generator_poly(nsym)
    msg_out = data + [0] * (len(gen) - 1)
    for i in range(len(data)):
        coef = msg_out[i]
        if coef != 0:
            lc = GF_LOG[coef]
            for j in range(1, len(gen)):
                if gen[j] != 0:
                    msg_out[i + j] ^= GF_EXP[GF_LOG[gen[j]] + lc]
    return bytes(msg_out[len(data) :])


def _rs_syndromes(msg: list[int], nsym: int) -> list[int]:
    # Index 0 is a placeholder so syndrome i lives at position i (1-based).
    return [0] + [gf_poly_eval(msg, GF_EXP[i]) for i in range(nsym)]


def rs_check(msg: bytes, nsym: int) -> bool:
    """True when every syndrome is zero (codeword is valid)."""
    return max(_rs_syndromes(list(bytes(msg)), nsym)) == 0


def _rs_find_error_locator(synd: list[int], nsym: int) -> list[int]:
    """Berlekamp-Massey: build the error-locator polynomial."""
    err_loc = [1]
    old_loc = [1]
    synd_shift = len(synd) - nsym if len(synd) > nsym else 0

    for i in range(nsym):
        k = i + synd_shift
        delta = synd[k]
        for j in range(1, len(err_loc)):
            delta ^= gf_mul(err_loc[-(j + 1)], synd[k - j])
        old_loc = [*old_loc, 0]
        if delta != 0:
            if len(old_loc) > len(err_loc):
                new_loc = gf_poly_scale(old_loc, delta)
                old_loc = gf_poly_scale(err_loc, gf_inverse(delta))
                err_loc = new_loc
            err_loc = gf_poly_add(err_loc, gf_poly_scale(old_loc, delta))

    while len(err_loc) and err_loc[0] == 0:
        del err_loc[0]
    if (len(err_loc) - 1) * 2 > nsym:
        raise ReedSolomonError("too many errors to correct")
    return err_loc


def _rs_find_errors(err_loc: list[int], nmess: int) -> list[int]:
    """Chien search: locate error positions from the locator polynomial."""
    errs = len(err_loc) - 1
    err_pos = [i for i in range(nmess) if gf_poly_eval(err_loc, GF_EXP[i]) == 0]
    err_pos = [nmess - 1 - p for p in err_pos]
    if len(err_pos) != errs:
        raise ReedSolomonError("could not locate errors")
    return err_pos


def _rs_errata_locator(coef_pos: list[int]) -> list[int]:
    e_loc = [1]
    for p in coef_pos:
        e_loc = gf_poly_mul(e_loc, gf_poly_add([1], [GF_EXP[p], 0]))
    return e_loc


def _rs_error_evaluator(synd: list[int], err_loc: list[int], nsym: int) -> list[int]:
    _, remainder = gf_poly_div(gf_poly_mul(synd, err_loc), [1] + [0] * (nsym + 1))
    return remainder


def _rs_correct_errata(
    msg: list[int], synd: list[int], err_pos: list[int]
) -> list[int]:
    """Forney algorithm: compute error magnitudes and correct the message."""
    coef_pos = [len(msg) - 1 - p for p in err_pos]
    err_loc = _rs_errata_locator(coef_pos)
    err_eval = _rs_error_evaluator(synd[::-1], err_loc, len(err_loc) - 1)[::-1]

    x_vals = [gf_pow(2, -(GF_ORDER - p)) for p in coef_pos]
    magnitudes = [0] * len(msg)
    for i, xi in enumerate(x_vals):
        xi_inv = gf_inverse(xi)
        prime = 1
        for j in range(len(x_vals)):
            if j != i:
                prime = gf_mul(prime, 1 ^ gf_mul(xi_inv, x_vals[j]))
        y = gf_poly_eval(err_eval[::-1], xi_inv)
        y = gf_mul(xi, y)
        magnitudes[err_pos[i]] = gf_div(y, prime)
    return gf_poly_add(msg, magnitudes)


def rs_decode(msg: bytes, nsym: int) -> tuple[bytes, int]:
    """Decode/correct a Reed-Solomon codeword.

    Returns ``(corrected_bytes, errors_corrected)``. ``errors_corrected`` is 0
    for an already-valid codeword.

    Raises:
        ReedSolomonError: when the codeword is beyond the correction radius.
    """
    msg = list(bytes(msg))
    nsym = int(nsym)
    if nsym <= 0 or nsym >= len(msg):
        raise ReedSolomonError("invalid nsym for this codeword length")
    if rs_check(bytes(msg), nsym):
        return bytes(msg), 0

    synd = _rs_syndromes(msg, nsym)
    err_loc = _rs_find_error_locator(synd, nsym)
    if len(err_loc) - 1 == 0:
        raise ReedSolomonError("no error locator found")
    err_pos = _rs_find_errors(err_loc[::-1], len(msg))
    corrected = _rs_correct_errata(msg, synd, err_pos)
    if not rs_check(bytes(corrected), nsym):
        raise ReedSolomonError("correction failed (too many errors)")
    return bytes(corrected), len(err_pos)


# --------------------------------------------------------------------------- #
# LDPC: small (3,4)-regular code with normalized min-sum decoding
# --------------------------------------------------------------------------- #

LDPC_K = 48  # information bits per block (6 bytes)
LDPC_M = 36  # parity checks per block
LDPC_N = LDPC_K + LDPC_M  # codeword length (84), rate 0.571
LDPC_DV = 3  # ones per message column  (variable-node degree)
LDPC_DC = 4  # ones per check row       (check-node degree)
LDPC_SEED = 20260918


def _build_ldpc_matrix(
    k: int = LDPC_K, m: int = LDPC_M, dv: int = LDPC_DV, dc: int = LDPC_DC
) -> np.ndarray:
    """Build a systematic ``(dv, dc)``-regular matrix ``H = [A | I]``.

    ``dv`` and ``dc`` describe the information-side block ``A``: every
    information column has ``dv`` ones and every row of ``A`` has ``dc`` ones.
    The identity block gives each parity variable degree one, so complete check
    rows have weight ``dc + 1``.  A cyclic construction is used when ``k`` is a
    multiple of ``m``; a deterministic balanced-stub fallback covers other
    dimensions.  Both paths reject duplicate column supports, which otherwise
    make the BP messages indistinguishable.
    """
    k, m, dv, dc = (int(k), int(m), int(dv), int(dc))
    if min(k, m, dv, dc) <= 0:
        raise ValueError("LDPC dimensions and degrees must be positive")
    if k * dv != m * dc:
        raise ValueError(
            f"k*dv ({k * dv}) must equal m*dc ({m * dc}) for a regular code"
        )
    if dv > m or dc > k:
        raise ValueError(
            f"degree (dv={dv}, dc={dc}) cannot fit k={k}, m={m} information block"
        )

    def complete(a: np.ndarray) -> np.ndarray | None:
        signatures = {tuple(np.flatnonzero(a[:, j])) for j in range(k)}
        if len(signatures) != k:
            return None
        return np.hstack([a, np.eye(m, dtype=np.uint8)]).astype(np.uint8)

    # Each group of m columns visits every row exactly dv times.  Across
    # groups = k/m this gives exactly groups*dv == dc ones per A row, without
    # the low acceptance probability of dealing shuffled row stubs.
    if k % m == 0 and dv < m:
        groups = k // m
        for seed_offset in range(5000):
            rng = np.random.default_rng(LDPC_SEED + seed_offset)
            offset_sets: list[tuple[int, ...]] = []
            seen: set[tuple[int, ...]] = set()
            for _ in range(groups):
                for _attempt in range(500):
                    offsets = tuple(
                        sorted(int(x) for x in rng.choice(m, size=dv, replace=False))
                    )
                    supports = [
                        tuple(sorted((base + off) % m for off in offsets))
                        for base in range(m)
                    ]
                    if all(support not in seen for support in supports):
                        offset_sets.append(offsets)
                        seen.update(supports)
                        break
                else:
                    break
            if len(offset_sets) != groups:
                continue
            a = np.zeros((m, k), dtype=np.uint8)
            for group, offsets in enumerate(offset_sets):
                for base in range(m):
                    column = group * m + base
                    for off in offsets:
                        a[(base + off) % m, column] = 1
            built = complete(a)
            if built is not None:
                return built

    # General fallback for dimensions where k is not divisible by m.
    for seed_offset in range(20_000):
        rng = np.random.default_rng(LDPC_SEED + seed_offset)
        pool = np.repeat(np.arange(m), dc)
        rng.shuffle(pool)
        a = np.zeros((m, k), dtype=np.uint8)
        ok = True
        for j in range(k):
            rows = pool[j * dv : (j + 1) * dv]
            if np.unique(rows).size != dv:
                ok = False
                break
            a[rows, j] = 1
        if not ok:
            continue
        built = complete(a)
        if built is not None:
            return built

    raise RuntimeError("could not construct a duplicate-free regular LDPC matrix")


def build_ldpc_matrix(
    k: int = LDPC_K, m: int = LDPC_M, dv: int = LDPC_DV, dc: int = LDPC_DC
) -> np.ndarray:
    """Public alias for the parametric LDPC matrix builder."""
    return _build_ldpc_matrix(k=k, m=m, dv=dv, dc=dc)


LDPC_H = _build_ldpc_matrix()


def ldpc_encode(bits: np.ndarray, h: np.ndarray | None = None) -> np.ndarray:
    """Systematic LDPC encode: returns ``[u | p]`` with ``p = A @ u mod 2``."""
    h = LDPC_H if h is None else h
    u = np.asarray(bits, dtype=np.uint8).ravel()
    k = h.shape[1] - h.shape[0]
    if u.size == 0:
        return np.array([], dtype=np.uint8)
    if u.size != k:
        raise ValueError(f"LDPC expects exactly {k} information bits, got {u.size}")
    a = h[:, :k]
    p = (a @ u) % 2
    return np.concatenate([u, p.astype(np.uint8)])


def ldpc_syndrome_ok(bits: np.ndarray, h: np.ndarray | None = None) -> bool:
    """True when ``H @ bits == 0 (mod 2)`` — i.e. the codeword is valid."""
    h = LDPC_H if h is None else h
    c = np.asarray(bits, dtype=np.uint8).ravel()
    if c.size != h.shape[1]:
        return False
    return bool(np.all((h @ c) % 2 == 0))


def _edge_structure(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten ``H`` into a padded ``(m, dc_max)`` edge table plus a validity mask.

    The padded layout lets every check-node update be expressed as a whole-array
    NumPy operation instead of a per-edge loop, which is what keeps belief
    propagation inside the §NFR-03 time budget on real captures.
    """
    m = h.shape[0]
    rows = [np.flatnonzero(h[c]) for c in range(m)]
    dc_max = max((r.size for r in rows), default=0)
    if dc_max == 0:
        return np.zeros((m, 0), dtype=np.intp), np.zeros((m, 0), dtype=bool)
    edge_v = np.zeros((m, dc_max), dtype=np.intp)
    valid = np.zeros((m, dc_max), dtype=bool)
    for c, r in enumerate(rows):
        edge_v[c, : r.size] = r
        valid[c, : r.size] = True
    return edge_v, valid


#: Precomputed edge table for the shared matrix. Building it once at import
#: time keeps the per-block decode allocation-free.
_LDPC_EDGES: tuple[np.ndarray, np.ndarray] = _edge_structure(LDPC_H)


def _cached_edges(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the edge table for ``h``.

    Only the shared ``LDPC_H`` is served from the module cache (it is a
    long-lived constant); a caller-supplied matrix is rebuilt each call so the
    cache can never go stale against a recycled object identity.
    """
    if h is LDPC_H:
        return _LDPC_EDGES
    return _edge_structure(h)


def ldpc_decode(
    bits: np.ndarray,
    h: np.ndarray | None = None,
    p_err: float = 0.05,
    max_iter: int = 60,
    alpha: float = 0.75,
) -> np.ndarray:
    """Normalized min-sum belief propagation for the systematic LDPC code.

    ``p_err`` is the assumed BSC crossover probability used to seed the
    channel LLRs. Returns the decoded codeword (``[u | p]``); check validity
    with :func:`ldpc_syndrome_ok`.

    Fully vectorized: the check-node min-sum is computed from a per-row sort
    (the two smallest magnitudes decide every outgoing message) and the
    variable-node sum uses ``bincount``, so no per-edge Python loop remains.
    """
    h = LDPC_H if h is None else h
    rx = np.asarray(bits, dtype=np.uint8).ravel()
    n = h.shape[1]

    if rx.size != n:
        raise ValueError(f"LDPC expects exactly {n} bits, got {rx.size}")

    p_err = float(min(max(p_err, 1e-6), 0.5 - 1e-6))
    llr_mag = float(np.log((1.0 - p_err) / p_err))
    llr0 = np.where(rx == 0, llr_mag, -llr_mag).astype(np.float64)

    edge_v, valid = _cached_edges(h)
    if edge_v.shape[1] == 0:
        return (llr0 < 0).astype(np.uint8)
    flat_v = edge_v.ravel()

    v2c = np.where(valid, llr0[edge_v], 0.0)
    total = llr0.copy()

    for _ in range(max_iter):
        # ---- check node update: normalized min-sum, self-excluded ----
        abs_msg = np.where(valid, np.abs(v2c), np.inf)
        signs = np.where(valid, np.where(v2c >= 0.0, 1.0, -1.0), 1.0)
        total_sign = np.prod(signs, axis=1, keepdims=True)

        order = np.argsort(abs_msg, axis=1, kind="stable")
        sorted_abs = np.take_along_axis(abs_msg, order, axis=1)
        min1 = sorted_abs[:, 0:1]
        min2 = sorted_abs[:, 1:2] if sorted_abs.shape[1] > 1 else min1
        # The smallest-magnitude edge must use the *second* smallest; every
        # other edge uses the smallest.
        smallest = np.zeros_like(valid)
        np.put_along_axis(smallest, order[:, 0:1], True, axis=1)
        others_mag = np.where(smallest, min2, min1)

        # prod(signs except i) == total_sign * signs[i] because signs are +/-1.
        c2v = np.where(valid, alpha * total_sign * signs * others_mag, 0.0)

        # ---- variable node update ----
        # c2v is already zero on padding edges, so a plain weighted bincount
        # gives sum(c2v) per variable node.
        total = llr0 + np.bincount(flat_v, weights=c2v.ravel(), minlength=n)
        v2c = np.where(valid, total[edge_v] - c2v, 0.0)

        decision = (total < 0).astype(np.uint8)
        if bool(np.all((h @ decision) % 2 == 0)):
            return decision

    return (total < 0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Concatenated code: RS outer + convolutional inner
# --------------------------------------------------------------------------- #

Interleaver = Callable[[np.ndarray], np.ndarray]


def concatenated_encode(
    payload: bytes, nsym: int = 32, interleave: Interleaver | None = None
) -> np.ndarray:
    """Encode the historical K=7 r=1/2 concatenated scheme.

    The general family form is :func:`concatenated_encode_general`; this
    compatibility wrapper deliberately keeps the original positional API.
    """
    return concatenated_encode_general(payload, nsym=nsym, interleave=interleave)


def concatenated_decode(
    bits: np.ndarray,
    nsym: int = 32,
    deinterleave: Interleaver | None = None,
) -> dict:
    """Decode the historical K=7 r=1/2 concatenated scheme.

    The general family form is :func:`concatenated_decode_general`; this
    wrapper preserves the original return shape and error handling.
    """
    return concatenated_decode_general(bits, nsym=nsym, deinterleave=deinterleave)


# --------------------------------------------------------------------------- #
# Blind candidate scoring (unchanged honest contract)
# --------------------------------------------------------------------------- #

CANDIDATES = ["Convolutional r=1/2 K=7", "RS(255,223)", "LDPC", "Concatenated"]


def _parity_score(bits: np.ndarray) -> float:
    if len(bits) < 16:
        return 0.0
    # Even-parity drift proxy for conv code.
    fails = int(np.sum(bits[1:] != bits[:-1]))
    return float(max(0.0, min(0.5, 0.35 - fails / len(bits) * 0.2)))


def score_fec_candidates(bits: np.ndarray) -> dict:
    """Blind FEC candidate scoring — confidence capped at 0.5 (never blind).

    This is intentionally *not* a detector: see ``info.md`` §30. For verified
    decoding use :func:`decode_hypotheses`.
    """
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


# --------------------------------------------------------------------------- #
# Generic convolutional code family (K = 5..9, r = 1/2 and 1/3)
# --------------------------------------------------------------------------- #
#
# The original release shipped exactly one convolutional code (K=7, r=1/2) and one LDPC
# matrix ((3,4)-regular, n=84). A blind search over a single code is not a
# blind search -- it is a guess with a decoder attached. This section adds the
# *family*: the constraint lengths and rates a real link actually uses, decoded
# with a soft-decision Viterbi so the search also survives the low SNR where a
# hard decision has already thrown the information away.
#
# Generator polynomials are the published octal sets for each (K, rate). Their
# optimality is not what matters here -- the search only needs each code to be
# decodable and *distinguishable*, and a wrong code hypothesis fails the CRC.


@dataclass(frozen=True)
class ConvCode:
    """One convolutional code: constraint length plus one generator per output.

    ``gens`` holds the octal generator polynomials, most significant bit first,
    one per output bit -- so ``len(gens)`` is the code's rate denominator (2 for
    r=1/2, 3 for r=1/3) and the code's constraint length is ``k``.
    """

    name: str
    k: int
    gens: tuple[int, ...]

    @property
    def rate_den(self) -> int:
        """Outputs per input bit (the rate denominator)."""
        return len(self.gens)

    @property
    def states(self) -> int:
        return 1 << (self.k - 1)

    @property
    def tail(self) -> int:
        """Zero tail bits the encoder appends to flush the trellis."""
        return self.k - 1


#: The code family the blind search sweeps.  Keep the project's own
#: ``K=7 r=1/2`` path first, but derive every row from :data:`CONV_GENERATORS`
#: so the public parameter table and the search trellises cannot drift.
_CONV_CODE_ORDER = (
    (7, "1/2"),
    (5, "1/2"),
    (6, "1/2"),
    (8, "1/2"),
    (9, "1/2"),
    (5, "1/3"),
    (6, "1/3"),
    (7, "1/3"),
    (8, "1/3"),
    (9, "1/3"),
)
CONV_CODES: tuple[ConvCode, ...] = tuple(
    ConvCode(f"K={k} r={rate}", k, CONV_GENERATORS[(k, rate)])
    for k, rate in _CONV_CODE_ORDER
)

CONV_CODE_BY_NAME: dict[str, ConvCode] = {c.name: c for c in CONV_CODES}


def _symbol_table(code: ConvCode) -> np.ndarray:
    """``out[reg]`` -> packed output symbol index for every register value.

    One lookup per code, computed once: the trellis is built from it and the
    encoder is built from it, so the two cannot disagree about the code.
    """
    width = 1 << code.k
    regs = np.arange(width, dtype=np.int64)
    table = np.zeros(width, dtype=np.int64)
    for generator in code.gens:
        table = (table << 1) | _parity_of(regs & generator)
    return table


def _parity_of(value: np.ndarray) -> np.ndarray:
    """Parity (mod-2 bit sum) of each element of an integer array."""
    v = np.asarray(value, dtype=np.int64).copy()
    out = np.zeros(v.shape, dtype=np.int64)
    while np.any(v):
        out ^= v & 1
        v >>= 1
    return out


_SYMBOL_TABLES: dict[str, np.ndarray] = {}


def _symbols_for(code: ConvCode) -> np.ndarray:
    table = _SYMBOL_TABLES.get(code.name)
    if table is None:
        table = _symbol_table(code)
        _SYMBOL_TABLES[code.name] = table
    return table


@dataclass(frozen=True)
class _Trellis:
    """Precomputed predecessor/output tables for one code."""

    in_bit: np.ndarray
    pred_a: np.ndarray
    pred_b: np.ndarray
    out_a: np.ndarray
    out_b: np.ndarray


_TRELLISES: dict[str, _Trellis] = {}


def _trellis_for(code: ConvCode) -> _Trellis:
    cached = _TRELLISES.get(code.name)
    if cached is not None:
        return cached
    symbols = _symbols_for(code)
    states = np.arange(code.states, dtype=np.int64)
    in_bit = states >> (code.k - 2)
    pred_a = (states & (code.states // 2 - 1)) << 1
    pred_b = pred_a | 1
    out_a = symbols[(in_bit << (code.k - 1)) | pred_a]
    out_b = symbols[(in_bit << (code.k - 1)) | pred_b]
    trellis = _Trellis(in_bit, pred_a, pred_b, out_a, out_b)
    _TRELLISES[code.name] = trellis
    return trellis


def conv_encode_generic(
    bits: np.ndarray, code: ConvCode, terminate: bool = True
) -> np.ndarray:
    """Encode with an arbitrary code from :data:`CONV_CODES`.

    With ``terminate=True`` the ``k - 1`` zero tail bits are appended so the
    trellis ends in state 0, which is what the decoder assumes.
    """
    arr = np.asarray(bits, dtype=np.uint8).ravel()
    if terminate:
        arr = np.concatenate([arr, np.zeros(code.tail, dtype=np.uint8)])
    if arr.size == 0:
        return np.array([], dtype=np.uint8)

    symbols = _symbols_for(code)
    rate_den = code.rate_den
    out = np.empty(arr.size * rate_den, dtype=np.uint8)
    state = 0
    for i, bit in enumerate(arr):
        reg = (int(bit) << (code.k - 1)) | state
        sym = int(symbols[reg])
        for j in range(rate_den):
            out[i * rate_den + j] = (sym >> (rate_den - 1 - j)) & 1
        # Next state keeps the newest k-1 register bits: the input bit just
        # shifted in plus the previous state's upper bits. (Masking the low
        # bits instead would drop the input and freeze the trellis in state 0,
        # a memoryless mapping -- this line is what makes it a real code.)
        state = reg >> 1
    return out


def _branch_metrics_hard(rx_sym: np.ndarray, width: int) -> np.ndarray:
    """Hamming distance from every received symbol to every ideal symbol.

    The family trellis supports at most three output bits, so a tiny lookup
    table is both clearer and faster than accidentally using the GF(2)
    *parity* of the XOR.  The distinction matters for rate-1/3: parity is not
    a Hamming distance and makes the hard-decision path fail on modest noise.
    """
    popcount = np.array([int(i).bit_count() for i in range(width)], dtype=np.int32)
    return popcount[np.asarray(rx_sym, dtype=np.intp)[:, None] ^ np.arange(width)]


def _resolve_conv_code(
    code: ConvCode | str | int | None = None,
    *,
    k: int | None = None,
    generators: tuple[int, ...] | None = None,
) -> ConvCode:
    """Resolve a code name/object or a ``k``/generator table request."""
    if isinstance(code, int):
        if k is not None and int(k) != code:
            raise ValueError("conflicting positional and keyword constraint lengths")
        k = code
        code = None
    if code is None:
        if k is None:
            return CONV_CODE_BY_NAME["K=7 r=1/2"]
        if generators is None:
            # The public table's default family row is rate 1/2 when no output
            # polynomials are supplied.
            generators = CONV_GENERATORS[(int(k), "1/2")]
        rate = f"1/{len(generators)}"
        return ConvCode(f"K={int(k)} r={rate}", int(k), tuple(generators))
    if isinstance(code, str):
        try:
            resolved = CONV_CODE_BY_NAME[code]
        except KeyError as exc:
            raise ValueError(f"unknown convolutional code {code!r}") from exc
    else:
        resolved = code
    if k is not None and int(k) != int(resolved.k):
        raise ValueError("code and k specify different constraint lengths")
    if generators is not None and tuple(generators) != tuple(resolved.gens):
        raise ValueError("code and generators specify different polynomials")
    return resolved


def viterbi_decode_soft(
    rx: np.ndarray | None = None,
    code: ConvCode | str | int | None = None,
    *,
    llr: np.ndarray | None = None,
    k: int | None = None,
    generators: tuple[int, ...] | None = None,
    soft_scale: float = 1.0,
) -> np.ndarray:
    """Soft-decision Viterbi for any code in :data:`CONV_CODES`.

    ``llr`` uses the usual convention ``log(P(bit=0)/P(bit=1))``: positive
    values prefer zero.  It may be supplied as the ``llr`` keyword alongside
    hard ``rx`` bits, or directly as the first positional argument.  A direct
    LLR argument is detected by its non-binary dtype/values, which keeps the
    historical ``viterbi_decode_soft(coded, code)`` call working unchanged.
    The optional ``k``/``generators`` form is retained for callers using the
    original family-table API.
    """
    conv = _resolve_conv_code(code, k=k, generators=generators)
    raw = np.asarray(llr if rx is None else rx)
    direct_llr = llr is None and (
        np.issubdtype(raw.dtype, np.floating)
        or (raw.size > 0 and not np.all((raw == 0) | (raw == 1)))
    )
    if direct_llr:
        llr = raw
        bits = (raw > 0).astype(np.uint8).ravel()
    else:
        bits = np.asarray(raw, dtype=np.uint8).ravel()
        if bits.size and not np.all((bits == 0) | (bits == 1)):
            raise ValueError("hard received bits must contain only 0/1")

    rate_den = conv.rate_den
    if bits.size % rate_den != 0:
        raise ValueError(
            f"{conv.name} input must be a multiple of {rate_den} bits, "
            f"got {bits.size}"
        )
    n_steps = bits.size // rate_den
    if n_steps == 0:
        return np.array([], dtype=np.uint8)

    width = 1 << rate_den
    grouped = bits.reshape(n_steps, rate_den).astype(np.int64)
    weights = 1 << np.arange(rate_den - 1, -1, -1, dtype=np.int64)
    rx_sym = grouped @ weights

    trellis = _trellis_for(conv)
    if llr is None:
        # Hard input: metric is -Hamming, so maximising it is minimising
        # distance.  Keep it integral for exact tie handling.
        metrics = -_branch_metrics_hard(rx_sym, width).astype(np.int32)
    else:
        soft = np.asarray(llr, dtype=np.float64).ravel() * float(soft_scale)
        if soft.size < bits.size:
            raise ValueError("llr must cover every received bit")
        soft = soft[: n_steps * rate_den].reshape(n_steps, rate_den)
        # Correlation with the ideal bipolar symbol, for all ideal symbols.
        ideal = np.zeros((width, rate_den), dtype=np.float64)
        for symbol in range(width):
            for j in range(rate_den):
                ideal[symbol, j] = 1.0 - 2.0 * ((symbol >> (rate_den - 1 - j)) & 1)
        metrics = (soft @ ideal.T) * 0.5

    n_states = conv.states
    neg = np.float64(-1e18) if llr is not None else np.int32(-(1 << 28))
    metric = np.full(n_states, neg, dtype=metrics.dtype)
    metric[0] = 0
    trace = np.empty((n_steps, n_states), dtype=np.int32)

    for t in range(n_steps):
        row = metrics[t]
        cand_a = metric[trellis.pred_a] + row[trellis.out_a]
        cand_b = metric[trellis.pred_b] + row[trellis.out_b]
        take_a = cand_a >= cand_b
        metric = np.where(take_a, cand_a, cand_b)
        trace[t] = np.where(take_a, trellis.pred_a, trellis.pred_b)

    # Terminated hard streams end in state 0.  With soft information, the
    # strongest final state is safer when the tail itself was uncertain.
    state = int(np.argmax(metric)) if llr is not None else 0
    decoded = np.empty(n_steps, dtype=np.uint8)
    for t in range(n_steps - 1, -1, -1):
        decoded[t] = (state >> (conv.k - 2)) & 1
        state = int(trace[t, state])
    return decoded


def _concatenated_failure(
    message: str, *, rs_ok: bool = False, errors: int = 0
) -> dict:
    """Uniform failure record for both concatenated APIs."""
    return {
        "crc_pass": False,
        "payload": b"",
        "errors_corrected": int(errors),
        "rs_ok": bool(rs_ok),
        "error": message,
    }


def concatenated_encode_general(
    payload: bytes,
    nsym: int = 32,
    code: ConvCode | str | int = "K=7 r=1/2",
    interleave: Interleaver | None = None,
    *,
    k: int | None = None,
    generators: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Encode CRC-16 -> RS outer -> convolutional inner for any family row.

    ``rs_encode`` returns parity only, so the systematic RS word is explicitly
    assembled before bit packing.  ``interleave`` is applied last, matching the
    transmit-side frame layout.
    """
    conv = _resolve_conv_code(code, k=k, generators=generators)
    framed = crc16_append(bytes(payload))
    outer = framed + rs_encode(framed, int(nsym))
    inner = conv_encode_generic(bytes_to_bits(outer), conv, terminate=True)
    if interleave is not None:
        inner = np.asarray(interleave(inner), dtype=np.uint8).ravel()
    return np.asarray(inner, dtype=np.uint8)


def concatenated_decode_general(
    bits: np.ndarray,
    nsym: int = 32,
    code: ConvCode | str | int = "K=7 r=1/2",
    deinterleave: Interleaver | None = None,
    *,
    k: int | None = None,
    generators: tuple[int, ...] | None = None,
    llr: np.ndarray | None = None,
    frame_bits: int | None = None,
) -> dict:
    """Decode a concatenated family row and require a real CRC-16 pass.

    If a length-preserving interleaver padded the stream, the decoder tries
    byte-aligned codeword prefixes after the inner Viterbi stage.  A candidate
    is accepted only when both the RS syndrome and the embedded CRC pass; this
    prevents padding from turning into an invented payload.
    """
    conv = _resolve_conv_code(code, k=k, generators=generators)
    stream = np.asarray(bits, dtype=np.uint8).ravel()
    soft = None if llr is None else np.asarray(llr, dtype=np.float64).ravel()
    if deinterleave is not None:
        try:
            stream = np.asarray(deinterleave(stream), dtype=np.uint8).ravel()
            if soft is not None:
                soft = np.asarray(deinterleave(soft), dtype=np.float64).ravel()
        except Exception as exc:
            return _concatenated_failure(f"deinterleaver failed: {exc}")
    if frame_bits is not None and int(frame_bits) > 0:
        stream = stream[: int(frame_bits)]
        if soft is not None:
            soft = soft[: int(frame_bits)]
    if stream.size == 0:
        return _concatenated_failure("empty concatenated stream")
    if stream.size % conv.rate_den != 0:
        return _concatenated_failure(
            f"inner stream length {stream.size} is not a multiple of {conv.rate_den}"
        )
    try:
        inner = viterbi_decode_soft(stream, conv, llr=soft)
    except (TypeError, ValueError) as exc:
        return _concatenated_failure(str(exc))
    if inner.size <= conv.tail:
        return _concatenated_failure("inner codeword is shorter than its tail")
    inner = inner[: inner.size - conv.tail]

    n_bytes = inner.size // 8
    min_bytes = int(nsym) + 3  # at least one payload byte + CRC-16
    candidates = [n_bytes]
    if frame_bits is None:
        candidates.extend(range(n_bytes - 1, min_bytes - 1, -1))
    last_error = "no valid RS codeword"
    for n_candidate in candidates:
        if n_candidate <= min_bytes - 1:
            continue
        word = bits_to_bytes(inner[: n_candidate * 8])
        try:
            corrected, n_err = rs_decode(word, int(nsym))
        except (ReedSolomonError, ValueError) as exc:
            last_error = str(exc)
            continue
        if len(corrected) <= int(nsym):
            last_error = "short codeword"
            continue
        ok, payload = scan_crc16(corrected[: len(corrected) - int(nsym)])
        if ok:
            return {
                "crc_pass": True,
                "payload": payload,
                "errors_corrected": int(n_err),
                "rs_ok": True,
            }
        last_error = "RS codeword valid but CRC did not pass"
    return _concatenated_failure(last_error, rs_ok=True)


def viterbi_metric(rx: np.ndarray, decoded: np.ndarray, code: ConvCode | str) -> dict:
    """Re-encode ``decoded`` and measure how far the received bits drifted.

    Returns ``{"n_bit_errors", "bit_error_rate", "metric"}``. A decoded path
    that is far from the received bits is evidence the hypothesis is wrong even
    when a CRC happens to pass, which is why the caller keeps this number.
    """
    conv = CONV_CODE_BY_NAME[code] if isinstance(code, str) else code
    reference = conv_encode_generic(decoded, conv, terminate=False)
    bits = np.asarray(rx, dtype=np.uint8).ravel()
    n = min(reference.size, bits.size)
    errors = int(np.sum(bits[:n] != reference[:n]))
    return {
        "n_bit_errors": errors,
        "bit_error_rate": (errors / n) if n else 0.0,
        "metric": float(errors),
    }


# --------------------------------------------------------------------------- #
# LDPC families and the extended Reed-Solomon parameter sweep
# --------------------------------------------------------------------------- #
#
# A single (3,4)-regular matrix is not "LDPC" in the sense the brief means --
# the family spans column/row weights and block lengths, and a receiver that
# only knows one of them cannot decode a capture built with another. The
# matrices here are all systematic (``H = [A | I]``) so encoding stays a
# matrix-vector product and the syndrome check is the independent evidence the
# CRC is combined with.

LDPC_FAMILIES: tuple[dict, ...] = (
    {"name": "LDPC (3,4) n=84", "k": 48, "m": 36, "dv": 3, "dc": 4},
    # The following rows are deliberately real parametric codes, not labels
    # for the historical matrix.  Their dimensions satisfy k*dv == m*dc and
    # the builder verifies the resulting A block before it is cached.
    {"name": "LDPC (2,6) n=64", "k": 48, "m": 16, "dv": 2, "dc": 6},
    {"name": "LDPC (4,4) n=96", "k": 48, "m": 48, "dv": 4, "dc": 4},
    {"name": "LDPC (3,6) n=96", "k": 64, "m": 32, "dv": 3, "dc": 6},
    {"name": "LDPC (2,4) n=72", "k": 48, "m": 24, "dv": 2, "dc": 4},
    {"name": "LDPC (4,8) n=96", "k": 64, "m": 32, "dv": 4, "dc": 8},
)

_LDPC_FAMILY_CACHE: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}


def ldpc_family_matrix(family: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Build (and cache) one family's ``H``, its edge table and its ``k``.

    Returns ``(h, edge_v, valid, k)``. The edge table is cached alongside the
    matrix because rebuilding it per block was the dominant cost of the family
    search, not the belief propagation itself.
    """
    key = str(family["name"])
    cached = _LDPC_FAMILY_CACHE.get(key)
    if cached is not None:
        return cached
    h = _build_ldpc_matrix(
        k=int(family["k"]),
        m=int(family["m"]),
        dv=int(family["dv"]),
        dc=int(family["dc"]),
    )
    edge_v, valid = _edge_structure(h)
    entry = (h, edge_v, valid, int(family["k"]))
    _LDPC_FAMILY_CACHE[key] = entry
    return entry


def ldpc_encode_family(bits: np.ndarray, family: dict) -> np.ndarray:
    """Systematic encode with one family's matrix: ``[u | p]``, ``p = A u mod 2``."""
    h, _, _, k = ldpc_family_matrix(family)
    u = np.asarray(bits, dtype=np.uint8).ravel()
    if u.size != k:
        raise ValueError(f"{family['name']} expects exactly {k} information bits")
    return np.concatenate([u, ((h[:, :k] @ u) % 2).astype(np.uint8)])


def ldpc_decode_family(
    bits: np.ndarray, family: dict, p_err: float = 0.05, max_iter: int = 60
) -> tuple[np.ndarray, bool]:
    """Min-sum decode one block with a family's matrix.

    Returns ``(decoded, syndrome_ok)``. The syndrome is the independent
    structural check -- a CRC pass on its own is a 1/65536 coincidence, and the
    search must not be able to manufacture a decode out of one.
    """
    h, edge_v, valid, k = ldpc_family_matrix(family)
    n = h.shape[1]
    rx = np.asarray(bits, dtype=np.uint8).ravel()
    if rx.size != n:
        raise ValueError(f"{family['name']} expects exactly {n} bits, got {rx.size}")

    p = float(min(max(p_err, 1e-6), 0.5 - 1e-6))
    magnitude = float(np.log((1.0 - p) / p))
    llr0 = np.where(rx == 0, magnitude, -magnitude).astype(np.float64)

    if edge_v.shape[1] == 0:
        decision = (llr0 < 0).astype(np.uint8)
        return decision, bool(np.all((h @ decision) % 2 == 0))

    flat_v = edge_v.ravel()
    v2c = np.where(valid, llr0[edge_v], 0.0)
    total = llr0.copy()
    alpha = 0.75

    for _ in range(max_iter):
        abs_msg = np.where(valid, np.abs(v2c), np.inf)
        signs = np.where(valid, np.where(v2c >= 0.0, 1.0, -1.0), 1.0)
        total_sign = np.prod(signs, axis=1, keepdims=True)
        order = np.argsort(abs_msg, axis=1, kind="stable")
        sorted_abs = np.take_along_axis(abs_msg, order, axis=1)
        min1 = sorted_abs[:, 0:1]
        min2 = sorted_abs[:, 1:2] if sorted_abs.shape[1] > 1 else min1
        smallest = np.zeros_like(valid)
        np.put_along_axis(smallest, order[:, 0:1], True, axis=1)
        others = np.where(smallest, min2, min1)
        c2v = np.where(valid, alpha * total_sign * signs * others, 0.0)
        total = llr0 + np.bincount(flat_v, weights=c2v.ravel(), minlength=n)
        v2c = np.where(valid, total[edge_v] - c2v, 0.0)
        decision = (total < 0).astype(np.uint8)
        if bool(np.all((h @ decision) % 2 == 0)):
            return decision, True

    decision = (total < 0).astype(np.uint8)
    return decision, bool(np.all((h @ decision) % 2 == 0))


#: Reed-Solomon parity lengths swept by the family search.  The four rows
#: required by the blind-search contract are all present; the additional
#: lengths cover common shortened links without pretending that the message
#: length is known in advance.
RS_NSYM_FAMILY: tuple[int, ...] = (32, 16, 8, 64, 4, 24, 12, 6, 20, 10, 28)


# --------------------------------------------------------------------------- #
# CFAR accounting for the CRC search
# --------------------------------------------------------------------------- #
#
# Every CRC-16 comparison is a 1/65536 chance of a random pass, so a search
# that runs *many* comparisons has a false-alarm rate that grows with the
# number of hypotheses. That is the same statistics as the sync-word search
# (`correlator.FALSE_ALARM_TARGET`), and it gets the same treatment: count the
# comparisons, report the expected false alarms, and say so when the search has
# spent more than the target allows. A decode is never *withheld* on that
# basis -- the CRC is the truth -- but the number travels with the result so a
# user can tell "verified" from "verified, with 3% of searches like this one
# expected to produce a coincidence".

#: Expected number of false CRC passes tolerated before the search reports that
#: its own evidence is weakening. Matches ``correlator.FALSE_ALARM_TARGET``.
DECODE_FALSE_ALARM_TARGET = 0.01

#: CRC-16 has 2**16 values, so one comparison is this much probability.
CRC16_SPACE = 1 << 16


class CrcTrialCounter:
    """Counts CRC-16 comparisons across a whole hypothesis search."""

    def __init__(self) -> None:
        self.n = 0

    def add(self, count: int) -> None:
        self.n += max(0, int(count))

    @property
    def expected_false_alarms(self) -> float:
        return self.n / CRC16_SPACE


def count_candidate_ends(buf: bytes, min_payload: int, max_pad: int | None) -> int:
    """How many CRC comparisons :func:`scan_crc16_strict` would perform."""
    return len(_candidate_ends(bytes(buf), min_payload, max_pad))


# --------------------------------------------------------------------------- #
# Convolutional-code pre-screen (dual-code syndrome)
# --------------------------------------------------------------------------- #
#
# The family search has a cost problem: a Viterbi decode of a 16 kbit stream
# takes ~0.44 s, and 6 interleavers x 10 codes is 60 of them -- 26 s, far past
# any wall-clock budget. Screening with the *decoder* is therefore hopeless;
# the screen has to be something that costs microseconds.
#
# A convolutional code is a linear code, so it has a dual: for a rate-1/2 code
# with generators G1, G2 the symbol streams satisfy ``G2*y1 + G1*y2 = 0``
# identically, and for rate 1/3 the dual is two-dimensional with generators
# ``(G2, G1, 0)`` and ``(G3, 0, G1)``. Evaluating that parity combination on
# the received stream is a handful of shift-XORs -- O(n) with no trellis -- and
# its *weight* separates a real code hypothesis from a wrong one by orders of
# magnitude: a valid codeword scores exactly 0, a wrong generator scores ~0.5
# (random), and a noisy codeword scores ~0.5 * (1 - (1-2p)**taps), i.e. linear
# in the channel BER.
#
# That is what makes the family search affordable: score every (interleaver,
# code) pair cheaply, then spend the expensive Viterbi decodes on the top few.
# The screen never *accepts* anything -- the CRC still decides -- so it cannot
# manufacture a decode, only fail to reach one.


def _poly_from_mask(mask: int, k: int) -> list[int]:
    """Coefficients by delay: index ``i`` is the coefficient of ``D**i``.

    The register convention is ``reg = (bit << (k-1)) | state``, so bit
    ``k-1-i`` of a generator mask is the coefficient of ``u[n-i]`` -- the mask
    read most-significant-bit first is the polynomial read in increasing delay.
    """
    return [(mask >> (k - 1 - i)) & 1 for i in range(k)]


def _poly_mul_gf2(a: list[int], b: list[int]) -> list[int]:
    out = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                if bj:
                    out[i + j] ^= 1
    return out


def _dual_relations(code: ConvCode) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Dual relations as per-output coefficient vectors (any rate in the family).

    Each returned relation is one tuple of coefficient vectors, one per output
    stream; the shift-XOR combination it describes is identically zero on any
    codeword, so its weight measures how far the stream is from the code.
    """
    polys = [_poly_from_mask(g, code.k) for g in code.gens]
    if code.rate_den == 2:
        return ((tuple(polys[1]), tuple(polys[0])),)
    if code.rate_den == 3:
        zero = (0,) * code.k
        return (
            (tuple(polys[1]), tuple(polys[0]), zero),
            (tuple(polys[2]), zero, tuple(polys[0])),
        )
    raise ValueError(f"no dual relation implemented for r=1/{code.rate_den}")


def conv_syndrome_weight(bits: np.ndarray, code: ConvCode | str) -> dict:
    """Normalised weight of the dual-code syndrome of ``bits``.

    Returns ``{"weight", "n_positions", "normalised", "n_relations"}`` where
    ``normalised`` is the syndrome weight per tested position per relation.
    ``0.0`` is a valid codeword, ``~0.5`` is a stream unrelated to the code.

    This is a *screen*, not a decoder and not a detection: it narrows which
    (interleaver, code) pairs are worth a Viterbi decode. A low score is not
    evidence of a decode -- only the CRC can be that.
    """
    conv = CONV_CODE_BY_NAME[code] if isinstance(code, str) else code
    arr = np.asarray(bits, dtype=np.uint8).ravel()
    rate_den = conv.rate_den
    n_groups = arr.size // rate_den
    if n_groups < 4:
        return {"weight": 0, "n_positions": 0, "normalised": 1.0, "n_relations": 0}

    outputs = arr[: n_groups * rate_den].reshape(n_groups, rate_den)
    relations = _dual_relations(conv)
    total_weight = 0
    tested = 0
    for relation in relations:
        acc = np.zeros(n_groups, dtype=np.uint8)
        for column, coeffs in enumerate(relation):
            taps = [i for i, c in enumerate(coeffs) if c]
            if not taps:
                continue
            stream = outputs[:, column]
            for delay in taps:
                if delay == 0:
                    acc ^= stream
                else:
                    acc[delay:] ^= stream[:-delay]
        # Positions inside the first (max degree) samples see a partially filled
        # register, so they are not part of the code and must not be scored.
        max_delay = max(len(c) for c in relation) - 1
        usable = acc[max_delay:] if acc.size > max_delay else acc[:0]
        total_weight += int(np.sum(usable))
        tested += int(usable.size)
    normalised = (total_weight / tested) if tested else 1.0
    return {
        "weight": int(total_weight),
        "n_positions": int(tested),
        "normalised": float(normalised),
        "n_relations": len(relations),
    }


#: How many (interleaver, code) pairs the screen hands to the Viterbi stage.
#: The screen ranks every pair; only this many actually cost a decode. Four is
#: enough to cover the true hypothesis plus the near-misses its interleaver
#: variants produce, and keeps the stage inside ~2 s at 16 kbit.
CONV_SCREEN_TOP_K = 4

#: A screen score at or below this counts as "the stream is consistent with
#: this code". Reported so the user can see whether the family search had a
#: candidate at all, rather than a bare "nothing found".
CONV_SCREEN_PLAUSIBLE = 0.10


# --------------------------------------------------------------------------- #
# Hypothesis-based verification (real decoders + CRC evidence)
# --------------------------------------------------------------------------- #

VALIDATED_CONFIDENCE = 0.99
# RS(255,223) is the named requirement; 8, 16, and 64 are the requested
# shortened parity variants.  Keep them in the inexpensive first-stage list so
# a valid shortened codeword is not hidden behind the larger family sweep.
RS_NSYM_HYPOTHESES = (64, 32, 16, 8)
#: A single RS codeword over GF(256) cannot exceed 255 symbols. Streams longer
#: than this can never be one codeword, so the RS hypotheses are skipped — a
#: large speedup on long captures, and a principled rejection rather than a
#: heuristic one.
RS_MAX_CODEWORD = 255


def _attempt(
    name: str,
    params: dict,
    decode: Callable[[], dict],
) -> dict:
    """Run one hypothesis; never raise — a failed hypothesis is just a miss."""
    record = {
        "candidate": name,
        "params": dict(params),
        "crc_pass": False,
        "confidence": 0.0,
        "payload_bytes": 0,
        "payload_hex": "",
        "errors_corrected": 0,
        "note": "",
    }
    try:
        result = decode()
    except Exception as exc:  # a bad hypothesis must never break the pipeline
        record["note"] = f"{type(exc).__name__}: {exc}"
        return record
    if result.get("crc_pass"):
        payload = bytes(result.get("payload", b""))
        record.update(
            crc_pass=True,
            confidence=VALIDATED_CONFIDENCE,
            payload_bytes=len(payload),
            payload_hex=payload.hex(),
            errors_corrected=int(result.get("errors_corrected", 0)),
        )
    else:
        record["note"] = str(result.get("error", "no CRC match"))
    return record


def decode_hypotheses(
    bits: np.ndarray,
    *,
    start_offset: int = 0,
    deinterleavers: dict[str, Interleaver] | None = None,
    max_bits: int = DECODE_MAX_BITS,
    time_budget_s: float = DECODE_TIME_BUDGET_S,
    frame_bits: int | None = None,
    llr: np.ndarray | None = None,
    families: bool = True,
) -> dict:
    """Try real FEC decoders on a hypothesis basis, validated by CRC-16.

    Nothing is claimed unless an independent CRC check passes, which keeps the
    honest line from ``info.md`` §30 while still *performing* requirement (iv).

    Hypotheses are ordered cheapest-first and the search stops at the first
    CRC-valid decode, so a genuine coded capture resolves in a few attempts
    while an uncoded one is bounded by ``time_budget_s``.

    Args:
        bits: Demodulated hard bits.
        start_offset: Skip this many leading bits (e.g. the sync word already
            consumed by header correlation) before attempting decodes.
        deinterleavers: ``{name: callable}`` hypotheses applied before the
            inner decoder. Defaults to no de-interleaving.
        max_bits: Cap on how many bits are examined (keeps 1M-sample captures
            inside the §NFR-03 time budget).
        time_budget_s: Wall-clock ceiling for the whole search.
        frame_bits: Length of the coded region after the sync word. Supplying it
            truncates the stream to exactly one frame, which is what a receiver
            needs when the capture continues past the end of the burst. Left as
            ``None`` the search assumes the capture ends at the frame boundary
            (a single-burst capture).
        llr: Optional per-bit soft information (log-likelihood ratios, positive
            = "more likely 0") aligned with ``bits``. When supplied the
            convolutional hypotheses decode softly, which is worth roughly 2 dB
            -- the difference between a 10 dB capture decoding and not. Hard
            bits still drive every other hypothesis, and the CRC still decides.
        families: Sweep the whole code family (K=5..9 at r=1/2 and r=1/3, four
            LDPC matrices, the shortened RS lengths) rather than only the
            project's own transmit scheme. The convolutional part of the family
            is pre-screened with :func:`conv_syndrome_weight` so only
            :data:`CONV_SCREEN_TOP_K` pairs ever pay for a trellis decode;
            ``screen`` in the result lists every pair's score so the narrowing
            is inspectable. Disable it to reproduce the original narrower search.

    Returns:
        ``{"attempts": [...], "validated": bool, "best": dict|None,
        "confidence": float, "budget_exhausted": bool, "screen": [...],
        "crc_trials": int, "expected_false_alarms": float}`` where ``best`` is
        the first validated attempt. ``expected_false_alarms`` is the number of
        coincidental CRC-16 passes the search *would* be expected to produce by
        chance given how many comparisons it made -- reported so "verified"
        can be read together with how much searching produced it.
    """
    try:
        max_bits = max(0, int(max_bits))
    except (TypeError, ValueError):
        max_bits = DECODE_MAX_BITS
    try:
        time_budget_s = float(time_budget_s)
    except (TypeError, ValueError):
        time_budget_s = DECODE_TIME_BUDGET_S
    if not np.isfinite(time_budget_s) or time_budget_s < 0.0:
        time_budget_s = DECODE_TIME_BUDGET_S

    stream = np.asarray(bits, dtype=np.uint8).ravel()
    if start_offset > 0:
        stream = stream[min(int(start_offset), stream.size) :]
    if frame_bits is not None and 0 < int(frame_bits) < stream.size:
        stream = stream[: int(frame_bits)]
    if stream.size > max_bits:
        stream = stream[:max_bits]

    attempts: list[dict] = []
    if stream.size < 32:
        return {
            "attempts": attempts,
            "validated": False,
            "best": None,
            "confidence": 0.0,
            "crc_trials": 0,
            "expected_false_alarms": 0.0,
            "false_alarm_target": DECODE_FALSE_ALARM_TARGET,
            "false_alarm_risk_high": False,
            "screen": [],
            "screen_top_k": CONV_SCREEN_TOP_K,
            "families_swept": bool(families),
            "max_bits": max_bits,
            "time_budget_s": time_budget_s,
            "note": "not enough bits for FEC hypothesis testing",
        }

    interleaver_map: dict[str, Interleaver | None] = {"none": None}
    if deinterleavers:
        interleaver_map.update(deinterleavers)

    # De-interleave once per candidate and reuse across every FEC hypothesis
    # (Viterbi over a 64-state trellis is the expensive part of this search).
    sources: dict[str, np.ndarray] = {}
    for name, deint in interleaver_map.items():
        try:
            sources[name] = (
                stream
                if deint is None
                else np.asarray(deint(stream), dtype=np.uint8).ravel()
            )
        except Exception:  # a bad hypothesis is a miss, never a crash
            sources[name] = np.array([], dtype=np.uint8)

    # Soft information travels through the same permutation as the hard bits,
    # so a de-interleaver can be applied to it too. If a scheme cannot accept a
    # float array the soft path simply abstains for that interleaver and the
    # hypothesis falls back to hard bits -- losing ~2 dB, never correctness.
    soft_sources: dict[str, np.ndarray | None] = {}
    for name, deint in interleaver_map.items():
        if llr is None or name not in sources or sources[name].size == 0:
            soft_sources[name] = None
            continue
        try:
            values = np.asarray(llr, dtype=np.float64).ravel()
            if start_offset > 0:
                values = values[min(int(start_offset), values.size) :]
            if frame_bits is not None and 0 < int(frame_bits) < values.size:
                values = values[: int(frame_bits)]
            values = values[: sources[name].size]
            if values.size < sources[name].size:
                values = np.concatenate(
                    [values, np.zeros(sources[name].size - values.size)]
                )
            soft_sources[name] = (
                None
                if deint is None
                else np.asarray(deint(values), dtype=np.float64).ravel()
            )
        except Exception:
            soft_sources[name] = None

    # Every CRC-16 comparison is a 1/65536 chance of a random pass, so the
    # search's false-alarm risk grows with the number of hypotheses it runs.
    # Count them and report the expectation with the result -- same treatment
    # the sync-word search gets in correlator.FALSE_ALARM_TARGET.
    crc_trials = CrcTrialCounter()

    def _scan(
        buf: bytes, *, free: bool = False, max_pad: int = 32, min_payload: int = 1
    ) -> tuple[bool, bytes]:
        """CRC-16 scan that also accounts for the risk it is taking on."""
        crc_trials.add(
            count_candidate_ends(buf, min_payload, None if free else max_pad)
        )
        if free:
            return scan_crc16(buf, min_payload=min_payload)
        return scan_crc16_strict(buf, min_payload=min_payload, max_pad=max_pad)

    # Hypothesis list ordered cheapest-first so early exit pays off: RS-only
    # needs no trellis search, Viterbi-based schemes come last.
    hypotheses: list[tuple[str, dict, Callable[[], dict]]] = []

    def _raw_attempt(src: np.ndarray) -> Callable[[], dict]:
        """Uncoded frame: the bits are already the CRC-16-framed message."""

        def run() -> dict:
            # No independent structural check exists here, so only accept a
            # framing anchored to the end of the (padding-stripped) buffer.
            ok, payload = _scan(bits_to_bytes(src))
            return {"crc_pass": ok, "payload": payload, "errors_corrected": 0}

        return run

    def _rs_attempt(src: np.ndarray, nsym: int) -> Callable[[], dict]:
        def run() -> dict:
            if src.size % 8 != 0:
                return {"crc_pass": False, "payload": b"", "error": "not byte-aligned"}
            if src.size // 8 > RS_MAX_CODEWORD:
                return {
                    "crc_pass": False,
                    "payload": b"",
                    "error": f"longer than RS_MAX_CODEWORD ({RS_MAX_CODEWORD}) bytes",
                }
            word = bits_to_bytes(src)
            try:
                corrected, n_err = rs_decode(word, nsym)
            except (ReedSolomonError, ValueError) as exc:
                return {"crc_pass": False, "payload": b"", "error": str(exc)}
            if len(corrected) <= nsym:
                return {"crc_pass": False, "payload": b"", "error": "short codeword"}
            # CRC-16 lives inside the RS message, before the parity block;
            # padding may follow it, so scan rather than assume a position. The
            # RS syndrome had to come out valid for us to be here at all, which
            # is the independent structural check that lets this scan be free
            # rather than windowed.
            ok, payload = _scan(corrected[: len(corrected) - nsym], free=True)
            return {"crc_pass": ok, "payload": payload, "errors_corrected": n_err}

        return run

    def _conv_attempt(src: np.ndarray) -> Callable[[], dict]:
        def run() -> dict:
            info = viterbi_decode(src)
            # Re-encoding the decoded path and diffing against the received
            # symbols counts exactly the hard bits Viterbi had to flip, which
            # is the observable evidence that error correction happened.
            reference = conv_encode(info, terminate=False)
            n_err = int(np.sum(src[: reference.size] != reference))
            if info.size > CONV_TAIL:
                info = info[: info.size - CONV_TAIL]
            # Viterbi leaves no structural signature of its own, so require the
            # CRC framing to be anchored to the end of the info bits.
            ok, payload = _scan(bits_to_bytes(info))
            return {"crc_pass": ok, "payload": payload, "errors_corrected": n_err}

        return run

    def _conv_family_attempt(
        src: np.ndarray, code: ConvCode, soft: np.ndarray | None
    ) -> Callable[[], dict]:
        def run() -> dict:
            info = viterbi_decode_soft(src, code, llr=soft)
            if info.size <= code.tail:
                return {"crc_pass": False, "payload": b"", "error": "too short"}
            quality = viterbi_metric(src, info, code)
            info = info[: info.size - code.tail]
            ok, payload = _scan(bits_to_bytes(info))
            return {
                "crc_pass": ok,
                "payload": payload,
                "errors_corrected": int(quality["n_bit_errors"]),
                "code_bit_error_rate": float(quality["bit_error_rate"]),
            }

        return run

    def _concat_attempt(src: np.ndarray, nsym: int) -> Callable[[], dict]:
        def run() -> dict:
            # Bound the RS word the Viterbi stage will produce: if it cannot fit
            # in one RS codeword the hypothesis is impossible, so reject it
            # before paying for the trellis search.
            info_bits = src.size // 2 - CONV_TAIL
            if info_bits <= 0 or (info_bits + 7) // 8 > RS_MAX_CODEWORD:
                return {
                    "crc_pass": False,
                    "payload": b"",
                    "error": f"inner word exceeds RS_MAX_CODEWORD ({RS_MAX_CODEWORD}) bytes",
                }
            res = concatenated_decode(src, nsym=nsym)
            return {
                "crc_pass": res["crc_pass"],
                "payload": res["payload"],
                "errors_corrected": res["errors_corrected"],
                "error": res.get("error", ""),
            }

        return run

    def _concat_family_attempt(
        src: np.ndarray, nsym: int, code: ConvCode
    ) -> Callable[[], dict]:
        def run() -> dict:
            info_bits = src.size // code.rate_den - code.tail
            if info_bits <= 0 or (info_bits + 7) // 8 > RS_MAX_CODEWORD:
                return {
                    "crc_pass": False,
                    "payload": b"",
                    "error": f"inner word exceeds RS_MAX_CODEWORD ({RS_MAX_CODEWORD}) bytes",
                }
            inner = viterbi_decode_soft(src, code)
            if inner.size <= code.tail:
                return {"crc_pass": False, "payload": b"", "error": "too short"}
            inner = inner[: inner.size - code.tail]
            try:
                outer, n_err = rs_decode(bits_to_bytes(inner), nsym)
            except (ReedSolomonError, ValueError) as exc:
                return {"crc_pass": False, "payload": b"", "error": str(exc)}
            if len(outer) <= nsym:
                return {"crc_pass": False, "payload": b"", "error": "short codeword"}
            ok, payload = _scan(outer[: len(outer) - nsym], free=True)
            return {"crc_pass": ok, "payload": payload, "errors_corrected": n_err}

        return run

    def _ldpc_attempt(src: np.ndarray) -> Callable[[], dict] | None:
        n_blocks = src.size // LDPC_N
        if n_blocks < 1:
            return None

        def run() -> dict:
            blocks = src[: n_blocks * LDPC_N].reshape(n_blocks, LDPC_N)
            info = []
            for blk in blocks:
                dec = ldpc_decode(blk)
                if not ldpc_syndrome_ok(dec):
                    return {
                        "crc_pass": False,
                        "payload": b"",
                        "error": "LDPC parity check failed",
                    }
                info.append(dec[:LDPC_K])
            packed = bits_to_bytes(np.concatenate(info))
            # The block stream is zero-padded, so the payload length is
            # unknown: scan for the CRC-16 framing rather than assuming it.
            ok, payload = _scan(packed, free=True)
            return {"crc_pass": ok, "payload": payload, "errors_corrected": 0}

        return run

    def _ldpc_family_attempt(
        family: dict, src: np.ndarray
    ) -> Callable[[], dict] | None:
        k = int(family["k"])
        n = k + int(family["m"])
        n_blocks = src.size // n
        if n_blocks < 1:
            return None

        def run() -> dict:
            blocks = src[: n_blocks * n].reshape(n_blocks, n)
            info = []
            for blk in blocks:
                dec, syndrome_ok = ldpc_decode_family(blk, family)
                if not syndrome_ok:
                    return {
                        "crc_pass": False,
                        "payload": b"",
                        "error": f"{family['name']} parity check failed",
                    }
                info.append(dec[:k])
            packed = bits_to_bytes(np.concatenate(info))
            ok, payload = _scan(packed, free=True)
            return {"crc_pass": ok, "payload": payload, "errors_corrected": 0}

        return run

    # ---- Screen the convolutional family before paying for any trellis ----
    # See the note above conv_syndrome_weight: a Viterbi decode of a 16 kbit
    # stream is ~0.44 s, so trying all 6 interleavers x 10 codes is 26 s. The
    # dual-code syndrome costs microseconds and ranks the pairs, so only the
    # top CONV_SCREEN_TOP_K are ever decoded.
    screen: list[dict] = []
    if families:
        for name, src in sources.items():
            if src.size == 0:
                continue
            for code in CONV_CODES:
                score = conv_syndrome_weight(src, code)
                screen.append(
                    {
                        "interleaver": name,
                        "code": code.name,
                        "rate_den": code.rate_den,
                        "syndrome_normalised": round(score["normalised"], 6),
                        "n_positions": score["n_positions"],
                        "plausible": score["normalised"] <= CONV_SCREEN_PLAUSIBLE,
                    }
                )
        screen.sort(key=lambda row: row["syndrome_normalised"])

    for name, src in sources.items():
        if src.size == 0:
            continue
        # Cheapest hypothesis first: no FEC at all, just CRC-16 framing.  This
        # preserves the fast path for genuinely uncoded frames; the acceptance
        # tests protect a protected payload bit so a raw CRC cannot mask the
        # structural RS hypothesis.
        hypotheses.append(
            (
                "None (CRC-16 only)",
                {"interleaver": name, "nsym": None},
                _raw_attempt(src),
            )
        )
        for nsym in RS_NSYM_HYPOTHESES:
            hypotheses.append(
                (
                    f"RS(255,{255 - nsym})",
                    {"interleaver": name, "nsym": nsym},
                    _rs_attempt(src, nsym),
                )
            )
        ldpc_fn = _ldpc_attempt(src)
        if ldpc_fn is not None:
            hypotheses.append(
                (
                    "LDPC",
                    {"interleaver": name, "n_blocks": src.size // LDPC_N},
                    ldpc_fn,
                )
            )
        # Concatenated is tried *before* the inner code alone: a Viterbi decode
        # of a concatenated stream yields the RS codeword, whose embedded CRC
        # would otherwise let the bare convolutional hypothesis claim a payload
        # that the outer RS stage actually protected (a mislabel, not a miss).
        for nsym in RS_NSYM_HYPOTHESES:
            hypotheses.append(
                (
                    f"Concatenated RS(255,{255 - nsym}) + Conv K=7",
                    {"interleaver": name, "nsym": nsym},
                    _concat_attempt(src, nsym),
                )
            )
        hypotheses.append(
            (
                "Convolutional r=1/2 K=7",
                {"interleaver": name, "nsym": None},
                _conv_attempt(src),
            )
        )

    # ---- Family stage: the rest of the code space, screened ----
    if families:
        for row in screen[:CONV_SCREEN_TOP_K]:
            code = CONV_CODE_BY_NAME[str(row["code"])]
            src = sources.get(str(row["interleaver"]))
            if src is None or src.size == 0:
                continue
            hypotheses.append(
                (
                    f"Convolutional r=1/{code.rate_den} {code.name}",
                    {
                        "interleaver": row["interleaver"],
                        "nsym": None,
                        "code": code.name,
                        "syndrome": row["syndrome_normalised"],
                    },
                    _conv_family_attempt(
                        src, code, soft_sources.get(str(row["interleaver"]))
                    ),
                )
            )
            # The concatenated variant of the same inner code, at the two named
            # RS parity lengths. A concatenated capture whose inner code is
            # right but whose outer length is not still needs the outer sweep,
            # so both are offered.
            for nsym in RS_NSYM_HYPOTHESES:
                hypotheses.append(
                    (
                        f"Concatenated RS(255,{255 - nsym}) + {code.name}",
                        {
                            "interleaver": row["interleaver"],
                            "nsym": nsym,
                            "code": code.name,
                        },
                        _concat_family_attempt(src, nsym, code),
                    )
                )

        for family in LDPC_FAMILIES:
            if str(family["name"]).endswith("n=84"):
                continue  # the (3,4) matrix is the core hypothesis already
            for name, src in sources.items():
                if src.size == 0:
                    continue
                fn = _ldpc_family_attempt(family, src)
                if fn is None:
                    continue
                hypotheses.append(
                    (
                        str(family["name"]),
                        {
                            "interleaver": name,
                            "k": int(family["k"]),
                            "n": int(family["k"]) + int(family["m"]),
                        },
                        fn,
                    )
                )

        # The shortened RS lengths a real link actually picks to fit its
        # payload. RS has no trellis, so this part of the family is cheap.
        for nsym in RS_NSYM_FAMILY:
            if nsym in RS_NSYM_HYPOTHESES:
                continue
            for name, src in sources.items():
                if src.size == 0:
                    continue
                hypotheses.append(
                    (
                        f"RS(255,{255 - nsym})",
                        {"interleaver": name, "nsym": nsym},
                        _rs_attempt(src, nsym),
                    )
                )

    best: dict | None = None
    started = time.monotonic()
    for name, params, fn in hypotheses:
        if time.monotonic() - started >= time_budget_s:
            attempts.append(
                {
                    "candidate": "(budget)",
                    "params": {},
                    "crc_pass": False,
                    "confidence": 0.0,
                    "payload_bytes": 0,
                    "payload_hex": "",
                    "errors_corrected": 0,
                    "note": f"stopped after {time_budget_s:.1f}s time budget",
                }
            )
            break
        record = _attempt(name, params, fn)
        attempts.append(record)
        if record["crc_pass"]:
            best = record
            break  # verified — no need to keep searching

    return {
        "attempts": attempts,
        "validated": best is not None,
        "best": best,
        "confidence": float(best["confidence"]) if best else 0.0,
        "n_attempts": len(attempts),
        "n_hypotheses": len(hypotheses),
        "budget_exhausted": bool(best is None and len(attempts) < len(hypotheses)),
        # How much searching produced the verdict, and what that much searching
        # is worth: `crc_trials` comparisons each carry a 1/65536 chance of a
        # coincidental pass, so `expected_false_alarms` is the number of false
        # positives this search would be expected to manufacture by chance
        # alone. A decode is never withheld because of it -- the CRC is the
        # truth -- but the number travels with the result.
        "crc_trials": int(crc_trials.n),
        "expected_false_alarms": float(crc_trials.expected_false_alarms),
        "false_alarm_target": DECODE_FALSE_ALARM_TARGET,
        "false_alarm_risk_high": bool(
            crc_trials.expected_false_alarms > DECODE_FALSE_ALARM_TARGET
        ),
        # The convolutional family's pre-screen, cheapest first: every
        # (interleaver, code) pair with its dual-code syndrome weight. A valid
        # codeword scores 0, an unrelated stream ~0.5, so this is what narrowed
        # 60 candidate pairs down to the CONV_SCREEN_TOP_K that were decoded.
        "screen": screen,
        "screen_top_k": CONV_SCREEN_TOP_K,
        "families_swept": bool(families),
        "max_bits": max_bits,
        "time_budget_s": time_budget_s,
    }
