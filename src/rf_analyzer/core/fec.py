"""Forward-error-correction codecs + hypothesis validation.

SIH26147 bullet (iv) asks for "FEC (short-constrained convolution codes with
Viterbi decoding, RS block codes, Concatenated codes, LDPC)". Full spec:
``info.md`` §12.5, with the honesty rules in §30.

Two clearly separated responsibilities live in this module:

1. **Real codecs** — from-scratch, pure-numpy implementations:
   * :func:`crc16_ccitt` / :func:`crc32_ieee` integrity checks;
   * a K=7, r=1/2 convolutional encoder (:func:`conv_encode`) and a
     hard-decision Viterbi decoder (:func:`viterbi_decode`) using the
     CCSDS / 802.11 generator pair ``G1=0o171, G2=0o133``;
   * GF(256) arithmetic plus a systematic Reed-Solomon codec
     (:func:`rs_encode` / :func:`rs_decode`) supporting RS(255,223) and
     shortened variants;
   * a small (3,4)-regular LDPC code with a normalized min-sum
     belief-propagation decoder (:func:`ldpc_encode` / :func:`ldpc_decode`);
   * a concatenated RS-outer / convolutional-inner scheme
     (:func:`concatenated_encode` / :func:`concatenated_decode`).

2. **Blind identification** — :func:`score_fec_candidates`, which is
   deliberately capped at 0.5 confidence and never claims detection.
   Fully blind FEC identification is research-grade (``info.md`` §30) and
   stays out of scope.

:func:`decode_hypotheses` bridges the two: it *tries* the real decoders on a
hypothesis basis and reports high confidence only when an independent
integrity check (CRC) actually passes. We never guess — we verify.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

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
    """
    n = len(buf)
    lo = min_payload + 2
    if window is not None:
        lo = max(lo, n - window)
    ends = list(range(lo, n + 1))

    # Safety net: the padding-stripped end is the structurally cleanest anchor,
    # so keep it as a candidate even when it falls outside a narrow window.
    zero_end = n
    while zero_end > 0 and buf[zero_end - 1] == 0:
        zero_end -= 1
    if zero_end >= min_payload + 2 and zero_end not in ends:
        ends.append(zero_end)
    return ends


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
    ``max_pad / 65536`` while still tolerating padding whose last byte happens to
    be non-zero.
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
# Convolutional code: K=7, r=1/2, hard-decision Viterbi
# --------------------------------------------------------------------------- #

CONV_K = 7
CONV_G1 = 0o171  # 1111001 -> 121
CONV_G2 = 0o133  # 1011011 -> 91
CONV_STATES = 1 << (CONV_K - 1)  # 64 trellis states
CONV_TAIL = CONV_K - 1  # 6 zero tail bits flush the encoder

# 7-bit parity lookup: register values are always < 128.
_PARITY7 = np.array([bin(i).count("1") & 1 for i in range(1 << CONV_K)], dtype=np.uint8)
# Hamming distance between 2-bit symbol pairs (values 0..3).
_HAMMING2 = np.array(
    [[bin(a ^ b).count("1") for b in range(4)] for a in range(4)], dtype=np.int16
)


def _conv_outputs(reg: np.ndarray) -> np.ndarray:
    """Map 7-bit shift registers to 2-bit output symbols (0..3)."""
    return (_PARITY7[reg & CONV_G1] << 1) | _PARITY7[reg & CONV_G2]


def conv_encode(bits: np.ndarray, terminate: bool = True) -> np.ndarray:
    """Rate-1/2, K=7 convolutional encoder (hard bits in, 2 bits per input).

    With ``terminate=True`` (default) ``CONV_TAIL`` zero bits are appended so
    the trellis ends in state 0, which is what the Viterbi decoder expects.
    """
    arr = np.asarray(bits, dtype=np.uint8).ravel()
    if terminate:
        arr = np.concatenate([arr, np.zeros(CONV_TAIL, dtype=np.uint8)])
    if arr.size == 0:
        return np.array([], dtype=np.uint8)

    # state = last K-1 input bits; reg = (new_bit << (K-1)) | state
    state = 0
    out = np.empty(arr.size * 2, dtype=np.uint8)
    for i, bit in enumerate(arr):
        reg = (int(bit) << (CONV_K - 1)) | state
        sym = int(_conv_outputs(np.array([reg], dtype=np.int32))[0])
        out[2 * i] = (sym >> 1) & 1
        out[2 * i + 1] = sym & 1
        state = reg >> 1
    return out


def viterbi_decode(rx_bits: np.ndarray) -> np.ndarray:
    """Hard-decision Viterbi decoder for the K=7, r=1/2 code above.

    Fully vectorized across the 64 trellis states. Returns the decoded
    information bits (including any tail bits the encoder appended).

    Raises:
        ValueError: when the received stream length is odd.
    """
    rx = np.asarray(rx_bits, dtype=np.uint8).ravel()
    if rx.size == 0:
        return np.array([], dtype=np.uint8)
    if rx.size % 2 != 0:
        raise ValueError("Viterbi input must have an even number of bits (r=1/2).")

    n_steps = rx.size // 2
    # Symbol values 0..3 for each received pair.
    rx_sym = (rx[0::2].astype(np.int16) << 1) | rx[1::2].astype(np.int16)

    # Predecessor tables: a transition into state n comes from
    # p = ((n & 0x1F) << 1) | lsb, carrying input bit (n >> 5).
    states = np.arange(CONV_STATES, dtype=np.int32)
    in_bit = states >> (CONV_K - 2)  # 0 for n < 32, 1 for n >= 32
    pred_a = (states & (CONV_STATES // 2 - 1)) << 1
    pred_b = pred_a | 1
    out_a = _conv_outputs((in_bit << (CONV_K - 1)) | pred_a)
    out_b = _conv_outputs((in_bit << (CONV_K - 1)) | pred_b)

    INF = np.int32(1 << 20)
    metric = np.full(CONV_STATES, INF, dtype=np.int32)
    metric[0] = 0
    trace = np.empty((n_steps, CONV_STATES), dtype=np.int16)

    for t in range(n_steps):
        sym = int(rx_sym[t])
        branch_a = _HAMMING2[sym, out_a].astype(np.int32)
        branch_b = _HAMMING2[sym, out_b].astype(np.int32)
        cand_a = metric[pred_a] + branch_a
        cand_b = metric[pred_b] + branch_b
        take_a = cand_a <= cand_b
        metric = np.where(take_a, cand_a, cand_b).astype(np.int32)
        trace[t] = np.where(take_a, pred_a, pred_b).astype(np.int16)

    # Terminated code -> end in state 0; otherwise pick the best final state.
    state = 0
    decoded = np.empty(n_steps, dtype=np.uint8)
    for t in range(n_steps - 1, -1, -1):
        decoded[t] = (state >> (CONV_K - 2)) & 1
        state = int(trace[t, state])
    return decoded


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
    """Build a systematic, ``(dv, dc)``-regular parity-check matrix ``H = [A | I]``.

    ``A`` is drawn by dealing from a fixed-size row pool, which guarantees
    exactly ``dc`` ones per row and ``dv`` per column — an irregular or
    duplicate-column matrix wrecks belief propagation (columns with identical
    support are indistinguishable, and weight-1 columns carry almost no
    information). Seeds are retried until the column signatures are unique.

    Systematic form means parity bits are just ``p = A @ u (mod 2)`` — no
    Gaussian elimination is needed to encode.
    """
    if k * dv != m * dc:
        raise ValueError(
            f"k*dv ({k * dv}) must equal m*dc ({m * dc}) for a regular code"
        )

    for seed_offset in range(500):
        rng = np.random.default_rng(LDPC_SEED + seed_offset)
        pool = np.repeat(np.arange(m), dc)
        rng.shuffle(pool)

        a = np.zeros((m, k), dtype=np.uint8)
        ok = True
        for j in range(k):
            rows = pool[j * dv : (j + 1) * dv]
            if np.unique(rows).size != dv:  # duplicate row in one column
                ok = False
                break
            a[rows, j] = 1
        if not ok:
            continue

        signatures = {tuple(np.flatnonzero(a[:, j])) for j in range(k)}
        if len(signatures) == k:  # every column distinguishable
            return np.hstack([a, np.eye(m, dtype=np.uint8)]).astype(np.uint8)

    raise RuntimeError("could not construct a duplicate-free regular LDPC matrix")


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
    """Encode ``payload`` as CRC-16 -> RS outer -> convolutional inner.

    The CRC-16 is computed over the raw payload and placed *inside* the RS
    protection, so a successful decode gives a verifiable integrity check.
    """
    framed = crc16_append(bytes(payload))
    # rs_encode returns the parity symbols only -> build the systematic word.
    outer = framed + rs_encode(framed, nsym)
    inner = conv_encode(bytes_to_bits(outer))
    if interleave is not None:
        inner = interleave(inner)
    return np.asarray(inner, dtype=np.uint8)


def concatenated_decode(
    bits: np.ndarray,
    nsym: int = 32,
    deinterleave: Interleaver | None = None,
) -> dict:
    """Reverse of :func:`concatenated_encode`.

    Returns a dict with ``crc_pass``, ``payload`` (bytes, empty on failure),
    ``errors_corrected`` and ``rs_ok``.
    """
    stream = np.asarray(bits, dtype=np.uint8).ravel()
    if deinterleave is not None:
        stream = np.asarray(deinterleave(stream), dtype=np.uint8)
    try:
        inner = viterbi_decode(stream)
    except ValueError as exc:
        return {
            "crc_pass": False,
            "payload": b"",
            "errors_corrected": 0,
            "rs_ok": False,
            "error": str(exc),
        }
    # Drop the K-1 tail bits the encoder appended.
    if inner.size > CONV_TAIL:
        inner = inner[: inner.size - CONV_TAIL]
    try:
        outer = bits_to_bytes(inner)
        corrected, n_err = rs_decode(outer, nsym)
    except (ReedSolomonError, ValueError) as exc:
        return {
            "crc_pass": False,
            "payload": b"",
            "errors_corrected": 0,
            "rs_ok": False,
            "error": str(exc),
        }
    # The RS parity symbols sit at the end; the CRC-16 is inside the RS
    # message, immediately after the payload. Block interleavers zero-pad to
    # a block boundary, so the payload length is unknown here -> scan the RS
    # message region for the CRC framing instead of assuming a position.
    if len(corrected) <= nsym:
        return {
            "crc_pass": False,
            "payload": b"",
            "errors_corrected": int(n_err),
            "rs_ok": True,
            "error": "codeword shorter than the parity block",
        }
    ok, payload = scan_crc16(corrected[: len(corrected) - nsym])
    return {
        "crc_pass": bool(ok),
        "payload": payload if ok else b"",
        "errors_corrected": int(n_err),
        "rs_ok": True,
    }


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
# Hypothesis-based verification (real decoders + CRC evidence)
# --------------------------------------------------------------------------- #

VALIDATED_CONFIDENCE = 0.99
# RS(255,223) is the named requirement; 16 is the common shortened variant.
RS_NSYM_HYPOTHESES = (32, 16)
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
    max_bits: int = 8192,
    time_budget_s: float = 3.0,
    frame_bits: int | None = None,
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

    Returns:
        ``{"attempts": [...], "validated": bool, "best": dict|None,
        "confidence": float, "budget_exhausted": bool}`` where ``best`` is the
        first validated attempt.
    """
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

    # Hypothesis list ordered cheapest-first so early exit pays off: RS-only
    # needs no trellis search, Viterbi-based schemes come last.
    hypotheses: list[tuple[str, dict, Callable[[], dict]]] = []

    def _raw_attempt(src: np.ndarray) -> Callable[[], dict]:
        """Uncoded frame: the bits are already the CRC-16-framed message."""

        def run() -> dict:
            # No independent structural check exists here, so only accept a
            # framing anchored to the end of the (padding-stripped) buffer.
            ok, payload = scan_crc16_strict(bits_to_bytes(src))
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
            # padding may follow it, so scan rather than assume a position.
            ok, payload = scan_crc16(corrected[: len(corrected) - nsym])
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
            ok, payload = scan_crc16_strict(bits_to_bytes(info))
            return {"crc_pass": ok, "payload": payload, "errors_corrected": n_err}

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
            ok, payload = scan_crc16(packed)
            return {"crc_pass": ok, "payload": payload, "errors_corrected": 0}

        return run

    for name, src in sources.items():
        if src.size == 0:
            continue
        # Cheapest hypothesis first: no FEC at all, just CRC-16 framing.
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

    best: dict | None = None
    started = time.monotonic()
    for name, params, fn in hypotheses:
        record = _attempt(name, params, fn)
        attempts.append(record)
        if record["crc_pass"]:
            best = record
            break  # verified — no need to keep searching
        if time.monotonic() - started > time_budget_s:
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

    return {
        "attempts": attempts,
        "validated": best is not None,
        "best": best,
        "confidence": float(best["confidence"]) if best else 0.0,
        "n_attempts": len(attempts),
        "n_hypotheses": len(hypotheses),
        "budget_exhausted": bool(best is None and len(attempts) < len(hypotheses)),
    }
