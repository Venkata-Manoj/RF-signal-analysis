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


def crc16_ccitt(data: bytes, init: int = 0xFFFF) -> int:
    """CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, no reflection/xorout)."""
    crc = init & 0xFFFF
    for byte in data:
        crc ^= (byte & 0xFF) << 8
        for _ in range(8):
            crc = (
                ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
            )
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


def scan_crc16(buf: bytes, min_payload: int = 1) -> tuple[bool, bytes]:
    """Find a valid CRC-16 framing *anywhere* in ``buf``.

    Needed when the payload length is unknown (e.g. a block-coded stream that
    was zero-padded to a fixed block size): every candidate split point is
    tested and the longest payload with a matching trailing CRC-16 wins.

    A 16-bit check has a 1/65536 false-positive rate per position, so callers
    should combine this with an independent structural check (e.g. LDPC parity)
    before trusting it.
    """
    buf = bytes(buf)
    for end in range(len(buf), min_payload + 1, -1):
        ok, payload = crc16_check(buf[:end])
        if ok:
            return True, payload
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
    """
    h = LDPC_H if h is None else h
    rx = np.asarray(bits, dtype=np.uint8).ravel()
    n = h.shape[1]
    m = h.shape[0]
    if rx.size != n:
        raise ValueError(f"LDPC expects exactly {n} bits, got {rx.size}")

    p_err = float(min(max(p_err, 1e-6), 0.5 - 1e-6))
    llr_mag = float(np.log((1.0 - p_err) / p_err))
    llr0 = np.where(rx == 0, llr_mag, -llr_mag).astype(np.float64)

    checks = [np.flatnonzero(h[c]) for c in range(m)]
    var_edges: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for c, vs in enumerate(checks):
        for i, v in enumerate(vs):
            var_edges[v].append((c, i))

    c2v = [np.zeros(vs.size, dtype=np.float64) for vs in checks]
    v2c = [llr0[vs].copy() for vs in checks]
    total = llr0.copy()

    for _ in range(max_iter):
        # ---- check node update (min-sum with self-exclusion) ----
        for c, _vs in enumerate(checks):
            msg = v2c[c]
            if msg.size == 1:
                c2v[c][0] = alpha * msg[0]
                continue
            abs_msg = np.abs(msg)
            signs = np.where(msg >= 0, 1.0, -1.0)
            for i in range(msg.size):
                others_mag = np.delete(abs_msg, i)
                others_sign = np.delete(signs, i)
                c2v[c][i] = (
                    alpha * float(np.prod(others_sign)) * float(others_mag.min())
                )

        # ---- variable node update ----
        for v in range(n):
            edges = var_edges[v]
            if not edges:
                total[v] = llr0[v]
                continue
            s = llr0[v]
            for c, i in edges:
                s += c2v[c][i]
            total[v] = s
            for c, i in edges:
                v2c[c][i] = s - c2v[c][i]

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
    # message, immediately after the payload.
    if len(corrected) <= nsym:
        return {
            "crc_pass": False,
            "payload": b"",
            "errors_corrected": int(n_err),
            "rs_ok": True,
            "error": "codeword shorter than the parity block",
        }
    ok, payload = crc16_check(corrected[: len(corrected) - nsym])
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
RS_NSYM_HYPOTHESES = (32, 16, 8)


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
    max_bits: int = 50_000,
) -> dict:
    """Try real FEC decoders on a hypothesis basis, validated by CRC-16.

    Nothing is claimed unless an independent CRC check passes, which keeps the
    honest line from ``info.md`` §30 while still *performing* requirement (iv).

    Args:
        bits: Demodulated hard bits.
        start_offset: Skip this many leading bits (e.g. the sync word already
            consumed by header correlation) before attempting decodes.
        deinterleavers: ``{name: callable}`` hypotheses applied before the
            inner decoder. Defaults to no de-interleaving.
        max_bits: Cap on how many bits are examined (keeps 1M-sample captures
            inside the §NFR-03 time budget).

    Returns:
        ``{"attempts": [...], "validated": bool, "best": dict|None,
        "confidence": float}`` where ``best`` is the first validated attempt.
    """
    stream = np.asarray(bits, dtype=np.uint8).ravel()
    if start_offset > 0:
        stream = stream[min(int(start_offset), stream.size) :]
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

    # --- (a) convolutional only ---
    for name, deint in interleaver_map.items():
        src = stream if deint is None else np.asarray(deint(stream), dtype=np.uint8)

        def _conv(src=src):
            info = viterbi_decode(src)
            if info.size > CONV_TAIL:
                info = info[: info.size - CONV_TAIL]
            ok, payload = crc16_check(bits_to_bytes(info))
            return {"crc_pass": ok, "payload": payload, "errors_corrected": 0}

        attempts.append(
            _attempt(
                "Convolutional r=1/2 K=7",
                {"interleaver": name, "nsym": None},
                _conv,
            )
        )

    # --- (b) Reed-Solomon only (byte-aligned) ---
    for nsym in RS_NSYM_HYPOTHESES:
        for name, deint in interleaver_map.items():
            src = stream if deint is None else np.asarray(deint(stream), dtype=np.uint8)
            n_bytes = src.size // 8

            def _rs(src=src, nsym=nsym, n_bytes=n_bytes):
                if n_bytes * 8 != src.size:
                    return {
                        "crc_pass": False,
                        "payload": b"",
                        "error": "not byte-aligned",
                    }
                word = bits_to_bytes(src)
                try:
                    corrected, n_err = rs_decode(word, nsym)
                except (ReedSolomonError, ValueError) as exc:
                    return {"crc_pass": False, "payload": b"", "error": str(exc)}
                if len(corrected) <= nsym:
                    return {
                        "crc_pass": False,
                        "payload": b"",
                        "error": "short codeword",
                    }
                # CRC-16 lives inside the RS message, before the parity block.
                ok, payload = crc16_check(corrected[: len(corrected) - nsym])
                return {
                    "crc_pass": ok,
                    "payload": payload,
                    "errors_corrected": n_err,
                }

            attempts.append(
                _attempt(
                    f"RS(255,{255 - nsym})", {"interleaver": name, "nsym": nsym}, _rs
                )
            )

    # --- (c) LDPC (block-aligned) ---
    n_blocks = stream.size // LDPC_N
    if n_blocks >= 1:
        for name, deint in interleaver_map.items():
            src = stream if deint is None else np.asarray(deint(stream), dtype=np.uint8)

            def _ldpc(src=src, n_blocks=n_blocks):
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

            attempts.append(
                _attempt("LDPC", {"interleaver": name, "n_blocks": n_blocks}, _ldpc)
            )

    # --- (d) concatenated RS outer + convolutional inner ---
    for nsym in RS_NSYM_HYPOTHESES:
        for name, deint in interleaver_map.items():
            src = stream

            def _concat(src=src, nsym=nsym, deint=deint):
                res = concatenated_decode(src, nsym=nsym, deinterleave=deint)
                return {
                    "crc_pass": res["crc_pass"],
                    "payload": res["payload"],
                    "errors_corrected": res["errors_corrected"],
                    "error": res.get("error", ""),
                }

            attempts.append(
                _attempt(
                    f"Concatenated RS(255,{255 - nsym}) + Conv K=7",
                    {"interleaver": name, "nsym": nsym},
                    _concat,
                )
            )

    validated = [a for a in attempts if a["crc_pass"]]
    best = validated[0] if validated else None
    return {
        "attempts": attempts,
        "validated": bool(validated),
        "best": best,
        "confidence": float(best["confidence"]) if best else 0.0,
        "n_attempts": len(attempts),
    }
