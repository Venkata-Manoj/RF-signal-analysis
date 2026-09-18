"""Transmit-side frame construction — the mirror of the analysis pipeline.

Requirements (ii)–(v) are receive-side. To *prove* they work end to end (and to
give the pipeline something honest to decode) there has to be a transmitter
that applies the same error-control schemes in a defined order. This module is
that transmitter, and it is the single source of truth for the frame layout
shared by ``scripts/generate_test_data.py`` and the integration tests.

Frame layout, in on-air bit order::

    [ sync word ][ interleave( FEC( crc16_append(message) ) ) ]

The CRC-16 is always computed over the raw message and always travels *inside*
the FEC protection, so a successful decode is independently verifiable. That is
what lets the receiver claim a payload instead of guessing one (``info.md`` §30).

Supported FEC schemes
---------------------
==============  ==========================================================
``none``        message + CRC-16 only; no error correction
``conv``        K=7, r=1/2 convolutional code, terminated (zero tail)
``rs``          systematic Reed-Solomon with CRC-16 inside the message
``ldpc``        (3,4)-regular systematic LDPC, 48-bit information blocks
``concatenated``RS outer + convolutional inner (the classic CCSDS pair)
==============  ==========================================================

Supported interleavers: ``none``, ``block``, ``convolutional``, ``diagonal``,
``pseudo-random`` (whole-stream) and ``pseudo-random-block`` (sub-block, the
variant that survives trailing samples).

This module is deliberately receive-agnostic: it never tells the pipeline which
scheme it used. The receiver has to work that out from the bit stream alone.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rf_analyzer.config import (
    DEFAULT_SYNC_WORD,
    FSK_FREQ_HIGH_HZ,
    FSK_FREQ_LOW_HZ,
    FSK_SYMBOL_DURATION_S,
)
from rf_analyzer.core.correlator import hex_to_bits
from rf_analyzer.core.deinterleave import (
    DEFAULT_COLS,
    DEFAULT_DEPTH,
    DEFAULT_RANDOM_BLOCK,
    DEFAULT_ROWS,
    DEFAULT_SEED,
    DEFAULT_SPACING,
    interleave_block,
    interleave_convolutional,
    interleave_diagonal,
    interleave_pseudo_random,
)
from rf_analyzer.core.demod import bits_to_qam16
from rf_analyzer.core.fec import (
    LDPC_K,
    bits_to_bytes,
    bytes_to_bits,
    conv_encode,
    crc16_append,
    ldpc_encode,
    rs_encode,
)

FEC_SCHEMES = ("none", "conv", "rs", "ldpc", "concatenated")
INTERLEAVER_SCHEMES = (
    "none",
    "block",
    "convolutional",
    "diagonal",
    "pseudo-random",
    "pseudo-random-block",
)
MODULATION_SCHEMES = ("BPSK", "QPSK", "16-QAM", "2-FSK")


def _pad_bits(bits: np.ndarray, block: int) -> np.ndarray:
    """Zero-pad to a multiple of ``block`` (no-op when already aligned)."""
    arr = np.asarray(bits, dtype=np.uint8).ravel()
    pad = (-arr.size) % block
    if pad:
        arr = np.concatenate([arr, np.zeros(pad, dtype=np.uint8)])
    return arr


def encode_fec(
    message: bytes, scheme: str = "concatenated", nsym: int = 32
) -> np.ndarray:
    """Apply the CRC framing plus the requested FEC scheme.

    Returns the coded bit array (``crc16_append(message)`` when ``scheme`` is
    ``"none"``), so the caller can interleave and modulate it.
    """
    message = bytes(message)
    scheme = scheme.lower()
    if scheme not in FEC_SCHEMES:
        raise ValueError(f"unknown FEC scheme {scheme!r}; pick one of {FEC_SCHEMES}")

    framed = crc16_append(message)
    if scheme == "none":
        return bytes_to_bits(framed)
    if scheme == "conv":
        return conv_encode(bytes_to_bits(framed))
    if scheme == "rs":
        # rs_encode returns parity symbols only -> build the systematic word.
        return bytes_to_bits(framed + rs_encode(framed, nsym))
    if scheme == "concatenated":
        outer = framed + rs_encode(framed, nsym)
        return conv_encode(bytes_to_bits(outer))
    # ldpc: chunk the framed bytes into 48-bit information blocks.
    payload_bits = bytes_to_bits(framed)
    padded = _pad_bits(payload_bits, LDPC_K)
    blocks = padded.reshape(-1, LDPC_K)
    return np.concatenate([ldpc_encode(blk) for blk in blocks]).astype(np.uint8)


def apply_interleaver(
    bits: np.ndarray,
    scheme: str = "none",
    *,
    rows: int = DEFAULT_ROWS,
    cols: int = DEFAULT_COLS,
    depth: int = DEFAULT_DEPTH,
    spacing: int = DEFAULT_SPACING,
    seed: int = DEFAULT_SEED,
    random_block: int = DEFAULT_RANDOM_BLOCK,
) -> np.ndarray:
    """Apply the requested interleaver (``"none"`` is a pass-through)."""
    scheme = scheme.lower()
    if scheme not in INTERLEAVER_SCHEMES:
        raise ValueError(
            f"unknown interleaver {scheme!r}; pick one of {INTERLEAVER_SCHEMES}"
        )
    arr = np.asarray(bits, dtype=np.uint8).ravel()
    if scheme == "none":
        return arr
    if scheme == "block":
        return interleave_block(arr, rows, cols)
    if scheme == "convolutional":
        return interleave_convolutional(arr, depth, spacing)
    if scheme == "diagonal":
        return interleave_diagonal(arr, rows, cols)
    if scheme == "pseudo-random":
        return interleave_pseudo_random(arr, seed)
    return interleave_pseudo_random(arr, seed, random_block)


def modulate_2fsk(
    bits: np.ndarray,
    sample_rate: float,
    *,
    symbol_duration: float = FSK_SYMBOL_DURATION_S,
    freq_low: float = FSK_FREQ_LOW_HZ,
    freq_high: float = FSK_FREQ_HIGH_HZ,
) -> np.ndarray:
    """Continuous-phase 2-FSK modulator: one tone per bit, ``sps`` samples each.

    ``sps = int(sample_rate * symbol_duration)``. Unlike the linear
    modulations this returns ``sps * len(bits)`` samples rather than one sample
    per bit, which is why it needs ``sample_rate`` and why the receiver must
    recover the period (``dsp.estimate_fsk_symbol_period``) before it can
    demodulate at all.

    The phase convention is the one in ``info.md`` §15.1, which this module is
    the single source of truth for: a symbol's phase carries over from the
    previous symbol's last sample, so the waveform is continuous and each
    symbol contributes ``sps - 1`` sample intervals plus one repeated boundary
    sample. ``scripts/generate_test_data.py`` calls this function rather than
    keeping its own copy, so the generator and the transmit path cannot drift.

    Parameters default to the project's reference 2-FSK configuration
    (1 ms symbols, +/-5 kHz), also held in ``config``.
    """
    arr = np.asarray(bits, dtype=np.uint8).ravel()
    samples_per_symbol = int(float(sample_rate) * float(symbol_duration))
    if samples_per_symbol < 2:
        raise ValueError(
            f"symbol_duration {symbol_duration!r} at sample_rate {sample_rate!r} "
            f"gives {samples_per_symbol} sample(s) per symbol; 2-FSK needs at least 2."
        )
    if arr.size == 0:
        return np.array([], dtype=np.complex64)

    tone = np.where(arr == 1, float(freq_high), float(freq_low))
    radians_per_hz = 2.0 * np.pi / float(sample_rate)
    # Phase carried into each symbol: the previous symbol advanced by
    # (sps - 1) sample intervals, matching info.md §15.1.
    advance = tone * ((samples_per_symbol - 1) * radians_per_hz)
    carried = np.concatenate([[0.0], np.cumsum(advance)[:-1]])
    within = (
        np.tile(np.arange(samples_per_symbol), arr.size)
        * np.repeat(tone, samples_per_symbol)
        * radians_per_hz
    )
    phase = np.repeat(carried, samples_per_symbol) + within
    return np.exp(1j * phase).astype(np.complex64)


def modulate(
    bits: np.ndarray,
    mode: str = "BPSK",
    *,
    sample_rate: float | None = None,
    symbol_duration: float = FSK_SYMBOL_DURATION_S,
    freq_low: float = FSK_FREQ_LOW_HZ,
    freq_high: float = FSK_FREQ_HIGH_HZ,
) -> np.ndarray:
    """Map hard bits onto complex baseband samples.

    BPSK, QPSK and 16-QAM return one symbol per bit (or per 2/4 bits), so no
    sample rate is needed. 2-FSK returns ``samples_per_symbol`` samples per
    bit and therefore *requires* ``sample_rate``; passing ``None`` raises rather
    than inventing a rate.
    """
    mode = mode.upper().replace("QAM16", "16-QAM")
    if mode not in MODULATION_SCHEMES:
        raise ValueError(
            f"unsupported modulation {mode!r}; pick one of {MODULATION_SCHEMES}"
        )
    if mode == "2-FSK":
        if sample_rate is None:
            raise ValueError(
                "2-FSK needs sample_rate (it produces many samples per bit, "
                "unlike BPSK/QPSK/16-QAM)."
            )
        return modulate_2fsk(
            bits,
            sample_rate,
            symbol_duration=symbol_duration,
            freq_low=freq_low,
            freq_high=freq_high,
        )
    arr = np.asarray(bits, dtype=np.uint8).ravel()
    if mode == "BPSK":
        return (2.0 * arr.astype(np.float32) - 1.0).astype(np.complex64)
    if mode == "16-QAM":
        return bits_to_qam16(arr)
    # QPSK: I and Q each carry one bit, both mapped to +/-1/sqrt(2).
    even = arr[: (arr.size // 2) * 2]
    i_sym = 2.0 * even[0::2].astype(np.float32) - 1.0
    q_sym = 2.0 * even[1::2].astype(np.float32) - 1.0
    return ((i_sym + 1j * q_sym) / np.sqrt(2.0)).astype(np.complex64)


def build_frame(
    message: bytes,
    *,
    sync_word: str = DEFAULT_SYNC_WORD,
    fec: str = "concatenated",
    interleaver: str = "none",
    nsym: int = 32,
    rows: int = DEFAULT_ROWS,
    cols: int = DEFAULT_COLS,
    depth: int = DEFAULT_DEPTH,
    spacing: int = DEFAULT_SPACING,
    seed: int = DEFAULT_SEED,
    random_block: int = DEFAULT_RANDOM_BLOCK,
) -> dict[str, Any]:
    """Build a complete frame: ``[sync][interleave(FEC(message))]``.

    Returns:
        ``{"bits", "sync_bits", "coded_bits", "frame_bits", "layout"}`` where
        ``frame_bits`` is the length of the coded region (the number a receiver
        needs when the capture continues past the end of the frame).
    """
    coded = encode_fec(message, fec, nsym)
    coded = apply_interleaver(
        coded,
        interleaver,
        rows=rows,
        cols=cols,
        depth=depth,
        spacing=spacing,
        seed=seed,
        random_block=random_block,
    )
    sync_bits = hex_to_bits(sync_word)
    return {
        "bits": np.concatenate([sync_bits, coded]).astype(np.uint8),
        "sync_bits": sync_bits,
        "coded_bits": coded,
        "frame_bits": int(coded.size),
        "layout": {
            "sync_word": sync_word,
            "sync_length": int(sync_bits.size),
            "fec": fec,
            "nsym": int(nsym) if fec in ("rs", "concatenated") else None,
            "interleaver": interleaver,
            "message_bytes": len(message),
            "coded_bits": int(coded.size),
        },
    }


def decode_frame_bits(bits: np.ndarray, sync_offset: int = 0) -> np.ndarray:
    """Convenience: strip the sync word from a received bit stream."""
    arr = np.asarray(bits, dtype=np.uint8).ravel()
    start = max(0, int(sync_offset))
    return arr[start:]


__all__ = [
    "FEC_SCHEMES",
    "INTERLEAVER_SCHEMES",
    "MODULATION_SCHEMES",
    "apply_interleaver",
    "build_frame",
    "decode_frame_bits",
    "encode_fec",
    "modulate",
    "modulate_2fsk",
]
