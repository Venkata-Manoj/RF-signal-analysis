"""BPSK/QPSK/8PSK/16-QAM/64-QAM/2-FSK/4-FSK demodulation. Full spec: info.md §12.3.

Phase-aligned slicers (no timing recovery here).
Robust sync lives in core/receiver.py — do not add Costas loops here.
"""

from __future__ import annotations

import numpy as np


def normalize_signal(samples: np.ndarray) -> np.ndarray:
    """Normalize signal amplitude."""
    samples = np.asarray(samples)
    if samples.size == 0:
        return samples
    max_val = np.max(np.abs(samples))
    if max_val == 0:
        return samples
    return samples / max_val


def demod_bpsk(samples: np.ndarray) -> np.ndarray:
    """Simple BPSK demodulation. Assumes phase-aligned signal."""
    samples = normalize_signal(samples)
    bits = (np.real(samples) > 0).astype(np.uint8)
    return bits


def demod_qpsk(samples: np.ndarray) -> np.ndarray:
    """Simple QPSK demodulation. Assumes phase-aligned signal."""
    samples = normalize_signal(samples)
    bit_i = (np.real(samples) > 0).astype(np.uint8)
    bit_q = (np.imag(samples) > 0).astype(np.uint8)
    bits = np.empty(bit_i.size * 2, dtype=np.uint8)
    bits[0::2] = bit_i
    bits[1::2] = bit_q
    return bits


def demod_2fsk(samples: np.ndarray, samples_per_symbol: int = 1) -> np.ndarray:
    """Simple 2-FSK demodulation using instantaneous frequency.

    When ``samples_per_symbol > 1``, downsamples the instantaneous
    frequency by taking the median within each symbol period — producing
    1 bit per symbol instead of 1 bit per sample.
    """
    samples = normalize_signal(samples)
    if len(samples) < 2:
        return np.array([], dtype=np.uint8)

    phase = np.angle(samples)
    phase_unwrapped = np.unwrap(phase)
    inst_freq = np.diff(phase_unwrapped)
    # Append the last frequency sample to preserve length = len(samples)
    inst_freq = np.concatenate([inst_freq, [inst_freq[-1]]])

    if samples_per_symbol > 1 and len(inst_freq) >= samples_per_symbol:
        n_symbols = len(inst_freq) // samples_per_symbol
        if n_symbols > 0:
            trimmed = inst_freq[: n_symbols * samples_per_symbol]
            reshaped = trimmed.reshape(n_symbols, samples_per_symbol)
            inst_freq = np.median(reshaped, axis=1)

    threshold = 0.0
    bits = (inst_freq > threshold).astype(np.uint8)
    return bits


def bits_to_bpsk(bits: np.ndarray) -> np.ndarray:
    """Map 0/1 bits to BPSK symbols (-1, +1)."""
    return 2.0 * bits.astype(np.float32) - 1.0


def bits_to_qam16(bits: np.ndarray) -> np.ndarray:
    """Gray-coded 16-QAM mapper, 4 bits/symbol.

    Input bits order per symbol: [i_msb, i_lsb, q_msb, q_lsb].
    Levels: -3,-1,1,3 / sqrt(10) so average power = 1.
    Truncates tail to a multiple of 4.
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    n = (len(bits) // 4) * 4
    if n == 0:
        return np.array([], dtype=np.complex64)
    bits = bits[:n]
    i_msb, i_lsb, q_msb, q_lsb = bits[0::4], bits[1::4], bits[2::4], bits[3::4]

    def level(msb: np.ndarray, lsb: np.ndarray) -> np.ndarray:
        is_pos = msb == 1
        is_inner = lsb == 1
        lvl = np.where(is_inner, 1.0, 3.0)
        return np.where(is_pos, lvl, -lvl)

    i = level(i_msb, i_lsb)
    q = level(q_msb, q_lsb)
    symbols = (i + 1j * q) / np.sqrt(10.0)
    return symbols.astype(np.complex64)


def demod_qam16(samples: np.ndarray) -> np.ndarray:
    """Naive 16-QAM demod, phase-aligned, no equalization (see receiver.lms_equalize).

    Thresholds on normalized I/Q: 0 separates sign (msb),
    |x| > 0.6 separates outer (lsb=0) vs inner (lsb=1).
    """
    samples = normalize_signal(np.asarray(samples))
    if samples.size == 0:
        return np.array([], dtype=np.uint8)
    real = np.real(samples)
    imag = np.imag(samples)
    bits = []
    for r, q in zip(real, imag):
        i_msb = 1 if r > 0 else 0
        i_lsb = 0 if abs(float(r)) > 0.6 else 1
        q_msb = 1 if q > 0 else 0
        q_lsb = 0 if abs(float(q)) > 0.6 else 1
        bits.extend([i_msb, i_lsb, q_msb, q_lsb])
    return np.array(bits, dtype=np.uint8)


# Aliases for pipeline/GUI tolerance.
demod_16qam = demod_qam16
demod_qam = demod_qam16


def demod_8psk(samples: np.ndarray) -> np.ndarray:
    """Naive 8PSK demod, phase-aligned, no carrier recovery (see receiver.costas_loop for the robust path).

    Phase sectors centred on ideal points (0,45,...315deg), Gray-labelled
    to match waveform.bits_to_symbols. Clean symbol-spaced captures hit
    BER 0.0; frequency offset inflates EVM (the coherent chain corrects it when locked).
    """
    x = normalize_signal(np.asarray(samples))
    if x.size == 0:
        return np.array([], dtype=np.uint8)
    try:
        from rf_analyzer.core import waveform as _wf

        return np.asarray(_wf.symbols_to_bits(x, "8PSK"), dtype=np.uint8)
    except Exception:
        pass
    pos = np.mod(np.round(np.angle(x) * 8.0 / (2.0 * np.pi)), 8.0).astype(np.int64)
    gray = pos ^ (pos >> 1)
    shifts = np.arange(2, -1, -1)
    return ((gray[:, None] >> shifts) & 1).astype(np.uint8).ravel()


def demod_64qam(samples: np.ndarray) -> np.ndarray:
    """Naive 64-QAM demod, phase-aligned, no equalization (see receiver.lms_equalize for the robust path).

    8-PAM per axis, Gray-labelled to match waveform.bits_to_symbols.
    """
    x = normalize_signal(np.asarray(samples))
    if x.size == 0:
        return np.array([], dtype=np.uint8)
    try:
        from rf_analyzer.core import waveform as _wf

        return np.asarray(_wf.symbols_to_bits(x, "64-QAM"), dtype=np.uint8)
    except Exception:
        pass
    scale = np.sqrt(42.0)
    table = np.arange(-7, 8, 2, dtype=float)
    width = 3
    shifts = np.arange(width - 1, -1, -1)
    bits = []
    for axis in (np.real(x) * scale, np.imag(x) * scale):
        pos = np.argmin(np.abs(axis[:, None] - table[None, :]), axis=1)
        gray = pos ^ (pos >> 1)
        bits.append(((gray[:, None] >> shifts) & 1).astype(np.uint8))
    return np.concatenate(bits, axis=1).ravel()


def demod_4fsk(samples: np.ndarray, samples_per_symbol: int = 1) -> np.ndarray:
    """Naive 4-FSK demod via instantaneous-frequency quantiles.

    Constant-envelope discriminator (diff unwrap angle), 4 tone levels from
    quartiles when sps==1 else median-per-symbol then quartiles. No Gray
    coding (matches waveform.modulate_mfsk tone-index map). Clean captures
    hit BER 0.0; non-orthogonal spacing costs dB (see waveform docs).
    """
    x = normalize_signal(np.asarray(samples))
    if x.size < 2:
        return np.array([], dtype=np.uint8)
    phase = np.unwrap(np.angle(x))
    inst = np.diff(phase)
    inst = np.concatenate([inst, [inst[-1]]])
    if samples_per_symbol > 1 and len(inst) >= samples_per_symbol:
        n_sym = len(inst) // samples_per_symbol
        if n_sym > 0:
            inst = np.median(
                inst[: n_sym * samples_per_symbol].reshape(n_sym, samples_per_symbol),
                axis=1,
            )
    if inst.size == 0:
        return np.array([], dtype=np.uint8)
    qs = np.quantile(inst, [0.125, 0.375, 0.625, 0.875])
    # Tone boundaries midway between quartile centres.
    bounds = [(qs[i] + qs[i + 1]) / 2.0 for i in range(3)]
    labels = np.digitize(inst, bounds).astype(np.int64)
    shifts = np.arange(1, -1, -1)
    return ((labels[:, None] >> shifts) & 1).astype(np.uint8).ravel()
