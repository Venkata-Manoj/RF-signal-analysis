"""Synthetic waveform generation for the extended demodulator/classifier.

The original ``framing.modulate`` emits **one sample per symbol** for BPSK, QPSK
and 16-QAM. That convention is what made the original demodulators viable with
no timing recovery at all, and it is why they could assume the receiver was
already phase-aligned. Neither assumption survives contact with a real capture,
so the extended chain needs a generator that produces genuinely oversampled,
pulse-shaped, phase-rotated waveforms to be tested against.

This module is that generator. It is deliberately independent of
``framing.modulate``: that function is the *transmit* path for the framed
payload and its 1-sample-per-symbol output is relied on by the pipeline's
existing tests. Nothing here changes it.

Everything is defined so the receiver can be built against it:

* **Gray coding** follows the existing ``demod.bits_to_qam16`` convention, so
  the slicers stay bit-compatible with the original ones. Per axis, the
  ``k`` bits (most significant first) are read as an integer ``b``, the *level
  position* is ``p = gray_to_binary(b)``, and the level is ``2p - (L - 1)``.
  Equivalently, position ``p`` carries the bit label ``p ^ (p >> 1)``, so
  walking the constellation one step moves the bit label by exactly one bit.
  For 16-QAM that reproduces ``-3, -1, +1, +3`` labelled ``00, 01, 11, 10``,
  which is what ``bits_to_qam16`` does. The inverse map (``binary_to_gray``) is
  *not* interchangeable here: the two permutations coincide for two-bit groups
  and differ for three, so using the wrong one is invisible at 16-QAM and
  breaks 64-QAM.
* **Pulse shaping** is root-raised-cosine, unit energy, so a matched filter at
  the receiver yields a raised-cosine Nyquist response with no ISI at the
  symbol instants.
* **Power** is normalised to unit average symbol power for every scheme, so a
  given ``snr_db`` means the same thing across modulations.

``info.md`` §15.1 remains the single source of truth for the 2-FSK phase
convention; :func:`modulate_mfsk` with ``order=2`` reproduces
``framing.modulate_2fsk``'s waveform.
"""

from __future__ import annotations

import numpy as np

#: Default oversampling: 8 samples per symbol. Enough for a Gardner detector to
#: find the midpoint without interpolating, and cheap enough to sweep.
DEFAULT_SPS = 8
#: Default root-raised-cosine roll-off. 0.35 is the usual link-budget choice:
#: it costs 35% excess bandwidth and keeps the timing-error S-curve well behaved.
DEFAULT_BETA = 0.35
#: Default RRC truncation, in symbols either side of the peak.
DEFAULT_SPAN = 8

#: Bits per symbol for every scheme this module can generate.
BITS_PER_SYMBOL = {
    "BPSK": 1,
    "QPSK": 2,
    "8PSK": 3,
    "16-QAM": 4,
    "64-QAM": 6,
    "2-FSK": 1,
    "4-FSK": 2,
}

#: Canonical spellings, so callers cannot drift between ``8PSK``/``8-PSK`` etc.
_SCHEME_ALIASES = {
    "BPSK": "BPSK",
    "QPSK": "QPSK",
    "4-QAM": "QPSK",
    "4QAM": "QPSK",
    "8PSK": "8PSK",
    "8-PSK": "8PSK",
    "16-QAM": "16-QAM",
    "QAM16": "16-QAM",
    "16QAM": "16-QAM",
    "64-QAM": "64-QAM",
    "QAM64": "64-QAM",
    "64QAM": "64-QAM",
    "2-FSK": "2-FSK",
    "2FSK": "2-FSK",
    "FSK": "2-FSK",
    "4-FSK": "4-FSK",
    "4FSK": "4-FSK",
}

#: Linear (constellation-carrying) schemes, i.e. everything but FSK.
LINEAR_SCHEMES = ("BPSK", "QPSK", "8PSK", "16-QAM", "64-QAM")
FSK_SCHEMES = ("2-FSK", "4-FSK")

#: Modulation index that makes an ``order``-tone M-FSK *orthogonal*.
#:
#: CPFSK tones are orthogonal over one symbol when their frequency spacing is a
#: multiple of ``1 / (2T)``, i.e. when the modulation index is a multiple of
#: ``1/2``. :func:`modulate_mfsk` spaces adjacent tones by
#: ``2 * f_dev / (order - 1)`` with ``f_dev = h * Rs / 2``, so orthogonality
#: needs ``h = (order - 1) / 2``: ``0.5`` for 2-FSK and ``1.5`` for 4-FSK.
#:
#: This matters for more than tidiness. With the *non*-orthogonal default the
#: 4-FSK tone spacing is ``Rs/3`` (``h = 1/3``), the four tones overlap
#: heavily and a discriminator receiver pays several dB. At 10 dB Eb/N0 that is
#: the difference between a comfortable pass and a BER above the acceptance
#: threshold, so the default has to be the orthogonal one.
ORTHOGONAL_MFSK_INDEX = {2: 0.5, 4: 1.5}


def orthogonal_modulation_index(order: int) -> float:
    """``(order - 1) / 2`` -- the index that makes ``order``-tone FSK orthogonal."""
    order = int(order)
    if order < 2 or (order & (order - 1)) != 0:
        raise ValueError("order must be a power of two >= 2")
    return (order - 1) / 2.0


def _bits_for_levels(levels: int) -> int:
    """``log2`` of a power of two, in exact integer arithmetic.

    ``levels`` must be a power of two; ``bit_length() - 1`` is then exact, where
    ``int(round(np.log2(levels)))`` is a float round-trip that ruff flags (the
    cast is redundant once ``round`` has already produced an int) and that can
    land one off for large values.
    """
    if levels < 1 or (levels & (levels - 1)) != 0:
        raise ValueError(f"expected a power of two, got {levels}")
    return levels.bit_length() - 1


def canonical_scheme(scheme: str) -> str:
    """Normalise a modulation label to its canonical spelling.

    Raises:
        ValueError: the label is not one this module can generate.
    """
    key = str(scheme).upper().replace("_", "-").strip()
    if key in _SCHEME_ALIASES:
        return _SCHEME_ALIASES[key]
    raise ValueError(
        f"unsupported modulation scheme {scheme!r}; "
        f"pick one of {tuple(sorted(set(_SCHEME_ALIASES.values())))}"
    )


def bits_per_symbol(scheme: str) -> int:
    """Bits carried by one symbol of ``scheme``."""
    return BITS_PER_SYMBOL[canonical_scheme(scheme)]


# --------------------------------------------------------------------------- #
# Gray coding
# --------------------------------------------------------------------------- #


def binary_to_gray(value: np.ndarray) -> np.ndarray:
    """``g = b ^ (b >> 1)``, elementwise."""
    b = np.asarray(value, dtype=np.int64)
    return b ^ (b >> 1)


def gray_to_binary(value: np.ndarray) -> np.ndarray:
    """Inverse of :func:`binary_to_gray`, elementwise."""
    g = np.asarray(value, dtype=np.int64).copy()
    result = g.copy()
    shift = 1
    while True:
        shifted = g >> shift
        if not np.any(shifted):
            break
        result ^= shifted
        shift += 1
    return result


def _pack_bits(bits: np.ndarray, width: int) -> np.ndarray:
    """Group a bit stream into integers, most significant bit first.

    Truncates any trailing partial group -- the caller decides what to do about
    the remainder, because silently zero-padding it would change the bit count
    and make a BER measurement meaningless.
    """
    arr = np.asarray(bits, dtype=np.uint8).ravel()
    usable = (arr.size // width) * width
    if usable == 0:
        return np.zeros(0, dtype=np.int64)
    grouped = arr[:usable].reshape(-1, width)
    weights = 1 << np.arange(width - 1, -1, -1, dtype=np.int64)
    return grouped.astype(np.int64) @ weights


def pam_levels(levels_per_axis: int) -> np.ndarray:
    """Symmetric PAM levels ``-(L-1) ... +(L-1)`` in steps of two."""
    return np.arange(-(levels_per_axis - 1), levels_per_axis, 2, dtype=np.float64)


def levels_from_index(index: np.ndarray, levels_per_axis: int) -> np.ndarray:
    """Map already-packed bit groups onto PAM levels, Gray-labelled.

    ``index`` holds the bit group as an integer (most significant bit first).
    The **level position** is ``p = gray_to_binary(index)`` and the level is
    ``2p - (L - 1)``, so position ``p`` carries the bit label
    ``p ^ (p >> 1)`` -- the binary-reflected Gray code, whose consecutive
    entries differ in exactly one bit. That is what makes adjacent levels
    one-bit neighbours, which is what makes a BER measurement mean anything.

    Note the direction: it is the *inverse* Gray map here, not
    ``binary_to_gray``. Using ``binary_to_gray`` is correct for two-bit groups
    by coincidence (the two permutations agree) and silently wrong for
    three-bit groups, which is how 64-QAM came out with two-bit neighbours
    across the axis centre.

    Call this directly when the bit groups are already packed; calling
    :func:`bits_to_pam` with packed groups would group them a second time.
    """
    position = gray_to_binary(np.asarray(index, dtype=np.int64))
    return (2.0 * position - (levels_per_axis - 1)).astype(np.float64)


def bits_to_pam(bits: np.ndarray, levels_per_axis: int) -> np.ndarray:
    """Gray-coded PAM mapping of a bit stream onto ``levels_per_axis`` levels.

    The convention is the one the existing ``demod.bits_to_qam16`` uses, so the
    two agree bit for bit: bit group ``b`` (MSB first) is read as a *position
    index*, ``p = gray_to_binary(b)``, and the level is ``2p - (L - 1)``.

    Note the direction -- it is the **inverse** Gray map, matching
    :func:`levels_from_index`. Writing ``g = b ^ (b >> 1)`` here instead is
    correct for two-bit groups by coincidence and silently wrong for three, so
    it is invisible at 16-QAM and breaks 64-QAM. This docstring used to say
    exactly that wrong thing; the code was right and the prose was not, which
    is how a reader reintroduces the bug.
    """
    width = _bits_for_levels(levels_per_axis)
    return levels_from_index(_pack_bits(bits, width), levels_per_axis)


def pam_to_bits(levels: np.ndarray, levels_per_axis: int) -> np.ndarray:
    """Inverse of :func:`bits_to_pam`: nearest level, then Gray-decode."""
    width = _bits_for_levels(levels_per_axis)
    table = pam_levels(levels_per_axis)
    positions = np.argmin(np.abs(np.asarray(levels)[:, None] - table[None, :]), axis=1)
    labels = binary_to_gray(positions)
    return (
        ((labels[:, None] >> np.arange(width - 1, -1, -1)) & 1).astype(np.uint8).ravel()
    )


def bits_to_symbols(bits: np.ndarray, scheme: str) -> np.ndarray:
    """Map a hard bit stream onto unit-average-power constellation points.

    Trailing bits that do not fill a whole symbol are dropped. ``2-FSK`` and
    ``4-FSK`` have no constellation and raise -- use :func:`modulate_mfsk`.
    """
    name = canonical_scheme(scheme)
    if name in FSK_SCHEMES:
        raise ValueError(f"{name} has no constellation; use modulate_mfsk()")
    arr = np.asarray(bits, dtype=np.uint8).ravel()

    if name == "BPSK":
        n = arr.size
        if n == 0:
            return np.zeros(0, dtype=np.complex128)
        return (2.0 * arr[:n].astype(np.float64) - 1.0).astype(np.complex128)

    if name == "QPSK":
        # One bit per axis, exactly as framing.modulate and demod.demod_qpsk
        # already do it: bit 1 -> I, bit 0 -> Q, and the bit decides the sign.
        # Walking the quadrants 00 -> 01 -> 11 -> 10 visits them in order and
        # every step flips one bit.
        grouped = _pack_bits(arr, 2)
        i = levels_from_index((grouped >> 1) & 1, 2)
        q = levels_from_index(grouped & 1, 2)
        return ((i + 1j * q) / np.sqrt(2.0)).astype(np.complex128)

    if name == "8PSK":
        # Position p carries the label p ^ (p >> 1), so the constellation walk
        # and the bit walk advance together one Gray step at a time.
        position = gray_to_binary(_pack_bits(arr, 3))
        return np.exp(1j * 2.0 * np.pi * position / 8.0).astype(np.complex128)

    if name == "16-QAM":
        # Bit order [i_msb, i_lsb, q_msb, q_lsb], matching demod.bits_to_qam16.
        grouped = _pack_bits(arr, 4)
        i = levels_from_index((grouped >> 2) & 0b11, 4)
        q = levels_from_index(grouped & 0b11, 4)
        return ((i + 1j * q) / np.sqrt(10.0)).astype(np.complex128)

    # 64-QAM: three bits per axis, most significant first.
    grouped = _pack_bits(arr, 6)
    i = levels_from_index((grouped >> 3) & 0b111, 8)
    q = levels_from_index(grouped & 0b111, 8)
    return ((i + 1j * q) / np.sqrt(42.0)).astype(np.complex128)


def symbols_to_bits(symbols: np.ndarray, scheme: str) -> np.ndarray:
    """Nearest-point decision back to hard bits (no carrier/timing recovery)."""
    name = canonical_scheme(scheme)
    if name in FSK_SCHEMES:
        raise ValueError(f"{name} carries no constellation; use an FSK demodulator")
    x = np.asarray(symbols, dtype=np.complex128).ravel()
    if x.size == 0:
        return np.zeros(0, dtype=np.uint8)

    if name == "BPSK":
        return (np.real(x) > 0.0).astype(np.uint8)
    if name == "QPSK":
        # Inverse of the mapper: the bit is just the sign of its axis, which is
        # also what demod.demod_qpsk does.
        i_bits = (np.real(x) > 0.0).astype(np.uint8)
        q_bits = (np.imag(x) > 0.0).astype(np.uint8)
        return np.stack([i_bits, q_bits], axis=1).ravel()
    if name == "8PSK":
        phase = np.angle(x)
        position = np.mod(np.round(phase * 8.0 / (2.0 * np.pi)), 8).astype(np.int64)
        labels = binary_to_gray(position)
        return ((labels[:, None] >> np.arange(2, -1, -1)) & 1).astype(np.uint8).ravel()

    scale = np.sqrt(10.0) if name == "16-QAM" else np.sqrt(42.0)
    levels = 4 if name == "16-QAM" else 8
    bits_per_axis = _bits_for_levels(levels)
    # Reshape to one row per symbol and concatenate along the *symbol* axis.
    # Concatenating the flat arrays instead would emit every I bit and then
    # every Q bit, whereas the mapper groups each symbol's bits as
    # [i_msb, i_lsb, q_msb, q_lsb].
    i_bits = pam_to_bits(np.real(x) * scale, levels).reshape(-1, bits_per_axis)
    q_bits = pam_to_bits(np.imag(x) * scale, levels).reshape(-1, bits_per_axis)
    return np.concatenate([i_bits, q_bits], axis=1).ravel()


# --------------------------------------------------------------------------- #
# Pulse shaping
# --------------------------------------------------------------------------- #


def rrc_taps(
    sps: int = DEFAULT_SPS, beta: float = DEFAULT_BETA, span: int = DEFAULT_SPAN
) -> np.ndarray:
    """Unit-energy root-raised-cosine impulse response.

    ``span`` symbols either side of the peak, so the filter is
    ``2 * span * sps + 1`` taps long. The three removable singularities --
    ``t = 0`` and ``t = +/- T / (4 beta)`` -- are filled in from their limits
    rather than left as ``0/0``.

    Args:
        sps: Samples per symbol.
        beta: Roll-off, ``0 <= beta <= 1``. ``beta = 0`` degenerates to a sinc.
        span: Half-length in symbols.

    Returns:
        Real, unit-energy FIR taps.
    """
    sps = int(sps)
    if sps < 1:
        raise ValueError("sps must be >= 1")
    beta = float(beta)
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")

    n = np.arange(-span * sps, span * sps + 1, dtype=np.float64)
    t = n / sps  # in symbol periods

    if beta == 0.0:
        h = np.sinc(t)
    else:
        numerator = np.sin(np.pi * t * (1.0 - beta)) + 4.0 * beta * t * np.cos(
            np.pi * t * (1.0 + beta)
        )
        denominator = np.pi * t * (1.0 - (4.0 * beta * t) ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            h = numerator / denominator
        # t = 0
        h[t == 0.0] = 1.0 - beta + 4.0 * beta / np.pi
        # t = +/- 1 / (4 beta)
        singular = np.isclose(np.abs(t), 1.0 / (4.0 * beta))
        h[singular] = (beta / np.sqrt(2.0)) * (
            (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
            - (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
        )

    energy = float(np.sum(h**2))
    if energy <= 0.0:
        raise ValueError("RRC filter has zero energy; check sps/beta/span")
    return h / np.sqrt(energy)


def upsample_and_pulse(
    symbols: np.ndarray,
    sps: int = DEFAULT_SPS,
    beta: float = DEFAULT_BETA,
    span: int = DEFAULT_SPAN,
    taps: np.ndarray | None = None,
) -> np.ndarray:
    """Insert ``sps - 1`` zeros between symbols and shape with an RRC filter.

    The result is ``len(symbols) * sps`` samples long: the filter's group delay
    is trimmed off both ends so symbol ``k`` is centred at sample
    ``k * sps + sps // 2``, which is what the timing-recovery loop expects.
    """
    x = np.asarray(symbols, dtype=np.complex128).ravel()
    if x.size == 0:
        return np.zeros(0, dtype=np.complex128)
    sps = int(sps)
    if taps is None:
        taps = rrc_taps(sps, beta, span)
    upsampled = np.zeros(x.size * sps, dtype=np.complex128)
    upsampled[::sps] = x
    shaped = np.convolve(upsampled, taps, mode="full")
    delay = (taps.size - 1) // 2
    return shaped[delay : delay + x.size * sps]


# --------------------------------------------------------------------------- #
# FSK
# --------------------------------------------------------------------------- #


def modulate_mfsk(
    bits: np.ndarray,
    order: int = 2,
    sps: int = DEFAULT_SPS,
    modulation_index: float = 0.5,
    sample_rate: float = 1.0,
) -> np.ndarray:
    """Continuous-phase M-FSK.

    ``order`` tones, spaced so the peak deviation gives the requested
    modulation index ``h = 2 * f_dev * T``. With ``order=2``,
    ``modulation_index=0.5``, ``sps`` samples per symbol and
    ``sample_rate = 1 / (symbol_duration * sps)`` this reproduces the waveform
    ``framing.modulate_2fsk`` produces at its default ``+/-5 kHz`` on 1 ms
    symbols -- see the test that pins the two against each other.

    Args:
        bits: Hard bit stream.
        order: Number of tones (2 or 4).
        sps: Samples per symbol.
        modulation_index: Peak deviation normalised by symbol rate.
        sample_rate: Sample rate in Hz; only sets the absolute tone spacing.

    Returns:
        Complex baseband samples, ``len(bits) * sps`` long, unit power.
    """
    order = int(order)
    if order < 2 or (order & (order - 1)) != 0:
        raise ValueError("order must be a power of two >= 2")
    arr = np.asarray(bits, dtype=np.uint8).ravel()
    if arr.size == 0:
        return np.zeros(0, dtype=np.complex64)
    sps = int(sps)
    width = _bits_for_levels(order)
    symbols = _pack_bits(arr, width)
    if symbols.size == 0:
        return np.zeros(0, dtype=np.complex64)

    symbol_rate = float(sample_rate) / sps
    peak_deviation = 0.5 * float(modulation_index) * symbol_rate
    # Symmetric tone set around DC: 2-FSK -> {-f, +f}, 4-FSK -> {-3f, -f, +f, +3f}
    # scaled so the outermost tone sits at the peak deviation.
    offsets = np.arange(order, dtype=np.float64) - (order - 1) / 2.0
    offsets *= 2.0 * peak_deviation / max(1.0, (order - 1))
    tone = offsets[symbols]

    radians_per_hz = 2.0 * np.pi / float(sample_rate)
    # Phase carries over symbol to symbol, matching info.md §15.1: a symbol
    # advances the phase by (sps - 1) intervals and the boundary sample is
    # shared with the next symbol.
    advance = tone * ((sps - 1) * radians_per_hz)
    carried = np.concatenate([[0.0], np.cumsum(advance)[:-1]])
    within = (
        np.tile(np.arange(sps), symbols.size) * np.repeat(tone, sps) * radians_per_hz
    )
    phase = np.repeat(carried, sps) + within
    wave = np.exp(1j * phase)
    return (wave / np.sqrt(np.mean(np.abs(wave) ** 2))).astype(np.complex64)


# --------------------------------------------------------------------------- #
# Noise
# --------------------------------------------------------------------------- #


def add_awgn(
    samples: np.ndarray, snr_db: float, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Add complex AWGN at the requested SNR.

    ``snr_db`` is defined against the signal's own measured power, so it is
    meaningful for any of the generators above without them having to report a
    power figure. Returns a new array; the input is not modified.
    """
    x = np.asarray(samples, dtype=np.complex128)
    if x.size == 0:
        return x
    rng = rng or np.random.default_rng(0)
    power = float(np.mean(np.abs(x) ** 2))
    if power <= 0.0:
        return x.copy()
    noise_power = power / (10.0 ** (float(snr_db) / 10.0))
    noise = (rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size)) / np.sqrt(
        2.0
    )
    return x + noise * np.sqrt(noise_power)


def ebn0_to_esn0(ebn0_db: float, bits_per_symbol: int) -> float:
    """Convert an energy-per-bit SNR to an energy-per-symbol SNR.

    ``Es/N0 = Eb/N0 + 10 log10(k)`` for ``k`` bits per symbol. This conversion
    is the whole reason a bare "10 dB" is not a usable BER operating point: at
    the same 10 dB the *per-symbol* SNR is 10 dB for BPSK but 17.8 dB for
    64-QAM, so comparing modes at a fixed ``Es/N0`` compares them at different
    per-bit energies and a dense constellation loses by construction.

    Every BER number in this project is therefore quoted at a stated ``Eb/N0``
    and the equivalent ``Es/N0`` is derived through this function, per mode.
    """
    k = int(bits_per_symbol)
    if k < 1:
        raise ValueError("bits_per_symbol must be >= 1")
    return float(ebn0_db) + 10.0 * float(np.log10(k))


def add_awgn_ebn0(
    samples: np.ndarray,
    ebn0_db: float,
    bits_per_symbol: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add AWGN at a stated energy-per-bit SNR, for a given constellation size.

    Thin wrapper over :func:`add_awgn` via :func:`ebn0_to_esn0`; see that
    function for why the per-bit convention is the honest one.
    """
    return add_awgn(samples, ebn0_to_esn0(ebn0_db, bits_per_symbol), rng)


def synth(
    scheme: str,
    n_bits: int,
    *,
    sps: int = DEFAULT_SPS,
    beta: float = DEFAULT_BETA,
    snr_db: float | None = None,
    ebn0_db: float | None = None,
    seed: int = 0,
    modulation_index: float | None = None,
    sample_rate: float = 1_000_000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a random capture and the exact bits that produced it.

    Returns ``(samples, transmitted_bits)``. ``transmitted_bits`` is truncated
    to a whole number of symbols, so a BER computed against it is exact -- no
    padding, no guessing about the tail.

    ``snr_db=None`` and ``ebn0_db=None`` return the noiseless waveform. The two
    are mutually exclusive; ``ebn0_db`` is the per-bit convention and is what
    every BER acceptance point in this project is stated in (see
    :func:`ebn0_to_esn0`). ``modulation_index=None`` uses the *orthogonal* index
    for the FSK order (see :data:`ORTHOGONAL_MFSK_INDEX`) rather than a
    spacing that makes the tones overlap.
    """
    if snr_db is not None and ebn0_db is not None:
        raise ValueError("pass snr_db or ebn0_db, not both")
    name = canonical_scheme(scheme)
    rng = np.random.default_rng(seed)
    width = bits_per_symbol(name)
    n_symbols = max(1, int(n_bits) // width)
    bits = rng.integers(0, 2, size=n_symbols * width, dtype=np.uint8)

    if name in FSK_SCHEMES:
        order = 2 if name == "2-FSK" else 4
        index = (
            orthogonal_modulation_index(order)
            if modulation_index is None
            else float(modulation_index)
        )
        samples = modulate_mfsk(
            bits,
            order=order,
            sps=sps,
            modulation_index=index,
            sample_rate=sample_rate,
        )
    else:
        symbols = bits_to_symbols(bits, name)
        samples = upsample_and_pulse(symbols, sps=sps, beta=beta)

    if ebn0_db is not None:
        samples = add_awgn_ebn0(samples, ebn0_db, width, rng)
    elif snr_db is not None:
        samples = add_awgn(samples, snr_db, rng)
    return np.asarray(samples, dtype=np.complex64), bits


def burst_in_noise(
    scheme: str,
    n_bits: int,
    *,
    snr_db: float | None = None,
    ebn0_db: float | None = None,
    sps: int = DEFAULT_SPS,
    beta: float = DEFAULT_BETA,
    lead: int = 2048,
    tail: int = 2048,
    seed: int = 0,
    modulation_index: float | None = None,
    sample_rate: float = 1_000_000.0,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """A frame-shaped burst surrounded by noise -- the realistic capture.

    Every synthetic test in this repo used to write the waveform as the *whole*
    file, which is a laboratory condition: a real recording is a burst that
    starts and ends somewhere in the middle of noise, and the noise is what
    breaks naive estimators and whole-capture statistics.

    The burst is generated noiselessly, then embedded at a random-free fixed
    offset in ``lead``/``tail`` samples of AWGN whose power is set from the
    stated SNR *relative to the burst*. Returns ``(samples, bits, region)``
    where ``region == (start_sample, end_sample)`` is the exact extent of the
    burst -- the ground truth a burst-region estimator is scored against.
    """
    if snr_db is not None and ebn0_db is not None:
        raise ValueError("pass snr_db or ebn0_db, not both")
    clean, bits = synth(
        scheme,
        n_bits,
        sps=sps,
        beta=beta,
        seed=seed,
        modulation_index=modulation_index,
        sample_rate=sample_rate,
    )
    rng = np.random.default_rng(seed + 1)
    power = float(np.mean(np.abs(clean.astype(np.complex128)) ** 2))
    if ebn0_db is not None:
        level_db = ebn0_to_esn0(ebn0_db, bits_per_symbol(scheme))
    elif snr_db is not None:
        level_db = float(snr_db)
    else:
        level_db = None
    noise_scale = (
        0.0
        if level_db is None or power <= 0.0
        else float(np.sqrt(power / (10.0 ** (level_db / 10.0))))
    )
    total = int(lead) + clean.size + int(tail)
    noise = (rng.standard_normal(total) + 1j * rng.standard_normal(total)) / np.sqrt(
        2.0
    )
    out = noise * noise_scale
    start = int(lead)
    out[start : start + clean.size] += clean.astype(np.complex128)
    return (
        out.astype(np.complex64),
        bits,
        (start, start + clean.size),
    )


__all__ = [
    "BITS_PER_SYMBOL",
    "DEFAULT_BETA",
    "DEFAULT_SPAN",
    "DEFAULT_SPS",
    "FSK_SCHEMES",
    "LINEAR_SCHEMES",
    "ORTHOGONAL_MFSK_INDEX",
    "add_awgn",
    "add_awgn_ebn0",
    "binary_to_gray",
    "bits_per_symbol",
    "bits_to_pam",
    "bits_to_symbols",
    "burst_in_noise",
    "canonical_scheme",
    "ebn0_to_esn0",
    "gray_to_binary",
    "levels_from_index",
    "modulate_mfsk",
    "orthogonal_modulation_index",
    "pam_levels",
    "pam_to_bits",
    "rrc_taps",
    "symbols_to_bits",
    "synth",
    "upsample_and_pulse",
]
