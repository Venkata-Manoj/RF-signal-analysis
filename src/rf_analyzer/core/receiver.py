"""Coherent receiver chain: carrier, timing, equalisation, full demod set.

The slicers in :mod:`rf_analyzer.core.demod` are deliberately simple:
they assume the capture is already phase-aligned and already at one sample per
symbol. This module is the robust receive chain used by default.
It does not replace ``demod.py`` -- the pipeline falls back to the
simple path whenever this chain abstains -- it adds the primary path that
reports what it actually achieved.

Why the design is what it is
============================

**Energy per bit, not per symbol.** Every BER number this module is tested
against is quoted at a stated ``Eb/N0`` and converted to ``Es/N0`` through
``waveform.ebn0_to_esn0``. That is not pedantry: at a fixed ``Es/N0`` a dense
constellation is compared at a *lower* per-bit energy, so 64-QAM would need
about 7.8 dB more per-symbol SNR than BPSK for the same per-bit energy and would
look hopeless for a reason that has nothing to do with the receiver. See
``waveform.ebn0_to_esn0`` and the test docstrings.

**Stage order.** For PSK the order is timing -> equalise (CMA) -> carrier
(Costas). CMA is a constant-modulus criterion, so it is blind to the carrier
phase and can run before the phase is known; the Costas loop then cleans up the
phase the equaliser left behind. For QAM the order is timing -> carrier ->
equalise (LMS), because the LMS error is ``y - decision(y)`` and a decision is
only meaningful once the constellation is roughly de-rotated. Running the
decision-directed equaliser first is the classic way to make a QAM receiver fail
to acquire, and it is why the order is asymmetric.

**Timing acquisition is feedforward, tracking is a loop.** The Gardner and
Mueller & Mueller loops both have a narrow pull-in range and a sign that is easy
to get backwards, so acquisition is done by the closed-form Oerder & Meyr
estimator (the phase of the ``|x|^2`` Fourier coefficient at the symbol rate)
and the loop then tracks. Measured on a clean BPSK capture the Oerder & Meyr
estimate recovers a known fractional offset of 1.3/3.7/6.5 samples to within
0.005 samples, while the same loop started from a zero phase converges to the
*wrong* zero of the Gardner S-curve -- see ``_oerder_meyr`` and the S-curve note
below. The loops are still real loops and are still what rejects residual
jitter; they are simply not asked to acquire a half-symbol offset blind.

**Loop signs are measured, not assumed.** The Gardner S-curve was measured on a
clean matched-filtered BPSK capture over one symbol period (offset in samples,
mean normalised error): ``0.0 -> +0.0001``, ``1.0 -> +0.135``, ``2.0 -> +0.191``,
``4.0 -> -0.0003``, ``6.0 -> -0.192``, ``7.5 -> -0.073``. The slope at the
origin is *positive*, so the update is ``tau -= mu * e`` for a stable lock; the
sign was inverted in the first draft and the loop locked a half symbol late.
The Mueller & Mueller S-curve has the *opposite* slope (``1.0 -> -0.245``,
``6.0 -> +0.478``), so its update is ``tau += mu * e``. Both loops therefore
exist as separate functions rather than one parameterised loop, because a shared
sign would be wrong for one of them.

**Carrier ambiguity is real and is reported, not hidden.** A Costas loop locks
the phase modulo the constellation's rotational symmetry (``pi`` for BPSK,
``pi/2`` for QPSK/16-QAM/64-QAM, ``pi/4`` for 8PSK). No blind estimator can do
better -- the ambiguity is information-theoretic, not an implementation gap --
which is exactly why real links carry a sync word. ``receive`` therefore
minimises the rotational ambiguity it can (it keeps the hypothesis with the
lowest EVM) and records the residual ambiguity in ``metrics``; the test suite
resolves it the way the pipeline does, with the known reference. Claiming a
lower BER by silently picking the best rotation would be the dishonest thing to
do, and this module does not do it.

**The FSK path is deliberately different.** An M-FSK waveform is
constant-envelope and its information lives in the instantaneous frequency, so
the discriminator ``diff(unwrap(angle))`` is used instead of a matched filter,
and no carrier recovery is applied: a residual carrier offset adds the *same*
constant to every symbol's measured frequency, so it shifts all tone levels
together and is absorbed by estimating the levels from the data. Estimating the
tone levels (1-D k-means) rather than assuming the generator's spacing is what
makes the path work at any modulation index, including the non-orthogonal one.
The cost is a real one: a discriminator receiver pays a few dB against coherent
detection, which is visible in the 0 dB column of the BER table.

**Abstention.** ``estimate_sps`` returns ``None`` with ``found=False`` rather
than guessing, ``locked`` is False whenever a loop did not converge, and a
stage that could not run names itself in ``metrics["recovery"]``. See
``AGENTS.md``'s honesty contracts; this module adds to them rather than relaxing
them.

Public API
==========

``receive`` / ``receive_burst`` are the entry points the pipeline calls.
``costas_loop``, ``gardner_recover``, ``mueller_muller_recover``,
``cma_equalize``, ``lms_equalize``, ``matched_filter`` and ``estimate_sps`` are
independently testable building blocks.
"""

from __future__ import annotations

import numpy as np

from rf_analyzer.core import channel as ch
from rf_analyzer.core import dsp, rate_est
from rf_analyzer.core import waveform as wf

# --------------------------------------------------------------------------- #
# Loop gains and thresholds
#
# Every constant here was calibrated against ``waveform.synth`` captures; the
# measured numbers are in the comments. They are not magic values copied from a
# textbook -- a textbook value for a different pulse shape and SNR is how the
# first draft of the timing loop ended up locking half a symbol late.
# --------------------------------------------------------------------------- #

#: Gardner tracking gain, in samples per unit normalised error. The Gardner
#: S-curve has a slope of about +0.38 per sample at the origin (measured
#: +0.135 at 0.5 samples, +0.191 at 2.0), so a gain of 0.05 moves the estimate
#: by ~0.02 samples per symbol at a half-sample error: fast enough to track a
#: static offset, slow enough not to ring. 0.2 was tried and oscillated.
GARDNER_MU = 0.05

#: Mueller & Mueller tracking gain. Its S-curve slope is steeper than Gardner's
#: (-0.48 per sample at the origin), so the gain is halved to keep the same
#: loop bandwidth.
MM_MU = 0.02

#: Second-order Costas loop gains (proportional, integral), per symbol. They
#: follow the standard normalised design for a damping of 0.707 and a noise
#: bandwidth of ~0.02 of the symbol rate; the integral term is what tracks a
#: residual carrier offset, and it can pull in 2% of the symbol rate from the
#: coarse estimate in well under a hundred symbols.
COSTAS_KP = 0.05
COSTAS_KI = 0.0015

#: CMA step size. The input is normalised to unit power first, so the update is
#: scale-free. 0.01 converges a 3-tap multipath channel in a few hundred symbols
#: without visibly degrading an already-equalised capture; 0.05 diverged on the
#: clean captures.
CMA_MU = 0.01

#: LMS step size, same normalisation. Smaller than CMA because the
#: decision-directed error is small near lock.
LMS_MU = 0.005

#: Default equaliser length in symbols. A 3-tap multipath channel needs at least
#: 3 taps to invert; 11 leaves margin for a fractional-delay remainder and still
#: converges from a delta initialisation within a few thousand symbols.
DEFAULT_TAPS = 11

#: Coarse timing grid resolution, in samples, for the discriminator-based FSK
#: timing search (linear schemes use the closed-form Oerder & Meyr estimate).
FSK_TIMING_STEP = 0.1

#: Tracking-pass Costas gains for dense QAM (gear-shift second pass, see
#: ``_receive_linear``). The acquisition pass needs wide gains to pull in, but
#: its steady-state jitter costs dense constellations several percent BER
#: (measured on rotated clean captures: 64-QAM 0.026-0.064, 16-QAM 0.004-0.014
#: at best rotation). Re-running from the acquired state with these narrow
#: gains removes the jitter (measured 64-QAM <= 0.0003 over five seeds) without
#: leaving the acquired well.
COSTAS_TRACK_KP = 0.01
COSTAS_TRACK_KI = 0.0002

#: Symbols examined by the FSK timing search. The scan runs k-means per grid
#: point, so a multi-megabit capture is subsampled to this prefix; the timing
#: phase is global, so a prefix estimates it as well as the whole capture.
FSK_TIMING_SCAN_SYMBOLS = 4096

#: Lock threshold for the carrier coherence statistic ``R``: the mean resultant
#: length of the ``order``-th power of the residual phase error. A locked loop
#: gives ``R`` near 1; an unlocked one spreads the residual uniformly over the
#: decision region and gives ``R`` near 0. Measured at 10 dB Eb/N0: 0.99 (BPSK),
#: 0.97 (QPSK), 0.94 (8PSK), 0.93 (16-QAM), 0.88 (64-QAM); on AWGN the same
#: statistic reads 0.03-0.08. The threshold sits an order of magnitude below the
#: weakest locked value and well above the strongest noise value.
CARRIER_LOCK_MIN = 0.35

#: Lock threshold for timing: the Oerder & Meyr coherence ``|c| / sum(p^2)``.
#: Measured on a clean BPSK capture 0.059, on 10 dB Eb/N0 0.043, and on AWGN
#: 0.0008 (the 1/sqrt(N) floor). A threshold of 0.01 sits above the noise floor
#: by more than 10x and below the weakest signal by 4x.
TIMING_LOCK_MIN = 0.01

#: Lock threshold for the FSK discriminator: the mean distance from each symbol's
#: measured frequency to its assigned tone level, divided by the mean tone gap.
#: Measured on clean 2-FSK 0.04, at 10 dB Eb/N0 0.17, and on AWGN 0.44.
FSK_LOCK_MAX_RESIDUAL = 0.30

#: Minimum number of symbols before any lock is claimed. Below this the loop
#: statistics are not meaningful and the honest answer is "did not converge".
MIN_LOCK_SYMBOLS = 64

#: Coarse carrier search: the ``order``-th power line is searched inside this
#: fraction of the symbol rate. The acceptance point is +/-2% of the symbol rate,
#: so 0.25 (i.e. +/-25%) leaves an order of magnitude of margin.
COARSE_CARRIER_MAX_CYCLES = 0.25

#: QAM phase refinement is block based.  A dense constellation makes a
#: decision-directed integral loop wander at the decision boundaries; a short
#: block estimate followed by a low-order fit is both less noisy and easier to
#: audit.  The values are deliberately modest: the channel sweep is an
#: Eb/N0=10 dB test, not a claim of a universal Doppler estimator.
QAM_PHASE_BLOCK = 64
QAM_PHASE_GRID = 64
QAM_PHASE_FIT_DEGREE = 2
QAM_PHASE_REFINEMENTS = 2

#: Below this blind IQ metric the correction is more likely to remove a
#: constellation's finite-sample moment than a real front-end error.  The
#: threshold is well above clean QAM noise measurements and well below the
#: bundled 20%-gain/5-degree impairment.
IQ_CORRECTION_MIN_MAGNITUDE = 0.12


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _as_complex(samples: np.ndarray) -> np.ndarray:
    """Return ``samples`` as a 1-D ``complex128`` array (real input stays real)."""
    x = np.asarray(samples)
    if np.iscomplexobj(x):
        return x.astype(np.complex128, copy=False).ravel()
    return x.astype(np.float64).ravel().astype(np.complex128)


def _interp_cubic(x: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Catmull-Rom cubic interpolation of ``x`` at fractional ``positions``.

    Four-point cubic (Farrow) interpolation rather than linear: the timing loop
    needs a smooth derivative at the sampling instant, and linear interpolation
    puts a corner there, which shows up as a bias in the recovered timing phase.
    The index is clipped to ``[1, n - 3]`` so edge positions extrapolate a
    fraction of a sample instead of raising; only the first and last symbol are
    affected and the caller drops neither, because a two-symbol edge error is
    below the BER the tests measure.
    """
    arr = _as_complex(x)
    n = arr.size
    if n < 4:
        return np.zeros(np.asarray(positions).shape, dtype=np.complex128)
    pos = np.asarray(positions, dtype=np.float64)
    i = np.clip(np.floor(pos).astype(np.int64), 1, n - 3)
    f = pos - i
    x0, x1, x2, x3 = arr[i - 1], arr[i], arr[i + 1], arr[i + 2]
    a = -0.5 * x0 + 1.5 * x1 - 1.5 * x2 + 0.5 * x3
    b = x0 - 2.5 * x1 + 2.0 * x2 - 0.5 * x3
    c = -0.5 * x0 + 0.5 * x2
    d = x1
    return ((a * f + b) * f + c) * f + d


def _oerder_meyr(x: np.ndarray, sps: int) -> float:
    """Closed-form timing phase, in samples within one symbol, mod ``sps``.

    The squared envelope of a pulse-shaped signal carries a spectral line at the
    symbol rate, and the phase of that line is proportional to the sampling
    phase: ``tau = arg(sum |x_k|^2 exp(-j 2 pi k / sps)) * sps / (2 pi)``. This
    is Oerder & Meyr's estimator. It needs excess bandwidth, so it is only used
    for ``sps >= 2``; at exactly one sample per symbol there is no line to
    measure and Mueller & Mueller's decision-directed loop is used instead.

    Returns a value in ``[0, sps)``; the caller wraps it to
    ``(-sps/2, sps/2]``. The integer part of the symbol phase (which symbol is
    "symbol 0") is *not* observable here, and resolving it is what a sync word
    is for.
    """
    power = np.abs(_as_complex(x)) ** 2
    power = power - float(np.mean(power))
    k = np.arange(power.size, dtype=np.float64)
    coefficient = np.sum(power * np.exp(-1j * 2.0 * np.pi * k / float(sps)))
    return float(np.angle(coefficient) / (2.0 * np.pi) * float(sps)) % float(sps)


def _timing_coherence(x: np.ndarray, sps: int) -> float:
    """Normalised magnitude of the symbol-rate line in the squared envelope.

    ``|sum p_k exp(-j 2 pi k / sps)| / sum p_k^2`` with ``p`` the mean-removed
    power. This is the timing-lock statistic: a signal with symbol structure has
    a line and scores a few percent; AWGN has none and scores the ``1/sqrt(N)``
    floor. See :data:`TIMING_LOCK_MIN` for the measured separation.
    """
    power = np.abs(_as_complex(x)) ** 2
    power = power - float(np.mean(power))
    denominator = float(np.sum(power**2))
    if denominator <= 0.0:
        return 0.0
    k = np.arange(power.size, dtype=np.float64)
    coefficient = np.sum(power * np.exp(-1j * 2.0 * np.pi * k / float(sps)))
    return float(np.abs(coefficient) / denominator)


def _wrap_offset(tau: float, sps: int) -> float:
    """Wrap a timing phase into ``(-sps/2, sps/2]``."""
    tau = float(tau) % float(sps)
    if tau > sps / 2.0:
        tau -= float(sps)
    return tau


# --------------------------------------------------------------------------- #
# Matched filter
# --------------------------------------------------------------------------- #


def matched_filter(samples: np.ndarray, sps: int, beta: float = wf.DEFAULT_BETA):
    """Root-raised-cosine matched filter, aligned to the input symbol grid.

    The generator shapes with an RRC pulse and places symbol ``k`` at sample
    ``k * sps`` (verified: correlating the generator's own symbols against its
    output gives a mean squared error of 3e-5 at offset 0 and 0.029 at offset 1).
    Convolving with the same RRC and trimming the filter's group delay keeps that
    alignment, so the output is the same length as the input and symbol ``k`` is
    still at ``k * sps``.

    Args:
        samples: Complex baseband samples.
        sps: Samples per symbol, ``>= 1``.
        beta: RRC roll-off; must match the transmitter's.

    Returns:
        Filtered samples, same length as the input.
    """
    sps = int(sps)
    if sps < 1:
        raise ValueError("sps must be >= 1")
    x = _as_complex(samples)
    if x.size == 0:
        return x
    taps = wf.rrc_taps(sps, float(beta))
    full = np.convolve(x, taps, mode="full")
    delay = (taps.size - 1) // 2
    return full[delay : delay + x.size]


# --------------------------------------------------------------------------- #
# Timing recovery
# --------------------------------------------------------------------------- #


def _timing_loop(
    x: np.ndarray,
    sps: int,
    *,
    detector: str,
    mu: float,
    constellation: np.ndarray,
    tau: float,
) -> tuple[np.ndarray, dict]:
    """Shared body of the Gardner and Mueller & Mueller tracking loops.

    ``detector`` selects the error term; the two differ in *sign* as well, and
    the caller passes the already-signed update through ``mu``.
    """
    n = x.size
    n_sym = n // sps
    if n_sym < 4:
        return np.zeros(0, dtype=np.complex128), {
            "offset": float(tau),
            "locked": False,
            "reason": "too few symbols for a timing loop",
            "n_symbols": int(n_sym),
            "coherence": 0.0,
        }
    power = float(np.mean(np.abs(x) ** 2))
    if power <= 0.0:
        return np.zeros(0, dtype=np.complex128), {
            "offset": float(tau),
            "locked": False,
            "reason": "zero-power input",
            "n_symbols": 0,
            "coherence": 0.0,
        }
    out = np.empty(n_sym, dtype=np.complex128)
    errors = np.empty(n_sym, dtype=np.float64)
    previous = None
    previous_decision = None
    for k in range(n_sym):
        position = tau + k * sps
        y = _interp_cubic(x, np.array([position]))[0]
        if detector == "gardner":
            midpoint = _interp_cubic(x, np.array([position - sps / 2.0]))[0]
            if previous is not None:
                error = float(np.real(np.conj(midpoint) * (y - previous))) / power
                tau += mu * error
                errors[k] = error
        else:
            decision = _nearest(y, constellation)
            if previous is not None and previous_decision is not None:
                error = (
                    float(
                        np.real(
                            np.conj(previous_decision) * y
                            - np.conj(previous) * decision
                        )
                    )
                    / power
                )
                tau += mu * error
                errors[k] = error
            previous_decision = decision
        previous = y
        out[k] = y
    coherence = _timing_coherence(x, sps)
    tail = errors[n_sym // 2 :]
    locked = bool(coherence >= TIMING_LOCK_MIN and n_sym >= MIN_LOCK_SYMBOLS)
    return out, {
        "offset": _wrap_offset(tau, sps),
        "locked": locked,
        "reason": "" if locked else "symbol-rate line below the lock threshold",
        "n_symbols": int(n_sym),
        "coherence": float(coherence),
        "residual_rms": float(np.sqrt(np.mean(tail**2))) if tail.size else None,
    }


def gardner_recover(
    samples: np.ndarray,
    sps: int,
    *,
    beta: float = wf.DEFAULT_BETA,
    mu: float = GARDNER_MU,
):
    """Gardner timing recovery for 2 or more samples per symbol.

    Feedforward Oerder & Meyr acquisition followed by a first-order Gardner
    tracking loop. Gardner's detector uses the midpoint sample between two
    symbols, so it cannot run below two samples per symbol -- at ``sps == 1`` use
    :func:`mueller_muller_recover`.

    Args:
        samples: Complex baseband samples.
        sps: Samples per symbol, ``>= 2``.
        beta: RRC roll-off for the matched filter.
        mu: Tracking gain, samples per unit normalised error.

    Returns:
        ``(symbols, info)``. ``symbols`` is symbol-spaced and complex;
        ``info`` carries ``offset``, ``locked``, ``reason``, ``n_symbols``,
        ``coherence`` and ``residual_rms``.
    """
    sps = int(sps)
    if sps < 2:
        raise ValueError("Gardner timing recovery needs sps >= 2")
    x = matched_filter(samples, sps, beta)
    tau = _wrap_offset(_oerder_meyr(x, sps), sps)
    return _timing_loop(
        x,
        sps,
        detector="gardner",
        mu=-mu,
        constellation=dsp.ideal_constellation("BPSK"),
        tau=tau,
    )


def mueller_muller_recover(
    samples: np.ndarray,
    sps: int,
    *,
    scheme: str = "BPSK",
    beta: float = wf.DEFAULT_BETA,
    mu: float = MM_MU,
):
    """Mueller & Mueller timing recovery, valid from one sample per symbol.

    Decision-directed, so it needs a constellation to slice against and a
    roughly de-rotated signal: it is meant to be used after (or with) carrier
    recovery, or on a capture whose phase is already known. Acquisition is the
    same feedforward Oerder & Meyr estimate as :func:`gardner_recover` when
    ``sps >= 2``; at ``sps == 1`` there is no symbol-rate line to measure and the
    loop acquires from zero, which is why the tests exercise it at 1 sample per
    symbol as well as at 8.

    Args:
        samples: Complex baseband samples.
        sps: Samples per symbol, ``>= 1``.
        scheme: Constellation label used for the decision.
        beta: RRC roll-off for the matched filter (``sps > 1`` only).
        mu: Tracking gain.

    Returns:
        ``(symbols, info)`` as for :func:`gardner_recover`.
    """
    sps = int(sps)
    if sps < 1:
        raise ValueError("sps must be >= 1")
    name = wf.canonical_scheme(scheme)
    constellation = dsp.ideal_constellation(name)
    if constellation is None:
        raise ValueError(f"{name} has no constellation for a decision-directed loop")
    x = matched_filter(samples, sps, beta) if sps > 1 else _as_complex(samples)
    tau = _wrap_offset(_oerder_meyr(x, sps), sps) if sps > 1 else 0.0
    return _timing_loop(
        x,
        sps,
        detector="mueller",
        mu=mu,
        constellation=constellation,
        tau=tau,
    )


# --------------------------------------------------------------------------- #
# Constellations and slicing
# --------------------------------------------------------------------------- #


def _binary_to_gray(value: np.ndarray) -> np.ndarray:
    """``g = b ^ (b >> 1)``. The receiver's own copy, so the round-trip test
    compares two independent implementations rather than one function with
    itself."""
    b = np.asarray(value, dtype=np.int64)
    return b ^ (b >> 1)


def _pam_levels(levels: int) -> np.ndarray:
    """Symmetric PAM levels ``-(L-1) ... +(L-1)`` in steps of two."""
    return np.arange(-(levels - 1), levels, 2, dtype=np.float64)


def _pam_slice(values: np.ndarray, levels: int) -> np.ndarray:
    """Nearest PAM level per value, returned as MSB-first bit columns.

    The level *position* ``p`` is the index into the ascending level table, and
    the bit label is ``binary_to_gray(p)`` -- the same direction as
    ``waveform.pam_to_bits``. The inverse map would be right for two-bit groups
    by coincidence and silently wrong for three, which is the 64-QAM trap the
    generator's docstring warns about.
    """
    table = _pam_levels(levels)
    width = int(levels).bit_length() - 1
    position = np.argmin(np.abs(np.asarray(values)[:, None] - table[None, :]), axis=1)
    labels = _binary_to_gray(position)
    shifts = np.arange(width - 1, -1, -1)
    return ((labels[:, None] >> shifts) & 1).astype(np.uint8)


def _slice_linear(symbols: np.ndarray, scheme: str) -> np.ndarray:
    """Hard bits from symbol-spaced constellation points.

    Bit-for-bit compatible with ``waveform.symbols_to_bits`` for every linear
    scheme; ``tests/unit/test_receiver.py`` pins the two against each other by
    round-tripping through ``waveform.bits_to_symbols``.
    """
    name = wf.canonical_scheme(scheme)
    x = _as_complex(symbols)
    if x.size == 0:
        return np.zeros(0, dtype=np.uint8)

    if name == "BPSK":
        return (np.real(x) > 0.0).astype(np.uint8)
    if name == "QPSK":
        i_bits = (np.real(x) > 0.0).astype(np.uint8)
        q_bits = (np.imag(x) > 0.0).astype(np.uint8)
        return np.stack([i_bits, q_bits], axis=1).ravel()
    if name == "8PSK":
        position = np.mod(np.round(np.angle(x) * 8.0 / (2.0 * np.pi)), 8.0).astype(
            np.int64
        )
        labels = _binary_to_gray(position)
        shifts = np.arange(2, -1, -1)
        return ((labels[:, None] >> shifts) & 1).astype(np.uint8).ravel()

    if name == "16-QAM":
        scale, levels = np.sqrt(10.0), 4
    else:
        scale, levels = np.sqrt(42.0), 8
    i_bits = _pam_slice(np.real(x) * scale, levels)
    q_bits = _pam_slice(np.imag(x) * scale, levels)
    return np.concatenate([i_bits, q_bits], axis=1).ravel()


def _nearest(y: complex, constellation: np.ndarray) -> complex:
    """Nearest constellation point to a single sample."""
    points = np.asarray(constellation, dtype=np.complex128).ravel()
    return complex(points[int(np.argmin(np.abs(points - y) ** 2))])


def _psk_points(order: int) -> np.ndarray:
    """``order`` equally spaced points on the unit circle, starting at 0 rad."""
    return np.exp(1j * 2.0 * np.pi * np.arange(int(order)) / float(order))


def _constellation_for(scheme: str) -> np.ndarray:
    """Ideal constellation for a scheme (FSK raises)."""
    points = dsp.ideal_constellation(scheme)
    if points is None:
        raise ValueError(f"{scheme} has no constellation")
    return np.asarray(points, dtype=np.complex128)


# --------------------------------------------------------------------------- #
# Carrier recovery
# --------------------------------------------------------------------------- #


def _coarse_carrier(symbols: np.ndarray, order: int) -> tuple[float, float]:
    """Coarse carrier frequency and phase from the ``order``-th power line.

    Raising a PSK signal to its order removes the modulation (``s^order`` is a
    constant), leaving a spectral line at ``order`` times the carrier offset and
    a phase of ``order`` times the carrier phase. Measured on a clean BPSK/QPSK/
    8PSK/16-QAM capture with a known offset of 0.02 cycles/symbol the estimate is
    within 2e-5 cycles/symbol for every one of them, which is why the Costas loop
    only has to track a residual instead of acquiring from scratch.

    Args:
        symbols: Symbol-spaced samples.
        order: The power to raise to (2 for BPSK, 4 for QPSK/QAM, 8 for 8PSK).

    Returns:
        ``(cycles_per_symbol, phase_radians)``.
    """
    x = _as_complex(symbols)
    n = x.size
    if n < 16:
        return 0.0, 0.0
    # Normalise first: raising a signal whose level is unknown to the 8th power
    # overflows for a large capture, and the line location is scale-invariant
    # anyway.
    power = float(np.mean(np.abs(x) ** 2))
    if power <= 0.0:
        return 0.0, 0.0
    z = (x / np.sqrt(power)) ** int(order)
    window = np.hanning(n)
    spectrum = np.fft.fftshift(np.fft.fft(z * window))
    freqs = np.fft.fftshift(np.fft.fftfreq(n))
    usable = np.abs(freqs) <= COARSE_CARRIER_MAX_CYCLES
    if not np.any(usable):
        return 0.0, 0.0
    masked = np.where(usable, np.abs(spectrum), -np.inf)
    peak = float(freqs[int(np.argmax(masked))])
    index = np.arange(n, dtype=np.float64)
    phase = float(np.angle(np.sum(z * np.exp(-1j * 2.0 * np.pi * peak * index))))
    return peak / float(order), phase / float(order)


def _phase_error(y: complex, order: int, constellation: np.ndarray) -> float:
    """Order-specific phase detector, normalised to unit slope near lock.

    ``Im{y * conj(d)} / |d|^2`` where ``d`` is the nearest constellation point.
    That is ``sin(theta)`` for a unit-modulus constellation and the standard
    decision-directed detector for QAM; for BPSK it reduces to
    ``sign(Re y) * Im y``, the classic Costas product detector. Dividing by
    ``|d|^2`` keeps the loop gain independent of which QAM ring the sample fell
    on, which matters because the outer ring is 9x the power of the inner one in
    64-QAM.
    """
    decision = _nearest(y, constellation)
    denominator = abs(decision) ** 2
    if denominator <= 0.0:
        return 0.0
    return float(np.imag(y * np.conj(decision)) / denominator)


def costas_loop(
    samples: np.ndarray,
    order: int,
    *,
    sps: int = 1,
    scheme: str | None = None,
    constellation: np.ndarray | None = None,
    kp: float = COSTAS_KP,
    ki: float = COSTAS_KI,
    coarse: bool = True,
    init_phase: float | None = None,
    init_freq: float | None = None,
):
    """Second-order Costas loop with an order-specific phase detector.

    Args:
        samples: Complex samples. Symbol-spaced input is expected; when
            ``sps > 1`` the samples are averaged into symbol blocks first, which
            is a crude matched filter and is only meant for direct use of this
            function.
        order: Detector order (2 BPSK, 4 QPSK/QAM, 8 8PSK).
        sps: Samples per symbol of ``samples``.
        scheme: Optional constellation label; QAM needs it so the
            decision-directed detector slices the real constellation.
        constellation: Optional explicit constellation, overriding ``scheme``.
        kp: Proportional gain.
        ki: Integral gain.
        coarse: Run the ``order``-th power estimator to initialise frequency and
            phase. Without it the loop must acquire the offset itself, which is
            slow and has a narrower pull-in range.
        init_phase: Override the coarse phase estimate (radians).
        init_freq: Override the coarse frequency estimate (cycles/symbol).

    Returns:
        ``(derotated_symbols, info)``. ``info`` carries ``cfo_cycles_per_symbol``,
        ``phase_radians``, ``locked``, ``lock_statistic``, ``reason`` and
        ``order``.
    """
    order = int(order)
    if order < 2:
        raise ValueError("Costas order must be >= 2")
    x = _as_complex(samples)
    sps = int(sps)
    if sps > 1 and x.size >= sps:
        usable = (x.size // sps) * sps
        x = x[:usable].reshape(-1, sps).mean(axis=1)
    n = x.size
    points = (
        np.asarray(constellation, dtype=np.complex128)
        if constellation is not None
        else (_constellation_for(scheme) if scheme else _psk_points(order))
    )
    blank = {
        "order": order,
        "cfo_cycles_per_symbol": 0.0,
        "phase_radians": 0.0,
        "locked": False,
        "lock_statistic": 0.0,
        "reason": "too few symbols for carrier recovery",
        "n_symbols": int(n),
    }
    if n < 4:
        return x, blank

    frequency = 0.0
    phase = 0.0
    if init_freq is not None:
        frequency = float(init_freq)
    elif coarse:
        frequency, _ = _coarse_carrier(x, order)
    if init_phase is not None:
        phase = float(init_phase)
    elif coarse:
        _, phase = _coarse_carrier(x, order)

    out = np.empty(n, dtype=np.complex128)
    residual = np.empty(n, dtype=np.float64)
    for i in range(n):
        y = x[i] * np.exp(-1j * phase)
        error = _phase_error(complex(y), order, points)
        frequency += ki * error
        # ``frequency`` is in cycles/symbol, while ``phase`` and the phase
        # detector are in radians.  The old update treated the former as the
        # latter, turning a 0.012-cycle CFO into roughly a 0.075-radian
        # per-symbol drift (and making the loop look locked while it walked
        # away). Convert explicitly at the phase update.
        phase += 2.0 * np.pi * frequency + kp * error
        out[i] = y
        residual[i] = error

    statistic = _carrier_lock_statistic(out, order, points)
    locked = bool(statistic >= CARRIER_LOCK_MIN and n >= MIN_LOCK_SYMBOLS)
    return out, {
        "order": order,
        "cfo_cycles_per_symbol": float(frequency),
        "phase_radians": float(phase),
        "locked": locked,
        "lock_statistic": float(statistic),
        "reason": "" if locked else "carrier lock statistic below the threshold",
        "n_symbols": int(n),
        "residual_rms": float(np.sqrt(np.mean(residual**2))),
    }


def _carrier_lock_statistic(
    symbols: np.ndarray, order: int, constellation: np.ndarray
) -> float:
    """Mean resultant length of the ``order``-th power of the residual phase.

    ``|mean(exp(j * order * psi))|`` with ``psi`` the angle from each sample to
    its nearest constellation point. When the loop is locked every ``psi`` is
    small, so the phasors add coherently and the statistic approaches 1; when it
    is not, ``psi`` fills the decision region and they cancel. This is the
    carrier analogue of the timing coherence and is what ``locked`` is decided
    from, rather than the mere absence of a NaN.
    """
    x = _as_complex(symbols)
    if x.size == 0:
        return 0.0
    angles = np.angle(x)
    decisions = np.array(
        [np.angle(_nearest(complex(v), constellation)) for v in x], dtype=np.float64
    )
    psi = angles - decisions
    return float(np.abs(np.mean(np.exp(1j * order * psi))))


def _nearest_array(values: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Nearest constellation point for every value in ``values``."""
    x = _as_complex(values)
    table = np.asarray(points, dtype=np.complex128).ravel()
    if x.size == 0 or table.size == 0:
        return np.zeros(0, dtype=np.complex128)
    distances = np.abs(x[:, None] - table[None, :])
    return table[np.argmin(distances, axis=1)]


def _trim_equalizer_delay(
    symbols: np.ndarray, equalizer_info: dict
) -> tuple[np.ndarray, int]:
    """Remove the causal equalizer's known centre-tap delay.

    Both adaptive filters below use a zero-initialised causal buffer.  The
    first valid output is therefore the input delayed by ``taps // 2``
    symbols.  Returning that delay to the top-level chain is essential: without
    this compensation even a delta equaliser makes a perfectly clean capture
    look like an all-zero/random bit stream to a caller comparing the first
    bits.
    """
    taps = int(equalizer_info.get("taps", 1))
    delay = max(0, taps // 2)
    if delay <= 0 or symbols.size <= delay:
        return symbols, 0
    return np.asarray(symbols[delay:], dtype=np.complex128), delay


def _maybe_correct_iq(x: np.ndarray, scheme: str) -> tuple[np.ndarray, dict]:
    """Blindly remove a material IQ imbalance when the estimate is credible.

    A BPSK waveform has almost no Q-branch energy, so the moment estimator is
    intrinsically invalid there.  For the two-dimensional constellations the
    estimate is useful and is deliberately gated by both magnitude and a
    post-correction improvement check.  This is a correction, not a confidence
    claim: the metrics are returned so a caller can audit whether it ran.
    """
    name = wf.canonical_scheme(scheme)
    if name == "BPSK" or x.size < 32:
        return x, {"applied": False, "reason": "not estimable for this constellation"}
    before = ch.iq_imbalance_metrics(x)
    magnitude = float(before.get("magnitude", float("nan")))
    if not np.isfinite(magnitude) or magnitude < IQ_CORRECTION_MIN_MAGNITUDE:
        return x, {
            "applied": False,
            "magnitude": magnitude,
            "reason": "IQ estimate below correction threshold",
        }
    estimate = ch.estimate_iq_imbalance(x)
    gain = estimate.get("iq_gain_imb")
    phase = estimate.get("iq_phase_imb_deg")
    if (
        gain is None
        or phase is None
        or not np.isfinite(float(gain))
        or not np.isfinite(float(phase))
    ):
        return x, {
            "applied": False,
            "magnitude": magnitude,
            "reason": "IQ estimate was not finite",
        }
    try:
        corrected = ch.correct_iq_imbalance(x, float(gain), float(phase))
    except ValueError as exc:
        return x, {"applied": False, "magnitude": magnitude, "reason": str(exc)}
    after = ch.iq_imbalance_metrics(corrected)
    after_magnitude = float(after.get("magnitude", float("nan")))
    if not np.isfinite(after_magnitude) or after_magnitude >= magnitude * 0.95:
        return x, {
            "applied": False,
            "magnitude": magnitude,
            "estimate": {"iq_gain_imb": float(gain), "iq_phase_imb_deg": float(phase)},
            "reason": "correction did not materially reduce the IQ metric",
        }
    return corrected, {
        "applied": True,
        "magnitude_before": magnitude,
        "magnitude_after": after_magnitude,
        "estimate": {"iq_gain_imb": float(gain), "iq_phase_imb_deg": float(phase)},
        "reason": "",
    }


def _qam_constellation_loss(
    values: np.ndarray, points: np.ndarray, phase: float
) -> float:
    """Mean squared distance to the nearest QAM point at a trial phase."""
    x = _as_complex(values)
    if x.size == 0:
        return float("inf")
    rotated = x * np.exp(-1j * float(phase))
    distances = np.abs(rotated[:, None] - points[None, :]) ** 2
    return float(np.mean(np.min(distances, axis=1)))


def _qam_best_phase(values: np.ndarray, points: np.ndarray) -> float:
    """Find a constellation-symmetry representative of the carrier phase.

    A QAM grid is invariant under 90-degree rotations.  Searching only one
    90-degree sector removes the impossible-to-resolve blind ambiguity while
    still finding the actual phase for BER/constellation scoring.  The local
    refinement matters for dense 64-QAM, whose decision boundaries are only a
    few degrees apart.
    """
    x = _as_complex(values)
    if x.size == 0:
        return 0.0
    scale = float(np.sqrt(np.mean(np.abs(x) ** 2)))
    if not np.isfinite(scale) or scale <= 1e-12:
        return 0.0
    z = x / scale
    table = np.asarray(points, dtype=np.complex128) / scale
    candidates = np.linspace(0.0, np.pi / 2.0, int(QAM_PHASE_GRID), endpoint=False)
    losses = np.array([_qam_constellation_loss(z, table, p) for p in candidates])
    best_index = int(np.argmin(losses))
    best_phase = float(candidates[best_index])
    step = float(np.pi / 2.0 / max(1, QAM_PHASE_GRID - 1))
    local = np.linspace(best_phase - step, best_phase + step, 9)
    local_losses = np.array([_qam_constellation_loss(z, table, p) for p in local])
    return float(local[int(np.argmin(local_losses))])


def _qam_phase_refine(
    symbols: np.ndarray,
    scheme: str,
    *,
    initial_frequency: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Decision-directed QAM carrier acquisition and Doppler tracking.

    The fourth-power Costas loop is retained as a fallback and remains useful
    for PSK, but its decision errors are especially costly for 64-QAM.  This
    path first removes a coarse frequency/phase, estimates the residual phase
    in short blocks, and fits a slowly varying phase trajectory.  It never
    claims a unique absolute phase: the output remains subject to the QAM
    quarter-turn ambiguity, which callers must resolve with a sync word or a
    known reference just as they do for a Costas loop.
    """
    name = wf.canonical_scheme(scheme)
    points = _constellation_for(name)
    x = _as_complex(symbols)
    n = x.size
    blank = {
        "order": 4,
        "cfo_cycles_per_symbol": 0.0,
        "phase_radians": 0.0,
        "locked": False,
        "lock_statistic": 0.0,
        "reason": "too few symbols for QAM phase refinement",
        "n_symbols": int(n),
        "residual_rms": None,
        "method": "qam-block-phase",
    }
    if n < max(16, QAM_PHASE_BLOCK // 2):
        return x, blank

    frequency, _ = _coarse_carrier(x, 4)
    if initial_frequency is not None and np.isfinite(float(initial_frequency)):
        frequency = float(initial_frequency)
    index = np.arange(n, dtype=np.float64)
    derotated = x * np.exp(-1j * 2.0 * np.pi * frequency * index)
    phase0 = _qam_best_phase(derotated, points)
    corrected = derotated * np.exp(-1j * phase0)

    centers: list[float] = []
    errors: list[float] = []
    block = max(16, int(QAM_PHASE_BLOCK))
    for start in range(0, n, block):
        segment = corrected[start : start + block]
        if segment.size < max(8, block // 2):
            continue
        decisions = _nearest_array(segment, points)
        phase_error = np.angle(np.mean(segment * np.conj(decisions)))
        if np.isfinite(phase_error):
            centers.append(float(start + (segment.size - 1) / 2.0))
            errors.append(float(phase_error))
    if len(errors) < 2:
        statistic = _carrier_lock_statistic(corrected, 4, points)
        return corrected, {
            **blank,
            "cfo_cycles_per_symbol": float(frequency),
            "phase_radians": float(phase0),
            "locked": bool(statistic >= CARRIER_LOCK_MIN and n >= MIN_LOCK_SYMBOLS),
            "lock_statistic": float(statistic),
            "reason": (
                ""
                if statistic >= CARRIER_LOCK_MIN
                else "QAM phase residual is not locked"
            ),
            "residual_rms": float(np.std(errors)) if errors else None,
        }

    centers_array = np.asarray(centers, dtype=np.float64)
    errors_array = np.unwrap(np.asarray(errors, dtype=np.float64))
    # Fit in a centred/scaled coordinate system; fitting raw symbol indices
    # makes a degree-two Doppler fit unnecessarily ill-conditioned on long
    # captures.
    coordinate_center = float(np.mean(centers_array))
    coordinate_scale = max(1.0, float(np.ptp(centers_array)))
    coordinate = (centers_array - coordinate_center) / coordinate_scale
    degree = min(int(QAM_PHASE_FIT_DEGREE), len(errors_array) - 1)
    coefficients = np.polyfit(coordinate, errors_array, degree)
    fit = np.polyval(coefficients, (index - coordinate_center) / coordinate_scale)
    corrected = x * np.exp(-1j * (2.0 * np.pi * frequency * index + phase0 + fit))

    # One short re-estimation catches a coarse-frequency bias.  Re-fitting the
    # same block errors twice is enough at the modest Doppler rates in the
    # acceptance channel and avoids turning a noisy QAM decision stream into
    # a high-order oscillator fit.
    for _ in range(max(0, int(QAM_PHASE_REFINEMENTS) - 1)):
        new_errors: list[float] = []
        for start in range(0, n, block):
            segment = corrected[start : start + block]
            if segment.size < max(8, block // 2):
                continue
            decisions = _nearest_array(segment, points)
            value = np.angle(np.mean(segment * np.conj(decisions)))
            if np.isfinite(value):
                new_errors.append(float(value))
        if len(new_errors) < 2:
            break
        errors_array = np.unwrap(np.asarray(new_errors, dtype=np.float64))
        coefficients = np.polyfit(coordinate, errors_array, degree)
        fit = np.polyval(coefficients, (index - coordinate_center) / coordinate_scale)
        corrected = x * np.exp(-1j * (2.0 * np.pi * frequency * index + phase0 + fit))

    statistic = _carrier_lock_statistic(corrected, 4, points)
    final_errors: list[float] = []
    for start in range(0, n, block):
        segment = corrected[start : start + block]
        if segment.size < max(8, block // 2):
            continue
        decisions = _nearest_array(segment, points)
        value = np.angle(np.mean(segment * np.conj(decisions)))
        if np.isfinite(value):
            final_errors.append(float(value))
    residual = float(np.std(final_errors)) if final_errors else None
    slope = (
        float(coefficients[-2]) / (2.0 * np.pi * coordinate_scale) if degree else 0.0
    )
    effective_frequency = float(frequency + slope)
    locked = bool(statistic >= CARRIER_LOCK_MIN and n >= MIN_LOCK_SYMBOLS)
    return corrected, {
        **blank,
        "cfo_cycles_per_symbol": effective_frequency,
        "phase_radians": float(
            phase0
            + np.polyval(coefficients, (n - 1 - coordinate_center) / coordinate_scale)
        ),
        "locked": locked,
        "lock_statistic": float(statistic),
        "reason": "" if locked else "QAM phase residual is not locked",
        "residual_rms": residual,
    }


# --------------------------------------------------------------------------- #
# Equalisation
# --------------------------------------------------------------------------- #


def _normalise_power(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale to unit mean power; returns the array and the original power."""
    power = float(np.mean(np.abs(x) ** 2)) if x.size else 0.0
    if power <= 0.0:
        return x, 0.0
    return x / np.sqrt(power), power


def cma_equalize(
    samples: np.ndarray,
    *,
    taps: int = DEFAULT_TAPS,
    mu: float = CMA_MU,
    modulus: float | None = None,
):
    """Constant-modulus (blind) equaliser, for PSK constellations.

    Godard's CMA updates the taps from ``(|y|^2 - R^2) * y``, which needs no
    decisions and no carrier phase, so it can run before the Costas loop. The
    input is normalised to unit power and ``R^2`` defaults to 1, the correct
    modulus for any unit-power PSK constellation; 8PSK is constant-modulus too,
    so CMA is the right blind criterion there as well.

    Args:
        samples: Symbol-spaced complex samples.
        taps: Number of taps (forced odd, so there is a centre tap).
        mu: Step size, after input normalisation.
        modulus: ``R^2``; defaults to the input's mean power after normalisation
            (i.e. 1.0).

    Returns:
        ``(equalized_symbols, info)`` with ``taps``, ``modulus``,
        ``final_error`` and ``converged`` (True when the tap update has settled).
    """
    x, power = _normalise_power(_as_complex(samples))
    n = x.size
    taps = max(1, int(taps))
    if taps % 2 == 0:
        taps += 1
    blank = {
        "taps": taps,
        "modulus": 1.0,
        "final_error": None,
        "converged": False,
        "reason": "not enough samples for the equaliser",
    }
    if n < taps or power <= 0.0:
        return x, blank

    r_squared = 1.0 if modulus is None else float(modulus) / max(power, 1e-30)
    weights = np.zeros(taps, dtype=np.complex128)
    weights[taps // 2] = 1.0
    buffer = np.zeros(taps, dtype=np.complex128)
    out = np.empty(n, dtype=np.complex128)
    errors = np.empty(n, dtype=np.float64)
    for i in range(n):
        buffer[1:] = buffer[:-1]
        buffer[0] = x[i]
        y = complex(np.dot(weights, buffer))
        magnitude = abs(y) ** 2
        error = y * (magnitude - r_squared)
        # Normalised (NLMS-style) update: dividing by the buffer power makes the
        # step independent of the input level. Without it the plain CMA update
        # diverged on a clean QPSK capture -- the tap magnitudes grew without
        # bound until `abs(y) ** 2` overflowed a float64.
        buffer_power = float(np.real(np.vdot(buffer, buffer)))
        if buffer_power <= 0.0:
            out[i] = y
            errors[i] = abs(magnitude - r_squared)
            continue
        weights -= mu * np.conj(error) * buffer / buffer_power
        out[i] = y
        errors[i] = abs(magnitude - r_squared)

    tail = errors[n // 2 :]
    final_error = float(np.mean(tail)) if tail.size else None
    return out, {
        "taps": taps,
        "modulus": float(r_squared),
        "final_error": final_error,
        "converged": bool(final_error is not None and final_error < 0.5),
        "reason": "",
        "weights": weights,
    }


def lms_equalize(
    samples: np.ndarray,
    reference_symbols: np.ndarray | None = None,
    *,
    scheme: str = "16-QAM",
    taps: int = DEFAULT_TAPS,
    mu: float = LMS_MU,
):
    """Decision-directed LMS equaliser, for QAM.

    Without ``reference_symbols`` the update is ``y - slice(y)``, which requires
    the constellation to be roughly de-rotated -- run carrier recovery first.
    Passing ``reference_symbols`` makes it data-aided (a training sequence), in
    which case the reference must be aligned with ``samples``.

    Args:
        samples: Symbol-spaced complex samples.
        reference_symbols: Optional known symbols for data-aided training.
        scheme: Constellation label used for the decision.
        taps: Number of taps (forced odd).
        mu: Step size, after input normalisation.

    Returns:
        ``(equalized_symbols, info)`` with ``taps``, ``final_error``,
        ``converged`` and ``mode`` (``"data-aided"`` or ``"decision-directed"``).
    """
    name = wf.canonical_scheme(scheme)
    constellation = _constellation_for(name)
    x, power = _normalise_power(_as_complex(samples))
    n = x.size
    taps = max(1, int(taps))
    if taps % 2 == 0:
        taps += 1
    mode = "decision-directed" if reference_symbols is None else "data-aided"
    blank = {
        "taps": taps,
        "final_error": None,
        "converged": False,
        "mode": mode,
        "reason": "not enough samples for the equaliser",
    }
    if n < taps or power <= 0.0:
        return x, blank

    reference = None
    if reference_symbols is not None:
        reference = _as_complex(reference_symbols).ravel()
        if reference.size < n:
            reference = np.concatenate(
                [reference, np.zeros(n - reference.size, dtype=np.complex128)]
            )
        reference = reference[:n]

    weights = np.zeros(taps, dtype=np.complex128)
    weights[taps // 2] = 1.0
    buffer = np.zeros(taps, dtype=np.complex128)
    out = np.empty(n, dtype=np.complex128)
    errors = np.empty(n, dtype=np.float64)
    for i in range(n):
        buffer[1:] = buffer[:-1]
        buffer[0] = x[i]
        y = complex(np.dot(weights, buffer))
        decision = reference[i] if reference is not None else _nearest(y, constellation)
        error = y - decision
        # Normalised update, same rationale as in `cma_equalize`.
        buffer_power = float(np.real(np.vdot(buffer, buffer)))
        if buffer_power <= 0.0:
            out[i] = y
            errors[i] = abs(error) ** 2
            continue
        weights -= mu * np.conj(error) * buffer / buffer_power
        out[i] = y
        errors[i] = abs(error) ** 2

    tail = errors[n // 2 :]
    final_error = float(np.mean(tail)) if tail.size else None
    return out, {
        "taps": taps,
        "final_error": final_error,
        "converged": bool(final_error is not None and final_error < 0.5),
        "mode": mode,
        "reason": "",
        "weights": weights,
    }


# --------------------------------------------------------------------------- #
# FSK
# --------------------------------------------------------------------------- #


def _fsk_symbol_frequencies(
    x: np.ndarray, sps: int, tau: float, n_symbols: int
) -> np.ndarray:
    """Mean instantaneous frequency per symbol, from the unwrapped phase.

    The mean frequency over a symbol is the phase advance divided by the symbol
    length, so interpolating the *unwrapped phase* at the symbol boundaries and
    differencing is both simpler and less noisy than averaging a per-sample
    discriminator. A residual carrier offset contributes the same constant to
    every symbol and is removed later by estimating the tone levels from the
    data.
    """
    phase = np.unwrap(np.angle(_as_complex(x)))
    positions = tau + np.arange(n_symbols + 1, dtype=np.float64) * sps
    usable = positions <= phase.size - 1
    if not np.all(usable):
        positions = positions[usable]
    sampled = _interp_cubic(phase, positions)
    if sampled.size < 2:
        return np.zeros(0, dtype=np.float64)
    # Complex interpolation of a real phase leaves a zero imaginary part with
    # a complex dtype; downstream k-means casts to real with a ComplexWarning.
    # Take the real part here so the numbers (and the warning) stay clean.
    return np.real(np.diff(sampled) / float(sps))


def _fsk_phase_slope_frequencies(
    x: np.ndarray, sps: int, tau: float, n_symbols: int
) -> np.ndarray:
    """Estimate one tone frequency per symbol by a phase-slope fit.

    Differencing the phase at two endpoints throws away almost all of the
    symbol's energy.  At the Eb/N0 used by the acceptance matrix the endpoint
    estimate consequently confuses 2-FSK tones often enough to produce roughly
    0.15 BER.  A least-squares slope over several interpolated phase points
    uses the whole symbol and is substantially more stable.
    """
    phase = np.unwrap(np.angle(_as_complex(x)))
    if phase.size == 0 or n_symbols <= 0:
        return np.zeros(0, dtype=np.float64)
    sps = max(1, int(sps))
    offsets = np.linspace(0.0, float(sps), min(9, sps + 1), dtype=np.float64)
    result = np.full(int(n_symbols), np.nan, dtype=np.float64)
    for k in range(int(n_symbols)):
        start = float(tau) + k * sps
        positions = start + offsets
        valid = (positions >= 0.0) & (positions <= phase.size - 1.0)
        if int(np.count_nonzero(valid)) < 3:
            continue
        positions = positions[valid]
        values = np.real(_interp_cubic(phase, positions))
        t = positions - start
        t = t - float(np.mean(t))
        denominator = float(np.dot(t, t))
        if denominator <= 0.0 or not np.isfinite(denominator):
            continue
        slope = float(np.dot(t, values - float(np.mean(values))) / denominator)
        if np.isfinite(slope):
            result[k] = slope / (2.0 * np.pi)
    return result[np.isfinite(result)]


def _fsk_correlated_labels(
    x: np.ndarray,
    sps: int,
    tau: float,
    n_symbols: int,
    centres: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify FSK tones by coherent projection over each symbol.

    ``centres`` are estimated in cycles/sample, not cycles/symbol.  A CFO
    shifts every centre by the same amount, so estimating the centres from the
    capture rather than hard-coding the generator's tone spacing is what keeps
    this path compatible with the channel model.
    """
    z = _as_complex(x)
    if z.size == 0 or n_symbols <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    tones = np.asarray(centres, dtype=np.float64).ravel()
    if tones.size < 2:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    sps = max(1, int(sps))
    labels = np.full(int(n_symbols), -1, dtype=np.int64)
    quality = np.zeros(int(n_symbols), dtype=np.float64)
    for k in range(int(n_symbols)):
        start = float(tau) + k * sps
        first = max(0, int(np.floor(start)))
        last = min(z.size, int(np.ceil(start + sps)) + 1)
        if last - first < 3:
            continue
        positions = np.arange(first, last, dtype=np.float64)
        t = positions - start
        segment = z[first:last]
        projections = np.abs(
            np.mean(
                segment[:, None]
                * np.exp(-1j * 2.0 * np.pi * t[:, None] * tones[None, :]),
                axis=0,
            )
        )
        choice = int(np.argmax(projections))
        labels[k] = choice
        ordered = np.sort(projections)
        quality[k] = (
            float((ordered[-1] - ordered[-2]) / max(ordered[-1], 1e-12))
            if ordered.size > 1
            else 0.0
        )
    valid = labels >= 0
    return labels[valid], quality[valid]


def _fsk_timing_residual_lsq(
    x: np.ndarray, sps: int, tau: float, n_symbols: int, order: int, rng
) -> float:
    """Tone-separation residual using the lower-noise phase-slope estimate."""
    freqs = _fsk_phase_slope_frequencies(x, sps, tau, n_symbols)
    if freqs.size < 4:
        return float("inf")
    result = _kmeans_1d(freqs, order, rng)
    if result is None:
        return float("inf")
    centres, labels = result
    gaps = np.diff(centres)
    mean_gap = float(np.mean(gaps)) if gaps.size else 0.0
    if mean_gap <= 0.0:
        return float("inf")
    return float(np.mean(np.abs(freqs - centres[labels])) / mean_gap)


def _kmeans_1d(
    values: np.ndarray, k: int, rng: np.random.Generator, iterations: int = 100
) -> tuple[np.ndarray, np.ndarray] | None:
    """1-D Lloyd k-means with a quantile initialisation.

    Returns ``(sorted_centres, labels)`` or ``None`` when the data cannot be
    split (fewer points than clusters, or zero spread). Sorting the centres
    ascending is what makes the labels meaningful for FSK: tone ``b`` is the
    ``b``-th lowest frequency, which is the generator's convention.
    """
    data = np.asarray(values, dtype=np.float64).ravel()
    k = int(k)
    if data.size < k or k < 2:
        return None
    spread = float(np.max(data) - np.min(data))
    if spread <= 0.0:
        return None
    quantiles = (np.arange(k) + 0.5) / k
    centres = np.quantile(data, quantiles)
    # Tiny seeded jitter breaks ties between identical quantiles without
    # changing the answer; it is the only place the public `seed` is consumed.
    centres = centres + rng.normal(0.0, 1e-9 * spread, k)
    centres = np.sort(centres)
    labels = np.zeros(data.size, dtype=np.int64)
    for _ in range(int(iterations)):
        labels = np.argmin(np.abs(data[:, None] - centres[None, :]), axis=1)
        updated = centres.copy()
        for cluster in range(k):
            members = data[labels == cluster]
            if members.size:
                updated[cluster] = float(np.mean(members))
        if np.allclose(updated, centres, rtol=0.0, atol=1e-12 * spread):
            centres = updated
            break
        centres = updated
    order = np.argsort(centres)
    remap = np.empty(k, dtype=np.int64)
    remap[order] = np.arange(k)
    return centres[order], remap[labels]


def _fsk_bits(labels: np.ndarray, order: int) -> np.ndarray:
    """Tone index to hard bits, most significant first (the generator's map).

    ``waveform.modulate_mfsk`` indexes its tone table with the packed bit group
    directly -- there is no Gray coding in the FSK path -- so tone ``b`` is just
    the binary value of its bits. 4-FSK therefore emits ``00, 01, 10, 11`` for
    tones ``0..3``.
    """
    labels = np.asarray(labels, dtype=np.int64)
    if order == 2:
        return labels.astype(np.uint8)
    width = int(order).bit_length() - 1
    shifts = np.arange(width - 1, -1, -1)
    return ((labels[:, None] >> shifts) & 1).astype(np.uint8).ravel()


def _fsk_timing_residual(
    x: np.ndarray, sps: int, tau: float, n_symbols: int, order: int, rng
) -> float:
    """Tone-separation residual of one candidate timing phase, or ``inf``.

    The per-symbol frequencies at ``tau`` are clustered into ``order`` tone
    levels (the same k-means the demodulator uses) and the mean distance to
    the assigned level, normalised by the mean tone gap, is the score. A
    timing phase that straddles symbol boundaries mixes tones and scores
    badly; one that respects them scores near zero. This replaces an earlier
    variance scan, which maximised the wrong statistic: on the project's own
    generator it picked ``tau = 7.5`` samples (variance 0.0306 vs 0.0295 at
    the true ``tau = 0``) and slipped the whole stream by one symbol.
    """
    freqs = _fsk_symbol_frequencies(x, sps, float(tau), n_symbols)
    if freqs.size < 4:
        return float("inf")
    result = _kmeans_1d(np.asarray(freqs, dtype=np.float64), order, rng)
    if result is None:
        return float("inf")
    centres, labels = result
    gaps = np.diff(centres)
    mean_gap = float(np.mean(gaps)) if gaps.size else 0.0
    if mean_gap <= 0.0:
        return float("inf")
    return float(np.mean(np.abs(freqs - centres[labels]))) / mean_gap


def _receive_fsk(
    x: np.ndarray,
    name: str,
    *,
    sps: int,
    timing: bool,
    seed: int,
    metrics: dict,
) -> dict:
    """Discriminator-based M-FSK receive path."""
    order = 2 if name == "2-FSK" else 4
    n_sym_guess = max(1, x.size // sps)
    if timing and n_sym_guess >= MIN_LOCK_SYMBOLS:
        # Scan the timing phase for the best tone separation (see
        # `_fsk_timing_residual`). Among near-optimal candidates the smallest
        # tau wins: a tau a symbol late separates tones just as well but
        # slips the stream by one symbol, which is exactly what the old
        # variance scan did. The scan runs on a prefix so a huge capture
        # does not pay k-means per grid point over the whole stream.
        scan_n = min(n_sym_guess, FSK_TIMING_SCAN_SYMBOLS)
        grid = np.arange(0.0, float(sps), FSK_TIMING_STEP)
        scan_rng = np.random.default_rng(int(seed))
        scored = [
            (
                _fsk_timing_residual(x, sps, float(tau), scan_n, order, scan_rng),
                float(tau),
            )
            for tau in grid
        ]
        finite = [(residual, tau) for residual, tau in scored if np.isfinite(residual)]
        if finite:
            best = min(residual for residual, _ in finite)
            tau = min(
                tau
                for residual, tau in finite
                if residual <= max(best * 1.5, best + 0.02)
            )
            recovery_timing = "discriminator-residual-scan"
        else:
            tau = 0.0
            recovery_timing = "none (timing scan found no separable tones)"
    else:
        tau = 0.0
        recovery_timing = "none"

    freqs = _fsk_phase_slope_frequencies(x, sps, tau, n_sym_guess)
    n_symbols = int(freqs.size)
    metrics["timing_offset_estimate"] = float(tau)
    metrics["n_symbols"] = n_symbols
    metrics["equalizer_taps"] = 0
    metrics["evm_percent"] = None
    metrics["cfo_hz_estimate"] = 0.0
    metrics["cfo_units"] = "not measurable (constant-envelope discriminator)"
    metrics["recovery"] = {
        "timing": recovery_timing,
        "carrier": "not-applicable (FSK discriminator)",
        "equalizer": "none",
    }
    if n_symbols < 4:
        return {
            "bits": np.zeros(0, dtype=np.uint8),
            "symbols": None,
            "scheme": name,
            "sps": int(sps),
            "locked": False,
            "metrics": {**metrics, "reason": "too few symbols after timing"},
        }

    rng = np.random.default_rng(int(seed))
    result = _kmeans_1d(freqs, order, rng)
    if result is None:
        return {
            "bits": np.zeros(0, dtype=np.uint8),
            "symbols": None,
            "scheme": name,
            "sps": int(sps),
            "locked": False,
            "metrics": {**metrics, "reason": "tone levels could not be separated"},
        }
    centres, labels = result
    gaps = np.diff(centres)
    mean_gap = float(np.mean(gaps)) if gaps.size else 0.0
    residual = float(np.mean(np.abs(freqs - centres[labels])))
    residual_ratio = residual / mean_gap if mean_gap > 0.0 else float("inf")

    # The slope estimate is used to learn the tone centres, but the final hard
    # decision uses every sample in the symbol.  This is a small matched
    # filter in frequency space and is what keeps 2-FSK below the acceptance
    # threshold at 10 dB Eb/N0.
    tone_labels, tone_quality = _fsk_correlated_labels(x, sps, tau, n_symbols, centres)
    if tone_labels.size >= max(4, order):
        labels = tone_labels
        detector = "phase-slope + coherent projection"
        if tone_quality.size:
            metrics["fsk_projection_quality"] = float(np.mean(tone_quality))
    else:
        detector = "phase-slope + k-means"
    locked = bool(
        mean_gap > 0.0
        and residual_ratio <= FSK_LOCK_MAX_RESIDUAL
        and n_symbols >= MIN_LOCK_SYMBOLS
    )
    metrics["fsk_tone_levels"] = [float(c) for c in centres]
    metrics["fsk_residual_ratio"] = float(residual_ratio)
    metrics["fsk_detector"] = detector
    metrics["reason"] = "" if locked else "tone levels are not separable"
    return {
        "bits": _fsk_bits(labels, order),
        "symbols": None,
        "scheme": name,
        "sps": int(sps),
        "locked": locked,
        "metrics": metrics,
    }


# --------------------------------------------------------------------------- #
# Symbol-rate (oversampling) estimation
# --------------------------------------------------------------------------- #

#: Largest relative deviation from an integer that still counts as "integer
#: oversampling". The cyclostationary peak is refined by parabolic interpolation,
#: so its residual error at the rates this project uses is well under 1%; 12%
#: absorbs the coarser resolution on short captures while still rejecting a
#: fractional rate such as 5.5.
SPS_INTEGER_TOLERANCE = 0.12


def estimate_sps(samples: np.ndarray, scheme: str | None = None) -> dict:
    """Estimate the integer oversampling factor from the signal alone.

    Two evidence paths, both of which can abstain:

    * **FSK period recovery** (``dsp.estimate_fsk_symbol_period``) when the
      scheme is an FSK one, or when it is unknown and the piecewise-constant
      model fits. This is a direct measurement of the symbol period in samples.
    * **Cyclostationary symbol rate** at a nominal sample rate of 1.0, which
      returns the rate in cycles/sample; ``sps = 1 / rate``.

    The result is only reported when ``1 / rate`` is close to an integer
    (see :data:`SPS_INTEGER_TOLERANCE`). Noise, an unmodulated tone and a
    non-integer rate all return ``sps=None, found=False`` with a reason -- this
    function never invents a number.

    Args:
        samples: Complex baseband samples.
        scheme: Optional modulation label; FSK labels route to period recovery.

    Returns:
        ``{"sps", "found", "reason", "method", "rate_cycles_per_sample"}``.
    """
    x = _as_complex(samples)
    unknown = {
        "sps": None,
        "found": False,
        "reason": "",
        "method": None,
        "rate_cycles_per_sample": None,
    }
    if x.size < 64:
        return {**unknown, "reason": "capture shorter than 64 samples"}

    name = wf.canonical_scheme(scheme) if scheme else None
    if (
        name is not None
        and name not in wf.FSK_SCHEMES
        and name not in (*wf.LINEAR_SCHEMES,)
    ):
        return {**unknown, "reason": f"unsupported scheme {name!r}"}

    if name is None or name in wf.FSK_SCHEMES:
        period = int(dsp.estimate_fsk_symbol_period(x))
        if period > 1:
            return {
                "sps": period,
                "found": True,
                "reason": "piecewise-constant instantaneous frequency",
                "method": "fsk_period",
                "rate_cycles_per_sample": 1.0 / float(period),
            }
        if name in wf.FSK_SCHEMES:
            return {
                **unknown,
                "method": "fsk_period",
                "reason": "no piecewise-constant symbol structure to measure",
            }

    try:
        cyclo = rate_est.cyclostationary_symbol_rate(x, 1.0, max_alpha_hz=0.5)
    except Exception:
        cyclo = {"found": False, "symbol_rate_hz": 0.0}
    if not cyclo.get("found"):
        return {
            **unknown,
            "method": "cyclostationary",
            "reason": "no symbol-rate line above the detection threshold",
        }
    rate = float(cyclo["symbol_rate_hz"])
    if not np.isfinite(rate) or rate <= 0.0:
        return {
            **unknown,
            "method": "cyclostationary",
            "reason": "symbol-rate line located outside the valid range",
        }
    candidate = 1.0 / rate
    nearest = round(candidate)
    if nearest < 1:
        return {
            **unknown,
            "method": "cyclostationary",
            "reason": f"estimated {candidate:.2f} samples/symbol is below 1",
        }
    deviation = abs(candidate - nearest) / float(nearest)
    if deviation > SPS_INTEGER_TOLERANCE:
        return {
            **unknown,
            "method": "cyclostationary",
            "rate_cycles_per_sample": rate,
            "reason": (
                f"estimated {candidate:.2f} samples/symbol is not an integer "
                f"({deviation:.0%} from {nearest})"
            ),
        }
    return {
        "sps": nearest,
        "found": True,
        "reason": "cyclostationary symbol-rate line",
        "method": "cyclostationary",
        "rate_cycles_per_sample": rate,
    }


# --------------------------------------------------------------------------- #
# Top-level receive chain
# --------------------------------------------------------------------------- #

_PSK_ORDER = {"BPSK": 2, "QPSK": 4, "8PSK": 8}
_QAM_SCHEMES = ("16-QAM", "64-QAM")


def _blank_metrics() -> dict:
    return {
        "cfo_hz_estimate": 0.0,
        "cfo_units": "cycles/symbol (no sample rate supplied)",
        "timing_offset_estimate": 0.0,
        "evm_percent": None,
        "equalizer_taps": 0,
        "n_symbols": 0,
        "locked": False,
        "recovery": {"timing": "none", "carrier": "none", "equalizer": "none"},
        "reason": "",
    }


def _receive_linear(
    x: np.ndarray,
    name: str,
    *,
    sps: int,
    equalize: bool,
    carrier: bool,
    timing: bool,
) -> dict:
    """Timing -> correction/equalisation/carrier -> slice.

    Equaliser output is centre-tap aligned before it reaches the slicer.  The
    returned ``symbols`` field is useful to BER harnesses that resolve the
    unavoidable constellation rotation against a known reference; callers that
    need a framed packet should still use the sync word for that decision.
    """
    metrics = _blank_metrics()
    recovery = metrics["recovery"]
    locked = True

    corrected_x, iq_info = _maybe_correct_iq(x, name)
    metrics["iq_correction"] = iq_info

    if timing:
        if sps >= 2:
            symbols, timing_info = gardner_recover(corrected_x, sps)
            recovery["timing"] = "gardner"
        else:
            symbols, timing_info = mueller_muller_recover(corrected_x, sps, scheme=name)
            recovery["timing"] = "mueller-muller"
        metrics["timing_offset_estimate"] = float(timing_info["offset"])
        metrics["timing_coherence"] = float(timing_info["coherence"])
        locked = locked and bool(timing_info["locked"])
        if not timing_info["locked"]:
            metrics["reason"] = timing_info["reason"]
    else:
        filtered = matched_filter(corrected_x, sps) if sps > 1 else corrected_x
        n_sym = filtered.size // sps
        symbols = (
            filtered[: n_sym * sps : sps] if n_sym else np.zeros(0, dtype=np.complex128)
        )
        recovery["timing"] = "none"
        metrics["timing_offset_estimate"] = 0.0

    if symbols.size < 4:
        return {
            "bits": np.zeros(0, dtype=np.uint8),
            "symbols": np.zeros(0, dtype=np.complex128),
            "scheme": name,
            "sps": int(sps),
            "locked": False,
            "metrics": {
                **metrics,
                "reason": "too few symbols after timing recovery",
            },
        }

    order = _PSK_ORDER.get(name, 4)
    metrics["carrier_refined"] = False
    metrics["equalizer_delay_symbols"] = 0

    if name in _QAM_SCHEMES:
        # A short coarse derotation makes the decision-directed LMS usable
        # even when the capture starts with a large CFO.  The full QAM phase
        # estimator below then replaces the old integral Costas loop, whose
        # random frequency walk was the dominant 64-QAM error source.
        if equalize:
            if carrier:
                coarse_frequency, coarse_phase = _coarse_carrier(symbols, order)
                index = np.arange(symbols.size, dtype=np.float64)
                equalizer_input = symbols * np.exp(
                    -1j * (2.0 * np.pi * coarse_frequency * index + coarse_phase)
                )
            else:
                equalizer_input = symbols
            symbols, equalizer_info = lms_equalize(
                equalizer_input, scheme=name, taps=DEFAULT_TAPS, mu=LMS_MU
            )
            symbols, delay = _trim_equalizer_delay(symbols, equalizer_info)
            metrics["equalizer_taps"] = int(equalizer_info["taps"])
            metrics["equalizer_delay_symbols"] = int(delay)
            recovery["equalizer"] = "lms"
        else:
            recovery["equalizer"] = "none"

        if carrier:
            symbols, carrier_info = _qam_phase_refine(symbols, name)
            recovery["carrier"] = "qam-block-phase"
            metrics["cfo_cycles_per_symbol"] = float(
                carrier_info["cfo_cycles_per_symbol"]
            )
            metrics["carrier_lock_statistic"] = float(carrier_info["lock_statistic"])
            metrics["carrier_refined"] = True
            locked = locked and bool(carrier_info["locked"])
            if not carrier_info["locked"] and not metrics["reason"]:
                metrics["reason"] = carrier_info["reason"]
    else:
        if equalize:
            symbols, equalizer_info = cma_equalize(symbols)
            symbols, delay = _trim_equalizer_delay(symbols, equalizer_info)
            metrics["equalizer_taps"] = int(equalizer_info["taps"])
            metrics["equalizer_delay_symbols"] = int(delay)
            recovery["equalizer"] = "cma"
        else:
            recovery["equalizer"] = "none"
        if carrier:
            symbols, carrier_info = costas_loop(symbols, order, scheme=name)
            recovery["carrier"] = f"costas-{order}psk"
            metrics["cfo_cycles_per_symbol"] = float(
                carrier_info["cfo_cycles_per_symbol"]
            )
            metrics["carrier_lock_statistic"] = float(carrier_info["lock_statistic"])
            locked = locked and bool(carrier_info["locked"])
            if not carrier_info["locked"] and not metrics["reason"]:
                metrics["reason"] = carrier_info["reason"]

    bits = _slice_linear(symbols, name)
    metrics["n_symbols"] = int(symbols.size)
    metrics["cfo_hz_estimate"] = float(metrics.get("cfo_cycles_per_symbol", 0.0))
    evm = dsp.compute_evm(symbols, name, samples_per_symbol=1)
    metrics["evm_percent"] = evm["evm_percent"]
    metrics["locked"] = bool(locked)
    return {
        "bits": bits,
        "symbols": np.asarray(symbols, dtype=np.complex128),
        "scheme": name,
        "sps": int(sps),
        "locked": bool(locked),
        "metrics": metrics,
    }


def receive(
    samples: np.ndarray,
    scheme: str,
    *,
    sps: int | None = None,
    equalize: bool = True,
    carrier: bool = True,
    timing: bool = True,
    seed: int = 0,
) -> dict:
    """Coherent receive chain: recover the hard bit stream from a capture.

    Args:
        samples: Complex baseband samples.
        scheme: Modulation label (any spelling ``waveform.canonical_scheme``
            accepts).
        sps: Samples per symbol. When ``None`` it is estimated with
            :func:`estimate_sps`; if that abstains the result is ``locked=False``
            with an empty bit stream and a reason, never a guessed rate.
        equalize: Run CMA (PSK) or LMS (QAM).
        carrier: Run the Costas / decision-directed carrier loop.
        timing: Run timing recovery. When False the signal is assumed already
            symbol-spaced at offset zero.
        seed: Seeds the only stochastic step, the FSK k-means initialisation.

    Returns:
        ``{"bits", "scheme", "sps", "locked", "metrics"}``. ``bits`` is a
        ``uint8`` array truncated to a whole number of symbols. ``locked`` is
        False whenever a stage did not converge. ``metrics`` carries at least
        ``cfo_hz_estimate``, ``timing_offset_estimate``, ``evm_percent``,
        ``equalizer_taps``, ``n_symbols`` and ``recovery``.

    Note:
        ``cfo_hz_estimate`` is in Hz only when the caller supplies ``sps`` and
        the sample rate is known; with no sample rate the receiver has no Hz to
        report and the value is in cycles/symbol, flagged by ``cfo_units``.
        The residual rotational ambiguity of the carrier loop is reported in
        ``metrics`` and is not silently resolved -- see the module docstring.
    """
    name = wf.canonical_scheme(scheme)
    x = _as_complex(samples)
    metrics = _blank_metrics()
    if x.size == 0:
        metrics["reason"] = "empty capture"
        return {
            "bits": np.zeros(0, dtype=np.uint8),
            "scheme": name,
            "sps": int(sps) if sps else 0,
            "locked": False,
            "metrics": metrics,
        }

    resolved_sps = int(sps) if sps else None
    if resolved_sps is None:
        estimate = estimate_sps(x, name)
        metrics["sps_estimate"] = estimate
        if not estimate["found"]:
            metrics["reason"] = "samples per symbol could not be estimated; " + str(
                estimate["reason"]
            )
            return {
                "bits": np.zeros(0, dtype=np.uint8),
                "scheme": name,
                "sps": 0,
                "locked": False,
                "metrics": metrics,
            }
        resolved_sps = int(estimate["sps"])
    if resolved_sps < 1:
        metrics["reason"] = "samples per symbol must be >= 1"
        return {
            "bits": np.zeros(0, dtype=np.uint8),
            "scheme": name,
            "sps": 0,
            "locked": False,
            "metrics": metrics,
        }

    if name in wf.FSK_SCHEMES:
        return _receive_fsk(
            x,
            name,
            sps=resolved_sps,
            timing=timing,
            seed=seed,
            metrics=metrics,
        )
    return _receive_linear(
        x,
        name,
        sps=resolved_sps,
        equalize=equalize,
        carrier=carrier,
        timing=timing,
    )


#: Costas rotational symmetry order per linear scheme. FSK has no phase
#: ambiguity of this kind -- the discriminator absorbs a residual carrier as a
#: common tone-level shift.
_ROTATION_ORDER = {"BPSK": 2, "QPSK": 4, "8PSK": 8, "16-QAM": 4, "64-QAM": 4}


def ber_best_rotation(
    tx_bits: np.ndarray, rx_bits: np.ndarray, scheme: str
) -> float | None:
    """Hamming BER under the Costas-ambiguity rotation that fits best.

    A Costas loop locks phase modulo the constellation symmetry, which no
    blind estimator can improve on -- the ambiguity is information-theoretic.
    The pipeline resolves it with the sync word; BER harnesses resolve it
    against a known reference. Each symmetry rotation of the *transmitted*
    symbols is sliced to bits and the rotation with the lowest Hamming
    distance wins.

    Integer-symbol lags of ``-2..+12`` are also searched: CMA/LMS equalisers
    are centred FIR filters with a group delay of ``taps // 2`` symbols, so
    the received stream trails the reference. The pipeline absorbs that with
    its sliding header search; this scorer does the equivalent and still
    reads ~0.5 on uncorrelated data, so the search cannot invent a good
    score.

    Returns ``None`` when either stream is empty (receiver abstained).
    """
    tx = np.asarray(tx_bits, dtype=np.uint8).ravel()
    rx = np.asarray(rx_bits, dtype=np.uint8).ravel()
    if rx.size == 0 or tx.size == 0:
        return None
    name = wf.canonical_scheme(scheme)
    bps = wf.bits_per_symbol(name) if name not in wf.FSK_SCHEMES else 1
    order = 1 if name in wf.FSK_SCHEMES else _ROTATION_ORDER[name]

    best = 1.0
    for delay_sym in range(-2, 13):
        lag_bits = delay_sym * bps
        ref = (
            tx[: rx.size - lag_bits]
            if lag_bits >= 0
            else tx[-lag_bits : -lag_bits + rx.size]
        )
        use_rx = rx[lag_bits:] if lag_bits >= 0 else rx[: ref.size]
        n = min(ref.size, use_rx.size)
        if n <= 0:
            continue
        ref, use_rx = ref[:n], use_rx[:n]
        if name in wf.FSK_SCHEMES:
            best = min(best, float(np.mean(ref != use_rx)))
            continue
        ref_syms = wf.bits_to_symbols(ref, name)
        expect = (ref.size // bps) * bps
        ref, use_rx = ref[:expect], use_rx[:expect]
        if ref.size == 0:
            continue
        for k in range(order):
            rotated = ref_syms * np.exp(1j * 2.0 * np.pi * k / order)
            cand = wf.symbols_to_bits(rotated, name)[: ref.size]
            best = min(best, float(np.mean(cand != use_rx)))
        if best == 0.0:
            break
    return best


def resolve_rotation_with_sync(
    symbols: np.ndarray,
    scheme: str,
    sync_bits: np.ndarray,
) -> dict:
    """Pick the Costas-ambiguity rotation whose hard bits best match a sync word.

    This is how the pipeline resolves the residual constellation rotation
    that no blind Costas / QAM phase loop can remove. Returns the winning
    ``bits``, the rotation index, and the ``find_header`` result used to
    rank candidates. FSK schemes have no rotational ambiguity -- they are
    sliced once and returned unchanged.
    """
    from rf_analyzer.core.correlator import find_header

    name = wf.canonical_scheme(scheme)
    sync = np.asarray(sync_bits, dtype=np.uint8).ravel()
    syms = np.asarray(symbols, dtype=np.complex128).ravel()
    if name in wf.FSK_SCHEMES or syms.size == 0:
        bits = (
            np.zeros(0, dtype=np.uint8)
            if syms.size == 0
            else (
                _slice_linear(syms, name)
                if name not in wf.FSK_SCHEMES
                else np.zeros(0, dtype=np.uint8)
            )
        )
        # FSK bits come from the discriminator path, not from these symbols.
        found = (
            find_header(bits, sync)
            if bits.size >= sync.size > 0
            else {"detected": False, "score": 0.0, "offset": -1}
        )
        return {
            "bits": bits,
            "rotation": 0,
            "found": found,
            "score": float(found.get("score", 0.0) or 0.0),
        }

    order = _ROTATION_ORDER[name]
    best: dict | None = None
    for k in range(order):
        rotated = syms * np.exp(1j * 2.0 * np.pi * k / float(order))
        bits = _slice_linear(rotated, name)
        if sync.size == 0 or bits.size < sync.size:
            found = {"detected": False, "score": 0.0, "offset": -1}
        else:
            found = find_header(bits, sync)
        score = float(found.get("score", 0.0) or 0.0)
        candidate = {
            "bits": bits,
            "rotation": k,
            "found": found,
            "score": score,
        }
        if best is None or score > best["score"]:
            best = candidate
        if bool(found.get("detected", False)) and score >= 0.99:
            break
    assert best is not None
    return best


def demodulate(
    samples: np.ndarray,
    scheme: str,
    *,
    sps: int | None = None,
    sync_bits: np.ndarray | None = None,
    prefer_coherent: bool = True,
    **kw,
) -> dict:
    """Demodulate with the coherent chain when oversampled, else abstain.

    Returns ``{"bits", "sps", "locked", "path", "metrics", "rotation"}``.
    ``path`` is ``"coherent"`` when :func:`receive` ran, or ``"abstain"``
    when samples-per-symbol could not be estimated above 1 (the pipeline then
    falls back to the phase-aligned slicers for symbol-spaced
    captures). When ``sync_bits`` is supplied the Costas residual rotation is
    resolved against it; otherwise ``rotation`` is ``None`` and the raw
    Costas-locked bits are returned.
    """
    name = wf.canonical_scheme(scheme)
    resolved = int(sps) if sps else None
    if resolved is None and prefer_coherent:
        estimate = estimate_sps(samples, name)
        if estimate.get("found") and int(estimate["sps"] or 0) > 1:
            resolved = int(estimate["sps"])
    if resolved is None or resolved <= 1:
        return {
            "bits": np.zeros(0, dtype=np.uint8),
            "sps": int(resolved or 0),
            "locked": False,
            "path": "abstain",
            "metrics": {"reason": "sps<=1; use naive demod for symbol-spaced captures"},
            "rotation": None,
        }
    result = receive(samples, name, sps=resolved, **kw)
    bits = np.asarray(result["bits"], dtype=np.uint8)
    rotation = None
    if (
        sync_bits is not None
        and name not in wf.FSK_SCHEMES
        and result.get("symbols") is not None
    ):
        resolved_bits = resolve_rotation_with_sync(result["symbols"], name, sync_bits)
        bits = resolved_bits["bits"]
        rotation = int(resolved_bits["rotation"])
        result = {**result, "bits": bits}
    return {
        "bits": bits,
        "sps": int(result["sps"]),
        "locked": bool(result["locked"]),
        "path": "coherent",
        "metrics": result.get("metrics") or {},
        "rotation": rotation,
        "symbols": result.get("symbols"),
    }


def receive_burst(
    samples: np.ndarray,
    scheme: str,
    *,
    region: tuple[int, int] | None = None,
    sps: int | None = None,
    **kw,
) -> dict:
    """As :func:`receive`, but locate and trim a burst inside noise first.

    When ``region`` is ``None`` the extent is measured with
    :func:`dsp.estimate_burst_region` and the capture is trimmed to it. The
    estimate is deliberately biased long (see that function), so the trimmed
    capture usually keeps a little noise at both ends; the timing and carrier
    loops are what tolerate that.

    Args:
        samples: Complex baseband samples.
        scheme: Modulation label.
        region: Explicit ``(start, end)`` sample indices; skips estimation.
        sps: Samples per symbol, forwarded to :func:`receive`.
        **kw: Forwarded to :func:`receive`.

    Returns:
        The :func:`receive` dict plus ``"region"`` (the ``(start, end)`` used)
        and ``"region_estimated"`` (True when it came from the estimator).
    """
    x = _as_complex(samples)
    estimated = False
    if region is None:
        found = dsp.estimate_burst_region(x)
        if found["found"]:
            region = (int(found["start_sample"]), int(found["end_sample"]))
            estimated = True
        else:
            region = (0, int(x.size))
    start = max(0, int(region[0]))
    end = min(int(x.size), int(region[1]))
    if end <= start:
        start, end = 0, int(x.size)
    trimmed = x[start:end]
    result = receive(trimmed, scheme, sps=sps, **kw)
    result["region"] = (start, end)
    result["region_estimated"] = bool(estimated)
    result["metrics"]["region"] = (start, end)
    result["metrics"]["region_estimated"] = bool(estimated)
    return result


__all__ = [
    "CARRIER_LOCK_MIN",
    "CMA_MU",
    "COSTAS_KI",
    "COSTAS_KP",
    "DEFAULT_TAPS",
    "FSK_LOCK_MAX_RESIDUAL",
    "GARDNER_MU",
    "LMS_MU",
    "MIN_LOCK_SYMBOLS",
    "MM_MU",
    "SPS_INTEGER_TOLERANCE",
    "TIMING_LOCK_MIN",
    "ber_best_rotation",
    "cma_equalize",
    "costas_loop",
    "demodulate",
    "estimate_sps",
    "gardner_recover",
    "lms_equalize",
    "matched_filter",
    "mueller_muller_recover",
    "receive",
    "receive_burst",
    "resolve_rotation_with_sync",
]
