"""Fused symbol-rate and sampling-rate estimation.

The original code answered "what is the symbol rate?" with a single number produced by
one method (the ``|x|^2`` spectral line) and answered "what is the sampling
rate?" with ``2.2 x bandwidth``. Both are point estimates with no way to tell
a real measurement from a guess, which is exactly the failure mode this
project exists to avoid.

This module replaces them with **four evidence paths that can each abstain**,
plus a fusion step that reports how much independent agreement it found:

1. ``bandwidth``    -- occupied-bandwidth Nyquist guard. Always available when
   the PSD has a measurable span, but it bounds the *sampling* rate, not the
   symbol rate: for a root-raised-cosine spectrum ``BW = Rs (1 + beta)``, so
   bandwidth only tells us ``Rs <= BW``.
2. ``envelope``     -- the ``|x|^2`` spectral line at ``Rs``
   (:func:`dsp.estimate_symbol_rate`). Works for PSK/QAM, meaningless for
   constant-envelope FSK (there is no envelope to square).
3. ``cyclostationary`` -- the spectral-coherence peak over cycle frequency
   ``alpha``. This is the same physical feature as (2) -- a linearly modulated
   signal has non-zero cyclic autocorrelation at ``alpha = k/T`` -- but the
   coherence normalises out the noise PSD shape, so it survives lower SNR and
   does not fire on a coloured noise floor the way a raw PSD peak does.
4. ``fsk_period``   -- the recovered 2-FSK symbol period
   (:func:`dsp.estimate_fsk_symbol_period`), converted to a rate. A direct
   measurement, but it only exists for constant-envelope FSK.

Honesty contract (do not weaken)
--------------------------------
Every path may return "nothing". When **no** path produces a usable estimate
the fused result is ``0.0`` **plus a warning naming what was tried** -- never a
plausible-looking default. ``0.0`` means *not measurable*, exactly as
``dsp.estimate_bandwidth`` defines it for bandwidth. A single uncorroborated
path is reported with its confidence capped and a warning that it stands alone;
only agreement between two or more independent paths earns a high confidence.

This is still not blind estimation: every number carries the method that
produced it and the spread of the estimates that agreed. See ``info.md`` §30.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from rf_analyzer.core.dsp import estimate_fsk_symbol_period

# --------------------------------------------------------------------------- #
# Cyclostationary (spectral coherence) symbol-rate estimator
# --------------------------------------------------------------------------- #

#: FFT size for the cyclic periodogram. Sets the cycle-frequency resolution,
#: ``2 * fs / nfft``: at 2048 bins and a 100 kHz capture that is ~98 Hz, fine
#: enough to resolve the symbol rates in the bundled samples (1-20 kHz).
#: Smaller values trade resolution for speed and are used for short captures.
CYCLO_NFFT = 2048
#: Cap on averaged segments. More segments lower the estimator's variance, but
#: the cost is linear in them and 64 already averages well below the peak
#: width we care about.
CYCLO_MAX_SEGMENTS = 64
#: Smallest frequency-bin offset searched, in bins. **This is a leakage guard,
#: not a resolution choice.** The cyclic periodogram at bin offset ``d``
#: correlates FFT bins ``2d`` apart; a Hann window's main lobe is four bins
#: wide, so at ``d <= 2`` the two bins see the *same* spectral line and the
#: coherence is high for any signal with a line in it -- including a pure tone,
#: which has no symbol rate at all. Measured: with ``dmin = 2`` an unmodulated
#: tone plus noise scores a 29.5-sigma false peak; with ``dmin = 4`` the same
#: capture scores 6.4 sigma and is rejected, while every genuine modulation
#: keeps a 19-294 sigma peak. Do not lower this to reach a low symbol rate --
#: raise ``nfft`` instead.
CYCLO_MIN_BIN_OFFSET = 4
#: Peak height above the profile's own median, in robust sigmas
#: (``1.4826 * MAD``). Calibrated on the separation above: genuine modulations
#: sit at 13-200 sigma from 0 dB to noiseless, while AWGN, a tone, tone+noise
#: and a random-walk "audio" surrogate all stay under 6.5.
CYCLO_MIN_PROMINENCE = 10.0
#: Absolute floor on the normalised profile. Under the null (no
#: cyclostationarity) ``P(alpha)`` has expectation 1.0 by construction, so a
#: peak must at least clear it -- this is what rejects a perfectly flat
#: profile whose MAD is zero and whose prominence would otherwise be infinite.
CYCLO_MIN_PEAK = 1.4
#: A peak at ``alpha/2`` at least this strong (relative to the peak's own
#: prominence) means the peak we found is the second harmonic, not the symbol
#: rate. Reporting ``2 * Rs`` as ``Rs`` would be a factor-of-two error.
CYCLO_HARMONIC_RATIO = 0.6
#: Default upper limit on cycle frequency, as a fraction of the sample rate.
#: Above Nyquist/2 the cyclic periodogram has little support for a real capture.
CYCLO_MAX_ALPHA_FRAC = 0.45

#: Smallest FFT size :func:`choose_cyclo_nfft` will fall back to. Below this the
#: cycle-frequency resolution (``2 * fs / nfft``) is so coarse that a symbol
#: rate in the top decade of the band cannot be located at all.
CYCLO_MIN_NFFT = 256

#: Work budget for :func:`choose_cyclo_nfft`, in "bin-updates": the product of
#: the cycle-frequency bins searched, the averaged segments and the FFT size.
#:
#: **This is a wall-clock guard, not an accuracy choice, and it is why the
#: fused estimator takes an ``nfft`` at all.** The cost of
#: :func:`cyclostationary_profile` is ``(dmax - dmin) * n_segments * nfft``
#: with ``dmax ~ alpha_max * nfft / (2 fs)`` and ``n_segments`` capped at
#: :data:`CYCLO_MAX_SEGMENTS` -- so it grows as ``nfft ** 2`` and is
#: *independent of the capture length*. Measured on a 1 M-sample QPSK capture
#: at 10 dB (``bw`` = 351.6 kHz, so ``alpha_max`` = 1.2 bw):
#:
#: ==========  ===========  ===============  ==================
#: ``nfft``    profile cost  cycle-freq res.  symbol rate found
#: ==========  ===========  ===============  ==================
#: 2048        7.28 s        977 Hz           125 008 Hz
#: 1024        1.09 s        1 953 Hz         125 014 Hz
#: 512         0.45 s        3 906 Hz         125 104 Hz
#: ==========  ===========  ===============  ==================
#:
#: The true rate is 125 kHz, so 1024 costs 6.7x less than 2048 and is *more*
#: accurate on this capture (0.01% against 0.006%) -- the resolution is not the
#: limiting error at this symbol rate. 15 M bin-updates is the point where the
#: estimator stays near a second on a 1 M-sample capture, which is what the
#: §NFR-03 budget can afford alongside the rest of the analysis.
CYCLO_MAX_WORK = 15_000_000


def choose_cyclo_nfft(
    n_samples: int,
    sample_rate: float,
    alpha_max_hz: float,
    *,
    nfft: int = CYCLO_NFFT,
    max_work: int = CYCLO_MAX_WORK,
) -> int:
    """Largest power-of-two FFT size whose cyclic profile fits the work budget.

    See :data:`CYCLO_MAX_WORK` for the cost model and the measurements behind
    the budget. Returns a value in ``[CYCLO_MIN_NFFT, nfft]``, never below
    :data:`CYCLO_MIN_NFFT`: an unaffordable estimate at a usable resolution is
    worth more than an affordable one at a resolution that cannot resolve the
    rate. When even the floor exceeds the budget the floor is returned and the
    caller has to accept the cost -- the alternative would be to abstain from a
    measurement we can actually make.
    """
    nfft = int(nfft)
    if nfft <= CYCLO_MIN_NFFT:
        return max(64, nfft)
    fs = float(sample_rate)
    alpha_max = float(alpha_max_hz)
    if fs <= 0 or alpha_max <= 0:
        return nfft

    n = int(n_samples)
    while nfft > CYCLO_MIN_NFFT:
        dmax = alpha_max * nfft / (2.0 * fs)
        step = max(1, nfft // 2)
        segments = max(1, (n - nfft) // step + 1)
        segments = min(segments, CYCLO_MAX_SEGMENTS)
        if max(1.0, dmax) * segments * nfft <= float(max_work):
            return nfft
        nfft //= 2
    return CYCLO_MIN_NFFT


def cyclostationary_profile(
    samples: np.ndarray,
    sample_rate: float,
    nfft: int = CYCLO_NFFT,
    max_alpha_hz: float | None = None,
    max_segments: int = CYCLO_MAX_SEGMENTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Power-weighted spectral-coherence profile over cycle frequency.

    For each cycle frequency ``alpha`` this computes the time-smoothed cyclic
    periodogram

        ``S_x^alpha(f) = (1/K) * sum_k X_k(f + alpha/2) X_k*(f - alpha/2)``

    normalises it into the spectral coherence ``gamma^2 = |S_x^alpha|^2 /
    (P(f + alpha/2) P(f - alpha/2))``, and reduces it over ``f`` into a single
    number::

        ``P(alpha) = K * sum_f w(f) gamma^2(alpha, f) / sum_f w(f)``,
        ``w(f) = sqrt(P(f + alpha/2) P(f - alpha/2))``

    Two details make this work, and both were wrong in the first attempt:

    * **The frequency shift must wrap.** ``f + alpha/2`` and ``f - alpha/2``
      are DFT bins, and the DFT is periodic, so the shift is circular
      (``np.roll``). Clipping the shift to keep both bins in ``[0, nfft)``
      silently discards every ``f`` near DC -- which is exactly where the
      cyclic feature of a pulse-shaped signal lives. Measured on a clean BPSK
      capture: clipped, the profile is flat and the estimator finds nothing;
      wrapped, it peaks at 25.8 (null expectation 1.0) precisely on the symbol
      rate.
    * **Reduce by a power-weighted mean, not a maximum.** The maximum over
      ``f`` is an extreme-value statistic: with ``K`` segments its null value is
      ``~4.3/sqrt(K)`` regardless of the signal, so it reports a large peak on
      pure noise. Weighting by ``sqrt(P(f+a/2) P(f-a/2))`` concentrates the
      average on the bins that actually carry signal power, so AWGN scores 1.0
      while a 10 dB BPSK capture scores 9.1.

    Args:
        samples: Complex baseband samples.
        sample_rate: Sampling rate in Hz.
        nfft: FFT size; sets the cycle-frequency resolution ``2 * fs / nfft``.
        max_alpha_hz: Upper limit on cycle frequency. Defaults to ``0.45 * fs``.
            Pass ``~1.2 x bandwidth`` when the bandwidth is known: for any
            linear modulation ``Rs <= BW``, and limiting the search both cuts
            the cost and removes the high-alpha region where a noise-only
            profile is noisiest.
        max_segments: Cap on averaged segments.

    Returns:
        ``(alphas_hz, profile)``, both the same length and aligned. Empty arrays
        when the capture is too short to segment at all -- the caller must treat
        that as "no evidence", not as a zero rate.
    """
    x = np.asarray(samples, dtype=np.complex128).ravel()
    n = x.size
    if n < 64 or sample_rate <= 0:
        return np.zeros(0), np.zeros(0)

    nfft = int(min(int(nfft), n))
    # Round down to a power of two: the per-segment FFT dominates the cost and
    # a power-of-two length keeps it on the fast path. Then shrink it until
    # enough segments are available: the profile's variance falls as 1/K, and
    # on a short capture a 2048-point window leaves only six segments, which is
    # too noisy to clear the prominence threshold (measured: a 16-QAM capture of
    # 8000 samples scored a 3.3 peak at 2048 bins and 8.8 at 512). The cost is
    # cycle-frequency resolution, ``2 * fs / nfft`` -- reported honestly in the
    # peak's width rather than hidden.
    nfft = 1 << int(np.floor(np.log2(max(min(nfft, n // 8), 1))))
    nfft = max(128, min(nfft, CYCLO_NFFT))
    if nfft > n:
        return np.zeros(0), np.zeros(0)

    step = max(1, nfft // 2)
    n_seg = (n - nfft) // step + 1
    if n_seg < 2:
        return np.zeros(0), np.zeros(0)
    # Thin evenly across the capture rather than truncating it: a burst may sit
    # anywhere in the recording, and a prefix-only estimator would miss it.
    idx = (
        np.linspace(0, n_seg - 1, int(max_segments)).astype(int)
        if n_seg > int(max_segments)
        else np.arange(n_seg)
    )

    win = np.hanning(nfft)
    wnorm = float(np.sqrt(np.mean(win**2)))
    X = np.empty((idx.size, nfft), dtype=np.complex128)
    for j, k in enumerate(idx):
        start = int(k) * step
        X[j] = np.fft.fft(x[start : start + nfft] * win)
    X /= np.sqrt(nfft) * wnorm
    n_segments = X.shape[0]

    alpha_max = (
        float(max_alpha_hz)
        if max_alpha_hz
        else CYCLO_MAX_ALPHA_FRAC * float(sample_rate)
    )
    dmax = int(alpha_max * nfft / (2.0 * float(sample_rate)))
    dmax = max(1, min(dmax, nfft // 2 - 1))
    dmin = max(CYCLO_MIN_BIN_OFFSET, 1)
    if dmax <= dmin:
        return np.zeros(0), np.zeros(0)

    alphas = np.empty(dmax - dmin, dtype=np.float64)
    profile = np.zeros(dmax - dmin, dtype=np.float64)
    for d in range(dmin, dmax):
        a = np.roll(X, -d, axis=1)
        b = np.roll(X, d, axis=1)
        power_a = np.mean(np.abs(a) ** 2, axis=0)
        power_b = np.mean(np.abs(b) ** 2, axis=0)
        denominator = power_a * power_b
        root = np.sqrt(denominator)
        total = float(np.sum(root))
        alphas[d - dmin] = 2.0 * d * float(sample_rate) / nfft
        if total <= 0.0:
            continue
        cross = np.mean(a * np.conj(b), axis=0)
        # gamma^2 = |mean(A B*)|^2 / (mean|A|^2 * mean|B|^2); under the null
        # this has expectation 1/K, so multiplying by K puts the profile on a
        # scale where "no cyclic structure" reads as 1.0. The weighted mean is
        # sum(w * gamma^2) / sum(w) with w = sqrt(denominator), i.e.
        # sum(|cross|^2 / sqrt(denominator)) / sum(sqrt(denominator)) -- note
        # the exponent, because |cross|^2 already carries one power of the
        # denominator and using sqrt(den) * |cross|^2 instead weights the
        # statistic by the *cube* of the power, which makes a capture with a
        # large DC spike (any real SDR recording) report a profile in the
        # millions instead of order 1.
        usable = root > 0.0
        weighted = float(np.sum(np.abs(cross[usable]) ** 2 / root[usable]))
        profile[d - dmin] = n_segments * weighted / total
    return alphas, profile


def _refine_peak(profile: np.ndarray, index: int) -> float:
    """Sub-bin peak location by parabolic interpolation, in bins.

    Without this the estimate is quantised to the cycle-frequency resolution
    ``2 * fs / nfft`` -- ~98 Hz at 2048 bins and 100 kHz, which is a 10% error
    on a 1 kHz symbol rate. Returns the fractional offset added to ``index``.
    """
    if index <= 0 or index >= profile.size - 1:
        return float(index)
    y0 = float(np.log(profile[index - 1] + 1e-12))
    y1 = float(np.log(profile[index] + 1e-12))
    y2 = float(np.log(profile[index + 1] + 1e-12))
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return float(index)
    offset = 0.5 * (y0 - y2) / denom
    if not np.isfinite(offset) or abs(offset) > 1.0:
        return float(index)
    return float(index) + float(offset)


def cyclostationary_symbol_rate(
    samples: np.ndarray,
    sample_rate: float,
    nfft: int = CYCLO_NFFT,
    max_alpha_hz: float | None = None,
) -> dict:
    """Symbol rate from the cyclostationary spectral-coherence peak.

    Finds the strongest peak in :func:`cyclostationary_profile` and decides
    whether it is real. A peak counts only if it clears the absolute floor
    (:data:`CYCLO_MIN_PEAK`), sits :data:`CYCLO_MIN_PROMINENCE` robust sigmas
    above the profile's own median, and the profile is not flat (a pure tone has
    perfect coherence at *every* cycle frequency, so its MAD is zero and it has
    no peak to speak of).

    The peak is then tested against its own sub-harmonic: if ``alpha/2`` is
    nearly as prominent, the peak is the second harmonic of the symbol rate and
    the fundamental is reported instead. Reporting ``2 * Rs`` as ``Rs`` would be
    a silent factor-of-two error.

    Returns:
        ``{"symbol_rate_hz", "coherence", "found", "reason"}``. ``found`` is
        False and the rate is 0.0 whenever no peak qualifies -- the caller must
        not read a 0.0 rate as a measurement.
    """
    alphas, profile = cyclostationary_profile(
        samples, sample_rate, nfft=nfft, max_alpha_hz=max_alpha_hz
    )
    if alphas.size == 0:
        return {
            "symbol_rate_hz": 0.0,
            "coherence": 0.0,
            "found": False,
            "reason": "capture too short to segment for a cyclic periodogram",
        }

    median = float(np.median(profile))
    mad = float(np.median(np.abs(profile - median)))
    sigma = 1.4826 * mad
    index = int(np.argmax(profile))
    best = float(profile[index])

    if sigma <= 0.0:
        return {
            "symbol_rate_hz": 0.0,
            "coherence": best,
            "found": False,
            "reason": (
                "profile is flat (robust sigma 0): a rank-one signal such as an "
                "unmodulated tone is coherent at every cycle frequency and has "
                "no symbol rate to find"
            ),
        }
    prominence = (best - median) / sigma
    if best < CYCLO_MIN_PEAK or prominence < CYCLO_MIN_PROMINENCE:
        return {
            "symbol_rate_hz": 0.0,
            "coherence": best,
            "found": False,
            "reason": (
                f"peak {best:.2f} ({prominence:.1f} sigma above the profile "
                f"median) did not clear the detection threshold "
                f"({CYCLO_MIN_PEAK:.1f} absolute, {CYCLO_MIN_PROMINENCE:.0f} sigma)"
            ),
        }

    refined = _refine_peak(profile, index)
    step = float(alphas[1] - alphas[0]) if alphas.size > 1 else 0.0
    rate = float(alphas[0]) + refined * step

    # Harmonic resolution. A linearly modulated signal is cyclostationary at
    # every multiple of the symbol rate, and which multiple is tallest depends
    # on the pulse shape -- a 4-FSK capture was measured with its third harmonic
    # (3 x Rs) stronger than the fundamental. Walk the divisors and keep the
    # smallest one that is still nearly as prominent; reporting 3 x Rs as Rs
    # would be a silent factor-of-three error. Halving once is not enough.
    for divisor in (2, 3, 4, 5):
        candidate = rate / divisor
        nearest = int(np.argmin(np.abs(alphas - candidate)))
        if abs(float(alphas[nearest]) - candidate) > 1.5 * abs(step):
            continue
        sub_prominence = (float(profile[nearest]) - median) / sigma
        if sub_prominence >= CYCLO_HARMONIC_RATIO * prominence:
            rate = float(alphas[0]) + _refine_peak(profile, nearest) * step
            break

    if not np.isfinite(rate) or rate <= 0.0:
        return {
            "symbol_rate_hz": 0.0,
            "coherence": best,
            "found": False,
            "reason": "peak located outside the valid cycle-frequency range",
        }
    return {
        "symbol_rate_hz": rate,
        "coherence": best,
        "found": True,
        "reason": "spectral-coherence peak",
    }


# --------------------------------------------------------------------------- #
# Envelope (|x|^2 spectral line) symbol-rate estimator
# --------------------------------------------------------------------------- #

#: FFT size for the squared-envelope spectrum.
ENVELOPE_NFFT = 1024
#: Cap on Welch-averaged segments.
ENVELOPE_MAX_SEGMENTS = 64
#: Running-median window, in bins, used to estimate the squared-envelope
#: spectrum's *local* background. See :func:`envelope_symbol_rate` for why the
#: background has to be local.
ENVELOPE_MEDIAN_WINDOW = 31
#: How many robust sigmas above the local background the line must sit.
ENVELOPE_MIN_PROMINENCE = 8.0
#: Below this fraction of the sample rate we are inside the DC leakage of the
#: mean removal, not a symbol-rate line.
ENVELOPE_MIN_RATE_FRAC = 0.002


def envelope_symbol_rate(
    samples: np.ndarray,
    sample_rate: float,
    nfft: int = ENVELOPE_NFFT,
    max_rate_hz: float | None = None,
) -> dict:
    """Symbol rate from the ``|x|^2`` spectral line, Welch-averaged and CFAR-gated.

    A linearly modulated signal carries a spectral line at ``Rs`` in the
    spectrum of its squared envelope. Two things make the naive version of this
    estimator unreliable, and this function fixes both:

    * **Variance.** :func:`dsp.estimate_symbol_rate` uses a *single*
      periodogram and accepts a peak 6 dB above the median. A single
      periodogram of white noise has ~100% variance per bin, so its maximum
      sits about 5-6 dB above the median by construction -- the gate is right
      at the noise floor and the estimator fires on AWGN roughly half the time
      (measured: 70 kHz of noise reported as a symbol rate). Averaging ``K``
      overlapping Welch segments cuts the per-bin variance to ``~1/sqrt(K)``.
    * **Background shape.** The squared envelope of a pulse-shaped signal is
      *not* flat: it has a broad lobe around DC of width ``~2 x Rs`` and
      sloping tails. Measured on a 16-QAM capture, the prominence of the true
      line against the *global* median of the spectrum is 4.7 when the search
      is bounded to 0.2 x fs and 1.0e6 when it is not -- five orders of
      magnitude for the same line, because the global median moves with the
      band you happen to look at. The gate here therefore compares the peak to
      a **running median** of the spectrum, which tracks the lobe and the tails
      and leaves the line standing on its own.

    Args:
        samples: Complex baseband samples.
        sample_rate: Sampling rate in Hz.
        nfft: FFT size for each Welch segment.
        max_rate_hz: Upper limit on the rate *searched* (the background estimate
            still uses the whole positive band, so the result does not depend
            on this bound).

    Returns:
        ``{"symbol_rate_hz", "prominence", "found", "reason"}``. ``found`` is
        False and the rate 0.0 when no line qualifies.
    """
    x = np.asarray(samples, dtype=np.complex128).ravel()
    n = x.size
    if n < 64 or sample_rate <= 0:
        return {
            "symbol_rate_hz": 0.0,
            "prominence": 0.0,
            "found": False,
            "reason": "capture too short for a squared-envelope spectrum",
        }

    envelope = np.abs(x) ** 2
    envelope = envelope - float(np.mean(envelope))

    nfft = 1 << int(np.floor(np.log2(max(min(int(nfft), n), 1))))
    nfft = max(64, min(nfft, ENVELOPE_NFFT))
    step = max(1, nfft // 2)
    n_seg = (n - nfft) // step + 1
    if n_seg < 2:
        # Not enough for Welch averaging; fall back to one periodogram but say
        # so, because the variance argument above no longer holds.
        nfft = 1 << int(np.floor(np.log2(max(n // 2, 1))))
        nfft = max(32, min(nfft, n))
        if nfft < 32:
            return {
                "symbol_rate_hz": 0.0,
                "prominence": 0.0,
                "found": False,
                "reason": "capture too short for a squared-envelope spectrum",
            }
        step = max(1, nfft // 2)
        n_seg = max(1, (n - nfft) // step + 1)
    if n_seg > ENVELOPE_MAX_SEGMENTS:
        idx = np.linspace(0, n_seg - 1, ENVELOPE_MAX_SEGMENTS).astype(int)
    else:
        idx = np.arange(n_seg)

    win = np.hanning(nfft)
    wnorm = float(np.mean(win**2))
    psd = np.zeros(nfft, dtype=np.float64)
    for k in idx:
        start = int(k) * step
        segment = envelope[start : start + nfft] * win
        psd += np.abs(np.fft.fft(segment)) ** 2
    psd /= max(1, idx.size) * (nfft * wnorm)

    freqs = np.fft.fftfreq(nfft, d=1.0 / float(sample_rate))
    min_rate = ENVELOPE_MIN_RATE_FRAC * float(sample_rate)
    positive = freqs > min_rate
    if int(np.count_nonzero(positive)) < 8:
        return {
            "symbol_rate_hz": 0.0,
            "prominence": 0.0,
            "found": False,
            "reason": "too few positive-frequency bins to test",
        }

    band = psd[positive]
    pos_freqs = freqs[positive]
    # Local background: a running median, so a broad lobe or a sloping tail is
    # treated as background rather than as evidence of a line.
    background = median_filter(band, size=ENVELOPE_MEDIAN_WINDOW, mode="nearest")
    residual = band - background
    median = float(np.median(residual))
    sigma = 1.4826 * float(np.median(np.abs(residual - median)))
    if sigma <= 0.0:
        return {
            "symbol_rate_hz": 0.0,
            "prominence": 0.0,
            "found": False,
            "reason": "squared-envelope spectrum has no measurable noise floor",
        }

    searchable = np.ones(band.size, dtype=bool)
    if max_rate_hz and max_rate_hz > 0:
        searchable = pos_freqs <= float(max_rate_hz)
    if int(np.count_nonzero(searchable)) < 4:
        return {
            "symbol_rate_hz": 0.0,
            "prominence": 0.0,
            "found": False,
            "reason": "no positive-frequency bins inside the search bound",
        }

    masked = np.where(searchable, residual, -np.inf)
    index = int(np.argmax(masked))
    prominence = (float(residual[index]) - median) / sigma
    if prominence < ENVELOPE_MIN_PROMINENCE:
        return {
            "symbol_rate_hz": 0.0,
            "prominence": prominence,
            "found": False,
            "reason": (
                f"strongest envelope line is {prominence:.1f} sigma above its "
                f"local background; needs {ENVELOPE_MIN_PROMINENCE:.0f}"
            ),
        }

    refined = _refine_peak(np.maximum(residual - median, 0.0), index)
    step_hz = float(pos_freqs[1] - pos_freqs[0]) if pos_freqs.size > 1 else 0.0
    rate = float(pos_freqs[index]) + (refined - index) * step_hz
    return {
        "symbol_rate_hz": abs(rate),
        "prominence": prominence,
        "found": True,
        "reason": "Welch-averaged squared-envelope spectral line",
    }


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #

#: Two estimates this close (as a fraction of the larger) count as agreeing.
#: Deliberately loose: the paths measure the same quantity through different
#: windows and a 5% spread is normal even when all of them are right.
AGREEMENT_TOLERANCE = 0.05
#: Confidence when two or more independent paths agree.
CONFIDENCE_CORROBORATED = 0.8
#: Confidence when a path made a *direct* measurement (the FSK period, which is
#: recovered by fitting and residual-testing a piecewise-constant model rather
#: than by detecting a feature).
CONFIDENCE_DIRECT = 0.7
#: Weight per evidence path, used only to pick between agreeing estimates.
_PATH_WEIGHTS = {
    "fsk_period": 1.0,
    "cyclostationary": 0.9,
    "envelope": 0.7,
}


def _weighted_median(values: list[float], weights: list[float]) -> float:
    order = np.argsort(values)
    v = np.asarray(values, dtype=np.float64)[order]
    w = np.asarray(weights, dtype=np.float64)[order]
    cumulative = np.cumsum(w)
    if cumulative[-1] <= 0.0:
        return float(np.median(v))
    return float(v[int(np.searchsorted(cumulative, 0.5 * cumulative[-1]))])


def estimate_symbol_rate_fused(
    samples: np.ndarray,
    sample_rate: float,
    bandwidth_estimate: float | None = None,
    fsk_period: int | None = None,
    modulation_hint: str | None = None,
    nfft: int | None = None,
) -> dict:
    """Fuse every available symbol-rate path into one honest answer.

    The decision rule, in order:

    1. **A direct measurement wins.** When the 2-FSK symbol period is recovered,
       that is a measurement of the period, not a detection of a feature -- the
       estimator fits a piecewise-constant model and rejects it when the
       reconstruction residual is too large. It is reported at
       :data:`CONFIDENCE_DIRECT`. A disagreeing detector does *not* veto it, but
       the disagreement is reported.
    2. **Two detectors that agree** give a measurement at
       :data:`CONFIDENCE_CORROBORATED`, reduced by a weighted median.
    3. **Two detectors that disagree** give ``0.0``. Two independent estimates
       of the same quantity that differ by more than
       :data:`AGREEMENT_TOLERANCE` are evidence that at least one is wrong, and
       the honest answer to "what is the symbol rate?" is then "not measurable"
       -- not the louder of the two. This is what keeps speech, whose pitch
       genuinely *is* cyclostationary, from being reported as a symbol rate:
       the pitch line and the envelope line disagree, so nothing is claimed.
    4. **One detector alone** gives ``0.0`` as well, with the candidate recorded
       in ``components`` and named in the warning. A single feature detection is
       a hypothesis; this tool does not turn hypotheses into measurements.

    Args:
        samples: Complex baseband samples.
        sample_rate: Sampling rate in Hz.
        bandwidth_estimate: Occupied bandwidth in Hz, if already measured.
            Bounds the cycle-frequency search (``Rs <= BW``) and sanity-checks
            the fused result; never a point estimate of ``Rs``.
        fsk_period: Recovered 2-FSK symbol period in samples, if known.
        modulation_hint: The modulation label, when one is already available.
            Suppresses paths that are physically inapplicable -- the ``|x|^2``
            line is meaningless for constant-envelope FSK.
        nfft: FFT size for the cyclostationary path. ``None`` (the default) uses
            the module's own :data:`CYCLO_NFFT` cap, which is what the unit
            tests pin. A caller on a large capture should pass a size from
            :func:`choose_cyclo_nfft`, because the profile's cost grows as
            ``nfft ** 2`` -- see :data:`CYCLO_MAX_WORK`. The resolution actually
            achieved is reported in ``resolution_hz`` either way, so a coarser
            answer is never presented as a finer one.

    Returns:
        A dict with ``symbol_rate_hz`` (``0.0`` when not measurable),
        ``confidence``, ``method`` (``"direct"``/``"corroborated"``/
        ``"disagreement"``/``"single"``/``"none"``), ``components`` (every
        path's raw result, including the ones that abstained), ``agreement``,
        ``nfft``, ``resolution_hz`` and ``warnings``.
    """
    warnings: list[str] = []
    components: dict[str, float | None] = {
        "bandwidth": float(bandwidth_estimate) if bandwidth_estimate else None,
        "envelope": None,
        "cyclostationary": None,
        "fsk_period": None,
    }
    x = np.asarray(samples).ravel()
    sr = float(sample_rate)
    cyclo_nfft = int(min(int(nfft), CYCLO_NFFT)) if nfft else CYCLO_NFFT
    resolution_hz = (2.0 * sr / cyclo_nfft) if (sr > 0 and cyclo_nfft > 0) else 0.0

    bw = float(bandwidth_estimate) if bandwidth_estimate else 0.0
    is_fsk = bool(modulation_hint) and "FSK" in str(modulation_hint).upper()

    def _refuse(method: str, message: str) -> dict:
        warnings.append(message)
        return {
            "symbol_rate_hz": 0.0,
            "confidence": 0.0,
            "method": method,
            "components": components,
            "agreement": None,
            "nfft": cyclo_nfft,
            "resolution_hz": resolution_hz,
            "warnings": warnings,
        }

    # --- Path 4: FSK period (a direct measurement, when it exists) ---------- #
    period = int(fsk_period) if fsk_period else 0
    if period <= 0 and is_fsk and x.size and sr > 0:
        try:
            period = int(estimate_fsk_symbol_period(x))
        except Exception:
            period = 0
    if period > 1:
        components["fsk_period"] = sr / float(period)

    # --- Path 3: cyclostationary coherence --------------------------------- #
    # The occupied bandwidth bounds the search (Rs <= BW) and that is a useful
    # speed-up, but it is a *different estimator's* output and the project's own
    # documentation says it fails on flat-spectrum signals such as QPSK. Using
    # it as a hard bound would let one estimator's error hide a real symbol
    # rate, so a capped search that finds nothing is retried over the full
    # range before the path is allowed to abstain.
    if sr > 0 and x.size >= 128:
        cyclo = {"found": False, "symbol_rate_hz": 0.0, "reason": "estimator error"}
        for limit in ([1.2 * bw, None] if bw > 0 else [None]):
            try:
                cyclo = cyclostationary_symbol_rate(
                    x, sr, nfft=cyclo_nfft, max_alpha_hz=limit
                )
            except Exception:
                cyclo = {
                    "found": False,
                    "symbol_rate_hz": 0.0,
                    "reason": "estimator error",
                }
            if cyclo["found"]:
                break
        if cyclo["found"]:
            components["cyclostationary"] = float(cyclo["symbol_rate_hz"])

    # --- Path 2: |x|^2 spectral line --------------------------------------- #
    # Skipped for constant-envelope FSK: a constant-modulus signal has no
    # envelope line at Rs, and the estimator returns whatever the noise floor
    # peaks at instead (measured 395 kHz for a 1 kHz burst).
    if not is_fsk and sr > 0 and x.size >= 64:
        envelope = {"found": False, "symbol_rate_hz": 0.0}
        for limit in ([1.2 * bw, None] if bw > 0 else [None]):
            try:
                envelope = envelope_symbol_rate(x, sr, max_rate_hz=limit)
            except Exception:
                envelope = {"found": False, "symbol_rate_hz": 0.0}
            if envelope["found"]:
                break
        if envelope["found"]:
            components["envelope"] = float(envelope["symbol_rate_hz"])

    # --- Rule 1: a direct period measurement wins --------------------------- #
    direct = components["fsk_period"]
    if direct is not None and float(direct) > 0.0:
        fused = float(direct)
        detectors = [
            (name, float(value))
            for name, value in components.items()
            if name in ("envelope", "cyclostationary") and value is not None
        ]
        if detectors and any(
            abs(value - fused) > AGREEMENT_TOLERANCE * max(value, fused)
            for _, value in detectors
        ):
            warnings.append(
                "The feature detectors disagree with the directly measured FSK "
                f"symbol period ({fused:.0f} Hz vs "
                + ", ".join(f"{n}={v:.0f} Hz" for n, v in detectors)
                + "). The direct measurement is reported; the disagreement is "
                "recorded so it is not mistaken for corroboration."
            )
        if bw > 0.0 and fused > bw * 1.15:
            warnings.append(
                f"Symbol-rate estimate {fused:.0f} Hz exceeds the occupied "
                f"bandwidth {bw:.0f} Hz, which is not physically possible for a "
                "linear modulation (BW = Rs (1 + beta), beta >= 0). One of the "
                "two measurements is wrong; both are reported as measured."
            )
        return {
            "symbol_rate_hz": fused,
            "confidence": float(CONFIDENCE_DIRECT),
            "method": "direct",
            "components": components,
            "agreement": None,
            "nfft": cyclo_nfft,
            "resolution_hz": resolution_hz,
            "warnings": warnings,
        }

    # --- Rules 2-4: fuse the detectors -------------------------------------- #
    points = [
        (name, float(value))
        for name, value in components.items()
        if name in ("envelope", "cyclostationary") and value is not None
    ]

    if not points:
        tried = (
            "bandwidth Nyquist guard, cyclostationary coherence"
            if is_fsk
            else "|x|^2 spectral line, cyclostationary coherence"
        )
        return _refuse(
            "none",
            "Symbol rate not measurable: no evidence path produced a usable "
            f"estimate ({tried}, FSK period recovery all declined). Reported as "
            "0 Hz, meaning 'not measurable' -- not as a measurement.",
        )

    if len(points) == 1:
        name, value = points[0]
        return _refuse(
            "single",
            f"Symbol rate not measurable: only one evidence path produced a "
            f"number ({name} = {value:.0f} Hz) and nothing independent "
            "corroborated it. A single feature detection is a hypothesis, not a "
            "measurement, so 0 Hz is reported and the candidate is kept in "
            "components.",
        )

    values = [v for _, v in points]
    weights = [_PATH_WEIGHTS.get(name, 0.5) for name, _ in points]
    fused = _weighted_median(values, weights)
    spread = (max(values) - min(values)) / max(values) if max(values) > 0 else 0.0
    agreeing = [
        (name, value)
        for name, value in points
        if abs(value - fused) <= AGREEMENT_TOLERANCE * max(value, fused)
    ]

    if len(agreeing) < 2:
        return _refuse(
            "disagreement",
            "Symbol rate not measurable: the independent evidence paths "
            f"disagree ({', '.join(f'{n}={v:.0f} Hz' for n, v in points)}, a "
            f"{spread:.0%} spread, tolerance {AGREEMENT_TOLERANCE:.0%}). Two "
            "estimates of the same quantity that far apart mean at least one is "
            "wrong, so 0 Hz is reported rather than the louder of the two.",
        )

    fused = _weighted_median(
        [v for _, v in agreeing], [_PATH_WEIGHTS.get(n, 0.5) for n, _ in agreeing]
    )

    # Physical bound: for any linear modulation BW = Rs (1 + beta) with
    # beta >= 0, so Rs can never exceed the occupied bandwidth. A violation
    # means one of the two measurements is wrong; say so rather than picking.
    if bw > 0.0 and fused > bw * 1.15:
        warnings.append(
            f"Symbol-rate estimate {fused:.0f} Hz exceeds the occupied "
            f"bandwidth {bw:.0f} Hz, which is not physically possible for a "
            "linear modulation (BW = Rs (1 + beta), beta >= 0). One of the two "
            "measurements is wrong; both are reported as measured."
        )

    return {
        "symbol_rate_hz": float(fused),
        "confidence": float(CONFIDENCE_CORROBORATED),
        "method": "corroborated",
        "components": components,
        "agreement": float(spread),
        "nfft": cyclo_nfft,
        "resolution_hz": resolution_hz,
        "warnings": warnings,
    }


def estimate_sampling_rate_fused(
    samples: np.ndarray,
    sample_rate: float,
    bandwidth_estimate: float | None = None,
    symbol_rate_hz: float | None = None,
) -> dict:
    """Fuse Nyquist, oversampling and standard-rate evidence into one hint.

    Replaces ``bandwidth * 2.2``. The three sources are the same ones
    :func:`dsp.estimate_sampling_rate_candidates` ranks, but the fused answer
    is only reported when at least one of them exists at all:

    * Nyquist feasibility: ``fs >= 2 x BW`` (a hard physical requirement);
    * integer oversampling of the symbol rate, ``fs = k x Rs`` for the usual
      ``k`` in ``{2, 4, 8, 16}`` -- the tightest constraint, because a capture
      chain is normally designed to run at a small integer multiple;
    * a standard capture rate falling inside the plausible band.

    Returns:
        ``sampling_rate_hz`` (0.0 when not measurable), ``confidence``,
        ``candidates`` (the ranked list from the hypothesis tester),
        ``rationale`` and ``warnings``.
    """
    warnings: list[str] = []
    bw = float(bandwidth_estimate) if bandwidth_estimate else 0.0
    rs = float(symbol_rate_hz) if symbol_rate_hz else 0.0

    if bw <= 0.0 and rs <= 0.0:
        warnings.append(
            "Sampling rate not measurable: neither the occupied bandwidth nor "
            "the symbol rate could be estimated, and a raw .iq capture carries "
            "no rate metadata. Reported as 0 Hz, meaning 'not measurable'. "
            "Supply sample_rate= yourself, or use a .wav capture, which "
            "carries its own rate."
        )
        return {
            "sampling_rate_hz": 0.0,
            "confidence": 0.0,
            "candidates": [],
            "rationale": "no bandwidth and no symbol rate",
            "warnings": warnings,
        }

    # Imported here rather than at module scope: the candidate ranker is a
    # heavier function and this keeps the import graph flat.
    from rf_analyzer.core.dsp import estimate_sampling_rate_candidates

    try:
        candidates = estimate_sampling_rate_candidates(
            samples,
            sample_rate,
            bandwidth_estimate=bw or None,
            symbol_rate_estimate=rs or None,
        )
    except Exception:
        candidates = []

    nyquist = 2.0 * bw if bw > 0.0 else 0.0
    oversampled = rs * 2.0 if rs > 0.0 else 0.0
    floor = max(nyquist, oversampled)

    if floor <= 0.0:
        warnings.append(
            "Sampling rate not measurable: the evidence paths produced no "
            "usable constraint."
        )
        return {
            "sampling_rate_hz": 0.0,
            "confidence": 0.0,
            "candidates": candidates,
            "rationale": "no usable constraint",
            "warnings": warnings,
        }

    best = float(candidates[0]["rate_hz"]) if candidates else floor
    fused = max(floor, best)

    rationale_parts = []
    if nyquist > 0.0:
        rationale_parts.append(f"Nyquist 2 x BW = {nyquist:.0f} Hz")
    if oversampled > 0.0:
        rationale_parts.append(f"2 x Rs = {oversampled:.0f} Hz")
    if candidates:
        rationale_parts.append("ranked hypothesis: " + str(candidates[0]["rationale"]))

    confidence = 0.5
    if rs > 0.0 and bw > 0.0:
        confidence = 0.7
    if len(candidates) >= 3:
        # A well-separated top candidate is stronger evidence than a flat list.
        top, runner_up = candidates[0]["score"], candidates[1]["score"]
        if top - runner_up >= 0.15:
            confidence = min(0.85, confidence + 0.15)
    if bw <= 0.0 or rs <= 0.0:
        warnings.append(
            "Sampling-rate hint rests on a single constraint ("
            + ("bandwidth only" if bw > 0.0 else "symbol rate only")
            + "); it is a feasibility floor, not a measurement of the "
            "capture's actual rate."
        )

    return {
        "sampling_rate_hz": float(fused),
        "confidence": float(confidence),
        "candidates": candidates,
        "rationale": "; ".join(rationale_parts),
        "warnings": warnings,
    }


__all__ = [
    "AGREEMENT_TOLERANCE",
    "CONFIDENCE_CORROBORATED",
    "CONFIDENCE_DIRECT",
    "CYCLO_HARMONIC_RATIO",
    "CYCLO_MAX_WORK",
    "CYCLO_MIN_BIN_OFFSET",
    "CYCLO_MIN_NFFT",
    "CYCLO_MIN_PEAK",
    "CYCLO_MIN_PROMINENCE",
    "CYCLO_NFFT",
    "ENVELOPE_MIN_PROMINENCE",
    "ENVELOPE_NFFT",
    "choose_cyclo_nfft",
    "cyclostationary_profile",
    "cyclostationary_symbol_rate",
    "envelope_symbol_rate",
    "estimate_sampling_rate_fused",
    "estimate_symbol_rate_fused",
]
