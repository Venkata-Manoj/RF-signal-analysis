"""PSD/waterfall/parameter estimation. Full spec: info.md §12.2."""

from __future__ import annotations

import numpy as np

from rf_analyzer.config import (
    FSK_MAX_RECONSTRUCTION_RESIDUAL,
    FSK_MIN_SYMBOL_PERIOD,
)


def compute_psd(
    samples: np.ndarray, sample_rate: float, nfft: int = 4096
) -> tuple[np.ndarray, np.ndarray]:
    """Compute power spectral density. Returns frequency axis and PSD in dB."""
    samples = np.asarray(samples)
    if samples.size == 0:
        raise ValueError("Not enough samples for PSD computation.")
    nfft = min(nfft, len(samples))
    if nfft <= 0:
        raise ValueError("Not enough samples for PSD computation.")
    window = np.hanning(nfft)
    segment = samples[:nfft] * window
    fft_vals = np.fft.fftshift(np.fft.fft(segment, n=nfft))
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate))
    psd = np.abs(fft_vals) ** 2
    psd_db = 10.0 * np.log10(psd + 1e-12)
    return freqs, psd_db


def estimate_center_frequency(freqs: np.ndarray, psd_db: np.ndarray) -> float:
    """Estimate center frequency as PSD peak."""
    freqs = np.asarray(freqs)
    psd_db = np.asarray(psd_db)
    if freqs.size == 0 or psd_db.size == 0:
        raise ValueError("Empty spectrum input.")
    peak_index = int(np.argmax(psd_db))
    return float(freqs[peak_index])


def estimate_bandwidth(
    freqs: np.ndarray, psd_db: np.ndarray, threshold_db: float = 10.0
) -> float:
    """Estimate occupied bandwidth using noise-floor threshold.

    Returns the span between the lowest and highest bin that clears the median
    noise floor by ``threshold_db``, or ``0.0`` when no usable span exists.

    ``0.0`` means *unusable*, not *zero bandwidth*: a single bin above the
    floor is not a bandwidth, so at least two bins must qualify. Callers
    should treat ``0.0`` as "estimate unavailable" and say so rather than
    propagating it as a measurement.

    Known MVP limitation (§12, §32): the median-floor premise assumes the
    signal peaks well above a flat noise floor. That does not hold for
    flat-spectrum modulations such as QPSK, where ``max - median`` for a real
    signal (~10.3 dB) is no larger than for pure noise (~11.0 dB). Lowering
    the threshold therefore does *not* separate signal from noise -- it just
    starts classifying noise as a wideband signal. A robust bandwidth
    estimator (e.g. a Welch-averaged noise-floor subtraction or a
    symbol-rate-derived estimate) is V2 work.
    """
    freqs = np.asarray(freqs)
    psd_db = np.asarray(psd_db)
    if freqs.size == 0 or psd_db.size == 0:
        raise ValueError("Empty spectrum input.")
    noise_floor = float(np.median(psd_db))
    signal_mask = psd_db > (noise_floor + threshold_db)
    if int(np.count_nonzero(signal_mask)) < 2:
        return 0.0
    signal_freqs = freqs[signal_mask]
    return float(np.max(signal_freqs) - np.min(signal_freqs))


def estimate_snr(psd_db: np.ndarray) -> float:
    """Rough SNR estimate: peak PSD − median noise floor."""
    psd_db = np.asarray(psd_db)
    if psd_db.size == 0:
        raise ValueError("Empty spectrum input.")
    noise_floor = float(np.median(psd_db))
    peak = float(np.max(psd_db))
    return peak - noise_floor


def estimate_sampling_rate(bandwidth_estimate: float, factor: float = 2.2) -> float:
    """Naive sampling-rate hint: bandwidth * factor.

    MVP heuristic only (NTRO gap): for a band-limited signal the Nyquist
    minimum is ``2 * BW``; ``factor=2.2`` adds ~10% guard margin for the
    median-floor :func:`estimate_bandwidth` bias and filter roll-off.

    This is intentionally coarse — it uses only the PSD bandwidth and
    knows nothing about symbol rate or oversampling. For a refined
    estimate that fuses the ``|x|²`` symbol-rate line (see
    :func:`estimate_symbol_rate`), use
    :func:`estimate_sampling_rate_wideband`.

    Never claim precision — V2 will add cyclostationary hypothesis testing.
    """
    if bandwidth_estimate is None or bandwidth_estimate <= 0:
        return 0.0
    return float(bandwidth_estimate * float(factor))


def estimate_sampling_rate_wideband(
    bandwidth_estimate: float,
    symbol_rate_estimate: float | None = 0.0,
    factor: float = 2.2,
) -> float:
    """Fuse bandwidth and symbol-rate estimates into a sampling-rate hint.

    Combines the naive ``BW * factor`` Nyquist hint with the ``|x|²``
    spectral-line symbol-rate estimate from :func:`estimate_symbol_rate`:

    * If ``symbol_rate_estimate`` is missing/non-positive, falls back to
      :func:`estimate_sampling_rate` (BW-only path, backward compatible).
    * Otherwise returns ``max(BW, Rs) * factor`` — the sampling rate must
      satisfy Nyquist for the occupied bandwidth *and* oversample the
      symbol rate, so the tighter (larger) of the two constraints wins.

    Args:
        bandwidth_estimate: Occupied bandwidth in Hz (from
            :func:`estimate_bandwidth`).
        symbol_rate_estimate: Symbol rate in Hz (from
            :func:`estimate_symbol_rate`); ``None``/``<= 0`` means
            "no reliable line found".
        factor: Guard factor over Nyquist (default 2.2, same as
            :func:`estimate_sampling_rate`).

    Returns:
        Fused sampling-rate estimate in Hz, or 0.0 if bandwidth is invalid.
    """
    narrow = estimate_sampling_rate(bandwidth_estimate, factor)
    if symbol_rate_estimate is None or symbol_rate_estimate <= 0:
        return narrow
    if bandwidth_estimate is None or bandwidth_estimate <= 0:
        return 0.0
    return float(
        max(float(bandwidth_estimate), float(symbol_rate_estimate)) * float(factor)
    )


def compute_waterfall(
    samples: np.ndarray, sample_rate: float, nfft: int = 1024, overlap: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute waterfall matrix.

    Returns:
      freqs: frequency axis
      times: time axis
      waterfall_db: 2D array [time_bins, freq_bins]
    """
    samples = np.asarray(samples)
    if samples.size == 0:
        raise ValueError("Not enough samples for waterfall computation.")
    if len(samples) < 16:
        raise ValueError("Not enough samples for waterfall computation.")
    effective_nfft = min(int(nfft), len(samples))
    if effective_nfft <= 0:
        raise ValueError("Not enough samples for waterfall computation.")
    step = max(1, int(effective_nfft * (1.0 - overlap)))
    num_segments = max(1, (len(samples) - effective_nfft) // step + 1)
    window = np.hanning(effective_nfft)
    waterfall = np.zeros((num_segments, effective_nfft), dtype=np.float32)
    for i in range(num_segments):
        start = i * step
        end = start + effective_nfft
        if end > len(samples):
            break
        segment = samples[start:end] * window
        fft_vals = np.fft.fftshift(np.fft.fft(segment, n=effective_nfft))
        psd = np.abs(fft_vals) ** 2
        waterfall[i, :] = 10.0 * np.log10(psd + 1e-12)
    freqs = np.fft.fftshift(np.fft.fftfreq(effective_nfft, d=1.0 / sample_rate))
    times = np.arange(num_segments) * (step / sample_rate)
    return freqs, times, waterfall


def compute_eye(samples: np.ndarray, samples_per_symbol: int = 8) -> np.ndarray:
    """Reshape samples into overlapping eye-diagram traces.

    Each trace spans 2 symbols (2 * samples_per_symbol columns) with a
    1-symbol step (50% overlap). Trailing samples that do not fill a full
    trace are truncated.

    Returns:
      2D array of shape [num_traces, samples_per_symbol * 2].
    """
    samples = np.asarray(samples)
    if samples.size == 0:
        raise ValueError("Not enough samples for eye diagram computation.")
    if samples_per_symbol <= 0:
        raise ValueError("samples_per_symbol must be positive.")
    trace_len = int(samples_per_symbol * 2)
    step = int(samples_per_symbol)
    if len(samples) < trace_len:
        return np.asarray(samples).reshape(1, -1)
    num_traces = (len(samples) - trace_len) // step + 1
    if num_traces <= 0:
        raise ValueError("Not enough samples for eye diagram computation.")
    eye = np.zeros((num_traces, trace_len), dtype=samples.dtype)
    for i in range(num_traces):
        start = i * step
        eye[i, :] = samples[start : start + trace_len]
    return eye


def _piecewise_constant_residual(inst_freq: np.ndarray, period: int) -> float:
    """Reconstruction error of a piecewise-constant model at ``period`` samples.

    Splits the instantaneous frequency into blocks of ``period`` samples,
    replaces each block by its median (exactly what ``demod_2fsk`` does) and
    measures the mean absolute error of that reconstruction, normalised by the
    mean absolute instantaneous frequency.

    This is the discriminator that makes the period search work. The median
    alone is *not* enough: it always lands on a real sample value, so a wrong
    period still yields clean-looking symbols. The residual, by contrast, is
    near zero only when the block boundaries really do line up with the symbol
    boundaries. Measured on ``sample_data/fsk2.iq`` (true period 100): 0.0101
    at period 100 versus 0.2460 at period 99 and 0.2453 at 101.
    """
    n_symbols = inst_freq.size // period
    if n_symbols < 4:
        return float("inf")
    used = inst_freq[: n_symbols * period]
    medians = np.median(used.reshape(n_symbols, period), axis=1)
    scale = float(np.mean(np.abs(inst_freq)))
    if scale <= 0.0:
        return float("inf")
    return float(np.mean(np.abs(used - np.repeat(medians, period))) / scale)


def estimate_fsk_symbol_period(
    samples: np.ndarray,
    *,
    min_period: int = FSK_MIN_SYMBOL_PERIOD,
    max_residual: float = FSK_MAX_RECONSTRUCTION_RESIDUAL,
) -> int:
    """Recover the 2-FSK symbol period in samples; ``1`` when it cannot be told.

    A constant-modulus 2-FSK burst has a piecewise-constant instantaneous
    frequency: it only changes on a symbol boundary. Two properties of that
    structure make the period recoverable without any timing recovery loop.

    1. The normalised autocorrelation of the (mean-removed) instantaneous
       frequency decays linearly over the first few lags, and its slope is
       ``1/period``. That gives a coarse but very stable centre estimate: the
       first lag alone pins the period to within a few percent, which matters
       because the estimate is a *relative* one (a 1 % error is 10 samples at a
       period of 1000).

    2. Around that centre, the period that minimises
       :func:`_piecewise_constant_residual` is the true one -- and the minimum
       is sharp (see that function for measured numbers).

    Searching a narrow window instead of every candidate matters: any *divisor*
    of the true period also reconstructs perfectly, because whole blocks still
    fall inside single symbols. Those ties are broken by taking the largest
    near-optimal candidate, which is the true period rather than a divisor.
    Multiples are rejected automatically -- a block spanning two symbols
    reconstructs badly.

    Returns ``1`` -- meaning "no resolvable symbol structure, use the naive
    1-bit-per-sample path" -- when the signal is too short, is not constant
    modulus (BPSK/QPSK/noise/tone all measured), has a period below
    ``min_period``, or leaves a residual above ``max_residual``. It never
    guesses a period it cannot reconstruct.
    """
    x = np.asarray(samples, dtype=np.complex128).ravel()
    if x.size < 64:
        return 1

    inst_freq = np.diff(np.unwrap(np.angle(x)))
    centred = inst_freq - inst_freq.mean()
    if not np.any(centred):
        return 1

    # Autocorrelation via FFT; only lag 1 is used (best SNR, still in the
    # linear region for every period the MVP supports).
    nfft = 1 << int(np.ceil(np.log2(2 * centred.size)))
    spectrum = np.fft.rfft(centred, nfft)
    autocorr = np.fft.irfft(spectrum * np.conj(spectrum), nfft)
    if autocorr[0] <= 0.0:
        return 1
    r1 = float(autocorr[1] / autocorr[0])
    if not 0.0 < r1 < 1.0:
        return 1

    centre = 1.0 / (1.0 - r1)
    if centre < min_period:
        return 1

    # Window scaled to the estimate: the autocorrelation centre is good to a
    # few percent, not to a sample.
    tolerance = max(4, int(np.ceil(0.06 * centre)))
    lo = max(min_period, int(np.floor(centre)) - tolerance)
    hi = int(np.ceil(centre)) + tolerance

    scored = [
        (_piecewise_constant_residual(inst_freq, period), period)
        for period in range(lo, hi + 1)
    ]
    best_residual = min(residual for residual, _ in scored)
    best_period = max(
        period for residual, period in scored if residual == best_residual
    )
    if best_residual > max(max_residual, 2.0 / best_period):
        return 1

    # Divisor ties: keep the largest candidate that is (almost) as good.
    return max(period for residual, period in scored if residual <= best_residual * 1.6)


def estimate_symbol_rate(samples: np.ndarray, sample_rate: float) -> float:
    """Estimate symbol rate using the |x|² spectral-line method.

    For PSK/QAM signals, |x(t)|² produces spectral lines at ±Rs
    (the symbol rate). The strongest spectral peak in the squared
    signal's PSD (excluding DC) gives Rs.

    Returns 0.0 when no clear peak is found.
    """
    samples = np.asarray(samples)
    if samples.size < 64:
        return 0.0

    x_squared = np.abs(samples.astype(np.complex128)) ** 2
    x_squared = x_squared - np.mean(x_squared)

    nfft = min(8192, len(x_squared))
    freqs, psd_db = compute_psd(x_squared.astype(np.complex64), sample_rate, nfft=nfft)

    dc_margin = sample_rate * 0.01
    pos_mask = freqs > dc_margin
    if not np.any(pos_mask):
        return 0.0

    pos_freqs = freqs[pos_mask]
    pos_psd = psd_db[pos_mask]

    median_psd = float(np.median(pos_psd))
    peak_idx = int(np.argmax(pos_psd))
    if float(pos_psd[peak_idx]) < median_psd + 6.0:
        return 0.0

    return float(abs(pos_freqs[peak_idx]))


def estimate_cfo(samples: np.ndarray, sample_rate: float) -> float:
    """Estimate carrier frequency offset from constellation rotation.

    Uses linear fit to unwrapped phase to detect residual carrier rotation.
    Returns frequency offset in Hz.
    """
    samples = np.asarray(samples, dtype=np.complex128)
    if samples.size < 4:
        return 0.0

    phase = np.unwrap(np.angle(samples))
    n = np.arange(len(phase), dtype=np.float64)
    coeffs = np.polyfit(n, phase, 1)
    freq_offset = coeffs[0] * sample_rate / (2.0 * np.pi)
    return float(freq_offset)


def correct_cfo(
    samples: np.ndarray, sample_rate: float, freq_offset: float
) -> np.ndarray:
    """Apply CFO correction by counter-rotating the signal."""
    n = np.arange(len(samples), dtype=np.float64)
    correction = np.exp(-1j * 2.0 * np.pi * freq_offset * n / sample_rate)
    return samples * correction


def estimate_and_correct_cfo(
    samples: np.ndarray, sample_rate: float
) -> tuple[np.ndarray, float]:
    """Convenience wrapper: estimate CFO then counter-rotate.

    Calls :func:`estimate_cfo` to get the residual carrier offset in Hz,
    then :func:`correct_cfo` to remove it. Naive phase-slope estimator —
    same MVP caveats as :func:`estimate_cfo` (works on clean tones/PSK,
    not a PLL replacement).

    Returns:
        ``(corrected_samples, cfo_hz)`` where ``corrected_samples`` has
        the same shape/dtype-complex as the input.
    """
    cfo = float(estimate_cfo(samples, sample_rate))
    corrected = correct_cfo(np.asarray(samples), sample_rate, cfo)
    return corrected, cfo


# --------------------------------------------------------------------------- #
# Sampling-rate hypothesis testing
# --------------------------------------------------------------------------- #

# Rates a real capture chain plausibly runs at (sound cards, SDRs, lab gear).
STANDARD_SAMPLE_RATES = (
    8_000,
    16_000,
    22_050,
    32_000,
    44_100,
    48_000,
    96_000,
    192_000,
    250_000,
    500_000,
    1_000_000,
    1_228_800,
    2_000_000,
    2_400_000,
    5_000_000,
    10_000_000,
    20_000_000,
    40_000_000,
    61_440_000,
    122_880_000,
)
BW_GUARD_FACTORS = (2.0, 2.2, 2.5, 4.0)
OVERSAMPLING_HYPOTHESES = (2, 4, 8, 16)


def estimate_sampling_rate_candidates(
    samples: np.ndarray,
    sample_rate: float,
    bandwidth_estimate: float | None = None,
    symbol_rate_estimate: float | None = None,
    top_n: int = 5,
) -> list[dict]:
    """Rank plausible sampling rates instead of returning a single number.

    Raw ``.iq`` files carry no metadata, so the true sample rate is a
    hypothesis, not a measurement. This gathers candidates from three
    independent sources and scores each one:

    * Nyquist guards over the occupied bandwidth (``BW x {2.0, 2.2, 2.5, 4}``);
    * oversampling hypotheses over the ``|x|²`` symbol-rate line
      (``Rs x {2, 4, 8, 16}``);
    * standard capture rates that fall inside the plausible band.

    Scoring rewards Nyquist feasibility, proximity to the naive
    ``2.2 x BW`` hint, an integer oversampling ratio, and standard rates.

    Returns:
        Up to ``top_n`` dicts ``{"rate_hz", "score", "rationale"}`` sorted by
        descending score. Empty when no bandwidth information is available.

    This is hypothesis ranking, not blind estimation — see ``info.md`` §30.
    """
    x = np.asarray(samples)
    sr = float(sample_rate)

    bw = float(bandwidth_estimate) if bandwidth_estimate else 0.0
    if bw <= 0:
        if x.size == 0 or sr <= 0:
            return []
        freqs, psd_db = compute_psd(x, sr)
        bw = float(estimate_bandwidth(freqs, psd_db))
    if bw <= 0:
        return []

    rs = float(symbol_rate_estimate or 0.0)
    if rs <= 0 and x.size >= 64:
        try:
            rs = float(estimate_symbol_rate(x, sr))
        except Exception:
            rs = 0.0

    collected: dict[int, dict] = {}

    def _add(rate: float, rationale: str) -> None:
        if rate <= 0 or not np.isfinite(rate):
            return
        key = round(rate)
        entry = collected.setdefault(key, {"rate_hz": float(rate), "rationales": []})
        if rationale not in entry["rationales"]:
            entry["rationales"].append(rationale)

    for factor in BW_GUARD_FACTORS:
        _add(bw * factor, f"Nyquist x{factor:g} of occupied bandwidth")
    if rs > 0:
        for oversample in OVERSAMPLING_HYPOTHESES:
            _add(
                rs * oversample, f"{oversample}x oversampling of estimated symbol rate"
            )
    lo, hi = 2.0 * bw, 6.0 * bw
    for std in STANDARD_SAMPLE_RATES:
        if lo <= std <= hi:
            _add(float(std), "standard capture rate inside the plausible band")

    target = 2.2 * bw
    scored: list[dict] = []
    for entry in collected.values():
        rate = entry["rate_hz"]
        score = 0.40 if rate >= 2.0 * bw - 1e-9 else -0.25
        score += 0.25 * max(0.0, 1.0 - abs(rate - target) / target)
        if rs > 0:
            ratio = rate / rs
            if ratio >= 1.5:
                frac = abs(ratio - round(ratio))
                if frac <= 0.02:
                    score += 0.20
                elif frac <= 0.10:
                    score += 0.08
        if any(abs(rate - s) / s < 0.01 for s in STANDARD_SAMPLE_RATES):
            score += 0.15
        scored.append(
            {
                "rate_hz": rate,
                "score": round(float(min(max(score, 0.0), 1.0)), 4),
                "rationale": "; ".join(entry["rationales"]),
            }
        )

    scored.sort(key=lambda d: (-d["score"], d["rate_hz"]))
    return scored[: max(1, int(top_n))]


# --------------------------------------------------------------------------- #
# Constellation quality: EVM / MER / modulation order
# --------------------------------------------------------------------------- #

CONSTELLATIONS: dict[str, np.ndarray] = {}


def ideal_constellation(mode: str) -> np.ndarray | None:
    """Unit-average-power ideal constellation for ``mode``, or None if N/A.

    Every constellation returned has average symbol power 1.0, so a received
    signal normalised to unit power can be compared directly.
    """
    key = str(mode).upper().replace("_", "-").strip()

    if key in CONSTELLATIONS:
        return CONSTELLATIONS[key]

    points: np.ndarray | None = None
    if key == "BPSK":
        points = np.array([-1.0, 1.0], dtype=np.complex128)
    elif key in ("QPSK", "4-QAM", "4QAM"):
        levels = np.array([-1.0, 1.0])
        points = (levels[:, None] + 1j * levels[None, :]).ravel() / np.sqrt(2.0)
    elif key in ("16-QAM", "QAM16", "16QAM", "QAM"):
        levels = np.array([-3.0, -1.0, 1.0, 3.0])
        points = (levels[:, None] + 1j * levels[None, :]).ravel() / np.sqrt(10.0)
    elif key in ("8PSK", "8-PSK"):
        points = np.exp(1j * 2.0 * np.pi * np.arange(8) / 8.0)
    elif key in ("64-QAM", "QAM64", "64QAM"):
        levels = np.arange(-7, 8, 2, dtype=float)
        points = (levels[:, None] + 1j * levels[None, :]).ravel() / np.sqrt(42.0)

    if points is not None:
        CONSTELLATIONS[key] = points
    return points


def modulation_order(mode: str) -> int | None:
    """Bits per symbol for a modulation label (None when unknown)."""
    constellation = ideal_constellation(mode)
    if constellation is None or constellation.size < 2:
        return None
    return round(np.log2(constellation.size))


def compute_evm(
    samples: np.ndarray, mode: str = "BPSK", samples_per_symbol: int = 1
) -> dict:
    """Error-vector magnitude / MER for a phase-aligned constellation.

    Decides each received symbol to the nearest ideal point for the assumed
    modulation and measures the residual error. This doubles as a modulation
    sanity check: a wrong ``mode`` forces large errors, so a low EVM is
    independent evidence that the modulation guess is right.

    Caveat (MVP): no equalisation or carrier recovery, so a rotated or
    frequency-offset signal inflates EVM. Use :func:`estimate_and_correct_cfo`
    first when that matters.

    Returns:
        ``{"applicable", "mode", "modulation_order", "evm_percent",
        "mer_db", "snr_db_from_evm", "n_symbols"}``. ``applicable`` is False
        for modulations with no constellation (e.g. FSK).

    ``mer_db`` is ``None`` when the error vector is exactly zero (a noiseless
    synthetic capture), because the ratio is then unbounded. Reporting a
    literal ``inf`` would make the value non-serialisable -- ``Infinity`` is
    not valid JSON and JavaScript's ``JSON.parse`` rejects it -- so the
    absence of a measurable error is expressed as ``None`` instead.
    """
    constellation = ideal_constellation(mode)
    blank = {
        "applicable": False,
        "mode": str(mode),
        "modulation_order": None,
        "evm_percent": None,
        "mer_db": None,
        "snr_db_from_evm": None,
        "n_symbols": 0,
    }
    if constellation is None:
        return blank

    x = np.asarray(samples, dtype=np.complex128).ravel()
    if x.size == 0:
        return blank

    sps = max(1, int(samples_per_symbol or 1))
    if sps > 1:
        n_symbols = x.size // sps
        if n_symbols < 1:
            return blank
        # Sample at the centre of each symbol interval.
        x = x[: n_symbols * sps].reshape(n_symbols, sps).mean(axis=1)

    power = float(np.mean(np.abs(x) ** 2))
    if power <= 1e-20:
        return blank
    x = x / np.sqrt(power)

    distances = np.abs(x[:, None] - constellation[None, :])
    nearest = constellation[np.argmin(distances, axis=1)]

    error_power = float(np.mean(np.abs(x - nearest) ** 2))
    ideal_power = float(np.mean(np.abs(nearest) ** 2))
    if ideal_power <= 1e-20:
        return blank

    evm_ratio = float(np.sqrt(error_power / ideal_power))
    evm_percent = evm_ratio * 100.0
    # A zero error vector means an unbounded MER. Keep it JSON-safe: `None`
    # rather than `inf`, which is not representable in strict JSON.
    mer_db = float(-20.0 * np.log10(evm_ratio)) if evm_ratio > 0 else None

    return {
        "applicable": True,
        "mode": str(mode),
        "modulation_order": modulation_order(mode),
        "evm_percent": evm_percent,
        "mer_db": mer_db,
        "snr_db_from_evm": mer_db,
        "n_symbols": int(x.size),
    }
