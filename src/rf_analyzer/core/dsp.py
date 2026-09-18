"""PSD/waterfall/parameter estimation. Full spec: info.md §12.2."""

from __future__ import annotations

import numpy as np


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
    """Estimate occupied bandwidth using noise-floor threshold."""
    freqs = np.asarray(freqs)
    psd_db = np.asarray(psd_db)
    if freqs.size == 0 or psd_db.size == 0:
        raise ValueError("Empty spectrum input.")
    noise_floor = float(np.median(psd_db))
    signal_mask = psd_db > (noise_floor + threshold_db)
    if not np.any(signal_mask):
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
