"""PSD/waterfall/parameter estimation. Full spec: info.md §12.2."""

from __future__ import annotations

import numpy as np


def compute_psd(
    samples: np.ndarray, sample_rate: float, nfft: int = 4096
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectral density.
    Returns frequency axis and PSD in dB.
    """
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
    """
    Estimate center frequency as PSD peak.
    """
    freqs = np.asarray(freqs)
    psd_db = np.asarray(psd_db)
    if freqs.size == 0 or psd_db.size == 0:
        raise ValueError("Empty spectrum input.")
    peak_index = int(np.argmax(psd_db))
    return float(freqs[peak_index])


def estimate_bandwidth(
    freqs: np.ndarray, psd_db: np.ndarray, threshold_db: float = 10.0
) -> float:
    """
    Estimate occupied bandwidth using noise-floor threshold.
    """
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
    """
    Rough SNR estimate:
    SNR = peak PSD - median noise floor
    """
    psd_db = np.asarray(psd_db)
    if psd_db.size == 0:
        raise ValueError("Empty spectrum input.")
    noise_floor = float(np.median(psd_db))
    peak = float(np.max(psd_db))
    return peak - noise_floor


def compute_waterfall(
    samples: np.ndarray, sample_rate: float, nfft: int = 1024, overlap: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute waterfall matrix.

    Returns:
      freqs: frequency axis
      times: time axis
      waterfall_db: 2D array [time_bins, freq_bins]
    """
    samples = np.asarray(samples)
    if samples.size == 0:
        raise ValueError("Not enough samples for waterfall computation.")

    step = max(1, int(nfft * (1.0 - overlap)))
    num_segments = max(1, (len(samples) - nfft) // step + 1)

    window = np.hanning(nfft)
    waterfall = np.zeros((num_segments, nfft), dtype=np.float32)

    for i in range(num_segments):
        start = i * step
        end = start + nfft

        if end > len(samples):
            break

        segment = samples[start:end] * window
        fft_vals = np.fft.fftshift(np.fft.fft(segment, n=nfft))
        psd = np.abs(fft_vals) ** 2
        waterfall[i, :] = 10.0 * np.log10(psd + 1e-12)

    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate))
    times = np.arange(num_segments) * (step / sample_rate)

    return freqs, times, waterfall


def compute_eye(samples: np.ndarray, samples_per_symbol: int = 8) -> np.ndarray:
    """
    Reshape samples into overlapping eye-diagram traces.

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
        raise ValueError("Not enough samples for eye diagram computation.")

    num_traces = (len(samples) - trace_len) // step + 1
    if num_traces <= 0:
        raise ValueError("Not enough samples for eye diagram computation.")

    eye = np.zeros((num_traces, trace_len), dtype=samples.dtype)
    for i in range(num_traces):
        start = i * step
        eye[i, :] = samples[start : start + trace_len]

    return eye
