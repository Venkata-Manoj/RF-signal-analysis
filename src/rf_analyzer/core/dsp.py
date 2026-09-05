"""PSD/waterfall/parameter estimation. Full spec: info.md §12.2."""

from __future__ import annotations


def compute_psd(samples, sample_rate: float, nfft: int = 4096):
    """TODO (Milestone 3): implement per info.md §14.2."""
    raise NotImplementedError("compute_psd not implemented yet (Milestone 3)")


def compute_waterfall(samples, sample_rate: float, nfft: int = 1024, overlap: float = 0.5):
    """TODO (Milestone 3)."""
    raise NotImplementedError("compute_waterfall not implemented yet (Milestone 3)")


def estimate_center_frequency(freqs, psd_db) -> float:
    """TODO (Milestone 3)."""
    raise NotImplementedError("estimate_center_frequency not implemented yet (Milestone 3)")


def estimate_bandwidth(freqs, psd_db, threshold_db: float = 10.0) -> float:
    """TODO (Milestone 3)."""
    raise NotImplementedError("estimate_bandwidth not implemented yet (Milestone 3)")


def estimate_snr(psd_db) -> float:
    """TODO (Milestone 3)."""
    raise NotImplementedError("estimate_snr not implemented yet (Milestone 3)")


def compute_eye(samples, samples_per_symbol: int = 8):
    """TODO (Milestone 3)."""
    raise NotImplementedError("compute_eye not implemented yet (Milestone 3)")
