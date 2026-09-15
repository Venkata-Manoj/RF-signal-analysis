"""Higher-Order Cumulant (HOC) modulation classifier.

Replaces the naive variance heuristic in pipeline.py with a mathematically
rigorous approach based on 2nd and 4th order cumulants that are invariant
to carrier phase and amplitude scaling.

Theoretical cumulant values (unit power, zero mean):
  BPSK:  |C40| ≈ 2.0, |C42| ≈ 2.0
  QPSK:  |C40| ≈ 1.0, |C42| ≈ 1.0
  8PSK:  |C40| ≈ 0.0, |C42| ≈ 0.0
  16QAM: |C40| ≈ 0.68, |C42| ≈ 0.68
  2-FSK: Detected via instantaneous frequency variance
"""

from __future__ import annotations

import numpy as np


def compute_cumulants(samples: np.ndarray) -> dict:
    """Compute 2nd and 4th order cumulants for AMC.

    Normalizes input to unit power before computation.

    Returns dict with keys: C20, C21, C40, C42
    """
    x = np.asarray(samples, dtype=np.complex128).ravel()
    if x.size < 4:
        return {"C20": 0.0, "C21": 0.0, "C40": 0.0, "C42": 0.0}

    # Normalize to unit power
    power = float(np.mean(np.abs(x) ** 2))
    if power < 1e-12:
        return {"C20": 0.0, "C21": 0.0, "C40": 0.0, "C42": 0.0}
    x = x / np.sqrt(power)

    # Moments
    M20 = complex(np.mean(x**2))
    M21 = float(np.mean(np.abs(x) ** 2))
    M40 = complex(np.mean(x**4))
    M42 = complex(np.mean(np.abs(x) ** 2 * x**2))

    # Cumulants
    C20 = M20
    C21 = M21
    C40 = M40 - 3.0 * M20**2
    C42 = M42 - abs(M20) ** 2 - 2.0 * M21**2

    return {
        "C20": C20,
        "C21": C21,
        "C40": C40,
        "C42": C42,
    }


def _is_fsk(samples: np.ndarray) -> bool:
    """Detect FSK via low instantaneous-frequency variance."""
    if len(samples) < 4:
        return False
    phase = np.unwrap(np.angle(samples.astype(np.complex128)))
    inst_freq = np.diff(phase)
    return float(np.var(inst_freq)) < 1.0


def classify_modulation(
    samples: np.ndarray,
) -> tuple[str, float, list[str]]:
    """Classify modulation using Higher-Order Cumulants.

    Returns ``(estimated_type, confidence, alternatives)``.

    ``estimated_type`` is one of:
    ``'BPSK'``, ``'QPSK'``, ``'8PSK'``, ``'16-QAM'``, ``'2-FSK'``, ``'UNKNOWN'``

    ``confidence`` is in ``[0.0, 1.0]``.
    """
    flat = np.asarray(samples).ravel()
    if flat.size < 4:
        return "UNKNOWN", 0.5, ["BPSK", "QPSK", "2-FSK"]

    # --- Check FSK first (instantaneous-frequency heuristic) ---
    if _is_fsk(flat):
        return "2-FSK", 0.75, ["BPSK", "QPSK"]

    # --- HOC classification for PSK / QAM ---
    c = compute_cumulants(flat)
    c40_abs = abs(c["C40"])
    c42_abs = abs(c["C42"])

    if c40_abs > 1.5:
        return "BPSK", 0.85, ["QPSK", "2-FSK"]
    if 0.8 < c40_abs <= 1.5:
        return "QPSK", 0.80, ["BPSK", "8PSK"]
    if c40_abs < 0.3:
        if c42_abs < 0.5:
            return "8PSK", 0.70, ["QPSK", "16-QAM"]
        return "16-QAM", 0.65, ["8PSK", "QPSK"]

    # Fall-through: moderate C40 region — ambiguous
    return "UNKNOWN", 0.50, ["BPSK", "QPSK", "2-FSK"]
