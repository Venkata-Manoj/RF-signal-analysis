"""Real-channel impairments: CFO, Doppler drift, multipath, IQ imbalance, AWGN.

``waveform.synth`` / ``burst_in_noise`` model only AWGN, which is a laboratory
condition: a real capture arrives with a carrier offset, a slowly drifting
oscillator, echoes, and an imperfect quadrature front end. This module applies
those impairments in a fixed, documented order so a robustness sweep means one
thing:

1. **Carrier offset + linear Doppler drift** -- a complex exponential with a
   quadratic phase term. Applied first, while the signal is still clean, so the
   phase trajectory is exact.
2. **Multipath** -- a causal FIR filter ``y[n] = sum_k g_k x[n - d_k]`` with
   integer sample delays ``d_k`` and (possibly complex) gains ``g_k``.
3. **IQ imbalance** -- symmetric gain mismatch plus quadrature skew, applied to
   the real/imaginary branches.
4. **AWGN last** -- complex Gaussian noise at the requested ``snr_db``,
   measured against the post-impairment signal power (the same convention as
   ``waveform.add_awgn``).

Everything is seeded: the only stochastic step is the AWGN, drawn from
``numpy.random.default_rng(seed)``, so the same arguments always produce the
same capture. Pure NumPy, no compiled extensions.

Units: ``cfo_hz`` in Hz, ``doppler_rate_hz_s`` in Hz/s, multipath delays in
**samples**, ``iq_gain_imb`` as a fraction (``0`` = perfect, ``0.2`` = the I
branch 10% hot and the Q branch 10% cold -- see below), ``iq_phase_imb_deg``
in degrees (``0`` = perfect quadrature), ``snr_db`` against the impaired
signal power (``None`` = no noise).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "apply_channel",
    "channel_fir",
    "correct_iq_imbalance",
    "estimate_iq_imbalance",
    "iq_imbalance_metrics",
]


def _as_complex(samples: np.ndarray) -> np.ndarray:
    """Return ``samples`` as 1-D ``complex128`` without touching the caller."""
    x = np.asarray(samples)
    if np.iscomplexobj(x):
        return x.astype(np.complex128, copy=False).ravel()
    return x.astype(np.float64).ravel().astype(np.complex128)


def channel_fir(multipath: list[tuple[int, complex]] | tuple) -> np.ndarray:
    """Build the causal FIR tap array for a ``multipath`` profile.

    ``multipath`` is a sequence of ``(delay_samples, gain)`` pairs; the
    returned array has ``taps[d] == gain`` for each pair and ``0`` elsewhere,
    so ``np.convolve(x, taps)[:len(x)]`` is exactly the channel. The default
    profile ``[(0, 1.0)]`` yields ``[1.0]`` (identity). Delays must be
    non-negative integers; gains may be real or complex.
    """
    pairs = list(multipath) if multipath is not None else [(0, 1.0)]
    if len(pairs) == 0:
        raise ValueError("multipath must contain at least one (delay, gain) tap")
    delays: list[int] = []
    gains: list[complex] = []
    for entry in pairs:
        delay, gain = entry
        delay = int(delay)
        if delay < 0:
            raise ValueError(f"multipath delay must be >= 0 samples, got {delay}")
        if not np.isfinite(complex(gain)):
            raise ValueError(f"multipath gain must be finite, got {gain!r}")
        delays.append(delay)
        gains.append(complex(gain))
    taps = np.zeros(max(delays) + 1, dtype=np.complex128)
    for delay, gain in zip(delays, gains):
        taps[delay] += gain
    return taps


def apply_channel(
    samples: np.ndarray,
    sample_rate: float,
    cfo_hz: float = 0.0,
    doppler_rate_hz_s: float = 0.0,
    multipath: list[tuple[int, complex]] | tuple = ((0, 1.0),),
    iq_gain_imb: float = 0.0,
    iq_phase_imb_deg: float = 0.0,
    snr_db: float | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Pass ``samples`` through the impaired-channel model.

    Args:
        samples: Complex baseband input.
        sample_rate: Sample rate in Hz (``> 0``); sets the time axis for the
            CFO/Doppler phase ramp.
        cfo_hz: Constant carrier offset in Hz.
        doppler_rate_hz_s: Linear frequency drift in Hz/s. The instantaneous
            offset is ``cfo_hz + doppler_rate_hz_s * t`` and the applied phase
            is its integral, ``2*pi*(cfo*t + 0.5*rate*t**2)``.
        multipath: ``(delay_samples, gain)`` pairs; ``[(0, 1.0)]`` is identity.
        iq_gain_imb: Symmetric fractional gain mismatch. The I branch is scaled
            by ``(1 + g/2)`` and Q by ``(1 - g/2)``; ``0`` is perfect.
        iq_phase_imb_deg: Quadrature skew in degrees. The Q branch is replaced
            by ``I' * sin(phi) + Q' * cos(phi)``; ``0`` is perfect quadrature.
        snr_db: AWGN level against the impaired signal power, or ``None`` for
            no noise.
        seed: Seeds the AWGN generator only; every other stage is exact.

    Returns:
        Impaired samples as ``complex64``, same length as the input.
    """
    fs = float(sample_rate)
    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate!r}")
    x = _as_complex(samples)
    n = x.size
    if n == 0:
        return np.zeros(0, dtype=np.complex64)
    cfo = float(cfo_hz)
    rate = float(doppler_rate_hz_s)
    if not np.isfinite(cfo) or not np.isfinite(rate):
        raise ValueError("cfo_hz and doppler_rate_hz_s must be finite")
    gain = float(iq_gain_imb)
    phase_deg = float(iq_phase_imb_deg)
    if not np.isfinite(gain) or not np.isfinite(phase_deg):
        raise ValueError("iq_gain_imb and iq_phase_imb_deg must be finite")

    # 1. Carrier offset + linear Doppler drift (exact complex exponential).
    if cfo != 0.0 or rate != 0.0:
        t = np.arange(n, dtype=np.float64) / fs
        ramp = 2.0 * np.pi * (cfo * t + 0.5 * rate * t * t)
        x = x * np.exp(1j * ramp)

    # 2. Multipath FIR (causal, zero-padded at the start).
    taps = channel_fir(multipath)
    if not (taps.size == 1 and taps[0] == 1.0):
        x = np.convolve(x, taps, mode="full")[:n]

    # 3. IQ imbalance: symmetric gain, then quadrature skew.
    if gain != 0.0 or phase_deg != 0.0:
        phi = float(np.deg2rad(phase_deg))
        i_branch = np.real(x) * (1.0 + gain / 2.0)
        q_branch = np.imag(x) * (1.0 - gain / 2.0)
        q_out = i_branch * np.sin(phi) + q_branch * np.cos(phi)
        x = i_branch + 1j * q_out

    # 4. AWGN last, at snr_db against the impaired power (seeded).
    if snr_db is not None:
        level = float(snr_db)
        if not np.isfinite(level):
            raise ValueError(f"snr_db must be finite, got {snr_db!r}")
        power = float(np.mean(np.abs(x) ** 2))
        if power > 0.0:
            rng = np.random.default_rng(int(seed))
            noise_power = power / (10.0 ** (level / 10.0))
            noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(
                2.0
            )
            x = x + noise * np.sqrt(noise_power)

    return np.asarray(x, dtype=np.complex64)


def iq_imbalance_metrics(samples: np.ndarray) -> dict:
    """Blind IQ-imbalance statistics for a complex capture.

    For a balanced quadrature signal ``E[I^2] == E[Q^2]`` and
    ``E[I*Q] == 0``. Gain mismatch moves power between the branches and
    quadrature skew correlates them, so:

    * ``gain_metric = (P_I - P_Q) / (P_I + P_Q)`` -- ``0`` when balanced,
      positive when the I branch is hot.
    * ``quad_metric = mean(I*Q) / sqrt(P_I*P_Q)`` -- ``0`` for perfect
      quadrature, approximately ``sin(phi)`` under the skew model used by
      :func:`apply_channel` (exact when the underlying I/Q are uncorrelated
      with equal power, e.g. QPSK).
    * ``magnitude = sqrt(gain_metric**2 + quad_metric**2)`` -- one number for
      "how imbalanced", ``0`` for a clean capture.

    Returns ``NaN`` metrics for an empty or zero-power capture rather than
    raising: there is nothing to measure.
    """
    x = _as_complex(samples)
    blank = {
        "gain_metric": float("nan"),
        "quad_metric": float("nan"),
        "magnitude": float("nan"),
    }
    if x.size == 0:
        return blank
    i_vals = np.real(x)
    q_vals = np.imag(x)
    p_i = float(np.mean(i_vals**2))
    p_q = float(np.mean(q_vals**2))
    denom = p_i + p_q
    if not np.isfinite(denom) or denom <= 0.0:
        return blank
    gain_metric = (p_i - p_q) / denom
    scale = np.sqrt(p_i * p_q)
    quad_metric = (
        float(np.mean(i_vals * q_vals) / scale) if scale > 0.0 else float("nan")
    )
    magnitude = (
        float(np.sqrt(gain_metric**2 + quad_metric**2))
        if np.isfinite(quad_metric)
        else abs(float(gain_metric))
    )
    return {
        "gain_metric": float(gain_metric),
        "quad_metric": float(quad_metric),
        "magnitude": float(magnitude),
    }


def estimate_iq_imbalance(samples: np.ndarray) -> dict:
    """Invert :func:`iq_imbalance_metrics` into (gain, phase) estimates.

    Under :func:`apply_channel`'s symmetric model with an uncorrelated,
    equal-power baseband (QPSK is the calibration case):

    * ``r = sqrt(P_I / P_Q) = (1+g/2) / (1-g/2)``, hence
      ``g = 2*(r-1) / (r+1)``;
    * ``sin(phi) = mean(I*Q) / mean(I^2)`` (the skew term; uses the I power
      because the correlator sits on the I branch).

    Returns ``{"iq_gain_imb", "iq_phase_imb_deg"}`` with ``None`` values when
    the capture has nothing to estimate from. This is a coarse blind
    estimator, not a calibration -- it assumes the underlying signal is
    balanced, so a BPSK capture (all energy on I) must not be passed here.
    """
    x = _as_complex(samples)
    blank: dict = {"iq_gain_imb": None, "iq_phase_imb_deg": None}
    if x.size == 0:
        return blank
    i_vals = np.real(x)
    q_vals = np.imag(x)
    p_i = float(np.mean(i_vals**2))
    p_q = float(np.mean(q_vals**2))
    if not np.isfinite(p_i) or not np.isfinite(p_q) or p_i <= 0.0 or p_q <= 0.0:
        return blank
    ratio = float(np.sqrt(p_i / p_q))
    gain = 2.0 * (ratio - 1.0) / (ratio + 1.0)
    sine = float(np.mean(i_vals * q_vals) / p_i)
    sine = max(-1.0, min(1.0, sine))
    phase_deg = float(np.rad2deg(np.arcsin(sine)))
    return {"iq_gain_imb": float(gain), "iq_phase_imb_deg": float(phase_deg)}


def correct_iq_imbalance(
    samples: np.ndarray,
    iq_gain_imb: float,
    iq_phase_imb_deg: float,
) -> np.ndarray:
    """Exact inverse of :func:`apply_channel`'s IQ-imbalance stage.

    Given the same ``(iq_gain_imb, iq_phase_imb_deg)`` the channel was built
    with, this recovers the pre-imbalance samples to floating-point precision::

        I = Re(y) / (1 + g/2)
        Q = (Im(y) - Re(y)*sin(phi)) / (cos(phi) * (1 - g/2))

    Raises ``ValueError`` when the parameters are non-invertible (a zeroed
    branch at ``|g| == 2``, or ``cos(phi) == 0`` near 90 degrees).
    """
    gain = float(iq_gain_imb)
    phi = float(np.deg2rad(float(iq_phase_imb_deg)))
    if not np.isfinite(gain) or not np.isfinite(phi):
        raise ValueError("iq imbalance parameters must be finite")
    scale_i = 1.0 + gain / 2.0
    scale_q = 1.0 - gain / 2.0
    cos_phi = float(np.cos(phi))
    if abs(scale_i) < 1e-12 or abs(scale_q) < 1e-12:
        raise ValueError(f"iq_gain_imb={gain!r} zeroes a branch and cannot be inverted")
    if abs(cos_phi) < 1e-12:
        raise ValueError(
            f"iq_phase_imb_deg={iq_phase_imb_deg!r} is non-invertible (cos phase ~ 0)"
        )
    y = _as_complex(samples)
    if y.size == 0:
        return np.zeros(0, dtype=np.complex64)
    i_prime = np.real(y)
    q_prime = (np.imag(y) - i_prime * float(np.sin(phi))) / cos_phi
    i_clean = i_prime / scale_i
    q_clean = q_prime / scale_q
    return np.asarray(i_clean + 1j * q_clean, dtype=np.complex64)
