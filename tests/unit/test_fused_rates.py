"""Fused rate estimators exercised the way the pipeline calls them.

``pipeline.analyze_file`` replaced ``bandwidth * 2.2`` and the single
periodogram ``|x|^2`` line with
:func:`rate_est.estimate_symbol_rate_fused` and
:func:`rate_est.estimate_sampling_rate_fused`, fed by
:func:`dsp.estimate_bandwidth` and :func:`dsp.estimate_fsk_symbol_period`.
These tests pin that wiring against the oversampled generator
(:mod:`rf_analyzer.core.waveform`):

* clean BPSK/QPSK recover the true symbol rate within 5%;
* the same modulations still recover it at 20, 10 and 0 dB;
* a clean 2-FSK capture is reported by the *direct* path (the recovered
  period is a measurement, not a detection) and the ``|x|^2`` path stays
  out of it;
* AWGN and an unmodulated tone produce ``0.0`` -- "not measurable" -- plus
  a warning naming what was tried, never a plausible-looking rate;
* a burst embedded in noise (the realistic capture) still corroborates at
  20, 10 and 0 dB;
* the sampling-rate hint refuses on noise and degrades to a flagged
  feasibility floor (never a measurement) when only one constraint exists.

``0.0`` means "not measurable" throughout. A test that asserts a nonzero
rate for noise, or an exact rate without agreement between independent
paths, would be testing the thing this module exists to refuse.
"""

from __future__ import annotations

import numpy as np
import pytest

from rf_analyzer.core import waveform as wf
from rf_analyzer.core.dsp import (
    compute_psd,
    estimate_bandwidth,
    estimate_fsk_symbol_period,
)
from rf_analyzer.core.rate_est import (
    CONFIDENCE_CORROBORATED,
    CONFIDENCE_DIRECT,
    estimate_sampling_rate_fused,
    estimate_symbol_rate_fused,
)

#: Sample rate for every synthetic case. 1 MHz with 8 samples per symbol puts
#: the linear symbol rate at a round 125 kHz, far enough from the search
#: bounds that a sign error in the cycle-frequency axis cannot look correct.
SAMPLE_RATE = 1_000_000.0
SPS = 8
TRUE_SYMBOL_RATE = SAMPLE_RATE / SPS

#: 2-FSK oversampling for the direct-path tests. The recovered period is 16
#: samples, so the true rate is a round 62.5 kHz.
FSK_SPS = 16
TRUE_FSK_RATE = SAMPLE_RATE / FSK_SPS

#: Capture length for the linear tests. Long enough for the weak ``|x|^2``
#: line and the cyclic peak to agree (both need Welch averaging gain), short
#: enough that each fused call stays well under a second.
N_BITS_LINEAR = 4000

#: Noise levels swept for every "at 0-20 dB" case, plus the noiseless control
#: (``None``) wherever a rate is asserted.
SNR_DB_LEVELS = (20.0, 10.0, 0.0)


def _bandwidth(samples: np.ndarray, sample_rate: float) -> float:
    """Occupied bandwidth exactly as the pipeline measures it.

    :func:`dsp.compute_psd` looks at the first 4096 samples; on a
    burst-in-noise capture that window straddles the noise lead and the burst
    start, which is the realistic input, not a laboratory one.
    """
    freqs, psd_db = compute_psd(np.asarray(samples), sample_rate)
    return float(estimate_bandwidth(freqs, psd_db))


def _fuse_like_pipeline(
    samples: np.ndarray, sample_rate: float, modulation_hint: str | None
) -> dict:
    """Call the fused symbol-rate estimator the way ``analyze_file`` does.

    Bandwidth comes from :func:`dsp.estimate_bandwidth`, the FSK period from
    :func:`dsp.estimate_fsk_symbol_period` (``1`` when unresolvable, passed
    through only for an FSK hint -- otherwise the pipeline passes ``None``),
    and the modulation hint is the demodulator mode.
    """
    flat = np.asarray(samples)
    bandwidth = _bandwidth(flat, sample_rate)
    period = int(estimate_fsk_symbol_period(flat))
    fsk_period = (
        period
        if modulation_hint is not None and "FSK" in modulation_hint.upper()
        else None
    )
    return estimate_symbol_rate_fused(
        flat,
        sample_rate,
        bandwidth_estimate=bandwidth if bandwidth > 0.0 else None,
        fsk_period=fsk_period,
        modulation_hint=modulation_hint,
    )


# --------------------------------------------------------------------------- #
# Clean linear modulations: the rate must be right, confidently.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("scheme", ["BPSK", "QPSK"])
def test_fused_recovers_clean_linear_rate_within_5_percent(scheme: str) -> None:
    """A clean oversampled burst earns a corroborated number, not a guess."""
    samples, _ = wf.synth(
        scheme, N_BITS_LINEAR, sps=SPS, snr_db=None, seed=1, sample_rate=SAMPLE_RATE
    )
    result = _fuse_like_pipeline(samples, SAMPLE_RATE, scheme)

    assert result["method"] == "corroborated", result
    assert result["confidence"] == pytest.approx(CONFIDENCE_CORROBORATED)
    assert result["symbol_rate_hz"] == pytest.approx(TRUE_SYMBOL_RATE, rel=0.05)
    assert result["components"]["envelope"] is not None
    assert result["components"]["cyclostationary"] is not None


@pytest.mark.parametrize("scheme", ["BPSK", "QPSK"])
@pytest.mark.parametrize("snr_db", SNR_DB_LEVELS)
def test_fused_recovers_noisy_linear_rate_within_5_percent(
    scheme: str, snr_db: float
) -> None:
    """The same modulations at 20, 10 and 0 dB: still one honest number."""
    samples, _ = wf.synth(
        scheme, N_BITS_LINEAR, sps=SPS, snr_db=snr_db, seed=1, sample_rate=SAMPLE_RATE
    )
    result = _fuse_like_pipeline(samples, SAMPLE_RATE, scheme)

    assert result["method"] == "corroborated", (scheme, snr_db, result)
    assert result["symbol_rate_hz"] == pytest.approx(TRUE_SYMBOL_RATE, rel=0.05)


# --------------------------------------------------------------------------- #
# 2-FSK: a recovered period is a direct measurement.
# --------------------------------------------------------------------------- #


def test_fused_reports_clean_fsk_period_as_direct() -> None:
    """The FSK period outranks the feature detectors; it does not need them."""
    samples, _ = wf.synth(
        "2-FSK",
        N_BITS_LINEAR,
        sps=FSK_SPS,
        snr_db=None,
        seed=1,
        sample_rate=SAMPLE_RATE,
    )
    period = int(estimate_fsk_symbol_period(np.asarray(samples)))
    assert period > 1, "the noiseless FSK reference must have a recoverable period"
    assert period == FSK_SPS

    result = _fuse_like_pipeline(samples, SAMPLE_RATE, "2-FSK")

    assert result["method"] == "direct", result
    assert result["confidence"] == pytest.approx(CONFIDENCE_DIRECT)
    assert result["symbol_rate_hz"] == pytest.approx(SAMPLE_RATE / period, rel=1e-9)
    assert result["symbol_rate_hz"] == pytest.approx(TRUE_FSK_RATE, rel=0.05)
    # The |x|^2 line is meaningless for a constant-envelope signal (it once
    # reported 395 kHz for a 1 kHz burst), so it must stay out of the fusion.
    assert result["components"]["envelope"] is None


# --------------------------------------------------------------------------- #
# Noise and tone: 0.0 plus a warning, never a rate.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [0, 11])
def test_fused_returns_zero_on_awgn_with_warning(seed: int) -> None:
    """White noise has no symbol rate, and the report must say what it tried."""
    rng = np.random.default_rng(seed)
    n = 16000
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)

    result = _fuse_like_pipeline(noise, SAMPLE_RATE, None)

    assert result["symbol_rate_hz"] == 0.0, result
    assert result["confidence"] == 0.0
    assert result["method"] in ("none", "single", "disagreement")
    assert result["warnings"], "a refusal must name what was tried"


@pytest.mark.parametrize("noise_amplitude", [0.0, 0.5])
def test_fused_returns_zero_on_tone_with_warning(noise_amplitude: float) -> None:
    """An unmodulated tone has no symbol rate at any noise level.

    The pipeline classifies a tone as 2-FSK (its FSK test is a narrowband
    test), so the FSK hint is the pipeline-faithful one; the bare hint is
    covered too, because neither may invent a rate.
    """
    rng = np.random.default_rng(5)
    n = 16000
    tone = np.exp(1j * 2 * np.pi * 0.05 * np.arange(n))
    if noise_amplitude:
        tone = tone + noise_amplitude * (
            rng.standard_normal(n) + 1j * rng.standard_normal(n)
        )

    for hint in ("2-FSK", None):
        result = _fuse_like_pipeline(tone, SAMPLE_RATE, hint)

        assert result["symbol_rate_hz"] == 0.0, (hint, result)
        assert result["confidence"] == 0.0
        assert result["warnings"], "a refusal must name what was tried"


# --------------------------------------------------------------------------- #
# Burst in noise: the realistic capture, at 0-20 dB.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("scheme", ["BPSK", "QPSK"])
@pytest.mark.parametrize("snr_db", SNR_DB_LEVELS)
def test_fused_corroborates_burst_in_noise(scheme: str, snr_db: float) -> None:
    """A frame-shaped burst surrounded by noise still earns one number."""
    samples, _, _region = wf.burst_in_noise(
        scheme,
        N_BITS_LINEAR,
        snr_db=snr_db,
        sps=SPS,
        lead=2048,
        tail=2048,
        seed=4,
        sample_rate=SAMPLE_RATE,
    )
    result = _fuse_like_pipeline(samples, SAMPLE_RATE, scheme)

    assert result["method"] == "corroborated", (scheme, snr_db, result)
    assert result["symbol_rate_hz"] == pytest.approx(TRUE_SYMBOL_RATE, rel=0.05)


# --------------------------------------------------------------------------- #
# Sampling-rate hint: refuse on nothing, floor on one constraint.
# --------------------------------------------------------------------------- #


def test_sampling_fused_refuses_on_noise_with_warning() -> None:
    """No bandwidth and no symbol rate means no sampling rate -- and it says so."""
    rng = np.random.default_rng(23)
    n = 16000
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)

    result = estimate_sampling_rate_fused(
        noise, SAMPLE_RATE, bandwidth_estimate=0.0, symbol_rate_hz=0.0
    )

    assert result["sampling_rate_hz"] == 0.0
    assert result["confidence"] == 0.0
    assert result["warnings"]
    assert "not measurable" in result["warnings"][0]


def test_sampling_fused_gives_nyquist_floor_on_clean_linear() -> None:
    """With both constraints the hint respects Nyquist instead of guessing."""
    samples, _ = wf.synth(
        "BPSK", N_BITS_LINEAR, sps=SPS, snr_db=None, seed=1, sample_rate=SAMPLE_RATE
    )
    flat = np.asarray(samples)
    bandwidth = _bandwidth(flat, SAMPLE_RATE)
    assert bandwidth > 0.0, "the clean capture must have a measurable span"

    symbol = _fuse_like_pipeline(flat, SAMPLE_RATE, "BPSK")
    assert symbol["symbol_rate_hz"] > 0.0

    result = estimate_sampling_rate_fused(
        flat,
        SAMPLE_RATE,
        bandwidth_estimate=bandwidth,
        symbol_rate_hz=float(symbol["symbol_rate_hz"]),
    )

    assert result["sampling_rate_hz"] >= 2.0 * bandwidth
    assert result["sampling_rate_hz"] > 0.0
    assert result["candidates"]


def test_sampling_fused_tone_falls_back_to_bandwidth_floor() -> None:
    """A tone has no symbol rate, but its bandwidth still bounds the sampling.

    This mirrors the portal check: the sampling hint stays positive on a
    narrowband capture while the symbol rate honestly refuses -- and the
    single-constraint warning flags the floor as a floor, not a measurement.
    """
    n = 16000
    tone = np.exp(1j * 2 * np.pi * 0.05 * np.arange(n))
    flat = np.asarray(tone)
    bandwidth = _bandwidth(flat, SAMPLE_RATE)

    symbol = _fuse_like_pipeline(flat, SAMPLE_RATE, "2-FSK")
    assert symbol["symbol_rate_hz"] == 0.0

    result = estimate_sampling_rate_fused(
        flat,
        SAMPLE_RATE,
        bandwidth_estimate=bandwidth if bandwidth > 0.0 else None,
        symbol_rate_hz=None,
    )

    if bandwidth > 0.0:
        assert result["sampling_rate_hz"] > 0.0
        assert any("single constraint" in w for w in result["warnings"])
    else:
        assert result["sampling_rate_hz"] == 0.0
        assert result["warnings"]
