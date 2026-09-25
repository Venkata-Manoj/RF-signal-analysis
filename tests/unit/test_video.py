"""Unit tests for the NTSC line-sync hint detector (core/video.py).

The positive case is a synthetic AM signal with 15734.25 Hz sync dips plus
noise; the negatives are a pure tone, a pulse-shaped BPSK burst, and AWGN.
The real-capture case pins the measured truth for
``real_data/ntsc_10mhz_cf32.iq`` (50 ms analog NTSC at 10 MHz complex64):
envelope line at bin 15740 Hz (bin width 20 Hz, true rate 15734.25 Hz) with
confidence ~0.70. The ±250 Hz tolerance is the honest resolution budget, not
a fit: the single-FFT bin centre can only resolve the rate to half a bin,
and the line itself carries video-modulation skirts ~100 Hz wide.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from rf_analyzer.core.video import NTSC_LINE_RATE_HZ, detect_ntsc_lines

SAMPLE_RATE = 1_000_000.0
N_SAMPLES = 50_000  # 50 ms: short, but 20 Hz bins resolve the line well.

REAL_CAPTURE = Path(__file__).resolve().parents[2] / "real_data" / "ntsc_10mhz_cf32.iq"
REAL_SAMPLE_RATE = 10_000_000.0
# Honest resolution budget: 20 Hz bins plus ~100 Hz video sideband skirts.
REAL_RATE_TOLERANCE_HZ = 250.0


def make_am_sync(
    n: int = N_SAMPLES,
    sample_rate: float = SAMPLE_RATE,
    noise_std: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    """Carrier with narrow dips at the NTSC line rate, plus complex AWGN."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sample_rate
    sync = ((t * NTSC_LINE_RATE_HZ) % 1.0) < 0.075  # ~4.7 us of 63.5 us
    amp = 1.0 - 0.45 * sync.astype(float)
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2.0)
    return (amp + noise_std * noise).astype(np.complex64)


def make_tone(n: int = N_SAMPLES, sample_rate: float = SAMPLE_RATE) -> np.ndarray:
    t = np.arange(n) / sample_rate
    return np.exp(1j * 2.0 * np.pi * 100_000.0 * t).astype(np.complex64)


def test_synthetic_sync_found():
    # Arrange
    samples = make_am_sync()

    # Act
    result = detect_ntsc_lines(samples, SAMPLE_RATE)

    # Assert
    assert result["found"] is True
    assert result["line_rate_hz"] is not None
    assert abs(result["line_rate_hz"] - NTSC_LINE_RATE_HZ) <= 0.015 * NTSC_LINE_RATE_HZ
    assert result["confidence"] > 0.5
    assert result["confidence"] <= 0.8  # a hint, never a mode claim
    assert result["harmonic_ratio"] > 0.0
    assert result["method"] == "envelope-psd"
    assert result["reason"]


def test_pure_tone_not_found():
    # Arrange: constant envelope, so no line-sync structure can exist.
    samples = make_tone()

    # Act
    result = detect_ntsc_lines(samples, SAMPLE_RATE)

    # Assert
    assert result["found"] is False
    assert result["line_rate_hz"] is None
    assert result["confidence"] == 0.0
    assert result["reason"]


def test_bpsk_burst_not_found():
    # Arrange: pulse-shaped BPSK puts its envelope energy at the symbol rate
    # (125 kHz here), far from the line-sync search window.
    from rf_analyzer.core.waveform import synth

    samples, _ = synth("BPSK", 4000, sps=8, seed=1)

    # Act
    result = detect_ntsc_lines(samples, SAMPLE_RATE)

    # Assert
    assert result["found"] is False
    assert result["line_rate_hz"] is None
    assert result["reason"]


def test_awgn_not_found():
    # Arrange
    rng = np.random.default_rng(0)
    n = N_SAMPLES
    samples = (
        (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2.0)
    ).astype(np.complex64)

    # Act
    result = detect_ntsc_lines(samples, SAMPLE_RATE)

    # Assert
    assert result["found"] is False
    assert result["line_rate_hz"] is None
    assert result["reason"]


@pytest.mark.skipif(
    not REAL_CAPTURE.exists(), reason="real_data not fetched (see fetch_real_data.py)"
)
def test_real_ntsc_capture_found():
    # Arrange: measured truth -- envelope line at bin 15740 Hz (bin width
    # 20 Hz; true NTSC rate 15734.25 Hz), confidence ~0.70.
    samples = np.fromfile(str(REAL_CAPTURE), dtype=np.complex64)
    assert samples.size == 500_000

    # Act
    started = time.perf_counter()
    result = detect_ntsc_lines(samples, REAL_SAMPLE_RATE)
    elapsed = time.perf_counter() - started

    # Assert
    assert result["found"] is True
    assert result["line_rate_hz"] is not None
    assert abs(result["line_rate_hz"] - NTSC_LINE_RATE_HZ) <= REAL_RATE_TOLERANCE_HZ
    assert result["confidence"] > 0.5
    assert result["confidence"] <= 0.8
    assert elapsed < 5.0


def test_result_schema():
    # Both paths expose the same documented shape.
    for samples in (make_am_sync(), make_tone()):
        result = detect_ntsc_lines(samples, SAMPLE_RATE)
        assert set(result) == {
            "found",
            "line_rate_hz",
            "confidence",
            "harmonic_ratio",
            "method",
            "reason",
        }
        assert isinstance(result["found"], bool)
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 0.8
        assert isinstance(result["harmonic_ratio"], float)
        assert isinstance(result["reason"], str) and result["reason"]


def test_guard_rails_decline_with_reason():
    # Too short for any usable frequency resolution.
    short = make_am_sync(n=100)
    result = detect_ntsc_lines(short, SAMPLE_RATE)
    assert result["found"] is False
    assert result["reason"]

    # Sample rate too low to show the corroborating 2nd harmonic.
    result = detect_ntsc_lines(make_am_sync(), 8000.0)
    assert result["found"] is False
    assert result["reason"]
