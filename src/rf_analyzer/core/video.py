"""Analog video (NTSC) line-sync hint detector.

The headless pipeline classifies digital modulations (BPSK/QPSK/8PSK/QAM/FSK).
An analog NTSC capture is none of those: it is amplitude-modulated video whose
envelope carries the horizontal line-sync pulses at ``NTSC_LINE_RATE_HZ``
(15734.25 Hz). Without a dedicated test such a capture falls through to an
uncorroborated low-confidence label (measured: 4-FSK at 0.35 on
``real_data/ntsc_10mhz_cf32.iq``) -- an honest answer, but one that hides a
detectable signature.

This module is that test. It looks at the magnitude envelope ``|x|`` (DC
removed) with a single FFT and asks two questions:

1. Is there a spectral line at the NTSC line rate (±1.5%) that stands clearly
   above its local neighbourhood (``FUND_MIN_PROMINENCE``)?
2. Is its second harmonic present too (``HARM_MIN_PROMINENCE``)? Sync pulses
   are narrow (~4.7 us in a 63.5 us line, ~7% duty), so a genuine line-sync
   comb is rich in harmonics -- measured on the real capture the 2nd harmonic
   is as strong as the fundamental (ratio ~1.1), and the same holds for the
   synthetic AM-sync test signal (ratio ~0.9).

Both gates must pass. A pure tone has a constant envelope (no line at all), a
pulse-shaped BPSK burst puts its envelope energy at the symbol rate (125 kHz
in the test setup, nowhere near the search window), and white noise has no
sharp lines -- all three fail the gates with wide margins (see the measured
numbers in the gate comments and ``tests/unit/test_video.py``).

Honesty contracts (do not weaken):

* The result is a *hint*, not a mode claim: ``confidence`` is capped at
  ``MAX_CONFIDENCE`` (0.8) no matter how strong the line is.
* The reported rate is resolution-limited: a single FFT over ``n`` samples at
  ``fs`` has bin width ``fs / n`` (20 Hz for the 500k-sample real capture),
  so the true rate can be up to half a bin away. The bin width is quoted in
  ``reason``; callers must not treat the number as more precise than that.
* The detector never raises on bad input -- it returns ``found=False`` with a
  ``reason``. A hint must not be able to crash an analysis.
* Cost is one real FFT: O(n log n), measured ~0.1 s on the 500k-sample real
  capture, well inside the 5 s budget.
"""

from __future__ import annotations

import numpy as np

#: Nominal NTSC-M horizontal line rate in Hz (4.5 MHz / 286 = 15734.266...,
#: universally quoted as 15734.25).
NTSC_LINE_RATE_HZ = 15734.25

#: Half-width of the fundamental search window, as a fraction of nominal.
#: Covers crystal tolerance plus one FFT bin at every supported resolution.
LINE_SEARCH_FRACTION = 0.015

#: Minimum fundamental peak height relative to the local median. Measured:
#: real NTSC capture ~55, synthetic AM-sync ~203, BPSK burst 3.1, AWGN 2.1,
#: tone 0 (constant envelope, rejected earlier). The gate sits ~2.5x above
#: the strongest negative and ~7x below the weakest positive.
FUND_MIN_PROMINENCE = 8.0

#: Minimum 2nd-harmonic peak height relative to its local median. Measured:
#: real capture ~58, synthetic ~166, BPSK burst ~3, AWGN ~1. Same margins
#: as the fundamental gate.
HARM_MIN_PROMINENCE = 6.0

#: Confidence is never reported above this: a line-sync hint says "this looks
#: like analog video", never "this IS NTSC".
MAX_CONFIDENCE = 0.8

#: Envelopes varying less than this (relative std) count as constant, i.e. an
#: unmodulated carrier. A float tone is exactly 0; an int16-quantised tone is
#: ~7e-6; real NTSC is ~0.03 and the synthetic positive ~0.13.
MIN_ENVELOPE_VARIATION = 1e-4

#: Bins excluded around a peak when estimating its local background, and bins
#: searched around the expected harmonic (absorbs the x2 bin-quantisation of
#: the measured fundamental).
PEAK_EXCLUSION_BINS = 2

#: Half-width, in Hz, of the neighbourhood a peak's background is taken over.
#: The envelope line of a real video signal is not a pure tone: picture
#: content amplitude-modulates the sync pulses, so sideband skirts extend
#: ~100 Hz either side of the line (measured on real_data/ntsc_10mhz_cf32.iq:
#: a ±236 Hz background sits on the skirts and reads only 11x, while ±2 kHz
#: reads 55x). The background must therefore be wider than the skirts but
#: narrower than the 15.7 kHz comb spacing, so no neighbouring comb line can
#: enter it. 2 kHz satisfies both with margin.
MEDIAN_HALF_WIDTH_HZ = 2000.0

#: Minimum bins inside the search window for a meaningful peak-vs-background
#: comparison. Fewer means the FFT resolution is too coarse for this test.
MIN_WINDOW_BINS = 8


def _fail(reason: str) -> dict:
    """A negative result in the documented schema."""
    return {
        "found": False,
        "line_rate_hz": None,
        "confidence": 0.0,
        "harmonic_ratio": 0.0,
        "method": "envelope-psd",
        "reason": reason,
    }


def detect_ntsc_lines(samples: np.ndarray, sample_rate: float) -> dict:
    """Look for the NTSC horizontal line-sync signature in ``samples``.

    Args:
        samples: Complex (or real) baseband samples.
        sample_rate: Sample rate in Hz. Must exceed twice the top of the
            2nd-harmonic search band (~64 kHz); below that the corroborating
            harmonic is not observable and the detector declines.

    Returns:
        ``{"found", "line_rate_hz", "confidence", "harmonic_ratio", "method",
        "reason"}``. ``line_rate_hz`` is the FFT bin centre of the envelope
        line (``None`` when not found); ``confidence`` is in ``[0.0, 0.8]``
        (0.0 when not found); ``harmonic_ratio`` is the linear magnitude
        ratio ``|2nd harmonic| / |fundamental|`` (0.0 when unmeasurable);
        ``method`` is ``"envelope-psd"``; ``reason`` always explains the
        outcome and quotes the FFT bin width when a line is reported.
    """
    try:
        return _detect(samples, sample_rate)
    except Exception as exc:  # A hint must never crash an analysis.
        return _fail(f"detector failed: {exc}")


def _detect(samples: np.ndarray, sample_rate: float) -> dict:
    fs = float(sample_rate)
    if not np.isfinite(fs) or fs <= 0:
        return _fail("sample rate must be a positive finite number")

    x = np.asarray(samples).ravel()
    n = int(x.size)
    if n < 64:
        return _fail(f"too few samples ({n}) for a line-sync test")
    if not np.all(np.isfinite(x)):
        return _fail("input contains non-finite samples")

    window_top = NTSC_LINE_RATE_HZ * (1.0 + LINE_SEARCH_FRACTION)
    if fs / 2.0 < 2.0 * window_top:
        return _fail(
            f"sample rate {fs:.0f} Hz cannot show the 2nd harmonic ({2.0 * window_top:.0f} Hz needed inside Nyquist)"
        )

    # The sync pulses amplitude-modulate the carrier, so they appear in the
    # magnitude envelope. Complex and real inputs both work through abs().
    env = np.abs(x.astype(np.complex128))
    mean_env = float(np.mean(env))
    std_env = float(np.std(env))
    if not np.isfinite(mean_env) or not np.isfinite(std_env) or mean_env <= 0.0:
        return _fail("envelope has no usable dynamic range")
    if std_env / mean_env < MIN_ENVELOPE_VARIATION:
        return _fail("constant envelope: unmodulated carrier, no line-sync structure")
    env = env - mean_env

    # Single FFT over the whole capture: the line repeats, so every period
    # adds coherently and no spectrogram loop is needed.
    spectrum = np.fft.rfft(env * np.hanning(n))
    mag = np.abs(spectrum)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    bin_width = float(freqs[1] - freqs[0]) if freqs.size > 1 else fs

    lo = NTSC_LINE_RATE_HZ * (1.0 - LINE_SEARCH_FRACTION)
    hi = NTSC_LINE_RATE_HZ * (1.0 + LINE_SEARCH_FRACTION)
    in_window = np.flatnonzero((freqs >= lo) & (freqs <= hi))
    if in_window.size < MIN_WINDOW_BINS:
        return _fail(
            f"insufficient frequency resolution (bin width {bin_width:.1f} Hz, "
            f"only {in_window.size} bins in the search window)"
        )

    peak = int(in_window[int(np.argmax(mag[in_window]))])
    fund_mag = float(mag[peak])
    fund_freq = float(freqs[peak])
    half_bins = max(MIN_WINDOW_BINS, round(MEDIAN_HALF_WIDTH_HZ / bin_width))
    span_lo = max(0, peak - half_bins)
    span_hi = min(mag.size, peak + half_bins + 1)
    span = np.arange(span_lo, span_hi)
    keep = np.abs(span - peak) > PEAK_EXCLUSION_BINS
    if int(np.count_nonzero(keep)) < 4 or fund_mag <= 0.0:
        return _fail("no measurable envelope line in the search window")
    fund_med = float(np.median(mag[span[keep]]))
    if not np.isfinite(fund_med) or fund_med <= 0.0:
        return _fail("no measurable envelope background in the search window")
    fund_prom = fund_mag / fund_med
    if fund_prom < FUND_MIN_PROMINENCE:
        return _fail(
            f"no line at the NTSC rate: strongest candidate {fund_freq:.0f} Hz "
            f"at {fund_prom:.1f}x local background "
            f"(needs {FUND_MIN_PROMINENCE:.0f}x)"
        )

    # Corroboration: a real sync comb has a strong 2nd harmonic; a chance
    # noise bump at the line rate does not. The expected bin is twice the
    # *measured* peak (not twice nominal), with a small search for the x2
    # bin-quantisation error.
    expected = 2.0 * fund_freq
    centre = int(np.argmin(np.abs(freqs - expected)))
    span_lo = max(0, centre - PEAK_EXCLUSION_BINS)
    span_hi = min(mag.size - 1, centre + PEAK_EXCLUSION_BINS)
    harm_rel = int(np.argmax(mag[span_lo : span_hi + 1]))
    harm_bin = span_lo + harm_rel
    harm_mag = float(mag[harm_bin])
    half_bins_h = max(MIN_WINDOW_BINS, round(MEDIAN_HALF_WIDTH_HZ / bin_width))
    med_lo = max(0, harm_bin - half_bins_h)
    med_hi = min(mag.size, harm_bin + half_bins_h + 1)
    neighbourhood = np.arange(med_lo, med_hi)
    keep_h = np.abs(neighbourhood - harm_bin) > PEAK_EXCLUSION_BINS
    if int(np.count_nonzero(keep_h)) < 8 or harm_mag <= 0.0:
        return _fail("2nd harmonic of the line candidate is not measurable")
    harm_med = float(np.median(mag[neighbourhood[keep_h]]))
    if not np.isfinite(harm_med) or harm_med <= 0.0:
        return _fail("2nd harmonic background is not measurable")
    harm_prom = harm_mag / harm_med
    harmonic_ratio = harm_mag / fund_mag if fund_mag > 0.0 else 0.0
    if not np.isfinite(harmonic_ratio):
        harmonic_ratio = 0.0
    if harm_prom < HARM_MIN_PROMINENCE:
        return {
            **_fail(
                f"line candidate at {fund_freq:.0f} Hz "
                f"({fund_prom:.0f}x background) has no 2nd harmonic "
                f"({harm_prom:.1f}x, needs {HARM_MIN_PROMINENCE:.0f}x)"
            ),
            "harmonic_ratio": float(harmonic_ratio),
        }

    evidence_db = 10.0 * float(np.log10(min(fund_prom, harm_prom)))
    confidence = min(MAX_CONFIDENCE, max(0.0, evidence_db / 25.0))
    return {
        "found": True,
        "line_rate_hz": fund_freq,
        "confidence": float(confidence),
        "harmonic_ratio": float(harmonic_ratio),
        "method": "envelope-psd",
        "reason": (
            f"envelope line at {fund_freq:.0f} Hz "
            f"({fund_prom:.0f}x local background) with 2nd harmonic at "
            f"{float(freqs[harm_bin]):.0f} Hz ({harm_prom:.0f}x); "
            f"FFT bin width {bin_width:.1f} Hz limits the rate accuracy"
        ),
    }


__all__ = [
    "FUND_MIN_PROMINENCE",
    "HARM_MIN_PROMINENCE",
    "LINE_SEARCH_FRACTION",
    "MAX_CONFIDENCE",
    "MIN_ENVELOPE_VARIATION",
    "NTSC_LINE_RATE_HZ",
    "detect_ntsc_lines",
]
