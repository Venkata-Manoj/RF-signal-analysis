"""GPS L1 C/A acquisition + despread (additive, honest).

A 10 ms capture CANNOT yield navigation subframes or ephemeris (those need
6 s+ of continuous tracking), so this module never claims a message decode.
What it delivers is *acquisition evidence*: which space vehicles (SVs) are
present, at what Doppler and code phase, with prompt-correlator signs and a
C/N0 estimate. That is the standard first stage of any GPS receiver, and it
is the most this tool can honestly say about a short capture.

Method (textbook parallel code-phase search, pure numpy):

* The C/A code repeats every 1 ms (1023 chips at 1.023 Mcps). Each 1 ms block
  of the capture is carrier-wiped at a trial Doppler, then correlated against
  the PRN code for every code phase at once via FFT
  (``ifft(fft(block) * conj(fft(code)))``). Magnitudes are accumulated
  *non-coherently* (power sum) over the available 1 ms blocks, so a nav-bit
  flip inside the window can only cost one block, never the detection.
* Detection metric is peak / second-peak of the accumulated power-delay
  profile, with the second peak searched *outside* ±1 chip around the main
  peak (otherwise the triangle sidelobe next to the peak would always read
  as the runner-up). A fixed threshold gates the decision; pure noise sits
  near ~1.1-1.3 while a real signal reaches several units, so the gate has
  wide margins on both sides.
* Despread re-correlates each 1 ms block at the detected Doppler/code phase
  (the "prompt" correlator) and reports hard signs plus a moment-method C/N0
  estimate (``10*log10(|mean|^2/var) + 30 dB`` for the 1 kHz prompt rate).
  Noise lands near ~20 dB-Hz; healthy GPS lands 35-55 dB-Hz.

Resolution (documented, not hidden): code phase is resolved to one sample
(1023/block_len chips, ~0.26 chips at 4 MHz); Doppler to one search step
(250 Hz default, so ±125 Hz error). Bounded cost: at most ``max_blocks``
1 ms blocks (default 20) are examined however long the capture is, so even
a 100 MB file stays a bounded FFT workload.

Only ``numpy`` is used. No GNU Radio.
"""

from __future__ import annotations

import math

import numpy as np

#: C/A chipping rate (Hz) and code length (chips). The code repeats every 1 ms.
CHIP_RATE_HZ = 1_023_000.0
CODE_LENGTH_CHIPS = 1023
CODE_PERIOD_S = 0.001

#: G2 phase-selector taps per SV (1-indexed shift-register stages whose
#: outputs are XORed for that SV's G2 sequence). IS-GPS-200, Table 3-Ia,
#: PRN 1..32. G1 feedback is stages 3+10, G2 feedback stages 2+3+6+8+9+10,
#: both registers start all-ones; the output chip is G1-stage-10 XOR the
#: selected G2 taps. (Stages below are 1-indexed; subtract 1 for indices.)
G2_TAPS: dict[int, tuple[int, int]] = {
    1: (2, 6),
    2: (3, 7),
    3: (4, 8),
    4: (5, 9),
    5: (1, 9),
    6: (2, 10),
    7: (1, 8),
    8: (2, 9),
    9: (3, 10),
    10: (2, 3),
    11: (3, 4),
    12: (5, 6),
    13: (6, 7),
    14: (7, 8),
    15: (8, 9),
    16: (9, 10),
    17: (1, 4),
    18: (2, 5),
    19: (3, 6),
    20: (4, 7),
    21: (5, 8),
    22: (6, 9),
    23: (1, 3),
    24: (4, 6),
    25: (5, 7),
    26: (6, 8),
    27: (7, 9),
    28: (8, 10),
    29: (1, 6),
    30: (2, 7),
    31: (3, 8),
    32: (4, 9),
}

#: Default detection gate on the peak/second-peak metric. Measured margins:
#: pure complex noise over 4000-sample blocks lands at ~1.1-1.3, while a
#: healthy GPS signal reaches 3+. A marginal crossing still escalates via a
#: warning rather than reading as confident.
ACQUISITION_THRESHOLD = 2.5

#: Upper C/N0 plausibility bound for the warning text (dB-Hz). Healthy GPS is
#: 35-55 dB-Hz; a value far outside the plausible band means the prompt
#: statistics are not signal-like and is reported, not hidden.
CN0_PLAUSIBLE_DBHZ = (25.0, 60.0)

#: Minimum sample rate that can resolve the 1.023 Mcps code at all. Below
#: ~2x the chip rate the code aliases into mush, so acquisition declines
#: with a reason instead of searching.
MIN_SAMPLE_RATE_HZ = 2.0 * CHIP_RATE_HZ


def ca_code(sv: int, chips: int = CODE_LENGTH_CHIPS) -> np.ndarray:
    """C/A PRN code for SV 1..32 as ``0``/``1`` chips (``int8``).

    Generated from the G1/G2 10-stage LFSRs per the ICD definition above
    (see :data:`G2_TAPS`). SV 1's sequence starts ``110010...`` in this
    ``1``-means-high convention -- that prefix is pinned by
    ``tests/unit/test_gps.py`` as a regression check on the LFSR wiring.

    Args:
        sv: Space-vehicle number, 1..32 inclusive.
        chips: Chips to return; the 1023-chip period is tiled/truncated when
            this differs from 1023.

    Raises:
        ValueError: For an SV outside 1..32 or a non-positive chip count.
    """
    sv = int(sv)
    if sv not in G2_TAPS:
        raise ValueError(f"sv must be 1..32, got {sv!r}")
    chips = int(chips)
    if chips <= 0:
        raise ValueError(f"chips must be positive, got {chips!r}")
    tap_a, tap_b = G2_TAPS[sv]
    g1 = [1] * 10
    g2 = [1] * 10
    period = np.empty(CODE_LENGTH_CHIPS, dtype=np.int8)
    for i in range(CODE_LENGTH_CHIPS):
        g1_out = g1[9]
        g2_out = g2[tap_a - 1] ^ g2[tap_b - 1]
        period[i] = g1_out ^ g2_out
        g1_fb = g1[2] ^ g1[9]
        g2_fb = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        g1 = [g1_fb] + g1[:9]
        g2 = [g2_fb] + g2[:9]
    if chips == CODE_LENGTH_CHIPS:
        return period
    tiled = np.tile(period, (chips // CODE_LENGTH_CHIPS) + 1)
    return tiled[:chips].astype(np.int8)


def _code_pm1_upsampled(sv: int, sample_rate: float, n_samples: int) -> np.ndarray:
    """PRN code resampled to ``n_samples`` at ``sample_rate`` as ±1 floats.

    Sample ``i`` carries chip ``floor(i * chip_rate / fs) mod 1023`` with
    ``1 -> +1`` and ``0 -> -1``. Nearest-chip mapping (no filtering): the
    acquisition grid is one sample, so interpolation would only smear the
    peak it is trying to find.
    """
    chips = ca_code(sv).astype(np.float64)
    idx = ((np.arange(int(n_samples)) * CHIP_RATE_HZ / float(sample_rate)).astype(np.int64)) % CODE_LENGTH_CHIPS
    return np.where(chips[idx] > 0, 1.0, -1.0)


def _doppler_bins(doppler_range: float, doppler_step: float) -> np.ndarray:
    span = float(abs(doppler_range))
    step = float(abs(doppler_step))
    if step <= 0:
        raise ValueError(f"doppler_step must be positive, got {doppler_step!r}")
    n = int(round(span / step))
    return np.array([float(k) * step for k in range(-n, n + 1)], dtype=np.float64)


def acquire(
    samples: np.ndarray,
    sample_rate: float,
    svs: range | list[int] | tuple[int, ...] | None = None,
    doppler_range: float = 5000.0,
    doppler_step: float = 250.0,
    threshold: float = ACQUISITION_THRESHOLD,
    max_blocks: int = 20,
) -> dict[int, dict]:
    """Parallel code-phase acquisition over SVs x Doppler bins.

    For each SV and each trial Doppler the 1 ms blocks are carrier-wiped,
    FFT-correlated against the PRN code for all code phases at once, and
    accumulated non-coherently (power sum). The Doppler bin with the
    strongest peak wins; the SV counts as acquired only when its
    peak/second-peak metric clears ``threshold``.

    Cost is bounded: at most ``max_blocks`` 1 ms blocks are examined no
    matter how long the capture is (a 10 ms capture uses 10; a 100 MB file
    still uses 20). 10 ms at 4 MHz x 32 SVs x 41 Doppler bins runs in a few
    seconds with vectorized FFTs.

    Args:
        samples: Complex baseband samples (real input is accepted and read
            as I-only; the metric is scale-invariant, so no AGC is needed).
        sample_rate: Sample rate in Hz (must clear :data:`MIN_SAMPLE_RATE_HZ`
            or every SV declines with a reason instead of searching).
        svs: SVs to search (default 1..32).
        doppler_range: ± search span in Hz around 0 (baseband).
        doppler_step: Doppler bin spacing in Hz (resolution ±step/2).
        threshold: Peak/second-peak gate for ``acquired``.
        max_blocks: Cap on the 1 ms blocks examined (bounded runtime).

    Returns:
        Dict keyed by SV with ``{acquired, doppler_hz, code_phase_samples,
        code_phase_chips, metric, peak_over_mean, n_blocks, block_len,
        reason}``. ``reason`` is ``""`` when the search ran; a decline
        (undersampled input, no full 1 ms block) leaves ``acquired`` False
        for every SV with the reason named. Never raises on data -- only on
        invalid search parameters.
    """
    fs = float(sample_rate)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate!r}")
    step = float(doppler_step)
    if not np.isfinite(step) or step <= 0:
        raise ValueError(f"doppler_step must be positive, got {doppler_step!r}")
    if svs is None:
        wanted = list(range(1, 33))
    else:
        wanted = [int(s) for s in svs]
        for s in wanted:
            if s not in G2_TAPS:
                raise ValueError(f"sv must be 1..32, got {s!r}")
    max_blocks = int(max_blocks)
    if max_blocks <= 0:
        raise ValueError(f"max_blocks must be positive, got {max_blocks!r}")

    def _decline(reason: str) -> dict[int, dict]:
        return {
            sv: {
                "acquired": False,
                "doppler_hz": 0.0,
                "code_phase_samples": 0,
                "code_phase_chips": 0.0,
                "metric": 0.0,
                "peak_over_mean": 0.0,
                "n_blocks": 0,
                "block_len": 0,
                "reason": reason,
            }
            for sv in wanted
        }

    x = np.asarray(samples).ravel()
    if x.size == 0 or not np.all(np.isfinite(x.real)) or not np.all(np.isfinite(x.imag if np.iscomplexobj(x) else 0.0)):
        return _decline("no usable samples (empty or non-finite input)")
    xc = x.astype(np.complex128) if np.iscomplexobj(x) else x.astype(np.float64).astype(np.complex128)
    rms = float(np.sqrt(np.mean(np.abs(xc) ** 2)))
    if not np.isfinite(rms) or rms <= 0.0:
        return _decline("no usable samples (all-zero input)")
    xc = xc / rms  # scale-invariant metric; keeps float32-scale captures numerically sane

    if fs < MIN_SAMPLE_RATE_HZ:
        return _decline(
            f"sample rate {fs:.0f} Hz cannot resolve the 1.023 Mcps C/A code "
            f"(needs >={MIN_SAMPLE_RATE_HZ:.0f} Hz); acquisition declined"
        )
    block_len = int(round(fs * CODE_PERIOD_S))
    if block_len < 8:
        return _decline(f"1 ms block is only {block_len} samples; acquisition declined")
    n_blocks = int(xc.size // block_len)
    if n_blocks < 1:
        return _decline("capture shorter than one 1 ms code period; acquisition declined")
    n_blocks = min(n_blocks, max_blocks)
    trunc = xc[: n_blocks * block_len]
    # Global time base so the wipeoff phase is continuous across blocks
    # (continuity only matters within a block for the non-coherent sum, but
    # there is no reason to introduce a per-block phase jump).
    t = np.arange(n_blocks * block_len, dtype=np.float64) / fs

    dopplers = _doppler_bins(doppler_range, step)
    # Wiped-block spectra shared by all SVs: depends on Doppler only.
    wiped_fft: list[np.ndarray] = []
    for fd in dopplers:
        carrier = np.exp(-1j * 2.0 * np.pi * float(fd) * t)
        blocks = (trunc * carrier).reshape(n_blocks, block_len)
        wiped_fft.append(np.fft.fft(blocks, axis=1))

    # Second-peak exclusion: ±1 chip around the main peak, so the triangle
    # sidelobe next to the peak never reads as the runner-up.
    excl = max(1, int(math.ceil(block_len / CODE_LENGTH_CHIPS)))
    chips_per_sample = CODE_LENGTH_CHIPS / float(block_len)

    results: dict[int, dict] = {}
    for sv in wanted:
        code = _code_pm1_upsampled(sv, fs, block_len)
        code_conj = np.conj(np.fft.fft(code))
        best: dict | None = None
        for di, fd in enumerate(dopplers):
            acc = np.zeros(block_len, dtype=np.float64)
            spec = wiped_fft[di]
            for b in range(n_blocks):
                corr = np.fft.ifft(spec[b] * code_conj)
                acc += np.abs(corr) ** 2
            acc /= float(n_blocks)
            peak_idx = int(np.argmax(acc))
            peak = float(acc[peak_idx])
            mean = float(np.mean(acc))
            # Runner-up outside the main-lobe exclusion zone.
            lo = max(0, peak_idx - excl)
            hi = min(block_len, peak_idx + excl + 1)
            masked = acc.copy()
            masked[lo:hi] = -np.inf
            second = float(np.max(masked))
            if not np.isfinite(second) or second <= 0.0:
                # Degenerate (the exclusion covered the block, or an all-zero
                # profile): keep the metric finite so every JSON boundary stays
                # strict-JSON safe without a lossy ``inf -> None`` conversion.
                metric = 1e12 if peak > 0 else 0.0
            else:
                metric = min(peak / second, 1e12)
            cand = {
                "doppler_hz": float(fd),
                "code_phase_samples": peak_idx,
                "code_phase_chips": float(peak_idx * chips_per_sample),
                "metric": float(metric),
                "peak_over_mean": float(peak / mean) if mean > 0 else 0.0,
            }
            if best is None or peak > best[0]:
                best = (peak, cand)
        assert best is not None
        _, winner = best
        metric = float(winner["metric"])
        results[sv] = {
            "acquired": bool(metric >= float(threshold)),
            "doppler_hz": float(winner["doppler_hz"]),
            "code_phase_samples": int(winner["code_phase_samples"]),
            "code_phase_chips": float(winner["code_phase_chips"]),
            "metric": metric,
            "peak_over_mean": float(winner["peak_over_mean"]),
            "n_blocks": int(n_blocks),
            "block_len": int(block_len),
            "reason": "",
        }
    return results


def estimate_cn0_dbhz(prompt: np.ndarray) -> float | None:
    """Moment-method C/N0 from per-1 ms prompt values (dB-Hz) or None.

    ``C/N0 = 10*log10(|mean|^2 / var) + 30``: the prompt rate is 1 kHz, so
    the in-band SNR plus 30 dB is the carrier-to-noise-density ratio. A
    nav-bit flip inside the window inflates the variance, so this reads
    *low* (conservative) when one occurred -- never high. Pure noise lands
    near ``30 - 10*log10(n_blocks)`` (~20 dB-Hz for 10 blocks); healthy GPS
    lands 35-55 dB-Hz.
    """
    p = np.asarray(prompt).ravel()
    if p.size < 2:
        return None
    mean = complex(np.mean(p))
    var = float(np.mean(np.abs(p - mean) ** 2))
    sig = abs(mean) ** 2
    if not np.isfinite(sig) or not np.isfinite(var) or var <= 0.0 or sig <= 0.0:
        return None
    return float(10.0 * np.log10(sig / var) + 30.0)


def despread(
    samples: np.ndarray,
    sample_rate: float,
    sv: int,
    doppler_hz: float,
    code_phase_samples: int,
    max_blocks: int = 20,
) -> dict:
    """Prompt-correlate each 1 ms block at the detected Doppler/code phase.

    This is the "prompt arm" of a tracking loop without the loop: carrier
    wipeoff at ``doppler_hz``, then one complex correlation per 1 ms block
    against the code aligned to ``code_phase_samples``. Returns the prompt
    series, hard signs (``1`` when the prompt real part is >= 0 else ``0``,
    matching the pipeline's 0/1 bit convention), and the C/N0 estimate.

    The signs are *not* navigation data: a 10 ms window holds at most half
    a 20 ms nav bit, so no subframe, no ephemeris, no message is claimed --
    callers must say so. A residual Doppler error (up to half the
    acquisition step) rotates the prompt by up to 45° over 1 ms, which
    shrinks the real part but does not flip its sign by itself.
    """
    sv = int(sv)
    if sv not in G2_TAPS:
        raise ValueError(f"sv must be 1..32, got {sv!r}")
    fs = float(sample_rate)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate!r}")
    max_blocks = int(max_blocks)
    if max_blocks <= 0:
        raise ValueError(f"max_blocks must be positive, got {max_blocks!r}")
    x = np.asarray(samples).ravel()
    xc = x.astype(np.complex128) if np.iscomplexobj(x) else x.astype(np.float64).astype(np.complex128)
    block_len = int(round(fs * CODE_PERIOD_S))
    if block_len <= 0 or xc.size < block_len:
        return {
            "sv": sv,
            "doppler_hz": float(doppler_hz),
            "code_phase_samples": int(code_phase_samples),
            "n_blocks": 0,
            "prompt_real": [],
            "prompt_imag": [],
            "prompt_signs": [],
            "cn0_dbhz": None,
            "reason": "capture shorter than one 1 ms code period",
        }
    n_blocks = min(int(xc.size // block_len), max_blocks)
    phase = int(code_phase_samples) % block_len
    code = _code_pm1_upsampled(sv, fs, block_len)
    # Align the replica to the detected phase: the acquisition peak at index
    # k means the signal carries the code delayed by k, so correlating
    # against the replica rolled forward by k collapses it to the prompt.
    replica = np.roll(code, phase)
    t = np.arange(n_blocks * block_len, dtype=np.float64) / fs
    carrier = np.exp(-1j * 2.0 * np.pi * float(doppler_hz) * t)
    wiped = (xc[: n_blocks * block_len] * carrier).reshape(n_blocks, block_len)
    prompt = wiped @ replica  # one complex value per 1 ms block
    signs = [1 if float(v.real) >= 0.0 else 0 for v in prompt]
    return {
        "sv": sv,
        "doppler_hz": float(doppler_hz),
        "code_phase_samples": int(phase),
        "n_blocks": int(n_blocks),
        "prompt_real": [float(v.real) for v in prompt],
        "prompt_imag": [float(v.imag) for v in prompt],
        "prompt_signs": [int(s) for s in signs],
        "cn0_dbhz": estimate_cn0_dbhz(prompt),
        "reason": "",
    }


#: Alias for the spelling used in the task brief (``desspread`` vs ``despread``).
desspread = despread
