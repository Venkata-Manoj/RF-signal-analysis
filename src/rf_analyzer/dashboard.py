"""Zero-install web dashboard payloads.

The dashboard is a single self-contained HTML page served by
``scripts/serve_dashboard.py`` over Python's standard-library
``http.server``. Nothing beyond the project's existing dependencies is
required: no Flask, no npm, no CDN.

This module only *serialises* what the pipeline already computed plus a
handful of decimated plot arrays. It deliberately does not re-implement
any DSP decision logic -- ``pipeline.analyze_file`` remains the single
source of truth for the analysis, and the helpers here call the same
``core.dsp`` primitives the GUI and reports use.

Full spec: info.md §12.2 (PSD/waterfall), §13 (report schema).
"""

from __future__ import annotations

import base64
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from rf_analyzer.core.dsp import (
    compute_eye,
    compute_psd,
    compute_waterfall,
    ideal_constellation,
)
from rf_analyzer.core.io import load_iq, load_wav

# Plot budgets. These keep the JSON payload comfortably small (a few
# hundred KB) even for 100 MB captures, so the page stays responsive.
MAX_PLOT_SAMPLES = 2_000_000
CONSTELLATION_POINTS = 2_000
PSD_POINTS = 512
WATERFALL_SHAPE = (128, 256)
EYE_TRACES = 200
TIME_SERIES_POINTS = 1_000


def _decimate(values: np.ndarray, max_points: int) -> np.ndarray:
    """Evenly stride ``values`` down to at most ``max_points`` samples."""
    values = np.asarray(values)
    if max_points <= 0 or values.size <= max_points:
        return values
    step = int(np.ceil(values.size / max_points))
    return values[::step][:max_points]


def _round_list(values: np.ndarray, digits: int = 4) -> list[float]:
    """Convert to a plain JSON-friendly list of rounded floats."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return [round(float(v), digits) for v in arr]


def constellation_payload(
    samples: np.ndarray, mode: str, max_points: int = CONSTELLATION_POINTS
) -> dict[str, Any]:
    """Scatter data for the constellation plot, with the ideal points.

    Only the first ``max_points`` samples are sent; a real capture can hold
    millions of symbols and the browser canvas cannot resolve them anyway.
    """
    pts = _decimate(np.asarray(samples), max_points)
    payload: dict[str, Any] = {
        "i": _round_list(np.real(pts), 5),
        "q": _round_list(np.imag(pts), 5),
        "shown": int(pts.size),
    }
    ideal = ideal_constellation(mode)
    if ideal is not None:
        payload["ideal_i"] = _round_list(np.real(ideal), 5)
        payload["ideal_q"] = _round_list(np.imag(ideal), 5)
    return payload


def psd_payload(
    samples: np.ndarray, sample_rate: float, points: int = PSD_POINTS
) -> dict[str, Any]:
    """Frequency-domain line for the spectrum plot."""
    nfft = int(min(4096, max(64, np.asarray(samples).size)))
    freqs, psd_db = compute_psd(np.asarray(samples), sample_rate, nfft=nfft)
    idx = _decimate(np.arange(freqs.size), points)
    f_sel = freqs[idx]
    p_sel = psd_db[idx]
    peak = float(f_sel[int(np.argmax(p_sel))]) if p_sel.size else 0.0
    return {
        "freq_hz": _round_list(f_sel, 2),
        "power_db": _round_list(p_sel, 2),
        "peak_hz": round(peak, 2),
    }


def _bin_edges(n: int, bins: int) -> list[tuple[int, int]]:
    """Half-open index ranges that split ``n`` items into ``bins`` groups."""
    cuts = np.linspace(0, n, bins + 1).astype(int)
    return list(pairwise(cuts))


def waterfall_payload(
    samples: np.ndarray,
    sample_rate: float,
    shape: tuple[int, int] = WATERFALL_SHAPE,
) -> dict[str, Any]:
    """Waterfall as base64 uint8 -- ~43 KB instead of ~300 KB of JSON floats.

    The dB range is carried alongside so the client can map the byte back
    to a physical value for the colour legend.
    """
    samples = np.asarray(samples)
    rows, cols = int(shape[0]), int(shape[1])
    if samples.size < 32:
        return {"available": False, "reason": "not enough samples for a waterfall"}

    nfft = int(min(max(64, cols), samples.size))
    freqs, times, wf = compute_waterfall(samples, sample_rate, nfft=nfft, overlap=0.5)
    wf = np.asarray(wf, dtype=np.float64)
    if wf.ndim != 2 or wf.size == 0:
        return {"available": False, "reason": "waterfall computation returned no data"}

    # Collapse the time axis to `rows` bins by averaging.
    if wf.shape[0] > rows:
        groups = _bin_edges(wf.shape[0], rows)
        wf = np.vstack([wf[a:b].mean(axis=0) for a, b in groups])
        t_sel = np.array([float(times[a:b].mean()) for a, b in groups])
    else:
        t_sel = np.asarray(times, dtype=np.float64)

    # Collapse the frequency axis to `cols` bins.
    if wf.shape[1] > cols:
        groups = _bin_edges(wf.shape[1], cols)
        wf = np.hstack([wf[:, a:b].mean(axis=1, keepdims=True) for a, b in groups])
        f_sel = np.array([float(freqs[a:b].mean()) for a, b in groups])
    else:
        f_sel = np.asarray(freqs, dtype=np.float64)

    lo, hi = float(np.min(wf)), float(np.max(wf))
    span = hi - lo
    scale = 255.0 / span if span > 0 else 0.0
    scaled = ((wf - lo) * scale).astype(np.uint8)
    return {
        "available": True,
        "rows": int(scaled.shape[0]),
        "cols": int(scaled.shape[1]),
        "min_db": round(lo, 2),
        "max_db": round(hi, 2),
        "freq_hz": _round_list(f_sel, 2),
        "time_s": _round_list(t_sel, 6),
        "data_b64": base64.b64encode(scaled.tobytes()).decode("ascii"),
    }


def eye_payload(samples: np.ndarray, samples_per_symbol: int) -> dict[str, Any]:
    """Eye-diagram traces (each trace spans two symbols)."""
    sps = int(max(1, samples_per_symbol))
    try:
        eye = compute_eye(np.asarray(samples), samples_per_symbol=sps)
    except (ValueError, TypeError):
        return {"available": False, "reason": "not enough samples for an eye diagram"}
    if eye.ndim != 2 or eye.shape[0] == 0:
        return {"available": False, "reason": "not enough samples for an eye diagram"}
    if eye.shape[1] != sps * 2:
        # compute_eye falls back to one truncated trace when the capture is
        # shorter than two symbols. That is not an eye diagram, so report it
        # as unavailable rather than drawing a meaningless single line.
        return {
            "available": False,
            "reason": f"need at least {sps * 2} samples for an eye diagram",
        }
    traces = eye[:EYE_TRACES]
    return {
        "available": True,
        "samples_per_symbol": sps,
        "span": int(eye.shape[1]),
        "traces": [_round_list(np.real(t), 4) for t in traces],
    }


def time_payload(
    samples: np.ndarray, sample_rate: float, points: int = TIME_SERIES_POINTS
) -> dict[str, Any]:
    """Decimated I/Q time series plus the envelope."""
    samples = np.asarray(samples)
    if samples.size == 0 or sample_rate <= 0:
        return {"available": False, "reason": "no samples"}
    idx = _decimate(np.arange(samples.size), points)
    sel = samples[idx]
    t_ms = idx.astype(np.float64) / float(sample_rate) * 1000.0
    return {
        "available": True,
        "t_ms": _round_list(t_ms, 4),
        "i": _round_list(np.real(sel), 4),
        "q": _round_list(np.imag(sel), 4),
        "magnitude": _round_list(np.abs(sel), 4),
    }


def load_capture(
    file_path: str, *, sample_rate: float | None = None, iq_format: str = "complex64"
) -> tuple[np.ndarray, float]:
    """Re-read a capture for plotting.

    ``.wav`` carries its own sample rate; ``.iq`` is raw and therefore
    needs the caller-supplied rate (info.md §12.1).
    """
    suffix = Path(file_path).suffix.lower()
    if suffix == ".wav":
        return load_wav(file_path)
    if sample_rate is None or sample_rate <= 0:
        raise ValueError("A sample rate is required for raw .iq files.")
    samples = load_iq(file_path, dtype=iq_format)
    return samples, float(sample_rate)


def build_plots(
    samples: np.ndarray,
    sample_rate: float,
    *,
    mode: str,
    samples_per_symbol: int,
    max_samples: int = MAX_PLOT_SAMPLES,
) -> dict[str, Any]:
    """Assemble every plot payload for one capture."""
    samples = np.asarray(samples)
    if samples.size > max_samples:
        samples = samples[:max_samples]
    return {
        "constellation": constellation_payload(samples, mode),
        "psd": psd_payload(samples, sample_rate),
        "waterfall": waterfall_payload(samples, sample_rate),
        "eye": eye_payload(samples, samples_per_symbol),
        "time": time_payload(samples, sample_rate),
    }


def _fmt(value: Any, digits: int = 3, unit: str = "") -> str:
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        if value == 0:
            return f"0{unit}"
        if abs(value) >= 1e6 or abs(value) < 1e-3:
            return f"{value:.{digits}e}{unit}"
        return f"{value:.{digits}f}{unit}"
    return str(value)


def summary_rows(report: dict[str, Any]) -> list[dict[str, str]]:
    """Flat label/value rows for the dashboard's parameter table."""
    sig = report.get("signal", {}) or {}
    mod = report.get("modulation", {}) or {}
    dem = report.get("demodulation", {}) or {}
    cor = report.get("correlation", {}) or {}
    fec = report.get("fec", {}) or {}
    ilv = report.get("interleaving", {}) or {}
    inp = report.get("input", {}) or {}
    pay = report.get("payload", {}) or {}
    dec = pay.get("decoded", {}) or {}
    qual = report.get("quality", {}) or {}

    evm = qual.get("evm_percent")
    if not qual.get("applicable"):
        evm_text = "n/a (no constellation)"
    elif evm is not None and float(evm) < 0.01:
        # Noiseless synthetic captures land here; "<0.01 %" reads better than
        # a bare 0.00 % next to a several-hundred-dB MER.
        evm_text = "<0.01 %"
    else:
        evm_text = _fmt(evm, 2, " %")

    mer = qual.get("mer_db")
    if not qual.get("applicable"):
        mer_text = "--"
    elif mer is None:
        # No measurable error vector: MER is unbounded, not missing.
        mer_text = "unbounded"
    elif float(mer) > 100.0:
        mer_text = ">100 dB"
    else:
        mer_text = _fmt(mer, 2, " dB")

    decoded_ok = bool(dec.get("available"))
    return [
        {"label": "File", "value": str(inp.get("file_name", "--"))},
        {"label": "Sample rate", "value": _fmt(inp.get("sample_rate"), 0, " Hz")},
        {
            "label": "IQ format used",
            "value": str(inp.get("iq_format_used") or inp.get("iq_format") or "--"),
        },
        {"label": "Samples", "value": _fmt(sig.get("num_samples"))},
        {"label": "Duration", "value": _fmt(sig.get("duration_seconds"), 6, " s")},
        {"label": "Modulation", "value": str(mod.get("estimated_type", "--"))},
        {"label": "Modulation confidence", "value": _fmt(mod.get("confidence"))},
        {"label": "Demod mode", "value": str(dem.get("mode", "--"))},
        {"label": "Demodulated bits", "value": _fmt(dem.get("num_bits"))},
        {
            "label": "Center freq (est.)",
            "value": _fmt(sig.get("center_frequency_estimate"), 0, " Hz"),
        },
        {
            "label": "Bandwidth (est.)",
            "value": _fmt(sig.get("bandwidth_estimate"), 0, " Hz"),
        },
        {"label": "SNR", "value": _fmt(sig.get("snr_db"), 2, " dB")},
        {
            "label": "Symbol rate (est.)",
            "value": _fmt(sig.get("symbol_rate_estimate"), 0, " Hz"),
        },
        {
            "label": "Sample rate (est.)",
            "value": _fmt(sig.get("sample_rate_estimate"), 0, " Hz"),
        },
        {"label": "CFO", "value": _fmt(sig.get("cfo_estimate_hz"), 2, " Hz")},
        {"label": "EVM / MER", "value": evm_text},
        {"label": "MER", "value": mer_text},
        {
            "label": "Modulation order",
            "value": _fmt(qual.get("modulation_order"), 0, " bits/sym"),
        },
        {
            "label": "Sync word",
            "value": (
                (
                    str(cor.get("sync_word", "--"))
                    + (" (assumed)" if cor.get("assumed") else "")
                )
                if cor.get("sync_word")
                else "--"
            ),
        },
        {"label": "Header offset", "value": _fmt(cor.get("header_offset"))},
        {"label": "Correlation score", "value": _fmt(cor.get("score"), 4)},
        {"label": "Sync detected", "value": _fmt(bool(cor.get("detected")))},
        {"label": "FEC (blind guess)", "value": str(fec.get("candidate", "--"))},
        {"label": "FEC confidence", "value": _fmt(fec.get("confidence"))},
        {"label": "FEC verified", "value": _fmt(bool(fec.get("validated")))},
        {
            "label": "Interleaver (blind guess)",
            "value": str(ilv.get("candidate", "--")),
        },
        {"label": "Interleaver verified", "value": _fmt(bool(ilv.get("validated")))},
        {"label": "Payload bits", "value": _fmt(pay.get("payload_bits"))},
        {
            "label": "Payload bytes (raw)",
            "value": _fmt((pay.get("raw") or {}).get("bytes")),
        },
        {"label": "Decoded", "value": "yes" if decoded_ok else "no"},
        {"label": "Decoded scheme", "value": str(dec.get("scheme") or "--")},
        {"label": "Decoded bytes", "value": _fmt(dec.get("bytes"))},
        {"label": "CRC-16 passed", "value": _fmt(dec.get("crc_pass"))},
        {"label": "Decoded confidence", "value": _fmt(dec.get("confidence"))},
    ]


def dashboard_payload(
    report: dict[str, Any],
    samples: np.ndarray | None = None,
    sample_rate: float | None = None,
) -> dict[str, Any]:
    """Bundle the report, the plot data and the summary rows for the browser."""
    payload: dict[str, Any] = {
        "report": report,
        "summary": summary_rows(report),
        "plots": None,
        "plot_error": None,
    }
    if samples is None:
        return payload

    mode = str((report.get("display") or {}).get("mode") or "BPSK")
    sps = int((report.get("display") or {}).get("samples_per_symbol") or 1)
    rate = float(sample_rate or (report.get("input") or {}).get("sample_rate") or 0.0)
    if rate <= 0:
        payload["plot_error"] = "no usable sample rate, plots were skipped"
        return payload
    try:
        payload["plots"] = build_plots(samples, rate, mode=mode, samples_per_symbol=sps)
    except Exception as exc:  # plotting must never break the analysis view
        payload["plot_error"] = f"{type(exc).__name__}: {exc}"
    return payload
