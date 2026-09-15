"""End-to-end analysis entrypoint. See info.md §12.5 / §13 / §20.

Headless contract (see AGENTS.md): the GUI must call :func:`analyze_file`
and never duplicate DSP logic. The returned dict always follows the report
schema in info.md §13 (keys ``meta``/``input``/``signal``/``modulation``/
``demodulation``/``correlation``/``fec``/``interleaving``/``warnings``/
``errors``).

MVP notes (intentionally naive, per info.md §12):

* DSP: PSD-peak center frequency, median-floor bandwidth/SNR.
* Demod: phase-aligned BPSK/QPSK/16-QAM (``real > 0``/``imag > 0``),
  2-FSK via ``diff(unwrap(angle))``. No Costas loop / timing recovery (V2, §32).
* ``AUTO`` modulation is a variance heuristic only, not a real classifier.
* FEC/interleaving are candidate-score stubs — never claim blind detection.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rf_analyzer.config import CORRELATION_THRESHOLD, TOOL_NAME, VERSION
from rf_analyzer.core.classifier import classify_modulation as classify_modulation_hoc
from rf_analyzer.core.correlator import find_header, hex_to_bits
from rf_analyzer.core.deinterleave import score_deinterleave_candidates
from rf_analyzer.core.demod import demod_2fsk, demod_bpsk, demod_qam16, demod_qpsk
from rf_analyzer.core.dsp import (
    compute_psd,
    estimate_bandwidth,
    estimate_center_frequency,
    estimate_cfo,
    estimate_sampling_rate,
    estimate_snr,
    estimate_symbol_rate,
)
from rf_analyzer.core.fec import score_fec_candidates
from rf_analyzer.core.io import load_iq, load_wav

REQUIRED_REPORT_KEYS = (
    "meta",
    "input",
    "signal",
    "modulation",
    "demodulation",
    "correlation",
    "fec",
    "interleaving",
    "warnings",
    "errors",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _error_report(file_name: str, warnings: list, message: str) -> dict:
    """Graceful error report matching the info.md §20 failure branch."""
    return {
        "meta": {
            "tool_name": TOOL_NAME,
            "version": VERSION,
            "generated_at": _utc_now(),
        },
        "input": {
            "file_name": file_name,
            "file_type": None,
            "sample_rate": None,
            "center_frequency": None,
            "iq_format": None,
        },
        "signal": {},
        "modulation": {},
        "demodulation": {},
        "correlation": {},
        "fec": {},
        "interleaving": {},
        "display": {},
        "warnings": list(warnings),
        "errors": [message],
    }


def _classify_modulation(samples: np.ndarray) -> tuple[str, float, list[str]]:
    """Modulation classification using Higher-Order Cumulants with heuristic fallback."""
    try:
        mod_type, conf, alts = classify_modulation_hoc(samples)
        if mod_type != "UNKNOWN":
            return mod_type, conf, alts
    except Exception:
        pass

    flat = np.asarray(samples).ravel()
    if flat.size < 2:
        return "BPSK", 0.5, ["QPSK", "2-FSK", "16-QAM"]

    real = np.real(flat).astype(np.float64)
    imag = np.imag(flat).astype(np.float64)
    var_r = float(np.var(real))
    var_i = float(np.var(imag))
    peak = max(var_r, var_i)
    ratio = (min(var_r, var_i) / peak) if peak > 0 else 1.0
    if ratio < 0.5:
        return "BPSK", 0.8, ["QPSK", "2-FSK", "16-QAM"]

    # QAM: multi-level envelope => high magnitude variance
    mag = np.abs(flat)
    var_mag = float(np.var(mag))
    if var_mag > 0.04:
        return "16-QAM", 0.6, ["QPSK", "BPSK"]

    phase = np.unwrap(np.angle(flat.astype(np.complex128)))
    var_f = float(np.var(np.diff(phase)))
    if var_f < 1.0:
        return "2-FSK", 0.65, ["BPSK", "QPSK", "16-QAM"]
    return "QPSK", 0.7, ["BPSK", "2-FSK", "16-QAM"]


def analyze_file(request: dict) -> dict:
    """Coordinate load → DSP → demod → correlate → report.

    ``request`` example (info.md §12.5)::

        {
            "file_path": "sample_data/bpsk.iq",
            "sample_rate": 100000,       # required for .iq (raw has no metadata)
            "center_frequency": 0,
            "iq_format": "complex64",    # complex64 | int16 | uint8 ("auto" -> complex64)
            "modulation": "auto",        # auto | BPSK | QPSK | 2-FSK | 16-QAM | FSK
            "sync_word": "0x1ACFFC1D",   # optional hex string
        }

    Returns a report dict per info.md §13. Unsupported suffixes and
    processing failures yield an error report (``errors=[msg]``) instead of
    raising, except a missing ``sample_rate`` for ``.iq`` which raises
    ``ValueError`` (surfaced inside ``errors`` by the handler below).
    """
    raw_path = (
        request.get("file_path", "unknown") if isinstance(request, dict) else "unknown"
    )
    file_path = Path(raw_path) if raw_path else Path("unknown")
    suffix = file_path.suffix.lower()

    warnings: list[str] = []
    errors: list[str] = []

    if suffix not in (".iq", ".wav"):
        return _error_report(
            file_path.name,
            warnings,
            f"Unsupported file type: '{suffix or '(none)'}'. Supported: .iq, .wav",
        )

    try:
        if suffix == ".iq":
            iq_format = str(request.get("iq_format", "complex64") or "complex64")
            if iq_format.lower() == "auto":
                iq_format = "complex64"
            sample_rate_raw = request.get("sample_rate")
            if sample_rate_raw is None:
                raise ValueError(
                    "sample_rate is required for .iq files "
                    "(raw IQ has no metadata). Re-run with e.g. sample_rate=100000."
                )
            sample_rate = float(sample_rate_raw)
            samples = load_iq(str(file_path), dtype=iq_format)
            file_type = "iq"
        else:
            # WAV carries its own sample rate; any requested rate is ignored.
            samples, sample_rate = load_wav(str(file_path))
            sample_rate = float(sample_rate)
            file_type = "wav"

        freqs, psd_db = compute_psd(samples, sample_rate)

        center_freq = float(estimate_center_frequency(freqs, psd_db))
        bandwidth = float(estimate_bandwidth(freqs, psd_db))
        snr = float(estimate_snr(psd_db))
        sample_rate_estimate = float(estimate_sampling_rate(bandwidth))
        try:
            symbol_rate_estimate = float(estimate_symbol_rate(samples, sample_rate))
        except Exception:
            symbol_rate_estimate = 0.0
        try:
            cfo_estimate = float(estimate_cfo(samples, sample_rate))
        except Exception:
            cfo_estimate = 0.0

        requested = str(request.get("modulation", "auto") or "auto").upper()
        if requested == "AUTO":
            estimated_type, confidence, alternatives = _classify_modulation(samples)
            mode = estimated_type
        elif requested in ("BPSK", "QPSK", "2FSK", "2-FSK", "FSK"):
            estimated_type = {"2FSK": "2-FSK", "FSK": "2-FSK"}.get(requested, requested)
            confidence = 0.5  # user-selected, not estimated
            alternatives = []
            mode = estimated_type
        elif requested in ("16-QAM", "QAM16", "QAM"):
            estimated_type = "16-QAM"
            confidence = 0.5
            alternatives = ["BPSK", "QPSK", "2-FSK"]
            mode = "16-QAM"
        else:
            warnings.append(
                f"Unsupported modulation '{requested}'. Used BPSK fallback."
            )
            estimated_type = "BPSK"
            confidence = 0.5
            alternatives = []
            mode = "BPSK"

        if mode == "BPSK":
            bits = demod_bpsk(samples)
        elif mode == "QPSK":
            bits = demod_qpsk(samples)
        elif mode in ("2-FSK", "2FSK"):
            # Per-sample instantaneous-frequency demod; symbol-timing
            # recovery (samples_per_symbol handling) is a V2 concern.
            bits = demod_2fsk(samples)
        elif mode == "16-QAM":
            bits = demod_qam16(samples)
        else:  # pragma: no cover - defensive; mode is normalized above
            warnings.append(f"Unsupported modulation '{mode}'. Used BPSK fallback.")
            mode = "BPSK"
            bits = demod_bpsk(samples)

        sync_word = request.get("sync_word")
        if sync_word:
            sync_bits = hex_to_bits(str(sync_word))
            found = find_header(np.asarray(bits), sync_bits)
            offset = int(found.get("offset", -1))
            score = float(found.get("score", 0.0))
            detected = bool(found.get("detected", score >= CORRELATION_THRESHOLD))
            correlation = {
                "sync_word": sync_word,
                "header_offset": offset,
                "offset": offset,
                "score": score,
                "detected": detected,
            }
        else:
            correlation = {
                "sync_word": None,
                "header_offset": -1,
                "offset": -1,
                "score": 0.0,
                "detected": False,
            }

        fec_res = score_fec_candidates(bits)
        ilv_res = score_deinterleave_candidates(bits)

        report = {
            "meta": {
                "tool_name": TOOL_NAME,
                "version": VERSION,
                "generated_at": _utc_now(),
            },
            "input": {
                "file_name": file_path.name,
                "file_type": file_type,
                "sample_rate": sample_rate,
                "center_frequency": request.get("center_frequency"),
                "iq_format": request.get("iq_format", "auto"),
            },
            "signal": {
                "num_samples": len(samples),
                "duration_seconds": float(len(samples) / sample_rate),
                "center_frequency_estimate": center_freq,
                "bandwidth_estimate": bandwidth,
                "snr_db": snr,
                "sample_rate_estimate": sample_rate_estimate,
                "symbol_rate_estimate": symbol_rate_estimate,
                "cfo_estimate_hz": cfo_estimate,
            },
            "modulation": {
                "estimated_type": estimated_type,
                "confidence": float(confidence),
                "alternatives": list(alternatives),
            },
            "demodulation": {
                "mode": mode,
                "num_bits": len(bits),
                "bitstream_file": None,
                "bits_preview": [int(b) for b in list(bits[:2048])],
            },
            "correlation": correlation,
            # Candidate-score stubs only: the MVP never claims blind FEC /
            # interleaver detection (see info.md §30 demo-defense lines).
            "fec": {
                "candidate": fec_res["candidate"],
                "confidence": float(fec_res["confidence"]),
                "crc_pass": fec_res["crc_pass"],
                "candidates": fec_res.get("candidates", []),
            },
            "interleaving": {
                "candidate": ilv_res["candidate"],
                "depth": ilv_res["depth"],
                "confidence": float(ilv_res["confidence"]),
                "candidates": ilv_res.get("candidates", []),
            },
            # MVP hint for eye diagram: current synthetic generator uses
            # 1 sample/symbol for BPSK/QPSK/2-FSK.
            "display": {
                "samples_per_symbol": 1,
                "mode": mode,
            },
            "warnings": warnings,
            "errors": errors,
        }

        # Save bitstream/report artifacts (best-effort, never fail analysis).
        try:
            from rf_analyzer.core.report import save_bits, save_report

            out_dir = Path("output")
            out_dir.mkdir(exist_ok=True)
            bit_path = out_dir / f"{file_path.stem}_bits.bin"
            rep_path = out_dir / f"{file_path.stem}_report.json"
            if len(bits) > 0:
                save_bits(bits, str(bit_path))
                report["demodulation"]["bitstream_file"] = str(bit_path)
            save_report(report, str(rep_path))
        except Exception:
            pass

        return report

    except Exception as exc:
        return _error_report(file_path.name, warnings, str(exc))
