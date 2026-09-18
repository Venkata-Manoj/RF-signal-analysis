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

from rf_analyzer.config import (
    CORRELATION_THRESHOLD,
    DECODE_MAX_BITS,
    DECODE_TIME_BUDGET_S,
    DEFAULT_SYNC_WORD,
    MODULATION_FIT_CANDIDATES,
    MODULATION_FIT_EVM_WARN,
    MODULATION_FIT_MAX_SAMPLES,
    MODULATION_UNCORROBORATED_CONFIDENCE,
    PAYLOAD_PREVIEW_BYTES,
    TOOL_NAME,
    VERSION,
)
from rf_analyzer.core.classifier import classify_modulation as classify_modulation_hoc
from rf_analyzer.core.correlator import find_header, hex_to_bits
from rf_analyzer.core.deinterleave import (
    score_deinterleave_candidates,
    search_interleaver,
)
from rf_analyzer.core.demod import demod_2fsk, demod_bpsk, demod_qam16, demod_qpsk
from rf_analyzer.core.dsp import (
    compute_evm,
    compute_psd,
    estimate_bandwidth,
    estimate_center_frequency,
    estimate_cfo,
    estimate_fsk_symbol_period,
    estimate_sampling_rate,
    estimate_snr,
    estimate_symbol_rate,
)
from rf_analyzer.core.fec import score_fec_candidates
from rf_analyzer.core.io import detect_iq_format, load_iq, load_wav
from rf_analyzer.core.payload import payload_report

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
            "iq_format_used": None,
        },
        "signal": {},
        "modulation": {},
        "demodulation": {},
        "correlation": {},
        "payload": {},
        "fec": {},
        "interleaving": {},
        "display": {},
        "quality": {},
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
            "iq_format": "complex64",    # complex64 | int16 | uint8 | int8 | auto
            "modulation": "auto",        # auto | BPSK | QPSK | 2-FSK | 16-QAM | FSK
            "sync_word": "0x1ACFFC1D",   # optional hex string
            "decode": True,              # run the CRC-verified FEC search
            "frame_bits": None,          # optional coded-region length hint
            "decode_max_bits": 16384,    # optional search prefix cap
            "decode_time_budget_s": 4.0, # optional wall-clock ceiling
        }

    Returns a report dict per info.md §13, extended with a ``payload`` block
    (raw payload rendering plus any CRC-verified decode). Unsupported suffixes
    and processing failures yield an error report (``errors=[msg]``) instead of
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
    iq_format: str | None = None

    if suffix not in (".iq", ".wav"):
        return _error_report(
            file_path.name,
            warnings,
            f"Unsupported file type: '{suffix or '(none)'}'. Supported: .iq, .wav",
        )

    try:
        if suffix == ".iq":
            iq_format = str(request.get("iq_format", "complex64") or "complex64")
            detection: dict | None = None
            if iq_format.lower() == "auto":
                # Raw IQ carries no metadata: infer the on-disk dtype from the
                # statistics of the byte stream itself.
                detection = detect_iq_format(str(file_path))
                iq_format = str(detection.get("format", "complex64"))
                if detection.get("confidence", 0.0) < 0.6:
                    warnings.append(
                        "IQ format auto-detection was inconclusive "
                        f"(guessed '{iq_format}', confidence "
                        f"{float(detection.get('confidence', 0.0)):.2f}). "
                        "Pass iq_format explicitly if the result looks wrong."
                    )
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
        if bandwidth <= 0.0:
            # estimate_bandwidth returns 0.0 when no usable span exists. Left
            # unexplained, that reads as a measurement of "0 Hz" and silently
            # zeroes the derived sampling-rate estimate too, so say what
            # actually happened.
            warnings.append(
                "Occupied-bandwidth estimate unavailable: no frequency bin "
                "cleared the noise floor by 10 dB. The bandwidth and the "
                "sampling-rate estimate derived from it are reported as 0 Hz "
                "meaning 'not measurable', not as measurements."
            )
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

        # Samples per symbol actually used by the demodulator. BPSK/QPSK/16-QAM
        # in this MVP are one sample per symbol by construction; 2-FSK is not,
        # so its period is recovered from the instantaneous frequency below.
        samples_per_symbol = 1
        # Whether the estimated modulation was independently corroborated.
        # None = the check does not apply to this mode.
        corroborated: bool | None = None

        if mode == "BPSK":
            bits = demod_bpsk(samples)
        elif mode == "QPSK":
            bits = demod_qpsk(samples)
        elif mode in ("2-FSK", "2FSK"):
            # A real 2-FSK burst spends many samples on each bit, and the naive
            # one-bit-per-sample path would then emit the same bit dozens of
            # times in a row -- the sync word can never be found that way. The
            # period is recovered from the piecewise-constant instantaneous
            # frequency; if it cannot be recovered the estimator returns 1 and
            # the documented naive behaviour is preserved.
            samples_per_symbol = estimate_fsk_symbol_period(samples)
            bits = demod_2fsk(samples, samples_per_symbol)
            # A recovered period is independent evidence that this really is
            # constant-envelope FSK: the estimator was measured to recover the
            # period exactly for every FSK case tried and to decline for BPSK,
            # QPSK, QAM, AWGN, a tone and several audio recordings. Failing to
            # recover one does not prove the label wrong, but it does mean the
            # label is uncorroborated, and the classifier's FSK test is only a
            # narrowband test (it also fires for an unmodulated tone and for
            # music), so the tool must not stay confident about it.
            corroborated = samples_per_symbol > 1
            if samples_per_symbol > 1:
                # The |x|^2 spectral-line symbol-rate estimate assumes a
                # non-constant envelope and is meaningless for constant-modulus
                # FSK (it reported 395 kHz for a 1 kHz burst). The recovered
                # period is a direct measurement, so prefer it.
                symbol_rate_estimate = float(sample_rate) / float(samples_per_symbol)
            else:
                warnings.append(
                    "Could not recover the 2-FSK symbol period (the capture may "
                    "be shorter than a few symbols, or not constant-envelope "
                    "FSK). Demodulated at the MVP default of one bit per "
                    "sample, so the bit stream is oversampled by the unknown "
                    "number of samples per symbol."
                )
                if requested == "AUTO":
                    # Only downgrade an *estimate*. If the user asked for 2-FSK
                    # explicitly, report what was found and leave their choice
                    # alone; the warning above already carries the caveat.
                    confidence = min(confidence, MODULATION_UNCORROBORATED_CONFIDENCE)
                    warnings.append(
                        "The 2-FSK classification is uncorroborated, so its "
                        f"confidence is reported as {confidence:.2f} rather "
                        "than as a confident estimate."
                    )
        elif mode == "16-QAM":
            bits = demod_qam16(samples)
        else:  # pragma: no cover - defensive; mode is normalized above
            warnings.append(f"Unsupported modulation '{mode}'. Used BPSK fallback.")
            mode = "BPSK"
            bits = demod_bpsk(samples)

        sync_word = request.get("sync_word")
        assumed_sync_word: str | None = None
        probe_info: dict | None = None
        if not sync_word:
            # No sync word was supplied. Rather than silently starting the
            # payload at bit 0 -- which mis-frames every capture that *does*
            # carry a header, and makes the decode search miss -- probe with
            # the project default and, only if it correlates *significantly*,
            # use it as a framing hint. The probe is recorded as an assumption
            # in the report; it is never presented as a value the caller
            # supplied. `detected` already accounts for the search length, so
            # a long capture cannot produce a chance match (see
            # correlator.FALSE_ALARM_TARGET).
            probe = find_header(np.asarray(bits), hex_to_bits(DEFAULT_SYNC_WORD))
            probe_info = {
                "sync_word": DEFAULT_SYNC_WORD,
                "score": float(probe.get("score", 0.0)),
                "offset": int(probe.get("offset", -1)),
                "min_score": float(probe.get("min_score", 0.0)),
                "positions_searched": int(probe.get("positions_searched", 0)),
                "accepted": bool(probe.get("detected", False)),
            }
            if bool(probe.get("detected", False)) and int(probe.get("offset", -1)) >= 0:
                sync_word = DEFAULT_SYNC_WORD
                assumed_sync_word = DEFAULT_SYNC_WORD
                warnings.append(
                    "No sync_word was supplied; assumed the default "
                    f"{DEFAULT_SYNC_WORD} (correlation "
                    f"{float(probe.get('score', 0.0)):.2f} at bit "
                    f"{int(probe.get('offset', 0))}). Pass sync_word explicitly "
                    "if the capture uses a different header."
                )
            elif probe_info["score"] > 0.0:
                # A match was seen but is not significant for a capture this
                # long. Say so, rather than reporting a bare "no header" and
                # leaving the user to wonder whether the tool even looked.
                warnings.append(
                    f"No sync_word was supplied and the default "
                    f"{DEFAULT_SYNC_WORD} did not match significantly "
                    f"(best score {probe_info['score']:.3f} at bit "
                    f"{probe_info['offset']}, below the "
                    f"{probe_info['min_score']:.3f} required over "
                    f"{probe_info['positions_searched']} searched positions). "
                    "Treating the whole bit stream as payload; pass sync_word "
                    "explicitly if the capture really does carry a header."
                )

        if sync_word:
            sync_bits = hex_to_bits(str(sync_word))
            found = find_header(np.asarray(bits), sync_bits)
            offset = int(found.get("offset", -1))
            score = float(found.get("score", 0.0))
            detected = bool(found.get("detected", score >= CORRELATION_THRESHOLD))
            correlation = {
                "sync_word": sync_word,
                "assumed": assumed_sync_word is not None,
                "header_offset": offset,
                "offset": offset,
                "score": score,
                "detected": detected,
                "sync_length": int(sync_bits.size),
                # Statistical-significance context: a fixed score threshold is
                # weaker evidence on a long capture than a short one.
                "min_score": float(found.get("min_score", 0.0)),
                "min_matches": int(found.get("min_matches", 0)),
                "positions_searched": int(found.get("positions_searched", 0)),
                "probe": probe_info,
            }
        else:
            sync_bits = None
            correlation = {
                "sync_word": None,
                "assumed": False,
                "header_offset": -1,
                "offset": -1,
                "score": 0.0,
                "detected": False,
                "sync_length": 0,
                "min_score": 0.0,
                "min_matches": 0,
                "positions_searched": 0,
                "probe": probe_info,
            }

        # The payload starts right after a *detected* sync word; with no header
        # (or no confidence in one) the whole bit stream is treated as payload.
        if sync_bits is not None and correlation["detected"] and offset >= 0:
            payload_start = offset + int(sync_bits.size)
        else:
            payload_start = 0

        # ---- Blind candidate scores (honest: never a detection claim) ----
        fec_res = score_fec_candidates(bits)
        ilv_res = score_deinterleave_candidates(bits)

        # ---- Verified decode: de-interleave -> FEC -> CRC-16 ----
        # This is a *search*, not a detection: nothing is reported as decoded
        # unless an independent CRC-16 check passes (info.md §30).
        decode_enabled = bool(request.get("decode", True))
        frame_bits = request.get("frame_bits")
        # The search is wall-clock bounded so a huge capture cannot blow the
        # §NFR-03 budget. Both limits are overridable per request: a caller who
        # knows the frame is long can raise them, and a correctness test can
        # raise them to remove timing from the assertion.
        try:
            decode_max_bits = int(
                request.get("decode_max_bits", DECODE_MAX_BITS) or DECODE_MAX_BITS
            )
        except (TypeError, ValueError):
            decode_max_bits = DECODE_MAX_BITS
        try:
            decode_budget = float(
                request.get("decode_time_budget_s", DECODE_TIME_BUDGET_S)
                or DECODE_TIME_BUDGET_S
            )
        except (TypeError, ValueError):
            decode_budget = DECODE_TIME_BUDGET_S

        decode_res: dict | None = None
        if decode_enabled:
            try:
                decode_res = search_interleaver(
                    bits,
                    start_offset=payload_start,
                    max_bits=decode_max_bits,
                    time_budget_s=decode_budget,
                    frame_bits=int(frame_bits) if frame_bits else None,
                )
            except Exception as exc:  # a failed search is a miss, not an error
                warnings.append(f"FEC/interleaver search skipped: {exc}")
                decode_res = None

        validated = bool(decode_res and decode_res.get("validated"))
        best = decode_res.get("best") if decode_res else None

        payload_block = payload_report(
            bits,
            start_offset=payload_start,
            decoded=best,
            max_bytes=PAYLOAD_PREVIEW_BYTES,
        )
        payload_block["decode_search"] = {
            "enabled": decode_enabled,
            "attempts": len(decode_res.get("attempts", [])) if decode_res else 0,
            "validated": validated,
            "budget_exhausted": (
                bool(decode_res.get("budget_exhausted", False)) if decode_res else False
            ),
            "max_bits": decode_max_bits,
            "time_budget_s": decode_budget,
        }
        if decode_enabled and not validated:
            warnings.append(
                "No CRC-valid FEC/interleaver hypothesis matched. This is "
                "expected for a capture that carries no CRC-framed payload "
                "(e.g. an analog or unframed real-world recording); the bytes "
                "shown are the raw demodulated bit stream, not a decoded message."
            )

        if validated and best:
            # A CRC-verified decode replaces the blind guesses with facts.
            fec_block = {
                "candidate": best.get("candidate"),
                "confidence": float(best.get("confidence", 0.0)),
                "crc_pass": True,
                "validated": True,
                "params": best.get("params", {}),
                "errors_corrected": int(best.get("errors_corrected", 0)),
                "candidates": fec_res.get("candidates", []),
            }
            ilv_block = {
                "candidate": (best.get("params") or {}).get("interleaver"),
                "depth": ilv_res.get("depth"),
                "confidence": float(best.get("confidence", 0.0)),
                "validated": True,
                "candidates": ilv_res.get("candidates", []),
            }
        else:
            # No verified decode: report the capped heuristic scores only.
            fec_block = {
                "candidate": fec_res["candidate"],
                "confidence": float(fec_res["confidence"]),
                "crc_pass": fec_res["crc_pass"],
                "validated": False,
                "candidates": fec_res.get("candidates", []),
            }
            ilv_block = {
                "candidate": ilv_res["candidate"],
                "depth": ilv_res["depth"],
                "confidence": float(ilv_res["confidence"]),
                "validated": False,
                "candidates": ilv_res.get("candidates", []),
            }

        # ---- Constellation quality (EVM/MER): an independent sanity check ----
        # A low EVM means the received symbols really do land on the ideal
        # constellation for `mode`, so a wrong modulation guess shows up as a
        # large EVM instead of passing silently. FSK has no constellation and
        # reports applicable=False. Equalisation/carrier recovery are V2
        # (info.md §32), so a rotated or offset signal inflates this number.
        try:
            quality_block = compute_evm(
                samples, mode=mode, samples_per_symbol=samples_per_symbol
            )
        except Exception as exc:  # quality is a nicety, never a hard failure
            quality_block = {
                "applicable": False,
                "mode": mode,
                "modulation_order": None,
                "evm_percent": None,
                "mer_db": None,
                "snr_db_from_evm": None,
                "n_symbols": 0,
                "error": str(exc),
            }

        # ---- Cross-check the modulation guess against the constellation fit ----
        # The classifier works on higher-order statistics alone; EVM is an
        # independent test of how well the *chosen* constellation explains the
        # samples. A poor fit means the guess does not describe the signal.
        #
        # EVM is deliberately NOT used to rank modulations against each other:
        # a denser constellation can fit any point cloud better, so raw EVM
        # falls monotonically with constellation size (on a real GPS L1
        # capture: 93.5% at BPSK down to 26.2% at 64-QAM). The per-candidate
        # numbers are reported as context only, and the warning says so.
        evm_percent = quality_block.get("evm_percent")
        if quality_block.get("applicable") and evm_percent is not None:
            evm_value = float(evm_percent)
            fit_ok = evm_value <= MODULATION_FIT_EVM_WARN
            quality_block["fit_ok"] = fit_ok
            if not fit_ok:
                fit_samples = samples[:MODULATION_FIT_MAX_SAMPLES]
                table: list[dict] = []
                for candidate in MODULATION_FIT_CANDIDATES:
                    if str(candidate).upper() == mode.upper():
                        continue
                    try:
                        candidate_evm = compute_evm(
                            fit_samples,
                            mode=candidate,
                            samples_per_symbol=samples_per_symbol,
                        )
                    except Exception:
                        continue
                    if candidate_evm.get("applicable") and (
                        candidate_evm.get("evm_percent") is not None
                    ):
                        table.append(
                            {
                                "mode": str(candidate),
                                "modulation_order": candidate_evm.get(
                                    "modulation_order"
                                ),
                                "evm_percent": float(candidate_evm["evm_percent"]),
                            }
                        )
                table.sort(key=lambda row: row["evm_percent"])
                quality_block["candidates"] = table

                context = ""
                if table:
                    context = (
                        f" For context, EVM on this capture is "
                        f"{table[-1]['evm_percent']:.1f}% for "
                        f"{table[-1]['mode']} and {table[0]['evm_percent']:.1f}% "
                        f"for {table[0]['mode']}; a denser constellation fits any "
                        "sample cloud better, so those numbers are context, not a "
                        "ranking."
                    )
                warnings.append(
                    f"Modulation fit is poor: the {mode} constellation leaves "
                    f"{evm_value:.1f}% error-vector magnitude, so the modulation "
                    f"estimate is unreliable.{context} Set 'modulation' "
                    "explicitly if you know the scheme."
                )

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
                "iq_format_used": iq_format,
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
                # True = independently corroborated by the symbol-period
                # recovery; False = not corroborated (confidence is capped, see
                # MODULATION_UNCORROBORATED_CONFIDENCE); None = the check does
                # not apply to this mode.
                "corroborated": corroborated,
            },
            "demodulation": {
                "mode": mode,
                "num_bits": len(bits),
                "bitstream_file": None,
                "bits_preview": [int(b) for b in list(bits[:2048])],
            },
            "correlation": correlation,
            "payload": payload_block,
            # Constellation quality: EVM/MER plus the modulation order, so a
            # consumer can tell "BPSK at 12% EVM" from "BPSK at 180% EVM".
            "quality": quality_block,
            # Candidate scores are blind heuristics (capped, crc_pass unknown);
            # a CRC-verified decode is reported separately and only when it
            # actually happened. See info.md §30 demo-defense lines.
            "fec": fec_block,
            "interleaving": ilv_block,
            # MVP hint for eye diagram: current synthetic generator uses
            # 1 sample/symbol for BPSK/QPSK/2-FSK.
            "display": {
                "samples_per_symbol": samples_per_symbol,
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
            decoded_block = report["payload"].get("decoded") or {}
            if decoded_block.get("available"):
                payload_path = out_dir / f"{file_path.stem}_payload.bin"
                payload_path.write_bytes(bytes.fromhex(decoded_block.get("hex", "")))
                report["payload"]["decoded"]["file"] = str(payload_path)
            save_report(report, str(rep_path))
        except Exception:
            pass

        return report

    except Exception as exc:
        return _error_report(file_path.name, warnings, str(exc))
