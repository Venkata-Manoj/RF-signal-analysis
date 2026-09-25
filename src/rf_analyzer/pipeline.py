"""End-to-end analysis entrypoint. See info.md §12.5 / §13 / §20.

Headless contract (see AGENTS.md): the GUI must call :func:`analyze_file`
and never duplicate DSP logic. The returned dict always follows the report
schema in info.md §13 (keys ``meta``/``input``/``signal``/``modulation``/
``demodulation``/``correlation``/``fec``/``interleaving``/``warnings``/
``errors``).

Completed-system notes (see info.md §12):

* DSP: PSD-peak center frequency, median-floor bandwidth/SNR, fused rate estimation.
* Demod: coherent receive chain by default (Costas / Gardner / Mueller-Muller /
  CMA / LMS); honest fallback to phase-aligned slicers (``real > 0``/``imag > 0``),
  2-FSK via ``diff(unwrap(angle))``.
* ``AUTO`` modulation: ML head first, HOC/rules fallback.
* FEC/interleaving: real codecs + CRC-verified search — never claim blind detection.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rf_analyzer.config import (
    BURST_FRAME_SLACK_BITS,
    CORRELATION_THRESHOLD,
    DECODE_MAX_BITS,
    DECODE_TIME_BUDGET_S,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SYNC_WORD,
    MODULATION_FIT_CANDIDATES,
    MODULATION_FIT_EVM_WARN,
    MODULATION_FIT_MAX_SAMPLES,
    MODULATION_HEADER_CORROBORATED_CONFIDENCE,
    MODULATION_RETRY_MIN_SCORE,
    MODULATION_UNCORROBORATED_CONFIDENCE,
    PAYLOAD_PREVIEW_BYTES,
    TOOL_NAME,
    VERSION,
)
from rf_analyzer.core.classifier import (
    ANALOG_REJECT_LABELS,
)
from rf_analyzer.core.classifier import (
    classify_modulation as classify_modulation_hoc,
)
from rf_analyzer.core.correlator import discover_sync_word, find_header, hex_to_bits
from rf_analyzer.core.deinterleave import (
    score_deinterleave_candidates,
    search_interleaver,
)
from rf_analyzer.core.demod import (
    demod_2fsk,
    demod_4fsk,
    demod_8psk,
    demod_64qam,
    demod_bpsk,
    demod_qam16,
    demod_qpsk,
)
from rf_analyzer.core.dsp import (
    compute_evm,
    compute_psd,
    estimate_bandwidth,
    estimate_burst_region,
    estimate_center_frequency,
    estimate_cfo,
    estimate_fsk_symbol_period,
    estimate_sampling_rate_candidates,
    estimate_snr,
)
from rf_analyzer.core.fec import score_fec_candidates
from rf_analyzer.core.io import (
    FileTooLargeError,
    check_size,
    detect_iq_format,
    load_iq,
    load_sigmf_meta,
    load_wav,
)
from rf_analyzer.core.payload import payload_report
from rf_analyzer.core.rate_est import (
    choose_cyclo_nfft,
    estimate_sampling_rate_fused,
    estimate_symbol_rate_fused,
)
from rf_analyzer.core.receiver import demodulate as coherent_demodulate

#: Minimum required report keys per info.md §13 (mirrored by external consumers).
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

#: Additive top-level blocks defined beyond the core §13 schema.
ADDITIVE_REPORT_KEYS = (
    "payload",
    "display",
    "quality",
    "judgments",
    "gps",
    "video",
)

#: Complete set of top-level report keys returned by analyze_file() across both
#: success and error branches (ensuring unified top-level schema shape).
TOP_LEVEL_KEYS = frozenset(REQUIRED_REPORT_KEYS + ADDITIVE_REPORT_KEYS)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _judgment_block(
    samples: np.ndarray,
    bits: np.ndarray,
    sample_rate: float | None,
    sync_word: str | None,
    quality_block: dict,
    validated: bool,
) -> dict:
    """Additive TypeSafe-pattern judgments over the analyzed state.

    Asks independent Choice (modulation) / Noul (header, decode) / Score
    (quality) questions over the same capture and routes on confidence:
    ``act`` needs nothing, ``confirm`` is already covered by the pipeline's
    header arbitration + warnings, ``escalate`` appends an explicit note.
    A failed judgment degrades to ``available: False``, never to a guess.
    """
    try:
        from rf_analyzer.core import judgments as judgments_mod

        evm = None
        try:
            evm = quality_block.get("evm_percent")
            evm = None if evm is None else float(evm)
        except Exception:
            evm = None
        crc: bool | None = True if validated else None
        answers = judgments_mod.ask_all(
            np.asarray(samples),
            np.asarray(bits),
            sync_word=sync_word,
            evm_percent=evm,
            validated=bool(validated),
            crc_pass=crc,
            sample_rate=float(sample_rate) if sample_rate else None,
        )
        mod = answers["modulation"]
        hdr = answers["header_present"]
        qual = answers["signal_quality"]
        dec = answers["decode_trust"]
        assert isinstance(mod, judgments_mod.Choice) and isinstance(
            qual, judgments_mod.Score
        )
        assert isinstance(hdr, judgments_mod.Noul) and isinstance(
            dec, judgments_mod.Noul
        )
        routing = {
            "modulation": judgments_mod.route(float(mod.confidence)),
            "header": (
                "act"
                if float(hdr.noul) >= 0.9
                else ("confirm" if float(hdr.noul) >= 0.5 else "escalate")
            ),
            "decode": "act" if float(dec.noul) >= 0.9 else "escalate",
        }
        return {
            "available": True,
            "modulation": {
                "choice": mod.choice,
                "confidence": float(mod.confidence),
                "probabilities": {k: float(v) for k, v in mod.probabilities.items()},
                "method": mod.method,
                "route": routing["modulation"],
            },
            "header_present": {
                "noul": float(hdr.noul),
                "method": hdr.method,
                "route": routing["header"],
            },
            "signal_quality": {
                "score": float(qual.score),
                "levels": list(qual.legend),
                "probabilities": {k: float(v) for k, v in qual.probabilities.items()},
                "confidence": float(qual.confidence),
                "method": qual.method,
            },
            "decode_trust": {
                "noul": float(dec.noul),
                "method": dec.method,
                "route": routing["decode"],
            },
            "routing": routing,
        }
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


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
            "sample_rate_assumed": False,
            "sample_rate_source": None,
            "center_frequency": None,
            "iq_format": None,
            "iq_format_used": None,
            "iq_format_confidence": None,
            "auto_sample_rate": False,
            "auto_sync": True,
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
        "gps": {
            "attempted": False,
            "reason": message,
            "svs_searched": [],
            "acquired": [],
            "prompt_signs": [],
            "cn0_dbhz": None,
            "warnings": [],
        },
        "video": {
            "found": False,
            "line_rate_hz": None,
            "confidence": 0.0,
            "harmonic_ratio": None,
            "method": "envelope-fft",
            "reason": message,
        },
        "judgments": {
            "available": False,
            "reason": message,
        },
        "warnings": list(warnings),
        "errors": [message],
    }


def _classify_modulation(
    samples: np.ndarray, sample_rate: float | None = None
) -> tuple[str, float, list[str]]:
    """Modulation classification: ML head first, HOC/rules fallback.

    :func:`rf_analyzer.core.classifier.classify_modulation` already tries the
    trained ML head and falls back to the rule/HOC path on any failure, so a
    broken or missing model can only cost accuracy, never crash analysis.
    The heuristic below is the last resort when even the rules decline.
    """
    try:
        mod_type, conf, alts = classify_modulation_hoc(samples, sample_rate=sample_rate)
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


#: Modulations the header search may fall back to, cheapest first. BPSK and
#: QPSK are one decision each; 16-QAM and 2-FSK cost more, so they come last.
#: 8PSK/64-QAM/4-FSK included so a misclassified burst can still be recovered.
HEADER_FALLBACK_MODULATIONS = (
    "BPSK",
    "QPSK",
    "16-QAM",
    "8PSK",
    "64-QAM",
    "2-FSK",
    "4-FSK",
)


def _naive_demodulate(samples: np.ndarray, mode: str) -> tuple[np.ndarray | None, int]:
    """Phase-aligned slicers for symbol-spaced captures."""
    if mode == "BPSK":
        return demod_bpsk(samples), 1
    if mode == "QPSK":
        return demod_qpsk(samples), 1
    if mode == "16-QAM":
        return demod_qam16(samples), 1
    if mode == "8PSK":
        return demod_8psk(samples), 1
    if mode == "64-QAM":
        return demod_64qam(samples), 1
    if mode in ("2-FSK", "2FSK"):
        period = estimate_fsk_symbol_period(samples)
        return demod_2fsk(samples, period), period
    if mode in ("4-FSK", "4FSK"):
        period = estimate_fsk_symbol_period(samples)
        if period <= 1:
            period = 1
        return demod_4fsk(samples, period), period
    return None, 1


#: Rotational symmetry order per linear scheme. FSK carries its information
#: in the instantaneous frequency, so a residual carrier only shifts every
#: tone level together and there is no phase ambiguity to resolve.
_SYMMETRY_ORDER = {"BPSK": 2, "QPSK": 4, "8PSK": 8, "16-QAM": 4, "64-QAM": 4}

#: Bit widths per symbol for the naive symbol counts below.
_BITS_PER_SYMBOL = {
    "BPSK": 1,
    "QPSK": 2,
    "8PSK": 3,
    "16-QAM": 4,
    "64-QAM": 6,
    "2-FSK": 1,
    "2FSK": 1,
    "4-FSK": 2,
    "4FSK": 2,
}


def _normalize_recovery(metrics: dict, mode: str) -> dict:
    """Recovery stages as plain strings for the report.

    The linear and FSK receive paths name their stages differently; this
    maps both onto ``{timing, carrier, equalization, iq_correction}`` so a
    consumer reads one shape. Values are strings only (JSON-safe).
    """
    recovery = metrics.get("recovery") or {}
    iq = metrics.get("iq_correction")
    if isinstance(iq, dict):
        iq_text = (
            "applied"
            if iq.get("applied")
            else f"not applied ({iq.get('reason', 'no reason')})"
        )
    elif mode.upper() in ("2-FSK", "2FSK", "4-FSK", "4FSK"):
        iq_text = "n/a (FSK discriminator)"
    else:
        iq_text = "not estimated"
    return {
        "timing": str(recovery.get("timing", "none")),
        "carrier": str(recovery.get("carrier", "none")),
        "equalization": str(
            recovery.get("equalizer", recovery.get("equalization", "none"))
        ),
        "iq_correction": str(iq_text),
    }


def _ambiguity_block(mode: str, rotation: int | None) -> dict:
    """Residual carrier-rotation ambiguity for the report (JSON-safe)."""
    key = str(mode).upper()
    if key in ("2-FSK", "2FSK", "4-FSK", "4FSK"):
        return {
            "symmetry_order": 1,
            "rotation": 0,
            "resolved": True,
            "note": (
                "constant-envelope discriminator: a residual carrier "
                "shifts every tone level together and is absorbed by "
                "estimating the levels from the data"
            ),
        }
    return {
        "symmetry_order": int(_SYMMETRY_ORDER.get(key, 0) or 0),
        "rotation": None if rotation is None else int(rotation),
        "resolved": rotation is not None,
        "note": (
            "Costas loop locks modulo the constellation symmetry; "
            "resolved against the sync word when one was supplied, "
            "otherwise reported unresolved rather than guessed"
            if rotation is None
            else "rotation resolved against the supplied sync word"
        ),
    }


def _naive_receiver_info(
    requested: str, mode: str, sps: int, n_symbols: int | None, reason: str
) -> dict:
    """Report block for a bit stream the naive slicers produced."""
    return {
        "path": "naive",
        "requested": requested,
        "locked": None,
        "reason": reason,
        "sps": int(sps),
        "n_symbols": None if n_symbols is None else int(n_symbols),
        "recovery": {
            "timing": "none",
            "carrier": "none (phase-aligned slicer)",
            "equalization": "none",
            "iq_correction": "not estimated",
        },
        "ambiguity": _ambiguity_block(mode, None),
        "evm_percent": None,
        "rotation": None,
    }


def _demodulate_with_info(
    samples: np.ndarray,
    mode: str,
    *,
    sync_bits: np.ndarray | None = None,
    receiver: str = "auto",
) -> tuple[np.ndarray | None, int, dict]:
    """Demodulate one candidate, trying the coherent chain first in auto mode.

    Returns ``(bits, samples_per_symbol, receiver_info)`` where
    ``receiver_info`` always describes the path that produced ``bits``:
    ``path`` is ``"coherent"`` only when ``receiver.demodulate`` ran, and
    ``"naive"`` otherwise (requested naive, analog placeholder, abstain, or
    unconfirmed loops). ``locked`` is a bool for coherent attempts and
    ``None`` for naive slicers, which have no lock concept.

    A coherent result earns the bit stream only with independent evidence:
    when a sync word is known the bits must correlate it (a "locked" loop
    on a misestimated sps still slices garbage -- measured: sps=8 reported
    on a 584-sample symbol-spaced capture with a confident lock flag), and
    otherwise the loop lock itself decides. This keeps a wrong sps estimate
    from hiding a header the naive path would have found.
    """
    requested = str(receiver or "auto").strip().lower()
    if requested not in ("auto", "naive"):
        requested = "auto"
    bps = int(_BITS_PER_SYMBOL.get(str(mode).upper(), 1))

    def _naive(reason: str) -> tuple[np.ndarray | None, int, dict]:
        bits, sps = _naive_demodulate(samples, mode)
        n_symbols = None if bits is None else int(np.asarray(bits).size // max(bps, 1))
        return (
            bits,
            int(sps),
            _naive_receiver_info(requested, mode, int(sps), n_symbols, reason),
        )

    if requested == "naive":
        return _naive("receiver=naive requested: naive slicers used directly")

    try:
        coherent = coherent_demodulate(
            samples, mode, sync_bits=sync_bits, prefer_coherent=True
        )
    except Exception as exc:  # a failed estimator is a miss, never a crash
        return _naive(f"robust receiver raised {type(exc).__name__}: {exc}")
    if coherent.get("path") != "coherent" or np.asarray(coherent.get("bits")).size == 0:
        metrics = coherent.get("metrics") or {}
        return _naive(
            "robust receiver abstained "
            f"({metrics.get('reason', 'sps<=1; use naive demod')}); "
            "naive slicers used instead"
        )
    bits = np.asarray(coherent["bits"], dtype=np.uint8)
    sps = int(coherent.get("sps") or 0)
    metrics = coherent.get("metrics") or {}
    locked = bool(coherent.get("locked", False))
    rotation = coherent.get("rotation")
    rotation = None if rotation is None else int(rotation)
    # A coherent result earns the bit stream only with independent evidence:
    # when a sync word is known the bits must correlate it (a "locked" loop
    # on a misestimated sps still slices garbage -- measured: sps=8 reported
    # on a 584-sample symbol-spaced capture with a confident lock flag), and
    # otherwise the loop lock itself decides.
    sync = None if sync_bits is None else np.asarray(sync_bits).ravel()
    header_ok: bool | None = None
    if sync is not None and sync.size > 0 and bits.size >= sync.size:
        try:
            header_ok = bool(find_header(bits, sync).get("detected", False))
        except Exception:
            header_ok = False
    use_coherent = header_ok if header_ok is not None else locked
    if not use_coherent:
        if header_ok is False:
            reason = (
                "coherent bits missed the supplied sync word "
                "(below the CFAR threshold); naive slicers used instead"
            )
        else:
            reason = str(metrics.get("reason") or "carrier/timing did not lock")
            reason = (
                f"robust receiver did not lock ({reason}); "
                "naive slicers used instead"
            )
        bits0, naive_sps, info = _naive(reason)
        info["coherent_attempt"] = {
            "locked": bool(locked),
            "reason": str(metrics.get("reason") or ""),
            "sps": sps,
        }
        return bits0, naive_sps, info
    recovery = _normalize_recovery(metrics, mode)
    n_symbols = metrics.get("n_symbols")
    try:
        n_symbols = None if n_symbols is None else int(n_symbols)
    except (TypeError, ValueError):
        n_symbols = None
    evm = metrics.get("evm_percent")
    try:
        evm = None if evm is None else float(evm)
    except (TypeError, ValueError):
        evm = None
    return (
        bits,
        sps,
        {
            "path": "coherent",
            "requested": requested,
            "locked": True,
            "reason": "",
            "sps": sps,
            "n_symbols": n_symbols,
            "recovery": recovery,
            "ambiguity": _ambiguity_block(mode, rotation),
            "evm_percent": evm,
            "rotation": rotation,
        },
    )


def _demodulate_for(
    samples: np.ndarray,
    mode: str,
    *,
    sync_bits: np.ndarray | None = None,
    receiver: str = "auto",
) -> tuple[np.ndarray | None, int]:
    """Demodulate with one candidate modulation.

    Prefers the coherent receive chain (Costas / Gardner / CMA-LMS) when
    samples-per-symbol can be estimated above 1. Symbol-spaced captures
    (``framing.modulate``, one sample per symbol) keep the phase-aligned
    naive path -- the coherent chain abstains rather than inventing an
    oversampling factor. Returns ``(bits, samples_per_symbol)``.
    """
    bits, sps, _ = _demodulate_with_info(
        samples, mode, sync_bits=sync_bits, receiver=receiver
    )
    return bits, sps


def _corroborate_modulation(
    samples: np.ndarray,
    sync_bits: np.ndarray,
    *,
    current: str,
    receiver: str = "auto",
) -> tuple[str, np.ndarray, int, dict, bool | None, dict] | None:
    """Find a modulation whose demodulation can actually see the sync word.

    The classifier decides from whole-capture statistics, and those are not
    invariant to how much noise surrounds the burst -- a clean BPSK burst is
    labelled QPSK or 8PSK once enough noise is padded either side of it. A wrong
    label hides the header, so without this the report says "no frame found" and
    sends the user looking for a framing bug that is really a classification one.

    Header correlation is independent of the classifier, so it is a fair
    arbiter. Returns ``(mode, bits, samples_per_symbol, found, corroborated,
    receiver_info)`` for the first candidate that correlates the header, or
    ``None`` when the current modulation is as good as any. ``receiver_info``
    always describes the path that produced ``bits``.
    """
    wanted = str(current).upper()
    for candidate in HEADER_FALLBACK_MODULATIONS:
        if candidate.upper() == wanted:
            continue
        try:
            bits, samples_per_symbol, info = _demodulate_with_info(
                samples, candidate, sync_bits=sync_bits, receiver=receiver
            )
        except Exception:  # a bad candidate is a miss, never a crash
            continue
        if bits is None or bits.size < sync_bits.size:
            continue
        found = find_header(np.asarray(bits), sync_bits)
        if bool(found.get("detected", False)):
            corroborated = samples_per_symbol > 1 if candidate == "2-FSK" else None
            return candidate, bits, samples_per_symbol, found, corroborated, info
    return None


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
            "auto_sample_rate": False,   # estimate rate when sample_rate missing
            "auto_sync": True,           # discover sync word when not supplied
            "receiver": "auto",          # auto (robust chain w/ honest fallback)
                                         # | naive (phase-aligned slicers)
        }

    Demodulation path (robust receiver):

    * ``receiver="auto"`` (default) demodulates the selected mode with
      ``core.receiver`` (matched filter, Gardner/Mueller timing, Costas /
      QAM phase recovery, CMA/LMS equalisation, blind IQ correction) at the
      ``estimate_sps`` oversampling factor, resolving the residual Costas
      rotation against an explicitly supplied sync word. Bit slicing uses the
      receiver's own slicer -- the naive ``core.demod`` slicers are not
      consulted for the final bit stream.
    * The receiver abstains rather than inventing a rate (``sps<=1``,
      unlocked loops, exceptions). A coherent result earns the final bit
      stream only with independent evidence: when a sync word is known the
      bits must correlate it (a "locked" loop on a misestimated sps still
      slices garbage), otherwise the loop lock decides. Otherwise the
      pipeline falls back to the naive slicers and records the fallback
      plus its reason in ``demodulation.receiver`` with a warning.
    * ``receiver="naive"`` keeps the previous behaviour bit-for-bit.
      ``display.samples_per_symbol`` always carries the sps actually used,
      so the burst-to-``frame_bits`` conversion stays correct on either path.

    Automation for unknown captures (Task 5):

    * ``iq_format="auto"`` infers the dtype via ``io.detect_iq_format`` and
      reports ``input.iq_format_used`` plus ``iq_format_confidence``; a
      confidence below 0.6 warns.
    * ``auto_sample_rate=True`` lets a ``.iq`` file without ``sample_rate``
      run: the SigMF sidecar is tried first, else
      ``dsp.estimate_sampling_rate_candidates`` +
      ``rate_est.estimate_sampling_rate_fused`` propose a working rate at a
      provisional ``DEFAULT_SAMPLE_RATE``. The value is flagged
      ``input.sample_rate_assumed=True`` with
      ``sample_rate_source="hypothesis"`` and a warning -- raw IQ carries no
      rate metadata, so this is a feasibility floor, not a measurement. An
      explicit ``sample_rate`` always wins.
    * ``auto_sync=True`` (default) scans ``correlator.discover_sync_word``
      candidates with the CFAR gate when no ``sync_word`` was supplied; the
      winner is reported with ``correlation.assumed=True`` plus a warning. A
      manual ``sync_word`` always wins.
    * ``frame_bits`` defaults to the ``signal.burst``-derived hint (biased
      long by ``BURST_FRAME_SLACK_BITS``); an explicit value always wins.

    Returns a report dict per info.md §13, extended with a ``payload`` block
    (raw payload rendering plus any CRC-verified decode). Unsupported suffixes
    and processing failures yield an error report (``errors=[msg]``) instead of
    raising, except a missing ``sample_rate`` for ``.iq`` without
    ``auto_sample_rate`` which raises ``ValueError`` (surfaced inside
    ``errors`` by the handler below).
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
        # NFR-01 size gate: reject oversize captures before any preview or
        # full load, surfacing an error report instead of loading. The
        # loaders re-check via io.check_size (TOCTOU defense in depth).
        try:
            check_size(file_path)
        except FileTooLargeError as exc:
            return _error_report(file_path.name, warnings, str(exc))

        # Automation flags (Task 5). Explicit values always win over auto.
        _auto_sample = bool(request.get("auto_sample_rate", False))
        _auto_sync_raw = request.get("auto_sync", True)
        if isinstance(_auto_sync_raw, str):
            auto_sync = _auto_sync_raw.strip().lower() in ("1", "true", "yes", "on")
        else:
            auto_sync = bool(_auto_sync_raw)
        receiver_requested = (
            str(request.get("receiver", "auto") or "auto").strip().lower()
        )
        if receiver_requested not in ("auto", "naive"):
            warnings.append(
                f"Unknown receiver '{request.get('receiver')}'. Using 'auto' "
                "(robust chain with honest naive fallback)."
            )
            receiver_requested = "auto"

        if suffix == ".iq":
            # Validate an explicit sample_rate before touching disk: a bad
            # rate must fail fast instead of wasting a large load + DSP.
            # A missing rate falls through to the auto_sample_rate branch
            # below, so auto_sample_rate / auto_sync behavior is unchanged.
            sample_rate_raw = request.get("sample_rate")
            if sample_rate_raw is not None:
                try:
                    _rate_probe = float(sample_rate_raw)
                except (TypeError, ValueError):
                    return _error_report(
                        file_path.name,
                        warnings,
                        f"Invalid sample_rate {sample_rate_raw!r}: must be a "
                        "number satisfying 0 < sample_rate (finite).",
                    )
                if (
                    isinstance(sample_rate_raw, bool)
                    or not np.isfinite(_rate_probe)
                    or not (_rate_probe > 0)
                ):
                    return _error_report(
                        file_path.name,
                        warnings,
                        f"Invalid sample_rate {sample_rate_raw!r}: must satisfy "
                        "0 < sample_rate (finite).",
                    )
            iq_format_requested = str(
                request.get("iq_format", "complex64") or "complex64"
            )
            iq_format = iq_format_requested
            detection: dict | None = None
            iq_confidence: float | None = None
            if iq_format.lower() == "auto":
                # Raw IQ carries no metadata: infer the on-disk dtype from the
                # statistics of the byte stream itself.
                detection = detect_iq_format(str(file_path))
                iq_format = str(detection.get("format", "complex64"))
                iq_confidence = float(detection.get("confidence", 0.0))
                if iq_confidence < 0.6:
                    warnings.append(
                        "IQ format auto-detection was inconclusive "
                        f"(guessed '{iq_format}', confidence "
                        f"{iq_confidence:.2f}). "
                        "Pass iq_format explicitly if the result looks wrong."
                    )
            # Load once: the rate estimator needs the samples, and the rate is
            # needed for DSP -- so the samples come first, the rate second.
            samples = load_iq(str(file_path), dtype=iq_format)
            file_type = "iq"

            # `sample_rate_raw` was validated before the load above; re-check
            # defensively here so a later edit cannot reintroduce an
            # unchecked float() on the DSP path.
            sample_rate_assumed = False
            sample_rate_source: str | None = "explicit"
            sample_rate_hint: dict | None = None
            if sample_rate_raw is not None:
                # Manual override wins, even when auto was also requested.
                try:
                    sample_rate = float(sample_rate_raw)
                except (TypeError, ValueError):
                    return _error_report(
                        file_path.name,
                        warnings,
                        f"Invalid sample_rate {sample_rate_raw!r}: must be a "
                        "number satisfying 0 < sample_rate (finite).",
                    )
                if (
                    isinstance(sample_rate_raw, bool)
                    or not np.isfinite(sample_rate)
                    or not (sample_rate > 0)
                ):
                    return _error_report(
                        file_path.name,
                        warnings,
                        f"Invalid sample_rate {sample_rate_raw!r}: must satisfy "
                        "0 < sample_rate (finite).",
                    )
            else:
                if not _auto_sample:
                    raise ValueError(
                        "sample_rate is required for .iq files "
                        "(raw IQ has no metadata). Re-run with e.g. "
                        "sample_rate=100000 or set auto_sample_rate=True to "
                        "estimate a working rate."
                    )
                # --- Auto sample-rate: SigMF metadata first, fused hint next.
                # Raw IQ carries no rate metadata, so an absolute rate in Hz
                # is fundamentally unknowable from the samples alone. The
                # honest order is: real metadata when it exists, else a
                # feasibility floor clearly flagged as assumed.
                sig_rate: float | None = None
                try:
                    meta = load_sigmf_meta(str(file_path))
                    if meta and meta.get("sample_rate"):
                        sig_rate = float(meta["sample_rate"])
                except Exception:
                    sig_rate = None
                if sig_rate is not None and np.isfinite(sig_rate) and sig_rate > 0:
                    sample_rate = float(sig_rate)
                    sample_rate_source = "sigmf"
                    sample_rate_assumed = False
                else:
                    provisional = float(DEFAULT_SAMPLE_RATE)
                    try:
                        freqs_p, psd_p = compute_psd(samples, provisional)
                        bw_p = float(estimate_bandwidth(freqs_p, psd_p))
                        cands = estimate_sampling_rate_candidates(
                            samples,
                            provisional,
                            bandwidth_estimate=bw_p or None,
                            symbol_rate_estimate=None,
                        )
                        fused = estimate_sampling_rate_fused(
                            samples,
                            provisional,
                            bandwidth_estimate=bw_p or None,
                            symbol_rate_hz=None,
                        )
                        fused_rate = float(fused.get("sampling_rate_hz", 0.0))
                        if fused_rate > 0 and np.isfinite(fused_rate):
                            sample_rate = fused_rate
                            sample_rate_source = "hypothesis"
                        else:
                            sample_rate = provisional
                            sample_rate_source = "hypothesis"
                        sample_rate_assumed = True
                        sample_rate_hint = {
                            "rate_hz": float(sample_rate),
                            "confidence": float(fused.get("confidence", 0.0)),
                            "candidates": list(fused.get("candidates", cands)),
                            "rationale": str(fused.get("rationale", "")),
                            "provisional_rate": provisional,
                            "bandwidth_at_provisional": bw_p,
                            "source": sample_rate_source,
                        }
                        for w in fused.get("warnings", []):
                            warnings.append(f"Auto sample-rate hint: {w}")
                        warnings.append(
                            f"sample_rate was not supplied; assumed "
                            f"{sample_rate:.0f} Hz ({sample_rate_source}, "
                            f"confidence "
                            f"{float(sample_rate_hint['confidence']):.2f}). "
                            "Raw .iq carries no rate metadata, so this is a "
                            "feasibility floor from the spectral shape, not a "
                            "measurement. All Hz-scaled estimates inherit this "
                            "assumption. Pass sample_rate explicitly if known."
                        )
                    except Exception as exc:
                        sample_rate = float(DEFAULT_SAMPLE_RATE)
                        sample_rate_source = "hypothesis"
                        sample_rate_assumed = True
                        sample_rate_hint = None
                        warnings.append(
                            "Auto sample-rate estimation failed "
                            f"({exc}); assumed {sample_rate:.0f} Hz. Pass "
                            "sample_rate explicitly if known."
                        )
        else:
            # WAV carries its own sample rate; any requested rate is ignored.
            samples, sample_rate = load_wav(str(file_path))
            sample_rate = float(sample_rate)
            file_type = "wav"
            detection = None
            iq_confidence = None
            iq_format_requested = str(request.get("iq_format", "auto") or "auto")
            # IQ dtype is meaningless for WAV; report what was asked and what
            # was (not) used so the field never ships invisible.
            iq_format = "n/a"
            sample_rate_assumed = False
            sample_rate_source = "wav-header"
            sample_rate_hint = None

        freqs, psd_db = compute_psd(samples, sample_rate)

        center_freq = float(estimate_center_frequency(freqs, psd_db))
        bandwidth = float(estimate_bandwidth(freqs, psd_db))
        snr = float(estimate_snr(psd_db))
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
            cfo_estimate = float(estimate_cfo(samples, sample_rate))
        except Exception:
            cfo_estimate = 0.0
        # Where does the signal actually end? A real recording is a burst in
        # noise, and the CRC anchor is searched near the end of the stream, so
        # this is what stops a noise tail from hiding the frame (see the decode
        # block below). Reported as a measurement in its own right, and `found`
        # is False for a continuous signal or a capture that is all burst, in
        # which case nothing downstream changes.
        burst = estimate_burst_region(samples)

        requested = str(request.get("modulation", "auto") or "auto").upper()
        if requested == "AUTO":
            estimated_type, confidence, alternatives = _classify_modulation(
                samples, sample_rate
            )
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
        elif requested in ("8PSK", "8-PSK"):
            estimated_type = "8PSK"
            confidence = 0.5
            alternatives = ["QPSK", "16-QAM"]
            mode = "8PSK"
        elif requested in ("64-QAM", "QAM64", "64QAM"):
            estimated_type = "64-QAM"
            confidence = 0.5
            alternatives = ["16-QAM", "8PSK"]
            mode = "64-QAM"
        elif requested in ("4-FSK", "4FSK"):
            estimated_type = "4-FSK"
            confidence = 0.5
            alternatives = ["2-FSK", "FM"]
            mode = "4-FSK"
        else:
            warnings.append(
                f"Unsupported modulation '{requested}'. Used BPSK fallback."
            )
            estimated_type = "BPSK"
            confidence = 0.5
            alternatives = []
            mode = "BPSK"

        # Samples per symbol actually used by the demodulator. BPSK/QPSK/16-QAM
        # symbol-spaced captures are one sample per symbol by construction; 2-FSK is not,
        # so its period is recovered from the instantaneous frequency below.
        samples_per_symbol = 1
        # Whether the estimated modulation was independently corroborated.
        # None = the check does not apply to this mode.
        corroborated: bool | None = None
        # Set when the header search replaced the classifier's own label, so the
        # report can show a revised estimate as revised.
        revised_from: str | None = None

        # Parse an explicit sync word *before* demod so the coherent path can
        # resolve Costas residual rotation against it. Auto discovery still
        # runs after demod when no sync was supplied.
        sync_requested_raw = request.get("sync_word")
        if isinstance(sync_requested_raw, str) and not sync_requested_raw.strip():
            sync_requested_raw = None
        sync_word = (
            str(sync_requested_raw).strip() if sync_requested_raw is not None else None
        )
        early_sync_bits: np.ndarray | None = None
        if sync_word:
            try:
                early_sync_bits = hex_to_bits(sync_word)
            except Exception:
                early_sync_bits = None

        digital_modes = {
            "BPSK",
            "QPSK",
            "8PSK",
            "16-QAM",
            "64-QAM",
            "2-FSK",
            "2FSK",
            "4-FSK",
            "4FSK",
        }
        if mode in digital_modes:
            # Robust receiver by default; naive slicers on honest fallback.
            # The receiver info always describes the path that produced
            # `bits`, so `display.samples_per_symbol` and the burst-to-
            # `frame_bits` conversion below stay correct on either path.
            bits, samples_per_symbol, receiver_info = _demodulate_with_info(
                samples, mode, sync_bits=early_sync_bits, receiver=receiver_requested
            )
            if bits is None:
                bits = demod_bpsk(samples)
                samples_per_symbol = 1
                receiver_info = _naive_receiver_info(
                    receiver_requested,
                    mode,
                    1,
                    int(np.asarray(bits).size),
                    "demodulator returned no bits; BPSK slicer used as a placeholder",
                )
            elif receiver_info.get("path") == "naive" and receiver_requested == "auto":
                warnings.append(
                    "Robust receiver fell back to the naive slicers "
                    f"({receiver_info.get('reason', 'no reason')}). The "
                    "reported bits come from the naive path."
                )
            if mode in ("2-FSK", "2FSK", "4-FSK", "4FSK"):
                # A recovered period is independent evidence that this really
                # is constant-envelope FSK. The classifier's FSK gate is only
                # a narrowband test (it also fires for a tone and for music).
                corroborated = samples_per_symbol > 1
                if samples_per_symbol <= 1:
                    warnings.append(
                        f"Could not recover the {mode} symbol period (the "
                        "capture may be shorter than a few symbols, or not "
                        "constant-envelope FSK). Demodulated at the "
                        "default of one bit per sample, so the bit stream is "
                        "oversampled by the unknown number of samples per "
                        "symbol."
                    )
                    if requested == "AUTO" and mode in ("2-FSK", "2FSK"):
                        # Only downgrade an *estimate*. Explicit user choice
                        # for 2-FSK is left alone; the warning carries the
                        # caveat.
                        confidence = min(
                            confidence, MODULATION_UNCORROBORATED_CONFIDENCE
                        )
                        warnings.append(
                            "The 2-FSK classification is uncorroborated, so "
                            f"its confidence is reported as {confidence:.2f} "
                            "rather than as a confident estimate."
                        )
        else:
            if mode in ANALOG_REJECT_LABELS:
                # A tone, AM/FM broadcast or audio-like capture has no bits
                # to demodulate. The BPSK read below is a placeholder so the
                # report keeps its shape; the cap plus the warning say the
                # bits are not meaningful.
                confidence = min(
                    float(confidence), MODULATION_UNCORROBORATED_CONFIDENCE
                )
                warnings.append(
                    f"The capture looks like {mode} rather than a digital "
                    "modulation, so there are no bits to demodulate. A BPSK "
                    "slicer was run as a placeholder and its output is not "
                    "meaningful; confidence is capped at "
                    f"{MODULATION_UNCORROBORATED_CONFIDENCE:.2f}."
                )
            else:
                warnings.append(f"Unsupported modulation '{mode}'. Used BPSK fallback.")
            revised_from = estimated_type
            estimated_type = "BPSK"
            mode = "BPSK"
            bits = demod_bpsk(samples)
            samples_per_symbol = 1
            receiver_info = _naive_receiver_info(
                receiver_requested,
                "BPSK",
                1,
                int(np.asarray(bits).size),
                f"placeholder BPSK read for {revised_from} (no bits to demodulate)",
            )

        # Manual sync_word always wins over discovery.
        assumed_sync_word: str | None = None
        probe_info: dict | None = None
        discovery_info: dict | None = None
        if sync_word:
            # Explicit user value: reported as asked, never flagged assumed.
            pass
        elif auto_sync and revised_from in ANALOG_REJECT_LABELS:
            # Placeholder bits (see the analog branch above) carry no
            # information: searching them for a preamble can only manufacture
            # structural false hits -- measured: a tone demodulates to a
            # periodic square wave, which the QPSK revision folding then
            # matches to 0x55555555 at score 1.0. Leave the header unset
            # rather than claim one.
            discovery_info = {"candidates_tried": [], "margin": 0.0}
            warnings.append(
                "Skipped preamble search: the demodulated bits are a "
                "placeholder for an analog-like capture, not a bit stream. "
                "Pass modulation explicitly if this is really a digital burst."
            )
        elif auto_sync:
            # Auto sync-word discovery (CFAR-controlled): scan the common
            # 32-bit preambles and keep the one that correlates
            # *significantly* (see correlator.FALSE_ALARM_TARGET). The result
            # is a framing hint, never a claim -- reported with assumed=True
            # plus a warning. A wrong modulation hides the header, so when the
            # current demodulation finds nothing and the modulation itself was
            # an estimate, the other candidates are tried before giving up
            # (the header is independent evidence, same arbiter as
            # _corroborate_modulation below).
            discovery = discover_sync_word(np.asarray(bits))
            discovery_info = {
                "candidates_tried": discovery.get("candidates_tried", []),
                "margin": float(discovery.get("margin", 0.0)),
            }
            if bool(discovery.get("detected")) and discovery.get("sync_word"):
                sync_word = str(discovery["sync_word"])
                assumed_sync_word = sync_word
                warnings.append(
                    "No sync_word was supplied; auto-discovery assumed "
                    f"{sync_word} (correlation "
                    f"{float(discovery.get('score', 0.0)):.2f} at bit "
                    f"{int(discovery.get('offset', -1))}, "
                    f"min {float(discovery.get('min_score', 0.0)):.3f} over "
                    f"{int(discovery.get('positions_searched', 0))} positions). "
                    "Pass sync_word explicitly if the capture uses a "
                    "different header."
                )
            elif requested == "AUTO":
                for candidate in HEADER_FALLBACK_MODULATIONS:
                    if candidate.upper() == str(mode).upper():
                        continue
                    try:
                        alt_bits, alt_sps, alt_info = _demodulate_with_info(
                            samples, candidate, receiver=receiver_requested
                        )
                    except Exception:
                        continue
                    if alt_bits is None or np.asarray(alt_bits).size < 32:
                        continue
                    alt_disc = discover_sync_word(np.asarray(alt_bits))
                    if bool(alt_disc.get("detected")) and alt_disc.get("sync_word"):
                        previous_type = estimated_type
                        mode = candidate
                        bits = np.asarray(alt_bits)
                        samples_per_symbol = int(alt_sps)
                        receiver_info = alt_info
                        estimated_type = mode
                        revised_from = previous_type
                        corroborated = (
                            samples_per_symbol > 1 if candidate == "2-FSK" else None
                        )
                        discovery = alt_disc
                        discovery_info = {
                            "candidates_tried": alt_disc.get("candidates_tried", []),
                            "margin": float(alt_disc.get("margin", 0.0)),
                            "revised_modulation_from": previous_type,
                        }
                        sync_word = str(alt_disc["sync_word"])
                        assumed_sync_word = sync_word
                        confidence = MODULATION_HEADER_CORROBORATED_CONFIDENCE
                        alternatives = [previous_type] + [
                            alt
                            for alt in alternatives
                            if str(alt).upper() != mode.upper()
                        ]
                        if mode == "2-FSK" and samples_per_symbol > 1:
                            pass  # symbol-rate fused block below re-derives it
                        warnings.append(
                            f"No sync_word was supplied; auto-discovery assumed "
                            f"{sync_word} under a revised {mode} demodulation "
                            f"(correlation "
                            f"{float(alt_disc.get('score', 0.0)):.2f} at bit "
                            f"{int(alt_disc.get('offset', -1))}). The "
                            f"classifier's {previous_type} label is unreliable "
                            "when a burst is surrounded by noise; the header "
                            "match is independent evidence."
                        )
                        break
                else:
                    best = discovery.get("candidates_tried", [])
                    best_score = max(
                        (float(e.get("score", 0.0)) for e in best), default=0.0
                    )
                    warnings.append(
                        "No sync_word was supplied and auto-discovery found no "
                        f"significant header among {len(best)} common "
                        f"preamble(s) (best score {best_score:.3f}). Treating "
                        "the whole bit stream as payload; pass sync_word "
                        "explicitly if the capture really does carry a header."
                    )
            else:
                best = discovery.get("candidates_tried", [])
                best_score = max(
                    (float(e.get("score", 0.0)) for e in best), default=0.0
                )
                warnings.append(
                    "No sync_word was supplied and auto-discovery found no "
                    f"significant header among {len(best)} common "
                    f"preamble(s) (best score {best_score:.3f}). Treating the "
                    "whole bit stream as payload; pass sync_word explicitly "
                    "if the capture really does carry a header."
                )
            if sync_word is None:
                # Discovery declined: keep the tried list for inspectability,
                # but no header is claimed.
                pass
        else:
            # Legacy single-probe path (auto_sync=False): probe the project
            # default only. Rather than silently starting the payload at bit 0
            # -- which mis-frames every capture that *does* carry a header --
            # use it as a framing hint only if it correlates *significantly*.
            # `detected` already accounts for the search length, so a long
            # capture cannot produce a chance match (see
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
            # Discovery winners carry their own bit pattern (Barker-free, all
            # hex, so re-parsing is exact); manual values are parsed here.
            # Manual sync_word always wins -- discovery never overrides it.
            if assumed_sync_word is not None and discovery_info is not None:
                # Re-derive bits from the winning label to keep one code path;
                # all discovery candidates are hex, so this is exact.
                sync_bits = hex_to_bits(str(sync_word))
            else:
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
                "discovery": discovery_info,
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
                "discovery": discovery_info,
            }

        # The payload starts right after a *detected* sync word; with no header
        # (or no confidence in one) the whole bit stream is treated as payload.
        if sync_bits is not None and correlation["detected"] and offset >= 0:
            payload_start = offset + int(sync_bits.size)
        else:
            payload_start = 0

        # ---- Modulation corroboration by header search --------------------
        # The classifier's label is not invariant to how much noise surrounds
        # the burst: a clean BPSK burst is labelled QPSK or 8PSK once enough
        # noise is padded either side of it, and a wrong label hides the sync
        # word. Without this step the report says "no frame found" and sends the
        # user hunting for a framing bug that is really a classification one.
        #
        # Header correlation is independent of the classifier, so it is a fair
        # arbiter. Only an *estimate* is revised: an explicit user request is
        # never overridden. The score gate keeps the retry off pure-noise
        # captures, where it could only cost time.
        if (
            requested == "AUTO"
            and sync_bits is not None
            and not correlation["detected"]
            and float(correlation["score"]) > MODULATION_RETRY_MIN_SCORE
            and revised_from not in ANALOG_REJECT_LABELS
        ):
            revision = _corroborate_modulation(
                samples, sync_bits, current=mode, receiver=receiver_requested
            )
            if revision is not None:
                previous_type = estimated_type
                mode, bits, samples_per_symbol, found, corroborated, rev_info = revision
                receiver_info = rev_info
                estimated_type = mode
                revised_from = previous_type
                offset = int(found.get("offset", -1))
                score = float(found.get("score", 0.0))
                correlation = {
                    "sync_word": sync_word,
                    "assumed": assumed_sync_word is not None,
                    "header_offset": offset,
                    "offset": offset,
                    "score": score,
                    "detected": True,
                    "sync_length": int(sync_bits.size),
                    "min_score": float(found.get("min_score", 0.0)),
                    "min_matches": int(found.get("min_matches", 0)),
                    "positions_searched": int(found.get("positions_searched", 0)),
                    "probe": probe_info,
                    "discovery": discovery_info,
                }
                payload_start = offset + int(sync_bits.size) if offset >= 0 else 0
                confidence = MODULATION_HEADER_CORROBORATED_CONFIDENCE
                alternatives = [previous_type] + [
                    alt for alt in alternatives if str(alt).upper() != mode.upper()
                ]
                warnings.append(
                    f"The {previous_type} demodulation found no header, but "
                    f"{mode} did (correlation {score:.2f} at bit {offset}), so "
                    f"the modulation estimate was revised to {mode}. The "
                    f"classifier's {previous_type} label is unreliable when a "
                    "burst is surrounded by noise; the header match is "
                    "independent evidence."
                )

        # ---- Fused symbol-rate and sampling-rate estimation ----------------
        # `bandwidth * 2.2` was the original whole answer to "what is the sampling
        # rate?", and the |x|^2 spectral line was its whole answer to "what is
        # the symbol rate?". Both are single point estimates with no way to
        # tell a measurement from a guess. The fused estimators below combine
        # several evidence paths that can each *abstain*, and report 0 Hz plus
        # a warning when nothing usable was found -- never a plausible default.
        #
        # This runs after the modulation decision because two of the paths are
        # mode-dependent: the |x|^2 line is physically meaningless for
        # constant-envelope FSK, and the recovered FSK symbol period is a
        # *direct* measurement that outranks the detectors.
        fsk_period_hint = int(samples_per_symbol) if mode == "2-FSK" else None
        # The cyclostationary profile's cost grows as nfft^2 and is independent
        # of the capture length (see rate_est.CYCLO_MAX_WORK), so the FFT size
        # is chosen from a work budget and the resolution actually achieved is
        # reported. Without this a 1 M-sample capture spent 7.3 s of its 10 s
        # budget on this one estimator.
        alpha_max_hint = 1.2 * bandwidth if bandwidth > 0.0 else 0.45 * sample_rate
        cyclo_nfft = choose_cyclo_nfft(samples.size, sample_rate, alpha_max_hint)
        try:
            symbol_rate_block = estimate_symbol_rate_fused(
                samples,
                sample_rate,
                bandwidth_estimate=bandwidth or None,
                fsk_period=fsk_period_hint,
                modulation_hint=mode,
                nfft=cyclo_nfft,
            )
        except Exception as exc:  # a failed estimator is not a failed analysis
            symbol_rate_block = {
                "symbol_rate_hz": 0.0,
                "confidence": 0.0,
                "method": "none",
                "components": {},
                "agreement": None,
                "nfft": cyclo_nfft,
                "resolution_hz": 0.0,
                "warnings": [f"Symbol-rate estimation failed: {exc}"],
            }
        symbol_rate_estimate = float(symbol_rate_block["symbol_rate_hz"])
        warnings.extend(symbol_rate_block.get("warnings", []))

        try:
            sampling_rate_block = estimate_sampling_rate_fused(
                samples,
                sample_rate,
                bandwidth_estimate=bandwidth or None,
                symbol_rate_hz=symbol_rate_estimate or None,
            )
        except Exception as exc:
            sampling_rate_block = {
                "sampling_rate_hz": 0.0,
                "confidence": 0.0,
                "candidates": [],
                "rationale": f"estimator error: {exc}",
                "warnings": [],
            }
        sample_rate_estimate = float(sampling_rate_block["sampling_rate_hz"])
        warnings.extend(sampling_rate_block.get("warnings", []))

        # ---- Blind candidate scores (honest: never a detection claim) ----
        fec_res = score_fec_candidates(bits)
        ilv_res = score_deinterleave_candidates(bits)

        # ---- Verified decode: de-interleave -> FEC -> CRC-16 ----
        # This is a *search*, not a detection: nothing is reported as decoded
        # unless an independent CRC-16 check passes (info.md §30).
        decode_enabled = bool(request.get("decode", True))
        frame_bits = request.get("frame_bits")
        # With no explicit hint, derive one from the burst region. The CRC anchor
        # is searched near the end of the stream, so a frame followed by a noise
        # tail has its anchor land on noise and the frame is never found -- the
        # most likely failure on a real recording, where the burst is surrounded
        # by noise. Telling the search where the burst ends is what fixes it.
        #
        # The hint is biased *long* (slack, plus an estimator that lands late by
        # construction). That asymmetry is deliberate: appending bits leaves every
        # earlier de-interleaver block intact, while truncating below the true
        # frame end corrupts the last block and destroys the CRC. Excess is
        # absorbed by the CRC search window, so the hint only ever removes a
        # reason to fail -- it can never manufacture a decode, because the CRC
        # still decides.
        auto_frame_bits: int | None = None
        if frame_bits is None and burst["found"] and correlation["detected"]:
            samples_per_bit = samples.size / max(bits.size, 1)
            burst_end_bits = round(int(burst["end_sample"]) / samples_per_bit)
            if burst_end_bits > payload_start:
                auto_frame_bits = (
                    burst_end_bits - payload_start + BURST_FRAME_SLACK_BITS
                )
                if auto_frame_bits < bits.size:
                    warnings.append(
                        f"The burst region ends at sample {burst['end_sample']}, "
                        f"so the decode search was bounded to {auto_frame_bits} "
                        "coded bits past the sync word instead of assuming the "
                        "capture ends at the frame. Estimated from the power "
                        "envelope; see signal.burst."
                    )
                else:
                    # The estimate covers the whole capture, so there is nothing
                    # to trim and the hint would be a no-op.
                    auto_frame_bits = None

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
                    frame_bits=(int(frame_bits) if frame_bits else auto_frame_bits),
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
            # The coded-region length the search actually used: the caller's
            # explicit hint, else the one derived from signal.burst, else None
            # (which means "assume the capture ends at the frame").
            "frame_bits": int(frame_bits) if frame_bits else auto_frame_bits,
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

        # ---- Analog-video hint (NTSC line sync, additive, evidence only) ----
        # A dedicated envelope test, not a mode claim: an analog TV capture
        # otherwise falls through to an uncorroborated low-confidence label.
        # The detector caps its own confidence and never raises; a miss costs
        # one FFT, never the analysis.
        try:
            from rf_analyzer.core.video import detect_ntsc_lines

            video_block = detect_ntsc_lines(samples, sample_rate)
        except Exception as exc:  # a hint must never fail the analysis
            video_block = {
                "found": False,
                "line_rate_hz": None,
                "confidence": 0.0,
                "harmonic_ratio": None,
                "method": "envelope-fft",
                "reason": f"detector error: {exc}",
            }
        if bool(video_block.get("found")):
            rate = video_block.get("line_rate_hz")
            rate_text = "unknown rate" if rate is None else f"~{float(rate):.0f} Hz"
            warnings.append(
                "Analog video line sync detected "
                f"({rate_text}, confidence "
                f"{float(video_block.get('confidence', 0.0)):.2f}). The "
                "capture is likely analog television, not a digital "
                f"burst -- the {mode} bits below are not meaningful and no "
                "payload is claimed."
            )

        # ---- GPS L1 C/A acquisition (opt-in, additive, evidence only) ----
        # A 10 ms capture cannot yield navigation subframes or ephemeris
        # (those need 6 s+ of continuous tracking), so this never claims a
        # message -- only acquisition evidence (SV, Doppler, code phase,
        # prompt signs, C/N0) via core.gps. Default OFF: the search costs
        # seconds on wide captures and the rest of the suite must stay fast.
        gps_requested_raw = request.get("gps", False)
        if isinstance(gps_requested_raw, str):
            gps_requested = gps_requested_raw.strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
        else:
            gps_requested = bool(gps_requested_raw)
        gps_block: dict = {
            "attempted": bool(gps_requested),
            "reason": "",
            "svs_searched": [],
            "doppler_range_hz": 5000.0,
            "doppler_step_hz": 250.0,
            "threshold": 2.5,
            "n_blocks": 0,
            "block_len": None,
            "code_resolution_chips": None,
            "acquired": [],
            "prompt_signs": [],
            "cn0_dbhz": None,
            "nav_note": (
                "A 10 ms capture cannot yield navigation subframes or "
                "ephemeris (needs 6 s+ of continuous tracking); no message "
                "is claimed -- acquisition evidence only."
            ),
            "warnings": [],
        }
        if gps_requested:
            try:
                from rf_analyzer.core import gps as gps_mod

                gps_block["threshold"] = float(gps_mod.ACQUISITION_THRESHOLD)
                results = gps_mod.acquire(samples, sample_rate)
                gps_block["svs_searched"] = sorted(int(s) for s in results)
                acquired_entries: list[dict] = []
                for sv in sorted(results):
                    entry = results[sv]
                    if not entry.get("acquired"):
                        continue
                    try:
                        spread = gps_mod.despread(
                            samples,
                            sample_rate,
                            sv,
                            float(entry["doppler_hz"]),
                            int(entry["code_phase_samples"]),
                        )
                    except Exception as exc:
                        spread = {
                            "prompt_signs": [],
                            "cn0_dbhz": None,
                            "n_blocks": int(entry.get("n_blocks", 0)),
                            "reason": f"despread failed: {exc}",
                        }
                    cn0 = spread.get("cn0_dbhz")
                    acquired_entries.append(
                        {
                            "sv": int(sv),
                            "doppler_hz": float(entry["doppler_hz"]),
                            "code_phase_samples": int(entry["code_phase_samples"]),
                            "code_phase_chips": float(entry["code_phase_chips"]),
                            "metric": float(entry["metric"]),
                            "peak_over_mean": float(entry["peak_over_mean"]),
                            "cn0_dbhz": None if cn0 is None else float(cn0),
                            "n_blocks": int(spread.get("n_blocks", 0)),
                            "prompt_signs": [
                                int(s) for s in spread.get("prompt_signs", [])
                            ],
                        }
                    )
                    if spread.get("reason"):
                        gps_block["warnings"].append(
                            f"GPS L1 SV{sv}: {spread['reason']}."
                        )
                acquired_entries.sort(key=lambda e: -e["metric"])
                gps_block["acquired"] = acquired_entries
                if acquired_entries:
                    best = acquired_entries[0]
                    gps_block["n_blocks"] = int(best["n_blocks"])
                    gps_block["prompt_signs"] = list(best["prompt_signs"])
                    gps_block["cn0_dbhz"] = best["cn0_dbhz"]
                else:
                    first = next(iter(results.values()), {})
                    gps_block["n_blocks"] = int(first.get("n_blocks", 0))
                    if first.get("block_len"):
                        gps_block["block_len"] = int(first["block_len"])
                        gps_block["code_resolution_chips"] = 1023.0 / float(
                            first["block_len"]
                        )
                if acquired_entries:
                    sample_entry = results[acquired_entries[0]["sv"]]
                    gps_block["block_len"] = int(sample_entry["block_len"])
                    gps_block["code_resolution_chips"] = 1023.0 / float(
                        sample_entry["block_len"]
                    )
                if acquired_entries:
                    found = ", ".join(
                        f"SV{e['sv']} (Doppler {e['doppler_hz']:.0f} Hz, "
                        f"code {e['code_phase_chips']:.1f} chips, "
                        f"metric {e['metric']:.2f})"
                        for e in acquired_entries
                    )
                    warnings.append(
                        f"GPS L1 acquisition: {len(acquired_entries)} SV(s) "
                        f"acquired: {found}. Evidence only -- a 10 ms capture "
                        "cannot yield a navigation message, so none is claimed."
                    )
                    for entry in acquired_entries:
                        if entry["metric"] < 1.5 * float(gps_block["threshold"]):
                            note = (
                                f"GPS L1 SV{entry['sv']} acquisition is "
                                f"marginal (metric {entry['metric']:.2f} vs "
                                f"threshold {float(gps_block['threshold']):.2f}); "
                                "treat as tentative."
                            )
                            gps_block["warnings"].append(note)
                            warnings.append(note)
                        cn0 = entry["cn0_dbhz"]
                        if cn0 is not None and (
                            cn0 < gps_mod.CN0_PLAUSIBLE_DBHZ[0]
                            or cn0 > gps_mod.CN0_PLAUSIBLE_DBHZ[1]
                        ):
                            note = (
                                f"GPS L1 SV{entry['sv']} C/N0 {cn0:.1f} dB-Hz "
                                "is outside the plausible "
                                f"{gps_mod.CN0_PLAUSIBLE_DBHZ[0]:.0f}-"
                                f"{gps_mod.CN0_PLAUSIBLE_DBHZ[1]:.0f} dB-Hz "
                                "band (likely a nav-bit transition inside the "
                                "short window, which reads the estimate low); "
                                "signs reported as measured, not trusted."
                            )
                            gps_block["warnings"].append(note)
                            warnings.append(note)
                else:
                    declined = next(iter(results.values()), {}).get("reason", "")
                    if declined:
                        note = f"GPS L1 acquisition declined: {declined}."
                    else:
                        peaked = max(
                            (float(e["metric"]) for e in results.values()),
                            default=0.0,
                        )
                        note = (
                            "GPS L1 acquisition attempted over "
                            f"{len(results)} SV(s) x Doppler bins "
                            f"(±{gps_block['doppler_range_hz']:.0f} Hz): no SV "
                            "crossed the "
                            f"threshold {float(gps_block['threshold']):.2f} "
                            f"(best metric {peaked:.2f}); declining rather "
                            "than guessing."
                        )
                    gps_block["reason"] = note
                    gps_block["warnings"].append(note)
                    warnings.append(note)
            except Exception as exc:  # a failed search is a miss, not an error
                note = f"GPS L1 acquisition skipped: {exc}"
                gps_block["reason"] = note
                gps_block["warnings"].append(note)
                warnings.append(note)
        else:
            gps_block["reason"] = (
                "GPS L1 acquisition not requested (gps=false, the default); "
                "pass gps=true to attempt it."
            )
            # Spread-spectrum hint: a wide occupied bandwidth with no measurable
            # symbol rate and a poor BPSK constellation fit is what a
            # below-noise DSSS signal looks like to the narrowband path.
            if bandwidth > 1_000_000.0 and symbol_rate_estimate <= 0.0:
                try:
                    from rf_analyzer.core.dsp import compute_evm as _evm

                    probe = _evm(
                        np.asarray(samples[:MODULATION_FIT_MAX_SAMPLES]),
                        mode="BPSK",
                        samples_per_symbol=1,
                    )
                    probe_evm = probe.get("evm_percent")
                except Exception:
                    probe_evm = None
                if probe_evm is None or float(probe_evm) > MODULATION_FIT_EVM_WARN:
                    warnings.append(
                        "possible spread-spectrum; re-run with gps=true to "
                        "attempt GPS L1 C/A acquisition "
                        f"(occupied BW {bandwidth:.0f} Hz with no measurable "
                        "symbol rate and a poor BPSK constellation fit is "
                        "what a below-noise DSSS signal looks like)."
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
                # Additive (Task 5 automation): how the working rate was chosen.
                # `sample_rate_assumed=True` means the value is a hint, never a
                # measurement -- every assumed value also carries a warning.
                "sample_rate_assumed": bool(sample_rate_assumed),
                "sample_rate_source": sample_rate_source,
                "center_frequency": request.get("center_frequency"),
                "iq_format": request.get("iq_format", "auto"),
                "iq_format_used": iq_format,
                "iq_format_confidence": iq_confidence,
                "auto_sample_rate": bool(_auto_sample),
                "auto_sync": bool(auto_sync),
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
                # Additive: the evidence behind the two rate numbers above.
                # `method` names which path produced the value
                # (direct/corroborated/...), `components` keeps every path's raw
                # result *including the ones that abstained*, and
                # `resolution_hz` is the cycle-frequency resolution the
                # estimator actually achieved -- so a coarse answer is never
                # presented as a fine one. A `method` of "none" means 0 Hz is
                # "not measurable", not a measurement of zero.
                "symbol_rate": {
                    "hz": symbol_rate_estimate,
                    "confidence": float(symbol_rate_block.get("confidence", 0.0)),
                    "method": symbol_rate_block.get("method"),
                    "components": symbol_rate_block.get("components", {}),
                    "agreement": symbol_rate_block.get("agreement"),
                    "nfft": symbol_rate_block.get("nfft"),
                    "resolution_hz": symbol_rate_block.get("resolution_hz"),
                },
                "sampling_rate": {
                    "hz": sample_rate_estimate,
                    "confidence": float(sampling_rate_block.get("confidence", 0.0)),
                    "rationale": sampling_rate_block.get("rationale"),
                    "candidates": sampling_rate_block.get("candidates", []),
                },
                # Additive (Task 5): the working-rate hint used when
                # `sample_rate` was missing and `auto_sample_rate` estimated
                # one. None when the rate was explicit or from a WAV header.
                # Documented alongside `input.sample_rate_assumed/source`.
                "sample_rate_hint": sample_rate_hint,
                # Additive: where the signal burst starts and ends, estimated
                # from the power envelope. `found` is False for a continuous
                # signal or a capture that is all burst, and then no burst
                # measurement is claimed. `end_sample` is what bounds the decode
                # search when the capture continues past the frame.
                "burst": burst,
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
                # The classifier's own label when the header search overrode it,
                # else None. Non-null means `estimated_type` is a *revised*
                # estimate, not the classifier's answer.
                "revised_from": revised_from,
            },
            "demodulation": {
                "mode": mode,
                "num_bits": len(bits),
                "bitstream_file": None,
                "bits_preview": [int(b) for b in list(bits[:2048])],
                # Additive: which demodulation path produced the bit stream.
                # `path` is "coherent" only when core.receiver ran; "naive"
                # covers requested-naive, abstain/unlocked fallback, and the
                # analog placeholder. `requested` echoes the `receiver`
                # request key. `locked` is a bool for coherent attempts and
                # None for naive slicers (no lock concept). Every value is a
                # plain bool/int/float/str/None/dict so the strict-JSON
                # serializer never sees a numpy type or tuple.
                "receiver": {
                    "path": str(receiver_info.get("path", "naive")),
                    "requested": str(
                        receiver_info.get("requested", receiver_requested)
                    ),
                    "locked": (
                        None
                        if receiver_info.get("locked") is None
                        else bool(receiver_info.get("locked"))
                    ),
                    "reason": str(receiver_info.get("reason", "")),
                    "sps": int(receiver_info.get("sps", samples_per_symbol) or 0),
                    "n_symbols": receiver_info.get("n_symbols"),
                    "recovery": dict(receiver_info.get("recovery", {})),
                    "ambiguity": dict(receiver_info.get("ambiguity", {})),
                    "evm_percent": receiver_info.get("evm_percent"),
                    "rotation": receiver_info.get("rotation"),
                },
            },
            "correlation": correlation,
            "payload": payload_block,
            # Constellation quality: EVM/MER plus the modulation order, so a
            # consumer can tell "BPSK at 12% EVM" from "BPSK at 180% EVM".
            "quality": quality_block,
            # GPS L1 C/A acquisition evidence (opt-in via gps=true; additive
            # and never a message claim -- see the block built above).
            "gps": gps_block,
            # Analog-video hint (NTSC line sync; additive and never a mode
            # claim -- see the block built above).
            "video": video_block,
            # Candidate scores are blind heuristics (capped, crc_pass unknown);
            # a CRC-verified decode is reported separately and only when it
            # actually happened. See info.md §30 demo-defense lines.
            "fec": fec_block,
            "interleaving": ilv_block,
            # Hint for eye diagram: current synthetic generator uses
            # 1 sample/symbol for BPSK/QPSK/2-FSK.
            "display": {
                "samples_per_symbol": samples_per_symbol,
                "mode": mode,
            },
            # Additive TypeSafe-pattern judgments (Choice/Noul/Score over the
            # same state, composed in code; see core/judgments.py). Never
            # replaces §13 fields; low confidence escalates via warnings.
            "judgments": _judgment_block(
                samples,
                bits,
                sample_rate,
                sync_word if isinstance(sync_word, str) else None,
                quality_block,
                validated,
            ),
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
