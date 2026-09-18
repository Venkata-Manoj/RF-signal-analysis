#!/usr/bin/env python
"""Headless pipeline run (no GUI).

The file may be given positionally or with ``--file``::

    python scripts/run_pipeline.py real_data/gps_l1_4mhz_cf32.iq \
        --sample-rate 4000000 --iq-format auto
    python scripts/run_pipeline.py --file sample_data/bpsk.iq --sample-rate 100000 \
        --iq-format complex64 --modulation BPSK --sync-word 0x1ACFFC1D

Exit code is 0 on success and 1 when the report carries errors, so the script
is usable from a shell pipeline or CI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rf_analyzer.core.report import save_report
from rf_analyzer.pipeline import analyze_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the RF analysis pipeline headlessly.",
        epilog="Sample rate is required for .iq (raw IQ has no metadata) and "
        "ignored for .wav (the file carries its own).",
    )
    parser.add_argument(
        "file_path",
        nargs="?",
        help="Path to a .iq or .wav file (may also be given as --file).",
    )
    parser.add_argument(
        "--file", dest="file_opt", default=None, help="Path to the capture."
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=None,
        help="Sample rate in Hz (required for .iq; ignored for .wav).",
    )
    parser.add_argument(
        "--iq-format",
        default="auto",
        help="IQ dtype for .iq files: auto | complex64 | int16 | uint8 | int8.",
    )
    parser.add_argument(
        "--modulation", default="auto", help="auto | BPSK | QPSK | 2-FSK | 16-QAM."
    )
    parser.add_argument(
        "--sync-word", default=None, help="Hex sync word, e.g. 0x1ACFFC1D."
    )
    parser.add_argument(
        "--no-decode",
        action="store_true",
        help="Skip the CRC-verified FEC/interleaver search.",
    )
    parser.add_argument(
        "--frame-bits",
        type=int,
        default=None,
        help="Coded-region length in bits, when known (default: auto).",
    )
    parser.add_argument(
        "--out", default="output/report.json", help="Where to save the JSON report."
    )
    return parser


def _print_report(report: dict, out_path: Path) -> None:
    inp = report.get("input", {})
    signal = report.get("signal", {})
    modulation = report.get("modulation", {})
    demodulation = report.get("demodulation", {})
    correlation = report.get("correlation", {})
    quality = report.get("quality", {})
    payload = report.get("payload", {})
    decoded = payload.get("decoded", {})

    print(f"File: {inp.get('file_name')} ({inp.get('file_type')})")
    if inp.get("iq_format_used"):
        print(
            f"IQ format: {inp.get('iq_format_used')} (requested '{inp.get('iq_format')}')"
        )
    rate = inp.get("sample_rate")
    duration = signal.get("duration_seconds") or 0.0
    print(f"Samples: {signal.get('num_samples')} ({duration:.4f} s @ {rate} Hz)")
    print(f"Center freq estimate: {signal.get('center_frequency_estimate')} Hz")
    print(f"Bandwidth estimate: {signal.get('bandwidth_estimate')} Hz")
    print(f"SNR estimate: {signal.get('snr_db')} dB")
    print(f"Symbol rate estimate: {signal.get('symbol_rate_estimate')} Hz")
    print(f"CFO estimate: {signal.get('cfo_estimate_hz')} Hz")
    print(
        f"Modulation: {modulation.get('estimated_type')} "
        f"(confidence {modulation.get('confidence')})"
    )
    if quality.get("applicable"):
        mer = quality.get("mer_db")
        mer_text = "unbounded" if mer is None else f"{mer:.2f} dB"
        fit = "ok" if quality.get("fit_ok") else "POOR"
        print(
            f"Constellation: EVM {quality.get('evm_percent'):.2f}%, MER {mer_text} "
            f"({quality.get('n_symbols')} symbols, fit {fit})"
        )
    print(
        f"Demod: mode={demodulation.get('mode')} num_bits={demodulation.get('num_bits')}"
    )
    print(
        f"Correlation: sync_word={correlation.get('sync_word')}"
        f"{' (assumed)' if correlation.get('assumed') else ''} "
        f"header_offset={correlation.get('header_offset')} "
        f"score={correlation.get('score')}"
    )
    print(
        f"FEC: {report.get('fec', {}).get('candidate')} "
        f"(validated={report.get('fec', {}).get('validated')})"
    )
    print(
        f"Interleaving: {report.get('interleaving', {}).get('candidate')} "
        f"(validated={report.get('interleaving', {}).get('validated')})"
    )

    if decoded.get("available"):
        print("\n--- DECODED PAYLOAD (CRC-16 verified) ---")
        print(
            f"scheme: {decoded.get('scheme')}  interleaver: {decoded.get('interleaver')}"
        )
        print(f"{decoded.get('bytes')} bytes")
        if decoded.get("text"):
            print(decoded["text"])
        for line in decoded.get("hexdump", []):
            print(line)
    else:
        print(
            f"\nPayload: {payload.get('payload_bits')} bits, NOT decoded "
            f"({(payload.get('raw') or {}).get('bytes')} raw bytes)"
        )

    warnings = report.get("warnings", [])
    if warnings:
        print()
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in report.get("errors", []):
        print(f"ERROR: {error}")
    print(f"\nReport saved to {out_path}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    file_path = args.file_opt or args.file_path
    if not file_path:
        build_parser().error("no capture given: pass a path or --file <path>")

    request: dict = {
        "file_path": file_path,
        "iq_format": args.iq_format,
        "modulation": args.modulation,
        "decode": not args.no_decode,
    }
    if args.sample_rate is not None:
        request["sample_rate"] = args.sample_rate
    if args.sync_word:
        request["sync_word"] = args.sync_word
    if args.frame_bits:
        request["frame_bits"] = args.frame_bits

    report = analyze_file(request)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_report(report, str(out_path))
    _print_report(report, out_path)

    return 1 if report.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
