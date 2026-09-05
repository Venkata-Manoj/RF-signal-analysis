"""Headless pipeline run (no GUI).

Example (PowerShell, from the repo root)::

    python scripts/run_pipeline.py --file sample_data/bpsk.iq --sample-rate 100000 ^
        --iq-format complex64 --modulation BPSK --sync-word 0x1ACFFC1D
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
        description="Run the RF analysis pipeline headlessly."
    )
    parser.add_argument("--file", required=True, help="Path to a .iq or .wav file.")
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=None,
        help="Sample rate in Hz (required for .iq; ignored for .wav).",
    )
    parser.add_argument(
        "--iq-format", default="complex64", help="IQ dtype for .iq files."
    )
    parser.add_argument(
        "--modulation", default="auto", help="auto | BPSK | QPSK | 2-FSK | FSK."
    )
    parser.add_argument(
        "--sync-word", default=None, help="Hex sync word, e.g. 0x1ACFFC1D."
    )
    parser.add_argument(
        "--out", default="output/report.json", help="Where to save the JSON report."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    request = {
        "file_path": args.file,
        "sample_rate": args.sample_rate,
        "iq_format": args.iq_format,
        "modulation": args.modulation,
        "sync_word": args.sync_word,
    }
    report = analyze_file(request)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_report(report, str(out_path))

    signal = report.get("signal", {})
    modulation = report.get("modulation", {})
    demodulation = report.get("demodulation", {})
    correlation = report.get("correlation", {})
    print(
        f"File: {report.get('input', {}).get('file_name')} ({report.get('input', {}).get('file_type')})"
    )
    print(
        f"Samples: {signal.get('num_samples')} "
        f"({signal.get('duration_seconds', 0.0):.4f} s @ {report.get('input', {}).get('sample_rate')} Hz)"
    )
    print(f"Center freq estimate: {signal.get('center_frequency_estimate')} Hz")
    print(f"Bandwidth estimate: {signal.get('bandwidth_estimate')} Hz")
    print(f"SNR estimate: {signal.get('snr_db')} dB")
    print(
        f"Modulation: {modulation.get('estimated_type')} "
        f"(confidence {modulation.get('confidence')})"
    )
    print(
        f"Demod: mode={demodulation.get('mode')} num_bits={demodulation.get('num_bits')}"
    )
    print(
        f"Correlation: sync_word={correlation.get('sync_word')} "
        f"header_offset={correlation.get('header_offset')} score={correlation.get('score')}"
    )
    for warning in report.get("warnings", []):
        print(f"WARNING: {warning}")
    for error in report.get("errors", []):
        print(f"ERROR: {error}")
    print(f"Report saved to {out_path}")

    return 1 if report.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
