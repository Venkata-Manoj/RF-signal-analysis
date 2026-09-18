"""Batch-analyse a folder of .iq/.wav captures and export CSV + HTML summaries.

Usage::

    python scripts/batch_analyze.py sample_data
    python scripts/batch_analyze.py captures/ --iq-format auto --no-decode
    python scripts/batch_analyze.py captures/ --csv out.csv --html out.html

Exit codes: 0 on success (even when some captures fail to load — those are
reported in the summary), 2 when the folder does not exist or holds no
captures.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rf_analyzer.batch import (
    analyze_folder,
    discover_captures,
    write_csv,
    write_html,
)

IQ_FORMAT_CHOICES = ("complex64", "int16", "uint8", "int8", "auto")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-analyse .iq/.wav captures and export CSV + HTML.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("folder", help="folder holding .iq / .wav captures")
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=100_000.0,
        help="sample rate applied to .iq files (.wav uses its own)",
    )
    parser.add_argument(
        "--iq-format",
        choices=IQ_FORMAT_CHOICES,
        default="complex64",
        help="raw .iq sample format",
    )
    parser.add_argument("--modulation", default="auto", help="auto or a fixed mode")
    parser.add_argument(
        "--sync-word", default="0x1ACFFC1D", help="hex sync word ('' to skip)"
    )
    parser.add_argument(
        "--no-decode",
        action="store_true",
        help="skip the CRC-verified FEC/de-interleaver search (faster)",
    )
    parser.add_argument("--recursive", action="store_true", help="walk sub-folders")
    parser.add_argument("--csv", default=None, help="CSV output path")
    parser.add_argument("--html", default=None, help="HTML output path")
    parser.add_argument(
        "--quiet", action="store_true", help="only print the final summary line"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"ERROR: not a folder: {folder}", file=sys.stderr)
        return 2

    captures = discover_captures(folder, recursive=args.recursive)
    if not captures:
        print(f"ERROR: no .iq/.wav captures found in {folder}", file=sys.stderr)
        return 2

    if not args.quiet:
        print(f"Analysing {len(captures)} capture(s) in {folder} …")

    def progress(path: Path, index: int, total: int) -> None:
        if not args.quiet:
            print(f"  [{index}/{total}] {path.name}")

    rows = analyze_folder(
        folder,
        sample_rate=args.sample_rate,
        iq_format=args.iq_format,
        modulation=args.modulation,
        sync_word=args.sync_word,
        decode=not args.no_decode,
        recursive=args.recursive,
        progress=progress,
    )

    header = f"{'file':<26}{'mod':<8}{'bits':>8}  {'fec':<34}{'verified':<9}message"
    print()
    print(header)
    print("-" * len(header))
    for row in rows:
        message = str(row.get("decoded_text", ""))[:40]
        verified = "yes" if row.get("decode_validated") == "yes" else "-"
        print(
            f"{row.get('file', '')!s:<26}"
            f"{row.get('modulation', '')!s:<8}"
            f"{row.get('num_bits', '')!s:>8}  "
            f"{str(row.get('fec_candidate', ''))[:33]:<34}"
            f"{verified:<9}{message}"
        )

    verified = sum(1 for r in rows if r.get("decode_validated") == "yes")
    failed = sum(1 for r in rows if r.get("errors"))
    print()
    print(
        f"{len(rows)} capture(s): {verified} CRC-verified payload(s), "
        f"{failed} load failure(s)."
    )

    out_dir = REPO_ROOT / "output"
    csv_path = Path(args.csv) if args.csv else out_dir / "batch_summary.csv"
    html_path = Path(args.html) if args.html else out_dir / "batch_summary.html"
    try:
        write_csv(rows, csv_path)
        print(f"CSV  -> {csv_path}")
    except Exception as exc:  # a failed export must not hide the analysis
        print(f"WARN: CSV export failed: {exc}", file=sys.stderr)
    try:
        write_html(rows, html_path, subtitle=f"Folder: {folder}")
        print(f"HTML -> {html_path}")
    except Exception as exc:
        print(f"WARN: HTML export failed: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
