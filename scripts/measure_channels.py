"""Channel-robustness proof: BER ranges under real impairments.

Runs a synthetic sweep through the coherent ``receiver`` chain and writes
``output/channel_robustness.json`` plus a console table. Every BER figure is
a **min-max range with its mean across seeds, never a single number**; timing
is a median with its min-max range.

Two grids, one script:

* **Quick (default)** -- 4 modulations x 4 impairments x 2 nominal Es/N0
  levels x 3 seeds (32 cells). Fast enough for CI iteration; writes only the
  ``--json`` path.
* **Full (``--full``)** -- 7 schemes (BPSK, QPSK, 8PSK, 16-QAM, 64-QAM, 2-FSK,
  4-FSK) x 5 SNRs (0, 5, 10, 15, 20 dB) x 6 impairments (clean + 4 singletons
  + 1 combined cell stacking CFO 1500 Hz + Doppler 8 kHz/s + 3-tap multipath
  + IQ 0.1/5deg) x 3 seeds = 210 cells. Also mirrors the JSON to
  ``docs/evidence/channel_robustness.json`` as the committed copy.
* **Burst summary (``--burst-summary``)** -- reuses ``eval_ota``'s burst grid
  (4 modulations x 5 noise levels x 7 paddings x 3 seeds = 420 cases) through
  the shipped ``pipeline.analyze_file`` and writes
  ``docs/evidence/burst_420_summary.json``: exact/declined/wrong counts with
  ``budget_exhausted`` flags. A decode only ever counts behind a CRC pass;
  blind scores are never claimed.

This is the quick proof script. ``scripts/eval_ota.py`` is the full harness:
the same synthetic matrix with per-process isolation, the 420-case
burst-in-noise check, and the ``real_data/`` OTA captures. Both share the
scoring helper and the nominal-Es/N0 convention (see
``eval_ota.SPS_GAIN_DB``), so numbers here reproduce there.

Usage::

    python scripts/measure_channels.py
    python scripts/measure_channels.py --full
    python scripts/measure_channels.py --burst-summary
    python scripts/measure_channels.py --full --burst-summary
    python scripts/measure_channels.py --json output/other.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_ota

MODS = ("BPSK", "QPSK", "16-QAM", "2-FSK")
IMPAIRMENTS = ("clean", "cfo-1.5kHz", "multipath-3tap", "iq-g0.2-p5deg")
SNRS_DB = (5.0, 15.0)

#: Full grid: everything ``eval_ota`` sweeps (7 schemes x 5 SNRs x 6
#: impairments, clean + 4 singletons + the combined cell).
FULL_MODS = eval_ota.SYNTH_MODS
FULL_SNRS_DB = eval_ota.SYNTH_SNRS_DB
FULL_IMPAIRMENTS = tuple(eval_ota.SYNTH_IMPAIRMENTS)

DEFAULT_JSON = ROOT / "output" / "channel_robustness.json"
EVIDENCE_CHANNEL_JSON = ROOT / "docs" / "evidence" / "channel_robustness.json"
EVIDENCE_BURST_JSON = ROOT / "docs" / "evidence" / "burst_420_summary.json"


def sweep(full: bool = False) -> tuple[list[dict], dict]:
    """Run the grid in this process; returns ``(rows, grid)``.

    ``full=False`` keeps the quick 4x4x2 grid for CI speed; ``full=True`` runs
    the 7x5x6 matrix. ``grid`` records which axes were swept.
    """
    mods = FULL_MODS if full else MODS
    impairments = FULL_IMPAIRMENTS if full else IMPAIRMENTS
    snrs = FULL_SNRS_DB if full else SNRS_DB
    grid = {
        "mode": "full" if full else "quick",
        "modulations": list(mods),
        "impairments": list(impairments),
        "snrs_db": list(snrs),
    }
    rows = []
    total = len(mods) * len(impairments) * len(snrs)
    done = 0
    for mod in mods:
        for impairment in impairments:
            for snr in snrs:
                case = {
                    "kind": "synth",
                    "mod": mod,
                    "snr_db": snr,
                    "impairment": impairment,
                }
                row = eval_ota._run_synth(case)
                rows.append(row)
                done += 1
                lo, hi, mean = row["ber_min"], row["ber_max"], row.get("ber_mean")
                span = f"{lo}-{hi} (mean {mean})" if lo is not None else "abstained"
                print(
                    f"  [{done:>3}/{total}] {mod:<8} {snr:>4.0f}dB {impairment:<24} BER {span}"
                )
    return rows, grid


def summarise(rows: list[dict], grid: dict) -> dict:
    mods = grid["modulations"]
    per_mod = {}
    for mod in mods:
        vals = [
            v
            for r in rows
            if r["mod"] == mod
            for v in (r["ber_min"], r["ber_max"])
            if v is not None
        ]
        means = [
            r["ber_mean"]
            for r in rows
            if r["mod"] == mod and r.get("ber_mean") is not None
        ]
        per_mod[mod] = {
            "ber_min": min(vals) if vals else None,
            "ber_max": max(vals) if vals else None,
            "ber_mean": (sum(means) / len(means)) if means else None,
            "n_cells": sum(1 for r in rows if r["mod"] == mod),
            "n_abstained": sum(r["n_abstained"] for r in rows if r["mod"] == mod),
        }
    timings = [t for r in rows for t in (r["t_min_s"], r["t_median_s"], r["t_max_s"])]
    return {
        "method": {
            "snr_axis": "nominal per-symbol Es/N0 (eval_ota.SPS_GAIN_DB compensated)",
            "seeds_per_cell": list(eval_ota.SYNTH_SEEDS),
            "bits_per_capture": eval_ota.N_BITS,
            "sps": eval_ota.SPS,
            "sample_rate_hz": eval_ota.SAMPLE_RATE,
            "ber_figure": "min-max range with mean across seeds, never a single draw",
            "ambiguity": "Costas rotation + equaliser group delay resolved against the reference (eval_ota._ber_best_rotation)",
            "impairment_kwargs": {
                name: {
                    key: (list(value) if isinstance(value, (list, tuple)) else value)
                    for key, value in eval_ota.SYNTH_IMPAIRMENTS[name].items()
                }
                for name in grid["impairments"]
            },
            "honesty": "BER only; no decode is claimed here and no blind FEC/interleaver score is reported.",
        },
        "grid": grid,
        "n_cells": len(rows),
        "cells": rows,
        "per_modulation_ber_span": per_mod,
        "timing_s": {
            "median": round(statistics.median(timings), 3),
            "min": round(min(timings), 3),
            "max": round(max(timings), 3),
        },
    }


def _fmt_span(row: dict) -> str:
    lo, hi, mean = row["ber_min"], row["ber_max"], row.get("ber_mean")
    if lo is None or hi is None:
        return "abstained"
    if mean is None:
        return f"{lo:.4f}-{hi:.4f}"
    return f"{lo:.4f}-{hi:.4f} (mean {mean:.4f})"


def print_table(rows: list[dict], summary: dict, grid: dict) -> None:
    print("\n=== Channel robustness: BER range with mean (min-max across 3 seeds) ===")
    print("SNR axis is nominal per-symbol Es/N0.")
    for snr in grid["snrs_db"]:
        print(f"\n--- Es/N0 {snr:.0f} dB ---")
        print(f"{'mod':<8} {'impairment':<24} {'BER min-max (mean)':<34} {'locked':>6}")
        for mod in grid["modulations"]:
            for impairment in grid["impairments"]:
                row = next(
                    r
                    for r in rows
                    if r["mod"] == mod
                    and r["impairment"] == impairment
                    and r["snr_db"] == snr
                )
                print(
                    f"{mod:<8} {impairment:<24} {_fmt_span(row):<34} {row['n_locked']:>3}/3"
                )
    print("\n--- Overall BER span per modulation (min-max across impairment x SNR) ---")
    for mod in grid["modulations"]:
        span = summary["per_modulation_ber_span"][mod]
        lo, hi, mean = span["ber_min"], span["ber_max"], span["ber_mean"]
        detail = (
            f"{lo:.4f}-{hi:.4f} (mean {mean:.4f})" if lo is not None else "abstained"
        )
        print(f"  {mod:<8} BER span {detail}   abstentions: {span['n_abstained']}")
    timing = summary["timing_s"]
    print(
        f"\nPer-cell receiver time: median {timing['median']}s range {timing['min']}-{timing['max']}s (in-process)"
    )


def sweep_burst() -> list[dict]:
    """Run ``eval_ota``'s 420-case burst grid in this process.

    One row per case; a case that raises records ``outcome: error`` with the
    exception text, so a crash can never read as a quiet decline.
    """
    cases = eval_ota._burst_cases()
    rows: list[dict] = []
    total = len(cases)
    for done, case in enumerate(cases, start=1):
        try:
            row = eval_ota._run_burst(case)
        except Exception as exc:  # a failed case is an error, never a skip
            row = {
                "kind": "burst",
                "mod": case["mod"],
                "noise_amp": case["noise_amp"],
                "pad": case["pad"],
                "seed": case["seed"],
                "outcome": "error",
                "decoded_available": False,
                "crc_pass": None,
                "budget_exhausted": False,
                "error": f"{type(exc).__name__}: {exc}",
                "seconds": 0.0,
            }
        rows.append(row)
        if done % 60 == 0 or done == total:
            tally = {
                k: sum(1 for r in rows if r["outcome"] == k)
                for k in ("exact", "declined", "WRONG", "error")
            }
            print(
                f"  [{done:>3}/{total}] exact={tally['exact']} declined={tally['declined']} WRONG={tally['WRONG']} error={tally['error']}"
            )
    return rows


def summarise_burst(rows: list[dict]) -> dict:
    per_mod = {}
    for mod in eval_ota.BURST_MODS:
        sub = [r for r in rows if r["mod"] == mod]
        per_mod[mod] = {
            "cases": len(sub),
            "exact": sum(1 for r in sub if r["outcome"] == "exact"),
            "declined": sum(1 for r in sub if r["outcome"] == "declined"),
            "wrong": sum(1 for r in sub if r["outcome"] == "WRONG"),
            "error": sum(1 for r in sub if r["outcome"] == "error"),
            "budget_exhausted": sum(1 for r in sub if r.get("budget_exhausted")),
        }
    totals = {
        "cases": len(rows),
        "exact": sum(1 for r in rows if r["outcome"] == "exact"),
        "declined": sum(1 for r in rows if r["outcome"] == "declined"),
        "wrong": sum(1 for r in rows if r["outcome"] in ("WRONG",) or r is None),
        "error": sum(1 for r in rows if r["outcome"] == "error"),
        "budget_exhausted": sum(1 for r in rows if r.get("budget_exhausted")),
    }
    timings = [r["seconds"] for r in rows if isinstance(r.get("seconds"), (int, float))]
    return {
        "method": {
            "grid": {
                "modulations": list(eval_ota.BURST_MODS),
                "noise_amplitudes": list(eval_ota.BURST_NOISE_AMPS),
                "paddings": list(eval_ota.BURST_PADS),
                "seeds": list(eval_ota.BURST_SEEDS),
            },
            "message": eval_ota.BURST_MESSAGE.decode("ascii"),
            "frame": "conv FEC + block interleaver behind sync 0x1ACFFC1D (eval_ota.build_frame/modulate)",
            "decode_budget": eval_ota.BURST_BUDGET,
            "path": "shipped pipeline.analyze_file with modulation=auto",
            "honesty": (
                "exact counts only CRC-verified decodes that equal the transmitted "
                "message byte for byte; anything else CRC-valid but different is "
                "counted wrong, and blind FEC/interleaver scores are never claimed."
            ),
        },
        "totals": totals,
        "per_modulation": per_mod,
        "timing_s": {
            "median": round(statistics.median(timings), 3) if timings else None,
            "min": round(min(timings), 3) if timings else None,
            "max": round(max(timings), 3) if timings else None,
        },
        "cases": rows,
    }


def print_burst_summary(summary: dict) -> None:
    totals = summary["totals"]
    print(
        "\n=== Burst-in-noise summary: 4 mods x 5 noise levels x 7 paddings x 3 seeds ==="
    )
    print(
        f"  cases: {totals['cases']}   exact: {totals['exact']}   declined: {totals['declined']} "
        f"  WRONG: {totals['wrong']}   error: {totals['error']}   budget_exhausted: {totals['budget_exhausted']}"
    )
    for mod, sub in summary["per_modulation"].items():
        print(
            f"    {mod:<8} exact {sub['exact']:>3}/{sub['cases']}   declined {sub['declined']:>3} "
            f"  WRONG {sub['wrong']}   budget {sub['budget_exhausted']}"
        )
    if totals["wrong"] or totals["error"]:
        print(
            "  ATTENTION: cases above reported a payload that is not the message (or errored) -- see rows."
        )
        for r in summary["cases"]:
            if r["outcome"] in ("WRONG", "error"):
                print(
                    f"    {r['outcome']}: mod={r['mod']} noise={r['noise_amp']} pad={r['pad']} seed={r['seed']}"
                )
    else:
        print(
            "  0 wrong payloads -- every claim is exactly the message."
            + (
                ""
                if totals["budget_exhausted"] == 0
                else " (some cases hit the search budget and declined)"
            )
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else "channel proof"
    )
    parser.add_argument(
        "--json", default=str(DEFAULT_JSON), help="where to write the proof JSON"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="run the full 7-scheme x 5-SNR x 6-impairment matrix (default: quick 4x4x2)",
    )
    parser.add_argument(
        "--evidence-copy",
        default=str(EVIDENCE_CHANNEL_JSON),
        help="committed copy of the channel JSON (written only with --full)",
    )
    parser.add_argument(
        "--burst-summary",
        action="store_true",
        help="also run the 420-case burst grid and write the burst summary JSON",
    )
    parser.add_argument(
        "--burst-only",
        action="store_true",
        help="run only the burst grid (skip the channel sweep)",
    )
    parser.add_argument(
        "--burst-json",
        default=str(EVIDENCE_BURST_JSON),
        help="where to write the burst summary JSON",
    )
    args = parser.parse_args(argv)

    if not args.burst_only:
        started = time.perf_counter()
        grid_label = "7x5x6 (210 cells)" if args.full else "4x4x2 (32 cells)"
        print(f"Channel robustness sweep ({grid_label}) x 3 seeds")
        rows, grid = sweep(full=args.full)
        summary = summarise(rows, grid)
        summary["wall_clock_s"] = round(time.perf_counter() - started, 1)
        print_table(rows, summary, grid)

        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"\nProof written to {out}")
        if args.full:
            evidence = Path(args.evidence_copy)
            evidence.parent.mkdir(parents=True, exist_ok=True)
            evidence.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            print(f"Committed evidence copy written to {evidence}")

    if args.burst_summary:
        started = time.perf_counter()
        print(
            "\nBurst-in-noise summary: 4 mods x 5 noise levels x 7 paddings x 3 seeds (420 cases)"
        )
        rows = sweep_burst()
        summary = summarise_burst(rows)
        summary["wall_clock_s"] = round(time.perf_counter() - started, 1)
        print_burst_summary(summary)

        out = Path(args.burst_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"\nBurst summary written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
