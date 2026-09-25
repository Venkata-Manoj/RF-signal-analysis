#!/usr/bin/env python
"""Robustness gate: receiver BER under combined CFO + Doppler + multipath + IQ.

Matrix: every shipped modulation x 2 channel conditions x 3 seeds, through the
coherent chain (``channel.apply_channel`` -> ``receiver.demodulate`` with the
Costas ambiguity resolved against the known reference, exactly like
``tests/integration/test_robust_demod.py:18-52``):

* ``clean`` -- noiseless ``waveform.synth``, no channel. Bar: BER < 0.01.
* ``combined-10dB`` -- ``waveform.synth(ebn0_db=10.0)`` plus the combined
  impairment ``cfo_hz=1500``, ``doppler_rate_hz_s=8000``,
  ``multipath=[(0, 1.0), (1, 0.3), (2, 0.15)]``, ``iq_gain_imb=0.1``,
  ``iq_phase_imb_deg=5.0`` at ``sample_rate=1 MHz``. Bar: BER < 0.05.

``sps=8``, ``N_BITS=3000`` (divisible by every constellation width, so the BER
is exact with no tail padding), seeds ``(3, 5, 7)`` -- seed 3 is the clean
acceptance seed and seed 5 the combined-impairment seed from the test suite.

The bar applies to the **worst seed** in each cell, not the mean: a gate that
passes on average but fails one draw in three is not a gate.

Known gap (``docs/acceptance_status.md:32``): 16-QAM / 64-QAM / 2-FSK are
documented to exceed 0.05 under a combined CFO + multipath + IQ channel. When
one of those schemes fails the ``combined-10dB`` bar it is reported as
``KNOWN_GAP`` with a warning -- an honest exception, never a fake pass -- and
does not fail the gate. Every other failure is ``FAIL`` and exits non-zero.
``--strict`` turns even listed exceptions into failures.

The naive ``core/demod`` slicers run alongside as an unresolved-carrier
diagnostic column (``naive_ber``); they are never gated, they only show what
the coherent chain buys under CFO.

Usage::

    python scripts/verify_robustness.py --quick   # BPSK/QPSK only, for CI
    python scripts/verify_robustness.py --full    # all 7 schemes (default)
    python scripts/verify_robustness.py --strict  # known gaps fail too
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from rf_analyzer.core import channel as ch
from rf_analyzer.core import demod as naive
from rf_analyzer.core import receiver as rx
from rf_analyzer.core import waveform as wf
from rf_analyzer.core.report import json_safe

SCHEMES_ALL = ("BPSK", "QPSK", "8PSK", "16-QAM", "64-QAM", "2-FSK", "4-FSK")
SCHEMES_QUICK = ("BPSK", "QPSK")

SPS = 8
N_BITS = 3000
SEEDS = (3, 5, 7)
SAMPLE_RATE = 1_000_000.0

CFO_HZ = 1500.0
DOPPLER_RATE_HZ_S = 8000.0
MULTIPATH = [(0, 1.0), (1, 0.3), (2, 0.15)]
IQ_GAIN_IMB = 0.1
IQ_PHASE_IMB_DEG = 5.0

BAR_CLEAN = 0.01
BAR_COMBINED = 0.05

#: Schemes documented (docs/acceptance_status.md:32) to exceed 0.05 under a
#: combined CFO + multipath + IQ channel. Allowed a KNOWN_GAP exception on the
#: combined cell only -- the clean cell has no impairment, so no exception
#: applies there.
KNOWN_GAP_COMBINED = ("16-QAM", "64-QAM", "2-FSK")

DEFAULT_JSON = ROOT / "docs" / "evidence" / "robustness_gate.json"


def _naive_ber(samples: np.ndarray, tx_bits: np.ndarray, scheme: str) -> float | None:
    """Baseline BER from the phase-aligned ``core/demod`` slicers.

    Linear schemes are decimated to symbol centres (symbol ``k`` sits at sample
    ``k * sps`` out of ``waveform.synth``); FSK slicers take the oversampled
    stream with ``samples_per_symbol``. No carrier correction, no ambiguity
    handling beyond the harness scorer -- purely diagnostic, never gated.
    Returns ``None`` instead of raising: a diagnostic must not break the gate.
    """
    try:
        name = wf.canonical_scheme(scheme)
        x = np.asarray(samples)
        if name in wf.FSK_SCHEMES:
            bits = (
                naive.demod_2fsk(x, samples_per_symbol=SPS)
                if name == "2-FSK"
                else naive.demod_4fsk(x, samples_per_symbol=SPS)
            )
        else:
            symbols = x[0::SPS]
            if name == "BPSK":
                bits = naive.demod_bpsk(symbols)
            elif name == "QPSK":
                bits = naive.demod_qpsk(symbols)
            elif name == "8PSK":
                bits = naive.demod_8psk(symbols)
            elif name == "16-QAM":
                bits = naive.demod_qam16(symbols)
            else:
                bits = naive.demod_64qam(symbols)
        if np.asarray(bits).size == 0:
            return None
        ber = rx.ber_best_rotation(tx_bits, np.asarray(bits, dtype=np.uint8), name)
        return None if ber is None else float(ber)
    except Exception:
        return None


def _run_seed(scheme: str, condition: str, seed: int) -> dict:
    """One (scheme, condition, seed) trial; returns BERs plus receiver status."""
    if condition == "clean":
        samples, tx = wf.synth(scheme, N_BITS, sps=SPS, seed=seed)
        stressed = samples
    else:
        samples, tx = wf.synth(scheme, N_BITS, sps=SPS, seed=seed, ebn0_db=10.0)
        stressed = ch.apply_channel(
            samples,
            sample_rate=SAMPLE_RATE,
            cfo_hz=CFO_HZ,
            doppler_rate_hz_s=DOPPLER_RATE_HZ_S,
            multipath=list(MULTIPATH),
            iq_gain_imb=IQ_GAIN_IMB,
            iq_phase_imb_deg=IQ_PHASE_IMB_DEG,
            seed=seed,
        )
    out = rx.demodulate(stressed, scheme, sps=SPS, seed=seed)
    rx_bits = np.asarray(out["bits"], dtype=np.uint8)
    ber = rx.ber_best_rotation(tx, rx_bits, scheme)
    return {
        "seed": int(seed),
        "ber": None if ber is None else float(ber),
        "naive_ber": _naive_ber(stressed, tx, scheme),
        "locked": bool(out.get("locked", False)),
        "path": str(out.get("path", "")),
    }


def run_gate(schemes: tuple[str, ...], strict: bool = False) -> dict:
    """Run the full matrix; returns the JSON-serialisable summary dict."""
    conditions = (("clean", BAR_CLEAN), ("combined-10dB", BAR_COMBINED))
    cells: list[dict] = []
    warnings: list[str] = []
    n_pass = n_gap = n_fail = 0
    total = len(schemes) * len(conditions) * len(SEEDS)
    done = 0
    for scheme in schemes:
        for condition, bar in conditions:
            trials = []
            for seed in SEEDS:
                trial = _run_seed(scheme, condition, seed)
                trials.append(trial)
                done += 1
                ber_text = (
                    "abstained" if trial["ber"] is None else f"{trial['ber']:.4f}"
                )
                print(
                    f"  [{done:>3}/{total}] {scheme:<8} {condition:<13} seed={seed} BER {ber_text}",
                    flush=True,
                )
            bers = [t["ber"] for t in trials if t["ber"] is not None]
            abstained = len(bers) < len(trials)
            worst = max(bers) if bers else None
            mean = float(sum(bers) / len(bers)) if bers else None
            ok = worst is not None and worst < bar
            exception = (
                not strict
                and not ok
                and condition == "combined-10dB"
                and scheme in KNOWN_GAP_COMBINED
            )
            if ok:
                status = "PASS"
                n_pass += 1
            elif exception:
                status = "KNOWN_GAP"
                n_gap += 1
                warnings.append(
                    f"{scheme} {condition}: worst-seed BER {worst:.4f} >= {bar} "
                    "-- listed known-gap exception per docs/acceptance_status.md:32, "
                    "reported honestly, not counted as a pass."
                )
            else:
                status = "FAIL"
                n_fail += 1
                reason = (
                    "receiver abstained on a seed"
                    if abstained
                    else f"worst-seed BER {worst:.4f} >= {bar}"
                )
                warnings.append(f"{scheme} {condition}: FAIL -- {reason}.")
            cells.append(
                {
                    "scheme": scheme,
                    "condition": condition,
                    "bar": bar,
                    "seeds": trials,
                    "ber_max": worst,
                    "ber_mean": mean,
                    "status": status,
                    "exception": bool(exception),
                }
            )
    return {
        "meta": {
            "gate": "receiver BER under combined CFO+Doppler+multipath+IQ",
            "harness": "channel.apply_channel -> receiver.demodulate -> ber_best_rotation",
            "reference": "tests/integration/test_robust_demod.py:18-52 pattern",
        },
        "config": {
            "schemes": list(schemes),
            "conditions": ["clean", "combined-10dB"],
            "bars": {"clean": BAR_CLEAN, "combined-10dB": BAR_COMBINED},
            "bar_applies_to": "worst seed per cell",
            "sps": SPS,
            "n_bits": N_BITS,
            "seeds": list(SEEDS),
            "sample_rate_hz": SAMPLE_RATE,
            "combined": {
                "cfo_hz": CFO_HZ,
                "doppler_rate_hz_s": DOPPLER_RATE_HZ_S,
                "multipath": [[d, g] for d, g in MULTIPATH],
                "iq_gain_imb": IQ_GAIN_IMB,
                "iq_phase_imb_deg": IQ_PHASE_IMB_DEG,
                "ebn0_db": 10.0,
            },
            "strict": bool(strict),
        },
        "known_gap_exceptions": {
            "schemes": list(KNOWN_GAP_COMBINED),
            "applies_to": "combined-10dB only",
            "source": "docs/acceptance_status.md:32",
        },
        "cells": cells,
        "summary": {
            "n_pass": n_pass,
            "n_known_gap": n_gap,
            "n_fail": n_fail,
            "overall": "PASS" if n_fail == 0 else "FAIL",
        },
        "warnings": warnings,
    }


def print_report(summary: dict) -> None:
    """Per-cell BER table with PASS / KNOWN_GAP / FAIL vs the bars."""
    print("\n=== Robustness gate: BER under combined CFO+Doppler+multipath+IQ ===")
    print(
        f"{'scheme':<8} {'condition':<13} {'seeds (BER)':<28} {'worst':<8} {'bar':<6} verdict"
    )
    for cell in summary["cells"]:
        seed_text = ",".join(
            "abst" if t["ber"] is None else f"{t['ber']:.4f}" for t in cell["seeds"]
        )
        worst_text = "abst" if cell["ber_max"] is None else f"{cell['ber_max']:.4f}"
        print(
            f"{cell['scheme']:<8} {cell['condition']:<13} {seed_text:<28} "
            f"{worst_text:<8} {cell['bar']:<6} {cell['status']}"
        )
    print()
    for warning in summary["warnings"]:
        print(f"WARNING: {warning}")
    counts = summary["summary"]
    print(
        f"\nGate: {counts['overall']} ({counts['n_pass']} pass, {counts['n_known_gap']} known-gap, {counts['n_fail']} fail)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Robustness gate: receiver BER under combined channel impairments."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="BPSK/QPSK only, for CI.")
    mode.add_argument("--full", action="store_true", help="All 7 schemes (default).")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Listed known-gap exceptions fail instead of warning.",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default=str(DEFAULT_JSON),
        help="Where to write the JSON summary.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    schemes = SCHEMES_QUICK if args.quick else SCHEMES_ALL
    print(
        f"Robustness gate: {len(schemes)} scheme(s) x 2 conditions x {len(SEEDS)} seeds (sps={SPS}, N_BITS={N_BITS})"
    )
    summary = run_gate(schemes, strict=args.strict)
    print_report(summary)

    out_path = Path(args.json_path)
    if out_path.parent != Path("") and str(out_path.parent) != ".":
        out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(summary), handle, indent=2, allow_nan=False)
    print(f"\nJSON summary: {out_path}")

    return 0 if summary["summary"]["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
