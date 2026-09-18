"""Characterise FEC capability so the documented claims stay reproducible.

`README.md` and `AGENTS.md` make specific numerical claims about LDPC tolerance
and about what the interleaver buys. Those claims are easy to state and easy to
get wrong -- an earlier revision of this project shipped one ("6 errors fail
while 8 and 12 succeed") that a re-measurement could not reproduce. This script
exists so any of them can be re-derived rather than trusted.

Two sweeps:

* ``scatter`` -- errors at uniformly random positions, the ordinary channel
  model.
* ``burst``   -- one contiguous run of flipped bits, the failure mode an
  interleaver actually exists to defend against.

Every cell is a success *rate* over several seeds. That matters: LDPC correction
is probabilistic, so at a fixed error count the outcome depends on how the errors
land across the 84-bit blocks, and any single seed is anecdote rather than
evidence.

Usage::

    python scripts/measure_fec_capability.py            # 5 seeds, both sweeps
    python scripts/measure_fec_capability.py --quick    # 2 seeds, faster
    python scripts/measure_fec_capability.py --sweep burst
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from rf_analyzer.core.framing import build_frame, modulate
from rf_analyzer.pipeline import analyze_file

SAMPLE_RATE = 100_000
SYNC_WORD = "0x1ACFFC1D"
MESSAGE = b"SIH26147 end-to-end verification payload: FEC plus de-interleaving."
BUDGET = {"decode_time_budget_s": 120.0, "decode_max_bits": 200_000}

SCATTER_LOADS = [0, 2, 4, 6, 8, 10, 12, 14, 16]
BURST_LOADS = [4, 8, 16, 24, 32, 48]
DEFAULT_SEEDS = [1, 7, 42, 99, 2024]


def _decoded(report: dict) -> bool:
    """True only when the recovered bytes equal the transmitted message."""
    decoded = (report.get("payload") or {}).get("decoded") or {}
    if not decoded.get("available") or not decoded.get("crc_pass"):
        return False
    return bytes.fromhex(decoded.get("hex", "")) == MESSAGE


def _run(fec: str, interleaver: str, n_errors: int, seed: int, burst: bool, tmp: Path):
    frame = build_frame(MESSAGE, fec=fec, interleaver=interleaver)
    sync_len = int(frame["sync_bits"].size)
    bits = frame["bits"].copy()

    if n_errors:
        rng = np.random.default_rng(seed)
        if burst:
            # Keep the run inside the coded region so the header survives and
            # the error is a genuine coding problem rather than a framing one.
            start = int(rng.integers(sync_len, max(sync_len + 1, bits.size - n_errors)))
            bits[start : start + n_errors] ^= 1
        else:
            idx = rng.choice(
                np.arange(sync_len, bits.size), size=n_errors, replace=False
            )
            bits[idx] ^= 1

    samples = modulate(bits, "BPSK", sample_rate=SAMPLE_RATE)
    path = tmp / f"{fec}_{interleaver}_{n_errors}_{seed}.iq"
    samples.astype(np.complex64).tofile(path)

    return _decoded(
        analyze_file(
            {
                "file_path": str(path),
                "sample_rate": SAMPLE_RATE,
                "sync_word": SYNC_WORD,
                "modulation": "auto",
                **BUDGET,
            }
        )
    )


def sweep(fec: str, interleavers, loads, seeds, *, burst: bool, tmp: Path) -> None:
    kind = "contiguous burst" if burst else "uniformly scattered"
    print(f"\n=== {fec} -- {kind} errors, {len(seeds)} seeds each ===")
    header = "interleaver".ljust(21) + "".join(f"{n:>7}" for n in loads)
    print(header)
    print("-" * len(header))

    for interleaver in interleavers:
        row = interleaver.ljust(21)
        for n_errors in loads:
            ok = sum(
                _run(fec, interleaver, n_errors, seed, burst, tmp) for seed in seeds
            )
            row += f"{ok:>4}/{len(seeds)}"
        print(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fec", default="ldpc", help="FEC scheme to characterise")
    parser.add_argument("--seeds", type=int, default=5, help="seeds per cell")
    parser.add_argument("--quick", action="store_true", help="2 seeds, both sweeps")
    parser.add_argument("--sweep", choices=("scatter", "burst", "both"), default="both")
    args = parser.parse_args()

    from rf_analyzer.core.framing import INTERLEAVER_SCHEMES

    seeds = DEFAULT_SEEDS[: 2 if args.quick else args.seeds]

    with tempfile.TemporaryDirectory(prefix="fec_capability_") as raw:
        tmp = Path(raw)
        if args.sweep in ("scatter", "both"):
            sweep(
                args.fec,
                INTERLEAVER_SCHEMES,
                SCATTER_LOADS,
                seeds,
                burst=False,
                tmp=tmp,
            )
        if args.sweep in ("burst", "both"):
            sweep(
                args.fec,
                INTERLEAVER_SCHEMES,
                BURST_LOADS,
                seeds,
                burst=True,
                tmp=tmp,
            )

    print(
        "\nCells are successes/seeds. LDPC correction is probabilistic, so a cell\n"
        "below 100% is a rate, not a bug -- read the shape of the column."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
