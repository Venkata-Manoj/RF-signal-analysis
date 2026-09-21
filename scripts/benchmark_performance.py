"""Performance budget benchmark (info.md NFR-01..03).

The NFRs are absolute claims about the shipped pipeline, so they need a shipped
measurement rather than an assertion in a comment:

* **NFR-03** -- analysis of 1 million samples in under 10 seconds.
* **NFR-02** -- visualisation stays responsive for 1-2 million samples.
* **NFR-01** -- a 100 MB file loads and analyses without crashing.

Every capture is a *realistic* one: a framed BPSK burst followed by a noise tail,
so the measurement exercises the whole funnel (PSD, waterfall, demodulation, the
decode search) rather than just the loader.

Two methodology choices, both because the obvious version of this script lies:

**Each case runs in its own process.** Cases interfere when measured back to
back: the 2 M case came out at 10.4 s behind the 1 M repeats and 4.1 s on its
own, because several large analyses in one process accumulate allocator and
page-cache pressure. A user runs one file per invocation, so the isolated number
is the honest one. The parent re-executes this file once per case.

**Each case reports median and range over repeats, never one run.** End-to-end
timings on a shared machine vary by more than the effect being measured. The 100 MB
case measured 32 s and 31 s in two back-to-back isolated runs, then 36 s and 499 s
during a longer batch on the same machine -- a 15x swing with no change in the work,
because a machine already holding 12 of its 16 GB was busy. Publishing either number
alone would be a lie; the range is the measurement.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from rf_analyzer.core.framing import build_frame, modulate
from rf_analyzer.pipeline import analyze_file

SAMPLE_RATE = 100_000
SYNC_WORD = "0x1ACFFC1D"
MESSAGE = b"SIH26147 performance budget payload."

#: NFR-03: analysis of 1 million samples must finish inside this many seconds.
BUDGET_1M_SECONDS = 10.0

#: ``(label, n_samples, repeats, budget_seconds or None)``. ``complex64`` is
#: 8 bytes per sample, so 12.5 M samples is the 100 MB file of NFR-01.
#:
#: Repeats fall off with size because only the 1 M case carries a budget and so
#: only that one needs a median worth defending. The 100 MB case is a single run
#: on purpose: at ~32 s each, three repeats cost a minute and a half to produce a
#: spread that is dominated by whatever else the machine is doing.
CASES: list[tuple[str, int, int, float | None]] = [
    ("NFR-03  1 M samples", 1_000_000, 3, BUDGET_1M_SECONDS),
    ("NFR-02  2 M samples", 2_000_000, 2, None),
    ("NFR-01  100 MB file", 12_500_000, 1, None),
]


def build_capture(n_samples: int, path: Path) -> float:
    """Write a framed BPSK burst padded with noise to ``n_samples``.

    Returns the file size in MB. The noise tail is what makes this the realistic
    case: it is the shape that exercises the burst estimate and the decode search.
    """
    frame = build_frame(MESSAGE, fec="conv", interleaver="block")
    samples = modulate(frame["bits"], "BPSK", sample_rate=SAMPLE_RATE)
    if samples.size < n_samples:
        rng = np.random.default_rng(7)
        pad = n_samples - samples.size
        noise = (rng.standard_normal(pad) + 1j * rng.standard_normal(pad)) * 0.05
        samples = np.concatenate([samples, noise.astype(np.complex64)])
    else:
        samples = samples[:n_samples]
    samples.astype(np.complex64).tofile(path)
    return path.stat().st_size / 1e6


def analyse(path: Path) -> dict:
    """Run the shipped headless funnel on one capture."""
    return analyze_file(
        {
            "file_path": str(path),
            "sample_rate": SAMPLE_RATE,
            "iq_format": "complex64",
            "modulation": "auto",
            "sync_word": SYNC_WORD,
        }
    )


def _measure_in_child(index: int) -> None:
    """Measure one case and print a single JSON line. Runs in the child process."""
    label, n_samples, repeats, budget = CASES[index]
    tmp = Path(tempfile.mkdtemp(prefix="rf_perf_"))
    path = tmp / f"{n_samples}.iq"
    size_mb = build_capture(n_samples, path)

    # Warm up so the first timed run does not pay one-off import and
    # bytecode-compile costs -- that would measure the interpreter, not the
    # analyser. The capture is re-read from disk on every run either way.
    analyse(path)

    times: list[float] = []
    report: dict = {}
    for _ in range(repeats):
        start = time.perf_counter()
        report = analyse(path)
        times.append(time.perf_counter() - start)

    print(
        json.dumps(
            {
                "label": label,
                "samples": n_samples,
                "mb": round(size_mb, 1),
                "median_s": round(statistics.median(times), 2),
                "min_s": round(min(times), 2),
                "max_s": round(max(times), 2),
                "budget_s": budget,
                "errors": report.get("errors") or [],
                "decoded": bool(
                    (report.get("payload") or {}).get("decoded", {}).get("available")
                ),
            }
        )
    )


def run_benchmark() -> dict:
    print(
        f"{'case':<22} {'samples':>12} {'MB':>7} {'median':>9} {'range':>17}  verdict"
    )
    print("-" * 88)

    results: dict = {"cases": [], "passed": True}
    for index in range(len(CASES)):
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--one", str(index)],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")]
        if proc.returncode != 0 or not lines:
            label = CASES[index][0]
            print(f"{label:<22} {'':>12} {'':>7} {'':>9} {'':>17}  ERROR")
            print(
                f"    child failed (exit {proc.returncode}): "
                f"{(proc.stderr or '').strip().splitlines()[-1:]}"
            )
            results["passed"] = False
            continue

        row = json.loads(lines[-1])
        budget = row["budget_s"]
        if row["errors"]:
            verdict = f"ERROR: {row['errors'][0]}"
            results["passed"] = False
        elif budget is not None:
            verdict = (
                "PASS" if row["median_s"] <= budget else f"FAIL (budget {budget:.0f} s)"
            )
            if row["median_s"] > budget:
                results["passed"] = False
        else:
            verdict = "OK (no time budget)"

        spread = (
            f"{row['min_s']:.2f}-{row['max_s']:.2f} s"
            if row["max_s"] > row["min_s"]
            else "(single run)"
        )
        print(
            f"{row['label']:<22} {row['samples']:>12,} {row['mb']:>7.1f} "
            f"{row['median_s']:>8.2f}s {spread:>17}  {verdict}"
        )
        row["verdict"] = verdict
        results["cases"].append(row)

    print()
    if results["passed"]:
        print("All performance budgets met.")
    else:
        print("PERFORMANCE BUDGET NOT MET -- see the verdict column.")
    return results


if __name__ == "__main__":
    if "--one" in sys.argv:
        _measure_in_child(int(sys.argv[sys.argv.index("--one") + 1]))
    else:
        sys.exit(0 if run_benchmark()["passed"] else 1)
