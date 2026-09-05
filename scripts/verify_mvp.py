"""MVP verification: generate data → pytest → pipeline checks → PASS/FAIL.

See info.md §23. Exits 0 and prints ``PASS`` only if every stage succeeds;
otherwise prints ``FAIL`` and exits 1.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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


def run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        print("FAIL")
        sys.exit(1)

    return result


def check_pipeline() -> bool:
    """Extra MVP checks beyond pytest: BER, correlation, report schema."""
    import numpy as np

    from rf_analyzer.core.demod import demod_bpsk
    from rf_analyzer.core.io import load_iq
    from rf_analyzer.pipeline import analyze_file

    ok = True

    def check(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        status = "OK" if passed else "FAIL"
        suffix = f": {detail}" if detail else ""
        print(f"  [{status}] {name}{suffix}")
        if not passed:
            ok = False

    sample_data = ROOT / "sample_data"
    report = analyze_file(
        {
            "file_path": str(sample_data / "bpsk.iq"),
            "sample_rate": 100000,
            "iq_format": "complex64",
            "modulation": "BPSK",
            "sync_word": "0x1ACFFC1D",
        }
    )

    check(
        "pipeline reports no errors",
        report.get("errors") == [],
        str(report.get("errors")),
    )

    missing = [key for key in REQUIRED_REPORT_KEYS if key not in report]
    check(
        "report has all §13 keys",
        not missing,
        f"missing={missing}" if missing else "all present",
    )

    samples = load_iq(str(sample_data / "bpsk.iq"), dtype="complex64")
    demod_bits = np.asarray(demod_bpsk(samples), dtype=np.uint8)
    truth = np.load(sample_data / "bpsk_bits.npy")
    n = min(len(demod_bits), len(truth))
    ber = float(np.mean(demod_bits[:n] != truth[:n])) if n else 1.0
    check("BPSK BER < 0.01", ber < 0.01, f"BER={ber:.6f} (n={n})")

    correlation = report.get("correlation", {})
    score = float(correlation.get("score", 0.0))
    check("correlation score > 0.9", score > 0.9, f"score={score}")
    check(
        "header offset detected",
        int(correlation.get("header_offset", -1)) >= 0,
        f"header_offset={correlation.get('header_offset')}",
    )

    return ok


def main() -> int:
    print("=== RF Analyzer MVP Verification ===")

    print("[1/3] Generating synthetic data...")
    run_command([sys.executable, "scripts/generate_test_data.py"])

    print("[2/3] Running pytest...")
    run_command([sys.executable, "-m", "pytest", "-q"])

    print("[3/3] Running MVP pipeline checks...")
    try:
        ok = check_pipeline()
    except Exception as exc:  # explicit, never silent
        print(f"  [FAIL] MVP pipeline checks raised: {exc}")
        ok = False

    if ok:
        print("MVP verification complete.")
        print("PASS")
        return 0

    print("MVP verification complete.")
    print("FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
