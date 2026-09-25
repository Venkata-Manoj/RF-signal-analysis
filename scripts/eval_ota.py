"""Over-the-air robustness harness: synthetic channel matrix + burst check + real data.

Three evidence paths, one script:

* **Synthetic matrix** -- every ``waveform`` modulation x SNR 0-20 dB x channel
  impairment (clean, CFO, Doppler drift, multipath, IQ imbalance, plus one
  combined cell stacking them all), 3 seeds per cell through the coherent
  ``receiver`` chain. BER is reported as a **min-max range with its mean
  across seeds, never a single number**: at a fixed SNR the
  outcome still depends on the noise draw, and publishing one draw would be
  anecdote.
* **Burst-in-noise matrix** -- 4 modulations x 5 noise levels x 7 paddings x
  3 seeds (420 cases) of ``noise | frame | noise`` through the shipped
  ``pipeline.analyze_file``. A reported decode must equal the transmitted
  message byte for byte; anything else is counted ``wrong`` and fails the run.
  Declining to decode is acceptable -- a wrong payload is not.
* **Real data** -- every ``real_data/*.iq`` with a known sample rate is
  analysed in process and reported (modulation label, fit, warnings). A
  verified decode is only ever accepted behind a CRC pass.

Methodology (same lesson as ``benchmark_performance.py``): **each case runs in
its own process**. Cases interfere when measured back to back (allocator and
page-cache pressure inflate later timings), and a loaded machine must not
decide a correctness result either. The parent re-executes this file once per
case with ``--one INDEX``; the child prints a single JSON line.

Usage::

    python scripts/eval_ota.py                 # full matrix (slow, ~15 min)
    python scripts/eval_ota.py --quick         # trimmed matrix for iteration
    python scripts/eval_ota.py --synth-only    # skip the burst sweep
    python scripts/eval_ota.py --burst-only    # skip the synthetic sweep
    python scripts/eval_ota.py --json out.json # also write machine-readable results
"""

from __future__ import annotations

import argparse
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

SAMPLE_RATE = 1_000_000.0
SPS = 8
N_BITS = 2400  # divisible by 12: whole symbols for every constellation width
SYNTH_SEEDS = (1, 7, 42)

#: The SNR axis is nominal *per-symbol* Es/N0. ``waveform.synth`` + AWGN define
#: SNR against the *oversampled waveform* power, whose mean is ``1/sps`` of the
#: symbol power (zero-stuffing spreads each symbol over ``sps`` samples), so a
#: matched filter recovers ``10*log10(sps)`` of processing gain -- 9.0 dB at
#: ``SPS=8``. Without compensation the axis would read 9 dB optimistic (BPSK
#: at "0 dB" measured BER 0.000 instead of the theoretical ~0.08). The child
#: therefore applies ``snr_waveform = snr_target - 10*log10(SPS)``. Exact for
#: the linear matched-filter path; approximate for the FSK discriminator,
#: whose non-coherent combining gains the same factor to first order.
SPS_GAIN_DB = float(10.0 * np.log10(SPS))

SYNTH_MODS = ("BPSK", "QPSK", "8PSK", "16-QAM", "64-QAM", "2-FSK", "4-FSK")
SYNTH_SNRS_DB = (0.0, 5.0, 10.0, 15.0, 20.0)

#: Impairment on top of the AWGN set by the SNR axis. Values sit inside the
#: receiver's pull-in (CFO 1.5 kHz is 1.2% of the 125 kHz symbol rate) so the
#: sweep measures degradation, not guaranteed loss of lock. ``combined-*``
#: stacks every singleton at once (the mild IQ corner 0.1/5deg, as in
#: ``tests/integration/test_robust_demod.py``) for the worst honest cell.
SYNTH_IMPAIRMENTS: dict[str, dict] = {
    "clean": {},
    "cfo-1.5kHz": {"cfo_hz": 1500.0},
    "doppler-8kHz/s": {"doppler_rate_hz_s": 8000.0},
    "multipath-3tap": {"multipath": [(0, 1.0), (1, 0.3), (2, 0.2)]},
    "iq-g0.2-p5deg": {"iq_gain_imb": 0.2, "iq_phase_imb_deg": 5.0},
    "combined-cfo-doppler-mp-iq": {
        "cfo_hz": 1500.0,
        "doppler_rate_hz_s": 8000.0,
        "multipath": [(0, 1.0), (1, 0.3), (2, 0.2)],
        "iq_gain_imb": 0.1,
        "iq_phase_imb_deg": 5.0,
    },
}

BURST_MODS = ("BPSK", "QPSK", "16-QAM", "2-FSK")
BURST_NOISE_AMPS = (0.02, 0.05, 0.10, 0.20, 0.35)
BURST_PADS = (0, 100, 200, 400, 800, 1600, 4000)
BURST_SEEDS = (11, 12, 13)
BURST_MESSAGE = b"SIH26147 OTA burst-in-noise robustness payload."
BURST_SAMPLE_RATE = 100_000
BURST_SYNC = "0x1ACFFC1D"
BURST_BUDGET = {"decode_time_budget_s": 120.0, "decode_max_bits": 200_000}

_ROTATION_ORDER = {"BPSK": 2, "QPSK": 4, "8PSK": 8, "16-QAM": 4, "64-QAM": 4}


def _synth_cases() -> list[dict]:
    cases = []
    for mod in SYNTH_MODS:
        for snr in SYNTH_SNRS_DB:
            for name in SYNTH_IMPAIRMENTS:
                cases.append(
                    {"kind": "synth", "mod": mod, "snr_db": snr, "impairment": name}
                )
    return cases


def _burst_cases() -> list[dict]:
    cases = []
    for mod in BURST_MODS:
        for amp in BURST_NOISE_AMPS:
            for pad in BURST_PADS:
                for seed in BURST_SEEDS:
                    cases.append(
                        {
                            "kind": "burst",
                            "mod": mod,
                            "noise_amp": amp,
                            "pad": pad,
                            "seed": seed,
                        }
                    )
    return cases


def _ber_best_rotation(
    tx_bits: np.ndarray, rx_bits: np.ndarray, scheme: str
) -> float | None:
    """Delegate to the shared receiver scorer (Costas ambiguity + equaliser lag)."""
    from rf_analyzer.core.receiver import ber_best_rotation

    return ber_best_rotation(tx_bits, rx_bits, scheme)


def _run_synth(case: dict) -> dict:
    import warnings

    from rf_analyzer.core import channel as ch
    from rf_analyzer.core import receiver as rx
    from rf_analyzer.core import waveform as wf

    mod, snr, imp_name = case["mod"], float(case["snr_db"]), case["impairment"]
    imp = SYNTH_IMPAIRMENTS[imp_name]
    # See SPS_GAIN_DB: the requested SNR is nominal Es/N0; the waveform-power
    # SNR the channel model takes is lower by the oversampling gain.
    snr_waveform = snr - SPS_GAIN_DB
    bers: list[float] = []
    times: list[float] = []
    locked = 0
    abstained = 0
    # The FSK k-means helper casts a complex-with-zero-imag array to real and
    # emits a ComplexWarning on every call (pre-existing receiver behaviour,
    # numbers unaffected). Silence it here so the harness output stays readable.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Casting complex values to real.*")
        for seed in SYNTH_SEEDS:
            clean, tx_bits = wf.synth(
                mod, N_BITS, sps=SPS, seed=seed, sample_rate=SAMPLE_RATE
            )
            impaired = ch.apply_channel(
                clean, SAMPLE_RATE, snr_db=snr_waveform, seed=seed + 1000, **imp
            )
            start = time.perf_counter()
            result = rx.receive(impaired, mod, sps=SPS, seed=seed)
            times.append(time.perf_counter() - start)
            ber = _ber_best_rotation(tx_bits, result["bits"], mod)
            if ber is None:
                abstained += 1
                continue
            bers.append(ber)
            locked += int(bool(result["locked"]))
    row: dict = {
        "kind": "synth",
        "mod": mod,
        "snr_db": snr,
        "impairment": imp_name,
        "ber_min": min(bers) if bers else None,
        "ber_max": max(bers) if bers else None,
        "ber_mean": (sum(bers) / len(bers)) if bers else None,
        "n_ber": len(bers),
        "n_abstained": abstained,
        "n_locked": locked,
        "t_min_s": round(min(times), 3),
        "t_median_s": round(statistics.median(times), 3),
        "t_max_s": round(max(times), 3),
    }
    return row


def _run_burst(case: dict) -> dict:
    from rf_analyzer.core.framing import build_frame, modulate
    from rf_analyzer.pipeline import analyze_file

    mod, amp, pad, seed = (
        case["mod"],
        float(case["noise_amp"]),
        int(case["pad"]),
        int(case["seed"]),
    )
    frame = build_frame(BURST_MESSAGE, fec="conv", interleaver="block")
    core = modulate(frame["bits"], mod, sample_rate=BURST_SAMPLE_RATE)
    rng = np.random.default_rng(seed)
    lead = (
        (rng.standard_normal(pad) + 1j * rng.standard_normal(pad)) * amp
        if pad
        else np.zeros(0, dtype=np.complex128)
    )
    trail = (
        (rng.standard_normal(pad) + 1j * rng.standard_normal(pad)) * amp
        if pad
        else np.zeros(0, dtype=np.complex128)
    )
    capture = np.concatenate([lead, core.astype(np.complex128), trail]).astype(
        np.complex64
    )

    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="rf_ota_burst_") as raw:
        path = Path(raw) / "burst.iq"
        capture.tofile(path)
        report = analyze_file(
            {
                "file_path": str(path),
                "sample_rate": BURST_SAMPLE_RATE,
                "iq_format": "complex64",
                "modulation": "auto",
                "sync_word": BURST_SYNC,
                **BURST_BUDGET,
            }
        )
    elapsed = time.perf_counter() - start

    decoded = (report.get("payload") or {}).get("decoded") or {}
    search = (report.get("payload") or {}).get("decode_search") or {}
    outcome = "declined"
    if decoded.get("available") and decoded.get("crc_pass"):
        outcome = (
            "exact"
            if bytes.fromhex(decoded.get("hex", "")) == BURST_MESSAGE
            else "WRONG"
        )
    return {
        "kind": "burst",
        "mod": mod,
        "noise_amp": amp,
        "pad": pad,
        "seed": seed,
        "outcome": outcome,
        # Honesty flags: a decode only ever counts behind a CRC pass (see
        # ``outcome``), and a case that hit the search budget says so here
        # instead of reading as a clean decline.
        "decoded_available": bool(decoded.get("available")),
        "crc_pass": bool(decoded.get("crc_pass")) if decoded.get("available") else None,
        "budget_exhausted": bool(search.get("budget_exhausted", False)),
        "seconds": round(elapsed, 3),
    }


def _run_one(index: int, cases: list[dict]) -> None:
    case = cases[index]
    row = _run_synth(case) if case["kind"] == "synth" else _run_burst(case)
    print(json.dumps(row))


def _spawn(index: int) -> dict | None:
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--one", str(index)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")]
    if proc.returncode != 0 or not lines:
        tail = (proc.stderr or "").strip().splitlines()[-1:] if proc.stderr else []
        print(f"    [child {index} FAILED exit={proc.returncode} {' '.join(tail)}]")
        return None
    return json.loads(lines[-1])


def _fmt_range(lo, hi, digits: int = 4) -> str:
    if lo is None or hi is None:
        return "abstained"
    return f"{lo:.{digits}f}-{hi:.{digits}f}"


def _report_synth(rows: list[dict]) -> None:
    print("\n=== Synthetic channel matrix: BER range (min-max across 3 seeds) ===")
    print("SNR axis is nominal per-symbol Es/N0 (see SPS_GAIN_DB).")
    print(
        f"{'mod':<8} {'snr':>6} {'impairment':<16} {'BER min-max':<23} {'locked':>6}  timing median(range)"
    )
    print("-" * 96)
    for row in rows:
        if row is None:
            continue
        timing = f"{row['t_median_s']:.2f}s ({row['t_min_s']:.2f}-{row['t_max_s']:.2f})"
        print(
            f"{row['mod']:<8} {row['snr_db']:>5.0f}dB {row['impairment']:<16} "
            f"{_fmt_range(row['ber_min'], row['ber_max']):<23} {row['n_locked']:>3}/3  {timing}"
        )
    print("\n--- Per-modulation overall BER span (min-max across SNR x impairment) ---")
    for mod in SYNTH_MODS:
        vals = [
            v
            for r in rows
            if r and r["mod"] == mod
            for v in ([r["ber_min"], r["ber_max"]] if r["ber_min"] is not None else [])
        ]
        n_abs = sum(r["n_abstained"] for r in rows if r and r["mod"] == mod)
        print(
            f"  {mod:<8} BER span {_fmt_range(min(vals), max(vals)) if vals else 'abstained'}   abstentions: {n_abs}"
        )


def _report_burst(rows: list[dict]) -> bool:
    exact = sum(1 for r in rows if r and r["outcome"] == "exact")
    declined = sum(1 for r in rows if r and r["outcome"] == "declined")
    wrong = sum(1 for r in rows if r and (r is None or r["outcome"] == "WRONG"))
    budgeted = sum(1 for r in rows if r and r.get("budget_exhausted"))
    total = len(rows)
    times = [r["seconds"] for r in rows if r]
    print(
        "\n=== Burst-in-noise matrix: 4 mods x 5 noise levels x 7 paddings x 3 seeds ==="
    )
    print(
        f"  cases: {total}   exact: {exact}   declined: {declined}   WRONG: {wrong}   budget_exhausted: {budgeted}"
    )
    if times:
        print(
            f"  per-case time (each in its own process): median {statistics.median(times):.2f}s range {min(times):.2f}-{max(times):.2f}s"
        )
    for mod in BURST_MODS:
        sub = [r for r in rows if r and r["mod"] == mod]
        e = sum(1 for r in sub if r["outcome"] == "exact")
        w = sum(1 for r in sub if r["outcome"] == "WRONG")
        print(f"    {mod:<8} exact {e:>3}/{len(sub)}   WRONG {w}")
    if wrong:
        print("  FAIL: a case reported a payload that is not the message.")
        bad = [r for r in rows if not r or (r and r["outcome"] == "WRONG")]
        for r in bad[:10]:
            print(f"    WRONG/ERROR: {r}")
        return False
    print("  PASS: 0 wrong payloads -- every claim is exactly the message.")
    return True


def _report_real_data() -> None:
    from rf_analyzer.pipeline import analyze_file

    try:
        from scripts.fetch_real_data import CAPTURES
    except ImportError:
        CAPTURES = ()
    known = {c["name"]: c for c in CAPTURES}
    data_dir = ROOT / "real_data"
    files = sorted(data_dir.glob("*.iq")) if data_dir.is_dir() else []
    print("\n=== Real OTA captures (real_data/) ===")
    if not files:
        print("  (none present -- run python scripts/fetch_real_data.py)")
        return
    for path in files:
        meta = known.get(path.name, {})
        rate = meta.get("sample_rate")
        if not rate:
            print(
                f"  {path.name}: no known sample rate -- skipped (pass it explicitly, raw IQ has no metadata)"
            )
            continue
        report = analyze_file(
            {
                "file_path": str(path),
                "sample_rate": rate,
                "iq_format": meta.get("iq_format", "auto"),
            }
        )
        mod = (report.get("modulation") or {}).get("estimated_type")
        fit = (report.get("quality") or {}).get("fit_ok")
        decoded = (report.get("payload") or {}).get("decoded") or {}
        claim = (
            f"available={decoded.get('available')} crc_pass={decoded.get('crc_pass')}"
        )
        honest = (
            "OK"
            if (not decoded.get("available") or decoded.get("crc_pass"))
            else "DISHONEST-CLAIM"
        )
        print(f"  {path.name}: mod={mod} fit_ok={fit} decode[{claim}] {honest}")
        for warning in (report.get("warnings") or [])[:3]:
            print(f"    warn: {warning[:120]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else "OTA harness"
    )
    parser.add_argument(
        "--one",
        type=int,
        default=None,
        help="run one case index in this process (child mode)",
    )
    parser.add_argument(
        "--quick", action="store_true", help="trimmed matrix for iteration"
    )
    parser.add_argument("--synth-only", action="store_true")
    parser.add_argument("--burst-only", action="store_true")
    parser.add_argument("--real-only", action="store_true")
    parser.add_argument(
        "--json", default=None, help="write machine-readable results to PATH"
    )
    args = parser.parse_args(argv)

    synth_cases = _synth_cases()
    burst_cases = _burst_cases()
    all_cases = synth_cases + burst_cases

    if args.one is not None:
        _run_one(args.one, all_cases)
        return 0

    if args.quick:
        synth_cases = [
            c
            for c in synth_cases
            if c["snr_db"] in (0.0, 10.0, 20.0)
            and c["impairment"]
            in ("clean", "cfo-1.5kHz", "multipath-3tap", "iq-g0.2-p5deg")
        ]
        burst_cases = [
            c
            for c in burst_cases
            if c["noise_amp"] in (0.05, 0.20)
            and c["pad"] in (0, 400, 1600)
            and c["seed"] == 11
        ]
        print(f"[quick] {len(synth_cases)} synth + {len(burst_cases)} burst cases")

    results: dict = {"synth": [], "burst": [], "burst_pass": None}
    ok = True

    run_synth = not args.burst_only and not args.real_only
    run_burst = not args.synth_only and not args.real_only
    run_real = not args.synth_only and not args.burst_only

    if run_synth:
        print(
            f"Running {len(synth_cases)} synthetic cases, each in its own process ..."
        )
        for case in synth_cases:
            global_index = all_cases.index(case)
            row = _spawn(global_index)
            if row is None:
                ok = False
            results["synth"].append(row)
        _report_synth(results["synth"])

    if run_burst:
        print(f"\nRunning {len(burst_cases)} burst cases, each in its own process ...")
        offset = len(_synth_cases())
        full_burst = _burst_cases()
        for case in burst_cases:
            global_index = offset + full_burst.index(case)
            row = _spawn(global_index)
            if row is None:
                ok = False
            results["burst"].append(row)
        burst_pass = _report_burst(results["burst"])
        results["burst_pass"] = burst_pass
        ok = ok and burst_pass

    if run_real:
        _report_real_data()

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"\nResults written to {out}")

    if not ok:
        print("\nEVAL FAILED -- see rows above.")
        return 1
    print("\nEVAL DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
