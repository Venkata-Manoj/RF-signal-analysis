"""MVP verification: generate data → pytest → pipeline checks → PASS/FAIL.

See info.md §23. Exits 0 and prints ``PASS`` only if every stage succeeds;
otherwise prints ``FAIL`` and exits 1.

Beyond the §23 checks this also verifies the problem statement's error-control
requirements end to end: every coded capture in ``sample_data/coded_manifest.json``
must be demodulated, de-interleaved, FEC-decoded and CRC-verified back to the
exact message that was transmitted — with the transmitter's scheme identified
from the bit stream alone.
"""

from __future__ import annotations

import json
import math
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


class Checks:
    """Collects check results so one failure does not hide the others."""

    def __init__(self) -> None:
        self.ok = True

    def check(self, name: str, passed: bool, detail: str = "") -> None:
        status = "OK" if passed else "FAIL"
        suffix = f": {detail}" if detail else ""
        print(f"  [{status}] {name}{suffix}")
        if not passed:
            self.ok = False


def check_pipeline(checks: Checks) -> None:
    """Core MVP checks beyond pytest: BER, correlation, report schema."""
    import numpy as np

    from rf_analyzer.core.demod import demod_bpsk
    from rf_analyzer.core.io import load_iq
    from rf_analyzer.pipeline import analyze_file

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

    checks.check(
        "pipeline reports no errors",
        report.get("errors") == [],
        str(report.get("errors")),
    )

    missing = [key for key in REQUIRED_REPORT_KEYS if key not in report]
    checks.check(
        "report has all §13 keys",
        not missing,
        f"missing={missing}" if missing else "all present",
    )

    samples = load_iq(str(sample_data / "bpsk.iq"), dtype="complex64")
    demod_bits = np.asarray(demod_bpsk(samples), dtype=np.uint8)
    truth = np.load(sample_data / "bpsk_bits.npy")
    n = min(len(demod_bits), len(truth))
    ber = float(np.mean(demod_bits[:n] != truth[:n])) if n else 1.0
    checks.check("BPSK BER < 0.01", ber < 0.01, f"BER={ber:.6f} (n={n})")

    correlation = report.get("correlation", {})
    score = float(correlation.get("score", 0.0))
    checks.check("correlation score > 0.9", score > 0.9, f"score={score}")
    checks.check(
        "header offset detected",
        int(correlation.get("header_offset", -1)) >= 0,
        f"header_offset={correlation.get('header_offset')}",
    )


def check_coded_captures(checks: Checks) -> None:
    """Every coded capture must decode back to its transmitted message."""
    from rf_analyzer.pipeline import analyze_file

    manifest_path = ROOT / "sample_data" / "coded_manifest.json"
    if not manifest_path.exists():
        checks.check("coded manifest present", False, str(manifest_path))
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    captures = manifest.get("captures", [])
    checks.check(
        "coded manifest lists captures", len(captures) >= 4, f"{len(captures)}"
    )

    for entry in captures:
        name = entry["file"]
        report = analyze_file(
            {
                "file_path": str(ROOT / "sample_data" / name),
                "sample_rate": entry["sample_rate"],
                "iq_format": entry["iq_format"],
                "modulation": "auto",
                "sync_word": entry["sync_word"],
            }
        )
        decoded = (report.get("payload") or {}).get("decoded") or {}
        verified = bool(decoded.get("available"))
        recovered = bytes.fromhex(decoded.get("hex", "")) if verified else b""
        expected = bytes.fromhex(entry["message_hex"])

        checks.check(
            f"{name}: CRC-verified payload recovered",
            verified and recovered == expected,
            f"fec={report.get('fec', {}).get('candidate')} "
            f"ilv={report.get('interleaving', {}).get('candidate')} "
            f"bytes={len(recovered)}/{len(expected)}",
        )
        checks.check(
            f"{name}: transmitter scheme identified",
            report.get("fec", {}).get("validated") is True
            and report.get("interleaving", {}).get("candidate") == entry["interleaver"],
            f"expected interleaver={entry['interleaver']}",
        )


def check_2fsk_symbol_recovery(checks: Checks) -> None:
    """2-FSK must be decimated to one bit per symbol, and say so honestly.

    "Demodulate FSK" is only genuinely satisfied if the receiver copes with a
    capture that spends many samples on each bit. The naive one-bit-per-sample
    path smears the sync word past every threshold, so this checks the recovered
    period, the derived symbol rate and the resulting bit count -- and that a
    signal with no symbol structure is *not* given a period.
    """
    import numpy as np

    from rf_analyzer.core.dsp import estimate_fsk_symbol_period
    from rf_analyzer.core.io import load_iq
    from rf_analyzer.pipeline import analyze_file

    fsk_path = ROOT / "sample_data" / "fsk2.iq"
    if not fsk_path.exists():
        checks.check("2-FSK sample present", False, str(fsk_path))
        return

    samples = load_iq(str(fsk_path), dtype="complex64")
    period = estimate_fsk_symbol_period(samples)
    checks.check(
        "2-FSK sample: symbol period recovered",
        period == 100,
        f"period={period} samples (generator used 100)",
    )

    report = analyze_file(
        {
            "file_path": str(fsk_path),
            "sample_rate": 100_000,
            "iq_format": "complex64",
            "modulation": "auto",
            "sync_word": "0x1ACFFC1D",
        }
    )
    checks.check(
        "2-FSK sample: modulation classified",
        report.get("modulation", {}).get("estimated_type") == "2-FSK",
        f"estimated={report.get('modulation', {}).get('estimated_type')}",
    )
    checks.check(
        "2-FSK sample: symbol rate derived from the period",
        report.get("signal", {}).get("symbol_rate_estimate") == 1000.0,
        f"symbol_rate={report.get('signal', {}).get('symbol_rate_estimate')} Hz "
        "(1 ms symbols at 100 kHz)",
    )
    correlation = report.get("correlation", {})
    checks.check(
        "2-FSK sample: sync word found in the decimated stream",
        correlation.get("detected") is True
        and float(correlation.get("score") or 0.0) > 0.9,
        f"score={correlation.get('score')} offset={correlation.get('offset')}",
    )

    # Negative controls: a period must never be invented for a signal that has
    # no symbol structure, because the pipeline would then decimate it.
    rng = np.random.default_rng(7)
    bits = rng.integers(0, 2, size=4096, dtype=np.uint8)
    bpsk = (2.0 * bits - 1.0).astype(np.complex64)
    noise = (rng.standard_normal(40_000) + 1j * rng.standard_normal(40_000)).astype(
        np.complex64
    )
    checks.check(
        "2-FSK period declines on non-FSK signals",
        estimate_fsk_symbol_period(bpsk) == 1
        and estimate_fsk_symbol_period(noise) == 1,
        f"bpsk={estimate_fsk_symbol_period(bpsk)} "
        f"noise={estimate_fsk_symbol_period(noise)}",
    )

    # The classifier's FSK test is a narrowband test, not an FSK test: it also
    # fires for an unmodulated tone and for audio. The period recovery is the
    # independent evidence, so an uncorroborated 2-FSK label must lose its
    # confidence instead of being reported as a confident estimate.
    from rf_analyzer.config import MODULATION_UNCORROBORATED_CONFIDENCE

    tone_path = ROOT / "sample_data" / "tone.iq"
    if tone_path.exists():
        tone_report = analyze_file(
            {
                "file_path": str(tone_path),
                "sample_rate": 100_000,
                "iq_format": "complex64",
                "modulation": "auto",
                "sync_word": "0x1ACFFC1D",
            }
        )
        tone_mod = tone_report.get("modulation", {})
        checks.check(
            "uncorroborated 2-FSK keeps its label but loses its confidence",
            tone_mod.get("estimated_type") == "2-FSK"
            and tone_mod.get("corroborated") is False
            and float(tone_mod.get("confidence", 1.0))
            <= MODULATION_UNCORROBORATED_CONFIDENCE
            and any("uncorroborated" in w for w in tone_report.get("warnings", [])),
            f"type={tone_mod.get('estimated_type')} "
            f"conf={tone_mod.get('confidence')} "
            f"corroborated={tone_mod.get('corroborated')}",
        )
    checks.check(
        "genuine 2-FSK classification is corroborated",
        report.get("modulation", {}).get("corroborated") is True,
        f"corroborated={report.get('modulation', {}).get('corroborated')}",
    )


def check_payload_honesty(checks: Checks) -> None:
    """Random bits must never be reported as a decoded message."""
    import numpy as np

    from rf_analyzer.pipeline import analyze_file

    rng = np.random.default_rng(2026)
    samples = (rng.integers(0, 2, size=8000).astype(np.float32) * 2.0 - 1.0).astype(
        np.complex64
    )
    out = ROOT / "output"
    out.mkdir(exist_ok=True)
    noise_path = out / "_verify_noise.iq"
    samples.tofile(noise_path)
    try:
        report = analyze_file(
            {
                "file_path": str(noise_path),
                "sample_rate": 100000,
                "modulation": "BPSK",
                "sync_word": "0x1ACFFC1D",
            }
        )
        payload = report.get("payload") or {}
        checks.check(
            "random bits produce no decoded payload",
            (payload.get("decoded") or {}).get("available") is False,
            f"candidate={report.get('fec', {}).get('candidate')}",
        )
        checks.check(
            "blind FEC score stays capped at 0.5",
            float(report.get("fec", {}).get("confidence", 1.0)) <= 0.5,
            f"confidence={report.get('fec', {}).get('confidence')}",
        )
        checks.check(
            "undecoded capture is warned about",
            any("not a decoded message" in w for w in report.get("warnings", [])),
            f"{len(report.get('warnings', []))} warning(s)",
        )
    finally:
        # Best-effort scratch cleanup. A scratch file that cannot be removed
        # (locked, read-only, or blocked by a sandbox policy) must not turn a
        # passing verification into a failure: what is being verified is the
        # report, not the tidiness of `output/`.
        try:
            noise_path.unlink(missing_ok=True)
        except OSError:
            pass


def check_auto_detection(checks: Checks) -> None:
    """iq_format='auto' must infer the on-disk dtype without being told."""
    from rf_analyzer.pipeline import analyze_file

    report = analyze_file(
        {
            "file_path": str(ROOT / "sample_data" / "bpsk.iq"),
            "sample_rate": 100000,
            "iq_format": "auto",
            "modulation": "BPSK",
            "sync_word": "0x1ACFFC1D",
        }
    )
    used = (report.get("input") or {}).get("iq_format_used")
    checks.check(
        "auto IQ-format detection picks complex64",
        used == "complex64",
        f"detected={used}",
    )
    checks.check(
        "auto-detected capture still correlates",
        float(report.get("correlation", {}).get("score", 0.0)) > 0.9,
        f"score={report.get('correlation', {}).get('score')}",
    )


def check_batch_exports(checks: Checks) -> None:
    """The batch runner must produce a CSV and a self-contained HTML page."""
    from rf_analyzer.batch import analyze_folder, write_csv, write_html

    rows = analyze_folder(
        ROOT / "sample_data",
        sample_rate=100000,
        iq_format="complex64",
        modulation="auto",
        sync_word="0x1ACFFC1D",
        decode=True,
    )
    verified = sum(1 for r in rows if r.get("decode_validated") == "yes")
    checks.check(
        "batch analysis runs over the sample folder",
        len(rows) >= 6,
        f"{len(rows)} capture(s), {verified} verified",
    )
    checks.check(
        "batch verifies every coded capture",
        verified >= 5,
        f"verified={verified}",
    )

    out = ROOT / "output"
    try:
        csv_path = write_csv(rows, out / "_verify_batch.csv")
        html_path = write_html(rows, out / "_verify_batch.html")
        text = html_path.read_text(encoding="utf-8")
        checks.check(
            "batch CSV export written",
            csv_path.exists() and csv_path.stat().st_size > 0,
            f"{csv_path.stat().st_size} bytes",
        )
        checks.check(
            "batch HTML export is self-contained",
            html_path.exists() and "http" not in text.replace("http-equiv", ""),
            f"{html_path.stat().st_size} bytes",
        )
    finally:
        # Best-effort scratch cleanup -- see the note in check_payload_honesty.
        for name in ("_verify_batch.csv", "_verify_batch.html"):
            try:
                (out / name).unlink(missing_ok=True)
            except OSError:
                pass


def check_report_schema(checks: Checks) -> None:
    """Every report must carry the info.md §13 keys plus the additive blocks.

    Success and error reports must expose the *same* top-level shape so a
    consumer can read a field without first checking which branch produced it.
    """
    from rf_analyzer.pipeline import analyze_file

    ok = analyze_file(
        {
            "file_path": str(ROOT / "sample_data" / "bpsk.iq"),
            "sample_rate": 100000,
            "iq_format": "complex64",
            "modulation": "auto",
            "sync_word": "0x1ACFFC1D",
        }
    )
    missing = [k for k in REQUIRED_REPORT_KEYS if k not in ok]
    checks.check(
        "report has every §13 key", not missing, f"missing={missing or 'none'}"
    )

    additive = ("payload", "display", "quality")
    absent = [k for k in additive if k not in ok]
    checks.check(
        "report has the additive blocks (payload/display/quality)",
        not absent,
        f"missing={absent or 'none'}",
    )

    bad = analyze_file({"file_path": str(ROOT / "sample_data" / "nope.xyz")})
    checks.check(
        "unsupported suffix yields an error report, not an exception",
        bool(bad.get("errors")),
        f"errors={bad.get('errors')}",
    )
    checks.check(
        "error and success reports share one top-level shape",
        set(bad) == set(ok),
        f"only_in_error={sorted(set(bad) - set(ok))} "
        f"only_in_success={sorted(set(ok) - set(bad))}",
    )

    # The on-disk report is consumed outside Python (the dashboard, any JS
    # tooling), and JSON.parse rejects the Infinity/NaN tokens json.dump
    # emits by default.
    disk = ROOT / "output" / "bpsk_report.json"
    if disk.is_file():
        text = disk.read_text(encoding="utf-8")
        try:
            json.loads(text, parse_constant=_reject_constant)
            strict_ok, detail = True, f"{len(text)} bytes"
        except ValueError as exc:
            strict_ok, detail = False, str(exc)
        checks.check("on-disk report is strict JSON", strict_ok, detail)


def _reject_constant(token: str):
    """json.loads hook that rejects non-finite tokens, as a browser does."""
    raise ValueError(f"non-finite JSON token: {token}")


def check_dashboard(checks: Checks) -> None:
    """The zero-install web dashboard must serve its page and analyse a capture.

    The server is started in-process on an ephemeral localhost port so the
    check exercises the real HTTP path (routing, query parsing, pipeline call,
    plot assembly) without needing a browser or a fixed port.
    """
    import importlib.util
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    spec = importlib.util.spec_from_file_location(
        "serve_dashboard", ROOT / "scripts" / "serve_dashboard.py"
    )
    if spec is None or spec.loader is None:
        checks.check("dashboard server module loads", False, "import spec unavailable")
        return
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    page = module.PAGE_PATH
    checks.check(
        "dashboard page exists",
        page.is_file(),
        f"{page.stat().st_size} bytes" if page.is_file() else str(page),
    )
    if page.is_file():
        html = page.read_text(encoding="utf-8")
        checks.check(
            "dashboard page is self-contained",
            "<script src=" not in html
            and "https://" not in html
            and "<link" not in html,
            "no CDN, no external fetches",
        )

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), module.DashboardHandler)
    httpd.daemon_threads = True
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{port}"
        with urllib.request.urlopen(base + "/api/samples", timeout=30) as resp:
            samples = json.loads(resp.read())
        captures = samples.get("captures", [])
        checks.check(
            "dashboard lists bundled captures",
            len(captures) >= 6,
            f"{len(captures)}",
        )

        url = (
            f"{base}/api/analyze?name=coded_rs.iq&sample_rate=100000"
            "&iq_format=auto&modulation=auto&sync_word=0x1ACFFC1D&decode=1"
        )
        with urllib.request.urlopen(url, timeout=180) as resp:
            raw_body = resp.read()
        # A browser's JSON.parse rejects `Infinity`/`NaN`, so the response must
        # be strict JSON -- this is how the MER=inf bug surfaced.
        body_text = raw_body.decode("utf-8")
        checks.check(
            "dashboard response is strict JSON (no Infinity/NaN)",
            "Infinity" not in body_text and "NaN" not in body_text,
            f"{len(raw_body)} bytes",
        )
        payload = json.loads(body_text, parse_constant=_reject_constant)
        report = payload.get("report", {})
        decoded = (report.get("payload") or {}).get("decoded") or {}
        checks.check(
            "dashboard decodes a coded capture",
            bool(decoded.get("available")) and decoded.get("crc_pass") is True,
            f"scheme={decoded.get('scheme')} bytes={decoded.get('bytes')}",
        )

        plots = payload.get("plots") or {}
        checks.check(
            "dashboard returns every plot",
            set(plots) == {"constellation", "psd", "waterfall", "eye", "time"},
            ",".join(sorted(plots)),
        )
        quality = report.get("quality") or {}
        evm = quality.get("evm_percent")
        # mer_db is None when the error vector is exactly zero (a noiseless
        # synthetic capture), because the ratio is unbounded -- and `inf`
        # would not be valid JSON. Either a finite MER or None is correct.
        checks.check(
            "dashboard reports constellation quality (EVM/MER)",
            quality.get("applicable") is True
            and evm is not None
            and math.isfinite(float(evm))
            and (
                quality.get("mer_db") is None or math.isfinite(float(quality["mer_db"]))
            ),
            f"EVM={evm} MER={quality.get('mer_db')}",
        )

        # Leaving the sync word out must still frame the capture (default
        # probe) -- and the assumption must be flagged, not hidden.
        url = f"{base}/api/analyze?name=coded_conv.iq&sample_rate=100000&iq_format=complex64"
        with urllib.request.urlopen(url, timeout=180) as resp:
            probe = json.loads(resp.read())
        cor = probe.get("report", {}).get("correlation", {})
        checks.check(
            "dashboard frames a capture with no sync word given",
            cor.get("assumed") is True and cor.get("detected") is True,
            f"assumed={cor.get('assumed')} score={cor.get('score')}",
        )
        checks.check(
            "dashboard still decodes without a sync word",
            bool(
                (
                    (probe.get("report", {}).get("payload") or {}).get("decoded") or {}
                ).get("available")
            ),
            "default probe recovered the payload",
        )
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def check_real_captures(checks: Checks) -> None:
    """Third-party over-the-air captures must be analysed honestly.

    These are real SDR recordings fetched by ``scripts/fetch_real_data.py``
    (see ``real_data/SOURCES.json``). They matter because the synthetic
    captures are generated by this project, so on their own they only prove
    the pipeline agrees with itself.

    Skipped with an explicit note when the captures are absent -- they are
    large third-party binaries and are deliberately not committed.
    """
    from rf_analyzer.pipeline import analyze_file

    manifest_path = ROOT / "real_data" / "SOURCES.json"
    if not manifest_path.exists():
        checks.check(
            "real-world captures available",
            True,
            "skipped (run scripts/fetch_real_data.py to enable this group)",
        )
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    present = [
        entry
        for entry in manifest.get("captures", [])
        if (ROOT / "real_data" / entry["name"]).is_file()
    ]
    checks.check(
        "real-world captures available",
        bool(present),
        f"{len(present)} of {len(manifest.get('captures', []))} present",
    )

    for entry in present:
        name = entry["name"]
        report = analyze_file(
            {
                "file_path": str(ROOT / "real_data" / name),
                "sample_rate": entry["sample_rate"],
                "iq_format": "auto",
                "modulation": "auto",
            }
        )
        checks.check(
            f"{name}: analysed without errors",
            report.get("errors") == [],
            f"errors={report.get('errors')}",
        )
        checks.check(
            f"{name}: auto IQ-format detection matched the known dtype",
            report.get("input", {}).get("iq_format_used") == entry["iq_format"],
            f"detected={report.get('input', {}).get('iq_format_used')} "
            f"expected={entry['iq_format']}",
        )
        # Neither real capture carries the project's sync word, so a detected
        # header would be a false alarm -- the exact bug this guards.
        correlation = report.get("correlation", {})
        checks.check(
            f"{name}: no false sync-word header claimed",
            correlation.get("detected") is False,
            f"detected={correlation.get('detected')} score={correlation.get('score')}",
        )
        # And no message may be invented from an unframed real signal.
        decoded = (report.get("payload") or {}).get("decoded") or {}
        checks.check(
            f"{name}: no payload invented from an unframed capture",
            decoded.get("available") is False,
            f"decoded={decoded.get('available')} scheme={decoded.get('scheme')}",
        )


def main() -> int:
    print("=== RF Analyzer MVP Verification ===")

    print("[1/5] Generating synthetic data...")
    run_command([sys.executable, "scripts/generate_test_data.py"])

    print("[2/5] Running pytest...")
    run_command([sys.executable, "-m", "pytest", "-q"])

    print("[3/5] Running MVP pipeline checks...")
    checks = Checks()
    for step, fn in (
        ("core pipeline", check_pipeline),
        ("coded captures (FEC + de-interleaving)", check_coded_captures),
        ("2-FSK symbol-period recovery", check_2fsk_symbol_recovery),
        ("payload honesty", check_payload_honesty),
        ("IQ-format auto-detection", check_auto_detection),
        ("batch export", check_batch_exports),
        ("web dashboard", check_dashboard),
        ("real-world captures", check_real_captures),
    ):
        try:
            fn(checks)
        except Exception as exc:  # explicit, never silent
            checks.check(f"{step} checks raised", False, f"{type(exc).__name__}: {exc}")

    print("[4/5] Verifying the report schema...")
    check_report_schema(checks)

    print("[5/5] Result")
    if checks.ok:
        print("MVP verification complete.")
        print("PASS")
        return 0

    print("MVP verification complete.")
    print("FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
