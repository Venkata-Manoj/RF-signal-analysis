#!/usr/bin/env python
"""Zero-install web dashboard for the RF Signal Analysis Workbench.

Serves a self-contained browser UI over Python's standard-library
``http.server``. There is nothing extra to install: no Flask, no npm, no
CDN -- the page is one local HTML file and every plot is drawn with the
browser's own canvas.

Usage (Windows PowerShell / cmd / bash all work)::

    python scripts/serve_dashboard.py                 # http://127.0.0.1:8765
    python scripts/serve_dashboard.py --port 9000
    python scripts/serve_dashboard.py --open          # open the browser for you

The server binds to 127.0.0.1 only and never talks to the network: it is
a local convenience wrapper around ``pipeline.analyze_file``, which stays
the single source of truth for the analysis.

Routes
------
``GET  /``                the dashboard page
``GET  /api/health``      liveness probe
``GET  /api/samples``     captures bundled in ``sample_data/`` and ``real_data/``
``GET  /api/analyze``     analyse a bundled capture by name (query params)
``POST /api/analyze``     analyse uploaded bytes (raw body + query params)

Query params for ``/api/analyze``: ``name``, ``sample_rate``, ``iq_format``,
``modulation``, ``sync_word``, ``decode``, ``frame_bits``,
``decode_max_bits``, ``decode_time_budget_s``, ``auto_sample_rate``,
``auto_sync``. The two ``auto_*`` flags are additive: an explicit ``sample_rate``
or ``sync_word`` in the same request always wins, and every automatically chosen
value is reported as assumed with a warning. The two ``decode_*`` params are
the per-request overrides for the pipeline's bounded decode search
(``DECODE_MAX_BITS`` / ``DECODE_TIME_BUDGET_S``); omitted means the shipped
defaults, which keep interactive use inside the wall-clock budget.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tempfile
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rf_analyzer.config import DECODE_MAX_BITS
from rf_analyzer.core.report import json_safe
from rf_analyzer.dashboard import dashboard_payload, load_capture
from rf_analyzer.pipeline import _error_report, analyze_file

PAGE_PATH = ROOT / "src" / "rf_analyzer" / "web" / "dashboard.html"
CAPTURE_DIRS = (ROOT / "sample_data", ROOT / "real_data")
SUPPORTED_SUFFIXES = (".iq", ".wav")
# Aligned with io.MAX_FILE_BYTES / info.md NFR-01 (100 MB): anything larger is
# rejected before buffering so neither the server nor the pipeline ever holds
# body + temp + loaded samples (triple memory) for an oversize capture.
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB ceiling (NFR-01)
#: Streaming chunk size for POST uploads (avoids one big read into RAM).
UPLOAD_CHUNK_BYTES = 64 * 1024
#: Total wall-clock budget for receiving one upload body.
UPLOAD_TIMEOUT_S = 60.0
DEFAULT_PORT = 8765

#: Hard ceiling for an explicit sample rate (Hz). Raw IQ carries no rate
#: metadata, so an unbounded value only feeds nonsense into the DSP; the
#: pipeline itself rejects non-positive / non-finite rates.
MAX_SAMPLE_RATE_HZ = 1e9
#: Longest accepted sync word, in hex digits (256 hex chars = 1024 bits).
#: Bounds the sliding-correlation work and rejects accidental pastes.
MAX_SYNC_HEX_CHARS = 256
#: Upper bound for the caller-supplied coded-region hint. Four times the
#: decode-search prefix cap is ample for any real frame while keeping a
#: crafted value from forcing an unbounded search.
MAX_FRAME_BITS = DECODE_MAX_BITS * 4
#: Upper bounds for the caller-supplied decode-search overrides. The caps sit
#: far above the generous budgets correctness tests use (200_000 bits / 120 s,
#: see tests/integration/test_coded_pipeline.py) while keeping a crafted value
#: from forcing an unbounded search or hanging a server thread.
MAX_DECODE_MAX_BITS = 1_000_000
MAX_DECODE_TIME_BUDGET_S = 600.0

#: Query-param allowlists (compared case-insensitively after stripping).
#: Anything else is a 400 -- the pipeline's BPSK fallback is for direct
#: callers, not for the HTTP boundary.
ALLOWED_IQ_FORMATS = frozenset(
    {"auto", "complex64", "int16", "uint8", "int8", "cf32", "ci16", "cu8", "ci8"}
)
ALLOWED_MODULATIONS = frozenset(
    {
        "auto",
        "bpsk",
        "qpsk",
        "2-fsk",
        "2fsk",
        "fsk",
        "4-fsk",
        "4fsk",
        "8psk",
        "8-psk",
        "16-qam",
        "qam16",
        "qam",
        "64-qam",
        "qam64",
        "64qam",
    }
)
_SYNC_HEX_RE = re.compile(r"^(?:0[xX])?[0-9a-fA-F]+$")


def discover_captures() -> list[dict]:
    """List bundled captures that the page can analyse without an upload."""
    found: list[dict] = []
    for folder in CAPTURE_DIRS:
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            found.append(
                {
                    "name": path.name,
                    "folder": folder.name,
                    "size_bytes": path.stat().st_size,
                    "suffix": path.suffix.lower().lstrip("."),
                }
            )
    return found


def resolve_capture(name: str) -> Path | None:
    """Map a client-supplied name to a bundled file, or None.

    Only files that :func:`discover_captures` already found are reachable,
    so a crafted ``name`` cannot walk out of the capture directories.
    """
    for entry in discover_captures():
        if entry["name"] == name:
            candidate = (ROOT / entry["folder"] / entry["name"]).resolve()
            for folder in CAPTURE_DIRS:
                try:
                    candidate.relative_to(folder.resolve())
                except ValueError:
                    continue
                if candidate.is_file():
                    return candidate
    return None


def _as_bool(raw: str | None, default: bool = True) -> bool:
    if raw is None:
        return default
    return str(raw).strip().lower() not in ("0", "false", "no", "off", "")


def _as_float(raw: str | None) -> float | None:
    if raw is None or str(raw).strip() == "":
        return None
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _validated_iq_format(raw: str | None) -> str:
    value = (raw or "auto").strip() or "auto"
    if value.lower() not in ALLOWED_IQ_FORMATS:
        allowed = sorted(ALLOWED_IQ_FORMATS)
        raise ValueError(f"invalid 'iq_format' {raw!r}; expected one of {allowed}")
    return value


def _validated_modulation(raw: str | None) -> str:
    value = (raw or "auto").strip() or "auto"
    if value.lower() not in ALLOWED_MODULATIONS:
        allowed = sorted(ALLOWED_MODULATIONS)
        raise ValueError(f"invalid 'modulation' {raw!r}; expected one of {allowed}")
    return value


def _validated_sync_word(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    hex_part = value[2:] if value.lower().startswith("0x") else value
    if (
        not hex_part
        or len(hex_part) > MAX_SYNC_HEX_CHARS
        or not _SYNC_HEX_RE.match(value)
    ):
        raise ValueError(
            f"invalid 'sync_word' {raw!r}: must be 1-{MAX_SYNC_HEX_CHARS} "
            "hex characters (optional 0x prefix)"
        )
    return value


def _validated_sample_rate(raw: str | None) -> float | None:
    if raw is None or str(raw).strip() == "":
        return None
    text = str(raw).strip()
    try:
        rate = float(text)
    except (TypeError, ValueError):
        raise ValueError(f"invalid 'sample_rate' {raw!r}: must be a number") from None
    if not math.isfinite(rate) or not (0 < rate <= MAX_SAMPLE_RATE_HZ):
        raise ValueError(
            f"invalid 'sample_rate' {raw!r}: must satisfy "
            f"0 < sample_rate <= {MAX_SAMPLE_RATE_HZ:.0f} and be finite"
        )
    return rate


def _validated_frame_bits(raw: str | None) -> int | None:
    if raw is None or str(raw).strip() == "":
        return None
    text = str(raw).strip()
    try:
        parsed = float(text)
    except (TypeError, ValueError):
        raise ValueError(f"invalid 'frame_bits' {raw!r}: must be a number") from None
    if not math.isfinite(parsed):
        raise ValueError(f"invalid 'frame_bits' {raw!r}: must be finite")
    if parsed <= 0:
        return None
    if parsed > MAX_FRAME_BITS:
        raise ValueError(
            f"invalid 'frame_bits' {raw!r}: must be <= {MAX_FRAME_BITS} "
            f"(DECODE_MAX_BITS*4)"
        )
    return int(parsed)


def _validated_decode_max_bits(raw: str | None) -> int | None:
    """Optional ``decode_max_bits`` override for the pipeline search prefix."""
    if raw is None or str(raw).strip() == "":
        return None
    text = str(raw).strip()
    try:
        parsed = float(text)
    except (TypeError, ValueError):
        raise ValueError(
            f"invalid 'decode_max_bits' {raw!r}: must be a number"
        ) from None
    if not math.isfinite(parsed):
        raise ValueError(f"invalid 'decode_max_bits' {raw!r}: must be finite")
    if parsed <= 0:
        return None
    if parsed > MAX_DECODE_MAX_BITS:
        raise ValueError(
            f"invalid 'decode_max_bits' {raw!r}: must be <= {MAX_DECODE_MAX_BITS}"
        )
    return int(parsed)


def _validated_decode_time_budget_s(raw: str | None) -> float | None:
    """Optional ``decode_time_budget_s`` override for the search wall-clock."""
    if raw is None or str(raw).strip() == "":
        return None
    text = str(raw).strip()
    try:
        parsed = float(text)
    except (TypeError, ValueError):
        raise ValueError(
            f"invalid 'decode_time_budget_s' {raw!r}: must be a number"
        ) from None
    if not math.isfinite(parsed):
        raise ValueError(f"invalid 'decode_time_budget_s' {raw!r}: must be finite")
    if parsed <= 0:
        return None
    if parsed > MAX_DECODE_TIME_BUDGET_S:
        raise ValueError(
            f"invalid 'decode_time_budget_s' {raw!r}: must be <= "
            f"{MAX_DECODE_TIME_BUDGET_S:.0f}"
        )
    return float(parsed)


def build_request(params: dict[str, list[str]], file_path: Path) -> dict:
    """Translate query params into a ``pipeline.analyze_file`` request.

    Raises:
        ValueError: on any invalid ``iq_format`` / ``modulation`` /
            ``sync_word`` / ``sample_rate`` / ``frame_bits`` /
            ``decode_max_bits`` / ``decode_time_budget_s`` value. The HTTP
            handler turns this into a 400 so bad input never reaches the DSP.
    """

    def first(key: str) -> str | None:
        values = params.get(key)
        return values[0] if values else None

    request: dict = {
        "file_path": str(file_path),
        "iq_format": _validated_iq_format(first("iq_format")),
        "modulation": _validated_modulation(first("modulation")),
        "decode": _as_bool(first("decode"), True),
        # Both automation flags are additive and never override an explicit
        # sample_rate / sync_word supplied in the same request.
        "auto_sample_rate": _as_bool(first("auto_sample_rate"), False),
        "auto_sync": _as_bool(first("auto_sync"), False),
    }
    rate = _validated_sample_rate(first("sample_rate"))
    if rate is not None:
        request["sample_rate"] = rate
    sync_word = _validated_sync_word(first("sync_word"))
    if sync_word:
        request["sync_word"] = sync_word
    frame_bits = _validated_frame_bits(first("frame_bits"))
    if frame_bits is not None:
        request["frame_bits"] = frame_bits
    decode_max_bits = _validated_decode_max_bits(first("decode_max_bits"))
    if decode_max_bits is not None:
        request["decode_max_bits"] = decode_max_bits
    decode_budget = _validated_decode_time_budget_s(first("decode_time_budget_s"))
    if decode_budget is not None:
        request["decode_time_budget_s"] = decode_budget
    return request


def analyse(file_path: Path, request: dict) -> dict:
    """Run the pipeline and attach plot data, never raising."""
    try:
        report = analyze_file(request)
    except Exception as exc:  # missing sample_rate for .iq lands here
        # Same shape as every other error report (judgments available=False)
        # so the browser never branches on a missing block.
        report = _error_report(file_path.name, [], f"{type(exc).__name__}: {exc}")
        return {
            "report": report,
            "summary": [],
            "plots": None,
            "plot_error": None,
        }

    samples = None
    if not report.get("errors"):
        try:
            samples, rate = load_capture(
                str(file_path),
                sample_rate=request.get("sample_rate"),
                iq_format=str(
                    report.get("input", {}).get("iq_format_used") or "complex64"
                ),
            )
            report.setdefault("input", {})["sample_rate"] = rate
        except Exception as exc:
            samples = None
            report.setdefault("warnings", []).append(f"Plots unavailable: {exc}")

    return dashboard_payload(report, samples)


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "RFWorkbenchDashboard/1.0"
    protocol_version = "HTTP/1.1"

    # ---- helpers -------------------------------------------------------
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        # json_safe + allow_nan=False: the browser's JSON.parse rejects the
        # `Infinity`/`NaN` tokens Python emits by default, so a non-finite
        # value here would surface as an opaque "network error" in the UI.
        body = json.dumps(json_safe(payload), default=str, allow_nan=False).encode(
            "utf-8"
        )
        self._send(status, body, "application/json; charset=utf-8")

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json({"ok": False, "error": message}, status=status)

    def log_message(self, fmt: str, *args) -> None:  # quieter default logging
        sys.stderr.write(f"[dashboard] {self.address_string()} - {fmt % args}\n")

    # ---- routes --------------------------------------------------------
    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path.rstrip("/") or "/"
        params = parse_qs(parsed.query)

        if route in ("/", "/index.html"):
            if not PAGE_PATH.is_file():
                self._send_error_json(500, f"dashboard page missing: {PAGE_PATH}")
                return
            self._send(200, PAGE_PATH.read_bytes(), "text/html; charset=utf-8")
            return

        if route == "/api/health":
            self._send_json({"ok": True, "page": PAGE_PATH.is_file()})
            return

        if route == "/api/samples":
            self._send_json({"ok": True, "captures": discover_captures()})
            return

        if route == "/api/analyze":
            self._handle_analyze(params, upload=None)
            return

        self._send_error_json(404, f"unknown route: {route}")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path.rstrip("/") or "/"
        if route != "/api/analyze":
            self._send_error_json(404, f"unknown route: {route}")
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._send_error_json(400, "invalid Content-Length")
            return
        if length <= 0:
            self._send_error_json(400, "empty upload body")
            return
        if length > MAX_UPLOAD_BYTES:
            # Reject before reading a single body byte; close so a lying
            # Content-Length cannot desync the keep-alive connection.
            self.close_connection = True
            self._send_error_json(
                413,
                f"upload too large ({length} bytes); limit is {MAX_UPLOAD_BYTES} bytes",
            )
            return
        params = parse_qs(parsed.query)
        # Validate the suffix before buffering the body: an unsupported type
        # is rejected without ever holding its bytes in memory.
        early_name = (params.get("name") or ["upload.iq"])[0]
        if Path(early_name).suffix.lower() not in SUPPORTED_SUFFIXES:
            self.close_connection = True
            self._send_error_json(
                400, f"unsupported upload '{early_name}'; expected .iq or .wav"
            )
            # Drain nothing: the connection is closed, so the unread body
            # cannot desync a subsequent request.
            return
        self._handle_upload_streamed(params, length)

    # ---- streamed upload path ------------------------------------------
    def _handle_upload_streamed(
        self, params: dict[str, list[str]], length: int
    ) -> None:
        """Stream exactly ``length`` body bytes to temp, then analyse.

        Chunked write with a hard cap, Content-Length/actual-bytes agreement
        and a total timeout. Only one chunk (~64 KB) is ever in RAM, so an
        upload costs temp + loaded samples -- never body + temp + load.
        """
        name = (params.get("name") or ["upload.iq"])[0]
        try:
            with tempfile.TemporaryDirectory(prefix="rf_dash_") as tmp:
                target = Path(tmp) / Path(name).name
                written = 0
                deadline = time.monotonic() + UPLOAD_TIMEOUT_S
                # Bound a stalled single read as well as a slow body: without
                # a socket timeout one hanging recv would sit past the
                # deadline without ever reaching the check below.
                prev_timeout = self.request.gettimeout()
                try:
                    self.request.settimeout(UPLOAD_TIMEOUT_S)
                    with open(target, "wb") as fh:
                        remaining = int(length)
                        while remaining > 0:
                            if time.monotonic() > deadline:
                                self.close_connection = True
                                self._send_error_json(
                                    408,
                                    f"upload timed out after {UPLOAD_TIMEOUT_S:.0f}s "
                                    f"({written}/{length} bytes received)",
                                )
                                return
                            try:
                                chunk = self.rfile.read(
                                    min(UPLOAD_CHUNK_BYTES, remaining)
                                )
                            except (TimeoutError, OSError) as exc:
                                self.close_connection = True
                                self._send_error_json(
                                    408, f"upload read timed out: {exc}"
                                )
                                return
                            if not chunk:
                                # Client went away early: actual != declared.
                                self.close_connection = True
                                self._send_error_json(
                                    400,
                                    f"upload truncated "
                                    f"({written}/{length} bytes received); "
                                    "Content-Length does not match actual bytes",
                                )
                                return
                            written += len(chunk)
                            if written > MAX_UPLOAD_BYTES:
                                self.close_connection = True
                                self._send_error_json(
                                    413,
                                    f"upload too large ({written} bytes); "
                                    f"limit is {MAX_UPLOAD_BYTES} bytes",
                                )
                                return
                            fh.write(chunk)
                            remaining -= len(chunk)
                finally:
                    try:
                        self.request.settimeout(prev_timeout)
                    except OSError:
                        pass
                if written != int(length):
                    self.close_connection = True
                    self._send_error_json(
                        400,
                        f"upload size mismatch "
                        f"({written}/{length} bytes); "
                        "Content-Length does not match actual bytes",
                    )
                    return
                request = build_request(params, target)
                payload = analyse(target, request)
                payload["ok"] = True
                payload["source"] = "upload"
                self._send_json(payload)
                return
        except BrokenPipeError:
            pass  # the browser navigated away mid-response
        except Exception as exc:
            try:
                self._send_error_json(500, f"{type(exc).__name__}: {exc}")
            except BrokenPipeError:
                pass

    # ---- shared analyse path -------------------------------------------
    def _handle_analyze(
        self, params: dict[str, list[str]], upload: bytes | None
    ) -> None:
        try:
            if upload is not None:
                name = (params.get("name") or ["upload.iq"])[0]
                suffix = Path(name).suffix.lower()
                if suffix not in SUPPORTED_SUFFIXES:
                    self._send_error_json(
                        400, f"unsupported upload '{name}'; expected .iq or .wav"
                    )
                    return
                if len(upload) > MAX_UPLOAD_BYTES:
                    self._send_error_json(
                        413,
                        f"upload too large ({len(upload)} bytes); "
                        f"limit is {MAX_UPLOAD_BYTES} bytes",
                    )
                    return
                with tempfile.TemporaryDirectory(prefix="rf_dash_") as tmp:
                    target = Path(tmp) / Path(name).name
                    target.write_bytes(upload)
                    request = build_request(params, target)
                    payload = analyse(target, request)
                payload["ok"] = True
                payload["source"] = "upload"
                self._send_json(payload)
                return

            name = (params.get("name") or [""])[0]
            if not name:
                self._send_error_json(400, "missing 'name' (or POST a file body)")
                return
            target = resolve_capture(name)
            if target is None:
                self._send_error_json(404, f"capture not found: {name}")
                return
            request = build_request(params, target)
            payload = analyse(target, request)
            payload["ok"] = True
            payload["source"] = "sample"
            self._send_json(payload)
        except BrokenPipeError:
            pass  # the browser navigated away mid-response
        except ValueError as exc:
            # Input validation from build_request: bad query params are the
            # caller's fault, never a server crash.
            self._send_error_json(400, str(exc))
        except Exception as exc:
            self._send_error_json(500, f"{type(exc).__name__}: {exc}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="port (default: 8765)"
    )
    parser.add_argument(
        "--open", action="store_true", help="open the dashboard in a browser"
    )
    args = parser.parse_args(argv)

    if not PAGE_PATH.is_file():
        print(f"ERROR: dashboard page not found at {PAGE_PATH}", file=sys.stderr)
        return 1

    httpd = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    httpd.daemon_threads = True
    url = f"http://{args.host}:{args.port}/"
    captures = discover_captures()
    print("RF Signal Analysis Workbench -- web dashboard")
    print(f"  serving : {url}")
    print(f"  page    : {PAGE_PATH}")
    print(f"  captures: {len(captures)} bundled in sample_data/ + real_data/")
    print("  press Ctrl+C to stop")
    if args.open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
