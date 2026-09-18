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
``modulation``, ``sync_word``, ``decode``, ``frame_bits``.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rf_analyzer.core.report import json_safe
from rf_analyzer.dashboard import dashboard_payload, load_capture
from rf_analyzer.pipeline import analyze_file

PAGE_PATH = ROOT / "src" / "rf_analyzer" / "web" / "dashboard.html"
CAPTURE_DIRS = (ROOT / "sample_data", ROOT / "real_data")
SUPPORTED_SUFFIXES = (".iq", ".wav")
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB ceiling
DEFAULT_PORT = 8765


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
        return float(raw)
    except (TypeError, ValueError):
        return None


def build_request(params: dict[str, list[str]], file_path: Path) -> dict:
    """Translate query params into a ``pipeline.analyze_file`` request."""

    def first(key: str) -> str | None:
        values = params.get(key)
        return values[0] if values else None

    request: dict = {
        "file_path": str(file_path),
        "iq_format": (first("iq_format") or "auto").strip(),
        "modulation": (first("modulation") or "auto").strip(),
        "decode": _as_bool(first("decode"), True),
    }
    rate = _as_float(first("sample_rate"))
    if rate is not None:
        request["sample_rate"] = rate
    sync_word = (first("sync_word") or "").strip()
    if sync_word:
        request["sync_word"] = sync_word
    frame_bits = _as_float(first("frame_bits"))
    if frame_bits is not None and frame_bits > 0:
        request["frame_bits"] = int(frame_bits)
    return request


def analyse(file_path: Path, request: dict) -> dict:
    """Run the pipeline and attach plot data, never raising."""
    try:
        report = analyze_file(request)
    except Exception as exc:  # missing sample_rate for .iq lands here
        return {
            "report": {"errors": [f"{type(exc).__name__}: {exc}"], "warnings": []},
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
            self._send_error_json(
                413,
                f"upload too large ({length} bytes); limit is {MAX_UPLOAD_BYTES} bytes",
            )
            return
        self._handle_analyze(parse_qs(parsed.query), upload=self.rfile.read(length))

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
