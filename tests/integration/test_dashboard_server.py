"""Integration tests for the zero-install dashboard HTTP server.

The server is started on an ephemeral localhost port in a daemon thread, so
these tests exercise the real request path (routing, query parsing, upload
handling, path-traversal rejection) without a browser or a fixed port.
"""

from __future__ import annotations

import importlib.util
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SERVER_PATH = ROOT / "scripts" / "serve_dashboard.py"


def _load_server_module():
    spec = importlib.util.spec_from_file_location("serve_dashboard", SERVER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def server():
    module = _load_server_module()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), module.DashboardHandler)
    httpd.daemon_threads = True
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield module, f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _get(url: str, timeout: float = 120.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.status, resp.read()


def _get_json(url: str, timeout: float = 120.0):
    status, body = _get(url, timeout=timeout)
    return status, json.loads(body)


def _post(url: str, data: bytes, timeout: float = 120.0):
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/octet-stream"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read())


# --------------------------------------------------------------------------- #
# static + metadata routes
# --------------------------------------------------------------------------- #
def test_root_serves_the_dashboard_page(server):
    _, base = server
    status, body = _get(base + "/")
    assert status == 200
    text = body.decode("utf-8")
    assert "<title>RF Signal Analysis Workbench</title>" in text
    # The page must be self-contained: no CDN, no external fetches.
    assert "http://" not in text.replace("http://127.0.0.1", "")
    assert "https://" not in text
    assert "<script src=" not in text
    assert "<link" not in text


def test_health_reports_the_page_is_present(server):
    _, base = server
    status, payload = _get_json(base + "/api/health")
    assert status == 200
    assert payload["ok"] is True
    assert payload["page"] is True


def test_samples_lists_bundled_captures(server):
    _, base = server
    status, payload = _get_json(base + "/api/samples")
    assert status == 200
    names = {c["name"] for c in payload["captures"]}
    assert "bpsk.iq" in names
    assert all(c["suffix"] in ("iq", "wav") for c in payload["captures"])


def test_unknown_route_returns_404_json(server):
    _, base = server
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get(base + "/nope")
    assert excinfo.value.code == 404
    assert json.loads(excinfo.value.read())["ok"] is False


# --------------------------------------------------------------------------- #
# analyse: bundled capture
# --------------------------------------------------------------------------- #
def test_analyze_bundled_capture_returns_report_and_plots(server):
    _, base = server
    url = f"{base}/api/analyze?name=bpsk.iq&sample_rate=100000&iq_format=complex64&modulation=auto"
    status, payload = _get_json(url)
    assert status == 200
    assert payload["ok"] is True
    assert payload["source"] == "sample"
    report = payload["report"]
    assert report["errors"] == []
    assert report["modulation"]["estimated_type"] == "BPSK"
    assert report["quality"]["applicable"] is True
    assert payload["plots"] is not None
    assert set(payload["plots"]) == {"constellation", "psd", "waterfall", "eye", "time"}
    assert payload["summary"]


# Correctness tests must not be decided by the wall-clock guard. The
# concatenated + diagonal decode needs ~2-4 s against the shipped 4 s budget
# (41 hypotheses), so on a loaded machine the default budget can expire before
# the search reaches the winner and turn a passing decode into a flaky
# failure. Same generous budget as test_coded_pipeline.GENEROUS_BUDGET; the
# shipped defaults still apply to every other dashboard request.
GENEROUS_DECODE_QUERY = "decode_max_bits=200000&decode_time_budget_s=120"


def test_analyze_recovers_a_coded_message_end_to_end(server):
    _, base = server
    url = (
        f"{base}/api/analyze?name=coded_concatenated.iq&sample_rate=100000"
        "&iq_format=auto&modulation=auto&sync_word=0x1ACFFC1D&decode=1"
        f"&{GENEROUS_DECODE_QUERY}"
    )
    _, payload = _get_json(url, timeout=180.0)
    decoded = payload["report"]["payload"]["decoded"]
    assert decoded["available"] is True
    assert decoded["crc_pass"] is True
    assert decoded["bytes"] > 0
    assert "Concatenated" in decoded["scheme"]


def test_analyze_frames_a_capture_when_no_sync_word_is_given(server):
    """Leaving the sync word blank must still work via the default probe."""
    _, base = server
    url = (
        f"{base}/api/analyze?name=coded_conv.iq&sample_rate=100000"
        "&iq_format=complex64&modulation=auto"
    )
    _, payload = _get_json(url, timeout=180.0)
    cor = payload["report"]["correlation"]
    assert cor["assumed"] is True
    assert cor["detected"] is True
    assert payload["report"]["payload"]["decoded"]["available"] is True


def test_analyze_requires_a_sample_rate_for_raw_iq(server):
    _, base = server
    _, payload = _get_json(f"{base}/api/analyze?name=bpsk.iq&iq_format=complex64")
    assert payload["report"]["errors"]
    assert any("sample_rate" in e for e in payload["report"]["errors"])


def test_analyze_missing_name_is_a_400(server):
    _, base = server
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get(base + "/api/analyze")
    assert excinfo.value.code == 400


def test_analyze_unknown_capture_is_a_404(server):
    _, base = server
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get(base + "/api/analyze?name=does_not_exist.iq&sample_rate=100000")
    assert excinfo.value.code == 404


# --------------------------------------------------------------------------- #
# path traversal
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name",
    [
        "../AGENTS.md",
        "..%2FAGENTS.md",
        "subdir/bpsk.iq",
        "/etc/passwd",
        "C:\\Windows\\win.ini",
    ],
)
def test_analyze_refuses_path_traversal(server, name):
    _, base = server
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get(f"{base}/api/analyze?name={name}&sample_rate=100000")
    assert excinfo.value.code in (400, 404)


def test_resolve_capture_only_reaches_discovered_files(server):
    module, _ = server
    assert module.resolve_capture("bpsk.iq") is not None
    assert module.resolve_capture("../AGENTS.md") is None
    assert module.resolve_capture("nope.iq") is None


# --------------------------------------------------------------------------- #
# analyse: upload
# --------------------------------------------------------------------------- #
def test_upload_analyzes_posted_bytes(server):
    _, base = server
    raw = (ROOT / "sample_data" / "bpsk.iq").read_bytes()
    status, payload = _post(
        f"{base}/api/analyze?name=uploaded.iq&sample_rate=100000&iq_format=complex64",
        raw,
    )
    assert status == 200
    assert payload["ok"] is True
    assert payload["source"] == "upload"
    assert payload["report"]["errors"] == []
    assert payload["report"]["input"]["file_name"] == "uploaded.iq"
    assert payload["report"]["modulation"]["estimated_type"] == "BPSK"


def test_upload_rejects_an_unsupported_suffix(server):
    _, base = server
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post(f"{base}/api/analyze?name=evil.exe&sample_rate=100000", b"\x00\x01\x02")
    assert excinfo.value.code == 400


def test_upload_rejects_an_empty_body(server):
    _, base = server
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post(f"{base}/api/analyze?name=empty.iq&sample_rate=100000", b"")
    assert excinfo.value.code == 400


def test_upload_over_the_size_limit_is_rejected_without_buffering(server):
    """A declared Content-Length above the cap must 413 before reading it."""
    module, base = server
    request = urllib.request.Request(
        f"{base}/api/analyze?name=big.iq&sample_rate=100000",
        data=b"\x00" * 16,
        headers={
            "Content-Type": "application/octet-stream",
            "Content-Length": str(module.MAX_UPLOAD_BYTES + 1),
        },
    )
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(request, timeout=30.0)
    assert excinfo.value.code == 413


# --------------------------------------------------------------------------- #
# query parsing
# --------------------------------------------------------------------------- #
def test_build_request_parses_every_parameter(server, sample_data_dir):
    module, _ = server
    request = module.build_request(
        {
            "sample_rate": ["48000"],
            "iq_format": ["int16"],
            "modulation": ["qpsk"],
            "sync_word": ["0xDEADBEEF"],
            "decode": ["0"],
            "frame_bits": ["512"],
        },
        sample_data_dir / "bpsk.iq",
    )
    assert request["sample_rate"] == 48000.0
    assert request["iq_format"] == "int16"
    assert request["modulation"] == "qpsk"
    assert request["sync_word"] == "0xDEADBEEF"
    assert request["decode"] is False
    assert request["frame_bits"] == 512


def test_build_request_omits_optional_values(server, sample_data_dir):
    module, _ = server
    request = module.build_request({}, sample_data_dir / "bpsk.iq")
    assert "sample_rate" not in request
    assert "sync_word" not in request
    assert "frame_bits" not in request
    assert request["decode"] is True
    assert request["iq_format"] == "auto"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1", True),
        ("true", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("off", False),
    ],
)
def test_bool_parsing(server, raw, expected):
    module, _ = server
    assert module._as_bool(raw) is expected


def test_float_parsing_handles_garbage(server):
    module, _ = server
    assert module._as_float("1000") == 1000.0
    assert module._as_float("abc") is None
    assert module._as_float("") is None
    assert module._as_float(None) is None


# --------------------------------------------------------------------------- #
# strict JSON over the wire
# --------------------------------------------------------------------------- #
def _strict_load(text: str):
    """Parse JSON the way a browser does: non-finite tokens are fatal."""

    def reject(token: str):
        raise ValueError(f"non-finite JSON token: {token}")

    return json.loads(text, parse_constant=reject)


@pytest.mark.parametrize("name", ["coded_rs.iq", "coded_ldpc.iq", "coded_uncoded.iq"])
def test_response_is_strict_json_for_noiseless_captures(server, name):
    """`Infinity` would make the browser's JSON.parse throw.

    These captures have a zero error vector, so their MER is unbounded; the
    server must emit `null`, never the invalid `Infinity` token.
    """
    _, base = server
    url = (
        f"{base}/api/analyze?name={name}&sample_rate=100000"
        "&iq_format=complex64&modulation=auto&sync_word=0x1ACFFC1D"
    )
    status, body = _get(url, timeout=180.0)
    assert status == 200
    text = body.decode("utf-8")
    assert "Infinity" not in text
    assert "NaN" not in text
    payload = _strict_load(text)
    assert payload["ok"] is True
    assert payload["report"]["quality"]["mer_db"] is None


def test_samples_response_is_strict_json(server):
    _, base = server
    _, body = _get(base + "/api/samples")
    assert "Infinity" not in body.decode("utf-8")
    assert _strict_load(body.decode("utf-8"))["ok"] is True
