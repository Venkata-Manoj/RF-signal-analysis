"""Input-validation tests for the dashboard HTTP boundary + pipeline rate guard.

Covers the hardening in ``scripts/serve_dashboard.py`` (allowlisted
``iq_format`` / ``modulation``, hex-only ``sync_word`` capped at 256 hex
chars, ``0 < sample_rate <= 1e9`` finite, ``frame_bits <= DECODE_MAX_BITS*4``)
and the ``sample_rate > 0`` finite guard in ``pipeline.analyze_file``.

Invalid query params must fail fast with a 400 (never reach the DSP);
an invalid explicit rate to the pipeline must return an error report
(never raise, never run the DSP).
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

from rf_analyzer.config import DECODE_MAX_BITS
from rf_analyzer.pipeline import analyze_file

ROOT = Path(__file__).resolve().parents[2]
SERVER_PATH = ROOT / "scripts" / "serve_dashboard.py"
MAX_FRAME_BITS = DECODE_MAX_BITS * 4


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


@pytest.fixture(scope="module")
def dash():
    return _load_server_module()


def _params(**kwargs: str) -> dict[str, list[str]]:
    return {key: [value] for key, value in kwargs.items()}


# --------------------------------------------------------------------------- #
# iq_format allowlist
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value",
    ["auto", "complex64", "int16", "uint8", "int8", "cf32", "ci16", "cu8", "ci8"],
)
def test_build_request_accepts_iq_format_allowlist(dash, sample_data_dir, value: str):
    request = dash.build_request({"iq_format": [value]}, sample_data_dir / "bpsk.iq")
    assert request["iq_format"] == value


@pytest.mark.parametrize("value", ["AUTO", "Complex64", " INT16 ", "  auto  "])
def test_build_request_iq_format_is_case_and_space_insensitive(
    dash, sample_data_dir, value: str
):
    request = dash.build_request({"iq_format": [value]}, sample_data_dir / "bpsk.iq")
    assert request["iq_format"].strip().lower() == value.strip().lower()


@pytest.mark.parametrize("value", ["float32", "evil", "complex128", "text"])
def test_build_request_rejects_bad_iq_format(dash, sample_data_dir, value: str):
    with pytest.raises(ValueError, match="iq_format"):
        dash.build_request({"iq_format": [value]}, sample_data_dir / "bpsk.iq")


# --------------------------------------------------------------------------- #
# modulation allowlist
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value",
    [
        "auto",
        "BPSK",
        "QPSK",
        "2-FSK",
        "2FSK",
        "FSK",
        "4-FSK",
        "4FSK",
        "8PSK",
        "8-PSK",
        "16-QAM",
        "QAM16",
        "QAM",
        "64-QAM",
        "QAM64",
        "64QAM",
    ],
)
def test_build_request_accepts_modulation_allowlist(dash, sample_data_dir, value: str):
    request = dash.build_request({"modulation": [value]}, sample_data_dir / "bpsk.iq")
    assert request["modulation"] == value


@pytest.mark.parametrize("value", ["qpsk", " bpsk ", "AUTO", "16-qam"])
def test_build_request_modulation_is_case_and_space_insensitive(
    dash, sample_data_dir, value: str
):
    request = dash.build_request({"modulation": [value]}, sample_data_dir / "bpsk.iq")
    assert request["modulation"].strip().lower() == value.strip().lower()


@pytest.mark.parametrize("value", ["FM", "AM", "USB", "evil", "QAM256", "OFDM"])
def test_build_request_rejects_bad_modulation(dash, sample_data_dir, value: str):
    with pytest.raises(ValueError, match="modulation"):
        dash.build_request({"modulation": [value]}, sample_data_dir / "bpsk.iq")


# --------------------------------------------------------------------------- #
# sync_word: hex-only, <= 256 hex chars
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value", ["0x1ACFFC1D", "1ACFFC1D", "0xdeadBEEF", "AB", "0x00", "ff"]
)
def test_build_request_accepts_valid_sync_word(dash, sample_data_dir, value: str):
    request = dash.build_request({"sync_word": [value]}, sample_data_dir / "bpsk.iq")
    assert request["sync_word"] == value.strip()


def test_build_request_sync_word_cap_is_256_hex_chars(dash, sample_data_dir):
    ok_plain = "A" * 256
    ok_prefixed = "0x" + "B" * 256
    assert (
        dash.build_request({"sync_word": [ok_plain]}, sample_data_dir / "bpsk.iq")[
            "sync_word"
        ]
        == ok_plain
    )
    assert (
        dash.build_request({"sync_word": [ok_prefixed]}, sample_data_dir / "bpsk.iq")[
            "sync_word"
        ]
        == ok_prefixed
    )
    with pytest.raises(ValueError, match="sync_word"):
        dash.build_request({"sync_word": ["A" * 257]}, sample_data_dir / "bpsk.iq")
    with pytest.raises(ValueError, match="sync_word"):
        dash.build_request(
            {"sync_word": ["0x" + "B" * 257]}, sample_data_dir / "bpsk.iq"
        )


@pytest.mark.parametrize(
    "value", ["0xZZZZ", "hello!", "0x", "", "   ", "0x1G", "12 34", "0x_12"]
)
def test_build_request_rejects_non_hex_sync_word(dash, sample_data_dir, value: str):
    if value.strip() == "":
        # Blank means "no explicit sync" (auto path), not an error.
        request = dash.build_request(
            {"sync_word": [value]}, sample_data_dir / "bpsk.iq"
        )
        assert "sync_word" not in request
    else:
        with pytest.raises(ValueError, match="sync_word"):
            dash.build_request({"sync_word": [value]}, sample_data_dir / "bpsk.iq")


def test_build_request_omits_blank_sync_word(dash, sample_data_dir):
    assert "sync_word" not in dash.build_request({}, sample_data_dir / "bpsk.iq")
    assert "sync_word" not in dash.build_request(
        {"sync_word": ["   "]}, sample_data_dir / "bpsk.iq"
    )


# --------------------------------------------------------------------------- #
# sample_rate: 0 < rate <= 1e9, finite
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value", ["1", "48000", "100000", "1000000000", "1e9"])
def test_build_request_accepts_valid_sample_rate(dash, sample_data_dir, value: str):
    request = dash.build_request({"sample_rate": [value]}, sample_data_dir / "bpsk.iq")
    assert request["sample_rate"] == pytest.approx(float(value))


@pytest.mark.parametrize(
    "value",
    ["0", "-1", "-100", "0.0", "nan", "NaN", "inf", "-inf", "Infinity", "abc", ""],
)
def test_build_request_rejects_bad_sample_rate(dash, sample_data_dir, value: str):
    if value.strip() == "":
        request = dash.build_request(
            {"sample_rate": [value]}, sample_data_dir / "bpsk.iq"
        )
        assert "sample_rate" not in request
    else:
        with pytest.raises(ValueError, match="sample_rate"):
            dash.build_request({"sample_rate": [value]}, sample_data_dir / "bpsk.iq")


def test_build_request_rejects_sample_rate_above_1e9(dash, sample_data_dir):
    for value in ["1000000001", "1e12", "2000000000"]:
        with pytest.raises(ValueError, match="sample_rate"):
            dash.build_request({"sample_rate": [value]}, sample_data_dir / "bpsk.iq")


def test_as_float_rejects_nonfinite(server):
    module, _ = server
    assert module._as_float("1000") == 1000.0
    assert module._as_float("abc") is None
    assert module._as_float("") is None
    assert module._as_float(None) is None
    assert module._as_float("nan") is None
    assert module._as_float("NaN") is None
    assert module._as_float("inf") is None
    assert module._as_float("-inf") is None
    assert module._as_float("Infinity") is None


# --------------------------------------------------------------------------- #
# frame_bits: <= DECODE_MAX_BITS * 4
# --------------------------------------------------------------------------- #
def test_build_request_accepts_valid_frame_bits(dash, sample_data_dir):
    request = dash.build_request({"frame_bits": ["512"]}, sample_data_dir / "bpsk.iq")
    assert request["frame_bits"] == 512
    request = dash.build_request(
        {"frame_bits": [str(MAX_FRAME_BITS)]}, sample_data_dir / "bpsk.iq"
    )
    assert request["frame_bits"] == MAX_FRAME_BITS


def test_build_request_rejects_frame_bits_above_cap(dash, sample_data_dir):
    with pytest.raises(ValueError, match="frame_bits"):
        dash.build_request(
            {"frame_bits": [str(MAX_FRAME_BITS + 1)]}, sample_data_dir / "bpsk.iq"
        )
    with pytest.raises(ValueError, match="frame_bits"):
        dash.build_request({"frame_bits": ["1000000000"]}, sample_data_dir / "bpsk.iq")


@pytest.mark.parametrize("value", ["abc", "nan", "inf", "-inf"])
def test_build_request_rejects_garbage_frame_bits(dash, sample_data_dir, value: str):
    with pytest.raises(ValueError, match="frame_bits"):
        dash.build_request({"frame_bits": [value]}, sample_data_dir / "bpsk.iq")


@pytest.mark.parametrize("value", ["0", "-5", "0.0", "", "   "])
def test_build_request_nonpositive_frame_bits_means_auto(
    dash, sample_data_dir, value: str
):
    request = dash.build_request({"frame_bits": [value]}, sample_data_dir / "bpsk.iq")
    assert "frame_bits" not in request


# --------------------------------------------------------------------------- #
# HTTP: invalid query params are 400s, never DSP crashes
# --------------------------------------------------------------------------- #
def _get_status(url: str) -> int:
    try:
        with urllib.request.urlopen(url, timeout=30.0):
            return 200
    except urllib.error.HTTPError as exc:
        return int(exc.code)


@pytest.mark.parametrize(
    "query",
    [
        "iq_format=evil",
        "iq_format=float32",
        "modulation=FM",
        "modulation=evil",
        "sync_word=0xZZZZ",
        "sync_word=hello!",
        f"sync_word={'A' * 257}",
        "sample_rate=0",
        "sample_rate=-1",
        "sample_rate=nan",
        "sample_rate=inf",
        "sample_rate=-inf",
        "sample_rate=abc",
        "sample_rate=2000000000",
        "frame_bits=abc",
        "frame_bits=nan",
        f"frame_bits={MAX_FRAME_BITS + 1}",
    ],
)
def test_analyze_rejects_invalid_params_with_400(server, query: str):
    _, base = server
    status = _get_status(f"{base}/api/analyze?name=bpsk.iq&{query}&sample_rate=100000")
    # sample_rate cases carry their own bad value; drop the good default.
    if query.startswith("sample_rate="):
        status = _get_status(f"{base}/api/analyze?name=bpsk.iq&{query}")
    assert status == 400


def test_analyze_error_body_is_ok_false(server):
    _, base = server
    url = f"{base}/api/analyze?name=bpsk.iq&sample_rate=0"
    try:
        with urllib.request.urlopen(url, timeout=30.0):
            pytest.fail("expected HTTP 400")
    except urllib.error.HTTPError as exc:
        assert exc.code == 400
        payload = json.loads(exc.read())
        assert payload["ok"] is False
        assert "sample_rate" in payload["error"]


def test_analyze_accepts_valid_params_still_200(server):
    _, base = server
    url = (
        f"{base}/api/analyze?name=bpsk.iq&sample_rate=100000"
        "&iq_format=complex64&modulation=BPSK&sync_word=0x1ACFFC1D&frame_bits=512"
    )
    with urllib.request.urlopen(url, timeout=120.0) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read())
        assert payload["ok"] is True
        assert payload["report"]["errors"] == []


# --------------------------------------------------------------------------- #
# pipeline: explicit sample_rate must be > 0 and finite
# --------------------------------------------------------------------------- #
def _iq_request(sample_data_dir: Path, **overrides) -> dict:
    request = {
        "file_path": str(sample_data_dir / "bpsk.iq"),
        "sample_rate": 100000,
        "iq_format": "complex64",
        "modulation": "BPSK",
    }
    request.update(overrides)
    return request


@pytest.mark.parametrize(
    "bad_rate", [0, 0.0, -1, -100.5, float("nan"), float("inf"), float("-inf")]
)
def test_pipeline_rejects_nonpositive_or_nonfinite_sample_rate(
    sample_data_dir, bad_rate
):
    iq_path = sample_data_dir / "bpsk.iq"
    if not iq_path.exists():
        pytest.skip(f"missing synthetic data: {iq_path}")
    report = analyze_file(_iq_request(sample_data_dir, sample_rate=bad_rate))
    assert report["errors"], f"expected an error report for sample_rate={bad_rate!r}"
    assert any("sample_rate" in str(e).lower() for e in report["errors"])


@pytest.mark.parametrize("bad_rate", ["abc", "", "not-a-number"])
def test_pipeline_rejects_non_numeric_sample_rate(sample_data_dir, bad_rate):
    iq_path = sample_data_dir / "bpsk.iq"
    if not iq_path.exists():
        pytest.skip(f"missing synthetic data: {iq_path}")
    report = analyze_file(_iq_request(sample_data_dir, sample_rate=bad_rate))
    assert report["errors"]
    assert any("sample_rate" in str(e).lower() for e in report["errors"])


def test_pipeline_explicit_bad_rate_wins_over_auto(sample_data_dir):
    """An explicit invalid rate must still fail when auto is also requested."""
    iq_path = sample_data_dir / "bpsk.iq"
    if not iq_path.exists():
        pytest.skip(f"missing synthetic data: {iq_path}")
    report = analyze_file(
        _iq_request(sample_data_dir, sample_rate=0, auto_sample_rate=True)
    )
    assert report["errors"]
    assert any("sample_rate" in str(e).lower() for e in report["errors"])


def test_pipeline_missing_rate_without_auto_still_errors(sample_data_dir):
    iq_path = sample_data_dir / "bpsk.iq"
    if not iq_path.exists():
        pytest.skip(f"missing synthetic data: {iq_path}")
    request = _iq_request(sample_data_dir)
    del request["sample_rate"]
    report = analyze_file(request)
    assert report["errors"]
    assert any("sample_rate" in str(e).lower() for e in report["errors"])


def test_pipeline_auto_sample_rate_still_works(sample_data_dir):
    """Keep auto_sample_rate/auto_sync behavior: missing rate + auto runs."""
    iq_path = sample_data_dir / "bpsk.iq"
    if not iq_path.exists():
        pytest.skip(f"missing synthetic data: {iq_path}")
    request = _iq_request(sample_data_dir, auto_sample_rate=True, auto_sync=True)
    del request["sample_rate"]
    report = analyze_file(request)
    # Auto is a feasibility floor, not a guarantee: it must run (no
    # sample_rate error), even if this particular capture warns.
    assert not any(
        "sample_rate is required" in str(e) for e in report.get("errors", [])
    )
    assert report["input"]["sample_rate"] is not None
    assert float(report["input"]["sample_rate"]) > 0
