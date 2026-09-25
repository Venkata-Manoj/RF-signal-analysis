"""The robust receiver is on the delivered path, with honest fallback.

``analyze_file`` demodulates the selected mode with ``core.receiver`` by
default (``receiver="auto"``) and records the outcome additively under
``demodulation.receiver``. ``receiver="naive"`` keeps the previous
phase-aligned slicers bit-for-bit. Whenever the receiver abstains
(``sps<=1``, unlocked loops, exceptions) the pipeline falls back to the
naive slicers, says so in the report plus a warning, and never reports a
decode the selected path did not produce (the decode search always runs on
the final bit stream).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from rf_analyzer.config import BURST_FRAME_SLACK_BITS
from rf_analyzer.core import waveform as wf
from rf_analyzer.core.framing import build_frame, modulate
from rf_analyzer.pipeline import analyze_file

SYNC_WORD = "0x1ACFFC1D"
SYNC_LEN = 32
BUDGET = {"decode_time_budget_s": 120.0, "decode_max_bits": 200_000}


def _request(path: Path, **overrides) -> dict:
    request = {
        "file_path": str(path),
        "sample_rate": 100000,
        "iq_format": "complex64",
        "modulation": "BPSK",
        "sync_word": SYNC_WORD,
        "decode": True,
        **BUDGET,
    }
    request.update(overrides)
    return request


def _symbol_spaced_burst(path: Path, *, pad: int = 400) -> bytes:
    """1-sps framed BPSK burst in noise (the naive path's home turf)."""
    message = b"Receiver fallback still decodes the naive capture."
    frame = build_frame(message, fec="conv", interleaver="block")
    core = modulate(frame["bits"], "BPSK", sample_rate=100000)
    rng = np.random.default_rng(7)
    lead = (rng.standard_normal(pad) + 1j * rng.standard_normal(pad)) * 0.05
    trail = (rng.standard_normal(pad) + 1j * rng.standard_normal(pad)) * 0.05
    np.concatenate([lead, core, trail]).astype(np.complex64).tofile(path)
    return message


def _oversampled_burst(path: Path, *, lead: int = 256, tail: int = 1024) -> bytes:
    """RRC-shaped BPSK burst at 8 samples/symbol inside a noise capture."""
    message = b"Coherent sps8 burst tail decode."
    frame = build_frame(message, fec="none", interleaver="none")
    symbols = wf.bits_to_symbols(frame["bits"], "BPSK")
    shaped = wf.upsample_and_pulse(symbols, sps=8)
    rng = np.random.default_rng(11)
    parts = []
    if lead:
        parts.append(
            (rng.standard_normal(lead) + 1j * rng.standard_normal(lead))
            / np.sqrt(2)
            * 0.05
        )
    parts.append(np.asarray(shaped, dtype=np.complex128))
    if tail:
        parts.append(
            (rng.standard_normal(tail) + 1j * rng.standard_normal(tail))
            / np.sqrt(2)
            * 0.05
        )
    np.concatenate(parts).astype(np.complex64).tofile(path)
    return message


def test_receiver_naive_request_keeps_the_old_path(tmp_path):
    path = tmp_path / "naive.iq"
    message = _symbol_spaced_burst(path)
    report = analyze_file(_request(path, receiver="naive"))
    receiver = report["demodulation"]["receiver"]
    assert receiver["requested"] == "naive"
    assert receiver["path"] == "naive"
    assert receiver["locked"] is None
    assert report["demodulation"]["mode"] == "BPSK"
    assert report["correlation"]["detected"] is True
    decoded = report["payload"]["decoded"]
    assert decoded["available"] is True
    assert bytes.fromhex(decoded["hex"]) == message


def test_receiver_auto_falls_back_honestly_on_symbol_spaced(tmp_path):
    """1 sample/symbol has no oversampling to recover: abstain, fallback, decode."""
    path = tmp_path / "fallback.iq"
    message = _symbol_spaced_burst(path)
    report = analyze_file(_request(path))  # receiver defaults to auto
    receiver = report["demodulation"]["receiver"]
    assert receiver["requested"] == "auto"
    assert receiver["path"] == "naive"
    assert receiver["locked"] is None
    assert "abstain" in receiver["reason"]
    assert any("fell back" in w for w in report["warnings"])
    # The fallback bit stream still decodes: the decode ran on it, not on
    # an empty coherent attempt.
    assert report["correlation"]["detected"] is True
    decoded = report["payload"]["decoded"]
    assert decoded["available"] is True
    assert bytes.fromhex(decoded["hex"]) == message


def test_coherent_oversampled_burst_uses_recovered_sps_for_frame_bits(tmp_path):
    """Noise-tail fix on the coherent path: burst end in samples becomes a
    frame bound in bits through the *recovered* sps, and the frame decodes."""
    path = tmp_path / "coherent.iq"
    message = _oversampled_burst(path)
    report = analyze_file(_request(path, sample_rate=1000000.0))
    demodulation = report["demodulation"]
    receiver = demodulation["receiver"]
    assert receiver["requested"] == "auto"
    assert receiver["path"] == "coherent", receiver["reason"]
    assert receiver["locked"] is True
    assert receiver["sps"] == 8
    assert report["display"]["samples_per_symbol"] == 8

    correlation = report["correlation"]
    assert correlation["detected"] is True
    assert correlation["header_offset"] == 32  # 256 lead samples at sps 8

    decoded = report["payload"]["decoded"]
    assert decoded["available"] is True
    assert bytes.fromhex(decoded["hex"]) == message

    # The decode bound used the recovered sps: recompute it from the report's
    # own burst measurement and require an exact match.
    burst = report["signal"]["burst"]
    assert burst["found"] is True
    n_bits = demodulation["num_bits"]
    n_samples = report["signal"]["num_samples"]
    samples_per_bit = n_samples / n_bits
    payload_start = correlation["header_offset"] + SYNC_LEN
    expected = (
        round(int(burst["end_sample"]) / samples_per_bit)
        - payload_start
        + BURST_FRAME_SLACK_BITS
    )
    decode_search = report["payload"]["decode_search"]
    assert decode_search["frame_bits"] == expected


def test_analog_placeholder_records_the_naive_path(tmp_path):
    n = 20000
    t = np.arange(n) / 100000.0
    tone = np.exp(1j * 2.0 * np.pi * 10000.0 * t).astype(np.complex64)
    path = tmp_path / "tone.iq"
    tone.tofile(path)
    report = analyze_file(_request(path, modulation="auto", sync_word=None))
    receiver = report["demodulation"]["receiver"]
    assert receiver["path"] == "naive"
    assert report["payload"]["decoded"]["available"] is False


def test_unknown_receiver_value_warns_and_uses_auto(tmp_path):
    path = tmp_path / "unknown.iq"
    _symbol_spaced_burst(path)
    report = analyze_file(_request(path, receiver="bogus"))
    assert report["demodulation"]["receiver"]["requested"] == "auto"
    assert any("Unknown receiver" in w for w in report["warnings"])
