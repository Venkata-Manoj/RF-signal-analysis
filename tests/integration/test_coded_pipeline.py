"""End-to-end test of the full receive chain on coded captures.

This is the test that proves the problem statement's error-control requirements
end to end: a real frame is written to disk as ``.iq``, read back through the
public pipeline entrypoint, demodulated, de-interleaved, FEC-decoded and
CRC-verified — and the recovered bytes must equal the message that was
transmitted, including the channel errors that had to be corrected.

The captures are generated on the fly (not read from ``sample_data/``) so the
test is self-contained and does not depend on ``generate_test_data.py`` having
been run first.
"""

from __future__ import annotations

import numpy as np
import pytest

from rf_analyzer.core.framing import build_frame, modulate
from rf_analyzer.pipeline import analyze_file

SAMPLE_RATE = 100_000
SYNC_WORD = "0x1ACFFC1D"

#: (fec, interleaver, modulation, injected bit errors, expected FEC candidate)
CASES = [
    ("none", "none", "BPSK", 0, "None (CRC-16 only)"),
    ("conv", "convolutional", "BPSK", 14, "Convolutional r=1/2 K=7"),
    ("rs", "block", "BPSK", 12, "RS(255,223)"),
    ("ldpc", "pseudo-random-block", "BPSK", 8, "LDPC"),
    ("concatenated", "diagonal", "QPSK", 16, "Concatenated RS(255,223) + Conv K=7"),
    # 2-FSK is the only case whose capture has more than one sample per symbol,
    # so it is the only one that proves the symbol-period recovery path. Without
    # it, "demodulate FSK" would never be verified past the decision device.
    ("conv", "block", "2-FSK", 10, "Convolutional r=1/2 K=7"),
]

MESSAGE = b"SIH26147 end-to-end verification payload: FEC plus de-interleaving."

#: Correctness tests must not be decided by the wall-clock guard. On a loaded
#: machine the default 4 s budget can expire before the search reaches the
#: winning hypothesis, which would turn a passing decode into a flaky failure.
#: These tests assert *correctness*, so they buy themselves room; the default
#: budget itself is asserted in test_default_budget_is_reported below.
GENEROUS_BUDGET = {"decode_time_budget_s": 120.0, "decode_max_bits": 200_000}


def _write_capture(path, fec, interleaver, modulation, n_errors, seed=99):
    """Build a frame, inject channel errors, modulate and save as raw IQ."""
    frame = build_frame(MESSAGE, fec=fec, interleaver=interleaver)
    sync_len = int(frame["sync_bits"].size)
    bits = frame["bits"].copy()
    if n_errors:
        rng = np.random.default_rng(seed)
        idx = rng.choice(np.arange(sync_len, bits.size), size=n_errors, replace=False)
        bits[idx] ^= 1
    # 2-FSK is the one modulation that needs a sample rate (it emits many
    # samples per bit); the linear modulations ignore the keyword.
    samples = modulate(bits, modulation, sample_rate=SAMPLE_RATE)
    samples.astype(np.complex64).tofile(path)
    return frame


@pytest.mark.parametrize(
    ("fec", "interleaver", "modulation", "n_errors", "expected_candidate"), CASES
)
def test_coded_capture_is_decoded_and_crc_verified(
    tmp_path, fec, interleaver, modulation, n_errors, expected_candidate
):
    path = tmp_path / f"coded_{fec}.iq"
    _write_capture(path, fec, interleaver, modulation, n_errors)

    report = analyze_file(
        {
            "file_path": str(path),
            "sample_rate": SAMPLE_RATE,
            "sync_word": SYNC_WORD,
            "modulation": "auto",
            **GENEROUS_BUDGET,
        }
    )

    assert report["errors"] == [], report["errors"]

    # The header must be found before anything downstream can be trusted.
    assert report["correlation"]["detected"] is True
    assert report["correlation"]["score"] > 0.9

    payload = report["payload"]
    assert payload["decode_search"]["validated"] is True

    decoded = payload["decoded"]
    assert decoded["available"] is True
    assert decoded["crc_pass"] is True
    assert decoded["bytes"] == len(MESSAGE)
    assert bytes.fromhex(decoded["hex"]) == MESSAGE, "recovered bytes differ"
    assert decoded["text"] == MESSAGE.decode("utf-8")
    assert decoded["content_type"] == "text"

    # The transmitter's scheme must be identified, not merely stumbled into.
    assert report["fec"]["candidate"] == expected_candidate
    assert report["fec"]["validated"] is True
    assert report["interleaving"]["candidate"] == interleaver


@pytest.mark.parametrize(
    ("fec", "interleaver", "modulation", "n_errors", "_candidate"), CASES
)
def test_raw_payload_is_labelled_as_still_coded(
    tmp_path, fec, interleaver, modulation, n_errors, _candidate
):
    """The raw rendering must never masquerade as the decoded message."""
    if fec == "none" and interleaver == "none":
        pytest.skip("an uncoded frame's raw payload *is* the message")

    path = tmp_path / f"raw_{fec}.iq"
    _write_capture(path, fec, interleaver, modulation, n_errors)

    report = analyze_file(
        {
            "file_path": str(path),
            "sample_rate": SAMPLE_RATE,
            "sync_word": SYNC_WORD,
            "modulation": "auto",
            **GENEROUS_BUDGET,
        }
    )

    raw = report["payload"]["raw"]
    assert raw["bytes"] > len(MESSAGE)  # FEC expands the frame
    assert raw["content_type"] in ("binary", "mixed")
    assert MESSAGE.decode("utf-8") not in raw["text_preview"]


def test_2fsk_symbol_period_is_recovered_not_assumed(tmp_path):
    """A 2-FSK capture must be decimated to one bit per symbol, not per sample.

    The naive MVP path emits one bit per *sample*. On a capture that spends 100
    samples on every symbol that yields 100 identical bits in a row, which
    smears the sync word past any threshold and leaves a payload that can never
    be framed. This test pins the recovered period, the derived symbol rate and
    the resulting bit count, so a regression to the naive path fails loudly
    instead of just quietly returning "no decode".
    """
    path = tmp_path / "fsk.iq"
    frame = _write_capture(path, "conv", "block", "2-FSK", 10)

    report = analyze_file(
        {
            "file_path": str(path),
            "sample_rate": SAMPLE_RATE,
            "sync_word": SYNC_WORD,
            "modulation": "auto",
            **GENEROUS_BUDGET,
        }
    )

    # 1 ms symbols at 100 kHz.
    assert report["modulation"]["estimated_type"] == "2-FSK"
    assert report["display"]["samples_per_symbol"] == 100
    assert report["signal"]["symbol_rate_estimate"] == pytest.approx(1000.0)
    assert report["demodulation"]["num_bits"] == frame["bits"].size

    assert report["correlation"]["detected"] is True
    assert report["correlation"]["score"] > 0.9
    assert report["payload"]["decoded"]["crc_pass"] is True
    assert bytes.fromhex(report["payload"]["decoded"]["hex"]) == MESSAGE


def test_decode_can_be_disabled(tmp_path):
    """The GUI's 'fast mode' must skip the search and say so honestly."""
    path = tmp_path / "fastmode.iq"
    _write_capture(path, "conv", "convolutional", "BPSK", 0)

    report = analyze_file(
        {
            "file_path": str(path),
            "sample_rate": SAMPLE_RATE,
            "sync_word": SYNC_WORD,
            "modulation": "BPSK",
            "decode": False,
        }
    )

    assert report["payload"]["decode_search"]["enabled"] is False
    assert report["payload"]["decode_search"]["attempts"] == 0
    assert report["payload"]["decoded"]["available"] is False
    # The blind heuristic scores are still reported, still capped.
    assert report["fec"]["validated"] is False
    assert report["fec"]["confidence"] <= 0.5


def test_uncoded_noise_is_never_claimed_as_decoded(tmp_path):
    """Random bits must not produce a payload claim."""
    rng = np.random.default_rng(4)
    samples = (rng.integers(0, 2, size=6000).astype(np.float32) * 2.0 - 1.0).astype(
        np.complex64
    )
    samples.tofile(tmp_path / "noise.iq")

    report = analyze_file(
        {
            "file_path": str(tmp_path / "noise.iq"),
            "sample_rate": SAMPLE_RATE,
            "modulation": "BPSK",
            "sync_word": SYNC_WORD,
        }
    )

    assert report["payload"]["decoded"]["available"] is False
    assert report["fec"]["validated"] is False
    assert any("not a decoded message" in w for w in report["warnings"])


def test_frame_bits_hint_handles_trailing_samples(tmp_path):
    """A capture that continues past the burst needs the frame-length hint."""
    path = tmp_path / "burst.iq"
    frame = _write_capture(path, "rs", "block", "BPSK", 10)

    base = {
        "sample_rate": SAMPLE_RATE,
        "sync_word": SYNC_WORD,
        "modulation": "BPSK",
    }

    report = analyze_file(
        {
            **base,
            "file_path": str(path),
            "frame_bits": frame["frame_bits"],
            **GENEROUS_BUDGET,
        }
    )
    assert report["payload"]["decoded"]["available"] is True
    assert bytes.fromhex(report["payload"]["decoded"]["hex"]) == MESSAGE


def test_report_keeps_every_required_schema_key(tmp_path):
    """Adding the payload block must not disturb the info.md §13 schema."""
    path = tmp_path / "schema.iq"
    _write_capture(path, "rs", "block", "BPSK", 4)

    report = analyze_file(
        {
            "file_path": str(path),
            "sample_rate": SAMPLE_RATE,
            "sync_word": SYNC_WORD,
            "modulation": "BPSK",
        }
    )

    for key in (
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
    ):
        assert key in report, f"missing §13 key: {key}"
    assert "payload" in report


def test_default_search_budget_is_reported(tmp_path):
    """The override must not hide the default: an ordinary request reports it.

    This keeps the §NFR-03 guard visible in every report instead of only in the
    tests that deliberately raise it.
    """
    from rf_analyzer.config import DECODE_MAX_BITS, DECODE_TIME_BUDGET_S

    path = tmp_path / "budget.iq"
    _write_capture(path, "rs", "block", "BPSK", 0)

    report = analyze_file(
        {
            "file_path": str(path),
            "sample_rate": SAMPLE_RATE,
            "sync_word": SYNC_WORD,
            "modulation": "BPSK",
        }
    )

    search = report["payload"]["decode_search"]
    assert search["max_bits"] == DECODE_MAX_BITS
    assert search["time_budget_s"] == DECODE_TIME_BUDGET_S


def test_search_budget_is_overridable_per_request(tmp_path):
    """A caller who knows the frame is long can raise the limits."""
    path = tmp_path / "override.iq"
    _write_capture(path, "rs", "block", "BPSK", 0)

    report = analyze_file(
        {
            "file_path": str(path),
            "sample_rate": SAMPLE_RATE,
            "sync_word": SYNC_WORD,
            "modulation": "BPSK",
            "decode_max_bits": 4096,
            "decode_time_budget_s": 0.5,
        }
    )

    search = report["payload"]["decode_search"]
    assert search["max_bits"] == 4096
    assert search["time_budget_s"] == 0.5


def test_bad_budget_values_fall_back_to_defaults(tmp_path):
    """Garbage in the request must not crash the analysis."""
    from rf_analyzer.config import DECODE_MAX_BITS, DECODE_TIME_BUDGET_S

    path = tmp_path / "badbudget.iq"
    _write_capture(path, "rs", "block", "BPSK", 0)

    report = analyze_file(
        {
            "file_path": str(path),
            "sample_rate": SAMPLE_RATE,
            "sync_word": SYNC_WORD,
            "modulation": "BPSK",
            "decode_max_bits": "not-a-number",
            "decode_time_budget_s": None,
        }
    )

    assert report["errors"] == []
    search = report["payload"]["decode_search"]
    assert search["max_bits"] == DECODE_MAX_BITS
    assert search["time_budget_s"] == DECODE_TIME_BUDGET_S
