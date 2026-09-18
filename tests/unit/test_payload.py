"""Unit tests for header/payload identification and payload presentation.

The point of these tests is the honesty contract: the ``raw`` rendering is the
still-coded bit stream and must never be dressed up as a decoded message, while
``decoded`` is only populated from a CRC-verified decode.
"""

from __future__ import annotations

import numpy as np
import pytest

from rf_analyzer.core.payload import (
    classify_bytes,
    hexdump,
    payload_report,
    printable_text,
    slice_payload,
)


class TestSlicePayload:
    def test_slices_after_the_sync_word(self):
        bits = np.array([1, 1, 1, 0, 0, 1], dtype=np.uint8)
        assert slice_payload(bits, 3).tolist() == [0, 0, 1]

    def test_offset_zero_returns_everything(self):
        bits = np.array([1, 0, 1], dtype=np.uint8)
        assert slice_payload(bits, 0).tolist() == [1, 0, 1]

    def test_offset_past_the_end_is_empty_not_an_error(self):
        bits = np.array([1, 0, 1], dtype=np.uint8)
        assert slice_payload(bits, 99).size == 0

    def test_negative_offset_is_clamped(self):
        bits = np.array([1, 0, 1], dtype=np.uint8)
        assert slice_payload(bits, -5).tolist() == [1, 0, 1]

    def test_empty_input(self):
        assert slice_payload(np.array([], dtype=np.uint8), 0).size == 0


class TestClassifyBytes:
    def test_empty(self):
        assert classify_bytes(b"") == "empty"

    def test_plain_ascii_is_text(self):
        assert classify_bytes(b"hello world, this is a payload") == "text"

    def test_whitespace_counts_as_text(self):
        assert classify_bytes(b"line one\r\nline two\ttabbed") == "text"

    def test_high_entropy_is_binary(self):
        assert classify_bytes(bytes(range(0, 32)) * 4) == "binary"

    def test_half_printable_is_mixed(self):
        assert classify_bytes(b"abcd" + bytes([0, 1, 2, 3])) == "mixed"


class TestPrintableText:
    def test_printable_passthrough(self):
        assert printable_text(b"ABC") == "ABC"

    def test_non_printable_becomes_a_dot(self):
        assert printable_text(b"A\x00B") == "A.B"

    def test_control_characters_are_escaped(self):
        assert printable_text(b"A\tB\nC\rD") == "A\tB\nC\rD"

    def test_truncates_and_marks_the_elision(self):
        assert printable_text(b"x" * 50, max_chars=10) == "x" * 10 + "..."


class TestHexdump:
    def test_known_layout(self):
        lines = hexdump(b"ABCDEFGHIJKLMNOP", width=16)
        assert len(lines) == 1
        assert lines[0].startswith("00000000  ")
        assert lines[0].endswith("|ABCDEFGHIJKLMNOP|")
        assert "41 42 43 44" in lines[0]

    def test_offsets_advance_per_row(self):
        lines = hexdump(bytes(range(32)), width=16)
        assert lines[0].startswith("00000000")
        assert lines[1].startswith("00000010")

    def test_short_final_row_is_padded_not_misaligned(self):
        lines = hexdump(b"AB", width=16)
        assert lines[0].endswith("|AB|")
        # ASCII column must stay aligned across rows of differing length.
        long_lines = hexdump(b"A" * 18, width=16)
        assert long_lines[0].index("|") == long_lines[1].index("|")

    def test_non_printable_renders_as_dot(self):
        lines = hexdump(b"\x00\x01\x02", width=16)
        assert lines[0].endswith("|...|")

    def test_truncation_is_announced(self):
        lines = hexdump(b"A" * 100, width=16, max_bytes=16)
        assert len(lines) == 2
        assert "84 more byte(s)" in lines[-1]

    def test_rejects_a_zero_width(self):
        with pytest.raises(ValueError):
            hexdump(b"AB", width=0)


class TestPayloadReport:
    def test_raw_only_when_nothing_decoded(self):
        bits = np.tile(np.array([1, 0, 1, 1, 0, 0, 0, 1], dtype=np.uint8), 4)
        block = payload_report(bits, start_offset=8)

        assert block["header_offset"] == 8
        assert block["payload_bits"] == 24
        assert block["raw"]["bytes"] == 3
        assert block["decoded"]["available"] is False
        assert block["decoded"]["crc_pass"] is None
        assert block["decoded"]["hex"] == ""
        assert "remains coded" in block["decoded"]["note"]

    def test_decoded_block_requires_crc_pass(self):
        bits = np.tile(np.array([1, 0, 1, 1, 0, 0, 0, 1], dtype=np.uint8), 4)
        unverified = {"candidate": "LDPC", "crc_pass": False, "payload_hex": "dead"}
        assert payload_report(bits, decoded=unverified)["decoded"]["available"] is False

    def test_decoded_block_is_populated_on_a_verified_decode(self):
        bits = np.zeros(64, dtype=np.uint8)
        verified = {
            "candidate": "RS(255,223)",
            "crc_pass": True,
            "confidence": 0.99,
            "payload_hex": b"hello payload".hex(),
            "params": {"interleaver": "block"},
            "errors_corrected": 3,
        }
        decoded = payload_report(bits, decoded=verified)["decoded"]

        assert decoded["available"] is True
        assert decoded["scheme"] == "RS(255,223)"
        assert decoded["interleaver"] == "block"
        assert decoded["bytes"] == len(b"hello payload")
        assert decoded["content_type"] == "text"
        assert decoded["text"] == "hello payload"
        assert decoded["errors_corrected"] == 3
        assert decoded["note"] == "CRC-16 verified"

    def test_malformed_hex_degrades_to_empty_rather_than_raising(self):
        bits = np.zeros(64, dtype=np.uint8)
        bad = {"candidate": "LDPC", "crc_pass": True, "payload_hex": "zz"}
        decoded = payload_report(bits, decoded=bad)["decoded"]
        assert decoded["available"] is True
        assert decoded["bytes"] == 0

    def test_preview_size_is_honoured(self):
        bits = np.ones(8 * 64, dtype=np.uint8)
        block = payload_report(bits, max_bytes=8)
        assert block["raw"]["bytes"] == 64
        assert len(block["raw"]["hex_preview"]) == 16
        assert len(block["raw"]["hexdump"]) == 2  # 1 row + truncation notice
