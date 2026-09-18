"""Unit tests for batch folder analysis and CSV/HTML export."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from rf_analyzer.batch import (
    SUMMARY_COLUMNS,
    analyze_folder,
    discover_captures,
    render_html,
    summarize_report,
    write_csv,
    write_html,
)
from rf_analyzer.core.framing import build_frame, modulate

SAMPLE_RATE = 100_000
SYNC_WORD = "0x1ACFFC1D"
MESSAGE = b"batch export unit test payload, long enough to be a real message."


def _write_coded(path, fec="rs", interleaver="block", modulation="BPSK"):
    frame = build_frame(MESSAGE, fec=fec, interleaver=interleaver)
    modulate(frame["bits"], modulation).astype(np.complex64).tofile(path)
    return frame


def _write_noise(path, n=2048):
    rng = np.random.default_rng(2)
    samples = (rng.integers(0, 2, size=n).astype(np.float32) * 2.0 - 1.0).astype(
        np.complex64
    )
    samples.tofile(path)


class TestDiscoverCaptures:
    def test_finds_only_supported_suffixes(self, tmp_path):
        _write_coded(tmp_path / "a.iq")
        _write_coded(tmp_path / "b.iq")
        (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
        (tmp_path / "readme.md").write_text("ignore me too", encoding="utf-8")

        found = discover_captures(tmp_path)
        assert [p.name for p in found] == ["a.iq", "b.iq"]

    def test_is_sorted_by_name(self, tmp_path):
        for name in ("z.iq", "a.iq", "m.iq"):
            _write_coded(tmp_path / name)
        assert [p.name for p in discover_captures(tmp_path)] == [
            "a.iq",
            "m.iq",
            "z.iq",
        ]

    def test_recursive_flag_walks_subfolders(self, tmp_path):
        (tmp_path / "nested").mkdir()
        _write_coded(tmp_path / "top.iq")
        _write_coded(tmp_path / "nested" / "deep.iq")

        assert len(discover_captures(tmp_path)) == 1
        assert len(discover_captures(tmp_path, recursive=True)) == 2

    def test_missing_folder_raises(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            discover_captures(tmp_path / "nope")


class TestAnalyzeFolder:
    def test_verifies_the_coded_capture_and_flags_the_noise(self, tmp_path):
        _write_coded(tmp_path / "good.iq", fec="rs", interleaver="block")
        _write_noise(tmp_path / "noise.iq")

        rows = analyze_folder(
            tmp_path, sample_rate=SAMPLE_RATE, sync_word=SYNC_WORD, decode=True
        )

        assert [r["file"] for r in rows] == ["good.iq", "noise.iq"]
        good, noise = rows
        assert good["decode_validated"] == "yes"
        assert good["decoded_text"] == MESSAGE.decode("utf-8")
        assert noise["decode_validated"] == "no"
        assert noise["decoded_text"] == ""

    def test_every_row_carries_every_column(self, tmp_path):
        _write_coded(tmp_path / "one.iq")
        rows = analyze_folder(tmp_path, sample_rate=SAMPLE_RATE, sync_word=SYNC_WORD)
        assert set(rows[0]) == {key for key, _ in SUMMARY_COLUMNS}

    def test_progress_callback_is_invoked(self, tmp_path):
        _write_coded(tmp_path / "a.iq")
        _write_coded(tmp_path / "b.iq")
        seen: list[tuple[str, int, int]] = []
        analyze_folder(
            tmp_path,
            sample_rate=SAMPLE_RATE,
            sync_word=SYNC_WORD,
            progress=lambda p, i, t: seen.append((p.name, i, t)),
        )
        assert seen == [("a.iq", 1, 2), ("b.iq", 2, 2)]

    def test_decode_false_skips_the_search(self, tmp_path):
        _write_coded(tmp_path / "good.iq", fec="rs", interleaver="block")
        rows = analyze_folder(
            tmp_path, sample_rate=SAMPLE_RATE, sync_word=SYNC_WORD, decode=False
        )
        assert rows[0]["decode_validated"] == "no"
        assert rows[0]["decode_attempts"] == 0

    def test_empty_folder_returns_no_rows(self, tmp_path):
        assert analyze_folder(tmp_path, sample_rate=SAMPLE_RATE) == []

    def test_unsupported_file_inside_the_folder_is_skipped(self, tmp_path):
        _write_coded(tmp_path / "a.iq")
        (tmp_path / "junk.bin").write_bytes(b"\x00\x01\x02")
        assert len(analyze_folder(tmp_path, sample_rate=SAMPLE_RATE)) == 1


class TestSummarizeReport:
    def test_flattens_the_report_and_rounds_floats(self):
        report = {
            "input": {"file_type": "iq", "iq_format_used": "complex64"},
            "signal": {"num_samples": 100, "snr_db": 18.23456789},
            "modulation": {"estimated_type": "BPSK"},
            "demodulation": {"mode": "BPSK", "num_bits": 100},
            "correlation": {"header_offset": 0, "score": 1.0},
            "payload": {
                "payload_bits": 68,
                "raw": {"bytes": 9},
                "decode_search": {"attempts": 3, "validated": True},
                "decoded": {"available": True, "bytes": 4, "text": "abcd"},
            },
            "warnings": ["w1"],
            "errors": [],
        }
        row = summarize_report("x.iq", report, 0.123456)

        assert row["file"] == "x.iq"
        assert row["snr_db"] == 18.2346
        assert row["elapsed_s"] == 0.1235
        assert row["decode_validated"] == "yes"
        assert row["decoded_text"] == "abcd"
        assert row["warnings"] == "w1"

    def test_missing_sections_do_not_raise(self):
        row = summarize_report("x.iq", {}, 0.0)
        assert row["file"] == "x.iq"
        assert row["decode_validated"] == "no"
        assert row["decoded_bytes"] == ""

    def test_load_failure_is_recorded(self):
        row = summarize_report("bad.iq", {"errors": ["Unsupported file type"]}, 0.0)
        assert row["errors"] == "Unsupported file type"


class TestCsvExport:
    def test_writes_a_header_and_one_row_per_capture(self, tmp_path):
        _write_coded(tmp_path / "a.iq")
        _write_coded(tmp_path / "b.iq")
        rows = analyze_folder(tmp_path, sample_rate=SAMPLE_RATE, sync_word=SYNC_WORD)

        out = write_csv(rows, tmp_path / "out" / "summary.csv")
        assert out.exists()

        with open(out, encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            written = list(reader)
        assert reader.fieldnames == [key for key, _ in SUMMARY_COLUMNS]
        assert [r["file"] for r in written] == ["a.iq", "b.iq"]

    def test_creates_parent_directories(self, tmp_path):
        out = write_csv([], tmp_path / "deep" / "nested" / "s.csv")
        assert out.exists()


class TestHtmlExport:
    def test_renders_a_self_contained_page(self, tmp_path):
        _write_coded(tmp_path / "good.iq", fec="rs", interleaver="block")
        rows = analyze_folder(tmp_path, sample_rate=SAMPLE_RATE, sync_word=SYNC_WORD)

        out = write_html(rows, tmp_path / "summary.html", subtitle="unit test")
        text = out.read_text(encoding="utf-8")

        assert text.startswith("<!DOCTYPE html>")
        assert text.rstrip().endswith("</html>")
        assert "unit test" in text
        # Self-contained: no network fetches at all.
        assert "http://" not in text
        assert "https://" not in text
        assert "<script" not in text.lower()

    def test_decoded_message_appears_in_the_page(self, tmp_path):
        _write_coded(tmp_path / "good.iq", fec="rs", interleaver="block")
        rows = analyze_folder(tmp_path, sample_rate=SAMPLE_RATE, sync_word=SYNC_WORD)
        text = render_html(rows)
        assert MESSAGE.decode("utf-8") in text

    def test_undecoded_capture_is_not_claimed_as_a_message(self, tmp_path):
        _write_noise(tmp_path / "noise.iq")
        rows = analyze_folder(tmp_path, sample_rate=SAMPLE_RATE, sync_word=SYNC_WORD)
        text = render_html(rows)
        # The honesty note is always present, and no message is ever claimed.
        assert "independent CRC-16 check" in text
        assert "Recovered messages" not in text
        assert MESSAGE.decode("utf-8") not in text

    def test_empty_input_renders_guidance_not_a_crash(self):
        text = render_html([])
        assert "No captures found" in text
        assert "generate_test_data.py" in text

    def test_html_escapes_capture_content(self, tmp_path):
        """A message containing markup must not break the page."""
        nasty = b"<script>alert(1)</script> payload"
        frame = build_frame(nasty, fec="rs", interleaver="block")
        modulate(frame["bits"], "BPSK").astype(np.complex64).tofile(
            tmp_path / "nasty.iq"
        )
        rows = analyze_folder(tmp_path, sample_rate=SAMPLE_RATE, sync_word=SYNC_WORD)
        assert rows[0]["decode_validated"] == "yes"

        text = render_html(rows)
        assert "<script>alert(1)</script>" not in text
        assert "&lt;script&gt;" in text
