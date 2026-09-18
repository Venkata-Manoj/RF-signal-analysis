"""GUI tests for the payload view, decode controls and demo/batch actions.

The GUI must surface the pipeline's honest distinction between "decoded and
CRC-verified" and "shown as received, still coded" — these tests pin that.
Headless-safe (QT_QPA_PLATFORM=offscreen, no modal dialogs).
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
from pathlib import Path

import numpy as np
import pytest

QtWidgets = pytest.importorskip("PyQt6.QtWidgets", reason="PyQt6 not installed")
from PyQt6.QtWidgets import QApplication

from rf_analyzer.core.framing import build_frame, modulate
from rf_analyzer.gui.main_window import MainWindow

SAMPLE_DATA = Path(__file__).resolve().parents[2] / "sample_data"
CODED_RS = SAMPLE_DATA / "coded_rs.iq"
BPSK_IQ = SAMPLE_DATA / "bpsk.iq"

MESSAGE = b"SIH26147 end-to-end verification payload: FEC plus de-interleaving."

#: Module-level QApplication holder. Without a lasting reference the only one
#: lives in a helper's local scope, and Python may collect the application —
#: which deletes every widget with it ("wrapped C/C++ object has been deleted").
_APP_HOLDER: list[QApplication] = []


def _get_app() -> QApplication:
    if not _APP_HOLDER:
        _APP_HOLDER.append(QApplication.instance() or QApplication(sys.argv))
    return _APP_HOLDER[0]


def _make_window() -> MainWindow:
    app = _get_app()
    window = MainWindow()
    window.show()
    app.processEvents()
    return window


def _table_value(window: MainWindow, key: str) -> str | None:
    table = window.results_table
    for row in range(table.rowCount()):
        k_item = table.item(row, 0)
        v_item = table.item(row, 1)
        if k_item is not None and k_item.text() == key:
            return v_item.text() if v_item is not None else ""
    return None


def test_payload_tab_exists_with_badge_and_views():
    window = _make_window()
    try:
        titles = [window.tabs.tabText(i) for i in range(window.tabs.count())]
        assert "Payload" in titles, f"Payload tab missing from {titles}"

        assert window.payload_badge.text()
        assert window.payload_decoded_text.isReadOnly()
        assert window.payload_raw_text.isReadOnly()
    finally:
        window.close()


def test_decode_controls_default_and_feed_the_request(tmp_path):
    window = _make_window()
    try:
        assert window.decode_check.isChecked() is True
        assert window.frame_bits_spin.value() == 0
        assert window.frame_bits_spin.text() == "auto"

        path = tmp_path / "x.iq"
        modulate(
            build_frame(MESSAGE, fec="none", interleaver="none")["bits"], "BPSK"
        ).astype(np.complex64).tofile(path)
        window.open_file(str(path))

        req = window._build_request()
        assert req["decode"] is True
        assert req["frame_bits"] is None  # 0 -> auto

        window.frame_bits_spin.setValue(512)
        window.decode_check.setChecked(False)
        req = window._build_request()
        assert req["frame_bits"] == 512
        assert req["decode"] is False
    finally:
        window.close()


def test_iq_format_combo_offers_int8_and_auto():
    window = _make_window()
    try:
        options = [
            window.iq_format_combo.itemText(i)
            for i in range(window.iq_format_combo.count())
        ]
        assert "auto" in options
        assert "int8" in options
        assert "complex64" in options
    finally:
        window.close()


def test_verified_payload_is_shown_as_verified(tmp_path):
    """A CRC-valid decode must light the badge and fill the decoded view."""
    window = _make_window()
    try:
        path = tmp_path / "coded.iq"
        modulate(
            build_frame(MESSAGE, fec="rs", interleaver="block")["bits"], "BPSK"
        ).astype(np.complex64).tofile(path)

        window.open_file(str(path))
        window.sync_word_edit.setText("0x1ACFFC1D")
        window.decode_check.setChecked(True)
        window.run_analysis()
        _get_app().processEvents()

        assert "VERIFIED" in window.payload_badge.text()
        assert window.payload_badge.property("verified") is True
        assert MESSAGE.decode("utf-8") in window.payload_decoded_text.toPlainText()
        assert _table_value(window, "Recovered message") == (
            f"{len(MESSAGE)} bytes (CRC-16 verified)"
        )
        assert _table_value(window, "FEC verified") == "yes"
        assert _table_value(window, "Interleaver verified") == "yes"
        assert _table_value(window, "Interleaver") == "block"
    finally:
        window.close()


def test_undecoded_capture_is_labelled_honestly(tmp_path):
    """Random bits must never be presented as a recovered message."""
    window = _make_window()
    try:
        rng = np.random.default_rng(5)
        samples = (rng.integers(0, 2, size=4096).astype(np.float32) * 2.0 - 1.0).astype(
            np.complex64
        )
        path = tmp_path / "noise.iq"
        samples.tofile(path)

        window.open_file(str(path))
        window.mod_combo.setCurrentText("BPSK")
        window.run_analysis()
        _get_app().processEvents()

        assert "NOT DECODED" in window.payload_badge.text()
        assert window.payload_badge.property("verified") is False
        assert "No CRC-16-valid decode" in window.payload_decoded_text.toPlainText()
        assert _table_value(window, "Recovered message") == "none"
        assert _table_value(window, "FEC verified") == "no"
        # The raw view must still be populated, and labelled as coded.
        assert "hexdump" in window.payload_raw_text.toPlainText()
        assert _table_value(window, "Payload bytes (raw)") not in ("", "0")
    finally:
        window.close()


def test_decode_checkbox_off_reports_no_search(tmp_path):
    window = _make_window()
    try:
        path = tmp_path / "coded.iq"
        modulate(
            build_frame(MESSAGE, fec="rs", interleaver="block")["bits"], "BPSK"
        ).astype(np.complex64).tofile(path)

        window.open_file(str(path))
        window.decode_check.setChecked(False)
        window.run_analysis()
        _get_app().processEvents()

        assert "NOT DECODED" in window.payload_badge.text()
        assert _table_value(window, "Decode search") == "0 tried"
        assert _table_value(window, "FEC verified") == "no"
    finally:
        window.close()


def test_payload_tab_survives_a_load_failure():
    """An error report must not leave stale payload text on screen."""
    window = _make_window()
    try:
        window.run_analysis()
        report = {
            "errors": ["Unsupported file type: '.txt'"],
            "warnings": [],
            "payload": {},
        }
        window._update_payload_tab(report)
        assert "NOT ANALYSED" in window.payload_badge.text()
        assert window.payload_decoded_text.toPlainText() == ""
    finally:
        window.close()


@pytest.mark.skipif(not BPSK_IQ.exists(), reason="run generate_test_data.py first")
def test_blank_bits_preview_still_renders_payload_meta():
    window = _make_window()
    try:
        window.open_file(str(BPSK_IQ))
        window.run_analysis()
        _get_app().processEvents()
        meta = window.payload_meta_label.text()
        assert "Header:" in meta
        assert "Payload:" in meta
        assert "decode search" in meta
    finally:
        window.close()


@pytest.mark.skipif(not CODED_RS.exists(), reason="run generate_test_data.py first")
def test_demo_captures_are_discoverable():
    window = _make_window()
    try:
        names = [p.name for p in window._demo_captures()]
        assert "coded_rs.iq" in names
        assert "coded_concatenated.iq" in names
    finally:
        window.close()


@pytest.mark.skipif(not CODED_RS.exists(), reason="run generate_test_data.py first")
def test_batch_analysis_writes_both_exports(tmp_path, monkeypatch):
    """The batch action must analyse a folder and write CSV + HTML."""
    monkeypatch.chdir(tmp_path)
    work = tmp_path / "captures"
    work.mkdir()
    for name in ("coded_rs.iq", "coded_conv.iq", "coded_concatenated.iq"):
        src = SAMPLE_DATA / name
        if src.exists():
            (work / name).write_bytes(src.read_bytes())

    window = _make_window()
    try:
        rows = window.run_batch_analysis(str(work), export=True)
        assert len(rows) == 3
        assert all(r["decode_validated"] == "yes" for r in rows), rows
        assert Path("output/batch_summary.csv").exists()
        assert Path("output/batch_summary.html").exists()
    finally:
        window.close()


@pytest.mark.skipif(not CODED_RS.exists(), reason="run generate_test_data.py first")
def test_batch_analysis_without_export(tmp_path):
    work = tmp_path / "captures"
    work.mkdir()
    (work / "coded_rs.iq").write_bytes(CODED_RS.read_bytes())

    window = _make_window()
    try:
        rows = window.run_batch_analysis(str(work), export=False)
        assert len(rows) == 1
        assert rows[0]["decode_validated"] == "yes"
    finally:
        window.close()


def test_open_demo_capture_by_name():
    if not CODED_RS.exists():
        pytest.skip("run generate_test_data.py first")
    window = _make_window()
    try:
        window.open_demo_capture("coded_rs.iq")
        _get_app().processEvents()
        assert window.current_file is not None
        assert window.current_file.name == "coded_rs.iq"
        assert window.last_report is not None
        assert window.last_report["payload"]["decoded"]["available"] is True
    finally:
        window.close()


def test_open_demo_capture_unknown_name_is_harmless():
    window = _make_window()
    try:
        window.open_demo_capture("definitely-not-a-capture.iq")
        assert window.current_file is None
    finally:
        window.close()
