"""GUI coverage of the whole pipeline report.

The desktop GUI was written before the pipeline grew its burst detector, its
EVM/MER quality block, the classifier's alternatives/corroboration, and the
ranked FEC/interleaver hypotheses. Each of those reached the report and stopped
there — the window showed a subset frozen at whatever the pipeline returned the
day it was written.

These tests pin the surfaces that close that gap, so a future report field
cannot go unshown without a test failing. They also pin the honesty rule the
table relies on: ``None`` means "not measured" and must never render as ``0``.

Headless-safe (QT_QPA_PLATFORM=offscreen, no modal dialogs).
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import numpy as np
import pytest

QtWidgets = pytest.importorskip("PyQt6.QtWidgets", reason="PyQt6 not installed")
pg = pytest.importorskip("pyqtgraph", reason="pyqtgraph not installed")
from PyQt6.QtWidgets import QApplication

from rf_analyzer.gui.main_window import MainWindow

#: Module-level holder: without a lasting reference Python may collect the
#: QApplication and take every widget with it (see test_gui_payload.py).
_APP_HOLDER: list[QApplication] = []

#: Section headings the results table must carry, one per report block.
SECTIONS = (
    "-- INPUT --",
    "-- SIGNAL --",
    "-- BURST DETECTION --",
    "-- MODULATION --",
    "-- QUALITY (EVM / MER) --",
    "-- DEMODULATION --",
    "-- CORRELATION --",
    "-- PAYLOAD --",
    "-- FEC --",
    "-- INTERLEAVING --",
    "-- RESULT --",
    "-- DIAGNOSTICS --",
)


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


def _table_keys(window: MainWindow) -> list[str]:
    table = window.results_table
    return [
        table.item(r, 0).text() for r in range(table.rowCount()) if table.item(r, 0)
    ]


def _empty_report(**overrides) -> dict:
    """A report with every block present but empty, for table-only tests."""
    report = {
        "input": {"file_name": "x.iq", "file_type": "iq", "sample_rate": 1000},
        "signal": {"num_samples": 10, "burst": {}},
        "modulation": {"estimated_type": "BPSK"},
        "quality": {},
        "demodulation": {},
        "correlation": {},
        "payload": {},
        "fec": {},
        "interleaving": {},
        "display": {},
        "warnings": [],
        "errors": [],
    }
    report.update(overrides)
    return report


def _time_regions(window: MainWindow) -> list:
    return [
        item
        for item in window.time_plot.getPlotItem().items
        if isinstance(item, pg.LinearRegionItem)
    ]


# --------------------------------------------------------------------------- #
# Results table: every block has a home
# --------------------------------------------------------------------------- #


def test_every_report_block_has_a_section():
    """A block the pipeline returns with no section is a block nobody sees."""
    window = _make_window()
    try:
        window._update_results_table(_empty_report())
        keys = set(_table_keys(window))
        missing = [s for s in SECTIONS if s not in keys]
        assert not missing, f"results table is missing sections: {missing}"
    finally:
        window.close()


def test_not_measured_is_never_rendered_as_zero():
    """``None`` is "not measured"; ``0.0`` is a measurement. They must differ.

    An uncoded capture genuinely has an EVM of 0.0%, so rendering a missing
    measurement the same way would show a perfect fit for data nobody fitted.
    """
    window = _make_window()
    try:
        measured = _empty_report(quality={"applicable": True, "evm_percent": 0.0})
        window._update_results_table(measured)
        assert _table_value(window, "EVM (%)") == "0"

        absent = _empty_report(quality={"applicable": False, "evm_percent": None})
        window._update_results_table(absent)
        assert _table_value(window, "EVM (%)") == "—"
        assert _table_value(window, "MER (dB)") == "—"
    finally:
        window.close()


def test_absent_blocks_do_not_raise_or_invent_values():
    """A failed analysis carries an empty report; the table must still render."""
    window = _make_window()
    try:
        window._update_results_table({"errors": ["Unsupported file type: '.txt'"]})
        assert _table_value(window, "Burst found") == "no"
        assert _table_value(window, "FEC verified") == "no"
        assert _table_value(window, "Recovered message") == "none"
        assert _table_value(window, "Errors") == "Unsupported file type: '.txt'"
    finally:
        window.close()


def test_quality_and_fec_detail_rows_carry_the_report_values():
    """The rows that were missing must show the report's numbers verbatim."""
    window = _make_window()
    try:
        report = _empty_report(
            quality={
                "applicable": True,
                "mode": "16-QAM",
                "evm_percent": 3.75,
                "mer_db": 28.5,
                "snr_db_from_evm": 28.5,
                "modulation_order": 4,
                "n_symbols": 1000,
                "fit_ok": True,
            },
            modulation={
                "estimated_type": "16-QAM",
                "alternatives": ["QPSK", "BPSK"],
                "corroborated": True,
                "revised_from": "QPSK",
            },
            fec={
                "candidate": "LDPC",
                "validated": True,
                "crc_pass": True,
                "errors_corrected": 7,
                "params": {"interleaver": "block", "nsym": None},
            },
            interleaving={"candidate": "block", "validated": True, "depth": 32},
        )
        window._update_results_table(report)
        assert _table_value(window, "EVM (%)") == "3.75"
        assert _table_value(window, "MER (dB)") == "28.5"
        assert _table_value(window, "Modulation order") == "4"
        assert _table_value(window, "Fit ok") == "yes"
        assert _table_value(window, "Mod alternatives") == "QPSK, BPSK"
        assert _table_value(window, "Mod corroborated") == "yes"
        assert _table_value(window, "Mod revised from") == "QPSK"
        assert _table_value(window, "FEC errors corrected") == "7"
        assert _table_value(window, "FEC CRC pass") == "yes"
        assert _table_value(window, "Interleaver depth") == "32"
    finally:
        window.close()


# --------------------------------------------------------------------------- #
# Hypotheses tab
# --------------------------------------------------------------------------- #


def test_hypotheses_tab_exists():
    window = _make_window()
    try:
        titles = [window.tabs.tabText(i) for i in range(window.tabs.count())]
        assert "Hypotheses" in titles, f"Hypotheses tab missing from {titles}"
    finally:
        window.close()


def test_candidate_tables_follow_the_report():
    window = _make_window()
    try:
        window._update_hypotheses_tab(
            _empty_report(
                fec={
                    "candidate": "LDPC",
                    "candidates": [
                        {"candidate": "LDPC", "confidence": 0.9},
                        {"candidate": "RS(255,223)", "confidence": 0.2},
                    ],
                },
                interleaving={
                    "candidates": [
                        {"candidate": "Block", "depth": 32, "confidence": 0.8}
                    ]
                },
                quality={
                    "candidates": [
                        {"mode": "QPSK", "modulation_order": 2, "evm_percent": 4.0}
                    ]
                },
            )
        )
        assert window.fec_candidates_table.rowCount() == 2
        assert window.ilv_candidates_table.rowCount() == 1
        assert window.qual_candidates_table.rowCount() == 1
        assert window.fec_candidates_table.item(0, 0).text() == "LDPC"
        assert window.ilv_candidates_table.item(0, 1).text() == "32"
    finally:
        window.close()


def test_candidate_tables_shrink_with_the_report():
    """A shorter new list must not leave the previous report's tail behind."""
    window = _make_window()
    try:
        window._update_hypotheses_tab(
            _empty_report(
                fec={
                    "candidates": [
                        {"candidate": f"C{i}", "confidence": 0.1 * i} for i in range(6)
                    ]
                },
                interleaving={
                    "candidates": [{"candidate": f"I{i}", "depth": i} for i in range(5)]
                },
            )
        )
        assert window.fec_candidates_table.rowCount() == 6
        assert window.ilv_candidates_table.rowCount() == 5

        window._update_hypotheses_tab(
            _empty_report(fec={"candidates": [{"candidate": "only"}]})
        )
        assert window.fec_candidates_table.rowCount() == 1
        assert window.ilv_candidates_table.rowCount() == 0
    finally:
        window.close()


def test_decode_search_off_says_so_instead_of_zero_hypotheses():
    window = _make_window()
    try:
        window._update_hypotheses_tab(
            _empty_report(payload={"decode_search": {"enabled": False, "attempts": 0}})
        )
        assert "off" in window.decode_search_label.text()

        window._update_hypotheses_tab(
            _empty_report(
                payload={
                    "decode_search": {
                        "enabled": True,
                        "attempts": 14,
                        "validated": False,
                    }
                },
                fec={"candidate": "LDPC"},
            )
        )
        text = window.decode_search_label.text()
        assert "14 hypotheses" in text
        assert "no CRC match" in text
        # Nothing was won, so it must not be called a winner.
        assert "winner" not in text
    finally:
        window.close()


# --------------------------------------------------------------------------- #
# Plots: burst span and quality annotation
# --------------------------------------------------------------------------- #


def test_burst_span_is_shaded_and_then_cleared():
    """The shading must track the report, including when the next run has none."""
    window = _make_window()
    try:
        samples = np.ones(4096, dtype=np.complex64)
        window._plot_time(
            samples,
            burst={"found": True, "start_sample": 384, "end_sample": 1344},
        )
        regions = _time_regions(window)
        assert len(regions) == 1, "a detected burst must be shaded on the time plot"
        assert regions[0].getRegion() == (384.0, 1344.0)
        assert "384" in window.time_plot.getPlotItem().titleLabel.text

        # Next run: no burst. A leftover marker would misreport the new capture.
        window._plot_time(samples, burst={"found": False})
        assert _time_regions(window) == []
        assert "burst" not in window.time_plot.getPlotItem().titleLabel.text.lower()
    finally:
        window.close()


def test_constellation_annotates_quality_without_faking_an_overlay():
    window = _make_window()
    try:
        samples = np.ones(256, dtype=np.complex64)
        window._plot_constellation(
            samples,
            quality={
                "applicable": True,
                "mode": "BPSK",
                "evm_percent": 3.5,
                "mer_db": 20.0,
                "fit_ok": True,
            },
        )
        title = window.const_plot.getPlotItem().titleLabel.text
        assert "EVM 3.5%" in title
        assert "MER 20" in title
        # Only the measured scatter: no ideal reference grid is drawn over
        # samples the GUI did not demodulate.
        assert len(window.const_plot.listDataItems()) == 1

        window._plot_constellation(
            samples, quality={"applicable": False, "mode": "2-FSK"}
        )
        title = window.const_plot.getPlotItem().titleLabel.text
        assert "2-FSK" in title
        assert "not applicable" in title
    finally:
        window.close()


# --------------------------------------------------------------------------- #
# End to end on a real capture
# --------------------------------------------------------------------------- #


def test_a_real_capture_fills_the_new_surfaces():
    """One full run must populate the table, the hypotheses tab and the plots."""
    window = _make_window()
    try:
        rng = np.random.default_rng(11)
        samples = (rng.integers(0, 2, size=8192).astype(np.float32) * 2.0 - 1.0).astype(
            np.complex64
        )
        path = os.path.join(os.environ.get("TEMP", "."), "gui_coverage_probe.iq")
        samples.tofile(path)

        window.open_file(path)
        window.mod_combo.setCurrentText("BPSK")
        window.run_analysis()
        _get_app().processEvents()

        assert window.last_report is not None
        keys = set(_table_keys(window))
        for section in SECTIONS:
            assert section in keys, f"missing section {section} after a real run"

        # The hypothesis tables were rebuilt from this report, not left empty.
        assert window.fec_candidates_table.rowCount() > 0
        assert window.decode_search_label.text().startswith("Decode search:")
    finally:
        window.close()
