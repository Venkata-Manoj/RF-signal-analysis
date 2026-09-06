"""GUI reactivity regression tests: hardcoded params / frozen plots.

Guards the "hardcoded params / frozen plots" regression: the GUI must honor
live UI params on Run, mark stale results, show real demod bits, and refresh
plots when switching files. Headless-safe (QT_QPA_PLATFORM=offscreen, no
modal dialogs). Sample data owned by scripts/generate_test_data.py — skip
gracefully when artifacts are absent.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import re
import sys
from pathlib import Path

import numpy as np
import pytest

QtWidgets = pytest.importorskip("PyQt6.QtWidgets", reason="PyQt6 not installed")
from PyQt6.QtWidgets import QApplication, QGroupBox, QLabel, QPushButton, QWidget  # noqa: E402

from rf_analyzer.gui.main_window import MainWindow  # noqa: E402

SAMPLE_DATA = Path(__file__).resolve().parents[2] / "sample_data"
BPSK_IQ = SAMPLE_DATA / "bpsk.iq"
TONE_IQ = SAMPLE_DATA / "tone.iq"

# Substrings (lowercase) that count as a stale/change note. Kept specific so
# the pre-fix GUI (no stale handling) does NOT accidentally match. NOTE:
# "refresh" is intentionally excluded: the old buggy preview path logged
# "WARN: live preview refresh failed ..." which must NOT count as stale.
_STALE_KEYWORDS = (
    "stale",
    "outdated",
    "obsolete",
    "changed",
    "modified",
    "re-run",
    "rerun",
    "re run",
    "run again",
)


def _get_app() -> QApplication:
    # Arrange helper: singleton QApplication for offscreen tests.
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app  # type: ignore[return-value]


def _make_window() -> MainWindow:
    # Arrange helper: fresh visible (offscreen) window with events flushed.
    app = _get_app()
    window = MainWindow()
    window.show()
    app.processEvents()
    return window


def _table_value(window: MainWindow, key: str) -> str | None:
    # Helper: look up results_table value column by parameter name.
    table = window.results_table
    for row in range(table.rowCount()):
        k_item = table.item(row, 0)
        v_item = table.item(row, 1)
        if k_item is not None and k_item.text() == key:
            return v_item.text() if v_item is not None else ""
    return None


def _snapshot_time_curves(window: MainWindow) -> list[np.ndarray]:
    # Helper: snapshot time_plot Y data (I and Q curves) for frozen-plot check.
    app = _get_app()
    app.processEvents()
    try:
        items = window.time_plot.listDataItems()  # type: ignore[attr-defined]
    except Exception:
        items = window.time_plot.plotItem.listDataItems()
    curves: list[np.ndarray] = []
    for item in items:
        try:
            _, y = item.getData()
        except Exception:
            continue
        if y is None:
            continue
        curves.append(np.asarray(y, dtype=float).copy())
    return curves


def _visible_stale_texts(window: MainWindow) -> list[str]:
    # Helper: collect visible widget texts that look like a stale indicator.
    found: list[str] = []
    # Preferred: dedicated badge if the UI provides one.
    badge = getattr(window, "stale_badge", None)
    candidates: list[QWidget] = []
    if badge is not None:
        candidates.append(badge)
    try:
        for lbl in window.findChildren(QLabel):
            if lbl not in candidates:
                candidates.append(lbl)
        for btn in window.findChildren(QPushButton):
            candidates.append(btn)
        for grp in window.findChildren(QGroupBox):
            candidates.append(grp)
    except Exception:
        pass
    for w in candidates:
        try:
            if hasattr(w, "isVisible") and not w.isVisible():
                continue
            text = ""
            if isinstance(w, QGroupBox):
                text = str(w.title())
            elif hasattr(w, "text"):
                text = str(w.text())  # type: ignore[attr-defined]
            low = text.lower()
            if any(k in low for k in _STALE_KEYWORDS) and text.strip():
                found.append(f"{type(w).__name__}:{text!r}")
        except Exception:
            continue
    return found


def _log_delta(before: str, after: str) -> str:
    # Helper: new log lines appended after a UI action.
    if len(after) > len(before) and after.startswith(before):
        return after[len(before) :]
    return after


def test_run_uses_ui_params():
    # Arrange: open BPSK capture programmatically (no file dialog).
    if not BPSK_IQ.exists():
        pytest.skip(f"missing synthetic data: {BPSK_IQ} (run generate_test_data.py)")
    app = _get_app()
    window = _make_window()
    try:
        window.open_file(str(BPSK_IQ))
        app.processEvents()
        assert window.current_file is not None, "open_file must set current_file"

        # Act: select QPSK in the UI then run.
        window.mod_combo.setCurrentText("QPSK")
        app.processEvents()
        window.run_analysis()
        app.processEvents()

        # Assert: pipeline request honored the live combo value.
        assert window.last_report is not None, "run must produce last_report"
        mode = window.last_report.get("demodulation", {}).get("mode")
        assert mode == "QPSK", f"last_report mode {mode!r} != UI selection 'QPSK'"

        # Assert: results table reflects the same live params.
        cell = _table_value(window, "Demod mode")
        assert cell == "QPSK", f"results_table Demod mode {cell!r} != 'QPSK'"
    finally:
        window.close()


def test_params_change_marks_stale_or_updates():
    # Arrange: run once to establish a baseline, then mutate a param.
    if not BPSK_IQ.exists():
        pytest.skip(f"missing synthetic data: {BPSK_IQ} (run generate_test_data.py)")
    app = _get_app()
    window = _make_window()
    try:
        window.open_file(str(BPSK_IQ))
        app.processEvents()
        window.mod_combo.setCurrentText("BPSK")
        app.processEvents()
        window.run_analysis()
        app.processEvents()
        assert window.last_report is not None

        before_log = window.log_console.toPlainText()
        old_rate = float(window.sample_rate_spin.value())
        new_rate = old_rate + 5000.0 if old_rate < 50_000_000 else old_rate - 5000.0

        # Act: change sample rate via the UI control.
        window.sample_rate_spin.setValue(new_rate)
        app.processEvents()
        app.processEvents()

        # Assert (either branch passes, to allow UI-agent freedom):
        # stale badge visible OR log notes the change — fail if neither.
        stale_texts = _visible_stale_texts(window)
        stale_visible = len(stale_texts) > 0
        # Explicit badge check (visible => stale) for clearer failure output.
        try:
            badge = getattr(window, "stale_badge", None)
            badge_visible = bool(badge is not None and badge.isVisible())
        except Exception:
            badge_visible = False
        after_log = window.log_console.toPlainText()
        delta = _log_delta(before_log, after_log).lower()
        log_notes_change = any(k in delta for k in _STALE_KEYWORDS)

        assert stale_visible or badge_visible or log_notes_change, (
            "changing sample_rate must mark stale (visible badge) OR log a change "
            f"note; badge_visible={badge_visible} stale_texts={stale_texts!r} "
            f"delta={delta[-500:]!r}"
        )
    finally:
        window.close()


def test_bitstream_shows_real_bits():
    # Arrange: run BPSK analysis to extract a known bitstream.
    if not BPSK_IQ.exists():
        pytest.skip(f"missing synthetic data: {BPSK_IQ} (run generate_test_data.py)")
    app = _get_app()
    window = _make_window()
    try:
        window.open_file(str(BPSK_IQ))
        app.processEvents()
        window.mod_combo.setCurrentText("BPSK")
        app.processEvents()

        # Act
        window.run_analysis()
        app.processEvents()

        # Assert: placeholder is gone and a real 0/1 preview is shown.
        text = window.bitstream_text.toPlainText()
        assert "Raw bit vector is held" not in text, (
            "bitstream_text still shows pipeline-held placeholder; must show real bits"
        )
        runs = re.findall(r"[01]{32,}", text)
        assert runs, (
            "bitstream_text must contain a 0/1 preview run "
            f"(>=32 chars); got {text[:300]!r}"
        )
        longest = max(len(r) for r in runs)
        assert longest >= 64, (
            f"0/1 preview too short ({longest} chars); expected >=64 real bits"
        )
    finally:
        window.close()


def test_switching_files_changes_plots():
    # Arrange: run on bpsk.iq and snapshot the time plot.
    if not BPSK_IQ.exists() or not TONE_IQ.exists():
        pytest.skip("missing bpsk.iq / tone.iq (run generate_test_data.py)")
    app = _get_app()
    window = _make_window()
    try:
        window.open_file(str(BPSK_IQ))
        app.processEvents()
        window.run_analysis()
        app.processEvents()
        before = _snapshot_time_curves(window)
        assert len(before) > 0, "time_plot must contain curves after first run"

        # Act: open a different file and re-run.
        window.open_file(str(TONE_IQ))
        app.processEvents()
        window.run_analysis()
        app.processEvents()
        after = _snapshot_time_curves(window)
        assert len(after) > 0, "time_plot must contain curves after second run"

        # Assert: plot data follows the newly opened file (not frozen).
        if len(before) != len(after):
            differ = True
        else:
            differ = any(
                b.shape != a.shape or not np.array_equal(b, a)
                for b, a in zip(before, after)
            )
        assert differ, "time_plot data unchanged after switching files (frozen plots)"
    finally:
        window.close()
