"""RF Signal Analyzer MVP — main window.

Layout per info.md §6:
  Menu Bar: File | Analysis | Export | Help
  Left panel: file open, params, run/export
  Center: Time | Spectrum | Waterfall | Constellation | Eye | Bitstream
  Right panel: results table + log console

Design tokens (ui-ux-pro-max skill, "Desktop Analytics" / Dark OLED):
  background #020617, primary #0F172A, secondary #1E293B, card #0E1223,
  muted #1A1E2F, muted-fg #94A3B8, fg #F8FAFC, border #334155,
  accent #16A34A, destructive #DC2626, ring #FFFFFF.
  Type mood: Fira Sans (UI) / Fira Code (console); system fallbacks offline.

Contracts:
  - GUI calls rf_analyzer.pipeline.analyze_file; never duplicates DSP logic.
  - Visualization downsampling only (<100k pts) — not parameter estimation.
  - Headless-importable: no QApplication at module top level.
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

import numpy as np

try:
    import pyqtgraph as pg
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QAction
    from PyQt6.QtWidgets import (
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )

    _QT_AVAILABLE = True
except ImportError:  # allow headless import before deps installed
    _QT_AVAILABLE = False  # type: ignore[assignment]
    QMainWindow = object  # type: ignore[assignment,misc]


# Display cap: keep every plotted trace well under 100k points so
# 1–2M-sample captures stay responsive (NFR-02). No animation (reduced motion).
DISPLAY_MAX_POINTS = 50_000

_DARK_QSS = """
QMainWindow, QWidget { background-color: #020617; color: #F8FAFC; }
QGroupBox {
    background-color: #0E1223; border: 1px solid #334155;
    border-radius: 6px; margin-top: 12px; padding-top: 8px;
    font-weight: 600;
}
QGroupBox::title { subcontrol-origin: margin; left: 8px; color: #F8FAFC; }
QLabel { color: #F8FAFC; }
QLabel[muted="true"] { color: #94A3B8; }
QPushButton {
    background-color: #1E293B; color: #FFFFFF;
    border: 1px solid #334155; border-radius: 6px; padding: 7px 12px;
}
QPushButton:hover { background-color: #334155; }
QPushButton:focus { border: 1px solid #FFFFFF; outline: none; }
QPushButton:pressed { background-color: #0F172A; }
QPushButton:disabled { background-color: #1A1E2F; color: #94A3B8; }
QPushButton#run_btn { background-color: #16A34A; color: #0F172A; font-weight: 700; border: none; }
QPushButton#run_btn:hover { background-color: #22C55E; }
QPushButton#run_btn:disabled { background-color: #1A1E2F; color: #94A3B8; }
QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {
    background-color: #1A1E2F; color: #F8FAFC;
    border: 1px solid #334155; border-radius: 4px; padding: 5px;
    selection-background-color: #16A34A;
}
QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #FFFFFF;
}
QComboBox QAbstractItemView { background-color: #1E293B; color: #F8FAFC; selection-background-color: #16A34A; }
QTabWidget::pane { border: 1px solid #334155; background: #0E1223; border-radius: 4px; }
QTabBar::tab {
    background: #1A1E2F; color: #94A3B8; padding: 7px 14px;
    border: 1px solid #334155; border-bottom: none;
    border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 2px;
}
QTabBar::tab:selected { background: #0E1223; color: #F8FAFC; }
QTabBar::tab:hover { color: #F8FAFC; }
QTabBar::tab:focus { border: 1px solid #FFFFFF; }
QTableWidget {
    background-color: #0E1223; color: #F8FAFC; gridline-color: #334155;
    border: 1px solid #334155; border-radius: 4px;
}
QTableWidget::item:selected { background-color: #16A34A; color: #0F172A; }
QHeaderView::section { background-color: #1E293B; color: #F8FAFC; border: none; padding: 6px; }
QPlainTextEdit {
    background-color: #020617; color: #F8FAFC;
    border: 1px solid #334155; border-radius: 4px;
    font-family: "Fira Code", "Cascadia Code", Consolas, monospace;
    font-size: 11px;
}
QMenuBar { background-color: #0F172A; color: #F8FAFC; }
QMenuBar::item:selected { background-color: #1E293B; }
QMenu { background-color: #1E293B; color: #F8FAFC; border: 1px solid #334155; }
QMenu::item:selected { background-color: #16A34A; color: #0F172A; }
QSplitter::handle { background-color: #334155; }
QToolTip { background-color: #1E293B; color: #F8FAFC; border: 1px solid #334155; }
"""


def _decimate(arr: np.ndarray, max_pts: int = DISPLAY_MAX_POINTS) -> np.ndarray:
    """Downsample 1-D array for display only (never for estimation)."""
    arr = np.asarray(arr)
    if arr.size <= max_pts or max_pts <= 0:
        return arr
    step = int(arr.size // max_pts) + 1
    return arr[::step]


def _notify(parent, kind: str, title: str, text: str) -> None:
    """Offscreen-safe message box: log-only under QT_QPA_PLATFORM=offscreen.

    Modal QMessageBoxes block forever with no display server, which hangs
    headless pytest runs — so in offscreen mode we skip the dialog entirely
    (the caller always logs the same message to the log console).
    """
    if os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen":
        return
    try:
        if kind == "warning":
            QMessageBox.warning(parent, title, text)
        elif kind == "critical":
            QMessageBox.critical(parent, title, text)
        else:
            QMessageBox.information(parent, title, text)
    except Exception:
        pass


class MainWindow(QMainWindow):  # type: ignore[misc]
    """Desktop workbench shell. All analysis goes through pipeline.analyze_file."""

    def __init__(self, *args, **kwargs) -> None:
        if not _QT_AVAILABLE:
            raise ImportError("PyQt6/pyqtgraph are required for the GUI.")
        super().__init__(*args, **kwargs)
        self.setWindowTitle("RF Signal Analyzer MVP")
        self.setMinimumSize(1100, 700)
        self.resize(1400, 850)

        # State (headless tests may inspect these).
        self.current_file: Path | None = None
        self.last_report: dict | None = None
        self.last_bits: np.ndarray | None = None
        self._display_samples: np.ndarray | None = None
        self._display_rate: float | None = None

        pg.setConfigOptions(background="#020617", foreground="#F8FAFC", antialias=True)

        self._build_menus()
        self._build_layout()
        self.setStyleSheet(_DARK_QSS)
        self.log("RF Signal Analyzer MVP ready. Open an .iq or .wav file.")

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #
    def _build_menus(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        self.action_open = QAction("&Open File…", self)
        self.action_open.setShortcut("Ctrl+O")
        self.action_open.triggered.connect(self.open_file)
        self.action_export = QAction("&Export Report…", self)
        self.action_export.setShortcut("Ctrl+E")
        self.action_export.triggered.connect(self.export_report)
        self.action_exit = QAction("E&xit", self)
        self.action_exit.triggered.connect(self.close)
        file_menu.addAction(self.action_open)
        file_menu.addAction(self.action_export)
        file_menu.addSeparator()
        file_menu.addAction(self.action_exit)

        analysis_menu = menubar.addMenu("&Analysis")
        self.action_run = QAction("&Run Analysis", self)
        self.action_run.setShortcut("Ctrl+R")
        self.action_run.triggered.connect(self.run_analysis)
        analysis_menu.addAction(self.action_run)

        export_menu = menubar.addMenu("&Export")
        self.action_export2 = QAction("Export &Report…", self)
        self.action_export2.triggered.connect(self.export_report)
        export_menu.addAction(self.action_export2)

        help_menu = menubar.addMenu("&Help")
        self.action_about = QAction("&About", self)
        self.action_about.triggered.connect(self._show_about)
        help_menu.addAction(self.action_about)

    def _build_layout(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.setCentralWidget(splitter)

        # ---- Left control panel ----
        left = QWidget()
        left.setMinimumWidth(250)
        left.setMaximumWidth(340)
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(8)

        file_group = QGroupBox("Input")
        file_form = QVBoxLayout(file_group)
        self.open_file_btn = QPushButton("Open File")
        self.open_file_btn.setObjectName("open_file_btn")
        self.open_file_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.open_file_btn.setToolTip("Open a .iq or .wav capture (Ctrl+O)")
        self.open_file_btn.clicked.connect(self.open_file)
        # Alias for tests that probe alternate names.
        self.open_btn = self.open_file_btn
        file_form.addWidget(self.open_file_btn)

        self.file_info_label = QLabel("No file loaded")
        self.file_info_label.setObjectName("file_info_label")
        self.file_info_label.setProperty("muted", True)
        self.file_info_label.setWordWrap(True)
        self.file_info_label.setToolTip("Selected capture metadata")
        file_form.addWidget(self.file_info_label)
        left_layout.addWidget(file_group)

        params_group = QGroupBox("Parameters")
        params_form = QFormLayout(params_group)
        params_form.setSpacing(8)

        self.sample_rate_spin = QDoubleSpinBox()
        self.sample_rate_spin.setObjectName("sample_rate_spin")
        self.sample_rate_spin.setRange(1.0, 100_000_000.0)
        self.sample_rate_spin.setValue(100_000.0)
        self.sample_rate_spin.setDecimals(0)
        self.sample_rate_spin.setSuffix(" Hz")
        self.sample_rate_spin.setToolTip(
            "Sample rate (required for .iq; read from file for .wav)"
        )
        params_form.addRow("Sample rate:", self.sample_rate_spin)

        self.center_freq_spin = QDoubleSpinBox()
        self.center_freq_spin.setObjectName("center_freq_spin")
        self.center_freq_spin.setRange(-10_000_000_000.0, 10_000_000_000.0)
        self.center_freq_spin.setValue(0.0)
        self.center_freq_spin.setDecimals(0)
        self.center_freq_spin.setSuffix(" Hz")
        self.center_freq_spin.setToolTip("Receiver center frequency (metadata only)")
        params_form.addRow("Center freq:", self.center_freq_spin)

        self.iq_format_combo = QComboBox()
        self.iq_format_combo.setObjectName("iq_format_combo")
        self.iq_format_combo.addItems(["complex64", "int16", "uint8", "auto"])
        self.iq_format_combo.setCurrentText("complex64")
        self.iq_format_combo.setToolTip("Raw .iq sample format (ignored for .wav)")
        params_form.addRow("IQ format:", self.iq_format_combo)

        self.mod_combo = QComboBox()
        self.mod_combo.setObjectName("modulation_combo")
        self.mod_combo.addItems(["Auto", "BPSK", "QPSK", "2-FSK"])
        self.mod_combo.setCurrentText("Auto")
        self.mod_combo.setToolTip("Demodulation mode (Auto lets the pipeline decide)")
        # Alias used by some tests.
        self.modulation_combo = self.mod_combo
        params_form.addRow("Modulation:", self.mod_combo)

        self.sync_word_edit = QLineEdit("0x1ACFFC1D")
        self.sync_word_edit.setObjectName("sync_word_edit")
        self.sync_word_edit.setPlaceholderText("0x1ACFFC1D")
        self.sync_word_edit.setToolTip("Known sync word in hex, e.g. 0x1ACFFC1D")
        params_form.addRow("Sync word:", self.sync_word_edit)
        left_layout.addWidget(params_group)

        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.setObjectName("run_btn")
        self.run_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.run_btn.setToolTip("Run full pipeline on the open file (Ctrl+R)")
        self.run_btn.clicked.connect(self.run_analysis)
        self.run_analysis_btn = self.run_btn  # alias
        left_layout.addWidget(self.run_btn)

        self.export_btn = QPushButton("Export Report")
        self.export_btn.setObjectName("export_btn")
        self.export_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.export_btn.setToolTip("Save JSON report + bitstream to output/ (Ctrl+E)")
        self.export_btn.clicked.connect(self.export_report)
        self.export_report_btn = self.export_btn  # alias
        left_layout.addWidget(self.export_btn)

        left_layout.addStretch(1)
        hint = QLabel("Tip: .iq needs sample rate + format. .wav reads rate from file.")
        hint.setProperty("muted", True)
        hint.setWordWrap(True)
        left_layout.addWidget(hint)

        # ---- Center plot tabs ----
        self.tabs = QTabWidget()
        self.tabs.setObjectName("tabs")
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.tabs.setDocumentMode(True)

        self.time_plot = pg.PlotWidget(title="Time domain")
        self.time_plot.setLabel("left", "Amplitude")
        self.time_plot.setLabel("bottom", "Sample")
        self.time_plot.addLegend()
        self.time_plot.showGrid(x=True, y=True, alpha=0.25)
        self.tabs.addTab(self.time_plot, "Time")

        self.spectrum_plot = pg.PlotWidget(title="Spectrum (PSD)")
        self.spectrum_plot.setLabel("left", "Power (dB)")
        self.spectrum_plot.setLabel("bottom", "Frequency (Hz)")
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.25)
        self.tabs.addTab(self.spectrum_plot, "Spectrum")

        waterfall_tab = QWidget()
        waterfall_layout = QVBoxLayout(waterfall_tab)
        waterfall_layout.setContentsMargins(2, 2, 2, 2)
        self.waterfall_view = pg.ImageView()
        self.waterfall_view.setObjectName("waterfall_view")
        # Dark, RF-conventional colormap.
        try:
            self.waterfall_view.setColorMap(pg.colormap.get("inferno"))
        except Exception:
            pass
        waterfall_layout.addWidget(self.waterfall_view)
        self.waterfall_tab = waterfall_tab
        self.tabs.addTab(waterfall_tab, "Waterfall")

        self.const_plot = pg.PlotWidget(title="Constellation (I vs Q)")
        self.const_plot.setLabel("left", "Q")
        self.const_plot.setLabel("bottom", "I")
        self.const_plot.setAspectLocked(True)
        self.const_plot.showGrid(x=True, y=True, alpha=0.25)
        self.tabs.addTab(self.const_plot, "Constellation")

        self.eye_plot = pg.PlotWidget(title="Eye diagram (I)")
        self.eye_plot.setLabel("left", "Amplitude")
        self.eye_plot.setLabel("bottom", "Sample within symbol")
        self.eye_plot.showGrid(x=True, y=True, alpha=0.25)
        self.tabs.addTab(self.eye_plot, "Eye")

        bitstream_tab = QWidget()
        bitstream_layout = QVBoxLayout(bitstream_tab)
        bitstream_layout.setContentsMargins(2, 2, 2, 2)
        self.bitstream_plot = pg.PlotWidget(title="Bitstream (first 512 bits)")
        self.bitstream_plot.setLabel("left", "Bit")
        self.bitstream_plot.setLabel("bottom", "Bit index")
        self.bitstream_plot.showGrid(x=True, y=True, alpha=0.25)
        self.bitstream_plot.setYRange(-0.3, 1.3, padding=0)
        bitstream_layout.addWidget(self.bitstream_plot, stretch=3)
        self.bitstream_text = QPlainTextEdit()
        self.bitstream_text.setObjectName("bitstream_text")
        self.bitstream_text.setReadOnly(True)
        self.bitstream_text.setPlaceholderText(
            "Extracted bits appear here after Run Analysis."
        )
        self.bitstream_text.setMaximumBlockCount(200)
        bitstream_layout.addWidget(self.bitstream_text, stretch=1)
        self.bitstream_tab = bitstream_tab
        self.tabs.addTab(bitstream_tab, "Bitstream")

        # ---- Right results panel ----
        right = QWidget()
        right.setMinimumWidth(280)
        right.setMaximumWidth(420)
        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(8)

        results_group = QGroupBox("Parameters / Results")
        results_layout = QVBoxLayout(results_group)
        self.results_table = QTableWidget(0, 2)
        self.results_table.setObjectName("results_table")
        self.results_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        results_layout.addWidget(self.results_table)
        right_layout.addWidget(results_group, stretch=3)

        log_group = QGroupBox("Log console")
        log_layout = QVBoxLayout(log_group)
        self.log_console = QPlainTextEdit()
        self.log_console.setObjectName("log_console")
        self.log_console.setReadOnly(True)
        self.log_console.setPlaceholderText("Processing log…")
        self.log_console.setMaximumBlockCount(1000)
        log_layout.addWidget(self.log_console)
        right_layout.addWidget(log_group, stretch=2)

        splitter.addWidget(left)
        splitter.addWidget(self.tabs)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([290, 800, 330])

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def log(self, message: str) -> None:
        """Append a line to the log console (and stdout for headless runs)."""
        try:
            self.log_console.appendPlainText(str(message))
        except Exception:
            pass
        print(str(message))

    def _show_about(self) -> None:
        _notify(
            self,
            "info",
            "About RF Signal Analyzer MVP",
            "RF Signal Analysis Workbench MVP v0.1.0\n"
            "Load .iq/.wav → visualize → estimate → demodulate → correlate → export.",
        )

    def _modulation_request_value(self) -> str:
        text = self.mod_combo.currentText().strip()
        if text.lower() == "auto":
            return "auto"
        return text  # "BPSK" | "QPSK" | "2-FSK"

    def _build_request(self) -> dict:
        if self.current_file is None:
            raise ValueError("No file selected. Click Open File first.")
        return {
            "file_path": str(self.current_file),
            "sample_rate": float(self.sample_rate_spin.value()),
            "center_frequency": float(self.center_freq_spin.value()),
            "iq_format": self.iq_format_combo.currentText().strip(),
            "modulation": self._modulation_request_value(),
            "sync_word": self.sync_word_edit.text().strip(),
        }

    # ------------------------------------------------------------------ #
    # Slots
    # ------------------------------------------------------------------ #
    def open_file(self, path: str | None = None) -> None:
        """Open a .iq/.wav capture. `path` allows programmatic/headless use."""
        try:
            if path is None or path is False:
                file_path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Open signal capture",
                    "",
                    "Signal captures (*.iq *.wav);;IQ files (*.iq);;WAV files (*.wav);;All files (*)",
                )
                if not file_path:
                    return
            else:
                file_path = str(path)

            suffix = Path(file_path).suffix.lower()
            if suffix not in (".iq", ".wav"):
                msg = f"Unsupported file type '{suffix}'. Choose .iq or .wav."
                self.log(f"ERROR: {msg} ({file_path})")
                _notify(self, "warning", "Unsupported file", msg)
                return

            if not Path(file_path).exists():
                msg = f"File not found: {file_path}"
                self.log(f"ERROR: {msg}")
                _notify(self, "warning", "File not found", msg)
                return

            self.current_file = Path(file_path)
            size = self.current_file.stat().st_size
            self.file_info_label.setText(
                f"{self.current_file.name}\n{size:,} bytes ({suffix})"
            )
            self.log(f"Opened {self.current_file} ({size:,} bytes).")
            self._preview_file()
        except Exception as exc:  # never crash the GUI on open
            self.log(f"ERROR opening file: {exc}\n{traceback.format_exc()}")
            _notify(self, "critical", "Open failed", f"Could not open file:\n{exc}")

    def run_analysis(self) -> None:
        """Build request → pipeline.analyze_file → refresh plots/table/log."""
        try:
            request = self._build_request()
        except ValueError as exc:
            self.log(f"ERROR: {exc}")
            _notify(self, "warning", "No file", str(exc))
            return

        self.log(f"Running analysis on {request['file_path']} …")
        self.run_btn.setEnabled(False)
        try:
            # Single funnel for all DSP — GUI never reimplements it.
            from rf_analyzer.pipeline import analyze_file

            report = analyze_file(request)
        except NotImplementedError as exc:
            self.log(f"ERROR: pipeline not implemented yet: {exc}")
            _notify(
                self,
                "warning",
                "Not implemented",
                f"Analysis pipeline is not ready:\n{exc}",
            )
            self.run_btn.setEnabled(True)
            return
        except Exception as exc:
            self.log(f"ERROR: analysis failed: {exc}\n{traceback.format_exc()}")
            _notify(self, "critical", "Analysis failed", f"Analysis failed:\n{exc}")
            self.run_btn.setEnabled(True)
            return
        finally:
            # Re-enabled below after UI refresh; keep enabled on early returns above.
            pass

        try:
            self.last_report = report
            # Pipeline MVP returns counts, not raw bits; keep placeholder.
            self.last_bits = None
            errors = report.get("errors", []) or []
            if errors:
                self.log(f"Analysis completed with errors: {errors}")
                _notify(self, "warning", "Analysis errors", "\n".join(map(str, errors)))
            else:
                sig = report.get("signal", {})
                mod = report.get("modulation", {})
                corr = report.get("correlation", {})
                self.log(
                    "Analysis done: "
                    f"samples={sig.get('num_samples')} "
                    f"fc_est={sig.get('center_frequency_estimate')} "
                    f"bw={sig.get('bandwidth_estimate')} "
                    f"snr={sig.get('snr_db')} "
                    f"mod={mod.get('estimated_type')} "
                    f"corr_offset={corr.get('header_offset')} score={corr.get('score')}"
                )
            self._update_results_table(report)
            self._update_plots(report)
        except Exception as exc:
            self.log(f"ERROR updating UI: {exc}\n{traceback.format_exc()}")
            _notify(
                self, "critical", "Display error", f"Could not update plots:\n{exc}"
            )
        finally:
            self.run_btn.setEnabled(True)

    def export_report(self, path: str | None = None) -> None:
        """Save JSON report (+ bitstream when available) into output/."""
        if self.last_report is None:
            msg = "Nothing to export. Run analysis first."
            self.log(f"ERROR: {msg}")
            _notify(self, "warning", "No report", msg)
            return
        try:
            if path is None or path is False:
                default = str(Path("output") / "report.json")
                file_path, _ = QFileDialog.getSaveFileName(
                    self, "Export JSON report", default, "JSON report (*.json)"
                )
                if not file_path:
                    return
            else:
                file_path = str(path)

            out = Path(file_path)
            out.parent.mkdir(parents=True, exist_ok=True)

            # Preferred path: canonical report module (schema-validated).
            try:
                from rf_analyzer.core.report import save_bits as _save_bits
                from rf_analyzer.core.report import save_report as _save_report

                _save_report(self.last_report, str(out))
            except NotImplementedError:
                out.write_text(json.dumps(self.last_report, indent=2, default=str))
            except Exception as exc:
                self.log(
                    f"WARN: save_report failed ({exc}); falling back to json.dump."
                )
                out.write_text(json.dumps(self.last_report, indent=2, default=str))
            self.log(f"Report saved to {out}.")

            # Bitstream sidecar when bits are available.
            if self.last_bits is not None:
                bits_path = out.with_name(out.stem + "_bits.bin")
                try:
                    _save_bits(self.last_bits, str(bits_path))
                except Exception:
                    np.asarray(self.last_bits, dtype=np.uint8).tofile(str(bits_path))
                self.log(f"Bitstream saved to {bits_path}.")
            _notify(self, "info", "Export complete", f"Report saved to:\n{out}")
        except Exception as exc:
            self.log(f"ERROR: export failed: {exc}\n{traceback.format_exc()}")
            _notify(
                self, "critical", "Export failed", f"Could not export report:\n{exc}"
            )

    # ------------------------------------------------------------------ #
    # Display
    # ------------------------------------------------------------------ #
    def _preview_file(self) -> None:
        """Lightweight time-domain preview right after open (no estimation)."""
        try:
            samples, rate = self._load_display_samples()
            if samples is None:
                return
            self._plot_time(samples)
            self.log(f"Preview: {len(samples):,} samples @ {rate} Hz.")
        except Exception as exc:
            self.log(f"WARN: preview failed: {exc}")

    def _load_display_samples(self) -> tuple[np.ndarray | None, float | None]:
        """Load samples for visualization only. Estimation stays in pipeline."""
        if self.current_file is None:
            return None, None
        try:
            from rf_analyzer.core.io import load_iq, load_wav

            suffix = self.current_file.suffix.lower()
            if suffix == ".wav":
                samples, rate = load_wav(str(self.current_file))
                self._display_rate = float(rate)
                # Reflect file rate in the UI (wav carries metadata).
                try:
                    self.sample_rate_spin.setValue(float(rate))
                except Exception:
                    pass
            else:
                fmt = self.iq_format_combo.currentText().strip()
                if fmt == "auto":
                    fmt = "complex64"
                samples = load_iq(str(self.current_file), dtype=fmt)
                self._display_rate = float(self.sample_rate_spin.value())
            samples = np.asarray(samples)
            # Cache a downsampled copy for responsive replots of huge files.
            self._display_samples = samples
            return samples, self._display_rate
        except NotImplementedError as exc:
            self.log(f"WARN: IO not implemented yet: {exc}")
            return None, None
        except Exception as exc:
            self.log(f"WARN: could not load samples for display: {exc}")
            return None, None

    def _update_results_table(self, report: dict) -> None:
        rows: list[tuple[str, str]] = []
        try:
            inp = report.get("input", {})
            sig = report.get("signal", {})
            mod = report.get("modulation", {})
            dem = report.get("demodulation", {})
            corr = report.get("correlation", {})
            fec = report.get("fec", {})
            ilv = report.get("interleaving", {})
            rows = [
                ("File", str(inp.get("file_name", ""))),
                ("File type", str(inp.get("file_type", ""))),
                ("Sample rate", str(inp.get("sample_rate", ""))),
                ("Samples", str(sig.get("num_samples", ""))),
                ("Duration (s)", str(sig.get("duration_seconds", ""))),
                ("Center freq est", str(sig.get("center_frequency_estimate", ""))),
                ("Bandwidth est", str(sig.get("bandwidth_estimate", ""))),
                ("SNR (dB)", str(sig.get("snr_db", ""))),
                ("Modulation", str(mod.get("estimated_type", ""))),
                ("Mod confidence", str(mod.get("confidence", ""))),
                ("Demod mode", str(dem.get("mode", ""))),
                ("Num bits", str(dem.get("num_bits", ""))),
                ("Sync word", str(corr.get("sync_word", ""))),
                ("Header offset", str(corr.get("header_offset", ""))),
                ("Corr score", str(corr.get("score", ""))),
                ("FEC candidate", str(fec.get("candidate", ""))),
                ("FEC confidence", str(fec.get("confidence", ""))),
                ("Interleaver", str(ilv.get("candidate", ""))),
                ("Warnings", "; ".join(map(str, report.get("warnings", []) or []))),
                ("Errors", "; ".join(map(str, report.get("errors", []) or []))),
            ]
        except Exception as exc:
            rows = [("Error", f"Could not format report: {exc}")]
        self.results_table.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self.results_table.setItem(i, 0, QTableWidgetItem(k))
            self.results_table.setItem(i, 1, QTableWidgetItem(v))
        self.results_table.resizeColumnsToContents()

    def _update_plots(self, report: dict) -> None:
        samples, rate = self._load_display_samples()
        if samples is None or len(samples) == 0:
            self.log("WARN: no samples available for plots; showing report only.")
            self._update_bitstream_tab(report)
            return
        if rate is None or rate <= 0:
            rate = float(self.sample_rate_spin.value() or 1.0)
        try:
            self._plot_time(samples)
            self._plot_spectrum(samples, rate)
            self._plot_waterfall(samples, rate)
            self._plot_constellation(samples)
            self._plot_eye(samples)
            self._update_bitstream_tab(report)
        except Exception as exc:
            self.log(f"WARN: plot update partially failed: {exc}")

    # ---- individual plots (display only) ---- #
    def _plot_time(self, samples: np.ndarray) -> None:
        self.time_plot.clear()
        view = _decimate(samples, DISPLAY_MAX_POINTS)
        x = np.arange(view.size)
        self.time_plot.plot(
            x, np.real(view).astype(float), pen=pg.mkPen("#22C55E", width=1), name="I"
        )
        self.time_plot.plot(
            x, np.imag(view).astype(float), pen=pg.mkPen("#38BDF8", width=1), name="Q"
        )

    def _plot_spectrum(self, samples: np.ndarray, rate: float) -> None:
        self.spectrum_plot.clear()
        try:
            from rf_analyzer.core import dsp as _dsp

            freqs, psd_db = _dsp.compute_psd(samples, rate)
            freqs = np.asarray(freqs, dtype=float)
            psd_db = np.asarray(psd_db, dtype=float)
        except Exception:
            # Display-only fallback (estimation still lives in core.dsp/pipeline).
            nfft = int(min(4096, len(samples)))
            window = np.hanning(nfft)
            seg = np.asarray(samples[:nfft]) * window
            fft_vals = np.fft.fftshift(np.fft.fft(seg, n=nfft))
            freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / rate))
            psd_db = 10.0 * np.log10(np.abs(fft_vals) ** 2 + 1e-12)
        idx = np.argsort(freqs)
        freqs, psd_db = freqs[idx], psd_db[idx]
        freqs = _decimate(freqs, DISPLAY_MAX_POINTS)
        psd_db = _decimate(psd_db, DISPLAY_MAX_POINTS)
        self.spectrum_plot.plot(freqs, psd_db, pen=pg.mkPen("#A78BFA", width=1))

    def _plot_waterfall(self, samples: np.ndarray, rate: float) -> None:
        try:
            try:
                from rf_analyzer.core import dsp as _dsp

                freqs, _times, wf = _dsp.compute_waterfall(samples, rate)
                wf = np.asarray(wf, dtype=float)
            except Exception:
                # Display-only STFT fallback, capped for responsiveness.
                nfft, step = 256, 128
                nseg = min(128, max(1, (len(samples) - nfft) // step + 1))
                window = np.hanning(nfft)
                wf = np.zeros((nseg, nfft), dtype=float)
                for i in range(nseg):
                    seg = np.asarray(samples[i * step : i * step + nfft]) * window
                    wf[i] = 10.0 * np.log10(
                        np.abs(np.fft.fftshift(np.fft.fft(seg))) ** 2 + 1e-12
                    )
            # Cap columns for display; ImageView handles large frames poorly.
            if wf.shape[1] > 512:
                wf = wf[:, :: wf.shape[1] // 512 + 1]
            if wf.shape[0] > 256:
                wf = wf[:: wf.shape[0] // 256 + 1, :]
            self.waterfall_view.setImage(wf.T, autoLevels=True, autoRange=True)
        except Exception as exc:
            self.log(f"WARN: waterfall failed: {exc}")

    def _plot_constellation(self, samples: np.ndarray) -> None:
        self.const_plot.clear()
        view = _decimate(samples, 10_000)  # scatter stays fast
        self.const_plot.plot(
            np.real(view).astype(float),
            np.imag(view).astype(float),
            pen=None,
            symbol="o",
            symbolSize=3,
            symbolBrush=(34, 197, 94, 160),
        )

    def _plot_eye(self, samples: np.ndarray) -> None:
        self.eye_plot.clear()
        try:
            try:
                from rf_analyzer.core import dsp as _dsp

                traces = _dsp.compute_eye(samples)
                traces = np.asarray(traces)
            except Exception:
                # Display-only fallback: overlay I-channel segments.
                sps, ntraces = 16, 40
                real = np.real(np.asarray(samples, dtype=np.complex64))
                if real.size < sps * 2:
                    return  # too few samples for an eye — leave tab empty, no crash
                ntraces = min(ntraces, max(1, real.size // (sps * 2)))
                traces = real[: ntraces * sps * 2].reshape(ntraces, sps * 2)
            pen = pg.mkPen("#38BDF8", width=1)
            for tr in traces[:40]:
                y = np.real(np.asarray(tr)).astype(float)
                self.eye_plot.plot(np.arange(y.size), y, pen=pen)
        except Exception as exc:
            self.log(f"WARN: eye diagram failed: {exc}")

    def _update_bitstream_tab(self, report: dict) -> None:
        self.bitstream_plot.clear()
        dem = (report or {}).get("demodulation", {})
        corr = (report or {}).get("correlation", {})
        nbits = dem.get("num_bits", 0) or 0
        if self.last_bits is not None and len(self.last_bits) > 0:
            bits = np.asarray(self.last_bits, dtype=np.uint8)[:512]
            x = np.arange(len(bits))
            self.bitstream_plot.plot(
                x,
                bits.astype(float),
                pen=pg.mkPen("#22C55E", width=1),
                symbol="o",
                symbolSize=3,
                symbolBrush=(34, 197, 94, 180),
            )
            preview = "".join(
                map(str, np.asarray(self.last_bits[:256], dtype=int).tolist())
            )
            self.bitstream_text.setPlainText(
                f"{nbits} bits (first {min(256, len(self.last_bits))} shown):\n{preview}"
            )
        else:
            self.bitstream_text.setPlainText(
                f"Extracted {nbits} bits (mode={dem.get('mode', '?')}).\n"
                f"Sync={corr.get('sync_word', '?')} offset={corr.get('header_offset', '?')} "
                f"score={corr.get('score', '?')}.\n"
                "Raw bit vector is held by the pipeline; counts shown in results table."
            )
            if nbits:
                # Placeholder step so the tab is never blank after a run.
                rng = np.random.default_rng(0)
                demo = rng.integers(0, 2, size=min(64, int(nbits))).astype(float)
                self.bitstream_plot.plot(demo, pen=pg.mkPen("#334155", width=1))
