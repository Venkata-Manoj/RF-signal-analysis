"""RF Signal Analyzer MVP — main window.

Layout per info.md §6:
  Menu Bar: File | Analysis | Export | Help
  Left panel: file open, params, run/export
  Center: Time | Spectrum | Waterfall | Constellation | Eye | Bitstream |
          Payload | Hypotheses
  Right panel: results table (whole report) + log console

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
    from PyQt6.QtGui import QAction, QFont
    from PyQt6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QInputDialog,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QSpinBox,
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
QPushButton#run_btn[stale="true"] { border: 2px solid #EA580C; background-color: #16A34A; }
QLabel#stale_badge {
    background-color: #EA580C; color: #000000;
    border: 1px solid #EA580C; border-radius: 4px; padding: 3px 8px;
    font-weight: 700;
}
QLabel#last_analysis_label { color: #94A3B8; font-size: 11px; }
QLabel#payload_badge {
    background-color: #1A1E2F; color: #94A3B8;
    border: 1px solid #334155; border-radius: 4px; padding: 6px 10px;
    font-weight: 700;
}
QLabel#payload_badge[verified="true"] {
    background-color: #14532D; color: #4ADE80; border: 1px solid #16A34A;
}
QLabel#payload_meta_label { color: #94A3B8; font-size: 11px; }
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


def _decimation_step(size: int, max_pts: int = DISPLAY_MAX_POINTS) -> int:
    """Stride that :func:`_decimate` would use, so display indices can be mapped.

    Kept next to ``_decimate`` and derived from the same expression: a burst
    marker placed with a different stride would silently point at the wrong
    samples, which is worse than not drawing it.
    """
    size = int(size)
    if size <= max_pts or max_pts <= 0:
        return 1
    return int(size // max_pts) + 1


def _fmt_value(value: object) -> str:
    """Render one report value for the results table.

    ``None`` becomes an em dash on purpose: the pipeline uses ``None`` for
    "not measured" and a real ``0`` for "measured as zero" (an uncoded capture
    really does have an EVM of 0.0%). Rendering both as ``0`` would let an
    absent measurement read as a perfect one.
    """
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, dict):
        if not value:
            return "—"
        return ", ".join(f"{k}={_fmt_value(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        if not value:
            return "—"
        return ", ".join(_fmt_value(v) for v in value)
    return str(value)


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
        # Last successful Run request (for STALE tracking + "Last analysis" label).
        self._last_request: dict | None = None
        self._suppress_param_signals: bool = False

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
        self.action_open_demo = QAction("Open &Demo Capture…", self)
        self.action_open_demo.setToolTip(
            "Pick one of the generated sample_data captures and load it"
        )
        self.action_open_demo.triggered.connect(self.open_demo_capture)
        self.action_export = QAction("&Export Report…", self)
        self.action_export.setShortcut("Ctrl+E")
        self.action_export.triggered.connect(self.export_report)
        self.action_exit = QAction("E&xit", self)
        self.action_exit.triggered.connect(self.close)
        file_menu.addAction(self.action_open)
        file_menu.addAction(self.action_open_demo)
        file_menu.addAction(self.action_export)
        file_menu.addSeparator()
        file_menu.addAction(self.action_exit)

        analysis_menu = menubar.addMenu("&Analysis")
        self.action_run = QAction("&Run Analysis", self)
        self.action_run.setShortcut("Ctrl+R")
        self.action_run.triggered.connect(self.run_analysis)
        analysis_menu.addAction(self.action_run)
        analysis_menu.addSeparator()
        self.action_batch = QAction("&Batch Folder Analysis…", self)
        self.action_batch.setToolTip(
            "Analyse every .iq/.wav in a folder and export a CSV + HTML summary"
        )
        self.action_batch.triggered.connect(self.run_batch_analysis)
        analysis_menu.addAction(self.action_batch)
        self.action_demo_all = QAction("Run Full &Demo (all sample captures)", self)
        self.action_demo_all.setToolTip(
            "Analyse every generated sample capture and print a one-screen summary"
        )
        self.action_demo_all.triggered.connect(self.run_demo_all)
        analysis_menu.addAction(self.action_demo_all)

        export_menu = menubar.addMenu("&Export")
        self.action_export2 = QAction("Export &Report…", self)
        self.action_export2.triggered.connect(self.export_report)
        export_menu.addAction(self.action_export2)
        self.action_export_csv = QAction("Batch Summary (&CSV)…", self)
        self.action_export_csv.triggered.connect(
            lambda: self.run_batch_analysis(fmt="csv")
        )
        self.action_export_html = QAction("Batch Summary (&HTML)…", self)
        self.action_export_html.triggered.connect(
            lambda: self.run_batch_analysis(fmt="html")
        )
        export_menu.addSeparator()
        export_menu.addAction(self.action_export_csv)
        export_menu.addAction(self.action_export_html)

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
        self.center_freq_spin.setToolTip(
            "Receiver center frequency (metadata only — recorded in report, "
            "not used for estimation)"
        )
        params_form.addRow("Center freq:", self.center_freq_spin)

        self.iq_format_combo = QComboBox()
        self.iq_format_combo.setObjectName("iq_format_combo")
        self.iq_format_combo.addItems(["complex64", "int16", "uint8", "int8", "auto"])
        self.iq_format_combo.setCurrentText("complex64")
        self.iq_format_combo.setToolTip(
            "Raw .iq sample format (ignored for .wav — rate comes from file). "
            "'auto' infers the dtype from the byte statistics of the capture."
        )
        params_form.addRow("IQ format:", self.iq_format_combo)

        self.mod_combo = QComboBox()
        self.mod_combo.setObjectName("modulation_combo")
        self.mod_combo.addItems(["Auto", "BPSK", "QPSK", "2-FSK", "16-QAM", "QAM"])
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

        self.decode_check = QCheckBox("Verify FEC + decode payload")
        self.decode_check.setObjectName("decode_check")
        self.decode_check.setChecked(True)
        self.decode_check.setToolTip(
            "Search de-interleavers and FEC decoders and only report a payload "
            "when a CRC-16 check passes. Uncheck for a faster run that reports "
            "candidate scores only."
        )
        params_form.addRow("Decode:", self.decode_check)

        self.frame_bits_spin = QSpinBox()
        self.frame_bits_spin.setObjectName("frame_bits_spin")
        self.frame_bits_spin.setRange(0, 10_000_000)
        self.frame_bits_spin.setValue(0)
        self.frame_bits_spin.setSpecialValueText("auto")
        self.frame_bits_spin.setToolTip(
            "Optional length of the coded region after the sync word. Leave at "
            "'auto' for a capture that ends at the frame boundary; set it when "
            "the recording continues past the end of the burst."
        )
        params_form.addRow("Frame bits:", self.frame_bits_spin)
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

        # ---- Payload tab: header/payload identification ----
        payload_tab = QWidget()
        payload_layout = QVBoxLayout(payload_tab)
        payload_layout.setContentsMargins(6, 6, 6, 6)
        payload_layout.setSpacing(6)

        self.payload_badge = QLabel("NO PAYLOAD YET")
        self.payload_badge.setObjectName("payload_badge")
        self.payload_badge.setProperty("verified", False)
        self.payload_badge.setWordWrap(True)
        payload_layout.addWidget(self.payload_badge)

        self.payload_meta_label = QLabel(
            "Open a capture and press Run Analysis to identify the header and payload."
        )
        self.payload_meta_label.setObjectName("payload_meta_label")
        self.payload_meta_label.setWordWrap(True)
        payload_layout.addWidget(self.payload_meta_label)

        payload_split = QSplitter(Qt.Orientation.Vertical)

        decoded_group = QGroupBox("Recovered message (CRC-16 verified only)")
        decoded_layout = QVBoxLayout(decoded_group)
        self.payload_decoded_text = QPlainTextEdit()
        self.payload_decoded_text.setObjectName("payload_decoded_text")
        self.payload_decoded_text.setReadOnly(True)
        self.payload_decoded_text.setPlaceholderText(
            "A message appears here only after an independent CRC-16 check "
            "passes. Nothing is guessed."
        )
        self.payload_decoded_text.setMaximumBlockCount(400)
        decoded_layout.addWidget(self.payload_decoded_text)
        payload_split.addWidget(decoded_group)

        raw_group = QGroupBox("Raw payload as received (still coded)")
        raw_layout = QVBoxLayout(raw_group)
        self.payload_raw_text = QPlainTextEdit()
        self.payload_raw_text.setObjectName("payload_raw_text")
        self.payload_raw_text.setReadOnly(True)
        self.payload_raw_text.setPlaceholderText(
            "Demodulated bits after the sync word, packed as bytes. For a coded "
            "transmission this is the encoded stream, not the message."
        )
        self.payload_raw_text.setMaximumBlockCount(400)
        raw_layout.addWidget(self.payload_raw_text)
        payload_split.addWidget(raw_group)

        payload_split.setSizes([260, 260])
        payload_layout.addWidget(payload_split, stretch=1)
        self.payload_tab = payload_tab
        self.tabs.addTab(payload_tab, "Payload")

        # ---- Hypotheses tab: the ranked candidates behind the answer ----
        # The pipeline scores several FEC and interleaver hypotheses before it
        # commits to one. The results table shows only the winner, so the
        # runners-up are surfaced here rather than discarded.
        hyp_tab = QWidget()
        hyp_layout = QVBoxLayout(hyp_tab)
        hyp_layout.setContentsMargins(6, 6, 6, 6)
        hyp_layout.setSpacing(6)

        self.hypotheses_intro = QLabel(
            "Ranked candidates considered by the classifier. These scores are "
            "the pre-decode ranking; once a decode is validated its own "
            "confidence is reported in the results table."
        )
        self.hypotheses_intro.setObjectName("hypotheses_intro")
        self.hypotheses_intro.setWordWrap(True)
        hyp_layout.addWidget(self.hypotheses_intro)

        self.decode_search_label = QLabel("Decode search: — (no run yet)")
        self.decode_search_label.setObjectName("decode_search_label")
        self.decode_search_label.setWordWrap(True)
        hyp_layout.addWidget(self.decode_search_label)

        fec_group = QGroupBox("FEC hypotheses (ranked)")
        fec_group_layout = QVBoxLayout(fec_group)
        self.fec_candidates_table = QTableWidget(0, 3)
        self.fec_candidates_table.setObjectName("fec_candidates_table")
        self.fec_candidates_table.setHorizontalHeaderLabels(
            ["Candidate", "Confidence", "CRC pass"]
        )
        self.fec_candidates_table.horizontalHeader().setStretchLastSection(True)
        self.fec_candidates_table.verticalHeader().setVisible(False)
        self.fec_candidates_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        fec_group_layout.addWidget(self.fec_candidates_table)
        hyp_layout.addWidget(fec_group, stretch=1)

        ilv_group = QGroupBox("Interleaver hypotheses (ranked)")
        ilv_group_layout = QVBoxLayout(ilv_group)
        self.ilv_candidates_table = QTableWidget(0, 3)
        self.ilv_candidates_table.setObjectName("ilv_candidates_table")
        self.ilv_candidates_table.setHorizontalHeaderLabels(
            ["Candidate", "Depth", "Confidence"]
        )
        self.ilv_candidates_table.horizontalHeader().setStretchLastSection(True)
        self.ilv_candidates_table.verticalHeader().setVisible(False)
        self.ilv_candidates_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        ilv_group_layout.addWidget(self.ilv_candidates_table)
        hyp_layout.addWidget(ilv_group, stretch=1)

        qual_group = QGroupBox("Modulation fit (ranked by EVM)")
        qual_group_layout = QVBoxLayout(qual_group)
        self.qual_candidates_table = QTableWidget(0, 3)
        self.qual_candidates_table.setObjectName("qual_candidates_table")
        self.qual_candidates_table.setHorizontalHeaderLabels(
            ["Mode", "Order", "EVM (%)"]
        )
        self.qual_candidates_table.horizontalHeader().setStretchLastSection(True)
        self.qual_candidates_table.verticalHeader().setVisible(False)
        self.qual_candidates_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        qual_group_layout.addWidget(self.qual_candidates_table)
        hyp_layout.addWidget(qual_group, stretch=1)

        self.hypotheses_tab = hyp_tab
        self.tabs.addTab(hyp_tab, "Hypotheses")

        # ---- Right results panel ----
        right = QWidget()
        right.setMinimumWidth(280)
        right.setMaximumWidth(420)
        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(8)

        results_group = QGroupBox("Parameters / Results")
        results_layout = QVBoxLayout(results_group)
        # Active-params feedback: what the last Run actually used.
        self.last_analysis_label = QLabel("Last analysis: — (no run yet)")
        self.last_analysis_label.setObjectName("last_analysis_label")
        self.last_analysis_label.setWordWrap(True)
        self.last_analysis_label.setToolTip(
            "Parameters used by the last Run vs current UI values"
        )
        results_layout.addWidget(self.last_analysis_label)
        # STALE badge: accent fill (--color-accent #EA580C) when UI differs
        # from the last Run. Hidden when results are fresh.
        self.stale_badge = QLabel("STALE — params changed, Run to refresh")
        self.stale_badge.setObjectName("stale_badge")
        self.stale_badge.setToolTip(
            "Current inputs differ from the last analysis; press Run Analysis"
        )
        self.stale_badge.setVisible(False)
        results_layout.addWidget(self.stale_badge)
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

        # Live param reactivity: any input change marks results STALE and
        # refreshes the display-only time+spectrum preview (no estimation).
        # Full DSP estimation still runs only on Run Analysis.
        try:
            self.sample_rate_spin.valueChanged.connect(self._on_params_changed)
            self.center_freq_spin.valueChanged.connect(self._on_params_changed)
            self.iq_format_combo.currentTextChanged.connect(self._on_params_changed)
            self.mod_combo.currentTextChanged.connect(self._on_params_changed)
            self.sync_word_edit.textChanged.connect(self._on_params_changed)
            self.decode_check.toggled.connect(self._on_params_changed)
            self.frame_bits_spin.valueChanged.connect(self._on_params_changed)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def log(self, message: str) -> None:
        """Append a line to the log console (and stdout for headless runs)."""
        try:
            self.log_console.appendPlainText(str(message))
        except Exception:
            pass
        # Windows cp1252 consoles cannot encode some unicode (e.g. arrows);
        # never let a log line crash param-change/live-refresh handlers.
        try:
            print(str(message))
        except UnicodeEncodeError:
            try:
                print(str(message).encode("ascii", "replace").decode("ascii"))
            except Exception:
                pass

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
        frame_bits = int(self.frame_bits_spin.value())
        return {
            "file_path": str(self.current_file),
            "sample_rate": float(self.sample_rate_spin.value()),
            "center_frequency": float(self.center_freq_spin.value()),
            "iq_format": self.iq_format_combo.currentText().strip(),
            "modulation": self._modulation_request_value(),
            "sync_word": self.sync_word_edit.text().strip(),
            "decode": bool(self.decode_check.isChecked()),
            "frame_bits": frame_bits if frame_bits > 0 else None,
        }

    def _current_ui_request(self) -> dict | None:
        """Best-effort snapshot of current UI inputs (None when no file)."""
        try:
            if self.current_file is None:
                return None
            return self._build_request()
        except Exception:
            return None

    @staticmethod
    def _request_summary(req: dict | None) -> str:
        if not req:
            return "—"
        try:
            name = Path(str(req.get("file_path", "—"))).name
        except Exception:
            name = str(req.get("file_path", "—"))
        return (
            f"{name}|rate={req.get('sample_rate')}|fmt={req.get('iq_format')}|"
            f"mod={req.get('modulation')}|sync={req.get('sync_word')}|"
            f"decode={req.get('decode')}|frame_bits={req.get('frame_bits')}"
        )

    def _update_last_analysis_label(self) -> None:
        """Show last-Run params vs current UI values (active-params label)."""
        try:
            if self._last_request is None:
                self.last_analysis_label.setText("Last analysis: — (no run yet)")
                return
            last = self._request_summary(self._last_request)
            cur = self._request_summary(self._current_ui_request())
            self.last_analysis_label.setText(
                f"Last analysis: {last}\nCurrent UI: {cur}"
            )
        except Exception:
            pass

    def _set_stale(self, stale: bool) -> None:
        """Show/hide STALE badge + Run highlight (accent #EA580C)."""
        try:
            self.stale_badge.setVisible(bool(stale))
            # Also toggle hidden state explicitly for offscreen visibility checks.
            self.stale_badge.setHidden(not bool(stale))
            self.run_btn.setProperty("stale", bool(stale))
            # Re-polish so QSS [stale="true"] applies immediately.
            try:
                self.run_btn.style().unpolish(self.run_btn)
                self.run_btn.style().polish(self.run_btn)
            except Exception:
                pass
            if stale:
                self.run_btn.setToolTip(
                    "Parameters changed — press to re-run full pipeline (Ctrl+R)"
                )
            else:
                self.run_btn.setToolTip("Run full pipeline on the open file (Ctrl+R)")
        except Exception:
            pass

    def _on_params_changed(self, *args) -> None:
        """Live reactivity: log change, flag STALE, refresh display preview.

        Display-only refresh (time + spectrum) so the spectrum x-axis
        responds to the sample-rate spin immediately. Full DSP estimation
        still runs only on Run Analysis. Never duplicates pipeline logic.
        """
        if getattr(self, "_suppress_param_signals", False):
            return
        try:
            sender = self.sender()
            name = "param"
            try:
                if sender is not None:
                    name = str(sender.objectName() or type(sender).__name__)
            except Exception:
                pass
            cur = self._current_ui_request()
            if cur is None:
                return
            if self._last_request is None:
                # No baseline yet — just live-refresh the preview.
                self._update_last_analysis_label()
            else:
                # Diff keys for an honest log line.
                diffs: list[str] = []
                for k in (
                    "file_path",
                    "sample_rate",
                    "center_frequency",
                    "iq_format",
                    "modulation",
                    "sync_word",
                    "decode",
                    "frame_bits",
                ):
                    if str(cur.get(k)) != str(self._last_request.get(k)):
                        diffs.append(
                            f"{k}: {self._last_request.get(k)} -> {cur.get(k)}"
                        )
                if diffs:
                    self.log(
                        f"Param changed ({name}): "
                        + "; ".join(diffs)
                        + " — results STALE; Run to refresh."
                    )
                    self._set_stale(True)
                else:
                    self._set_stale(False)
                self._update_last_analysis_label()
            # Live display-only refresh (no estimation).
            if self.current_file is not None:
                try:
                    samples, rate = self._load_display_samples(_from_param_change=True)
                    if samples is not None and len(samples) > 0:
                        if rate is None or rate <= 0:
                            rate = float(self.sample_rate_spin.value() or 1.0)
                        self._plot_time(samples)
                        self._plot_spectrum(samples, rate)
                except Exception as exc:
                    self.log(f"WARN: live preview refresh failed: {exc}")
        except Exception as exc:
            self.log(f"WARN: param-change handler failed: {exc}")

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
            # IQ format is irrelevant for .wav (rate comes from file header).
            try:
                if suffix == ".wav":
                    self.iq_format_combo.setEnabled(False)
                    self.iq_format_combo.setToolTip(
                        "Ignored for .wav — sample rate comes from file header"
                    )
                else:
                    self.iq_format_combo.setEnabled(True)
                    self.iq_format_combo.setToolTip(
                        "Raw .iq sample format (required for .iq)"
                    )
            except Exception:
                pass
            self.log(f"Opened {self.current_file} ({size:,} bytes).")
            self._preview_file()
            # A newly opened file differs from any prior Run → mark STALE.
            try:
                if self._last_request is not None:
                    if str(self._last_request.get("file_path")) != str(
                        self.current_file
                    ):
                        self.log(
                            "File changed since last analysis — results STALE; "
                            "Run to refresh."
                        )
                        self._set_stale(True)
                self._update_last_analysis_label()
            except Exception:
                pass
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

        # Full param feedback: log exactly what this Run uses. center_freq
        # is metadata-only (recorded in report, not used for estimation).
        self.log(f"Run request: {request}")
        self.log(
            "Note: center_frequency is metadata-only "
            f"({request.get('center_frequency')} Hz recorded, not estimated)."
        )
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
            # Real bits come ONLY from pipeline demodulation preview.
            # Never call demod_* from the GUI, never synthesize bits.
            try:
                dem = (report or {}).get("demodulation", {}) or {}
                preview = dem.get("bits_preview", None)
                if preview is not None and len(preview) > 0:
                    self.last_bits = np.asarray(preview, dtype=np.uint8)
                    self.log(
                        f"Bitstream preview: {len(self.last_bits):,} bits "
                        f"(mode={dem.get('mode', '?')})."
                    )
                else:
                    self.last_bits = None
                    self.log(
                        "WARN: report has no demodulation.bits_preview — "
                        "showing counts only, no bit waveform."
                    )
            except Exception as exc:
                self.last_bits = None
                self.log(f"WARN: could not read bits_preview: {exc}")
            # This Run is now the fresh baseline for STALE tracking.
            try:
                self._last_request = dict(request)
            except Exception:
                self._last_request = request
            self._set_stale(False)
            self._update_last_analysis_label()
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
                self._log_payload_summary(report)
                self._log_detection_summary(report)
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
        """Lightweight preview right after open (no estimation).

        Plots time + spectrum + constellation so switching files visibly
        changes tabs before Run. Full estimation still runs only on Run.
        """
        try:
            samples, rate = self._load_display_samples()
            if samples is None or len(samples) == 0:
                return
            if rate is None or rate <= 0:
                rate = float(self.sample_rate_spin.value() or 1.0)
            # Each _plot_* starts with .clear() so stale traces never linger.
            self._plot_time(samples)
            self._plot_spectrum(samples, rate)
            self._plot_constellation(samples)
            self.log(f"Preview: {len(samples):,} samples @ {rate} Hz.")
        except Exception as exc:
            self.log(f"WARN: preview failed: {exc}")

    def _load_display_samples(
        self, _from_param_change: bool = False
    ) -> tuple[np.ndarray | None, float | None]:
        """Load samples for visualization only. Estimation stays in pipeline."""
        if self.current_file is None:
            return None, None
        try:
            from rf_analyzer.core.io import load_iq, load_wav

            suffix = self.current_file.suffix.lower()
            # Keep IQ-format control in sync with file type every load.
            try:
                if suffix == ".wav":
                    if self.iq_format_combo.isEnabled():
                        self.iq_format_combo.setEnabled(False)
                    self.iq_format_combo.setToolTip(
                        "Ignored for .wav — sample rate comes from file header"
                    )
                else:
                    if not self.iq_format_combo.isEnabled():
                        self.iq_format_combo.setEnabled(True)
                    self.iq_format_combo.setToolTip(
                        "Raw .iq sample format (required for .iq)"
                    )
            except Exception:
                pass
            if suffix == ".wav":
                samples, rate = load_wav(str(self.current_file))
                self._display_rate = float(rate)
                # Reflect file rate in the UI (wav carries metadata).
                # Block signals so the programmatic setValue does not
                # re-enter _on_params_changed → infinite live-refresh loop.
                try:
                    if float(self.sample_rate_spin.value()) != float(rate):
                        self._suppress_param_signals = True
                        try:
                            self.sample_rate_spin.setValue(float(rate))
                        finally:
                            self._suppress_param_signals = False
                except Exception:
                    try:
                        self._suppress_param_signals = False
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
        """Render every block of the pipeline report.

        The pipeline grew well past what this table used to show, so each block
        it returns now has a section here: burst detection, the classifier's
        alternatives and corroboration, the EVM/MER quality block, the FEC and
        interleaver parameters, and the sync-search detail. Section headings
        are rows whose key starts with ``"-- "`` and span both columns, so a
        parameter lookup by exact name (``_table_value`` in the tests) can
        never collide with one.
        """
        rows: list[tuple[str, str]] = []
        try:
            inp = report.get("input", {}) or {}
            sig = report.get("signal", {}) or {}
            burst = sig.get("burst", {}) or {}
            mod = report.get("modulation", {}) or {}
            qual = report.get("quality", {}) or {}
            dem = report.get("demodulation", {}) or {}
            corr = report.get("correlation", {}) or {}
            pay = report.get("payload", {}) or {}
            dec = pay.get("decoded", {}) or {}
            search = pay.get("decode_search", {}) or {}
            raw = pay.get("raw", {}) or {}
            fec = report.get("fec", {}) or {}
            ilv = report.get("interleaving", {}) or {}
            disp = report.get("display", {}) or {}

            if burst.get("found") and burst.get("start_sample") is not None:
                burst_span = f"{burst.get('start_sample')}-{burst.get('end_sample')}"
            else:
                burst_span = "—"

            rows = [
                ("-- INPUT --", ""),
                ("File", str(inp.get("file_name", ""))),
                ("File type", str(inp.get("file_type", ""))),
                ("Sample rate", str(inp.get("sample_rate", ""))),
                ("IQ format used", str(inp.get("iq_format_used", ""))),
                ("Center freq (set)", _fmt_value(inp.get("center_frequency"))),
                ("-- SIGNAL --", ""),
                ("Samples", str(sig.get("num_samples", ""))),
                ("Duration (s)", str(sig.get("duration_seconds", ""))),
                ("Center freq est", str(sig.get("center_frequency_estimate", ""))),
                ("Bandwidth est", str(sig.get("bandwidth_estimate", ""))),
                ("SNR (dB)", str(sig.get("snr_db", ""))),
                ("Sample rate est", _fmt_value(sig.get("sample_rate_estimate"))),
                ("Symbol rate est", str(sig.get("symbol_rate_estimate", ""))),
                ("CFO est (Hz)", str(sig.get("cfo_estimate_hz", ""))),
                ("-- BURST DETECTION --", ""),
                ("Burst found", "yes" if burst.get("found") else "no"),
                ("Burst span", burst_span),
                ("Burst level (dB)", _fmt_value(burst.get("burst_level_db"))),
                ("Burst noise floor (dB)", _fmt_value(burst.get("noise_floor_db"))),
                ("Burst threshold (dB)", _fmt_value(burst.get("threshold_db"))),
                ("Burst SNR (dB)", _fmt_value(burst.get("snr_db"))),
                ("Burst window", _fmt_value(burst.get("window"))),
                ("Burst method", _fmt_value(burst.get("method"))),
                ("Burst note", _fmt_value(burst.get("reason"))),
                ("-- MODULATION --", ""),
                ("Modulation", str(mod.get("estimated_type", ""))),
                ("Mod confidence", str(mod.get("confidence", ""))),
                ("Mod alternatives", _fmt_value(mod.get("alternatives"))),
                ("Mod corroborated", _fmt_value(mod.get("corroborated"))),
                ("Mod revised from", _fmt_value(mod.get("revised_from"))),
                ("-- QUALITY (EVM / MER) --", ""),
                ("Quality applies", "yes" if qual.get("applicable") else "no"),
                ("Quality mode", _fmt_value(qual.get("mode"))),
                ("EVM (%)", _fmt_value(qual.get("evm_percent"))),
                ("MER (dB)", _fmt_value(qual.get("mer_db"))),
                ("SNR from EVM (dB)", _fmt_value(qual.get("snr_db_from_evm"))),
                ("Modulation order", _fmt_value(qual.get("modulation_order"))),
                ("Quality symbols", _fmt_value(qual.get("n_symbols"))),
                ("Fit ok", _fmt_value(qual.get("fit_ok"))),
                ("-- DEMODULATION --", ""),
                ("Demod mode", str(dem.get("mode", ""))),
                ("Num bits", str(dem.get("num_bits", ""))),
                ("Display mode", _fmt_value(disp.get("mode"))),
                ("Samples per symbol", _fmt_value(disp.get("samples_per_symbol"))),
                ("Bitstream file", _fmt_value(dem.get("bitstream_file"))),
                ("-- CORRELATION --", ""),
                ("Sync word", str(corr.get("sync_word", ""))),
                ("Sync detected", _fmt_value(corr.get("detected"))),
                ("Sync assumed", _fmt_value(corr.get("assumed"))),
                ("Sync length (bits)", _fmt_value(corr.get("sync_length"))),
                ("Header offset", str(corr.get("header_offset", ""))),
                ("Corr score", str(corr.get("score", ""))),
                ("Corr min score", _fmt_value(corr.get("min_score"))),
                ("Corr min matches", _fmt_value(corr.get("min_matches"))),
                ("Positions searched", _fmt_value(corr.get("positions_searched"))),
                ("-- PAYLOAD --", ""),
                ("Payload bits", str(pay.get("payload_bits", ""))),
                ("Payload bit offset", _fmt_value(pay.get("payload_bit_offset"))),
                ("Payload bytes (raw)", str(raw.get("bytes", ""))),
                (
                    "Decode search",
                    f"{search.get('attempts', 0)} tried"
                    + (" (budget hit)" if search.get("budget_exhausted") else ""),
                ),
                ("-- FEC --", ""),
                ("FEC candidate", str(fec.get("candidate", ""))),
                ("FEC confidence", str(fec.get("confidence", ""))),
                ("FEC verified", "yes" if fec.get("validated") else "no"),
                ("FEC CRC pass", _fmt_value(fec.get("crc_pass"))),
                ("FEC errors corrected", _fmt_value(fec.get("errors_corrected"))),
                ("FEC params", _fmt_value(fec.get("params"))),
                ("-- INTERLEAVING --", ""),
                ("Interleaver", str(ilv.get("candidate", ""))),
                ("Interleaver verified", "yes" if ilv.get("validated") else "no"),
                ("Interleaver depth", _fmt_value(ilv.get("depth"))),
                ("-- RESULT --", ""),
                (
                    "Recovered message",
                    (
                        f"{dec.get('bytes', 0)} bytes (CRC-16 verified)"
                        if dec.get("available")
                        else "none"
                    ),
                ),
                ("-- DIAGNOSTICS --", ""),
                ("Warnings", "; ".join(map(str, report.get("warnings", []) or []))),
                ("Errors", "; ".join(map(str, report.get("errors", []) or []))),
            ]
        except Exception as exc:
            rows = [("Error", f"Could not format report: {exc}")]

        table = self.results_table
        table.clearSpans()
        table.setRowCount(len(rows))
        try:
            section_font = QFont()
            section_font.setBold(True)
        except Exception:
            section_font = None
        for i, (key, value) in enumerate(rows):
            key_item = QTableWidgetItem(key)
            if key.startswith("-- "):
                if section_font is not None:
                    key_item.setFont(section_font)
                table.setItem(i, 0, key_item)
                table.setSpan(i, 0, 1, 2)
                continue
            table.setItem(i, 0, key_item)
            table.setItem(i, 1, QTableWidgetItem(value))
        table.resizeColumnsToContents()

    # ------------------------------------------------------------------ #
    # Demo / batch helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sample_data_dir() -> Path:
        """Locate sample_data/ relative to the repo root (src/../..)."""
        return Path(__file__).resolve().parents[3] / "sample_data"

    def _demo_captures(self) -> list[Path]:
        try:
            from rf_analyzer.batch import discover_captures

            folder = self._sample_data_dir()
            if folder.is_dir():
                return discover_captures(folder)
        except Exception:
            pass
        return []

    def open_demo_capture(self, name: str | None = None) -> None:
        """Load one of the generated sample captures (no dialog when ``name`` given)."""
        captures = self._demo_captures()
        if not captures:
            msg = (
                "No sample captures found. Generate them first with "
                "python scripts/generate_test_data.py"
            )
            self.log(f"ERROR: {msg}")
            _notify(self, "warning", "No demo captures", msg)
            return
        if name is None:
            labels = [p.name for p in captures]
            choice, ok = QInputDialog.getItem(
                self, "Open demo capture", "Capture:", labels, 0, False
            )
            if not ok or not choice:
                return
            name = str(choice)
        target = next((p for p in captures if p.name == name), None)
        if target is None:
            self.log(f"ERROR: demo capture '{name}' not found.")
            return
        self.log(f"Loading demo capture: {target.name}")
        self.open_file(str(target))
        try:
            self.run_analysis()
        except Exception as exc:
            self.log(f"WARN: demo analysis failed: {exc}")

    def run_batch_analysis(
        self,
        folder: str | None = None,
        fmt: str | None = None,
        *,
        export: bool = True,
    ) -> list[dict]:
        """Analyse every capture in a folder and optionally export CSV + HTML.

        With no ``folder`` a directory picker is shown; passing ``folder`` keeps
        this callable from tests and scripts without any dialog.
        """
        from rf_analyzer.batch import analyze_folder, write_csv, write_html

        if folder is None:
            folder = QFileDialog.getExistingDirectory(
                self, "Choose a folder of .iq / .wav captures"
            )
            if not folder:
                return []
        try:
            rows = analyze_folder(
                folder,
                sample_rate=float(self.sample_rate_spin.value()),
                iq_format=self.iq_format_combo.currentText().strip(),
                modulation=self._modulation_request_value(),
                sync_word=self.sync_word_edit.text().strip(),
                decode=bool(self.decode_check.isChecked()),
                progress=lambda path, i, total: self.log(f"[{i}/{total}] {path.name}"),
            )
        except Exception as exc:
            self.log(f"ERROR: batch analysis failed: {exc}\n{traceback.format_exc()}")
            _notify(self, "critical", "Batch failed", str(exc))
            return []

        if not rows:
            self.log("Batch: no .iq/.wav captures found in that folder.")
            return []

        verified = sum(1 for r in rows if r.get("decode_validated") == "yes")
        self.log(
            f"Batch done: {len(rows)} capture(s), {verified} CRC-verified payload(s)."
        )
        for row in rows:
            if row.get("decode_validated") == "yes":
                self.log(
                    f"  {row['file']}: {row.get('fec_candidate')} / "
                    f"{row.get('interleaver')} -> {row.get('decoded_text', '')[:80]!r}"
                )

        if export:
            out_dir = Path("output")
            wanted = {fmt} if fmt in ("csv", "html") else {"csv", "html"}
            try:
                if "csv" in wanted:
                    path = write_csv(rows, out_dir / "batch_summary.csv")
                    self.log(f"Batch CSV written to {path}")
                if "html" in wanted:
                    path = write_html(
                        rows,
                        out_dir / "batch_summary.html",
                        subtitle=f"Folder: {Path(folder)}",
                    )
                    self.log(f"Batch HTML written to {path}")
            except Exception as exc:
                self.log(f"WARN: batch export failed: {exc}")
        return rows

    def run_demo_all(self, folder: str | None = None) -> list[dict]:
        """Analyse every generated sample capture and log a one-screen summary."""
        target = folder or str(self._sample_data_dir())
        self.log(f"Full demo: analysing every capture in {target} …")
        return self.run_batch_analysis(target, export=True)

    def _log_payload_summary(self, report: dict) -> None:
        """One honest log line about header/payload identification."""
        try:
            pay = (report or {}).get("payload", {}) or {}
            if not pay:
                return
            decoded = pay.get("decoded", {}) or {}
            search = pay.get("decode_search", {}) or {}
            if decoded.get("available"):
                self.log(
                    f"Payload: CRC-16 VERIFIED — {decoded.get('bytes')} bytes via "
                    f"{decoded.get('scheme')} / {decoded.get('interleaver')} "
                    f"interleaver ({search.get('attempts', 0)} hypotheses tried)."
                )
                preview = str(decoded.get("text", ""))
                if preview:
                    self.log(f"  message: {preview[:160]}")
            else:
                self.log(
                    "Payload: NOT DECODED — no CRC-valid FEC hypothesis "
                    f"({search.get('attempts', 0)} tried"
                    + (
                        ", time budget reached"
                        if search.get("budget_exhausted")
                        else ""
                    )
                    + "). Raw payload view shows the still-coded stream."
                )
        except Exception as exc:
            self.log(f"WARN: could not summarise payload: {exc}")

    def _log_detection_summary(self, report: dict) -> None:
        """Log the burst and quality blocks, the way the table now carries them.

        Both say "not measured" out loud rather than staying silent: a silent
        log reads as "nothing to report", which is the wrong story when the
        pipeline tried and declined.
        """
        try:
            burst = ((report or {}).get("signal", {}) or {}).get("burst", {}) or {}
            if burst:
                if burst.get("found"):
                    self.log(
                        "Burst: detected at samples "
                        f"{burst.get('start_sample')}-{burst.get('end_sample')} "
                        f"(level {_fmt_value(burst.get('burst_level_db'))} dB vs "
                        f"floor {_fmt_value(burst.get('noise_floor_db'))} dB, "
                        f"SNR {_fmt_value(burst.get('snr_db'))} dB, "
                        f"method {burst.get('method')})."
                    )
                else:
                    self.log(
                        f"Burst: none found — "
                        f"{burst.get('reason') or 'no reason given'} "
                        f"(method {burst.get('method')})."
                    )
            qual = (report or {}).get("quality", {}) or {}
            if qual.get("applicable") and qual.get("evm_percent") is not None:
                mer = qual.get("mer_db")
                self.log(
                    f"Quality: EVM {qual.get('evm_percent'):.3g}%"
                    + (f", MER {mer:.4g} dB" if mer is not None else "")
                    + f" over {qual.get('n_symbols')} symbols "
                    f"(order {qual.get('modulation_order')}, "
                    f"fit_ok={_fmt_value(qual.get('fit_ok'))})."
                )
            elif qual:
                self.log(
                    "Quality: EVM/MER not applicable to "
                    f"{qual.get('mode') or 'this modulation'}."
                )
        except Exception as exc:
            self.log(f"WARN: could not summarise detection: {exc}")

    def _set_payload_badge(self, text: str, verified: bool) -> None:
        """Set the payload status badge and re-polish so the QSS property applies."""
        try:
            self.payload_badge.setText(text)
            self.payload_badge.setProperty("verified", bool(verified))
            style = self.payload_badge.style()
            style.unpolish(self.payload_badge)
            style.polish(self.payload_badge)
        except Exception:
            pass

    def _update_payload_tab(self, report: dict) -> None:
        """Render header/payload identification.

        The badge is the honest headline: it reads VERIFIED only when a real
        decoder produced a CRC-valid payload. Otherwise it says so plainly and
        the raw view is labelled as still-coded.
        """
        report = report or {}
        pay = report.get("payload", {}) or {}
        search = pay.get("decode_search", {}) or {}
        decoded = pay.get("decoded", {}) or {}
        raw = pay.get("raw", {}) or {}

        # A failed analysis explains itself first: an error report carries an
        # empty payload block, so "no payload block" would tell the wrong story.
        errors = report.get("errors") or []
        if errors:
            self._set_payload_badge(f"NOT ANALYSED — {errors[0]}", False)
            self.payload_meta_label.setText("")
            self.payload_decoded_text.setPlainText("")
            self.payload_raw_text.setPlainText("")
            return

        if not pay:
            self._set_payload_badge(
                "NO PAYLOAD — analysis returned no payload block", False
            )
            self.payload_meta_label.setText(
                "The report has no payload section (the file may have failed to load)."
            )
            self.payload_decoded_text.setPlainText("")
            self.payload_raw_text.setPlainText("")
            return

        corr = report.get("correlation", {}) or {}
        header_offset = pay.get("header_offset", 0)
        if corr.get("sync_word") and corr.get("detected"):
            header_note = (
                f"sync {corr.get('sync_word')} found at bit {corr.get('header_offset')} "
                f"(score {float(corr.get('score', 0.0)):.3f})"
            )
        elif corr.get("sync_word"):
            header_note = (
                f"sync {corr.get('sync_word')} not found "
                f"(best score {float(corr.get('score', 0.0)):.3f}); "
                "whole stream treated as payload"
            )
        else:
            header_note = "no sync word supplied; whole stream treated as payload"

        attempts = search.get("attempts", 0)
        if not search.get("enabled", True):
            search_note = "decode search off"
        elif search.get("validated"):
            search_note = f"decode search: {attempts} hypotheses, validated"
        elif search.get("budget_exhausted"):
            search_note = f"decode search: {attempts} hypotheses, time budget reached"
        else:
            search_note = f"decode search: {attempts} hypotheses, no CRC match"

        self.payload_meta_label.setText(
            f"Header: {header_note}\n"
            f"Payload: {pay.get('payload_bits', 0)} bits from bit {header_offset} "
            f"({raw.get('bytes', 0)} bytes packed) · {search_note}"
        )

        if decoded.get("available"):
            self._set_payload_badge(
                f"CRC-16 VERIFIED — {decoded.get('scheme')} "
                f"via {decoded.get('interleaver')} interleaver · "
                f"{decoded.get('bytes')} bytes recovered",
                True,
            )
            body = [
                f"scheme        : {decoded.get('scheme')}",
                f"interleaver   : {decoded.get('interleaver')}",
                f"errors fixed  : {decoded.get('errors_corrected', 0)}",
                f"confidence    : {decoded.get('confidence')}",
                f"content type  : {decoded.get('content_type')}",
                "",
                "text:",
                str(decoded.get("text", "")),
                "",
                "hex:",
                str(decoded.get("hex", "")),
                "",
                "hexdump:",
                *[str(line) for line in (decoded.get("hexdump") or [])],
            ]
            self.payload_decoded_text.setPlainText("\n".join(body))
        else:
            self._set_payload_badge(
                "NOT DECODED — payload shown exactly as received, still coded",
                False,
            )
            note = decoded.get("note", "no CRC-valid decode")
            self.payload_decoded_text.setPlainText(
                "No CRC-16-valid decode was found, so no message is claimed.\n\n"
                f"reason: {note}\n\n"
                "The raw view below is the demodulated bit stream packed as "
                "bytes. For a coded transmission it is the encoded stream, not "
                "the transmitted text."
            )

        raw_body = [
            f"content type  : {raw.get('content_type')}",
            f"bytes         : {raw.get('bytes')}",
            "",
            "hexdump:",
            *[str(line) for line in (raw.get("hexdump") or [])],
        ]
        self.payload_raw_text.setPlainText("\n".join(raw_body))

    @staticmethod
    def _fill_candidate_table(table, candidates, columns: list[str]) -> None:
        """Fill a candidate table from a ranked hypothesis list.

        Rebuilt from scratch each run, so a short new list never leaves the
        previous report's longer tail on screen.
        """
        try:
            items = list(candidates or [])
        except Exception:
            items = []
        table.setRowCount(len(items))
        for row, entry in enumerate(items):
            cand = entry if isinstance(entry, dict) else {"candidate": entry}
            for col, key in enumerate(columns):
                table.setItem(row, col, QTableWidgetItem(_fmt_value(cand.get(key))))
        table.resizeColumnsToContents()

    def _update_hypotheses_tab(self, report: dict) -> None:
        """Render the ranked FEC / interleaver candidates and the decode search.

        The results table can only show the winning hypothesis; this tab shows
        what it was chosen from. The scores are the classifier's pre-decode
        ranking, so the winner here can carry a lower score than the validated
        decode's own confidence — the intro text says so, because otherwise
        that looks like a contradiction.
        """
        report = report or {}
        search = (report.get("payload") or {}).get("decode_search") or {}
        fec = report.get("fec") or {}
        ilv = report.get("interleaving") or {}

        if not search.get("enabled", True):
            self.decode_search_label.setText(
                "Decode search: off — tick 'Verify FEC + decode payload' to rank "
                "hypotheses."
            )
        else:
            self.decode_search_label.setText(
                f"Decode search: {search.get('attempts', 0)} hypotheses tried"
                + (" · time budget reached" if search.get("budget_exhausted") else "")
                + (" · validated" if search.get("validated") else " · no CRC match")
                # "top-ranked", not "winner": without a CRC pass nothing was
                # won, and the highest-scoring hypothesis is not a result.
                + f" · top-ranked: {fec.get('candidate') or '—'}"
            )

        self._fill_candidate_table(
            self.fec_candidates_table,
            fec.get("candidates"),
            ["candidate", "confidence", "crc_pass"],
        )
        self._fill_candidate_table(
            self.ilv_candidates_table,
            ilv.get("candidates"),
            ["candidate", "depth", "confidence"],
        )
        qual = report.get("quality") or {}
        self._fill_candidate_table(
            self.qual_candidates_table,
            qual.get("candidates"),
            ["mode", "modulation_order", "evm_percent"],
        )
        # Mark the hypothesis the pipeline actually committed to, so the
        # ranking and the decision are readable in one glance.
        try:
            winner = fec.get("candidate")
            for row in range(self.fec_candidates_table.rowCount()):
                item = self.fec_candidates_table.item(row, 0)
                if item is not None and winner and item.text() == str(winner):
                    font = item.font()
                    font.setBold(True)
                    for col in range(self.fec_candidates_table.columnCount()):
                        cell = self.fec_candidates_table.item(row, col)
                        if cell is not None:
                            cell.setFont(font)
        except Exception:
            pass

    def _update_plots(self, report: dict) -> None:
        # The payload view is derived purely from the report, so refresh it
        # first — it must still populate when samples are unavailable.
        self._update_payload_tab(report)
        self._update_hypotheses_tab(report)
        samples, rate = self._load_display_samples()
        if samples is None or len(samples) == 0:
            self.log("WARN: no samples available for plots; showing report only.")
            self._update_bitstream_tab(report)
            return
        if rate is None or rate <= 0:
            rate = float(self.sample_rate_spin.value() or 1.0)
        try:
            # Every plot starts with .clear()/setImage so runs never overlay.
            burst = ((report or {}).get("signal", {}) or {}).get("burst", {}) or {}
            quality = (report or {}).get("quality", {}) or {}
            self._plot_time(samples, burst=burst)
            self._plot_spectrum(samples, rate)
            self._plot_waterfall(samples, rate)
            self._plot_constellation(samples, quality=quality)
            # Eye overlay depends on symbol density: prefer pipeline hint
            # report["display"]["samples_per_symbol"], else demod mode.
            try:
                disp = (report or {}).get("display", {}) or {}
                sps = disp.get("samples_per_symbol", None)
                mode = ((report or {}).get("demodulation", {}) or {}).get("mode", "")
                if sps is None:
                    sps = 1 if str(mode).upper() in ("BPSK", "QPSK") else 8
                sps = int(sps)
            except Exception:
                sps = 8
                mode = ""
            self._plot_eye(samples, samples_per_symbol=sps)
            self._update_bitstream_tab(report)
        except Exception as exc:
            self.log(f"WARN: plot update partially failed: {exc}")

    # ---- individual plots (display only) ---- #
    def _plot_time(self, samples: np.ndarray, burst: dict | None = None) -> None:
        """Time-domain view, with the detected burst span shaded when there is one.

        The span is mapped through the same stride ``_decimate`` used, so the
        shaded region lines up with the decimated trace instead of pointing at
        original-index positions in display space.
        """
        self.time_plot.clear()
        self.time_plot.setTitle("Time domain")
        view = _decimate(samples, DISPLAY_MAX_POINTS)
        x = np.arange(view.size)
        self.time_plot.plot(
            x, np.real(view).astype(float), pen=pg.mkPen("#22C55E", width=1), name="I"
        )
        self.time_plot.plot(
            x, np.imag(view).astype(float), pen=pg.mkPen("#38BDF8", width=1), name="Q"
        )
        try:
            info = burst or {}
            if info.get("found") and info.get("start_sample") is not None:
                step = _decimation_step(np.asarray(samples).size, DISPLAY_MAX_POINTS)
                region = pg.LinearRegionItem(
                    values=(
                        float(info.get("start_sample", 0)) / step,
                        float(info.get("end_sample", 0)) / step,
                    ),
                    movable=False,
                    brush=pg.mkBrush(234, 88, 12, 40),
                    pen=pg.mkPen(234, 88, 12, 170),
                )
                self.time_plot.addItem(region)
                self.time_plot.setTitle(
                    "Time domain — burst "
                    f"[{info.get('start_sample')}-{info.get('end_sample')} samples]"
                )
        except Exception as exc:
            self.log(f"WARN: could not shade the burst region: {exc}")

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

                # Adaptive core DSP (handles short/long captures); GUI only
                # caps the display size for responsiveness.
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
            # Display caps: max 256 rows x 512 cols; ImageView is refreshed
            # via setImage every run so tabs visibly change per file/params.
            if wf.shape[1] > 512:
                wf = wf[:, :: wf.shape[1] // 512 + 1]
            if wf.shape[0] > 256:
                wf = wf[:: wf.shape[0] // 256 + 1, :]
            self.waterfall_view.setImage(wf.T, autoLevels=True, autoRange=True)
            self.log(f"Waterfall: {wf.shape[0]} rows x {wf.shape[1]} cols.")
        except Exception as exc:
            self.log(f"WARN: waterfall failed: {exc}")

    def _plot_constellation(
        self, samples: np.ndarray, quality: dict | None = None
    ) -> None:
        """I/Q scatter, annotated with the pipeline's measured EVM/MER.

        The annotation only repeats what the pipeline computed; no ideal
        reference points are drawn over the scatter. The displayed samples are
        raw (not the pipeline's symbol decisions), so overlaying an ideal grid
        would imply an agreement the GUI cannot actually demonstrate.
        """
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
        try:
            qual = quality or {}
            mode = qual.get("mode")
            evm = qual.get("evm_percent")
            mer = qual.get("mer_db")
            if qual.get("applicable") and evm is not None:
                title = f"Constellation (I vs Q) — {mode}, EVM {evm:.3g}%"
                if mer is not None:
                    title += f", MER {mer:.4g} dB"
                if qual.get("fit_ok") is False:
                    title += " (fit poor — burst likely padded with noise)"
                self.const_plot.setTitle(title)
            elif mode:
                self.const_plot.setTitle(
                    f"Constellation (I vs Q) — {mode}; EVM/MER not applicable"
                )
            else:
                self.const_plot.setTitle("Constellation (I vs Q)")
        except Exception:
            self.const_plot.setTitle("Constellation (I vs Q)")

    def _plot_eye(self, samples: np.ndarray, samples_per_symbol: int = 8) -> None:
        self.eye_plot.clear()
        try:
            sps = int(samples_per_symbol or 8)
            if sps <= 0:
                sps = 8
            flat = np.asarray(samples)
            if flat.size == 0:
                self.log("WARN: eye diagram skipped — no samples.")
                return
            if sps <= 1:
                # Honest 1-sps waveform view: overlay I-channel chunks.
                # No symbol timing exists at 1 sample/symbol, so never fake
                # a 2-symbol eye — show chunked overlay instead.
                try:
                    self.eye_plot.setTitle("Eye diagram (I) (1 sps: overlay view)")
                except Exception:
                    pass
                real = np.real(flat).astype(float)
                chunk, ntraces = 32, 40
                if real.size < chunk:
                    # Too short: single trace, never crash.
                    self.eye_plot.plot(
                        np.arange(real.size),
                        real,
                        pen=pg.mkPen("#38BDF8", width=1),
                    )
                    return
                ntraces = min(ntraces, max(1, real.size // chunk))
                traces = real[: ntraces * chunk].reshape(ntraces, chunk)
                pen = pg.mkPen("#38BDF8", width=1)
                for tr in traces[:40]:
                    self.eye_plot.plot(np.arange(tr.size), tr.astype(float), pen=pen)
                return
            try:
                self.eye_plot.setTitle("Eye diagram (I)")
            except Exception:
                pass
            try:
                from rf_analyzer.core import dsp as _dsp

                traces = _dsp.compute_eye(flat, samples_per_symbol=sps)
                traces = np.asarray(traces)
            except Exception:
                # Display-only fallback: overlay I-channel segments.
                real = np.real(np.asarray(flat, dtype=np.complex64))
                if real.size < sps * 2:
                    return  # too few samples for an eye — leave tab empty, no crash
                ntraces = min(40, max(1, real.size // (sps * 2)))
                traces = real[: ntraces * sps * 2].reshape(ntraces, sps * 2)
            pen = pg.mkPen("#38BDF8", width=1)
            for tr in traces[:40]:
                y = np.real(np.asarray(tr)).astype(float)
                self.eye_plot.plot(np.arange(y.size), y, pen=pen)
        except Exception as exc:
            self.log(f"WARN: eye diagram failed: {exc}")

    def _update_bitstream_tab(self, report: dict) -> None:
        self.bitstream_plot.clear()
        dem = (report or {}).get("demodulation", {}) or {}
        corr = (report or {}).get("correlation", {}) or {}
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
            header = f"{nbits} bits (first {min(256, len(self.last_bits))} shown):"
            bitfile = dem.get("bitstream_file")
            if bitfile:
                # The pipeline writes the full stream next to its output; the
                # preview above is truncated, so say where the rest went.
                header += f"\nfull bitstream written to: {bitfile}"
            self.bitstream_text.setPlainText(f"{header}\n{preview}")
        else:
            # Honest empty state: never synthesize a fake wave with
            # fixed-seed random. Counts/mode come from the pipeline report.
            self.bitstream_text.setPlainText(
                f"No bits returned (mode={dem.get('mode', '?')}, "
                f"num_bits={nbits}).\n"
                f"Sync={corr.get('sync_word', '?')} offset={corr.get('header_offset', '?')} "
                f"score={corr.get('score', '?')}.\n"
                "Pipeline returned no bits_preview; re-run analysis."
            )
            # Plot stays cleared (blank) so stale/fake traces never linger.
