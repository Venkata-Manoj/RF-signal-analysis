"""CLI --gps flag and GUI GPS checkbox reach the pipeline request."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from rf_analyzer.core.framing import build_frame, modulate

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_PIPELINE = REPO_ROOT / "scripts" / "run_pipeline.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("run_pipeline_cli", RUN_PIPELINE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_pipeline_cli"] = module
    spec.loader.exec_module(module)
    return module


def _tiny_iq(path: Path) -> Path:
    frame = build_frame(b"GPS flag", fec="none", interleaver="none")
    modulate(frame["bits"], "BPSK").astype(np.complex64).tofile(path)
    return path


def test_cli_gps_flag_reaches_report(tmp_path):
    cli = _load_cli()
    iq = _tiny_iq(tmp_path / "flag.iq")
    out = tmp_path / "report.json"
    rc = cli.main(
        [
            str(iq),
            "--sample-rate",
            "100000",
            "--iq-format",
            "complex64",
            "--modulation",
            "BPSK",
            "--sync-word",
            "0x1ACFFC1D",
            "--no-decode",
            "--gps",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["gps"]["attempted"] is True


def test_cli_without_gps_does_not_attempt(tmp_path):
    cli = _load_cli()
    iq = _tiny_iq(tmp_path / "noflag.iq")
    out = tmp_path / "report.json"
    rc = cli.main(
        [
            str(iq),
            "--sample-rate",
            "100000",
            "--iq-format",
            "complex64",
            "--modulation",
            "BPSK",
            "--sync-word",
            "0x1ACFFC1D",
            "--no-decode",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["gps"]["attempted"] is False


_APP_HOLDER: list = []


def _make_window():
    from PyQt6.QtWidgets import QApplication

    from rf_analyzer.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    _APP_HOLDER.append(app)
    window = MainWindow()
    window.show()
    app.processEvents()
    return window


def test_gui_gps_checkbox_defaults_off_and_feeds_request(tmp_path):
    window = _make_window()
    try:
        assert window.gps_check.objectName() == "gps_check"
        assert window.gps_check.isChecked() is False
        window.gps_check.setChecked(True)
        window.current_file = _tiny_iq(tmp_path / "gui.iq")
        req = window._build_request()
        assert req["gps"] is True
    finally:
        window.close()


def test_gui_results_table_renders_after_report(tmp_path):
    from rf_analyzer.pipeline import analyze_file

    window = _make_window()
    try:
        report = analyze_file(
            {
                "file_path": str(_tiny_iq(tmp_path / "t.iq")),
                "sample_rate": 100000,
                "iq_format": "complex64",
                "modulation": "BPSK",
                "sync_word": "0x1ACFFC1D",
                "decode": False,
            }
        )
        window._update_results_table(report)
        assert window.results_table.rowCount() > 0
    finally:
        window.close()
