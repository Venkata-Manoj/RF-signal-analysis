"""GUI smoke test: main window opens headless (info.md §19).

Run headless with QT_QPA_PLATFORM=offscreen (see AGENTS.md).
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

PyQt6 = pytest.importorskip("PyQt6.QtWidgets", reason="PyQt6 not installed")
from PyQt6.QtWidgets import QApplication  # noqa: E402

from rf_analyzer.gui.main_window import MainWindow  # noqa: E402


def test_main_window_opens():
    # Arrange
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Act
    window = MainWindow()
    window.setWindowTitle("RF Signal Analyzer MVP")

    # Assert
    assert window.windowTitle() == "RF Signal Analyzer MVP"

    window.close()
