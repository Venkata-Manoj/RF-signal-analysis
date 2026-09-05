"""Main window shell. Full spec: info.md §6. TODO (Milestone 7).

Must call rf_analyzer.pipeline.analyze_file; never duplicate DSP logic.
Use the ui-ux-pro-max skill in .opencode/skills/ for styling.
"""

from __future__ import annotations

try:
    from PyQt6.QtWidgets import QMainWindow
except ImportError:  # allow headless import before deps installed
    QMainWindow = object  # type: ignore[assignment,misc]


class MainWindow(QMainWindow):  # type: ignore[misc]
    """TODO (Milestone 7): file open, param inputs, plot tabs, run/export."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("MainWindow not implemented yet (Milestone 7)")
