"""Launch the PyQt6 GUI workbench."""

from __future__ import annotations

import sys


def main() -> None:
    from PyQt6.QtWidgets import QApplication

    from rf_analyzer.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
