"""Milestone 1 setup check: imports + packaging resolve via pythonpath=src."""

import rf_analyzer
import rf_analyzer.config


def test_package_imports():
    assert rf_analyzer.__version__ == "0.1.0"
    assert rf_analyzer.config.TOOL_NAME.startswith("RF Signal")
