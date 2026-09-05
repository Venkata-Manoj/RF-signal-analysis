"""JSON report + bitstream export. Full spec: info.md §12.6 / §13."""

from __future__ import annotations


def save_report(report: dict, path: str) -> None:
    """TODO (Milestone 6)."""
    raise NotImplementedError("save_report not implemented yet (Milestone 6)")


def save_bits(bits, path: str) -> None:
    """TODO (Milestone 6)."""
    raise NotImplementedError("save_bits not implemented yet (Milestone 6)")


def validate_report(report: dict) -> list[str]:
    """TODO (Milestone 6): return list of missing-field errors."""
    raise NotImplementedError("validate_report not implemented yet (Milestone 6)")
