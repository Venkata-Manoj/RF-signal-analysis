"""End-to-end analysis entrypoint. Full spec: info.md §12.5."""

from __future__ import annotations


def analyze_file(request: dict) -> dict:
    """Coordinate load → DSP → demod → correlate → report.

    TODO (Milestone 6): implement per info.md §20. GUI must call this;
    never duplicate DSP logic in the GUI layer.
    """
    raise NotImplementedError("pipeline not implemented yet (Milestone 6)")
