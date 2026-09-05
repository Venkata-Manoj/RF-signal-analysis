"""BPSK/QPSK/2-FSK demodulation. Full spec: info.md §12.3.

MVP is intentionally naive (phase-aligned, no timing recovery).
Robust sync is V2 — do not add Costas loops here.
"""

from __future__ import annotations


def normalize_signal(samples):
    """TODO (Milestone 4)."""
    raise NotImplementedError("normalize_signal not implemented yet (Milestone 4)")


def demod_bpsk(samples):
    """TODO (Milestone 4)."""
    raise NotImplementedError("demod_bpsk not implemented yet (Milestone 4)")


def demod_qpsk(samples):
    """TODO (Milestone 4)."""
    raise NotImplementedError("demod_qpsk not implemented yet (Milestone 4)")


def demod_2fsk(samples):
    """TODO (Milestone 4)."""
    raise NotImplementedError("demod_2fsk not implemented yet (Milestone 4)")
