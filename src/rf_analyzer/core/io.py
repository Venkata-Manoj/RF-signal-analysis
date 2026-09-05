"""Raw .IQ / .wav loading. Full spec: info.md §12.1.

Contracts (do not change without updating AGENTS.md):
- complex64 native; int16 interleaved I/Q ÷32768; uint8 offset-128 ÷128.
- Truncate odd trailing byte. Little-endian default.
- WAV stereo L=I/R=Q; mono = real-only. Sample rate from file for WAV,
  user-supplied for .IQ.
"""

from __future__ import annotations


def load_iq(file_path: str, dtype: str = "complex64", endian: str = "little"):
    """TODO (Milestone 2): implement per info.md §14.1."""
    raise NotImplementedError("load_iq not implemented yet (Milestone 2)")


def load_wav(file_path: str):
    """TODO (Milestone 2): implement per info.md §14.1."""
    raise NotImplementedError("load_wav not implemented yet (Milestone 2)")


def detect_file_type(file_path: str) -> str:
    """TODO (Milestone 2): return 'iq' or 'wav' from suffix."""
    raise NotImplementedError("detect_file_type not implemented yet (Milestone 2)")
