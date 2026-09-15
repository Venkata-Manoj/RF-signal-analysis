"""Raw .IQ / .wav loading. Full spec: info.md §12.1.

Contracts (do not change without updating AGENTS.md):
- complex64 native; int16 interleaved I/Q ÷32768; uint8 offset-128 ÷128.
- Truncate odd trailing byte. Little-endian default.
- WAV stereo L=I/R=Q; mono = real-only. Sample rate from file for WAV,
  user-supplied for .IQ.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def load_iq(
    file_path: str, dtype: str = "complex64", endian: str = "little"
) -> np.ndarray:
    """
    Load raw IQ file.

    Supported dtype:
    - complex64
    - int16
    - uint8
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"IQ file not found: {file_path}")

    if dtype == "complex64":
        samples = np.fromfile(file_path, dtype=np.complex64)

    elif dtype == "int16":
        raw = np.fromfile(file_path, dtype=np.int16)
        if endian == "big":
            raw = raw.byteswap()
        if len(raw) % 2 != 0:
            raw = raw[:-1]
        samples = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
        samples = samples / 32768.0

    elif dtype == "uint8":
        raw = np.fromfile(file_path, dtype=np.uint8)
        if len(raw) % 2 != 0:
            raw = raw[:-1]
        samples = (raw[0::2].astype(np.float32) - 128.0) + 1j * (
            raw[1::2].astype(np.float32) - 128.0
        )
        samples = samples / 128.0

    else:
        raise ValueError(f"Unsupported IQ dtype: {dtype}")

    return samples.astype(np.complex64)


def load_wav(file_path: str) -> tuple[np.ndarray, float]:
    """
    Load WAV file.

    If stereo:
      left = I
      right = Q

    If mono:
      treat as real/I channel.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"WAV file not found: {file_path}")

    data, sample_rate = sf.read(str(file_path), dtype="float32")

    if data.ndim == 2:
        if data.shape[1] < 2:
            raise ValueError("Stereo WAV must have at least two channels.")
        samples = data[:, 0] + 1j * data[:, 1]
    else:
        samples = data.astype(np.complex64)

    return samples.astype(np.complex64), float(sample_rate)


def detect_file_type(file_path: str) -> str:
    """Return 'iq' or 'wav' from file suffix (case-insensitive).

    Extended for real-world captures: .cf32, .ci16, .raw, .dat,
    .sigmf-data are treated as raw IQ.
    """
    suffix = Path(file_path).suffix.lower()
    if suffix in (".iq", ".cf32", ".ci16", ".raw", ".dat", ".sigmf-data"):
        return "iq"
    if suffix == ".wav":
        return "wav"
    raise ValueError(f"Unsupported file type: {suffix}")


def load_sigmf_meta(file_path: str) -> dict | None:
    """Read SigMF sidecar metadata if it exists.

    Given a data file (e.g. ``signal.sigmf-data`` or ``signal.iq``),
    looks for a corresponding ``.sigmf-meta`` JSON file and extracts:

    * ``sample_rate`` from ``global.core:sample_rate``
    * ``center_frequency`` from the first capture's ``core:frequency``
    * ``datatype`` from ``global.core:datatype``

    Returns ``None`` if the sidecar is absent or malformed.
    """
    import json as _json

    fp = Path(file_path)
    # Try exact companion: signal.sigmf-data → signal.sigmf-meta
    meta_path = fp.with_suffix(".sigmf-meta")
    if not meta_path.exists():
        # Also check stem.sigmf-meta for files like signal.iq
        meta_path = fp.parent / (fp.stem + ".sigmf-meta")
    if not meta_path.exists():
        return None

    try:
        with open(meta_path, encoding="utf-8") as fh:
            meta = _json.load(fh)
    except Exception:
        return None

    result: dict = {}
    glob = meta.get("global", {})
    result["sample_rate"] = glob.get("core:sample_rate")
    result["datatype"] = glob.get("core:datatype")

    captures = meta.get("captures", [])
    if captures:
        result["center_frequency"] = captures[0].get("core:frequency")
    else:
        result["center_frequency"] = None

    return result
