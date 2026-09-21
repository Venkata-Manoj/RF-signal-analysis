"""Raw .IQ / .wav loading. Full spec: info.md §12.1.

Contracts (do not change without updating AGENTS.md):
- complex64 native; int16 interleaved I/Q ÷32768; uint8 offset-128 ÷128;
  int8 (HackRF signed-8 native) interleaved I/Q ÷128.
- Truncate odd trailing byte. Little-endian default.
- WAV stereo L=I/R=Q; mono = real-only. Sample rate from file for WAV,
  user-supplied for .IQ.
- Dtype aliases (case-insensitive): ci8→int8, cu8→uint8, ci16→int16,
  cf32→complex64. Kept for SigMF/HackRF naming compatibility.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def load_iq(
    file_path: str | Path, dtype: str = "complex64", endian: str = "little"
) -> np.ndarray:
    """
    Load raw IQ file.

    Supported dtype (case-insensitive, backward compatible):
    - complex64 (alias: cf32)
    - int16 (alias: ci16)
    - uint8 (alias: cu8)
    - int8 (alias: ci8, HackRF signed-8 native: interleaved I/Q ÷128.0)

    ``endian`` applies to multi-byte dtypes (int16); single-byte dtypes
    (int8/uint8) accept the parameter for API symmetry but it is a no-op.
    Odd trailing elements are truncated so I/Q stay paired.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"IQ file not found: {file_path}")

    # Normalize dtype + resolve SigMF/HackRF aliases. Keep backward compat
    # with the historic exact strings ("complex64", "int16", "uint8").
    _aliases = {
        "ci8": "int8",
        "cu8": "uint8",
        "ci16": "int16",
        "cf32": "complex64",
    }
    dtype_norm = str(dtype).lower()
    dtype_norm = _aliases.get(dtype_norm, dtype_norm)

    if dtype_norm == "complex64":
        samples = np.fromfile(file_path, dtype=np.complex64)

    elif dtype_norm == "int16":
        raw = np.fromfile(file_path, dtype=np.int16)
        if endian == "big":
            raw = raw.byteswap()
        if len(raw) % 2 != 0:
            raw = raw[:-1]
        samples = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
        samples = samples / 32768.0

    elif dtype_norm == "int8":
        raw = np.fromfile(file_path, dtype=np.int8)
        if endian == "big":
            # No-op for single-byte elements; kept for API symmetry
            # with the int16 path.
            raw = raw.byteswap()
        if len(raw) % 2 != 0:
            raw = raw[:-1]
        samples = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
        samples = samples / 128.0

    elif dtype_norm == "uint8":
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


def load_wav(file_path: str | Path) -> tuple[np.ndarray, float]:
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


# --------------------------------------------------------------------------- #
# Statistical IQ-format auto-detection
# --------------------------------------------------------------------------- #

IQ_FORMATS = ("complex64", "int16", "uint8", "int8")

# Full-scale span of each raw format, used to judge dynamic-range usage.
_FULL_SCALE = {"int16": 65535.0, "uint8": 255.0, "int8": 255.0}


def _read_preview(file_path: Path, max_bytes: int) -> bytes:
    with open(file_path, "rb") as handle:
        return handle.read(max(8, int(max_bytes)))


def _score_complex64(samples: np.ndarray) -> float:
    """Plausibility of a complex64 interpretation.

    The decisive signal: raw integer data read as float32 lands almost
    entirely in the denormal range (magnitudes around 1e-38), because the
    integer bytes become exponent bits. Genuine float captures sit in a
    sane magnitude band.
    """
    if samples.size < 16:
        return 0.0
    magnitude = np.abs(samples)
    finite = np.isfinite(magnitude)
    if float(np.mean(finite)) < 0.99:
        return 0.0
    magnitude = magnitude[finite]
    peak = float(np.max(magnitude))
    if peak < 1e-6 or peak > 1e15:
        return 0.0
    if float(np.mean(magnitude < 1e-30)) > 0.5:
        return 0.0
    return 1.0


def _score_real_iq(samples: np.ndarray, dtype_name: str) -> float:
    """Plausibility of an integer IQ interpretation.

    Two features do the real work, both measured empirically against a
    16-cell confusion matrix of true-format x tested-format combinations:

    * **smoothness** — a real capture is oversampled, so ``I[n]`` tracks
      ``I[n-1]`` closely. Every *correct* interpretation of an oversampled
      capture scores ~0.97 here; every mis-typed one collapses (<=0.76), and
      for float32-bytes-read-as-int16 it falls to ~0.00.
    * **I/Q decorrelation** — I and Q of a genuine capture are independent.
      Mis-typing the sample pairing makes the two channels near-identical
      (|corr| up to 0.97), which is a strong tell.

    The score is their product, so *both* must hold. Gating on AC coupling,
    non-degenerate I/Q spread and DC offset rejects the remaining cases.

    Honest limitation: for a *white* signal at one sample per symbol (no
    oversampling) no format is distinguishable this way, and the score stays
    near zero for all of them. That is reported as low confidence rather than
    a confident wrong answer.
    """
    if samples.size < 16:
        return 0.0

    real = np.real(samples)
    imag = np.imag(samples)
    std_i = float(np.std(real))
    std_q = float(np.std(imag))
    peak_std = max(std_i, std_q)
    if peak_std <= 1e-12:
        return 0.0

    # Gate 1: a genuine capture is AC-coupled (small DC relative to spread).
    dc = max(abs(float(np.mean(real))), abs(float(np.mean(imag)))) / peak_std
    if dc > 1.5:
        return 0.0

    # Gate 2: both channels must carry energy.
    if min(std_i, std_q) / peak_std < 0.05:
        return 0.0

    # Feature 1: I/Q must be uncorrelated.
    correlation = 1.0
    if std_i > 0.0 and std_q > 0.0:
        with np.errstate(invalid="ignore", divide="ignore"):
            value = float(np.corrcoef(real, imag)[0, 1])
        correlation = abs(value) if np.isfinite(value) else 1.0

    # Feature 2: oversampled captures are smooth from sample to sample.
    power = float(np.mean(real**2))
    smoothness = (
        float(np.abs(np.mean(real[1:] * real[:-1])) / power) if power > 0 else 0.0
    )

    score = smoothness * (1.0 - min(correlation, 1.0))

    # Gentle tie-break: prefer the reading that uses more of its own range.
    span = max(
        float(np.max(real) - np.min(real)),
        float(np.max(imag) - np.min(imag)),
    )
    usage = min(1.0, span / _FULL_SCALE.get(dtype_name, 65535.0))
    score *= 0.9 + 0.1 * usage
    score *= max(0.0, 1.0 - 0.5 * dc)

    return float(min(max(score, 0.0), 1.0))


def detect_iq_format(file_path: str, max_bytes: int = 2_000_000) -> dict:
    """Guess the raw sample format of an ``.iq`` file from its statistics.

    Raw IQ has no header, so a non-expert user cannot know whether their
    capture is complex64, int16, uint8 or int8. This scores all four
    interpretations against the heuristics above and returns the winner.

    Returns:
        ``{"format", "confidence", "scores", "detail"}`` where ``confidence``
        is the winner's score normalised against the runner-up (so a clear
        win approaches 1.0 and an ambiguous file sits near 0.5).

    This is a heuristic, not a guarantee — always allow the user to override
    it (``info.md`` NFR-10).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"IQ file not found: {path}")

    raw = _read_preview(path, max_bytes)
    scores: dict[str, float] = {}
    detail: dict[str, str] = {}

    # --- complex64 ---
    usable = (len(raw) // 8) * 8
    if usable >= 128:
        block = np.frombuffer(raw[:usable], dtype=np.complex64)
        scores["complex64"] = _score_complex64(block)
        detail["complex64"] = f"peak|z|={float(np.max(np.abs(block))):.3g}"
    else:
        scores["complex64"] = 0.0
        detail["complex64"] = "too few bytes"

    # --- integer interleaved I/Q ---
    for name, dtype, offset in (
        ("int16", np.int16, 0.0),
        ("int8", np.int8, 0.0),
        ("uint8", np.uint8, 128.0),
    ):
        item = np.dtype(dtype).itemsize
        aligned = raw[: (len(raw) // item) * item]  # frombuffer needs exact fit
        arr = np.frombuffer(aligned, dtype=dtype) if aligned else np.array([], dtype)
        if arr.size % 2:
            arr = arr[:-1]
        if arr.size < 32:
            scores[name] = 0.0
            detail[name] = "too few samples"
            continue
        scale = 32768.0 if name == "int16" else 128.0
        iq = (arr[0::2].astype(np.float64) - offset) + 1j * (
            arr[1::2].astype(np.float64) - offset
        )
        iq = iq / scale
        scores[name] = _score_real_iq(iq, name)
        detail[name] = (
            f"std(I)={float(np.std(iq.real)):.4g} std(Q)={float(np.std(iq.imag)):.4g}"
        )

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_name, best_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0

    if best_score <= 0.0:
        confidence = 0.0
    elif best_score >= 1.0:
        confidence = 1.0
    else:
        margin = best_score - runner_up
        confidence = float(min(0.99, max(0.5, 0.5 + margin)))

    return {
        "format": best_name,
        "confidence": round(float(confidence), 4),
        "scores": {k: round(float(v), 4) for k, v in scores.items()},
        "detail": detail,
    }


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
