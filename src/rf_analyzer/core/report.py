"""JSON report + bitstream export. Full spec: info.md §12.6 / §13."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REQUIRED_TOP_KEYS = [
    "meta",
    "input",
    "signal",
    "modulation",
    "demodulation",
    "correlation",
    "fec",
    "interleaving",
    "warnings",
    "errors",
]


def save_report(report: dict, path: str) -> None:
    """Save report dict as pretty-printed JSON (indent 2)."""
    out = Path(path)
    if out.parent != Path("") and str(out.parent) != ".":
        out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def save_bits(bits, path: str) -> None:
    """Save bitstream to .npy (raw array), .txt (0/1 chars), else packed .bin."""
    out = Path(path)
    if out.parent != Path("") and str(out.parent) != ".":
        out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(bits, dtype=np.uint8).ravel()
    suffix = out.suffix.lower()
    if suffix == ".npy":
        np.save(str(out), arr)
    elif suffix == ".txt":
        with open(out, "w", encoding="utf-8") as f:
            f.write("".join(str(int(b)) for b in arr))
    else:
        # Default packed binary (e.g. .bin): pack bits MSB-first.
        np.packbits(arr).tofile(str(out))


def validate_report(report: dict) -> list[str]:
    """Return list of missing-field errors for required top-level keys."""
    errors: list[str] = []
    for key in REQUIRED_TOP_KEYS:
        if key not in report:
            errors.append(f"Missing required field: {key}")
    return errors
