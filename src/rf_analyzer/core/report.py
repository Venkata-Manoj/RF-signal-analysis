"""JSON report + bitstream export. Full spec: info.md §12.6 / §13."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

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


def json_safe(value: Any) -> Any:
    """Recursively make ``value`` strict-JSON safe.

    ``json.dump`` writes ``Infinity``/``NaN`` by default, but those tokens are
    *not* valid JSON: ``JSON.parse`` in a browser rejects them outright, so a
    report containing one is unreadable to every JavaScript consumer (the web
    dashboard, any downstream tooling). Reports are also consumed outside
    Python, so non-finite floats become ``None`` here and numpy scalars become
    plain Python values.

    Used by :func:`save_report` and by the dashboard's HTTP responses; both
    also pass ``allow_nan=False`` so a leak fails loudly instead of silently
    producing a file or response nobody can parse.
    """
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, bool):  # before the numeric checks: bool is an int
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def save_report(report: dict, path: str) -> None:
    """Save report dict as pretty-printed, strict-JSON JSON (indent 2)."""
    out = Path(path)
    if out.parent != Path("") and str(out.parent) != ".":
        out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(json_safe(report), f, indent=2, allow_nan=False)


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
