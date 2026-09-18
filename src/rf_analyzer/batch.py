"""Batch folder analysis with CSV / HTML export.

Analyzing one capture at a time is fine for a demo but useless for a field
survey. This module walks a folder, runs the same public
:func:`rf_analyzer.pipeline.analyze_file` funnel over every capture, and emits
a flat summary that can be exported as CSV (for spreadsheets) or as a
self-contained HTML page (for a report).

The HTML output is deliberately dependency-free — no CDN, no JavaScript — so it
opens correctly offline, which is what a demo machine usually is.
"""

from __future__ import annotations

import csv
import html
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rf_analyzer.config import DEFAULT_IQ_FORMAT, DEFAULT_SYNC_WORD
from rf_analyzer.pipeline import analyze_file

if TYPE_CHECKING:
    from collections.abc import Iterable

#: Suffixes the pipeline can read.
SUPPORTED_SUFFIXES = (".iq", ".wav")

#: Column order for the CSV export and the HTML table.
SUMMARY_COLUMNS: tuple[tuple[str, str], ...] = (
    ("file", "File"),
    ("file_type", "Type"),
    ("iq_format_used", "IQ format"),
    ("sample_rate", "Sample rate"),
    ("num_samples", "Samples"),
    ("duration_s", "Duration (s)"),
    ("center_freq_est", "Centre freq (Hz)"),
    ("bandwidth_est", "Bandwidth (Hz)"),
    ("snr_db", "SNR (dB)"),
    ("symbol_rate_est", "Symbol rate"),
    ("modulation", "Modulation"),
    ("mod_confidence", "Mod conf"),
    ("demod_mode", "Demod mode"),
    ("num_bits", "Bits"),
    ("header_offset", "Header at"),
    ("corr_score", "Corr score"),
    ("payload_bits", "Payload bits"),
    ("payload_bytes_raw", "Payload bytes (raw)"),
    ("decode_attempts", "Decode attempts"),
    ("decode_validated", "Decode verified"),
    ("fec_candidate", "FEC candidate"),
    ("interleaver", "Interleaver"),
    ("decoded_bytes", "Decoded bytes"),
    ("decoded_text", "Decoded message"),
    ("elapsed_s", "Elapsed (s)"),
    ("warnings", "Warnings"),
    ("errors", "Errors"),
)


def discover_captures(folder: str | Path, recursive: bool = False) -> list[Path]:
    """Return every supported capture in ``folder``, sorted by name."""
    root = Path(folder)
    if not root.is_dir():
        raise NotADirectoryError(f"not a folder: {root}")
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(
        p for p in iterator if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


def summarize_report(file_name: str, report: dict, elapsed_s: float) -> dict[str, Any]:
    """Flatten one §13 report into a single summary row."""
    inp = report.get("input", {}) or {}
    sig = report.get("signal", {}) or {}
    mod = report.get("modulation", {}) or {}
    dem = report.get("demodulation", {}) or {}
    corr = report.get("correlation", {}) or {}
    fec = report.get("fec", {}) or {}
    ilv = report.get("interleaving", {}) or {}
    pay = report.get("payload", {}) or {}
    dec = pay.get("decoded", {}) or {}
    search = pay.get("decode_search", {}) or {}
    raw = pay.get("raw", {}) or {}

    return {
        "file": file_name,
        "file_type": inp.get("file_type", ""),
        "iq_format_used": inp.get("iq_format_used", ""),
        "sample_rate": inp.get("sample_rate", ""),
        "num_samples": sig.get("num_samples", ""),
        "duration_s": _round(sig.get("duration_seconds")),
        "center_freq_est": _round(sig.get("center_frequency_estimate")),
        "bandwidth_est": _round(sig.get("bandwidth_estimate")),
        "snr_db": _round(sig.get("snr_db")),
        "symbol_rate_est": _round(sig.get("symbol_rate_estimate")),
        "modulation": mod.get("estimated_type", ""),
        "mod_confidence": _round(mod.get("confidence")),
        "demod_mode": dem.get("mode", ""),
        "num_bits": dem.get("num_bits", ""),
        "header_offset": corr.get("header_offset", ""),
        "corr_score": _round(corr.get("score")),
        "payload_bits": pay.get("payload_bits", ""),
        "payload_bytes_raw": raw.get("bytes", ""),
        "decode_attempts": search.get("attempts", 0),
        "decode_validated": "yes" if search.get("validated") else "no",
        "fec_candidate": fec.get("candidate", ""),
        "interleaver": ilv.get("candidate", ""),
        "decoded_bytes": dec.get("bytes", 0) if dec.get("available") else "",
        "decoded_text": str(dec.get("text", "")) if dec.get("available") else "",
        "elapsed_s": _round(elapsed_s),
        "warnings": "; ".join(map(str, report.get("warnings", []) or [])),
        "errors": "; ".join(map(str, report.get("errors", []) or [])),
    }


def _round(value: Any, digits: int = 4) -> Any:
    try:
        if value is None or value == "":
            return ""
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def analyze_folder(
    folder: str | Path,
    *,
    sample_rate: float = 100_000.0,
    iq_format: str = DEFAULT_IQ_FORMAT,
    modulation: str = "auto",
    sync_word: str | None = DEFAULT_SYNC_WORD,
    decode: bool = True,
    recursive: bool = False,
    progress: Any = None,
) -> list[dict[str, Any]]:
    """Run the pipeline over every capture in ``folder``.

    Args:
        folder: Directory to scan.
        sample_rate: Rate applied to ``.iq`` files (``.wav`` uses its own).
        iq_format: ``complex64`` | ``int16`` | ``uint8`` | ``int8`` | ``auto``.
        modulation: ``auto`` or an explicit mode.
        sync_word: Hex sync word, or ``None``/``""`` to skip correlation.
        decode: Run the CRC-verified FEC/de-interleaver search.
        recursive: Walk sub-folders too.
        progress: Optional ``callable(path, index, total)`` for UI feedback.

    Returns:
        One summary row per capture, in file-name order. Files that fail to
        load still produce a row (with the error recorded) so the batch never
        dies halfway through.
    """
    captures = discover_captures(folder, recursive=recursive)
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(captures, start=1):
        if progress is not None:
            try:
                progress(path, index, len(captures))
            except Exception:
                pass
        request = {
            "file_path": str(path),
            "sample_rate": float(sample_rate),
            "iq_format": str(iq_format),
            "modulation": str(modulation),
            "sync_word": sync_word or "",
            "decode": bool(decode),
        }
        started = time.perf_counter()
        report = analyze_file(request)
        elapsed = time.perf_counter() - started
        rows.append(summarize_report(path.name, report, elapsed))
    return rows


def write_csv(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    """Write summary rows as CSV (UTF-8 with BOM so Excel reads it correctly)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = [key for key, _ in SUMMARY_COLUMNS]
    with open(out, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})
    return out


_HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{
  --bg: #020617; --card: #0E1223; --panel: #0F172A; --muted-bg: #1A1E2F;
  --fg: #F8FAFC; --muted: #94A3B8; --border: #334155;
  --accent: #16A34A; --accent-soft: #14532D; --accent-fg: #4ADE80;
  --warn: #EA580C; --bad: #DC2626;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; padding: 32px; background: var(--bg); color: var(--fg);
  font-family: "Fira Sans", "Segoe UI", system-ui, -apple-system, sans-serif;
  line-height: 1.5;
}}
h1 {{ font-size: 24px; margin: 0 0 4px; }}
h2 {{ font-size: 17px; margin: 28px 0 10px; }}
.sub {{ color: var(--muted); font-size: 13px; margin-bottom: 24px; }}
.cards {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 8px; }}
.card {{
  background: var(--card); border: 1px solid var(--border); border-radius: 8px;
  padding: 14px 18px; min-width: 150px;
}}
.card .k {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .06em; }}
.card .v {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
table {{
  width: 100%; border-collapse: collapse; background: var(--card);
  border-radius: 8px; overflow: hidden; font-size: 13px;
}}
th, td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border); vertical-align: top; }}
th {{ background: var(--panel); font-weight: 600; white-space: nowrap; }}
tr:last-child td {{ border-bottom: none; }}
tbody tr:hover {{ background: var(--muted-bg); }}
.ok {{ color: var(--accent-fg); font-weight: 700; }}
.no {{ color: var(--muted); }}
.err {{ color: var(--bad); }}
.mono {{ font-family: "Fira Code", Consolas, monospace; font-size: 12px; }}
.msg {{
  background: var(--muted-bg); border-left: 3px solid var(--accent);
  border-radius: 4px; padding: 8px 10px; margin: 4px 0 0;
  font-family: "Fira Code", Consolas, monospace; font-size: 12px;
  white-space: pre-wrap; word-break: break-word;
}}
.note {{
  background: var(--muted-bg); border-left: 3px solid var(--warn);
  border-radius: 4px; padding: 10px 12px; color: var(--muted); font-size: 12px;
  margin-top: 8px;
}}
footer {{ color: var(--muted); font-size: 12px; margin-top: 32px; }}
</style>
</head>
<body>
"""


def render_html(
    rows: list[dict[str, Any]],
    title: str = "RF Signal Analysis — Batch Summary",
    subtitle: str = "",
) -> str:
    """Render summary rows as a self-contained dark-theme HTML page."""
    total = len(rows)
    decoded = sum(1 for r in rows if r.get("decode_validated") == "yes")
    failed = sum(1 for r in rows if r.get("errors"))
    total_bits = 0
    for r in rows:
        try:
            total_bits += int(r.get("num_bits") or 0)
        except (TypeError, ValueError):
            pass

    parts: list[str] = [
        _HTML_HEAD.format(title=html.escape(title)),
        f"<h1>{html.escape(title)}</h1>",
        f'<div class="sub">{html.escape(subtitle)}</div>' if subtitle else "",
        '<div class="cards">',
        _card("Captures", total),
        _card("Payloads verified", f"{decoded}/{total}"),
        _card("Bits demodulated", f"{total_bits:,}"),
        _card("Load failures", failed),
        "</div>",
    ]

    if not rows:
        parts.append(
            '<div class="note">No captures found. Generate the sample set with '
            "<span class='mono'>python scripts/generate_test_data.py</span> or "
            "point the batch runner at a folder containing <span class='mono'>"
            ".iq</span> or <span class='mono'>.wav</span> files.</div>"
        )
    else:
        headers = "".join(
            f"<th>{html.escape(label)}</th>" for _, label in SUMMARY_COLUMNS
        )
        body: list[str] = []
        for row in rows:
            cells: list[str] = []
            for key, _label in SUMMARY_COLUMNS:
                value = row.get(key, "")
                text = html.escape(str(value))
                if key == "decode_validated":
                    css = "ok" if value == "yes" else "no"
                    cells.append(f'<td class="{css}">{text}</td>')
                elif key in ("errors", "warnings"):
                    css = "err" if (key == "errors" and value) else "no"
                    cells.append(f'<td class="{css}">{text}</td>')
                elif key in ("file", "decoded_text", "fec_candidate", "interleaver"):
                    cells.append(f'<td class="mono">{text}</td>')
                else:
                    cells.append(f"<td>{text}</td>")
            body.append("<tr>" + "".join(cells) + "</tr>")
        parts.append("<table><thead><tr>" + headers + "</tr></thead><tbody>")
        parts.extend(body)
        parts.append("</tbody></table>")

        verified = [r for r in rows if r.get("decode_validated") == "yes"]
        if verified:
            parts.append("<h2>Recovered messages (CRC-16 verified)</h2>")
            for row in verified:
                parts.append(
                    f'<h2 style="font-size:14px;margin:14px 0 2px">'
                    f"{html.escape(str(row.get('file', '')))}</h2>"
                    f'<div class="sub" style="margin:0">'
                    f"{html.escape(str(row.get('fec_candidate', '')))} · "
                    f"{html.escape(str(row.get('interleaver', '')))} interleaver · "
                    f"{html.escape(str(row.get('decoded_bytes', '')))} bytes</div>"
                    f'<div class="msg">{html.escape(str(row.get("decoded_text", "")))}</div>'
                )

    parts.append(
        '<div class="note">Only payloads that passed an independent CRC-16 check '
        "are reported as decoded. Everything else is shown as the raw, still-coded "
        "bit stream — the tool never guesses a message.</div>"
    )
    parts.append(
        "<footer>Generated by the RF Signal Analysis Workbench MVP.</footer>"
        "</body></html>"
    )
    return "\n".join(parts)


def _card(label: str, value: Any) -> str:
    return (
        f'<div class="card"><div class="k">{html.escape(label)}</div>'
        f'<div class="v">{html.escape(str(value))}</div></div>'
    )


def write_html(
    rows: list[dict[str, Any]],
    path: str | Path,
    title: str = "RF Signal Analysis — Batch Summary",
    subtitle: str = "",
) -> Path:
    """Write the batch summary as a self-contained HTML page."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(rows, title=title, subtitle=subtitle), encoding="utf-8")
    return out
