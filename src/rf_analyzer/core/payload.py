"""Header/payload identification and payload presentation.

The problem statement asks for "header/payload identification" and a readable
view of the recovered bit stream. This module turns the demodulated bit stream
into something a human can inspect:

* :func:`slice_payload` — the bits that follow the detected sync word.
* :func:`hexdump` / :func:`printable_text` — classic hex + ASCII rendering.
* :func:`payload_report` — the ``payload`` block of the ``info.md`` §13 report.

Honesty contract (``info.md`` §30)
----------------------------------
Two very different things are reported side by side and never conflated:

* ``raw`` — the demodulated bits *as received*, packed into bytes. For a coded
  transmission this is the still-encoded bit stream: it looks like noise and
  must never be presented as the transmitted message.
* ``decoded`` — only populated when a real decoder produced a CRC-16-valid
  payload. ``decoded["available"]`` is ``False`` otherwise.

Nothing in this module ever guesses a message.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rf_analyzer.core.fec import bits_to_bytes

#: Byte values that render as themselves in the ASCII column.
_TAB = "\t"
_CR = "\r"
_LF = "\n"

#: How many bytes of the raw payload get rendered in the report.
DEFAULT_PREVIEW_BYTES = 256
#: Classic hexdump row width.
HEXDUMP_WIDTH = 16


def slice_payload(bits: np.ndarray, start_offset: int) -> np.ndarray:
    """Return the bits after ``start_offset`` (the consumed sync word).

    Out-of-range offsets yield an empty array rather than raising, so a
    truncated capture degrades to "no payload" instead of an error.
    """
    arr = np.asarray(bits, dtype=np.uint8).ravel()
    start = max(0, int(start_offset))
    if start >= arr.size:
        return np.array([], dtype=np.uint8)
    return arr[start:]


def classify_bytes(data: bytes) -> str:
    """Coarse content class: ``empty`` | ``text`` | ``mixed`` | ``binary``.

    A coded stream almost always lands on ``binary``, which is the point: it
    tells the user at a glance that a decoder is still required.
    """
    if not data:
        return "empty"
    printable = sum(1 for b in data if 32 <= b < 127 or b in (9, 10, 13))
    ratio = printable / len(data)
    if ratio >= 0.9:
        return "text"
    if ratio >= 0.5:
        return "mixed"
    return "binary"


def printable_text(data: bytes, max_chars: int = DEFAULT_PREVIEW_BYTES) -> str:
    """Render bytes as text, escaping control characters and eliding the tail."""
    out: list[str] = []
    for b in bytes(data[:max_chars]):
        if 32 <= b < 127:
            out.append(chr(b))
        elif b == 9:
            out.append(_TAB)
        elif b == 10:
            out.append(_LF)
        elif b == 13:
            out.append(_CR)
        else:
            out.append(".")
    text = "".join(out)
    if len(data) > max_chars:
        text += "..."
    return text


def hexdump(
    data: bytes, width: int = HEXDUMP_WIDTH, max_bytes: int = DEFAULT_PREVIEW_BYTES
) -> list[str]:
    """Classic ``offset  hex  |ascii|`` dump, truncated to ``max_bytes``."""
    if width < 1:
        raise ValueError("width must be >= 1")
    view = bytes(data[:max_bytes])
    lines: list[str] = []
    for off in range(0, len(view), width):
        chunk = view[off : off + width]
        hex_part = " ".join(f"{b:02x}" for b in chunk).ljust(width * 3 - 1)
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append(f"{off:08x}  {hex_part}  |{ascii_part}|")
    if len(data) > max_bytes:
        lines.append(f"... {len(data) - max_bytes} more byte(s) not shown")
    return lines


def _describe(data: bytes, max_bytes: int) -> dict[str, Any]:
    """Common presentation block for a byte string."""
    return {
        "bytes": len(data),
        "content_type": classify_bytes(data),
        "hex_preview": data[:max_bytes].hex(),
        "text_preview": printable_text(data, max_bytes),
        "hexdump": hexdump(data, max_bytes=max_bytes),
    }


def payload_report(
    bits: np.ndarray,
    *,
    start_offset: int = 0,
    decoded: dict | None = None,
    max_bytes: int = DEFAULT_PREVIEW_BYTES,
) -> dict[str, Any]:
    """Build the ``payload`` report block.

    Args:
        bits: Full demodulated hard-bit stream.
        start_offset: Bit index where the payload begins (i.e. the sync-word
            offset plus the sync-word length). ``0`` means "no header found".
        decoded: A validated decode record (``crc_pass`` must be ``True`` for
            it to be reported as available). Typically
            ``search_interleaver(...)["best"]``.
        max_bytes: Preview size for hex/text/hexdump rendering.

    Returns:
        Dict with ``header_offset``, ``payload_bit_offset``, ``payload_bits``,
        ``raw`` and ``decoded``.
    """
    payload_bits = slice_payload(bits, start_offset)
    raw_bytes = bits_to_bytes(payload_bits)

    block: dict[str, Any] = {
        "header_offset": int(start_offset),
        "payload_bit_offset": int(max(0, start_offset)),
        "payload_bits": int(payload_bits.size),
        "raw": _describe(raw_bytes, max_bytes),
        "decoded": {
            "available": False,
            "scheme": None,
            "interleaver": None,
            "crc_pass": None,
            "confidence": 0.0,
            "bytes": 0,
            "content_type": "empty",
            "hex": "",
            "text": "",
            "hexdump": [],
            "note": "no CRC-valid decode; payload remains coded",
        },
    }

    if decoded and decoded.get("crc_pass"):
        try:
            data = bytes.fromhex(str(decoded.get("payload_hex", "")))
        except ValueError:
            data = b""
        params = decoded.get("params") or {}
        block["decoded"] = {
            "available": True,
            "scheme": decoded.get("candidate"),
            "interleaver": params.get("interleaver"),
            "crc_pass": True,
            "confidence": float(decoded.get("confidence", 0.0)),
            "bytes": len(data),
            "content_type": classify_bytes(data),
            "hex": data.hex(),
            "text": printable_text(data, max_bytes),
            "hexdump": hexdump(data, max_bytes=max_bytes),
            "errors_corrected": int(decoded.get("errors_corrected", 0)),
            "note": "CRC-16 verified",
        }

    return block
