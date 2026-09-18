"""Shared MVP constants. Full report schema lives in info.md §13."""

TOOL_NAME = "RF Signal Analysis Workbench MVP"
VERSION = "0.1.0"

DEFAULT_SAMPLE_RATE = 100_000.0
DEFAULT_IQ_FORMAT = "complex64"
DEFAULT_SYNC_WORD = "0x1ACFFC1D"
CORRELATION_THRESHOLD = 0.85

# Verified-decode search budget. The blind hypothesis search (de-interleave ->
# FEC decode -> CRC-16) must stay well inside the §NFR-03 wall-clock target even
# for multi-megabit captures, so it only ever examines a bounded prefix.
DECODE_MAX_BITS = 16_384
DECODE_TIME_BUDGET_S = 4.0
#: Preview size for payload hex/text rendering in the report.
PAYLOAD_PREVIEW_BYTES = 256
