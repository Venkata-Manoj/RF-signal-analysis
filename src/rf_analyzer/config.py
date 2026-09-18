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

# Constellation-fit cross-check for the modulation estimate.
#
# The classifier picks a modulation from higher-order statistics; EVM then
# measures how well the *chosen* constellation explains the received symbols.
# A poor fit is strong evidence the estimate is wrong -- a real GPS L1 capture
# (BPSK) was classified as 16-QAM and fitted at 39% EVM, while a correct BPSK
# fit is a few percent. Above this threshold the pipeline warns and names any
# alternative that fits better, instead of presenting a bad guess as fact.
MODULATION_FIT_EVM_WARN = 30.0

#: Constellations evaluated when the chosen one fits poorly. Reported as
#: context only -- see the note in pipeline.analyze_file about EVM not being
#: comparable across constellation sizes.
MODULATION_FIT_CANDIDATES = ("BPSK", "QPSK", "8PSK", "16-QAM", "64-QAM")

#: Symbol cap for the fit cross-check, so a multi-megabit capture stays fast.
MODULATION_FIT_MAX_SAMPLES = 100_000
