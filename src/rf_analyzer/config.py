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

# 2-FSK symbol-period (samples per symbol) recovery.
#
# The MVP demodulator works one bit per sample, which is only correct when the
# capture already has one sample per symbol. A real 2-FSK burst does not: the
# project's own sample_data/fsk2.iq spends 100 samples on every bit, so the
# naive path smears each bit across 100 demodulated samples and the sync word
# can never be found. estimate_fsk_symbol_period recovers the true period by
# fitting a piecewise-constant model to the instantaneous frequency.
#
# Below this period the piecewise-constant model has too few samples per
# symbol to be distinguishable from noise, so the estimator declines and the
# demodulator falls back to its documented 1-bit-per-sample behaviour.
FSK_MIN_SYMBOL_PERIOD = 3

#: Largest reconstruction residual accepted as "this really is the period".
#: A correct period reconstructs the instantaneous frequency almost exactly
#: (residual ~ 0, or ~1/period for a generator that repeats one sample at each
#: symbol boundary); a wrong one leaves ~0.25 of the mean amplitude unexplained.
FSK_MAX_RECONSTRUCTION_RESIDUAL = 0.15

#: Transmit-side 2-FSK defaults, shared by framing.modulate and the generator.
FSK_SYMBOL_DURATION_S = 0.001
FSK_FREQ_LOW_HZ = -5000.0
FSK_FREQ_HIGH_HZ = 5000.0
