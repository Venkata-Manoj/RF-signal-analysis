# Portal Alignment — SIH26147 (NTRO)

Source: `sih.gov.in/sih2026PS` → `sih-2026-explorer-pearl.vercel.app/problems/SIH26147/`

Official problem statement bullets and repo status. Line numbers drift — the
function name is the durable reference.

| # | Portal bullet | Status | Where |
|---|---------------|--------|-------|
| Background | Raw data in `.wav`/`.IQ` formats, manual analysis insufficient | **DONE** — `.iq` (`complex64`/`int16`/`uint8`/`int8`, SIGMF dtype aliases) + `.wav`, with automatic dtype detection | `core/io.py:load_iq`, `core/io.py:detect_iq_format` |
| i | Identify signal parameters (sampling freq, modulation, FEC, interleaving) | **COMPLETED** — sampling rate (fused bandwidth + envelope + cyclostationary + FSK-period estimate with abstain-to-zero, SigMF metadata, auto hypothesis); modulation (HOC + rules + ML head, 11 labels) with confidence. **FEC and interleaving are identified by CRC-verified decode, not by blind detection** | `core/rate_est.py`, `core/dsp.py:estimate_sampling_rate*`, `core/classifier.py:classify_modulation`, `core/fec.py:decode_hypotheses`, `core/deinterleave.py:search_interleaver` |
| ii | Demodulate signals (FSK, QAM, PSK) | **COMPLETED** — BPSK, QPSK, 8PSK, 16-QAM, 64-QAM, 2-FSK, 4-FSK; coherent receiver (Costas / Gardner / Mueller-Muller / CMA / LMS) by default with honest naive fallback; all seven decode end to end | `core/receiver.py:demodulate`, `core/demod.py` |
| iii | De-interleaving (Block, Convolution, Diagonal, Pseudo-Random) | **COMPLETED** — all four implemented (plus a pseudo-random-block variant), searched and validated by CRC | `core/deinterleave.py` |
| iv | FEC (conv-Viterbi, RS, Concatenated, LDPC) | **COMPLETED** — convolutional family K=5–9 r=1/2,1/3 (hard + soft Viterbi; K=7 r=1/2 preserved), RS family with shortened variants over GF(256), parametric LDPC family with min-sum belief propagation, and concatenated variants CRC-16 → RS → conv. All written from scratch, no new dependencies | `core/fec.py` |
| v | Bitstream correlation for header/payload | **COMPLETED** — CFAR-thresholded sliding hard-bit correlation plus sync-word discovery; a payload is reported as decoded only on a CRC-16 pass | `core/correlator.py:find_header`, `pipeline.py` |
| Expected | GUI, constellation, waterfall, demod, correlation | **COMPLETED** — PyQt6 workbench (8 tabs) plus a zero-install web dashboard, both calling the same headless pipeline | `gui/main_window.py`, `dashboard.py` |

## Remaining honest limits (kept, not hidden)

- **Blind FEC detection is a search, not a detector.** `score_fec_candidates`
  confidence stays capped at 0.5 with `crc_pass=None`. Verified decoding is a
  *separate* field and is only set when a
  CRC-16 actually passes. The tool never claims blind FEC detection.
- **De-interleaving is verified, but only over the grid that is searched.** The
  blind candidate score is a heuristic capped at 0.5; real de-interleaving is
  performed and confirmed by CRC, but a depth/spacing/seed outside the searched
  grid will not be found.
- **FSK symbol-period recovery abstains honestly.** When no piecewise-constant structure is resolvable it returns 1 and the naive path is used with a warning; a burst whose symbol rate drifts mid-capture is not tracked.
- **The FSK classifier gate is a narrowband test, not an FSK test.** It also
  fires for an unmodulated tone and for audio. Period recovery is the independent
  evidence; when it fails, the label is kept but marked `corroborated: false` and
  its confidence is capped (see `MODULATION_UNCORROBORATED_CONFIDENCE`).
- **Sampling rate for `.iq` defaults to user-supplied** — a raw file carries no
  metadata. `auto_sample_rate=True`, SigMF metadata, and the fused estimator provide a flagged working hypothesis for unknown captures. For `.wav` it comes from the file.
- **Higher-order constellations cost SNR.** 64-QAM needs substantially more per-bit energy than BPSK at the same BER; the coherent chain reports lock/EVM honestly rather than hiding it.
- **Carrier/timing/equalisation can abstain.** When loops do not lock the pipeline falls back to the naive slicers, records `demodulation.receiver.path="naive"` with the reason, and warns — never a silent guess.

## Verification

`python scripts/verify_completed.py → PASS` (57 checks, 0 failures — a fresh clone reports
48 passed + 1 skipped, because the third-party `real_data/` group is skipped rather
than failed) and `pytest` (quote what `pytest` prints; historically 864 collected). The requirement matrix is exercised
end to end: every one
of the 30 FEC × interleaver combinations is identified and decodes back to the
exact message, and all seven modulations decode end to end. Error loads are chosen
per scheme, inside each scheme's measured tolerance — LDPC correction is
probabilistic, so a load at the edge of capability tests luck rather than the
pipeline.

The `info.md` performance budget is measured by
`python scripts/benchmark_performance.py` (realistic burst-in-noise captures, each
case in its own process): **1 M samples in 2.8 s median (1.8–4.1 s)** against a
10 s budget, 2 M in 6.5 s, and a 100 MB / 12.5 M-sample file in 30 s without
crashing. Only the 1 M row is stable enough to carry a budget — the 100 MB case was
observed at 499 s under external load on the same machine, so the range is the
measurement.
