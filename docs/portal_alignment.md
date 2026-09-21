# Portal Alignment — SIH26147 (NTRO)

Source: `sih.gov.in/sih2026PS` → `sih-2026-explorer-pearl.vercel.app/problems/SIH26147/`

Official problem statement bullets and repo status. Line numbers drift — the
function name is the durable reference.

| # | Portal bullet | Status | Where |
|---|---------------|--------|-------|
| Background | Raw data in `.wav`/`.IQ` formats, manual analysis insufficient | **DONE** — `.iq` (`complex64`/`int16`/`uint8`/`int8`, SIGMF dtype aliases) + `.wav`, with automatic dtype detection | `core/io.py:load_iq`, `core/io.py:detect_iq_format` |
| i | Identify signal parameters (sampling freq, modulation, FEC, interleaving) | **DONE** — sampling rate (narrowband heuristic, wideband estimate, candidate hypotheses, and derived from the 2-FSK symbol period); modulation with a confidence. **FEC and interleaving are identified by CRC-verified decode, not by blind detection** | `core/dsp.py:estimate_sampling_rate*`, `core/fec.py:decode_hypotheses`, `core/deinterleave.py:search_interleaver` |
| ii | Demodulate signals (FSK, QAM, PSK) | **DONE** — 2-FSK (with symbol-period recovery), 16-QAM, BPSK, QPSK; all four decode end to end | `core/demod.py:demod_2fsk`, `demod_qam16`, `demod_bpsk`, `demod_qpsk` |
| iii | De-interleaving (Block, Convolution, Diagonal, Pseudo-Random) | **DONE** — all four implemented (plus a pseudo-random-block variant), searched and validated by CRC | `core/deinterleave.py` |
| iv | FEC (conv-Viterbi, RS, Concatenated, LDPC) | **DONE** — K=7 r=1/2 convolutional + hard-decision Viterbi, RS(255,223)/RS(255,239) over GF(256), (3,4)-regular LDPC with min-sum belief propagation, and concatenated CRC-16 → RS → conv. All written from scratch, no new dependencies | `core/fec.py` |
| v | Bitstream correlation for header/payload | **DONE** — CFAR-thresholded sliding hard-bit correlation; a payload is reported as decoded only on a CRC-16 pass | `core/correlator.py:find_header`, `pipeline.py` |
| Expected | GUI, constellation, waterfall, demod, correlation | **DONE** — PyQt6 workbench plus a zero-install web dashboard, both calling the same headless pipeline | `gui/main_window.py`, `dashboard.py` |

## Honest limitations (per `info.md §30`)

- **Blind FEC detection is a search, not a detector.** `score_fec_candidates`
  confidence stays capped at 0.5 with `crc_pass=None`. Verified decoding is a
  *separate* field (`fec.decoded`, `validated=True`) and is only set when a
  CRC-16 actually passes. The tool never claims blind FEC detection.
- **De-interleaving is verified, but only over the grid that is searched.** The
  blind candidate score is a heuristic capped at 0.5; real de-interleaving is
  performed and confirmed by CRC, but a depth/spacing/seed outside the searched
  grid will not be found.
- **The 2-FSK symbol period is recovered, not tracked.** No timing-recovery loop,
  so a burst whose symbol rate drifts will not be followed.
- **The FSK classifier gate is a narrowband test, not an FSK test.** It also
  fires for an unmodulated tone and for audio. Period recovery is the independent
  evidence; when it fails, the label is kept but marked `corroborated: false` and
  its confidence is capped (see `MODULATION_UNCORROBORATED_CONFIDENCE`).
- **Sampling rate for `.iq` must be user-supplied** — a raw file carries no
  metadata saying otherwise. For `.wav` it comes from the file.
- **Sampling-rate estimation is a heuristic.** Narrowband uses `bandwidth × 2.2`
  plus a candidate-hypothesis search; it is not blind. For 2-FSK the rate is
  derived from the recovered symbol period instead.
- **No QAM beyond 16-QAM**; higher-order QAM is V2.
- **No Costas loop / timing recovery / equalization** (V2 §32).

## Verification

`python scripts/verify_mvp.py → PASS` (57 checks, 0 failures — a fresh clone reports
48 passed + 1 skipped, because the third-party `real_data/` group is skipped rather
than failed) and `pytest` (473 passed, 1 skipped). The requirement matrix is exercised
end to end: every one
of the 30 FEC × interleaver combinations is identified and decodes back to the
exact message, and all four modulations decode end to end. Error loads are chosen
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
