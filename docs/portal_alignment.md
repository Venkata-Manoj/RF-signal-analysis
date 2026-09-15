# Portal Alignment — SIH26147 (NTRO)

Source: `sih.gov.in/sih2026PS` → `sih-2026-explorer-pearl.vercel.app/problems/SIH26147/`

Official problem statement bullets and repo status:

| # | Portal bullet | Status | File:line |
|---|---------------|--------|-----------|
| Background | Raw data in `.wav`/`.IQ` formats, manual analysis insufficient | **DONE** | `src/rf_analyzer/core/io.py:17` |
| i | Identify signal parameters (sampling freq, modulation, FEC, interleaving) | **DONE (sampling freq + modulation)**; FEC/interleaving candidate-only | `src/rf_analyzer/core/dsp.py:91`, `pipeline.py:177` |
| ii | Demodulate signals (FSK, QAM, PSK) | **DONE** (2-FSK, 16-QAM, BPSK/QPSK) | `src/rf_analyzer/core/demod.py` |
| iii | De-interleaving (Block, Convolution, Diagonal, Pseudo-Random) | **DONE (candidate scores, not blind)** | `src/rf_analyzer/core/deinterleave.py` |
| iv | FEC (conv-Viterbi, RS, Concatenated, LDPC) | **DONE (candidate scores ≤0.5, not blind)** | `src/rf_analyzer/core/fec.py` |
| v | Bitstream correlation for header/payload | **DONE** | `src/rf_analyzer/core/correlator.py` |
| Expected | GUI, constellation, waterfall, demod, correlation | **DONE** | `src/rf_analyzer/gui/main_window.py` |

## Honest limitations (per `info.md §30`)

- FEC confidence capped at 0.5 — **never claims blind FEC detection**.
- De-interleaving is candidate-score only; real de-interleave requires protocol-specific constraints (V2).
- Sampling-rate estimate is a heuristic (`bandwidth * 2.2`), not blind — V2 will add cyclostationary hypothesis testing.
- No QAM beyond 16-QAM; higher-order QAM is V2.
- No Costas loop / timing recovery / equalization (V2 §32).
