# AGENTS.md — RF Signal Analysis Workbench MVP

## Status / source of truth
- Repo at Milestone 1+ (Gap-Closure complete): 60 pytest tests passing, `python scripts/verify_mvp.py → PASS`, ruff/black clean. Implements 16-QAM demod, sampling-rate estimation, FEC/de-interleave candidate scoring, portal task alignment, CI workflow.
- `info.md` is authoritative for scope, schemas, and starter code. Implemented `info.md` §9–§12 gap-closure per `docs/superpowers/plans/2026-09-15-sih26147-gap-closure.md`.
- Target layout (§10): `src/rf_analyzer/{config.py,pipeline.py,core/{io,dsp,demod,correlator,report}.py,gui/main_window.py}`, `scripts/{generate_test_data,run_app,run_pipeline,verify_mvp}.py`, `tests/{unit,integration}`, `sample_data/`, `output/`.

## Stack (do not swap)
- Python 3.11, `numpy/scipy/soundfile/PyQt6/pyqtgraph`; dev `pytest/ruff/black/mypy`. No GNU Radio.
- `pytest.ini` must set `testpaths=tests`, `pythonpath=src`.

## Contracts agents get wrong
- Headless entrypoint is `src/rf_analyzer/pipeline.py:analyze_file(request: dict) -> report dict` — GUI must call it, never duplicate DSP logic. Report schema is `info.md` §13 (keys `meta/input/signal/modulation/demodulation/correlation/fec/interleaving/warnings/errors`).
- `core/io.py`: `.iq` dtypes `complex64` (native), `int16` (interleaved I/Q, ÷32768.0), `uint8` (offset 128, ÷128.0); truncate odd trailing byte; little-endian default. `.wav`: stereo L=I/R=Q, mono=real-only; sample rate comes from file for `.wav`, but **must be user-supplied for `.iq`** (raw has no metadata). Fail gracefully on unsupported suffix.
- MVP DSP/demod are intentionally naive (§12): PSD-peak center freq, median-floor bandwidth/SNR, phase-aligned BPSK/QPSK (`real>0`/`imag>0`), 2-FSK via `diff(unwrap(angle))`. No Costas loop / timing recovery / equalization — that is V2 (§32).
- Sync words are hex strings (e.g. `0x1ACFFC1D`) via `hex_to_bits` + sliding hard-bit `sliding_correlate`/`find_header` (threshold 0.85). FEC/interleaving **blind identification** stays candidate-score only — never claim blind detection (see demo-defense lines, §30).
- `core/fec.py` has TWO layers, do not conflate them:
  1. Real codecs (from scratch, pure numpy): `crc16_ccitt`/`crc32_ieee`, K=7 r=1/2 `conv_encode`+hard-decision `viterbi_decode` (G1=0o171, G2=0o133), GF(256) + `rs_encode`/`rs_decode` (RS(255,223) and shortened), `(3,4)`-regular systematic `ldpc_encode`/`ldpc_decode` (min-sum BP, `H=[A|I]`, `p = A@u mod 2`), `concatenated_encode`/`concatenated_decode` (CRC-16 → RS outer → conv inner).
  2. `score_fec_candidates` = blind heuristic, confidence capped at 0.5, `crc_pass=None`. `decode_hypotheses(bits, start_offset=...)` performs the real decodes and only reports >0.9 confidence when a CRC-16 actually passes. Never raise `fec.confidence` above 0.5 — verified decoding goes in the separate `fec.decoded` field.
- `rs_encode` returns **parity symbols only**; build systematic codewords as `msg + rs_encode(msg, nsym)`. The CRC-16 sits *inside* the RS message (payload+CRC+parity), so strip the trailing `nsym` bytes before `crc16_check`.

## Commands (Windows PowerShell host — no `make`, no `source`, no `head`)
```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-dev.txt
python scripts/generate_test_data.py   # required before integration tests; seeded (42)
pytest                                  # all
pytest tests/unit/test_io.py -q         # single file/module pattern
$env:QT_QPA_PLATFORM="offscreen"; pytest tests/integration -q  # GUI smoke test needs this
python scripts/verify_mvp.py            # must print PASS (§23)
python scripts/run_app.py               # GUI; headless check: scripts/run_pipeline.py
```
- Order matters: `install → generate_test_data → pytest`. Integration tests read `sample_data/bpsk.iq|bpsk.wav` and fail if data wasn't generated.
- Perf budget (NFR): ≤100 MB files, 1–2 M samples responsive, 1 M samples <10 s. No RF hardware / network needed for tests or demo.

## Gotchas
- `sample_data/*.iq|*.wav` and `output/` are gitignored artifacts — never commit; `output/` holds reports/bitstreams.
- BPSK clean synthetic must hit BER <0.01 and correlation score >0.9 or the pipeline is broken (§16/§22).

## UI skill (installed)
- `ui-ux-pro-max` lives in `.opencode/skills/` (OpenCode-native path); use it for all GUI work in `gui/main_window.py` (PyQt6 + pyqtgraph plots, result table, log console per `info.md` §6).
- Search script needs `python`, not `python3`, on this Windows host: `python .opencode/skills/ui-ux-pro-max/scripts/search.py "<need>" --design-system -p "<Page>"`. Never `pip install` for the skill (stdlib-only).
