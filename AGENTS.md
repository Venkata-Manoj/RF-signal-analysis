# AGENTS.md — RF Signal Analysis Workbench MVP

## Status / source of truth
- Repo is greenfield: `info.md` (full PRD + build plan + starter code + tests) plus `.opencode/skills/` (ui-ux-pro-max). No app code, no git, no manifests yet.
- `info.md` is authoritative for scope, schemas, and starter code. Don't re-derive architecture — implement `info.md` §9–§12 in order (§34): skeleton → `requirements.txt` → `core/io.py` → `scripts/generate_test_data.py` → `core/dsp.py` → `core/demod.py` → `core/correlator.py` → `pipeline.py` → tests → GUI shell.
- Target layout (§10): `src/rf_analyzer/{config.py,pipeline.py,core/{io,dsp,demod,correlator,report}.py,gui/main_window.py}`, `scripts/{generate_test_data,run_app,run_pipeline,verify_mvp}.py`, `tests/{unit,integration}`, `sample_data/`, `output/`.

## Stack (do not swap)
- Python 3.11, `numpy/scipy/soundfile/PyQt6/pyqtgraph`; dev `pytest/ruff/black/mypy`. No GNU Radio.
- `pytest.ini` must set `testpaths=tests`, `pythonpath=src`.

## Contracts agents get wrong
- Headless entrypoint is `src/rf_analyzer/pipeline.py:analyze_file(request: dict) -> report dict` — GUI must call it, never duplicate DSP logic. Report schema is `info.md` §13 (keys `meta/input/signal/modulation/demodulation/correlation/fec/interleaving/warnings/errors`).
- `core/io.py`: `.iq` dtypes `complex64` (native), `int16` (interleaved I/Q, ÷32768.0), `uint8` (offset 128, ÷128.0); truncate odd trailing byte; little-endian default. `.wav`: stereo L=I/R=Q, mono=real-only; sample rate comes from file for `.wav`, but **must be user-supplied for `.iq`** (raw has no metadata). Fail gracefully on unsupported suffix.
- MVP DSP/demod are intentionally naive (§12): PSD-peak center freq, median-floor bandwidth/SNR, phase-aligned BPSK/QPSK (`real>0`/`imag>0`), 2-FSK via `diff(unwrap(angle))`. No Costas loop / timing recovery / equalization — that is V2 (§32).
- Sync words are hex strings (e.g. `0x1ACFFC1D`) via `hex_to_bits` + sliding hard-bit `sliding_correlate`/`find_header` (threshold 0.85). FEC/interleaving are candidate-score stubs only — never claim blind detection (see demo-defense lines, §30).

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
