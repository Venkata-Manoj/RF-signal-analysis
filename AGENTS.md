# AGENTS.md — RF Signal Analysis Workbench MVP

## Status / source of truth
- Repo is feature-complete for the MVP brief: **426 pytest tests passing (1 skipped)**, `python scripts/verify_mvp.py → PASS` (57 individual checks), ruff/black clean. Implements 16-QAM demod, sampling-rate estimation, 2-FSK symbol-period recovery, real FEC codecs + verified decoding, real interleavers, payload extraction, batch analysis, a PyQt6 GUI and a zero-install web dashboard.
- `README.md` is the user-facing document (written for a non-technical reader) — keep it accurate when behaviour changes.
- `info.md` is authoritative for scope, schemas, and starter code. Implemented `info.md` §9–§12 gap-closure per `docs/superpowers/plans/2026-09-15-sih26147-gap-closure.md`.
- Actual layout: `src/rf_analyzer/{config,pipeline,batch,dashboard}.py`, `core/{io,dsp,classifier,demod,correlator,fec,deinterleave,framing,payload,report}.py`, `gui/main_window.py`, `web/dashboard.html`; `scripts/{generate_test_data,run_app,run_pipeline,serve_dashboard,batch_analyze,fetch_real_data,verify_mvp,benchmark_snr}.py`; `tests/{unit,integration}`; `sample_data/`, `real_data/`, `output/`.

## Stack (do not swap)
- Python 3.11, `numpy/scipy/soundfile/PyQt6/pyqtgraph`; dev `pytest/ruff/black/mypy`. No GNU Radio.
- `pytest.ini` must set `testpaths=tests`, `pythonpath=src`.

## Contracts agents get wrong
- Headless entrypoint is `src/rf_analyzer/pipeline.py:analyze_file(request: dict) -> report dict` — GUI must call it, never duplicate DSP logic. Report schema is `info.md` §13 (keys `meta/input/signal/modulation/demodulation/correlation/fec/interleaving/warnings/errors`).
- `core/io.py`: `.iq` dtypes `complex64` (native), `int16` (interleaved I/Q, ÷32768.0), `uint8` (offset 128, ÷128.0); truncate odd trailing byte; little-endian default. `.wav`: stereo L=I/R=Q, mono=real-only; sample rate comes from file for `.wav`, but **must be user-supplied for `.iq`** (raw has no metadata). Fail gracefully on unsupported suffix.
- MVP DSP/demod are intentionally naive (§12): PSD-peak center freq, median-floor bandwidth/SNR, phase-aligned BPSK/QPSK (`real>0`/`imag>0`), 2-FSK via `diff(unwrap(angle))`. No Costas loop / timing recovery / equalization — that is V2 (§32). **One exception:** 2-FSK *does* recover its symbol period, because a real FSK burst is never one sample per bit and without it the sync word is unreachable. See the honesty contract below.
- Sync words are hex strings (e.g. `0x1ACFFC1D`) via `hex_to_bits` + sliding hard-bit `sliding_correlate`/`find_header` (threshold 0.85). FEC/interleaving **blind identification** stays candidate-score only — never claim blind detection (see demo-defense lines, §30).
- `core/fec.py` has TWO layers, do not conflate them:
  1. Real codecs (from scratch, pure numpy): `crc16_ccitt`/`crc32_ieee`, K=7 r=1/2 `conv_encode`+hard-decision `viterbi_decode` (G1=0o171, G2=0o133), GF(256) + `rs_encode`/`rs_decode` (RS(255,223) and shortened), `(3,4)`-regular systematic `ldpc_encode`/`ldpc_decode` (min-sum BP, `H=[A|I]`, `p = A@u mod 2`), `concatenated_encode`/`concatenated_decode` (CRC-16 → RS outer → conv inner).
  2. `score_fec_candidates` = blind heuristic, confidence capped at 0.5, `crc_pass=None`. `decode_hypotheses(bits, start_offset=...)` performs the real decodes and only reports >0.9 confidence when a CRC-16 actually passes. Never raise `fec.confidence` above 0.5 — verified decoding goes in the separate `fec.decoded` field.
- `rs_encode` returns **parity symbols only**; build systematic codewords as `msg + rs_encode(msg, nsym)`. The CRC-16 sits *inside* the RS message (payload+CRC+parity), so strip the trailing `nsym` bytes before `crc16_check`.
- Report schema is §13 **plus additive blocks** that must not replace it: `payload` (raw rendering + CRC-verified decode), `display`, `quality` (EVM/MER/`fit_ok`), plus one additive *nested* key `modulation.corroborated` (`True`/`False`/`None`). `REQUIRED_REPORT_KEYS` deliberately mirrors §13 only. Error and success reports must expose the **same** top-level shape.

## Honesty contracts (do not weaken these)
- **Never claim a decode without a CRC pass.** Blind `score_fec_candidates`/`score_deinterleave_candidates` stay capped at 0.5 with `crc_pass=None`; only `decode_hypotheses`/`search_interleaver` may set `validated=True`, and only on a real CRC-16 pass. Tests and `verify_mvp.py` enforce this — a noise capture must produce no decode.
- **`json_safe` + `allow_nan=False` at every JSON boundary.** `json.dump` writes `Infinity`/`NaN` by default, which is invalid JSON and makes a browser's `JSON.parse` throw. Non-finite floats become `None` (e.g. `quality.mer_db is None` means "unbounded MER"). `save_report` and the dashboard's HTTP responses both go through it.
- **Sync detection is length-aware (CFAR).** `min_matches_for_length` raises the required match count with the number of positions searched, so the expected false alarms stay ≤0.01. A fixed 0.85 threshold gives ~4.8 false positives on a 500k-bit capture — this was a real bug. Never compare a raw `score` against a constant without `min_score`.
- **EVM is not comparable across constellation sizes.** A denser constellation fits any point cloud better (measured on a real GPS capture: 93.5% BPSK → 26.2% 64-QAM). Use EVM only to judge the *chosen* constellation's fit (`fit_ok`); never to rank modulations.
- **`bandwidth_estimate == 0.0` means "not measurable", not zero.** `estimate_bandwidth` needs ≥2 bins above the noise floor; the pipeline warns when it fails and the derived sample-rate estimate is 0 too. Do not "fix" this by lowering the threshold — for flat-spectrum signals (QPSK) noise is *peakier* than the signal.
- **2-FSK symbol-period recovery never invents a period.** `dsp.estimate_fsk_symbol_period` returns `1` ("no resolvable structure, use the naive path") for BPSK/QPSK/noise/tone, for periods below `FSK_MIN_SYMBOL_PERIOD`, and whenever the piecewise-constant reconstruction residual exceeds `FSK_MAX_RECONSTRUCTION_RESIDUAL`. It searches a window around an autocorrelation-derived centre (a fixed window misses at large periods — the centre is only good to a few percent) and breaks divisor ties by taking the largest near-optimal candidate. The pipeline then derives `symbol_rate_estimate` from the period, overriding the |x|² estimate, which is meaningless for constant-envelope FSK (it reported 395 kHz for a 1 kHz burst). `scripts/generate_test_data.py` and `framing.modulate_2fsk` share one waveform definition (info.md §15.1 phase convention) — do not reintroduce a second copy.
- **An uncorroborated 2-FSK label keeps its label but loses its confidence.** The classifier's FSK gate is a *narrowband* test, not an FSK test: measured on the bundled samples it also fires for an unmodulated tone (instantaneous-frequency variance 1e-16), narrowband audio (0.18) and speech (0.79), against 0.097 for a genuine 2-FSK burst. Period recovery is the independent evidence, so `modulation.corroborated` records `True`/`False`/`None` and, when the label is an *estimate* (`requested == "AUTO"`) and no period could be recovered, `modulation.confidence` is capped to `MODULATION_UNCORROBORATED_CONFIDENCE` (0.35) with a warning. An **explicit** user request for `2-FSK` is left alone — report what was found, do not downgrade the user's own choice. Do not swap in a different discriminator without validating it: median-split separation and the bimodality coefficient were both tried and are numerically degenerate here (a tone's instantaneous frequency is constant to ~1e-8), so they would only replace one wrong label with another.
- **LDPC correction is probabilistic — never test it at the edge of its capability.** At a fixed error count the outcome depends on how the errors land across the 84-bit blocks, so a single seed is not evidence either way. Measured over five seeds: with scattered errors the no-interleaver case simply degrades with load (5/5 up to 8 errors, then 4/5, 3/5, 2/5, 2/5), and the *block* interleaver is genuinely non-monotone (3/5 at 8 errors against 5/5 at 10). The interleaver's real benefit is against **contiguous bursts**: at a burst of 8 bits every interleaver except pseudo-random-block fails outright (0/5), and that one still recovers the message 3/5 of the time. Use the per-scheme loads in `MATRIX_ERRORS` (`tests/integration/test_coded_pipeline.py`) — a load at the edge tests luck, not the pipeline.
- **A framed burst *inside noise* is the realistic capture, and the MVP handles it poorly.** Every coded-capture test writes the frame as the *whole file*, so the suite does not cover burst-in-noise at all. Two measured limitations, both pinned in `tests/integration/test_burst_in_noise.py`:
  1. **The modulation label is not invariant to the surrounding noise.** The same byte-identical BPSK burst is labelled QPSK or 8PSK once enough noise is padded either side, at 0.7–0.85 confidence and with nothing in the report warning that the label is unreliable. When the label is wrong the header usually fails to correlate, so the user sees "no frame found" rather than "the modulation is uncertain" — pointing at the wrong problem. Do **not** try to fix this with EVM: a sparser constellation always fits a denser one's points, so EVM cannot separate BPSK from QPSK. There is no cheap corroboration here, unlike the 2-FSK case.
  2. **The decode search assumes the capture ends at the frame boundary** (`decode_hypotheses(frame_bits=None)`). A frame followed by a trailing tail never decodes, even when the modulation is correct and the sync word is found at the right offset with score 1.0 and `budget_exhausted` is False. Capping `decode_max_bits` to just past the frame restores the decode — which is the evidence that the cause is a missing frame-length estimate, not the noise itself.
  Do not claim real-capture decode support without checking both.

## Commands (Windows PowerShell host — no `make`, no `source`, no `head`)
```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-dev.txt
python scripts/generate_test_data.py   # required before integration tests; seeded (42)
pytest                                  # all
pytest tests/unit/test_io.py -q         # single file/module pattern
$env:QT_QPA_PLATFORM="offscreen"; pytest tests/integration -q  # GUI smoke test needs this
python scripts/verify_mvp.py            # must print PASS (§23); 57 checks
python scripts/run_app.py               # GUI; headless check: scripts/run_pipeline.py
python scripts/serve_dashboard.py --open  # zero-install web dashboard (127.0.0.1:8765)
python scripts/run_pipeline.py sample_data/bpsk.iq --sample-rate 100000 --iq-format auto
python scripts/batch_analyze.py sample_data --csv output/s.csv --html output/s.html
python scripts/fetch_real_data.py       # real OTA captures (network); --list for sources
python scripts/measure_fec_capability.py --quick  # re-derive the documented FEC tolerance numbers
```
- Order matters: `install → generate_test_data → pytest`. Integration tests read `sample_data/bpsk.iq|bpsk.wav` and fail if data wasn't generated.
- Perf budget (NFR): ≤100 MB files, 1–2 M samples responsive, 1 M samples <10 s. No RF hardware needed for tests; only `fetch_real_data.py` touches the network.
- Decode search is bounded by `DECODE_MAX_BITS`/`DECODE_TIME_BUDGET_S` (per-request overridable via `decode_max_bits`/`decode_time_budget_s`). Correctness tests pass generous budgets so a loaded machine cannot decide a result.

## Gotchas
- `sample_data/*.iq|*.wav|*.npy`, `sample_data/coded_manifest.json`, `real_data/*` and `output/` are gitignored artifacts — never commit; `output/` holds reports/bitstreams/payloads.
- BPSK clean synthetic must hit BER <0.01 and correlation score >0.9 or the pipeline is broken (§16/§22).
- `coded_manifest.json` is the ground truth for the 5 coded captures: `file`, `message_hex`, `fec`, `interleaver`, `frame_bits`. `verify_mvp.py` asserts each decodes back to the exact message **and** that the transmitter's scheme is identified from the bit stream alone.
- **Deterministic CRC-16 false positive:** for CRC-16/CCITT-FALSE, `crc16(M || C_hi) == C_lo << 8` always holds, so whenever the byte after a frame is `0x00` a one-byte-longer framing also validates. `_candidate_ends` therefore scans **shortest payload first**. Do not "simplify" it back to longest-first.
- `ldpc_decode` is fully vectorized (sort-based min-sum + `bincount`). The earlier per-edge `np.delete` loop made the suite ~4× slower; keep it vectorized.

## UI skill (installed)
- `ui-ux-pro-max` lives in `.opencode/skills/` (OpenCode-native path); use it for all GUI work in `gui/main_window.py` (PyQt6 + pyqtgraph plots, result table, log console per `info.md` §6).
- Search script needs `python`, not `python3`, on this Windows host: `python .opencode/skills/ui-ux-pro-max/scripts/search.py "<need>" --design-system -p "<Page>"`. Never `pip install` for the skill (stdlib-only).
