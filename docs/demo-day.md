# Demo Day — RF Signal Analysis Workbench

> Single-file demo plan. Maps to `info.md` §§28–29 (Manual Demo Checklist + Demo Script for Judges), with defense lines from §30.
> Sources of truth: `AGENTS.md` (commands), `README.md` Known limitations (§“Known limitations”), `docs/portal_alignment.md` honesty block.
> Audience: non-technical judge. No RF hardware, no internet needed on the laptop.

---

## Part 1 — Pre-demo checklist (do this the night before + morning of)

Offline-first. The only script that touches the network is `scripts/fetch_real_data.py` — do **not** run it in the demo.

- [ ] **Install once:** `python -m pip install -r requirements.txt -r requirements-dev.txt`
- [ ] **Generate data FIRST (required):** `python scripts/generate_test_data.py` — seeded (42). Integration tests read `sample_data/bpsk.iq` / `bpsk.wav` and fail without this. Order is `install → generate_test_data → pytest` (`AGENTS.md`).
- [ ] **Fresh-clone expectation:** `python scripts/verify_completed.py` prints `PASS`. With `real_data/` fetched it runs 57 checks; without it, it runs fewer (48 passed + 1 skipped) and **names its skips**. Quote whatever number it prints — do not quote a fixed number (`AGENTS.md`).
- [ ] **`real_data/` skip explanation (say this if asked):** “The skipped group is third-party over-the-air captures downloaded by `fetch_real_data.py`. They need internet, so on an offline laptop the harness skips them instead of failing. Everything else still runs and passes.”
- [ ] **Headless fallback if the GUI will not show:** `$env:QT_QPA_PLATFORM="offscreen"; pytest tests/integration -q` — this is the GUI smoke test path from `AGENTS.md`. If this passes, the app logic is fine even on a machine with no display.
- [ ] **Laptop needs no internet.** Confirm Wi-Fi off: pipeline, GUI, and web dashboard all run on local files only (`127.0.0.1:8765` for the dashboard).
- [ ] **GUI opens in <10 s:** `python scripts/run_app.py` → window titled “RF Signal Analyzer” appears. Close and reopen once to warm the disk cache.
- [ ] **Spot-check one headless run:** `python scripts/run_pipeline.py sample_data/bpsk.iq --sample-rate 100000 --iq-format auto` → exits 0, prints `Modulation: BPSK`, `Correlation: … score=1.0`-ish (must be `> 0.9`).
- [ ] **Bring backups:** PPT with architecture diagram + completion summary; backup demo video; this file printed or open in a second tab (`info.md` §28).
- [ ] **One line the whole team can say:** “We show what we can prove — anything unproven is labelled as a guess, never as a decode.” (`README.md` Known limitations, `docs/portal_alignment.md` honesty block.)

> `.iq` files carry no metadata, so **sample rate must be supplied for `.iq`** (e.g. `--sample-rate 100000`). `.wav` carries its own rate from the file (`core/io.py`).

---

## Part 2 — 3–5 minute timed judge script (say + type exactly this)

Total ~4 min. All commands run from the repo root (`E:\SIH26`). Copy-paste the literal lines.

### 0:00–0:30 — Opening (say)

> “We built an RF Signal Analysis Workbench. It takes a raw radio capture — `.IQ` or `.wav` — shows you what is in it, guesses the modulation, demodulates it, finds the header, and exports a report. Everything you will see runs offline on this laptop.”

### 0:30–1:00 — Step 1: make the test signals (type)

```powershell
python scripts/generate_test_data.py
```

**Say:** “These are our known-truth test signals — the same seed every time, so the demo is reproducible.”
**Expected on screen:** `Synthetic test data generated in sample_data/` plus `tone.iq / bpsk.iq / qpsk.iq / fsk2.iq` and `.wav` versions.

### 1:00–2:00 — Step 2: analyse one capture headlessly (type)

```powershell
python scripts/run_pipeline.py sample_data/bpsk.iq --sample-rate 100000 --iq-format auto
```

**Say:** “No GUI needed — this is the same pipeline the GUI and the web page both call.”
**Expected on screen (clean BPSK must hit these or the pipeline is broken, `AGENTS.md` §22):**

```text
File: bpsk.iq (iq)
Samples: 1032 (... s @ 100000 Hz)
Modulation: BPSK (confidence ~0.8)
Demod: mode=BPSK num_bits=1032
Correlation: sync_word=0x1ACFFC1D header_offset=0 score=1.0
```

Point at two numbers for the judge: **BER < 0.01** (bits match truth — the score of 1.0 is the visible proof) and **correlation score > 0.9** (header found at offset 0). On a coded capture you also get a `--- DECODED PAYLOAD (CRC-16 verified) ---` block — that word “verified” is the whole honesty story (Part 3).

### 2:00–2:45 — Step 3: zero-install web dashboard (type)

```powershell
python scripts/serve_dashboard.py --open
```

**Say:** “No install, no internet — it is one local page on `http://127.0.0.1:8765` with spectrum, waterfall, constellation, and the same report. Good fallback if the desktop GUI misbehaves on stage.”
**Expected on screen:** browser opens, capture list shows `sample_data/` files, clicking Analyse shows the same modulation + header offset as Step 2.

### 2:45–3:30 — Step 4: batch over the whole folder (type)

```powershell
python scripts/batch_analyze.py sample_data --csv output/s.csv --html output/s.html
```

**Say:** “For a folder of captures — one row per file, plus a spreadsheet and a web summary.”
**Expected on screen:** a table (`file | mod | bits | fec | verified | message`), then `N capture(s): K CRC-verified payload(s), 0 load failure(s).`, then `CSV -> output/s.csv` and `HTML -> output/s.html`. Open `output/s.html` in the browser if time allows.

### 3:30–4:00 — Step 5: desktop GUI finish (click)

1. `python scripts/run_app.py`
2. **File > Open Demo Capture** (loads the bundled BPSK capture for you).
3. **Analysis > Run Full Demo (all sample captures)** — or press Run Analysis for the single file.

**Say:** “Same engine, now with time plot, spectrum, waterfall, constellation, eye diagram, bitstream, and Export Report to JSON.”
**Expected on screen:** plots draw, results table fills (one `-- SECTION --` row per report block), log console narrates each step, Export writes `output/report.json`.

### 4:00 — Closing (say)

> “That is the full pipeline: file in, pictures, parameters, bits, header, report out. Blind FEC staying capped, timing/carrier abstaining honestly, and higher-order QAM costing SNR are documented behaviours — the tool says so on screen instead of guessing.”

**If anything crashes on stage:** keep talking, switch to the backup video, and quote the headless result from Step 2 — it proves the engine works even if a plot widget fails.

---

## Part 3 — Defense one-pager (if judges probe; one sentence each + file pointer)

Read the bold sentence aloud. The file pointer is for the technical judge who wants to check.

1. **Blind guesses are capped at 0.5 and never called decodes — only a real CRC-16 pass reports a decode with >0.9 confidence.** → `src/rf_analyzer/core/fec.py`: `score_fec_candidates` (heuristic, `confidence ≤ 0.5`, `crc_pass=None`) vs `decode_hypotheses` (sets `validated=True` only on a CRC-16 pass; noise yields no decode).
2. **Sync detection is length-aware (CFAR) so the expected false alarms stay ≤ 0.01 — a fixed 0.85 threshold alone gives ~5 false hits on a 500k-bit capture.** → `src/rf_analyzer/core/correlator.py`: `min_matches_for_length` / `find_header` (never compare raw `score` to a constant without `min_score`).
3. **EVM judges only the chosen constellation’s fit (`fit_ok`) and is never used to rank modulations, because a denser constellation always fits better (measured 93.5% BPSK → 26.2% 64-QAM on a real GPS capture).** → `src/rf_analyzer/pipeline.py`: `quality.evm_percent / mer_db / fit_ok` (non-finite MER becomes `None` via `json_safe`).
4. **An uncorroborated 2-FSK label keeps its name but loses its confidence (capped to 0.35 with a warning), because the FSK gate is really a narrowband test that also fires for a tone or speech — period recovery is the independent evidence.** → `src/rf_analyzer/pipeline.py`: `modulation.corroborated` + `MODULATION_UNCORROBORATED_CONFIDENCE` (explicit user `2-FSK` request is never downgraded); period recovery in `src/rf_analyzer/core/dsp.py:estimate_fsk_symbol_period` (returns 1 = “no structure” rather than inventing a period).
5. **The burst-length estimate is deliberately biased long (never truncated short) and can only cost a decode, never fake one, because the CRC still decides — the length actually used is reported as `payload.decode_search.frame_bits`.** → `src/rf_analyzer/core/dsp.py:estimate_burst_region` (Otsu threshold on log-envelope, `signal.burst`) + `src/rf_analyzer/pipeline.py`: `BURST_FRAME_SLACK_BITS` slack and shortest-first `_candidate_ends` (protects the deterministic CRC-16 one-extra-byte collision).

**Honesty lines to keep verbatim (from `docs/portal_alignment.md` + `README.md` Known limitations):** no GNU Radio by choice (brief lists “GNU Radio, python, C++” as alternatives; from-scratch NumPy stays testable); `bandwidth_estimate == 0.0` means “not measurable”, not zero; `.iq` sample rate is user-supplied; decode search is bounded (`DECODE_MAX_BITS` / 4 s, overridable); 420-case burst-in-noise sweep produced **0 wrong payloads**.
