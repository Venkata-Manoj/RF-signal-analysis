# Backup demo video — storyboard (≤ 3 min)

> **Goal:** a pre-recorded backup of the SIH live demo that plays if the laptop,
> projector, or network fails on stage. Target **≤ 180 s** including title cards.
>
> **Sources (exact):**
> - Demo beats + narration quotes — `info.md` §29 "Demo Script for Judges"
>   (Steps 1–7, `info.md:2279-2305`; Opening `info.md:2275-2277`, Closing `info.md:2307-2309`).
> - Pre-demo checklist — `info.md` §28 (`info.md:2253-2267`); the three items named in
>   the audit brief are `info.md:2260` (constellation appears), `info.md:2261`
>   (bitstream appears), `info.md:2262` (header correlation works).
> - Presentation readiness — `info.md` §31: tests `info.md:2374-2376`
>   (synthetic BPSK / correlation / CI pass) and presentation items `info.md:2380-2384`
>   (architecture diagram, demo script, backup video, completion summary, limitations).
>   (The brief cited `2374-2378`; the presentation block itself starts at line 2380 —
>   both ranges are covered here.)
> - Completion summary content — `deliverables/slides.html` slide 17, `RF_Signal_Analysis_Workbench.md` §9.

## 0. Pre-flight (do once, before pressing record)

From §28 (`info.md:2253-2267`) — all must be true:

| # | Check | Command / action |
|---|-------|------------------|
| 1 | Laptop runs with **no internet** (dashboard is stdlib-only, no CDN) | Disconnect Wi-Fi, then rehearse once |
| 2 | Sample `.iq` files present | `python scripts/generate_test_data.py` (seeded, deterministic) |
| 3 | Sample `.wav` files present | Same command; `sample_data/bpsk.wav` |
| 4 | GUI opens in < 10 s | `python scripts/run_app.py` (time it) |
| 5 | Large file does not crash | Open `sample_data/coded_fsk.iq` (~1.1 MB) in GUI |
| 6 | Spectrum appears | GUI Spectrum tab / dashboard PSD plot |
| 7 | Waterfall appears | GUI Waterfall tab / dashboard waterfall |
| 8 | **Constellation appears** (`:2260`) | GUI Constellation tab |
| 9 | **Bitstream appears** (`:2261`) | GUI Bitstream tab |
| 10 | **Header correlation works** (`:2262`) | `run_pipeline.py` reports `header_offset=0 score=1.0` on `bpsk.iq` |
| 11 | JSON report exports | `python scripts/run_pipeline.py sample_data/bpsk.iq --sample-rate 100000 --iq-format auto` → `output/` |
| 12 | PPT has architecture diagram | `docs/architecture.svg` inserted (see §4 Export steps below) |
| 13 | PPT has completion summary | slides.html slide 17 |
| 14 | Speaker can state limitations in one breath | "No carrier/timing recovery or equalisation yet; blind scores capped at 0.5; no decode without a CRC-16 pass." |

From §31 tests (`info.md:2374-2376`): `pytest -q` green (or at least the BPSK +
correlation tests), so the narration claim "verified" is true on the recording day.
Re-run `python scripts/verify_completed.py` and show its `PASS` line on camera if time permits.

## 1. Shot list (175 s total — 5 s spare)

Record at 1080p, system audio off, mic on. One continuous take is fine; title cards
are plain fullscreen slides (use `slides.html` slides 1 and 18, or black cards with white text).

| TC | Dur | Beat (§29) | On screen | Narration (say this — quotes are verbatim from §29) |
|----|-----|------------|-----------|------------------------------------------------------|
| 0:00 | 10 s | Opening (`:2275-77`) | Title card: project name + team | "We have built an RF Signal Analysis Workbench that automates the workflow from raw IQ/WAV capture to parameter extraction, demodulation, and header correlation." |
| 0:10 | 15 s | Step 1: Open file (`:2279-81`) | Dashboard in browser: drag `sample_data/bpsk.iq`, type sample rate `100000`, press **Analyze capture** | "We open a captured IQ file. The user can provide sample rate and IQ format." |
| 0:25 | 30 s | Step 2: Visualize (`:2283-85`) | Plots appearing one by one: time-domain → spectrum/PSD → waterfall → constellation → eye diagram (scroll slowly) | "The system shows time-domain, spectrum, waterfall, constellation, and eye diagram." Hold 3 s on the constellation. |
| 0:55 | 20 s | Step 3: Parameters (`:2287-89`) | Report panel: `center_freq`, `bandwidth_estimate`, `snr_db` highlighted | "It estimates center frequency, occupied bandwidth, and SNR." If bandwidth reads `0.0`, say: "zero means not measurable, not zero — the tool says so instead of inventing a number." |
| 1:15 | 20 s | Step 4: Modulation (`:2291-93`) | Report panel: `modulation: BPSK (confidence 0.85)` + alternatives | "It classifies likely modulation and provides a confidence score." |
| 1:35 | 20 s | Step 5: Demodulation (`:2295-97`) | Terminal: `python scripts/run_pipeline.py sample_data/coded_conv.iq --sample-rate 100000` → bits scroll; cut to decoded payload block | "It demodulates the selected signal and extracts the bitstream." |
| 1:55 | 25 s | Step 6: Correlation (`:2299-2301`) | Same terminal output: `sync_word=0x1ACFFC1D header_offset=0 score=1.0`; GUI Bitstream tab with header region shaded | "We provide a known sync word. The system detects the header offset and separates payload." |
| 2:20 | 20 s | Step 7: Report (`:2303-05`) | `output/*.json` opened in editor (valid JSON), plus batch HTML report if available | "Finally, it exports a structured JSON report for further analysis." |
| 2:40 | 15 s | Closing (`:2307-09`) | Completion summary card (static) | "The system demonstrates the complete pipeline: seven modulations, verified FEC/interleaver decoding, and honest reporting of everything it cannot prove." |

**Total: 175 s.** Do not add a blooper reel. If any beat overruns, trim Step 2 (the plots
speak for themselves) — never trim Steps 6–7, they are the proof.

## 2. How to record (Windows, no new software)

1. Close everything except the browser (dashboard at `http://127.0.0.1:8765`) and one terminal.
2. Start the dashboard first: `python scripts/serve_dashboard.py --open`, then disconnect Wi-Fi (proves zero-install claim).
3. Press `Win + Alt + R` (Xbox Game Bar) to record the dashboard window; `Win + Alt + R` to stop.
   Alternative: OBS Studio → Sources → Window Capture → Settings → Output → Recording Format `mp4`, 1080p.
4. Speak the narration column verbatim; keep the mic 15–20 cm away.
5. Save as `deliverables/backup-demo-3min.mp4` (gitignored — copy to the submission USB/portal manually, see §4).
6. Watch once at 1×: every command typed must match §1, every number claimed must be visible on screen.

## 3. If the live demo fails on stage (the point of this file)

| Failure | Fallback line | Action |
|---------|---------------|--------|
| GUI won't open | "The live app needs a display driver this projector lacks — here is the identical pipeline in the browser." | Play TC 0:10–1:35 (dashboard beats) |
| Dashboard port blocked | "Same function, no server — straight to the CLI." | Play TC 1:35–2:40 (terminal beats) |
| Nothing runs | "The recording is the demo — same seeded files, same outputs." | Play the full 175 s from 0:00 |

## 4. Submission export steps (do these by hand — no binary is generated by the repo)

> `deliverables/` is **gitignored** (see `.gitignore`: "presentation material only, not
> source"), so nothing here reaches the portal via git. The canonical, reviewable assets
> live in `docs/` (committed); the portal-ready binaries are produced once, by hand,
> with the exact steps below. `python-pptx` 1.0.2 is installed in `.venv`, but no PPTX
> is generated in-repo: a deck hand-built from `slides.html` (18 slides, live canvases,
> speaker notes in §29) beats any auto-converted dump, and the conversion is a
> one-time manual task, not a build step.

1. **Slides PDF** — open `deliverables/slides.html` in Chrome/Edge → `Ctrl+P` →
   Destination "Save as PDF", Layout **Landscape**, Background graphics **on** →
   save as `deliverables/slides.pdf`. Verify: 18 pages, slide 5 shows the 10-stage
   flow, slide 17 the completion summary.
2. **Slides PPTX** — in PowerPoint: New blank 16:9 deck → 18 slides, copy
   headings + cards from `slides.html` (text only; skip the two `<canvas>` live demos —
   replace with screenshots of the dashboard). Insert `docs/architecture.svg` on the
   slide matching deck slide 5 (Insert → Pictures). Save as `deliverables/slides.pptx`.
   (Scripted alternative exists — `python-pptx` is available — but hand layout is required
   for a judged deck; do not auto-generate.)
3. **Architecture image** — `docs/architecture.svg` is the portable figure (also for the PPT).
   PNG if the portal demands raster: open the SVG in Chrome → screenshot at 200%, or
   paste `docs/architecture.mmd` into https://mermaid.live → Export PNG.
4. **Backup video** — the recording from §2, saved as `deliverables/backup-demo-3min.mp4`,
   ≤ 3 min, 1080p. Upload to the portal video slot AND keep a copy on the demo laptop
   + a USB stick (three copies, two media).
5. **Final manual check** — §31 presentation block (`info.md:2380-2384`): architecture
   diagram in deck ☐ · demo script printed ☐ · backup video present ☐ · completion
   slide ☐ · limitations one-liner rehearsed ☐.

## 5. Relation to the live-demo plan + what this file does NOT cover

- `docs/demo-day.md` is the **live** demo plan (timed judge script + §30 defense lines).
  This file is the **backup recording** plan (shot list, recording how-to, submission
  exports). Rehearse both; if live fails, play the recording per §3 above.

- Live-demo Q&A prep (see `RF_Signal_Analysis_Workbench.md` §4 honesty contracts —
  every contract is a likely judge question with its answer attached).
- Benchmark numbers narration (use §8: "2.8 s median, 1.8–4.1 s range" — quote the
  range, never a single number).
