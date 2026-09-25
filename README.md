# RF Signal Analysis Workbench

**Point it at a captured radio signal file and it tells you what the signal is, what
it says, and how confident it is about both.**

This is our completed system for **SIH 2026, problem statement SIH26147 / NTRO** — *"Automated model
for analysis of .IQ and .wav files along with signal parameter extraction."*

You do not need to know anything about radio to use it. If you have a `.iq` or `.wav`
file and want to know what is inside it, start at [Quick start](#quick-start).

---

## Contents

- [What it does](#what-it-does)
- [Quick start](#quick-start)
- [Three ways to use it](#three-ways-to-use-it)
  - [1. Desktop app (the GUI)](#1-desktop-app-the-gui)
  - [2. Web dashboard (no-install alternative)](#2-web-dashboard-no-install-alternative)
  - [3. Command line](#3-command-line)
- [Understanding the output](#understanding-the-output)
- [What "verified" means (and what it does not)](#what-verified-means-and-what-it-does-not)
- [Supported files and schemes](#supported-files-and-schemes)
- [Analysing many files at once](#analysing-many-files-at-once)
- [Testing with real radio recordings](#testing-with-real-radio-recordings)
- [Proving it works](#proving-it-works)
- [Performance](#performance)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)
- [How this maps to the problem statement](#how-this-maps-to-the-problem-statement)
<!-- - [Known limitations](#known-limitations) -->

---

## What it does

Radio signals are recorded as **IQ files**: two streams of numbers (the "in-phase" and
"quadrature" parts of the waveform) that together describe a radio signal exactly as it
arrived at the antenna. They look like noise in a normal audio player. This tool reads
them and works out:

| Question | What you get |
| --- | --- |
| What kind of signal is this? | Modulation type (BPSK / QPSK / 8PSK / 16-QAM / 64-QAM / 2-FSK / 4-FSK) plus a confidence |
| How fast is it? | Sample rate, bandwidth, symbol rate, carrier offset |
| How clean is it? | SNR in dB, plus EVM/MER from the constellation |
| Where does the message start? | Sync-word ("header") position in the bit stream |
| Is there error-correction coding? | FEC scheme + interleaver, **only when proven by a CRC check** |
| What does it actually say? | The decoded message as text, hex and ASCII |

It also draws the standard RF engineer's plots — constellation, spectrum, waterfall,
eye diagram — so you can see the signal rather than just read numbers about it.

---

## Quick start

Requires **Python 3.11+**. On Windows use PowerShell or Command Prompt.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Optional — only needed to run the test suite and the verification harness
python -m pip install -r requirements-dev.txt

# Create the sample signals used by the demos and tests
python scripts/generate_test_data.py

# Launch the desktop GUI — this is the "GUI based model" the brief asks for
python scripts/run_app.py
```

The window opens with eight tabs — **Time, Spectrum, Waterfall, Constellation, Eye,
Bitstream, Payload, Hypotheses**. Use **Open Demo Capture** for a bundled signal, or
**File → Open** for your own `.iq`/`.wav`. Set the sample rate if you are using a raw `.iq`
file (see [Supported files](#supported-files-and-schemes)), then press **Run analysis**.

The right-hand table is the full report, grouped by block: input, signal, burst detection,
modulation, EVM/MER quality, demodulation, correlation, payload, FEC, interleaving and
diagnostics. The **Hypotheses** tab holds what the results table can only summarise — the
ranked FEC, interleaver and modulation-fit candidates the classifier chose between.

Don't want to open a window? `python scripts/serve_dashboard.py --open` gives you the same
analysis in a browser instead. It is an extra, not a substitute for the desktop app — both
call the identical analysis code.

On macOS or Linux the only difference is the activation step:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt   # optional: tests + verification only
python scripts/generate_test_data.py
python scripts/run_app.py                       # launch the desktop GUI
```

> **Nothing else is installed.** The dashboard is served by Python's own standard
> library — no Flask, no npm, no CDN, and it makes no network requests. Every plot is
> drawn by your browser.

---

## Three ways to use it

All three call exactly the same analysis code (`pipeline.analyze_file`), so they can
never disagree with each other.

### 1. Desktop app (the GUI)

The brief asks for a **GUI-based model**, and this is it: a full PyQt6 workbench with eight
tabs — Time, Spectrum, Waterfall, Constellation, Eye, Bitstream, Payload, Hypotheses — plus a **Payload** tab showing the decoded message and a raw hex/ASCII dump.

```powershell
python scripts/run_app.py
```

Useful menu items: **Open Demo Capture**, **Batch Folder Analysis**, **Run Full Demo**.

### 2. Web dashboard (no-install alternative)

The same analysis with no window at all, served by Python's own standard library. It is an
extra beyond the brief, and useful when you cannot run a Qt app — a locked-down machine, or a
demo over a shared screen.

```powershell
python scripts/serve_dashboard.py            # then open http://127.0.0.1:8765
python scripts/serve_dashboard.py --open     # opens the browser for you
python scripts/serve_dashboard.py --port 9000
```

Drag an `.iq` or `.wav` file onto the page, or pick one of the bundled captures. Set the
sample rate if you are using a raw `.iq` file (see [Supported files](#supported-files-and-schemes)),
then press **Analyze capture**. **Analyze every bundled capture** compares them all in one
table.

You can share a link to a specific analysis — `http://127.0.0.1:8765/?name=bpsk.iq&auto=1`
opens that capture and analyses it immediately.

The server listens on `127.0.0.1` only, so it is not reachable from other machines.

### 3. Command line

Best for scripting and for CI. Prints a readable summary and saves a JSON report.

```powershell
python scripts/run_pipeline.py sample_data/bpsk.iq --sample-rate 100000 --iq-format auto
python scripts/run_pipeline.py real_data/gps_l1_4mhz_cf32.iq --sample-rate 4000000 --iq-format auto
```

Example of a successful decode:

```
Modulation: BPSK (confidence 0.85)
Constellation: EVM 0.00%, MER unbounded (1168 symbols, fit ok)
Correlation: sync_word=0x1ACFFC1D (assumed) header_offset=0 score=1.0
FEC: Convolutional r=1/2 K=7 (validated=True)
Interleaving: convolutional (validated=True)

--- DECODED PAYLOAD (CRC-16 verified) ---
scheme: Convolutional r=1/2 K=7  interleaver: convolutional
68 bytes
SIH26147 convolutional frame: K=7 r=1/2 Viterbi, Forney interleaver.
00000000  53 49 48 32 36 31 34 37 20 63 6f 6e 76 6f 6c 75  |SIH26147 convolu|
...
```

The script exits `1` if the report contains errors, so it is safe to use in a pipeline.

---

## Understanding the output

The names below are jargon, but the ideas are simple.

**Signal measurements**

| Field | Plain meaning |
| --- | --- |
| `sample_rate` | How many IQ samples per second the file was recorded at. For `.iq` files **you must supply this** — a raw file has no header saying so. |
| `bandwidth_estimate` | Roughly how much spectrum the signal occupies. |
| `snr_db` | Signal-to-noise ratio. Higher is cleaner. |
| `symbol_rate_estimate` | How many symbols per second the signal carries. |
| `burst` | Where the signal actually starts and ends inside the file, measured from the power envelope — a real recording is a burst surrounded by noise. `found: false` means the capture is continuous (or is all burst), so there is no burst to measure. This is also what lets a frame followed by noise still decode. |
| `cfo_estimate_hz` | Carrier frequency offset — how far the signal drifted from where it should be. |
| `center_frequency_estimate` | Where in the spectrum the strongest energy sits. |

**Quality measurements**

| Field | Plain meaning |
| --- | --- |
| `evm_percent` | **Error vector magnitude** — how far the received symbols sit from where they should be. Small is good: a clean signal is a few percent. |
| `mer_db` | **Modulation error ratio** — the same thing expressed in dB. Shown as `unbounded` when the signal is mathematically perfect. |
| `fit_ok` | Whether the EVM is low enough that the guessed modulation actually explains the samples. |

**Plots**

- **Constellation** — every received symbol as a dot. Clean signals cluster into tight
  groups; the green circles are where the groups *should* be.
- **Power spectral density** — how much energy is at each frequency.
- **Waterfall** — the same spectrum over time. Colour is energy; bright bands are signal.
- **Eye diagram** — overlaid symbol transitions, the classic "how open is the eye" view.
- **Bit stream ribbon** — the demodulated bits, with the header and payload regions shaded.

**Payload** — if a CRC check passes you get the decoded message as text and as a hex
dump. If it does not, you get the raw bits **clearly labelled as still-coded**, never as
a message.

**The `(uncorroborated)` marker** — sometimes the tool guesses the modulation is 2-FSK
but cannot independently confirm it. When that happens it keeps the label, but writes
`(uncorroborated)` beside the confidence so the number is not mistaken for a normal
estimate. Read it as "we are not sure about this one". This mostly happens on signals
that carry no data at all, such as an unmodulated tone or a piece of audio — see
[Known limitations](#known-limitations).

---

## What "verified" means (and what it does not)

This is the most important section in this document.

Working out whether an unknown signal uses error-correcting codes is genuinely hard.
It is very easy to write a tool that *guesses* and sounds confident. We deliberately did
not.

So the report always separates two different things:

| | Meaning |
| --- | --- |
| **Blind guess** (`FEC (blind guess)`, `Interleaver (blind guess)`) | A heuristic score from the bit statistics. Confidence is **capped at 0.5** because it is a guess. |
| **Verified** (`validated: true`) | We actually de-interleaved, decoded and got a **CRC-16 checksum to pass**. Only then do we call it verified, and only then do we show you a message. |

The practical consequence: **if the tool shows you a decoded message, that message is
real.** A CRC-16 is a 1-in-65,536 check on top of the decode, so a false positive is
very unlikely. If the tool cannot prove a decode, it says so and shows you the raw bits
instead.

You will also see honest warnings rather than silent bad numbers:

- `Occupied-bandwidth estimate unavailable: ... reported as 0 Hz meaning 'not measurable', not as measurements.`
- `Modulation fit is poor: the 16-QAM constellation leaves 39.2% error-vector magnitude, so the modulation estimate is unreliable.`
- `No sync_word was supplied and the default 0x1ACFFC1D did not match significantly ... Treating the whole bit stream as payload.`

We would rather tell you we are unsure than give you a confident wrong answer.

---

## Supported files and schemes

**Files**

| Format | Notes |
| --- | --- |
| `.iq` (raw IQ) | `complex64` (also called cf32), `int16` (ci16), `uint8` (cu8), `int8` (ci8, HackRF). Little-endian by default. **A sample rate is required** — raw files carry no metadata. |
| `.wav` | Stereo is read as left = I, right = Q; mono is treated as a real signal. The sample rate comes from the file. |

Set **IQ format** to `auto-detect` and the tool infers the storage type from the file's
own statistics — it works out whether the numbers are floats, 16-bit integers, etc.
It tells you which format it used, and warns you if the guess was not confident.

**Modulation:** BPSK, QPSK, 8PSK, 16-QAM, 64-QAM, 2-FSK, 4-FSK (or `auto`). Demodulation prefers the coherent receive chain (`core/receiver.py`: matched filter, Gardner / Mueller-Muller timing, Costas carrier, CMA/LMS equalisation) with honest fallback to the phase-aligned slicers when the receiver abstains.

**Receiver:** `auto` (the default — robust chain with honest naive fallback) or `naive` (legacy slicers, bit-for-bit the old behaviour). The GUI has a **Receiver** dropdown next to Modulation; the CLI takes `--receiver auto|naive`. The report records the outcome under `demodulation.receiver` (`path`, `requested`, `locked`, `reason`, `sps`), so you can always tell which path produced the bits.

**About 2-FSK:** it is the one modulation where a bit is *not* one sample. A real 2-FSK
burst spends many samples on each tone (the included sample spends 100), so the tool
first recovers the symbol period from the signal itself, then decimates to one bit per
symbol. It reports the recovered period and derives the symbol rate from it. If a period
cannot be recovered — the signal is not constant-envelope FSK, or it is shorter than a
few symbols — the tool says so and falls back to the documented one-bit-per-sample
behaviour rather than guessing a period.

**Forward error correction** (verified, decoded from scratch in pure NumPy):

| Scheme | Detail |
| --- | --- |
| CRC-16 only | Frame integrity with no coding |
| Convolutional | K=5–9, rate 1/2 and 1/3 with hard/soft-decision Viterbi (historical K=7, rate 1/2 G1=0o171, G2=0o133 preserved) |
| Reed–Solomon | RS(255,223), RS(255,239) and shortened/truncated variants over GF(256) |
| LDPC | Parametric systematic family with min-sum belief propagation (original (3,4)-regular as default member) |
| Concatenated | CRC-16 → Reed–Solomon → convolutional (multiple nsym/code variants) |

**Interleaving:** block, convolutional (Forney), diagonal, pseudo-random, and
pseudo-random over sub-blocks.

**Why the interleaver matters.** Interleaving is not decoration. LDPC correction is
*probabilistic*: at a fixed error count the outcome depends on how the errors happen to
land across the 84-bit code blocks, so it is not reliably monotone. Measured over five
seeds, the block interleaver decodes 8 errors *less* often than 10 (3 times out of 5,
against 5 out of 5).

The interleaver's real benefit shows against **bursts**, which is what it exists for. With
a contiguous burst of 8 bits, no interleaver recovers the message at all (0 times out of
5), while the pseudo-random sub-block interleaver still recovers it 3 times out of 5 — and
without an interleaver a burst of 8 is already fatal. So it roughly doubles the burst
length tolerated. It does *not* help against uniformly scattered errors, because those are
spread across blocks already. Every FEC × interleaver combination the tool offers is
identified and decodes back to the exact message, at error loads inside each scheme's
measured tolerance.

**Sync words:** any hex string, e.g. `0x1ACFFC1D`. Leave it blank and the tool probes
with the default — and tells you when it had to assume.

---

## Analysing many files at once

```powershell
python scripts/batch_analyze.py sample_data --sample-rate 100000 --csv output/summary.csv --html output/summary.html
```

Produces a CSV and a self-contained HTML report (one file, no internet needed) with one
row per capture. The GUI has **Batch Folder Analysis**, and the dashboard has **Analyze
every bundled capture**.

---

## Testing with real radio recordings

The bundled sample signals are generated by this project, so on their own they only
prove the tool agrees with itself. To test against genuine third-party recordings:

```powershell
python scripts/fetch_real_data.py          # downloads 2 real OTA captures (~4 MB)
python scripts/fetch_real_data.py --list   # just show the curated sources
```

This fetches real SDR recordings published by the [PySDR](https://pysdr.org) project —
a GPS L1 satellite downlink and an analog NTSC television transmission — and verifies
each against a pinned SHA-256 so a corrupted download is reported rather than analysed.
Provenance is written to `real_data/SOURCES.json`.

```powershell
python scripts/run_pipeline.py real_data/gps_l1_4mhz_cf32.iq --sample-rate 4000000 --iq-format auto --gps
python scripts/run_pipeline.py real_data/ntsc_10mhz_cf32.iq --sample-rate 10000000 --iq-format auto
```

These are deliberately *hard* cases, and the tool's honesty is the point: neither carries
our sync word, so it correctly reports **no header and no decoded message**. Beyond that:
- With `--gps`, the GPS capture yields **acquisition evidence, not a message**: 7 satellites
  found with Doppler, code phase and C/N0 in the `gps` report block (a 10 ms capture cannot
  yield navigation subframes, so none is claimed). The GUI has a **GPS L1** checkbox for the same.
- The NTSC capture yields an **analog-video hint**: line sync detected at ~15740 Hz in the
  `video` report block with an explicit warning, while the modulation label stays capped
  and no payload is invented.

`scripts/fetch_real_data.py --list` also lists larger public sources (SDRangel
satellite/ADS-B/AIS recordings, the Mendeley 2.4 GHz dataset, DeepSig RadioML, IQEngine)
if you want more material for a live demo.

---

## Proving it works

```powershell
python scripts/verify_completed.py
```

*(Needs the optional `requirements-dev.txt` install from
[Quick start](#quick-start) — it runs the test suite. If you skipped it, the harness
says so and exits `2` rather than failing obscurely.)*

This regenerates the test data, runs the whole test suite, and then performs **57
individual checks** end to end — it prints that count itself, so the number here
cannot drift away from what actually ran. It prints `PASS` and exits `0` only if
nothing fails; a check whose inputs are missing is printed as `[SKIP]` and left out
of the passed count rather than folded into it.

**The total depends on one thing: the third-party recordings.** `real_data/` is not
committed (they are large binaries — see
[Testing with real radio recordings](#testing-with-real-radio-recordings)), so a fresh
clone reports **48 passed, 1 skipped** — that group is skipped rather than failed —
against the full **57** once you run `python scripts/fetch_real_data.py`. The harness
prints whichever it actually ran, and names any skips so "N passed" never quietly
includes a group that never ran.

It covers, among others:

- the core pipeline on a clean BPSK capture (BER < 0.01, correlation > 0.9)
- all six coded captures decoding back to the **exact transmitted message**, with the
  transmitter's FEC scheme and interleaver identified from the bit stream alone (the test
  suite sweeps all 30 FEC × interleaver combinations, so the search cannot hide behind
  the six captures we happen to ship)
- **2-FSK symbol-period recovery** (the one modulation that is not one sample per bit)
- **no payload invented** from random bits, and blind FEC scores staying capped
- IQ-format auto-detection
- batch CSV/HTML export
- the web dashboard (serves its page, decodes a capture, returns every plot, and its
  HTTP responses are valid strict JSON)
- real third-party captures analysed honestly
- the report schema and that on-disk reports are valid strict JSON

Other useful commands:

```powershell
pytest                                        # the full suite (as of 2026-09-25: 864 collected; quote what `pytest` prints)
pytest tests/unit/test_fec_codecs.py -q       # one module
python scripts/benchmark_snr.py               # BER vs SNR sweep
python scripts/measure_fec_capability.py      # re-derive the FEC tolerance numbers below
python scripts/benchmark_performance.py       # re-derive the timings in "Performance"
$env:QT_QPA_PLATFORM="offscreen"; pytest tests/integration -q   # GUI tests, headless
```

`pytest` reports **864 collected** (as of 2026-09-25; quote what `pytest` prints for passed/skipped). The skip is deliberate and it says so itself:
one parametrisation of "the raw payload must never masquerade as the decoded message" is
the no-coding case, where the raw payload genuinely *is* the message, so the test's premise
does not apply. Run `pytest -rs` to see skips named rather than counted.

---

## Performance

The stated requirement is **1 million samples analysed in under 10 seconds**. Measured
with `scripts/benchmark_performance.py`, which builds realistic captures (a framed burst
followed by a noise tail) and times the whole funnel — load, PSD, waterfall,
demodulation and the decode search:

| Capture | File size | Typical | Observed range |
| --- | --- | --- | --- |
| 1 million samples | 8 MB | 2.8 s | 1.8 – 4.1 s |
| 2 million samples | 16 MB | 6.5 s | 4.1 – 12.1 s |
| 12.5 million samples | 100 MB | 30 s | 29 – 36 s quiet, 499 s under load |

The budget is met with roughly **3–4× headroom**, and cost grows close to linearly with
the sample count, so about **3 seconds per million samples** is a fair rule of thumb.

**The larger rows are indicative, not guaranteed.** These are wall-clock timings on a
shared machine and they move with whatever else is running: the 100 MB case measured
32 s and 31 s back to back, then 36 s and 499 s during a longer batch, with no change in
the work at all. Only the 1 M row is stable enough to carry a budget. Run the script on
your own hardware if you need your own numbers — it prints the range, not just a median,
for exactly this reason.

A capture is read and analysed whole, so a file needs room for it: **8 bytes per sample**
for the samples themselves, plus working room for the FFTs. There is no streaming mode.

---

## Project layout

```
src/rf_analyzer/
  pipeline.py          analyze_file(request) -> report   <- the single source of truth
  config.py            shared constants and thresholds
  batch.py             folder-wide analysis + CSV/HTML export
  dashboard.py         builds the JSON that the web page draws
  web/dashboard.html   the self-contained browser UI
  core/
    io.py              .iq / .wav loading, IQ-format auto-detection
    dsp.py             spectrum, waterfall, bandwidth/SNR/CFO, EVM/MER, burst region
    rate_est.py        fused symbol/sampling-rate estimation (BW + |x|^2 + cyclostationary + FSK)
    classifier.py      modulation classification (HOC + rules + ML head)
    demod.py           naive phase-aligned slicers (the honest fallback)
    receiver.py        coherent chain: matched filter, Gardner/Mueller-Muller, Costas, CMA/LMS
    waveform.py        seeded synthetic waveform generator (tests and training)
    channel.py         real-channel impairments (CFO/Doppler, multipath, IQ imbalance, AWGN)
    gps.py             GPS L1 C/A acquisition + despread (evidence only, opt-in)
    video.py           analog-video line-sync hint detector
    judgments.py       typed Choice/Noul/Score judgments with confidence-gated routing
    correlator.py      sync-word correlation (with false-alarm control)
    fec.py             CRC-16/32, Viterbi, Reed-Solomon, LDPC, concatenated
    deinterleave.py    block / convolutional / diagonal / pseudo-random
    framing.py         transmit-side frame builder (mirror of the receive chain)
    payload.py         payload slicing, classification, hexdump
    report.py          report serialisation (strict JSON)
  gui/main_window.py   the PyQt6 desktop app
scripts/               run_app, run_pipeline, serve_dashboard, batch_analyze,
                       generate_test_data, fetch_real_data, verify_completed, benchmark_snr
tests/                 unit + integration, incl. headless GUI (as of 2026-09-25: 864 collected; quote what `pytest` prints)
sample_data/           generated synthetic captures (gitignored)
real_data/             downloaded real captures (gitignored)
output/                reports, bitstreams and decoded payloads (gitignored)
info.md                the full PRD, build plan and verification matrix
```

**Architecture rule:** `pipeline.analyze_file()` is the only place the analysis happens.
The GUI, the web dashboard and the CLI all call it. That is why they can never disagree.

---

## Troubleshooting

**"sample_rate is required for .iq files"**
Raw IQ files have no header, so nothing in the file says how fast it was recorded. Look
at the filename (e.g. `gps_l1_4mhz_cf32.iq` → 4,000,000), your SDR software's recording
settings, or the source you downloaded it from. WAV files carry their own rate, so you
never need to supply one for those.

**The constellation looks like a fuzzy circle instead of tight dots**
Usually a carrier frequency offset or the wrong modulation — check the `cfo_estimate_hz` value and the `demodulation.receiver` path/lock status, and try setting **Modulation** explicitly
instead of `auto-detect`. The tool will warn you when the constellation fit is poor.

**"No CRC-valid FEC/interleaver hypothesis matched"**
Not necessarily a problem. It means the tool could not *prove* a decode. For a real-world
recording that was never CRC-framed (analog video, unframed telemetry), this is the
correct and expected answer. If you know the frame length, pass `--frame-bits`.

**The dashboard says "server unreachable"**
The server is not running. Start it with `python scripts/serve_dashboard.py`.

**Nothing appears in `sample_data/`**
Run `python scripts/generate_test_data.py` first. The integration tests need it.

**`verify_completed.py` says "pytest is not installed"**
The test runner is an optional install, deliberately kept out of the runtime
requirements so you do not need it just to *use* the tool. Add it with
`python -m pip install -r requirements-dev.txt`. The harness detects this case and
exits `2` (rather than `1`, which means a check actually failed) so a script can tell
the two apart.

**`python3` not found on Windows**
Use `python`, not `python3`.

---

## How this maps to the problem statement

| Requirement | Where it is handled |
| --- | --- |
| Analyse `.IQ` and `.wav` files | `core/io.py` — raw IQ in four dtypes plus WAV, with auto-detection |
| Extract signal parameters (sampling frequency, modulation, FEC, interleaving) | `core/dsp.py`, `core/classifier.py`, `core/fec.py`, `core/deinterleave.py` |
| Demodulate FSK, PSK, QAM | **COMPLETED** — `core/receiver.py` + `core/demod.py` — BPSK, QPSK, 8PSK, 16-QAM, 64-QAM, 2-FSK, 4-FSK. Coherent chain (Costas / Gardner / Mueller-Muller / CMA / LMS) is the default with honest naive fallback; 2-FSK/4-FSK recover their symbol period, so a real burst with many samples per bit is decodable end to end |
| De-interleave: Block, Convolution, Diagonal, Pseudo-Random | `core/deinterleave.py` — all four, plus a sub-block pseudo-random variant |
| FEC: convolutional + Viterbi, Reed–Solomon, concatenated, LDPC | **COMPLETED** — `core/fec.py` — convolutional family K=5–9 r=1/2,1/3 (hard + soft Viterbi), RS family with shortened variants, parametric LDPC family, concatenated variants; all implemented from scratch and CRC-verified |
| Bit stream correlation | `core/correlator.py` — with false-alarm control so long captures cannot fake a header |
| GUI-based model | `gui/main_window.py` + `scripts/run_app.py` — a PyQt6 desktop app with eight tabs. The browser dashboard is an extra, not the GUI the brief asks for |
| Improved feature visibility — constellation, waterfall | Both UIs, plus spectrum, eye diagram and a bit-stream ribbon |
| Automated analysis | `pipeline.analyze_file()`, `batch.py`, `scripts/batch_analyze.py` |
| Error correction | `core/fec.py`, verified by CRC-16 |
| Header/payload identification | `core/correlator.py` + `core/payload.py` |
| Toolchain — "advanced models such as GNU Radio, python, C++" | Python 3.11 + NumPy/SciPy, one of the alternatives the brief names. The DSP, FEC codecs and interleavers are implemented from scratch rather than assembled from library blocks, so each one is unit-tested against known-answer vectors and gated on a CRC-16 pass |

**Extra features beyond the brief:** zero-install web dashboard; IQ-format auto-detection;
one-click demo and full-demo runs; batch folder analysis with CSV and self-contained HTML
export; sampling-rate hypothesis ranking; EVM/MER constellation quality with a
modulation-fit cross-check; payload extraction with text/hex viewer; shareable deep links;
constant-false-alarm-rate sync detection; a reproducible real-data fetch with checksum
verification.

---
<!-- 
## Known limitations

We would rather list these than have you discover them:

- **Signal conditioning is built in.** Centre frequency is the PSD peak, bandwidth
  uses a median-noise-floor rule with fused sampling/symbol-rate estimation, and the coherent receive chain provides matched filtering, Gardner / Mueller-Muller timing recovery, Costas carrier recovery and CMA/LMS equalisation with honest fallback when loops do not lock. A rotated or frequency-offset signal is corrected when lock is achieved and otherwise reported with its EVM and a warning.
- **2-FSK symbol-period recovery needs a few samples per symbol.** Below about five
  samples per symbol the piecewise-constant model cannot be told apart from noise, so the
  period is reported as unrecovered and the naive one-bit-per-sample path is used. The
  tool never invents a period it cannot reconstruct.
- **The bandwidth estimator cannot measure flat-spectrum signals** such as QPSK, where a
  real signal's peak-above-noise-floor is no larger than pure noise's. Rather than lower
  the threshold (which would start classifying noise as signal), the tool reports the
  estimate as unavailable and says so.
- **Modulation classification is a heuristic**, and it gets real-world signals wrong —
  the GPS L1 capture comes back as low-confidence BPSK with an explicit warning rather
  than a real answer (spread-spectrum looks like noise to a constellation classifier).
  The tool flags the poor fit rather than hiding it. EVM is *not* used to pick a
  modulation, because it always favours denser constellations. For what the GPS
  capture actually contains, use `--gps` (or the GUI checkbox): the acquisition
  search reports the satellites it can prove, with Doppler, code phase and C/N0.
- **The modulation label is corroborated by the header, not by the classifier alone.** The
  classifier works from whole-capture statistics, which are not invariant to how much noise
  surrounds a burst: the same BPSK burst is labelled QPSK or 8PSK once enough noise is added
  either side, at 0.7–0.85 confidence. That is a real problem, because a wrong label hides
  the sync word and the report then says "no frame found" — pointing at a framing bug that
  is really a classification one. So the tool no longer trusts the classifier's answer on
  its own: when the chosen demodulation sees a hint of the sync word but cannot confirm it,
  the other demodulations are tried and the one that actually correlates the header wins,
  with `modulation.revised_from` recording what the classifier originally said and a warning
  explaining the revision. Two caveats remain, and they are honest ones: the arbitration
  needs a sync word (with none supplied, nothing is revised), and a wrong label that happens
  to correlate is still reported as right. EVM cannot help here — a sparser constellation
  always fits a denser one's points, so EVM cannot separate BPSK from QPSK. Pinned by
  `tests/integration/test_burst_in_noise.py`.
- **A signal with no data on it is still given a label.** Measured on the bundled
  samples, the FSK test (low instantaneous-frequency variance) fires for an unmodulated
  tone, for narrowband audio and for speech, as well as for genuine 2-FSK. Those cases
  are not accepted quietly: the tool tries to recover a symbol period, fails, and both
  warns and **caps the reported confidence**, so the label is shown as uncorroborated
  instead of as a confident estimate. It never claims a payload from them. Telling
  "unmodulated carrier" apart from "data signal" properly needs its own class, which is
  a scoped follow-up rather than something to bolt onto the existing heuristic — and
  swapping in a different discriminator without validating it would just replace one
  wrong label with another.
- **Blind FEC detection is a search, not a detector.** Nothing is reported as decoded
  without a CRC-16 pass, so an unusual or unlisted scheme will simply not be found.
- **An uncoded frame corrects nothing, by design.** With no FEC, a single flipped bit
  breaks the CRC-16 and the tool reports no decode rather than guessing. This is the
  control that makes the error-correction claims meaningful.
- **The decode search examines a bounded prefix** (16,384 bits, 4 s by default) to respect
  the performance budget. Raise `decode_max_bits` / `decode_time_budget_s` for long frames.
- **The burst measurement needs a noise gap, and declines without one.** The start and end of
  the burst are found from the power envelope, so a continuous recording — or a capture that
  is all burst, like the bundled bare-frame samples — reports `found: false` and no burst is
  claimed. That is deliberate, because the estimate only ever *narrows* the decode search:
  the CRC-16 still decides, so a wrong estimate can cost a decode but can never invent one.
  Measured across bursts occupying 0.4%–69% of their capture, the estimated edge lands within
  a couple of envelope blocks of the true edge — in *either* direction, because a boundary
  block that is mostly noise can miss the threshold and stop the estimate short. Both
  directions are absorbed downstream: the search is biased long by `BURST_FRAME_SLACK_BITS`
  and the CRC anchor tolerates excess. Across a 420-case sweep (4 modulations × 5 noise
  levels × 7 paddings × 3 seeds) every capture that decoded returned the exact message, and
  **no case ever produced a wrong payload**. Pinned by
  `tests/integration/test_burst_in_noise.py` and `tests/unit/test_dsp.py`.
- **No GNU Radio, and that is a choice rather than a gap.** The problem statement offers
  "advanced models such as GNU Radio, python, C++", and Python is one of the alternatives it
  names, so the requirement is met by picking it. The DSP, FEC codecs and interleavers here are
  written from scratch in NumPy instead of being assembled from library blocks — which is what
  makes the behaviour checkable to the level the rest of this section describes. A library block
  is convenient, but you cannot point a test at the inside of it. GNU Radio's real advantage is
  driving live radio hardware, and there is none here: everything runs on files. -->

---

<!-- ## Completion status (completed, nearly finished)

All `description.txt` (SIH26147) requirements are implemented and tested: fused rate estimation, automation (IQ auto-detect,
auto sample rate/sync/frame bits, one-click Auto-Analyze in GUI/dashboard/CLI),
typed judgments, seven-modulation coherent demodulation with GUI/CLI receiver control,
ML classifier head, FEC families, channel model and
OTA harness (GPS acquisition evidence, analog-video hint). Tracked in `docs/acceptance_status.md` (R1–R7 with
evidence). The system is complete against the problem statement; `python scripts/verify_completed.py` must still print PASS as the final gate. -->

## License and attribution

Third-party test recordings downloaded by `scripts/fetch_real_data.py` are published by
the [PySDR project](https://pysdr.org) and are used here as test material only. See
`real_data/SOURCES.json` for per-file provenance.

See `info.md` for the full PRD, schema definitions, build plan and verification matrix.
