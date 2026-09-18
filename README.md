# RF Signal Analysis Workbench

**Point it at a captured radio signal file and it tells you what the signal is, what
it says, and how confident it is about both.**

This is our MVP for **SIH 2026, problem statement SIH26147 / NTRO** — *"Automated model
for analysis of .IQ and .wav files along with signal parameter extraction."*

You do not need to know anything about radio to use it. If you have a `.iq` or `.wav`
file and want to know what is inside it, start at [Quick start](#quick-start).

---

## Contents

- [What it does](#what-it-does)
- [Quick start](#quick-start)
- [Three ways to use it](#three-ways-to-use-it)
  - [1. Web dashboard (easiest)](#1-web-dashboard-easiest)
  - [2. Desktop app](#2-desktop-app)
  - [3. Command line](#3-command-line)
- [Understanding the output](#understanding-the-output)
- [What "verified" means (and what it does not)](#what-verified-means-and-what-it-does-not)
- [Supported files and schemes](#supported-files-and-schemes)
- [Analysing many files at once](#analysing-many-files-at-once)
- [Testing with real radio recordings](#testing-with-real-radio-recordings)
- [Proving it works](#proving-it-works)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)
- [How this maps to the problem statement](#how-this-maps-to-the-problem-statement)
- [Known limitations](#known-limitations)

---

## What it does

Radio signals are recorded as **IQ files**: two streams of numbers (the "in-phase" and
"quadrature" parts of the waveform) that together describe a radio signal exactly as it
arrived at the antenna. They look like noise in a normal audio player. This tool reads
them and works out:

| Question | What you get |
| --- | --- |
| What kind of signal is this? | Modulation type (BPSK / QPSK / 2-FSK / 16-QAM) plus a confidence |
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

# Create the sample signals used by the demos and tests
python scripts/generate_test_data.py

# Easiest way in: a browser dashboard on your own machine
python scripts/serve_dashboard.py --open
```

Your browser opens at `http://127.0.0.1:8765`. Pick a capture from the dropdown, leave
the sample rate at `100000`, and press **Analyze capture**.

On macOS or Linux the only difference is the activation step:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/generate_test_data.py
python scripts/serve_dashboard.py --open
```

> **Nothing else is installed.** The dashboard is served by Python's own standard
> library — no Flask, no npm, no CDN, and it makes no network requests. Every plot is
> drawn by your browser.

---

## Three ways to use it

All three call exactly the same analysis code (`pipeline.analyze_file`), so they can
never disagree with each other.

### 1. Web dashboard (easiest)

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

### 2. Desktop app

A full PyQt6 workbench with the same plots plus a **Payload** tab showing the decoded
message and a raw hex/ASCII dump.

```powershell
python scripts/run_app.py
```

Useful menu items: **Open Demo Capture**, **Batch Folder Analysis**, **Run Full Demo**.

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

**Modulation:** BPSK, QPSK, 2-FSK, 16-QAM (or `auto`).

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
| Convolutional | K=7, rate 1/2 (G1=0o171, G2=0o133) with hard-decision Viterbi |
| Reed–Solomon | RS(255,223) and RS(255,239) over GF(256) |
| LDPC | (3,4)-regular systematic code with min-sum belief propagation |
| Concatenated | CRC-16 → Reed–Solomon → convolutional |

**Interleaving:** block, convolutional (Forney), diagonal, pseudo-random, and
pseudo-random over sub-blocks.

**Why the interleaver matters.** Interleaving is not decoration. While auditing this
project we measured that LDPC *without* an interleaver is not even monotone in the
number of errors — 6 errors failed while 8 and 12 succeeded. The errors land inside the
same code block, and a block whose error count exceeds the code's correction capability
fails on its own. Spreading the same errors across blocks is exactly what the interleaver
does: with it the behaviour becomes monotone and the tolerance more than doubles. Every
FEC × interleaver combination the tool offers decodes back to the exact message.

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
python scripts/run_pipeline.py real_data/gps_l1_4mhz_cf32.iq --sample-rate 4000000 --iq-format auto
python scripts/run_pipeline.py real_data/ntsc_10mhz_cf32.iq --sample-rate 10000000 --iq-format auto
```

These are deliberately *hard* cases, and the tool's honesty is the point: neither carries
our sync word, so it correctly reports **no header and no decoded message**, and warns
that the modulation fit on the GPS capture is poor rather than pretending to know.

`scripts/fetch_real_data.py --list` also lists larger public sources (SDRangel
satellite/ADS-B/AIS recordings, the Mendeley 2.4 GHz dataset, DeepSig RadioML, IQEngine)
if you want more material for a live demo.

---

## Proving it works

```powershell
python scripts/verify_mvp.py
```

This regenerates the test data, runs the whole test suite, and then performs **55
individual checks** end to end. It prints `PASS` and exits `0` only if every one
succeeds. It covers, among others:

- the core pipeline on a clean BPSK capture (BER < 0.01, correlation > 0.9)
- all six coded captures decoding back to the **exact transmitted message**, with the
  transmitter's FEC scheme and interleaver identified from the bit stream alone
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
pytest                                        # the full suite (389 tests)
pytest tests/unit/test_fec_codecs.py -q       # one module
python scripts/benchmark_snr.py               # BER vs SNR sweep
$env:QT_QPA_PLATFORM="offscreen"; pytest tests/integration -q   # GUI tests, headless
```

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
    dsp.py             spectrum, waterfall, bandwidth/SNR/CFO, EVM/MER
    classifier.py      modulation classification
    demod.py           BPSK / QPSK / 2-FSK / 16-QAM demodulation
    correlator.py      sync-word correlation (with false-alarm control)
    fec.py             CRC-16/32, Viterbi, Reed-Solomon, LDPC, concatenated
    deinterleave.py    block / convolutional / diagonal / pseudo-random
    framing.py         transmit-side frame builder (mirror of the receive chain)
    payload.py         payload slicing, classification, hexdump
    report.py          report serialisation (strict JSON)
  gui/main_window.py   the PyQt6 desktop app
scripts/               run_app, run_pipeline, serve_dashboard, batch_analyze,
                       generate_test_data, fetch_real_data, verify_mvp, benchmark_snr
tests/                 366 tests: unit + integration (incl. headless GUI)
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
Usually a carrier frequency offset or the wrong modulation — the MVP has no carrier
recovery. Check the `cfo_estimate_hz` value, and try setting **Modulation** explicitly
instead of `auto-detect`. The tool will warn you when the constellation fit is poor.

**"No CRC-valid FEC/interleaver hypothesis matched"**
Not necessarily a problem. It means the tool could not *prove* a decode. For a real-world
recording that was never CRC-framed (analog video, unframed telemetry), this is the
correct and expected answer. If you know the frame length, pass `--frame-bits`.

**The dashboard says "server unreachable"**
The server is not running. Start it with `python scripts/serve_dashboard.py`.

**Nothing appears in `sample_data/`**
Run `python scripts/generate_test_data.py` first. The integration tests need it.

**`python3` not found on Windows**
Use `python`, not `python3`.

---

## How this maps to the problem statement

| Requirement | Where it is handled |
| --- | --- |
| Analyse `.IQ` and `.wav` files | `core/io.py` — raw IQ in four dtypes plus WAV, with auto-detection |
| Extract signal parameters (sampling frequency, modulation, FEC, interleaving) | `core/dsp.py`, `core/classifier.py`, `core/fec.py`, `core/deinterleave.py` |
| Demodulate FSK, PSK, QAM | `core/demod.py` — 2-FSK, BPSK, QPSK, 16-QAM. 2-FSK also recovers its own symbol period (`core/dsp.py`), so a real burst with many samples per bit is decodable end to end |
| De-interleave: Block, Convolution, Diagonal, Pseudo-Random | `core/deinterleave.py` — all four, plus a sub-block pseudo-random variant |
| FEC: convolutional + Viterbi, Reed–Solomon, concatenated, LDPC | `core/fec.py` — all implemented from scratch and CRC-verified |
| Bit stream correlation | `core/correlator.py` — with false-alarm control so long captures cannot fake a header |
| GUI-based model | `gui/main_window.py` (desktop) and `web/dashboard.html` (browser) |
| Improved feature visibility — constellation, waterfall | Both UIs, plus spectrum, eye diagram and a bit-stream ribbon |
| Automated analysis | `pipeline.analyze_file()`, `batch.py`, `scripts/batch_analyze.py` |
| Error correction | `core/fec.py`, verified by CRC-16 |
| Header/payload identification | `core/correlator.py` + `core/payload.py` |

**Extra features beyond the brief:** zero-install web dashboard; IQ-format auto-detection;
one-click demo and full-demo runs; batch folder analysis with CSV and self-contained HTML
export; sampling-rate hypothesis ranking; EVM/MER constellation quality with a
modulation-fit cross-check; payload extraction with text/hex viewer; shareable deep links;
constant-false-alarm-rate sync detection; a reproducible real-data fetch with checksum
verification.

---

## Known limitations

We would rather list these than have you discover them:

- **The MVP DSP is intentionally simple.** Centre frequency is the PSD peak, bandwidth
  uses a median-noise-floor rule, and there is no timing recovery, carrier recovery or
  equalisation. A rotated or frequency-offset signal inflates the EVM. These are
  documented V2 items in `info.md` §32.
- **2-FSK symbol-period recovery needs a few samples per symbol.** Below about five
  samples per symbol the piecewise-constant model cannot be told apart from noise, so the
  period is reported as unrecovered and the naive one-bit-per-sample path is used. The
  tool never invents a period it cannot reconstruct.
- **The bandwidth estimator cannot measure flat-spectrum signals** such as QPSK, where a
  real signal's peak-above-noise-floor is no larger than pure noise's. Rather than lower
  the threshold (which would start classifying noise as signal), the tool reports the
  estimate as unavailable and says so.
- **Modulation classification is a heuristic**, and it gets real-world signals wrong —
  the GPS L1 capture is classified as 16-QAM. The tool flags the poor fit rather than
  hiding it. EVM is *not* used to pick a modulation, because it always favours denser
  constellations.
- **Blind FEC detection is a search, not a detector.** Nothing is reported as decoded
  without a CRC-16 pass, so an unusual or unlisted scheme will simply not be found.
- **An uncoded frame corrects nothing, by design.** With no FEC, a single flipped bit
  breaks the CRC-16 and the tool reports no decode rather than guessing. This is the
  control that makes the error-correction claims meaningful.
- **The decode search examines a bounded prefix** (16,384 bits, 4 s by default) to respect
  the performance budget. Raise `decode_max_bits` / `decode_time_budget_s` for long frames.
- **No GNU Radio and no RF hardware.** Everything runs on files.

---

## License and attribution

Third-party test recordings downloaded by `scripts/fetch_real_data.py` are published by
the [PySDR project](https://pysdr.org) and are used here as test material only. See
`real_data/SOURCES.json` for per-file provenance.

See `info.md` for the full PRD, schema definitions, build plan and verification matrix.
