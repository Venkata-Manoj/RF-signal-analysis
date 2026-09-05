Below is a complete **MVP PRD + Technical Build Plan + Automated Verification Plan** for your SIH’26 Problem Statement.

Use this as your execution document.

---

# MVP PRD: RF Signal Analyzer for `.IQ` and `.wav` Files

## Project Name

**RF Signal Analysis Workbench MVP**

## Problem Statement

Build an automated GUI-based system to analyze `.IQ` and `.wav` signal recordings, extract signal parameters, visualize spectral features, demodulate basic signals, and correlate bitstreams for header/payload identification.

## MVP Objective

Build a working prototype that demonstrates the complete pipeline:

```text
Input .IQ/.wav
    ↓
Signal Visualization
    ↓
Basic Parameter Extraction
    ↓
Modulation Estimation
    ↓
Limited Demodulation
    ↓
Bitstream Extraction
    ↓
Header/Sync Word Correlation
    ↓
Report Export
```

This MVP is not a full blind SIGINT system. It is a hackathon-grade working prototype with a clear upgrade path.

---

# 1. MVP Scope

## 1.1 In Scope for MVP

| Area | MVP Requirement |
| --- | --- |
| Input files | Load raw `.IQ` and `.wav` files |
| IQ formats | Complex Float32, Int16 IQ, UInt8 IQ |
| WAV support | Mono and stereo WAV; stereo can be treated as I/Q |
| Metadata entry | User can enter sample rate, center frequency, format |
| Visualization | Time plot, spectrum, waterfall, constellation, eye diagram |
| Parameter extraction | Center frequency, occupied bandwidth, SNR estimate |
| Modulation estimation | Basic FSK/PSK/QAM classification with confidence |
| Demodulation | 2-FSK, BPSK, QPSK |
| FEC/Interleaving | Candidate-based validation only, not blind full detection |
| Bitstream correlation | Detect known sync word/header |
| Report | Export JSON report and bitstream file |
| GUI | Desktop GUI with file open, plots, run analysis, export |
| Automated verification | Synthetic test data + pytest + headless pipeline test |

---

## 1.2 Out of Scope for MVP

These should be presented as Version 2 features:

1. Fully blind sampling rate estimation.
2. Fully blind FEC identification.
3. Fully blind interleaver recovery.
4. Robust demodulation of weak, noisy, Doppler-affected signals.
5. Full LDPC, Turbo, Reed-Solomon, concatenated-code auto-detection.
6. Full protocol reverse engineering.
7. Real-time SDR input.
8. Cloud deployment.
9. Multi-user authentication.
10. Production-grade security and audit logging.

---

# 2. Target Users

## Primary User

RF/signal analysis operator or defense/telecom analyst who needs to inspect captured `.IQ` or `.wav` files quickly.

## User Needs

The user wants to:

1. Open a signal capture.
2. See spectrum and waterfall.
3. Identify likely signal parameters.
4. Attempt demodulation.
5. Extract bits.
6. Detect header/payload.
7. Export a report.

---

# 3. MVP Success Criteria

The MVP is successful if it can demonstrate the following:

## 3.1 Functional Success

1. Loads a synthetic or real `.IQ` file without crashing.
2. Loads a synthetic or real `.wav` file without crashing.
3. Displays time-domain, spectrum, waterfall, constellation, and eye diagram.
4. Estimates center frequency within acceptable tolerance on synthetic test tone.
5. Estimates bandwidth and SNR.
6. Detects a known BPSK/QPSK/2-FSK signal in clean synthetic conditions.
7. Demodulates a clean synthetic BPSK/QPSK/2-FSK signal with low BER.
8. Detects a known sync word/header in the bitstream.
9. Exports a structured JSON report.
10. GUI opens and runs analysis without crashing.

## 3.2 Automated Verification Success

1. All unit tests pass.
2. All integration tests pass.
3. Synthetic BPSK test achieves BER < 0.01 in clean conditions.
4. Sync word correlation detects correct offset.
5. Report contains all required fields.
6. GUI smoke test passes in headless mode.

---

# 4. Product Requirements

## 4.1 Functional Requirements

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-01 | System shall allow opening `.IQ` files | Must |
| FR-02 | System shall allow opening `.wav` files | Must |
| FR-03 | System shall allow user to enter sample rate | Must |
| FR-04 | System shall allow user to select IQ data type | Must |
| FR-05 | System shall display time-domain signal | Must |
| FR-06 | System shall display frequency spectrum | Must |
| FR-07 | System shall display waterfall plot | Must |
| FR-08 | System shall display constellation plot | Must |
| FR-09 | System shall display eye diagram | Must |
| FR-10 | System shall estimate center frequency | Must |
| FR-11 | System shall estimate occupied bandwidth | Must |
| FR-12 | System shall estimate SNR approximately | Must |
| FR-13 | System shall estimate possible modulation type | Should |
| FR-14 | System shall demodulate BPSK | Must |
| FR-15 | System shall demodulate QPSK | Should |
| FR-16 | System shall demodulate 2-FSK | Should |
| FR-17 | System shall display extracted bitstream | Must |
| FR-18 | System shall correlate known sync word/header | Must |
| FR-19 | System shall provide FEC/interleaver candidate scores | Optional for MVP |
| FR-20 | System shall export JSON report | Must |
| FR-21 | System shall export bitstream as `.bin` or `.txt` | Should |
| FR-22 | System shall show confidence scores | Should |
| FR-23 | System shall log processing steps | Should |

---

## 4.2 Non-Functional Requirements

| ID | Requirement |
| --- | --- |
| NFR-01 | MVP should load files up to 100 MB without crashing |
| NFR-02 | Visualization should remain responsive for 1–2 million samples |
| NFR-03 | Analysis of 1 million samples should complete in under 10 seconds on a normal laptop |
| NFR-04 | GUI should be usable on Windows/Linux |
| NFR-05 | Code should be modular and testable |
| NFR-06 | Automated tests should run without real RF hardware |
| NFR-07 | System should fail gracefully for unsupported files |
| NFR-08 | No external RF hardware required for demo |
| NFR-09 | All synthetic test data should be reproducible |
| NFR-10 | GUI should support manual override of parameters |

---

# 5. MVP User Flow

## Main Flow

1. User opens the application.
2. User clicks **Open File**.
3. User selects `.IQ` or `.wav`.
4. User enters sample rate if required.
5. User selects IQ format if required.
6. User clicks **Analyze**.
7. System displays:
   - Time plot
   - Spectrum
   - Waterfall
   - Constellation
   - Eye diagram
8. System estimates:
   - Center frequency
   - Bandwidth
   - SNR
   - Modulation
9. User selects demodulation mode:
   - Auto
   - BPSK
   - QPSK
   - 2-FSK
10. System extracts bits.
11. User enters known sync word or header bits.
12. System correlates bitstream and identifies header offset.
13. System exports report and bitstream.

---

# 6. GUI Requirements

## 6.1 Main Window Layout

```text
+---------------------------------------------------------------+
| Menu Bar: File | Analysis | Export | Help                     |
+---------------------------------------------------------------+
| Left Panel                 | Center Panel                     |
|                            |                                  |
| Open File                  | Tabs:                            |
| File Info                  |  Time                            |
| Sample Rate                |  Spectrum                        |
| Center Frequency           |  Waterfall                       |
| IQ Format                  |  Constellation                   |
| Modulation Mode            |  Eye Diagram                     |
| Sync Word                  |  Bitstream                       |
| Run Analysis               |                                  |
| Export Report              |                                  |
+---------------------------------------------------------------+
| Right Panel: Parameters / Results / Logs                      |
+---------------------------------------------------------------+
```

## 6.2 Required Controls

1. File open button.
2. Sample rate input.
3. Center frequency input.
4. IQ format dropdown.
5. Modulation dropdown.
6. Sync word input.
7. Run analysis button.
8. Export button.
9. Log console.
10. Parameter result table.

## 6.3 Required Plots

1. Time-domain magnitude/real/imag.
2. Power spectral density.
3. Waterfall spectrogram.
4. Constellation.
5. Eye diagram.
6. Bitstream view.

---

# 7. Technical Architecture

## 7.1 High-Level Architecture

```text
+----------------------+
|       GUI Layer      |
| PyQt6 / PySide6      |
+----------+-----------+
           |
+----------v-----------+
|   Analysis Pipeline  |
+----------+-----------+
           |
+----------v-----------+
| Core Processing      |
| IO / DSP / Demod /   |
| Correlation / Report |
+----------------------+
```

## 7.2 Modules

1. **File Loader**
   - Loads `.IQ`
   - Loads `.wav`
   - Converts to complex NumPy array

2. **Signal Visualization**
   - Time plot
   - FFT/PSD
   - Waterfall
   - Constellation
   - Eye diagram

3. **Parameter Estimator**
   - Center frequency
   - Bandwidth
   - SNR
   - Basic signal quality

4. **Modulation Classifier**
   - Rule-based features
   - Optional ML model later

5. **Demodulator**
   - BPSK
   - QPSK
   - 2-FSK

6. **Bitstream Processor**
   - Bit extraction
   - Sync word correlation
   - Header offset detection

7. **FEC/Interleaver Assistant**
   - Candidate testing
   - Confidence score
   - CRC validation if possible

8. **Report Generator**
   - JSON report
   - Bitstream export
   - Plot export

---

# 8. Recommended Tech Stack

| Layer | Technology |
| --- | --- |
| Language | Python 3.11 |
| Numerical processing | NumPy, SciPy |
| Audio/WAV | soundfile |
| GUI | PyQt6 or PySide6 |
| Fast plotting | pyqtgraph |
| Testing | pytest |
| Linting | ruff |
| Formatting | black |
| Optional type checking | mypy |
| Optional ML | PyTorch, scikit-learn |
| Optional DSP library | scikit-commpy |

For MVP, avoid GNU Radio initially unless your team already knows it.

---

# 9. Project Bootstrap

Create the project directory:

```bash
mkdir sih-rf-analyzer
cd sih-rf-analyzer
git init
```

Create Python virtual environment:

```bash
python -m venv .venv
```

Activate on Linux/macOS:

```bash
source .venv/bin/activate
```

Activate on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Create folder structure:

```bash
mkdir -p src/rf_analyzer/core
mkdir -p src/rf_analyzer/gui
mkdir -p scripts
mkdir -p tests/unit
mkdir -p tests/integration
mkdir -p tests/fixtures
mkdir -p sample_data
mkdir -p output
```

Create empty files:

```bash
touch README.md
touch requirements.txt
touch requirements-dev.txt
touch pytest.ini
touch .gitignore
touch Makefile
touch src/rf_analyzer/__init__.py
touch src/rf_analyzer/config.py
touch src/rf_analyzer/pipeline.py
touch src/rf_analyzer/core/__init__.py
touch src/rf_analyzer/core/io.py
touch src/rf_analyzer/core/dsp.py
touch src/rf_analyzer/core/demod.py
touch src/rf_analyzer/core/correlator.py
touch src/rf_analyzer/core/report.py
touch src/rf_analyzer/gui/__init__.py
touch src/rf_analyzer/gui/main_window.py
touch scripts/generate_test_data.py
touch scripts/run_app.py
touch scripts/run_pipeline.py
touch tests/conftest.py
touch tests/unit/test_io.py
touch tests/unit/test_dsp.py
touch tests/unit/test_demod.py
touch tests/unit/test_correlator.py
touch tests/integration/test_pipeline.py
```

---

# 10. Final Folder Structure

```text
sih-rf-analyzer/
│
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── .gitignore
├── Makefile
│
├── scripts/
│   ├── generate_test_data.py
│   ├── run_app.py
│   └── run_pipeline.py
│
├── src/
│   └── rf_analyzer/
│       ├── __init__.py
│       ├── config.py
│       ├── pipeline.py
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── io.py
│       │   ├── dsp.py
│       │   ├── demod.py
│       │   ├── correlator.py
│       │   └── report.py
│       │
│       └── gui/
│           ├── __init__.py
│           └── main_window.py
│
├── sample_data/
├── output/
│
└── tests/
    ├── conftest.py
    ├── fixtures/
    ├── unit/
    │   ├── test_io.py
    │   ├── test_dsp.py
    │   ├── test_demod.py
    │   └── test_correlator.py
    └── integration/
        └── test_pipeline.py
```

---

# 11. Dependency Files

## `requirements.txt`

```text
numpy>=1.26
scipy>=1.11
soundfile>=0.12
PyQt6>=6.6
pyqtgraph>=0.13
```

## `requirements-dev.txt`

```text
pytest>=8.0
ruff>=0.4
black>=24.0
mypy>=1.10
```

## `.gitignore`

```text
.venv/
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/
output/
sample_data/*.iq
sample_data/*.wav
*.egg-info/
```

## `pytest.ini`

```ini
[pytest]
testpaths = tests
pythonpath = src
addopts = -v
```

## `Makefile`

```makefile
.PHONY: install dev data test verify run clean

install:
 python -m pip install -r requirements.txt

dev:
 python -m pip install -r requirements.txt -r requirements-dev.txt

data:
 python scripts/generate_test_data.py

test:
 pytest

verify: dev data test

run:
 python scripts/run_app.py

clean:
 rm -rf .pytest_cache .mypy_cache .ruff_cache __pycache__
```

For Windows users without `make`, use:

```powershell
python -m pip install -r requirements.txt -r requirements-dev.txt
python scripts/generate_test_data.py
pytest
python scripts/run_app.py
```

---

# 12. Module Specifications

---

## 12.1 File Loader Module

File:

```text
src/rf_analyzer/core/io.py
```

Responsibilities:

1. Load raw `.IQ` files.
2. Load `.wav` files.
3. Return complex samples.
4. Return sample rate if available.

Functions:

```python
load_iq(file_path, dtype="complex64", endian="little")
load_wav(file_path)
detect_file_type(file_path)
```

Supported IQ formats:

| Format | Description |
| --- | --- |
| `complex64` | 32-bit float I + 32-bit float Q |
| `int16` | 16-bit integer I and Q interleaved |
| `uint8` | 8-bit unsigned integer I and Q interleaved |

Expected output:

```python
samples: np.ndarray
```

Example:

```python
samples = load_iq("sample.iq", dtype="int16")
```

---

## 12.2 DSP Module

File:

```text
src/rf_analyzer/core/dsp.py
```

Responsibilities:

1. Compute FFT/PSD.
2. Generate spectrogram/waterfall data.
3. Estimate center frequency.
4. Estimate bandwidth.
5. Estimate SNR.
6. Generate constellation data.
7. Generate eye diagram data.

Functions:

```python
compute_psd(samples, sample_rate, nfft=4096)
compute_waterfall(samples, sample_rate, nfft=1024, overlap=0.5)
estimate_center_frequency(freqs, psd_db)
estimate_bandwidth(freqs, psd_db, threshold_db=10)
estimate_snr(psd_db)
compute_eye(samples, samples_per_symbol=8)
```

---

## 12.3 Demodulation Module

File:

```text
src/rf_analyzer/core/demod.py
```

Responsibilities:

1. Demodulate BPSK.
2. Demodulate QPSK.
3. Demodulate 2-FSK.
4. Convert symbols to bits.

Functions:

```python
demod_bpsk(samples)
demod_qpsk(samples)
demod_2fsk(samples)
normalize_signal(samples)
```

MVP assumptions:

1. For clean synthetic signals, symbol alignment is known.
2. For real files, user may need to select symbol rate manually.
3. Advanced timing recovery is Version 2.

---

## 12.4 Correlation Module

File:

```text
src/rf_analyzer/core/correlator.py
```

Responsibilities:

1. Convert sync word hex to bits.
2. Correlate received bitstream with known bits.
3. Find header offset.
4. Return correlation confidence.

Functions:

```python
hex_to_bits(hex_string, bit_length=None)
sliding_correlate(received_bits, known_bits)
find_header(bits, sync_bits)
```

Output:

```python
{
    "offset": 1024,
    "score": 0.98,
    "sync_length": 32
}
```

---

## 12.5 Pipeline Module

File:

```text
src/rf_analyzer/pipeline.py
```

Responsibilities:

1. Coordinate full analysis.
2. Load file.
3. Run DSP.
4. Run classification.
5. Run demodulation.
6. Run correlation.
7. Generate report.

Main function:

```python
analyze_file(request)
```

Input example:

```python
request = {
    "file_path": "sample_data/bpsk.iq",
    "sample_rate": 100000,
    "center_frequency": 0,
    "iq_format": "complex64",
    "modulation": "auto",
    "sync_word": "0x1ACFFC1D"
}
```

Output:

```python
report = {
    "file": "bpsk.iq",
    "sample_rate": 100000,
    "center_frequency_estimate": 12.5,
    "bandwidth_estimate": 24000,
    "snr_db": 18.2,
    "modulation": "BPSK",
    "modulation_confidence": 0.83,
    "header_offset": 1024,
    "header_score": 0.97,
    "num_bits": 20000
}
```

---

## 12.6 Report Module

File:

```text
src/rf_analyzer/core/report.py
```

Responsibilities:

1. Validate report fields.
2. Save JSON report.
3. Save bitstream.
4. Save plots if required.

Functions:

```python
save_report(report, path)
save_bits(bits, path)
validate_report(report)
```

---

# 13. Report Schema

The MVP should export this JSON structure:

```json
{
  "meta": {
    "tool_name": "RF Signal Analysis Workbench MVP",
    "version": "0.1.0",
    "generated_at": "2026-09-06T12:00:00Z"
  },

  "input": {
    "file_name": "sample.iq",
    "file_type": "iq",
    "sample_rate": 100000,
    "center_frequency": null,
    "iq_format": "complex64"
  },

  "signal": {
    "num_samples": 100000,
    "duration_seconds": 1.0,
    "center_frequency_estimate": 12500,
    "bandwidth_estimate": 24000,
    "snr_db": 18.2
  },

  "modulation": {
    "estimated_type": "BPSK",
    "confidence": 0.82,
    "alternatives": ["QPSK", "2-FSK"]
  },

  "demodulation": {
    "mode": "BPSK",
    "num_bits": 20000,
    "bitstream_file": "bits.bin"
  },

  "correlation": {
    "sync_word": "0x1ACFFC1D",
    "header_offset": 1024,
    "score": 0.97
  },

  "fec": {
    "candidate": "Convolutional r=1/2 K=7",
    "confidence": 0.35,
    "crc_pass": null
  },

  "interleaving": {
    "candidate": "Block",
    "depth": 32,
    "confidence": 0.28
  },

  "warnings": [],
  "errors": []
}
```

---

# 14. Starter Code

Below is starter code for the core MVP.

---

## 14.1 `src/rf_analyzer/core/io.py`

```python
import numpy as np
import soundfile as sf
from pathlib import Path


def load_iq(file_path: str, dtype: str = "complex64", endian: str = "little") -> np.ndarray:
    """
    Load raw IQ file.

    Supported dtype:
    - complex64
    - int16
    - uint8
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"IQ file not found: {file_path}")

    if dtype == "complex64":
        samples = np.fromfile(file_path, dtype=np.complex64)

    elif dtype == "int16":
        raw = np.fromfile(file_path, dtype=np.int16)
        if endian == "big":
            raw = raw.byteswap()
        if len(raw) % 2 != 0:
            raw = raw[:-1]
        samples = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
        samples = samples / 32768.0

    elif dtype == "uint8":
        raw = np.fromfile(file_path, dtype=np.uint8)
        if len(raw) % 2 != 0:
            raw = raw[:-1]
        samples = (raw[0::2].astype(np.float32) - 128.0) + 1j * (raw[1::2].astype(np.float32) - 128.0)
        samples = samples / 128.0

    else:
        raise ValueError(f"Unsupported IQ dtype: {dtype}")

    return samples.astype(np.complex64)


def load_wav(file_path: str) -> tuple[np.ndarray, float]:
    """
    Load WAV file.

    If stereo:
      left = I
      right = Q

    If mono:
      treat as real/I channel.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"WAV file not found: {file_path}")

    data, sample_rate = sf.read(str(file_path), dtype="float32")

    if data.ndim == 2:
        if data.shape[1] < 2:
            raise ValueError("Stereo WAV must have at least two channels.")
        samples = data[:, 0] + 1j * data[:, 1]
    else:
        samples = data.astype(np.complex64)

    return samples.astype(np.complex64), float(sample_rate)
```

---

## 14.2 `src/rf_analyzer/core/dsp.py`

```python
import numpy as np


def compute_psd(samples: np.ndarray, sample_rate: float, nfft: int = 4096):
    """
    Compute power spectral density.
    Returns frequency axis and PSD in dB.
    """
    nfft = min(nfft, len(samples))

    if nfft <= 0:
        raise ValueError("Not enough samples for PSD computation.")

    window = np.hanning(nfft)
    segment = samples[:nfft] * window

    fft_vals = np.fft.fftshift(np.fft.fft(segment, n=nfft))
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate))

    psd = np.abs(fft_vals) ** 2
    psd_db = 10.0 * np.log10(psd + 1e-12)

    return freqs, psd_db


def estimate_center_frequency(freqs: np.ndarray, psd_db: np.ndarray) -> float:
    """
    Estimate center frequency as PSD peak.
    """
    peak_index = int(np.argmax(psd_db))
    return float(freqs[peak_index])


def estimate_bandwidth(freqs: np.ndarray, psd_db: np.ndarray, threshold_db: float = 10.0) -> float:
    """
    Estimate occupied bandwidth using noise-floor threshold.
    """
    noise_floor = float(np.median(psd_db))
    signal_mask = psd_db > (noise_floor + threshold_db)

    if not np.any(signal_mask):
        return 0.0

    signal_freqs = freqs[signal_mask]
    return float(np.max(signal_freqs) - np.min(signal_freqs))


def estimate_snr(psd_db: np.ndarray) -> float:
    """
    Rough SNR estimate:
    SNR = peak PSD - median noise floor
    """
    noise_floor = float(np.median(psd_db))
    peak = float(np.max(psd_db))
    return peak - noise_floor


def compute_waterfall(samples: np.ndarray, sample_rate: float, nfft: int = 1024, overlap: float = 0.5):
    """
    Compute waterfall matrix.

    Returns:
      freqs: frequency axis
      times: time axis
      waterfall_db: 2D array [time_bins, freq_bins]
    """
    step = max(1, int(nfft * (1.0 - overlap)))
    num_segments = max(1, (len(samples) - nfft) // step + 1)

    window = np.hanning(nfft)
    waterfall = np.zeros((num_segments, nfft), dtype=np.float32)

    for i in range(num_segments):
        start = i * step
        end = start + nfft

        if end > len(samples):
            break

        segment = samples[start:end] * window
        fft_vals = np.fft.fftshift(np.fft.fft(segment, n=nfft))
        psd = np.abs(fft_vals) ** 2
        waterfall[i, :] = 10.0 * np.log10(psd + 1e-12)

    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate))
    times = np.arange(num_segments) * (step / sample_rate)

    return freqs, times, waterfall
```

---

## 14.3 `src/rf_analyzer/core/demod.py`

```python
import numpy as np


def normalize_signal(samples: np.ndarray) -> np.ndarray:
    """
    Normalize signal amplitude.
    """
    max_val = np.max(np.abs(samples))
    if max_val == 0:
        return samples
    return samples / max_val


def demod_bpsk(samples: np.ndarray) -> np.ndarray:
    """
    Simple BPSK demodulation.
    Assumes phase-aligned signal.
    """
    samples = normalize_signal(samples)
    bits = (np.real(samples) > 0).astype(np.uint8)
    return bits


def demod_qpsk(samples: np.ndarray) -> np.ndarray:
    """
    Simple QPSK demodulation.
    Assumes phase-aligned signal.
    """
    samples = normalize_signal(samples)

    bits = []
    for sample in samples:
        bit_i = 1 if np.real(sample) > 0 else 0
        bit_q = 1 if np.imag(sample) > 0 else 0
        bits.append(bit_i)
        bits.append(bit_q)

    return np.array(bits, dtype=np.uint8)


def demod_2fsk(samples: np.ndarray) -> np.ndarray:
    """
    Simple 2-FSK demodulation using instantaneous frequency.
    """
    samples = normalize_signal(samples)

    phase = np.angle(samples)
    phase_unwrapped = np.unwrap(phase)
    inst_freq = np.diff(phase_unwrapped)

    threshold = 0.0
    bits = (inst_freq > threshold).astype(np.uint8)

    return bits
```

---

## 14.4 `src/rf_analyzer/core/correlator.py`

```python
import numpy as np


def hex_to_bits(hex_string: str, bit_length: int | None = None) -> np.ndarray:
    """
    Convert hex string to bit array.

    Example:
      "0x1A" -> [0,0,0,1,1,0,1,0]
    """
    hex_string = hex_string.lower().strip()

    if hex_string.startswith("0x"):
        hex_string = hex_string[2:]

    value = int(hex_string, 16)

    if bit_length is None:
        bit_length = len(hex_string) * 4

    bits = [(value >> (bit_length - 1 - i)) & 1 for i in range(bit_length)]
    return np.array(bits, dtype=np.uint8)


def sliding_correlate(received_bits: np.ndarray, known_bits: np.ndarray) -> tuple[int, float]:
    """
    Simple sliding hard-bit correlation.

    Returns:
      best_offset, best_score
    """
    received_bits = np.asarray(received_bits, dtype=np.uint8)
    known_bits = np.asarray(known_bits, dtype=np.uint8)

    if len(known_bits) == 0:
        raise ValueError("Known bits must not be empty.")

    if len(received_bits) < len(known_bits):
        return -1, 0.0

    best_offset = -1
    best_score = -1.0

    known_len = len(known_bits)

    for offset in range(len(received_bits) - known_len + 1):
        segment = received_bits[offset:offset + known_len]
        matches = np.sum(segment == known_bits)
        score = float(matches) / known_len

        if score > best_score:
            best_score = score
            best_offset = offset

    return best_offset, best_score


def find_header(received_bits: np.ndarray, sync_bits: np.ndarray, threshold: float = 0.85):
    """
    Find header using sync bits.
    """
    offset, score = sliding_correlate(received_bits, sync_bits)

    return {
        "offset": offset,
        "score": score,
        "detected": score >= threshold
    }
```

---

# 15. Synthetic Test Data Generation

You need automated test data because real RF files may not be available.

Create:

```text
scripts/generate_test_data.py
```

This script should generate:

1. Complex tone for center-frequency test.
2. BPSK signal with sync word.
3. QPSK signal with sync word.
4. 2-FSK signal.
5. WAV versions of the same.

---

## 15.1 Basic Synthetic Generation Script

```python
import numpy as np
import soundfile as sf
from pathlib import Path


SAMPLE_RATE = 100_000
OUTPUT_DIR = Path("sample_data")
OUTPUT_DIR.mkdir(exist_ok=True)


def save_iq(path: Path, samples: np.ndarray):
    samples.astype(np.complex64).tofile(path)


def save_wav_iq(path: Path, samples: np.ndarray, sample_rate: float):
    stereo = np.column_stack([samples.real, samples.imag]).astype(np.float32)
    sf.write(str(path), stereo, int(sample_rate), subtype="FLOAT")


def add_awgn(samples: np.ndarray, snr_db: float) -> np.ndarray:
    signal_power = np.mean(np.abs(samples) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power / 2.0) * (
        np.random.randn(len(samples)) + 1j * np.random.randn(len(samples))
    )
    return samples + noise


def generate_tone(freq: float = 10_000, sample_rate: float = SAMPLE_RATE, duration: float = 0.1):
    t = np.arange(int(sample_rate * duration)) / sample_rate
    samples = np.exp(1j * 2 * np.pi * freq * t)
    return samples.astype(np.complex64)


def bits_to_bpsk(bits: np.ndarray) -> np.ndarray:
    return 2.0 * bits.astype(np.float32) - 1.0


def generate_bpsk(sync_bits: np.ndarray, payload_bits: np.ndarray, snr_db: float = 30.0):
    bits = np.concatenate([sync_bits, payload_bits])
    symbols = bits_to_bpsk(bits)
    samples = symbols.astype(np.complex64)
    samples = add_awgn(samples, snr_db)
    return bits, samples.astype(np.complex64)


def bits_to_qpsk(bits: np.ndarray) -> np.ndarray:
    if len(bits) % 2 != 0:
        bits = bits[:-1]

    i_bits = bits[0::2]
    q_bits = bits[1::2]

    i_sym = 2.0 * i_bits.astype(np.float32) - 1.0
    q_sym = 2.0 * q_bits.astype(np.float32) - 1.0

    symbols = (i_sym + 1j * q_sym) / np.sqrt(2.0)
    return symbols.astype(np.complex64)


def generate_qpsk(sync_bits: np.ndarray, payload_bits: np.ndarray, snr_db: float = 30.0):
    bits = np.concatenate([sync_bits, payload_bits])
    symbols = bits_to_qpsk(bits)
    samples = add_awgn(symbols, snr_db)
    return bits, samples.astype(np.complex64)


def generate_2fsk(bits: np.ndarray, sample_rate: float = SAMPLE_RATE, symbol_duration: float = 0.001, freq_low: float = -5000, freq_high: float = 5000):
    samples_per_symbol = int(sample_rate * symbol_duration)
    phase = 0.0
    samples = []

    for bit in bits:
        freq = freq_high if bit == 1 else freq_low
        t = np.arange(samples_per_symbol) / sample_rate
        segment = np.exp(1j * (2 * np.pi * freq * t + phase))
        samples.append(segment)
        phase = np.angle(segment[-1])

    samples = np.concatenate(samples).astype(np.complex64)
    return samples


def hex_to_bits(hex_string: str, bit_length: int | None = None):
    hex_string = hex_string.lower().replace("0x", "")
    value = int(hex_string, 16)

    if bit_length is None:
        bit_length = len(hex_string) * 4

    bits = [(value >> (bit_length - 1 - i)) & 1 for i in range(bit_length)]
    return np.array(bits, dtype=np.uint8)


def main():
    np.random.seed(42)

    sync_word = "0x1ACFFC1D"
    sync_bits = hex_to_bits(sync_word)

    payload_bits = np.random.randint(0, 2, size=1000, dtype=np.uint8)

    # Tone
    tone = generate_tone()
    save_iq(OUTPUT_DIR / "tone.iq", tone)
    save_wav_iq(OUTPUT_DIR / "tone.wav", tone, SAMPLE_RATE)

    # BPSK
    bpsk_bits, bpsk_samples = generate_bpsk(sync_bits, payload_bits)
    save_iq(OUTPUT_DIR / "bpsk.iq", bpsk_samples)
    save_wav_iq(OUTPUT_DIR / "bpsk.wav", bpsk_samples, SAMPLE_RATE)
    np.save(OUTPUT_DIR / "bpsk_bits.npy", bpsk_bits)

    # QPSK
    qpsk_bits, qpsk_samples = generate_qpsk(sync_bits, payload_bits)
    save_iq(OUTPUT_DIR / "qpsk.iq", qpsk_samples)
    save_wav_iq(OUTPUT_DIR / "qpsk.wav", qpsk_samples, SAMPLE_RATE)
    np.save(OUTPUT_DIR / "qpsk_bits.npy", qpsk_bits)

    # 2-FSK
    fsk_bits = np.concatenate([sync_bits, payload_bits])
    fsk_samples = generate_2fsk(fsk_bits)
    save_iq(OUTPUT_DIR / "fsk2.iq", fsk_samples)
    save_wav_iq(OUTPUT_DIR / "fsk2.wav", fsk_samples, SAMPLE_RATE)
    np.save(OUTPUT_DIR / "fsk2_bits.npy", fsk_bits)

    print("Synthetic test data generated in sample_data/")


if __name__ == "__main__":
    main()
```

Run:

```bash
python scripts/generate_test_data.py
```

Expected output:

```text
sample_data/
├── tone.iq
├── tone.wav
├── bpsk.iq
├── bpsk.wav
├── bpsk_bits.npy
├── qpsk.iq
├── qpsk.wav
├── qpsk_bits.npy
├── fsk2.iq
├── fsk2.wav
└── fsk2_bits.npy
```

---

# 16. Automated Verification Plan

This is the most important part for making your MVP credible.

## 16.1 Verification Levels

| Level | What Is Tested | How |
| --- | --- | --- |
| Unit tests | Individual functions | pytest |
| Integration tests | Full pipeline | pytest |
| Synthetic signal tests | Known signals with known ground truth | generated test data |
| GUI smoke test | App opens without crashing | PyQt offscreen |
| Report validation | JSON schema and required fields | pytest |
| Performance smoke test | Load/analysis time | optional pytest |

---

# 17. Unit Tests

## 17.1 `tests/unit/test_io.py`

```python
import numpy as np
from rf_analyzer.core.io import load_iq, load_wav


def test_load_complex_iq(tmp_path):
    samples = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex64)
    file_path = tmp_path / "test.iq"
    samples.tofile(file_path)

    loaded = load_iq(str(file_path), dtype="complex64")

    assert len(loaded) == len(samples)
    assert np.allclose(loaded, samples, atol=1e-6)


def test_load_int16_iq(tmp_path):
    raw = np.array([1000, -1000, 2000, -2000], dtype=np.int16)
    file_path = tmp_path / "test_int16.iq"
    raw.tofile(file_path)

    loaded = load_iq(str(file_path), dtype="int16")

    assert len(loaded) == 2
    assert np.isclose(loaded[0].real, 1000 / 32768.0, atol=1e-4)
    assert np.isclose(loaded[0].imag, -1000 / 32768.0, atol=1e-4)


def test_load_wav_stereo_iq(tmp_path):
    import soundfile as sf

    samples = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    file_path = tmp_path / "test.wav"
    sf.write(str(file_path), samples, 100000, subtype="FLOAT")

    iq, sample_rate = load_wav(str(file_path))

    assert sample_rate == 100000
    assert len(iq) == 2
    assert np.isclose(iq[0].real, 0.1, atol=1e-6)
    assert np.isclose(iq[0].imag, 0.2, atol=1e-6)
```

---

## 17.2 `tests/unit/test_dsp.py`

```python
import numpy as np
from rf_analyzer.core.dsp import (
    compute_psd,
    estimate_center_frequency,
    estimate_bandwidth,
    estimate_snr,
)


def generate_tone(freq=10000, sample_rate=100000, duration=0.1):
    t = np.arange(int(sample_rate * duration)) / sample_rate
    return np.exp(1j * 2 * np.pi * freq * t).astype(np.complex64)


def test_psd_peak_center_frequency():
    sample_rate = 100000
    target_freq = 10000
    samples = generate_tone(freq=target_freq, sample_rate=sample_rate)

    freqs, psd_db = compute_psd(samples, sample_rate, nfft=4096)
    estimated_freq = estimate_center_frequency(freqs, psd_db)

    assert abs(estimated_freq - target_freq) < 1000


def test_bandwidth_positive():
    sample_rate = 100000
    samples = generate_tone(freq=10000, sample_rate=sample_rate)

    freqs, psd_db = compute_psd(samples, sample_rate, nfft=4096)
    bw = estimate_bandwidth(freqs, psd_db)

    assert bw >= 0.0


def test_snr_positive_for_tone():
    sample_rate = 100000
    samples = generate_tone(freq=10000, sample_rate=sample_rate)

    freqs, psd_db = compute_psd(samples, sample_rate, nfft=4096)
    snr = estimate_snr(psd_db)

    assert snr > 0
```

---

## 17.3 `tests/unit/test_demod.py`

```python
import numpy as np
from rf_analyzer.core.demod import demod_bpsk, demod_qpsk


def test_bpsk_demod_clean():
    bits = np.array([0, 1, 0, 1, 1], dtype=np.uint8)
    symbols = 2.0 * bits.astype(np.float32) - 1.0
    samples = symbols.astype(np.complex64)

    demod_bits = demod_bpsk(samples)

    assert np.array_equal(demod_bits, bits)


def test_qpsk_demod_clean():
    # Bits: I,Q pairs
    bits = np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8)

    i_bits = bits[0::2]
    q_bits = bits[1::2]

    i_sym = 2.0 * i_bits.astype(np.float32) - 1.0
    q_sym = 2.0 * q_bits.astype(np.float32) - 1.0

    samples = (i_sym + 1j * q_sym) / np.sqrt(2.0)
    samples = samples.astype(np.complex64)

    demod_bits = demod_qpsk(samples)

    assert np.array_equal(demod_bits, bits)
```

---

## 17.4 `tests/unit/test_correlator.py`

```python
import numpy as np
from rf_analyzer.core.correlator import hex_to_bits, sliding_correlate, find_header


def test_hex_to_bits():
    bits = hex_to_bits("0x1", bit_length=4)
    assert np.array_equal(bits, np.array([0, 0, 0, 1], dtype=np.uint8))


def test_sliding_correlate_finds_offset():
    sync_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    random_prefix = np.random.randint(0, 2, size=100, dtype=np.uint8)
    random_suffix = np.random.randint(0, 2, size=100, dtype=np.uint8)

    received = np.concatenate([random_prefix, sync_bits, random_suffix])

    offset, score = sliding_correlate(received, sync_bits)

    assert offset == 100
    assert score > 0.99


def test_find_header():
    sync_bits = np.array([1, 1, 0, 0, 1, 0, 1, 0], dtype=np.uint8)
    prefix = np.zeros(50, dtype=np.uint8)
    suffix = np.ones(50, dtype=np.uint8)

    received = np.concatenate([prefix, sync_bits, suffix])

    result = find_header(received, sync_bits, threshold=0.9)

    assert result["detected"] is True
    assert result["offset"] == 50
```

---

# 18. Integration Test

Create:

```text
tests/integration/test_pipeline.py
```

This tests the full MVP pipeline without GUI.

## Example Pipeline Test

```python
import numpy as np
from pathlib import Path
from rf_analyzer.pipeline import analyze_file


SAMPLE_DATA = Path("sample_data")


def test_bpsk_pipeline():
    request = {
        "file_path": str(SAMPLE_DATA / "bpsk.iq"),
        "sample_rate": 100000,
        "iq_format": "complex64",
        "modulation": "BPSK",
        "sync_word": "0x1ACFFC1D",
    }

    report = analyze_file(request)

    assert report["errors"] == []
    assert report["input"]["file_type"] == "iq"
    assert report["signal"]["num_samples"] > 0
    assert report["demodulation"]["num_bits"] > 0
    assert report["correlation"]["header_offset"] >= 0
    assert report["correlation"]["score"] > 0.9


def test_wav_pipeline():
    request = {
        "file_path": str(SAMPLE_DATA / "bpsk.wav"),
        "iq_format": "auto",
        "modulation": "BPSK",
        "sync_word": "0x1ACFFC1D",
    }

    report = analyze_file(request)

    assert report["errors"] == []
    assert report["input"]["file_type"] == "wav"
    assert report["signal"]["num_samples"] > 0
```

---

# 19. GUI Smoke Test

Create:

```text
tests/integration/test_gui_smoke.py
```

Run GUI in offscreen mode.

Set environment variable:

Linux/macOS:

```bash
export QT_QPA_PLATFORM=offscreen
```

Windows PowerShell:

```powershell
$env:QT_QPA_PLATFORM="offscreen"
```

Test:

```python
import sys
from PyQt6.QtWidgets import QApplication
from rf_analyzer.gui.main_window import MainWindow


def test_main_window_opens():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = MainWindow()
    window.setWindowTitle("RF Signal Analyzer MVP")

    assert window.windowTitle() == "RF Signal Analyzer MVP"

    window.close()
```

---

# 20. Pipeline Implementation Sketch

Create:

```text
src/rf_analyzer/pipeline.py
```

Basic structure:

```python
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

from rf_analyzer.core.io import load_iq, load_wav
from rf_analyzer.core.dsp import (
    compute_psd,
    estimate_center_frequency,
    estimate_bandwidth,
    estimate_snr,
)
from rf_analyzer.core.demod import demod_bpsk, demod_qpsk, demod_2fsk
from rf_analyzer.core.correlator import hex_to_bits, find_header


def analyze_file(request: dict) -> dict:
    file_path = Path(request["file_path"])
    suffix = file_path.suffix.lower()

    errors = []
    warnings = []

    try:
        if suffix == ".iq":
            samples = load_iq(
                str(file_path),
                dtype=request.get("iq_format", "complex64")
            )
            sample_rate = float(request.get("sample_rate", 1.0))
            file_type = "iq"

        elif suffix == ".wav":
            samples, sample_rate = load_wav(str(file_path))
            file_type = "wav"

        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        freqs, psd_db = compute_psd(samples, sample_rate)

        center_freq = estimate_center_frequency(freqs, psd_db)
        bandwidth = estimate_bandwidth(freqs, psd_db)
        snr = estimate_snr(psd_db)

        modulation = request.get("modulation", "BPSK").upper()

        if modulation == "BPSK":
            bits = demod_bpsk(samples)
        elif modulation == "QPSK":
            bits = demod_qpsk(samples)
        elif modulation in ["2FSK", "FSK"]:
            bits = demod_2fsk(samples)
        else:
            bits = demod_bpsk(samples)
            warnings.append("Unsupported modulation selected. Used BPSK fallback.")

        correlation_result = {
            "sync_word": request.get("sync_word"),
            "header_offset": -1,
            "score": 0.0,
            "detected": False,
        }

        if request.get("sync_word"):
            sync_bits = hex_to_bits(request["sync_word"])
            correlation_result = find_header(bits, sync_bits)
            correlation_result["sync_word"] = request["sync_word"]

        report = {
            "meta": {
                "tool_name": "RF Signal Analysis Workbench MVP",
                "version": "0.1.0",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            "input": {
                "file_name": file_path.name,
                "file_type": file_type,
                "sample_rate": sample_rate,
                "center_frequency": request.get("center_frequency"),
                "iq_format": request.get("iq_format", "auto"),
            },
            "signal": {
                "num_samples": int(len(samples)),
                "duration_seconds": float(len(samples) / sample_rate),
                "center_frequency_estimate": center_freq,
                "bandwidth_estimate": bandwidth,
                "snr_db": snr,
            },
            "modulation": {
                "estimated_type": modulation,
                "confidence": 0.5,
                "alternatives": [],
            },
            "demodulation": {
                "mode": modulation,
                "num_bits": int(len(bits)),
                "bitstream_file": None,
            },
            "correlation": correlation_result,
            "fec": {
                "candidate": None,
                "confidence": 0.0,
                "crc_pass": None,
            },
            "interleaving": {
                "candidate": None,
                "depth": None,
                "confidence": 0.0,
            },
            "warnings": warnings,
            "errors": errors,
        }

        return report

    except Exception as e:
        return {
            "meta": {
                "tool_name": "RF Signal Analysis Workbench MVP",
                "version": "0.1.0",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            "input": {
                "file_name": file_path.name,
                "file_type": None,
                "sample_rate": None,
                "center_frequency": None,
                "iq_format": None,
            },
            "signal": {},
            "modulation": {},
            "demodulation": {},
            "correlation": {},
            "fec": {},
            "interleaving": {},
            "warnings": warnings,
            "errors": [str(e)],
        }
```

This is not the final polished version, but it is enough to start.

---

# 21. Automated Verification Commands

After installing dependencies:

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
```

Generate test data:

```bash
python scripts/generate_test_data.py
```

Run tests:

```bash
pytest
```

Run only unit tests:

```bash
pytest tests/unit
```

Run only integration tests:

```bash
pytest tests/integration
```

Run with coverage:

```bash
pip install pytest-cov
pytest --cov=src/rf_analyzer
```

Full verification:

```bash
make verify
```

or manually:

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python scripts/generate_test_data.py
pytest
```

---

# 22. MVP Acceptance Test Matrix

Use this table to verify the MVP.

| Test ID | Test | Input | Expected Result |
| --- | --- | --- | --- |
| AT-01 | Load complex IQ | `tone.iq` | Samples loaded, length > 0 |
| AT-02 | Load int16 IQ | synthetic int16 file | Samples normalized correctly |
| AT-03 | Load WAV | `bpsk.wav` | Samples and sample rate loaded |
| AT-04 | Spectrum plot | `tone.iq` | Peak visible near expected frequency |
| AT-05 | Center frequency | `tone.iq` at 10 kHz | Estimate within tolerance |
| AT-06 | Bandwidth estimate | `tone.iq` | Non-negative value |
| AT-07 | SNR estimate | clean tone | Positive value |
| AT-08 | BPSK demod | clean `bpsk.iq` | BER < 0.01 |
| AT-09 | QPSK demod | clean `qpsk.iq` | BER < 0.05 for simple aligned case |
| AT-10 | 2-FSK demod | clean `fsk2.iq` | Bits approximately recovered |
| AT-11 | Sync correlation | BPSK bits with sync word | Offset detected |
| AT-12 | Report export | pipeline run | JSON contains required fields |
| AT-13 | GUI opens | app launch | Window opens without crash |
| AT-14 | GUI analysis | load file and run | Plots and results update |
| AT-15 | Error handling | invalid file | Error shown gracefully |

---

# 23. Automated MVP Verification Script

Create:

```text
scripts/verify_mvp.py
```

Purpose:

1. Generate synthetic data.
2. Run pipeline on sample files.
3. Check required report fields.
4. Check correlation result.
5. Print PASS/FAIL.

Basic structure:

```python
import json
import subprocess
import sys
from pathlib import Path


def run_command(cmd):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)

    return result


def main():
    print("=== RF Analyzer MVP Verification ===")

    print("[1/3] Generating synthetic data...")
    run_command([sys.executable, "scripts/generate_test_data.py"])

    print("[2/3] Running pytest...")
    run_command([sys.executable, "-m", "pytest", "-q"])

    print("[3/3] MVP verification complete.")
    print("PASS")


if __name__ == "__main__":
    main()
```

Run:

```bash
python scripts/verify_mvp.py
```

Expected final output:

```text
=== RF Analyzer MVP Verification ===
[1/3] Generating synthetic data...
[2/3] Running pytest...
[3/3] MVP verification complete.
PASS
```

---

# 24. CI Pipeline

Create:

```text
.github/workflows/verify.yml
```

```yaml
name: Verify MVP

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest

    env:
      QT_QPA_PLATFORM: offscreen

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Generate synthetic test data
        run: |
          python scripts/generate_test_data.py

      - name: Run tests
        run: |
          pytest
```

This gives you automated verification on every commit.

---

# 25. Development Milestones

## Milestone 1: Project Setup

Deliverables:

- Repository created.
- Folder structure created.
- Virtual environment works.
- Dependencies install.
- pytest runs.

Definition of Done:

```text
pytest runs with dummy test and passes.
```

---

## Milestone 2: File Loading

Deliverables:

- Load `.IQ`.
- Load `.wav`.
- Unit tests pass.

Definition of Done:

```text
tests/unit/test_io.py passes.
```

---

## Milestone 3: DSP and Visualization Data

Deliverables:

- PSD calculation.
- Waterfall calculation.
- Center frequency estimation.
- Bandwidth estimation.
- SNR estimation.

Definition of Done:

```text
tests/unit/test_dsp.py passes.
```

---

## Milestone 4: Demodulation

Deliverables:

- BPSK demodulation.
- QPSK demodulation.
- 2-FSK demodulation.
- BER tests on synthetic clean signals.

Definition of Done:

```text
tests/unit/test_demod.py passes.
```

---

## Milestone 5: Bitstream Correlation

Deliverables:

- Hex to bits.
- Sliding correlation.
- Header detection.

Definition of Done:

```text
tests/unit/test_correlator.py passes.
```

---

## Milestone 6: Pipeline

Deliverables:

- End-to-end `analyze_file()` function.
- JSON report generation.
- Integration tests pass.

Definition of Done:

```text
tests/integration/test_pipeline.py passes.
```

---

## Milestone 7: GUI

Deliverables:

- Main window.
- File open dialog.
- Parameter input fields.
- Plot tabs.
- Run analysis button.
- Result panel.

Definition of Done:

```text
GUI opens and runs analysis on synthetic file without crashing.
```

---

## Milestone 8: MVP Verification

Deliverables:

- Synthetic data generator.
- pytest suite.
- CI pipeline.
- MVP verification script.

Definition of Done:

```text
python scripts/verify_mvp.py prints PASS.
```

---

# 26. Task Board for Team

Use this as your GitHub Issues or SIH team task list.

## Backend Tasks

1. Create project skeleton.
2. Implement IQ loader.
3. Implement WAV loader.
4. Implement PSD computation.
5. Implement waterfall computation.
6. Implement parameter estimation.
7. Implement BPSK demodulator.
8. Implement QPSK demodulator.
9. Implement 2-FSK demodulator.
10. Implement sync-word correlator.
11. Implement report generator.
12. Implement pipeline orchestrator.

## GUI Tasks

1. Create main window.
2. Add file open button.
3. Add metadata input panel.
4. Add plot tabs.
5. Add result table.
6. Add log console.
7. Add export button.
8. Connect backend pipeline to GUI.
9. Add error popups.
10. Polish UI theme.

## Testing Tasks

1. Create synthetic data generator.
2. Write IO tests.
3. Write DSP tests.
4. Write demodulation tests.
5. Write correlation tests.
6. Write pipeline tests.
7. Write GUI smoke test.
8. Add CI workflow.
9. Add MVP verification script.
10. Create acceptance test checklist.

## Documentation Tasks

1. Write README.
2. Write setup instructions.
3. Write demo script.
4. Write judge presentation.
5. Create architecture diagram.
6. Create Version 2 roadmap slide.

---

# 27. README Template

Use this in your repository.

```md
# RF Signal Analysis Workbench MVP

MVP for automated analysis of `.IQ` and `.wav` files with signal parameter extraction, basic demodulation, and bitstream correlation.

## Features

- Load raw IQ and WAV files
- Spectrum, waterfall, constellation, and eye visualization
- Basic parameter estimation
- BPSK/QPSK/2-FSK demodulation
- Sync-word correlation
- JSON report export

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

## Generate Test Data

```bash
python scripts/generate_test_data.py
```

## Run Tests

```bash
pytest
```

## Run GUI

```bash
python scripts/run_app.py
```

## MVP Scope

This MVP demonstrates the end-to-end workflow. Advanced blind FEC detection, interleaving recovery, and robust synchronization are planned for Version 2.

```

---

# 28. Manual Demo Checklist

Before SIH demo, verify:

## Pre-Demo Checklist

- [ ] Laptop can run app without internet.
- [ ] Sample `.IQ` files are available.
- [ ] Sample `.wav` files are available.
- [ ] GUI opens in less than 10 seconds.
- [ ] No crash when opening large file.
- [ ] Spectrum appears.
- [ ] Waterfall appears.
- [ ] Constellation appears.
- [ ] Bitstream appears.
- [ ] Header correlation works.
- [ ] JSON report exports.
- [ ] Backup demo video is ready.
- [ ] PPT has architecture diagram.
- [ ] PPT has Version 2 roadmap.
- [ ] Team can explain limitations confidently.

---

# 29. Demo Script for Judges

Use this during the 3–5 minute demo.

## Opening

> “We have built an RF Signal Analysis Workbench MVP that automates the workflow from raw IQ/WAV capture to parameter extraction, demodulation, and header correlation.”

## Step 1: Open File

> “We open a captured IQ file. The user can provide sample rate and IQ format.”

## Step 2: Visualize

> “The system shows time-domain, spectrum, waterfall, constellation, and eye diagram.”

## Step 3: Parameter Extraction

> “It estimates center frequency, occupied bandwidth, and SNR.”

## Step 4: Modulation

> “It classifies likely modulation and provides a confidence score.”

## Step 5: Demodulation

> “It demodulates the selected signal and extracts the bitstream.”

## Step 6: Correlation

> “We provide a known sync word. The system detects the header offset and separates payload.”

## Step 7: Report

> “Finally, it exports a structured JSON report for further analysis.”

## Closing

> “The MVP demonstrates the complete pipeline. Version 2 will add advanced blind FEC/interleaver detection, wider modulation support, and robust synchronization for real-world noisy captures.”

---

# 30. What to Say If Judges Ask About Missing Full Features

If they ask:

> “Did you complete FEC and interleaving detection?”

Answer:

> “The MVP uses candidate-based validation instead of fully blind detection because blind FEC and interleaving recovery are highly constrained problems. It tests likely configurations and validates them using header/CRC checks. Full blind detection is planned for Version 2 using protocol-specific plugins and ML-assisted hypothesis testing.”

If they ask:

> “Can it detect sampling rate automatically?”

Answer:

> “The current MVP allows user-provided sample rate because raw IQ files often do not contain metadata. Version 2 will include sample-rate hypothesis testing using known bandwidths and cyclostationary features.”

If they ask:

> “Is this production ready?”

Answer:

> “It is a working prototype that validates the core workflow. For production, we need real dataset tuning, robust synchronization, protocol-specific libraries, and large-file optimization.”

---

# 31. MVP Definition of Done

The MVP is complete when all of these are true:

## Code

- [ ] Project runs locally.
- [ ] Dependencies install cleanly.
- [ ] GUI opens.
- [ ] Pipeline runs without GUI.
- [ ] Report exports.

## Features

- [ ] Loads `.IQ`.
- [ ] Loads `.wav`.
- [ ] Shows spectrum.
- [ ] Shows waterfall.
- [ ] Shows constellation.
- [ ] Estimates center frequency.
- [ ] Estimates bandwidth.
- [ ] Estimates SNR.
- [ ] Demodulates BPSK.
- [ ] Correlates sync word.
- [ ] Exports JSON.

## Tests

- [ ] Unit tests pass.
- [ ] Integration tests pass.
- [ ] GUI smoke test passes.
- [ ] Synthetic BPSK test passes.
- [ ] Correlation test passes.
- [ ] CI passes.

## Presentation

- [ ] Architecture diagram ready.
- [ ] Demo script ready.
- [ ] Backup video ready.
- [ ] Version 2 roadmap ready.
- [ ] Limitations clearly explained.

---

# 32. Version 2 Roadmap

After the hackathon, propose this:

## Version 2 Features

1. Advanced synchronization:
   - Costas loop
   - Polyphase timing sync
   - Equalization
   - Frequency offset correction

2. Expanded modulation support:
   - 4-FSK
   - 8PSK
   - 16-QAM
   - 64-QAM
   - GFSK
   - AFSK

3. ML modulation classifier:
   - CNN on spectrograms
   - CNN on constellation images
   - Feature-based classifier
   - Confidence calibration

4. FEC engine:
   - Viterbi decoder
   - Reed-Solomon decoder
   - LDPC decoder
   - CRC-based hypothesis validation

5. Interleaver engine:
   - Block de-interleaver
   - Convolutional de-interleaver
   - Pseudo-random seed search
   - Entropy-based validation

6. Protocol plugins:
   - Known header templates
   - Frame parser
   - Payload decoder

7. Performance:
   - Large-file streaming
   - GPU FFT
   - Parallel batch processing

8. GUI improvements:
   - Annotation tools
   - Manual signal region selection
   - Multi-file comparison
   - Session save/load

---

# 33. Final Recommendation

Do this:

> Build the MVP exactly as defined above.  
> Present the full vision.  
> Demonstrate the working pipeline.  
> Clearly show Version 2 roadmap.

Do not say:

> “We fully completed blind FEC and interleaving detection.”

Instead say:

> “We built a working MVP that demonstrates the end-to-end signal-analysis workflow. Advanced blind detection modules are designed as future extensions because they require protocol-specific constraints and real-world training data.”

This is the strongest and most credible SIH strategy.

---

# 34. Next Immediate Build Step

Your next step is to create the repository and implement in this order:

1. Folder structure  
2. `requirements.txt`  
3. IQ/WAV loader  
4. Synthetic data generator  
5. DSP functions  
6. Demodulation functions  
7. Correlator  
8. Pipeline  
9. Tests  
10. GUI shell  

If you follow the above PRD, your MVP will be:

- Buildable
- Testable
- Demoable
- Honest about limitations
- Strong enough for SIH evaluation

If you want, I can next give you the **complete starter code repository files one by one**, starting with:

1. `requirements.txt`  
2. `src/rf_analyzer/core/io.py`  
3. `scripts/generate_test_data.py`  
4. `tests/unit/test_io.py`
