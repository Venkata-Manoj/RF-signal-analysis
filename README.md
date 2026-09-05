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

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-dev.txt
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

See `info.md` for the full PRD, build plan, and verification matrix.
