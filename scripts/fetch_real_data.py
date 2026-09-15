"""Download and prepare real-world RF test captures for SIH 2026 PS 147.

Provides links, documentation, and tools to acquire real-world
satellite, aircraft, and radio IQ/WAV captures for off-the-air testing.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REAL_DATA_DIR = ROOT / "real_data"


def print_manual_download_guide() -> None:
    """Print curated URLs for full-bandwidth real satellite and SDR captures."""
    guide = """
================================================================================
REAL-WORLD RF CAPTURE REPOSITORY & DOWNLOAD GUIDE (SIH 2026 PS 147)
================================================================================

For Grand Finale live demonstrations, evaluators expect real over-the-air signals:

1. SDRangel Real IQ Collection (WAV / SDRIQ):
   - Website: https://www.sdrangel.org/
   - Available captures:
     * NOAA APT Satellite (137 MHz WAV): Weather satellite downlink
     * ADS-B Aircraft Transponders (1090 MHz WAV)
     * AIS Ship Tracking (162 MHz GMSK)
     * AX.25 Satellite Packet Radio (VHF WAV)

2. Mendeley Real-World 2.4 GHz IQ Dataset (Labeled BPSK/QPSK/QAM/OFDM):
   - URL: https://data.mendeley.com/datasets/tjzsbph49x/2
   - Contains multipath and line-of-sight real RF captures across diverse SNR conditions.

3. DeepSig RadioML 2016.10A / 2018.01A:
   - URL: https://www.deepsig.ai/datasets
   - Standard reference dataset for AMC benchmarking (11 modulation types, SNR -20 to +18 dB).

4. IQEngine (SigMF recordings browser):
   - URL: https://www.iqengine.org/
   - Features standard SigMF metadata sidecars with real-world satellite, aviation, and ham radio captures.

Place any downloaded .iq or .wav files into the `real_data/` folder and test them:
   python scripts/run_pipeline.py real_data/<filename>
================================================================================
"""
    REAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Safe printing under cp1252 Windows console
    try:
        print(guide)
    except UnicodeEncodeError:
        print(guide.encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    print_manual_download_guide()
