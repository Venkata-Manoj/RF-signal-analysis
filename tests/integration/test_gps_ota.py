"""GPS L1 over-the-air test on the bundled PySDR capture.

Measured reality (deterministic: same file, FFT-based search, no randomness):
10 ms at 4 MHz -> 10 non-coherent 1 ms blocks. Full 32-SV x 41-Doppler search
(~1.5 s) acquires SVs {2, 11, 12, 22, 25, 31, 32}; every other SV sits at
metric <= ~1.4 against the 2.5 gate, i.e. wide separation, no borderline
rejections. Strongest: SV12 (metric ~18, C/N0 ~43.6 dB-Hz) and SV31 (metric
~15, C/N0 ~42 dB-Hz). SV2 is the weakest acquisition (metric ~3.3) and must
escalate via a marginal warning rather than read as confident.

If this file ever yields nothing (different recording, broken fetch), the
honest assertions below still hold: attempted, empty acquired list, a decline
reason, errors empty, and no invented payload. What is NEVER asserted is a
navigation message -- a 10 ms capture cannot yield subframes/ephemeris.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from rf_analyzer.core import gps
from rf_analyzer.pipeline import analyze_file

REAL_DATA = Path(__file__).resolve().parents[2] / "real_data"
CAPTURE = REAL_DATA / "gps_l1_4mhz_cf32.iq"
FS = 4_000_000.0

#: Full measured acquisition set on the bundled capture (see module docstring).
MEASURED_ACQUIRED = {2, 11, 12, 22, 25, 31, 32}


def _require_capture() -> np.ndarray:
    if not CAPTURE.is_file():
        pytest.skip(f"missing real capture: {CAPTURE} (run scripts/fetch_real_data.py)")
    return np.fromfile(str(CAPTURE), dtype=np.complex64)


def test_ota_acquisition_table_matches_measured_reality():
    samples = _require_capture()
    started = time.perf_counter()
    res = gps.acquire(samples, FS)
    elapsed = time.perf_counter() - started
    assert elapsed < 60.0, f"32-SV search took {elapsed:.1f}s (budget 60s)"

    acquired = {sv for sv, r in res.items() if r["acquired"]}
    assert MEASURED_ACQUIRED <= acquired, f"missing SVs: {MEASURED_ACQUIRED - acquired} (got {sorted(acquired)})"
    refused = {sv for sv, r in res.items() if not r["acquired"]}
    assert all(float(res[sv]["metric"]) < gps.ACQUISITION_THRESHOLD for sv in refused)

    # Strong, far-from-gate detections pinned exactly (discrete Doppler bins
    # and sample code phases do not move for a deterministic search).
    sv12 = res[12]
    assert sv12["doppler_hz"] == -250.0
    assert sv12["code_phase_samples"] == 3408
    assert float(sv12["metric"]) > 10.0
    sv31 = res[31]
    assert sv31["doppler_hz"] == 5000.0
    assert sv31["code_phase_samples"] == 1812
    assert float(sv31["metric"]) > 10.0

    # Plausible C/N0 (35-55 dB-Hz) for the two healthy SVs, with signs output.
    for sv in (12, 31):
        out = gps.despread(samples, FS, sv, float(res[sv]["doppler_hz"]), int(res[sv]["code_phase_samples"]))
        assert out["n_blocks"] == 10
        assert len(out["prompt_signs"]) == 10
        assert out["cn0_dbhz"] is not None and 35.0 <= float(out["cn0_dbhz"]) <= 55.0, f"SV{sv} C/N0={out['cn0_dbhz']}"


def test_pipeline_gps_attempt_reports_evidence_not_messages():
    _require_capture()
    report = analyze_file(
        {
            "file_path": str(CAPTURE),
            "sample_rate": FS,
            "iq_format": "auto",
            "modulation": "auto",
            "gps": True,
        }
    )
    assert report["errors"] == []
    block = report["gps"]
    assert block["attempted"] is True
    acquired = {e["sv"] for e in block["acquired"]}
    assert MEASURED_ACQUIRED <= acquired
    # Evidence fields present; message fields absent by construction.
    assert block["prompt_signs"] != []
    assert block["cn0_dbhz"] is not None
    assert "decoded" not in block and "message" not in block
    assert "no message" in block["nav_note"]
    # No payload invented from the spread-spectrum capture.
    assert (report["payload"].get("decoded") or {}).get("available") is False
    # Low confidence escalates: SV2 (~3.3 vs gate 2.5) must read as marginal.
    marginal = [e for e in block["acquired"] if e["metric"] < 1.5 * block["threshold"]]
    assert marginal, "expected at least the weak SV2 acquisition to read marginal"
    for entry in marginal:
        assert any(f"SV{entry['sv']}" in w and "marginal" in w for w in report["warnings"]), report["warnings"]
    # Untouched semantics: the narrowband path is exactly what gps=false sees.
    plain = analyze_file(
        {
            "file_path": str(CAPTURE),
            "sample_rate": FS,
            "iq_format": "auto",
            "modulation": "auto",
        }
    )
    assert report["display"]["samples_per_symbol"] == plain["display"]["samples_per_symbol"]
    assert report["demodulation"]["mode"] == plain["demodulation"]["mode"]


def test_pipeline_suggests_gps_when_off():
    _require_capture()
    report = analyze_file(
        {
            "file_path": str(CAPTURE),
            "sample_rate": FS,
            "iq_format": "auto",
            "modulation": "auto",
        }
    )
    assert report["errors"] == []
    assert report["gps"]["attempted"] is False
    assert report["gps"]["acquired"] == []
    assert any("possible spread-spectrum; re-run with gps=true" in w for w in report["warnings"]), report["warnings"]


def test_pipeline_gps_declines_gracefully_on_noise(tmp_path):
    rng = np.random.default_rng(9)
    noise = ((rng.standard_normal(40000) + 1j * rng.standard_normal(40000)) / np.sqrt(2.0)).astype(np.complex64)
    path = tmp_path / "noise4m.iq"
    noise.tofile(path)
    report = analyze_file(
        {
            "file_path": str(path),
            "sample_rate": FS,
            "iq_format": "complex64",
            "modulation": "auto",
            "decode": False,
            "gps": True,
        }
    )
    assert report["errors"] == []
    block = report["gps"]
    assert block["attempted"] is True
    assert block["acquired"] == []
    assert block["prompt_signs"] == []
    assert block["cn0_dbhz"] is None
    assert any("no SV crossed" in w for w in report["warnings"]), report["warnings"]
    assert (report["payload"].get("decoded") or {}).get("available") is False
