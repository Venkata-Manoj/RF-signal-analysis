"""Unit tests for GPS L1 C/A acquisition + despread (core.gps).

Covers the PRN generator against the ICD LFSR definition (SV1 prefix pin,
balance, periodic Gold bounds over all 32 SVs), synthetic-signal acquisition
(known SV/Doppler/code phase recovered; pure noise abstains), despread signs
and C/N0, and the decline paths. No real capture needed here.
"""

from __future__ import annotations

import numpy as np
import pytest

from rf_analyzer.core import gps

FS = 4_000_000.0
BLOCK = int(FS * gps.CODE_PERIOD_S)  # 4000 samples per 1 ms


def _synthetic(sv: int, doppler_hz: float, phase_samples: int, *, seed: int = 0, noise_std: float = 2.0):
    """10 ms GPS-like baseband: PRN code x carrier at known Doppler + AWGN."""
    rng = np.random.default_rng(seed)
    code = gps._code_pm1_upsampled(sv, FS, BLOCK)
    sig = np.tile(np.roll(code, phase_samples), 10)
    t = np.arange(sig.size) / FS
    sig = sig * np.exp(1j * 2.0 * np.pi * doppler_hz * t)
    noise = (rng.standard_normal(sig.size) + 1j * rng.standard_normal(sig.size)) / np.sqrt(2.0) * noise_std
    return (sig + noise).astype(np.complex64)


def test_ca_code_length_and_balance():
    for sv in (1, 7, 19, 32):
        code = gps.ca_code(sv)
        assert code.shape == (1023,)
        assert code.dtype == np.int8
        assert set(np.unique(code).tolist()) <= {0, 1}
        # Gold codes are near-balanced: 512 ones / 511 zeros (or the reverse).
        assert int(code.sum()) in (511, 512)


def test_ca_code_sv1_prefix_pinned_to_lfsr_definition():
    # First chips out of the G1/G2 registers (all-ones start, IS-GPS-200 taps):
    # G1>>..., G2(2,6)>>... for SV1. Pinned so a wiring change fails loudly.
    assert gps.ca_code(1)[:10].tolist() == [1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
    assert "".join(map(str, gps.ca_code(1)[:20].tolist())) == "11001000001110010100"


def test_ca_code_periodic_gold_bounds_over_all_svs():
    # Circular (periodic) autocorrelation sidelobes and cross-correlations of
    # 1023-chip Gold codes are bounded by t(10) = 65. Linear (aperiodic)
    # sidelobes are larger (~78), so the bound below only holds circularly.
    pm = {sv: np.where(gps.ca_code(sv) > 0, 1.0, -1.0) for sv in range(1, 33)}
    for sv, code in pm.items():
        circ = np.abs(np.fft.ifft(np.abs(np.fft.fft(code)) ** 2))
        assert circ[0] == pytest.approx(1023.0)
        assert float(np.max(circ[1:])) <= 65.0 + 1e-6, f"SV{sv} autosidelobe"
    for a in (1, 7, 19):
        for b in (2, 8, 20, 32):
            if a == b:
                continue
            cross = np.abs(np.fft.ifft(np.fft.fft(pm[a]) * np.conj(np.fft.fft(pm[b]))))
            assert float(np.max(cross)) <= 65.0 + 1e-6, f"SV{a}xSV{b}"


def test_ca_code_rejects_bad_sv_and_chips():
    for bad in (0, 33, -1):
        try:
            gps.ca_code(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"sv={bad} should raise")
    try:
        gps.ca_code(1, chips=0)
    except ValueError:
        pass
    else:
        raise AssertionError("chips=0 should raise")


def test_ca_code_tiling_and_determinism():
    assert np.array_equal(gps.ca_code(5), gps.ca_code(5))
    long_code = gps.ca_code(5, chips=2046)
    assert long_code.shape == (2046,)
    assert np.array_equal(long_code[:1023], gps.ca_code(5))
    assert np.array_equal(long_code[1023:], gps.ca_code(5))
    assert not np.array_equal(gps.ca_code(1), gps.ca_code(2))


def test_acquire_finds_synthetic_signal():
    x = _synthetic(sv=7, doppler_hz=1500.0, phase_samples=1234)
    res = gps.acquire(x, FS, svs=list(range(1, 9)))
    hit = res[7]
    assert hit["acquired"] is True
    assert abs(float(hit["doppler_hz"]) - 1500.0) <= 250.0
    expected_chips = 1234 * 1023.0 / BLOCK
    assert abs(float(hit["code_phase_chips"]) - expected_chips) <= 2.0
    assert int(hit["code_phase_samples"]) == 1234
    # The true SV owns the global metric maximum; nothing else crosses.
    assert hit["metric"] == max(float(r["metric"]) for r in res.values())
    for sv, entry in res.items():
        if sv != 7:
            assert entry["acquired"] is False, f"SV{sv} false alarm: {entry}"


def test_acquire_abstains_on_pure_noise():
    rng = np.random.default_rng(3)
    noise = ((rng.standard_normal(40000) + 1j * rng.standard_normal(40000)) / np.sqrt(2.0)).astype(np.complex64)
    res = gps.acquire(noise, FS, svs=list(range(1, 9)))
    assert all(entry["acquired"] is False for entry in res.values())
    assert all(entry["reason"] == "" for entry in res.values())  # the search ran
    assert max(float(e["metric"]) for e in res.values()) < gps.ACQUISITION_THRESHOLD


def test_acquire_declines_when_undersampled():
    rng = np.random.default_rng(4)
    x = ((rng.standard_normal(4000) + 1j * rng.standard_normal(4000)) / np.sqrt(2.0)).astype(np.complex64)
    res = gps.acquire(x, 100_000.0, svs=[1, 2])
    assert all(entry["acquired"] is False for entry in res.values())
    assert all("sample rate" in entry["reason"] for entry in res.values())


def test_despread_synthetic_signs_and_cn0():
    x = _synthetic(sv=7, doppler_hz=1500.0, phase_samples=1234)
    res = gps.acquire(x, FS, svs=[7])
    hit = res[7]
    assert hit["acquired"] is True
    out = gps.despread(x, FS, 7, float(hit["doppler_hz"]), int(hit["code_phase_samples"]))
    assert out["n_blocks"] == 10
    # No nav-bit flips in the synthetic: every prompt sign agrees.
    assert len(out["prompt_signs"]) == 10
    assert all(s == out["prompt_signs"][0] for s in out["prompt_signs"])
    assert out["cn0_dbhz"] is not None and 30.0 < float(out["cn0_dbhz"]) < 70.0


def test_despread_short_input_declines():
    out = gps.despread(np.ones(100, dtype=np.complex64), FS, 1, 0.0, 0)
    assert out["n_blocks"] == 0
    assert out["cn0_dbhz"] is None
    assert out["prompt_signs"] == []
    assert out["reason"] != ""


def test_desspread_alias_matches_despread():
    assert gps.desspread is gps.despread


def test_cn0_moment_estimator_noise_floor():
    rng = np.random.default_rng(5)
    prompt = (rng.standard_normal(10) + 1j * rng.standard_normal(10)) / np.sqrt(2.0)
    cn0 = gps.estimate_cn0_dbhz(prompt)
    assert cn0 is not None and cn0 < 30.0  # ~20 dB-Hz for 10 noise blocks
    assert gps.estimate_cn0_dbhz(np.ones(1)) is None
