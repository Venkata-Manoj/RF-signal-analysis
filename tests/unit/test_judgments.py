"""Tests for the TypeSafe-pattern typed judgments (offline, deterministic)."""

import numpy as np

from rf_analyzer.core import judgments as judgments_mod
from rf_analyzer.core.correlator import hex_to_bits


def test_confidence_peaked_is_one_and_flat_is_zero():
    assert judgments_mod.confidence_from_probs(np.array([1.0, 0.0, 0.0])) == 1.0
    assert judgments_mod.confidence_from_probs(np.array([1 / 3] * 3)) == 0.0


def test_route_thresholds():
    assert judgments_mod.route(0.9) == "act"
    assert judgments_mod.route(0.6) == "confirm"
    assert judgments_mod.route(0.2) == "escalate"


def test_modulation_choice_clean_bpsk_is_confident():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=2000, dtype=np.uint8)
    symbols = (2.0 * bits.astype(float) - 1.0).astype(np.complex64)
    ans = judgments_mod.modulation_choice(symbols)
    assert ans.choice == "BPSK"
    assert ans.confidence > 0.5
    assert abs(sum(ans.probabilities.values()) - 1.0) < 1e-6


def test_modulation_choice_tone_never_confident_fsk():
    n = 20000
    t = np.arange(n) / 100000.0
    tone = np.exp(1j * 2 * np.pi * 10000.0 * t).astype(np.complex64)
    ans = judgments_mod.modulation_choice(tone)
    assert not (ans.choice in ("2-FSK", "4-FSK") and ans.confidence > 0.35)


def test_header_noul_real_sync_high_and_noise_low():
    sync = hex_to_bits("0x1ACFFC1D")
    rng = np.random.default_rng(3)
    prefix = rng.integers(0, 2, size=500, dtype=np.uint8)
    suffix = rng.integers(0, 2, size=500, dtype=np.uint8)
    stream = np.concatenate([prefix, sync, suffix])
    assert judgments_mod.header_noul(stream).noul >= 0.9
    noise = rng.integers(0, 2, size=20000, dtype=np.uint8)
    assert judgments_mod.header_noul(noise).noul < 0.5


def test_quality_score_ordering_and_flat_on_none():
    excellent = judgments_mod.quality_score(2.0)
    bad = judgments_mod.quality_score(45.0)
    assert excellent.score < bad.score
    flat = judgments_mod.quality_score(None)
    assert flat.confidence == 0.0


def test_decode_noul_only_on_crc():
    assert judgments_mod.decode_noul(validated=True, crc_pass=True).noul == 0.99
    assert judgments_mod.decode_noul(validated=False, crc_pass=None).noul < 0.5


def test_ask_all_keys_and_json_safe():
    import json

    rng = np.random.default_rng(4)
    bits = rng.integers(0, 2, size=1000, dtype=np.uint8)
    syms = (2.0 * bits.astype(float) - 1.0).astype(np.complex64)
    out = judgments_mod.ask_all(
        syms,
        bits,
        sync_word="0x1ACFFC1D",
        evm_percent=5.0,
        validated=False,
        crc_pass=None,
    )
    assert set(out) == {
        "modulation",
        "signal_quality",
        "decode_trust",
        "header_present",
    }
    json.dumps(
        {
            "modulation": {
                "choice": out["modulation"].choice,
                "probabilities": dict(out["modulation"].probabilities),
                "confidence": out["modulation"].confidence,
            }
        },
        allow_nan=False,
    )
