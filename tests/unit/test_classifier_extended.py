"""Extended classifier: spectral features, rejects, ML head, and bursts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from rf_analyzer.core import classifier as clf
from rf_analyzer.core import waveform as waveform_mod
from rf_analyzer.core.io import load_iq, load_wav
from rf_analyzer.pipeline import _classify_modulation, analyze_file

SAMPLE_RATE = 1_000_000.0
BURST_LENGTHS = (512, 896, 3000, 6000)
BURST_PADS = (0, 800, 2048, 8192)
BURST_SNRS = (6.0, 10.0, 15.0, 20.0)
DIGITAL_SCHEMES = ("BPSK", "QPSK", "8PSK", "16-QAM", "64-QAM", "2-FSK", "4-FSK")


def _burst(
    scheme: str, n_samples: int, pad: int, snr_db: float, seed: int
) -> np.ndarray:
    """Make a clean burst of an exact sample length inside AWGN."""
    width = waveform_mod.bits_per_symbol(scheme)
    n_bits = max(width, round(n_samples * width / waveform_mod.DEFAULT_SPS))
    samples, _, _ = waveform_mod.burst_in_noise(
        scheme,
        n_bits,
        snr_db=snr_db,
        lead=pad,
        tail=pad,
        seed=seed,
    )
    return samples


def _tone(n: int = 48_000, freq: float = 10_000.0) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    return np.exp(1j * 2.0 * np.pi * freq * t).astype(np.complex64)


def _am(n: int = 48_000) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    wave = (1.0 + 0.8 * np.sin(2.0 * np.pi * 1000.0 * t)) * np.exp(
        1j * 2.0 * np.pi * 8000.0 * t
    )
    wave = wave / np.sqrt(float(np.mean(np.abs(wave) ** 2)))
    return wave.astype(np.complex64)


def _fm(n: int = 48_000) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    return np.exp(
        1j * (2.0 * np.pi * 8000.0 * t + 5.0 * np.sin(2.0 * np.pi * 1200.0 * t))
    ).astype(np.complex64)


def _audio(n: int = 48_000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    wave = np.zeros(n)
    for freq in (440.0, 880.0, 1320.0):
        wave = wave + np.sin(2.0 * np.pi * freq * t + rng.uniform(0, 6.28))
    wave = wave + 0.1 * rng.standard_normal(n)
    wave = wave / np.sqrt(float(np.mean(wave**2)))
    return wave.astype(np.complex64)  # real-only, like mono .wav speech


def test_feature_extractor_keys_and_finite():
    samples, _ = waveform_mod.synth("QPSK", 3000, seed=0)
    feats = clf.extract_features(samples, SAMPLE_RATE)
    assert list(feats.keys()) == list(clf.FEATURE_NAMES)
    assert all(np.isfinite(v) for v in feats.values())
    vec = clf.feature_vector(samples, SAMPLE_RATE)
    assert vec.shape == (len(clf.FEATURE_NAMES),)


def test_feature_extractor_short_input():
    feats = clf.extract_features(np.array([1.0 + 1j]), SAMPLE_RATE)
    assert set(feats.keys()) == set(clf.FEATURE_NAMES)
    assert all(np.isfinite(v) for v in feats.values())


def test_rules_64qam_adjacent():
    # Fallback heuristic: clean 64-QAM lands on a QAM label (16/64 overlap
    # under seed variance); the ML head owns the exact split.
    samples, _ = waveform_mod.synth("64-QAM", 6000, seed=3)
    label, conf, _ = clf.classify_modulation(
        samples, sample_rate=SAMPLE_RATE, use_ml=False
    )
    assert label in ("64-QAM", "16-QAM")
    assert conf <= 0.65


def test_rules_4fsk_clean():
    samples, _ = waveform_mod.synth("4-FSK", 6000, seed=3)
    label, conf, _ = clf.classify_modulation(
        samples, sample_rate=SAMPLE_RATE, use_ml=False
    )
    assert label == "4-FSK"
    assert conf >= 0.8  # symbol period recovered: corroborated


def test_rules_am_fm_tone_clean():
    for sig, want in ((_am(), "AM"), (_fm(), "FM"), (_tone(), "TONE")):
        label, conf, _ = clf.classify_modulation(
            sig, sample_rate=SAMPLE_RATE, use_ml=False
        )
        assert label == want, f"expected {want}, got {label}"


def test_tone_never_confident_fsk():
    for sig in (_tone(), _tone(freq=-7000.0)):
        for use_ml in (False, True):
            label, conf, _ = clf.classify_modulation(
                sig, sample_rate=SAMPLE_RATE, use_ml=use_ml
            )
            assert not (label in clf.FSK_LABELS and conf > 0.5), (
                label,
                conf,
            )


def test_audio_never_confident_fsk():
    for seed in (0, 1):
        sig = _audio(seed=seed)
        for use_ml in (False, True):
            label, conf, _ = clf.classify_modulation(
                sig, sample_rate=SAMPLE_RATE, use_ml=use_ml
            )
            assert not (label in clf.FSK_LABELS and conf > 0.5), (
                label,
                conf,
            )


def test_noisy_tone_capped_not_confident():
    rng = np.random.default_rng(0)
    sig = _tone()
    noise = rng.standard_normal(sig.size) + 1j * rng.standard_normal(sig.size)
    sig_noisy = (sig + noise * np.sqrt(0.1)).astype(np.complex64)  # ~10 dB
    label, conf, _ = clf.classify_modulation(sig_noisy, sample_rate=SAMPLE_RATE)
    assert not (label in clf.FSK_LABELS and conf > 0.5), (label, conf)


def test_wav_iq_feature_parity(tmp_path):
    samples, _ = waveform_mod.synth("16-QAM", 3000, seed=5)
    iq_path = tmp_path / "cap.iq"
    np.asarray(samples, dtype=np.complex64).tofile(iq_path)
    back_iq = load_iq(iq_path, dtype="complex64")
    wav_path = tmp_path / "cap.wav"
    stereo = np.column_stack([samples.real, samples.imag]).astype(np.float32)
    sf.write(str(wav_path), stereo, int(SAMPLE_RATE), subtype="FLOAT")
    back_wav, _ = load_wav(wav_path)
    vec_iq = clf.feature_vector(back_iq, SAMPLE_RATE)
    vec_wav = clf.feature_vector(back_wav, SAMPLE_RATE)
    assert np.allclose(vec_iq, vec_wav, atol=1e-4)


def test_ml_model_artifact_present():
    assert (
        clf.MODEL_PATH.exists()
    ), "ml_model.npz missing -- run scripts/train_classifier.py"
    assert clf._load_model() is not None


def test_ml_head_clean_digital():
    for scheme in ("BPSK", "QPSK", "8PSK", "16-QAM", "64-QAM", "2-FSK", "4-FSK"):
        samples, _ = waveform_mod.synth(scheme, 6000, seed=11)
        label, conf, _ = clf.classify_with_ml(samples, SAMPLE_RATE)
        assert label == scheme, f"expected {scheme}, got {label}"
        assert 0.0 <= conf <= clf.ML_MAX_CONFIDENCE


def test_ml_heldout_accuracy():
    """Held-out digital captures stay above 90% across the full 0--20 dB grid."""
    cases: list[tuple[str, np.ndarray]] = []
    for scheme in DIGITAL_SCHEMES:
        for snr in (0.0, 5.0, 10.0, 15.0, 20.0):
            for seed in (300, 301):
                samples, _ = waveform_mod.synth(scheme, 3000, seed=seed, ebn0_db=snr)
                cases.append((scheme, samples))

    correct = 0
    for want, sig in cases:
        label, conf, _ = clf.classify_modulation(sig, sample_rate=SAMPLE_RATE)
        if label in clf.FSK_LABELS and conf > 0.5:
            assert want in clf.FSK_LABELS, f"{want} -> confident {label}"
        correct += label == want
    accuracy = correct / len(cases)
    assert accuracy >= 0.90, f"held-out digital accuracy {accuracy:.3f} < 0.90"

    # Analog rejects remain a separate honesty check; a correct digital row
    # must not be obtained by turning every low-SNR capture into a digital
    # label, and no analog row may become confident FSK.
    analogs = {"AM": _am(), "FM": _fm(), "TONE": _tone(), "AUDIO": _audio()}
    for name, sig in analogs.items():
        label, conf, _ = clf.classify_modulation(sig, sample_rate=SAMPLE_RATE)
        assert not (label in clf.FSK_LABELS and conf > 0.5), (name, label, conf)
        rng = np.random.default_rng(3)
        noise = rng.standard_normal(sig.size) + 1j * rng.standard_normal(sig.size)
        noisy = (sig + noise * np.sqrt(0.1)).astype(np.complex64)
        label, conf, _ = clf.classify_modulation(noisy, sample_rate=SAMPLE_RATE)
        assert not (label in clf.FSK_LABELS and conf > 0.5), (name, label, conf)


def test_ml_heldout_burst_matrix_accuracy():
    """Short digital bursts in noise stay digital across the requested matrix."""
    cases: list[tuple[str, np.ndarray]] = []
    for scheme in DIGITAL_SCHEMES:
        for n_samples in BURST_LENGTHS:
            for pad in BURST_PADS:
                for snr_db in BURST_SNRS:
                    # Fresh from both the full-capture and training burst seeds.
                    cases.append(
                        (scheme, _burst(scheme, n_samples, pad, snr_db, seed=4000))
                    )

    correct = 0
    for want, sig in cases:
        label, conf, _ = clf.classify_modulation(sig, sample_rate=SAMPLE_RATE)
        assert 0.0 <= conf <= clf.ML_MAX_CONFIDENCE
        if label in clf.FSK_LABELS and conf > 0.35:
            # A public FSK result above the uncorroborated cap must have
            # independent symbol-period evidence.
            assert clf._recovered_period(sig) > 1, (want, label, conf)
        correct += label == want
    accuracy = correct / len(cases)
    assert accuracy >= 0.90, f"held-out burst accuracy {accuracy:.3f} < 0.90"


def test_short_bpsk_burst_is_not_reported_as_audio():
    """Pin the original 896-sample/800-sample-padding regression directly."""
    sig = _burst("BPSK", n_samples=896, pad=800, snr_db=6.0, seed=7)
    label, conf, _ = clf.classify_modulation(sig, sample_rate=SAMPLE_RATE)
    assert label == "BPSK", (label, conf)


def test_ml_fallback_on_missing_model(monkeypatch, tmp_path):
    saved = dict(clf._MODEL_STATE)
    monkeypatch.setattr(clf, "MODEL_PATH", Path(tmp_path) / "nope.npz")
    clf._MODEL_STATE["model"] = None
    clf._MODEL_STATE["missing"] = False
    try:
        with pytest.raises(RuntimeError):
            clf.classify_with_ml(np.ones(64, dtype=np.complex64))
        # Public entry still answers via rules/HOC fallback.
        label, conf, alts = clf.classify_modulation(np.ones(64, dtype=np.complex64))
        assert isinstance(label, str) and 0.0 <= conf <= 1.0 and alts
    finally:
        clf._MODEL_STATE.update(saved)


def test_pipeline_tone_capped():
    label, conf, _ = _classify_modulation(_tone(n=10_000), SAMPLE_RATE)
    assert not (label in clf.FSK_LABELS and conf > 0.5), (label, conf)


def test_pipeline_analog_report_capped(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sig = _tone(n=10_000)
    stereo = np.column_stack([sig.real, sig.imag]).astype(np.float32)
    wav_path = tmp_path / "tone_probe.wav"
    sf.write(str(wav_path), stereo, int(SAMPLE_RATE), subtype="FLOAT")
    report = analyze_file({"file_path": str(wav_path), "modulation": "auto"})
    assert report["errors"] == []
    assert report["modulation"]["confidence"] <= 0.35
    assert any("placeholder" in w for w in report["warnings"])
