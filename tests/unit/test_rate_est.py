"""Fused symbol/sampling-rate estimation, and the waveform generator it stands on.

Two things are under test here and they are deliberately in one file because the
second is meaningless without the first:

* ``core/waveform.py`` -- the oversampled, pulse-shaped, Gray-coded generator
  the extended demodulators will be tested against. If its constellation
  mapping is not invertible, or its pulse shaping has ISI at the symbol
  instants, every BER number measured later is fiction.
* ``core/rate_est.py`` -- the fused rate estimator that replaces
  ``bandwidth * 2.2`` and the single-periodogram ``|x|^2`` line.

The honesty contract these tests exist to enforce: **0.0 means "not
measurable"**. The estimator must produce a real rate for a real modulation at
0 dB, and must produce 0.0 -- with a warning naming what was tried -- for AWGN,
an unmodulated tone, a random walk and real speech. Speech is the hard case and
the reason the fusion refuses on disagreement rather than picking a winner: a
voiced sound genuinely *is* cyclostationary at its pitch, so the detector is
right and the answer is still "no symbol rate here".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rf_analyzer.core import waveform as wf
from rf_analyzer.core.dsp import estimate_fsk_symbol_period
from rf_analyzer.core.rate_est import (
    AGREEMENT_TOLERANCE,
    CONFIDENCE_CORROBORATED,
    CONFIDENCE_DIRECT,
    cyclostationary_profile,
    cyclostationary_symbol_rate,
    envelope_symbol_rate,
    estimate_sampling_rate_fused,
    estimate_symbol_rate_fused,
)

#: Sample rate for every synthetic case. 1 MHz with 8 samples per symbol puts
#: the symbol rate at a round 125 kHz, far enough from the search bounds that a
#: sign error in the cycle-frequency axis cannot accidentally look correct.
SAMPLE_RATE = 1_000_000.0
SPS = 8
TRUE_SYMBOL_RATE = SAMPLE_RATE / SPS

#: Bits per scheme for the *cyclic* tests, chosen so every capture lands at
#: 8-16 k samples. That keeps the cyclic periodogram's FFT at 1024 rather than
#: 2048 and the whole file under a minute -- measured, not guessed: 16 k samples
#: takes 0.15 s and 32 k takes 1.5 s, because the per-``alpha`` cost is linear
#: in the number of segments times the FFT size.
BITS_FOR_SCHEME = {
    "BPSK": 2000,
    "QPSK": 2000,
    "8PSK": 3000,
    "16-QAM": 4000,
    "64-QAM": 6000,
    "2-FSK": 4000,
    "4-FSK": 4000,
}

#: Bits per scheme for the envelope and fusion tests. The ``|x|^2`` line of a
#: dense constellation is weak, and 6000 bits of 64-QAM is only 1000 symbols --
#: at that length its prominence (4.6) overlaps AWGN's (2.9) and no threshold
#: separates them. Longer captures are free here because the envelope estimator
#: is Welch-averaged and costs ~10 ms regardless.
BITS_FOR_FUSION = {
    "BPSK": 4000,
    "QPSK": 4000,
    "8PSK": 6000,
    "16-QAM": 12000,
    "64-QAM": 16000,
}

ROOT = Path(__file__).resolve().parents[2]
AUDIO_DIR = ROOT / "sample_data"
AUDIO_FILES = (
    "sample-3s-8khz.wav",
    "sample-3s-16khz.wav",
    "sample-3s-48khz.wav",
    "sample-3s-stereo.wav",
    "sample-speech-1m.wav",
)


def _synth(scheme: str, snr_db: float | None, seed: int = 1, sps: int = SPS):
    return wf.synth(
        scheme,
        BITS_FOR_SCHEME[scheme],
        sps=sps,
        snr_db=snr_db,
        seed=seed,
        sample_rate=SAMPLE_RATE,
    )


def _synth_long(scheme: str, snr_db: float | None, seed: int = 1, sps: int = SPS):
    """A capture long enough for the weak envelope line of a dense constellation."""
    return wf.synth(
        scheme,
        BITS_FOR_FUSION[scheme],
        sps=sps,
        snr_db=snr_db,
        seed=seed,
        sample_rate=SAMPLE_RATE,
    )


def _voiced_speech_surrogate(
    n: int = 16000, f0: float = 120.0, fs: float = 16000.0, seed: int = 0
):
    """A pitched harmonic stack with a syllable envelope.

    Stands in for real speech when ``sample_data/`` is not present. Voiced
    speech is a periodic glottal pulse train shaped by a formant filter, so it
    has genuine cyclostationarity at ``f0`` -- which is exactly the false
    positive the fusion has to survive. Pure noise would not test anything.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    signal = np.zeros(n)
    for harmonic in range(1, 25):
        if harmonic * f0 >= fs / 2:
            break
        amplitude = 1.0 / harmonic
        signal += amplitude * np.sin(
            2 * np.pi * harmonic * f0 * t + rng.uniform(0, 2 * np.pi)
        )
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)  # ~3 Hz syllable rate
    signal = signal * envelope
    signal += 0.02 * rng.standard_normal(n)
    return (signal / np.max(np.abs(signal))).astype(np.float64)


# --------------------------------------------------------------------------- #
# Waveform generator
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("scheme", wf.LINEAR_SCHEMES)
def test_constellation_round_trips(scheme: str) -> None:
    """Map -> decide must return the exact bits that went in.

    Without this, a BER measured against the generator's own output would
    report the mapping's own mistakes as channel errors.
    """
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=4096, dtype=np.uint8)
    symbols = wf.bits_to_symbols(bits, scheme)
    recovered = wf.symbols_to_bits(symbols, scheme)
    assert recovered.size == symbols.size * wf.bits_per_symbol(scheme)
    assert np.array_equal(bits[: recovered.size], recovered)


@pytest.mark.parametrize("scheme", wf.LINEAR_SCHEMES)
def test_constellation_has_unit_average_power(scheme: str) -> None:
    """Equal average power, so one ``snr_db`` means the same for every scheme."""
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, size=240_000, dtype=np.uint8)
    symbols = wf.bits_to_symbols(bits, scheme)
    assert np.mean(np.abs(symbols) ** 2) == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize("scheme", wf.LINEAR_SCHEMES)
def test_gray_coding_makes_neighbours_one_bit_apart(scheme: str) -> None:
    """Adjacent constellation points differ in exactly one bit.

    This is what makes the BER numbers meaningful: without Gray coding a single
    symbol error costs several bits, and a "BER < 0.01" claim would be measuring
    the labelling rather than the receiver. Built from the *complete* bit space
    rather than a random draw, so no point is missing from the test.
    """
    width = wf.bits_per_symbol(scheme)
    all_bits = np.array(
        [
            [(value >> shift) & 1 for shift in range(width - 1, -1, -1)]
            for value in range(1 << width)
        ],
        dtype=np.uint8,
    ).ravel()
    symbols = wf.bits_to_symbols(all_bits, scheme)
    assert symbols.size == 1 << width

    distances = np.abs(symbols[:, None] - symbols[None, :])
    np.fill_diagonal(distances, np.inf)
    minimum = distances.min()

    for i in range(symbols.size):
        for j in range(i + 1, symbols.size):
            if abs(distances[i, j] - minimum) < 1e-9:
                a = all_bits[i * width : (i + 1) * width]
                b = all_bits[j * width : (j + 1) * width]
                assert int(np.count_nonzero(a != b)) == 1, (scheme, a, b)


def test_rrc_has_no_isi_at_the_symbol_instants() -> None:
    """A matched-filter pair must be a Nyquist pulse: zero at every other symbol.

    If this fails, the receiver sees inter-symbol interference that is not
    channel noise, and every BER floor measured later is an artefact of the
    pulse shape.
    """
    taps = wf.rrc_taps(SPS, 0.35, 8)
    assert np.sum(taps**2) == pytest.approx(1.0, rel=1e-9)

    matched = np.convolve(taps, taps)
    centre = int(np.argmax(np.abs(matched)))
    peak = matched[centre]
    for offset in range(1, 6):
        for index in (centre - offset * SPS, centre + offset * SPS):
            assert abs(matched[index]) / abs(peak) < 1e-4


def test_mfsk_reproduces_framing_2fsk() -> None:
    """``modulate_mfsk`` and ``framing.modulate_2fsk`` must be one waveform.

    The project already paid for two copies of the FSK phase convention once
    (``info.md`` §15.1); this pins the new generator to the existing one so the
    generator and the transmit path cannot drift apart again.
    """
    from rf_analyzer.core.framing import modulate_2fsk

    bits = np.array([0, 1, 1, 0, 1, 0, 0, 1] * 4, dtype=np.uint8)
    sample_rate = 100_000.0
    symbol_duration = 0.001
    sps = int(sample_rate * symbol_duration)
    deviation = 5_000.0

    reference = modulate_2fsk(
        bits,
        sample_rate,
        symbol_duration=symbol_duration,
        freq_low=-deviation,
        freq_high=deviation,
    )
    # h = 2 * f_dev * T = 2 * 5000 * 0.001 = 10.
    candidate = wf.modulate_mfsk(
        bits, 2, sps=sps, modulation_index=10.0, sample_rate=sample_rate
    )
    assert candidate.size == reference.size
    assert np.max(np.abs(candidate - reference)) < 1e-5


@pytest.mark.parametrize("order", [0, 1, 3, 5])
def test_mfsk_rejects_a_non_power_of_two_order(order: int) -> None:
    with pytest.raises(ValueError):
        wf.modulate_mfsk(np.zeros(8, dtype=np.uint8), order=order)


def test_awgn_hits_the_requested_snr() -> None:
    """``snr_db`` must be measured against the signal's own power."""
    rng = np.random.default_rng(3)
    x = wf.bits_to_symbols(rng.integers(0, 2, 4000, dtype=np.uint8), "QPSK")
    for target in (0.0, 10.0, 20.0):
        noisy = wf.add_awgn(x, target, rng)
        measured = 10 * np.log10(
            np.mean(np.abs(x) ** 2) / np.mean(np.abs(noisy - x) ** 2)
        )
        assert measured == pytest.approx(target, abs=1.0)


# --------------------------------------------------------------------------- #
# Cyclostationary symbol-rate estimator
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("scheme", ["BPSK", "QPSK", "8PSK", "16-QAM", "64-QAM"])
@pytest.mark.parametrize("snr_db", [None, 20.0, 10.0, 0.0])
def test_cyclostationary_recovers_the_symbol_rate(
    scheme: str, snr_db: float | None
) -> None:
    """Every linear modulation, from noiseless down to 0 dB, within 2%."""
    samples, _ = _synth(scheme, snr_db)
    result = cyclostationary_symbol_rate(samples, SAMPLE_RATE, max_alpha_hz=200_000.0)
    assert result["found"], f"{scheme} at {snr_db} dB: {result['reason']}"
    assert result["symbol_rate_hz"] == pytest.approx(TRUE_SYMBOL_RATE, rel=0.02)


@pytest.mark.parametrize(
    "label,seed",
    [("awgn", 0), ("awgn", 11), ("awgn", 23)],
)
def test_cyclostationary_declines_on_awgn(label: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 16000
    samples = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    result = cyclostationary_symbol_rate(samples, SAMPLE_RATE)
    assert not result["found"]
    assert result["symbol_rate_hz"] == 0.0


@pytest.mark.parametrize("noise", [0.0, 0.1, 0.5])
def test_cyclostationary_declines_on_an_unmodulated_tone(noise: float) -> None:
    """A tone has no symbol rate, at any noise level.

    This is the case the *first* version of this estimator got wrong: taking the
    maximum of the coherence over frequency is an extreme-value statistic whose
    noise floor is ~4.3/sqrt(K) regardless of the signal, so a tone scored a
    large "peak". It is also why the cycle-frequency search starts four bins
    out -- closer than that, the two shifted bins see the same spectral line and
    the tone looks perfectly coherent with itself.
    """
    rng = np.random.default_rng(5)
    n = 16000
    t = np.arange(n)
    samples = np.exp(1j * 2 * np.pi * 0.05 * t)
    if noise:
        samples = samples + noise * (
            rng.standard_normal(n) + 1j * rng.standard_normal(n)
        )
    result = cyclostationary_symbol_rate(samples, SAMPLE_RATE)
    assert not result["found"], result
    assert result["symbol_rate_hz"] == 0.0


def test_cyclostationary_declines_on_a_random_walk() -> None:
    """A 1/f-dominated signal is the hardest synthetic negative."""
    rng = np.random.default_rng(7)
    samples = np.cumsum(rng.standard_normal(16000)).astype(np.complex128)
    result = cyclostationary_symbol_rate(samples, SAMPLE_RATE)
    assert not result["found"], result


def test_cyclostationary_resolves_the_harmonic() -> None:
    """The reported rate must be the fundamental, not a stronger harmonic.

    A 4-FSK burst was measured with its third harmonic taller than the
    fundamental. Halving once is not enough; the estimator walks the divisors.
    """
    samples, _ = _synth("4-FSK", None)
    result = cyclostationary_symbol_rate(samples, SAMPLE_RATE, max_alpha_hz=200_000.0)
    assert result["found"], result["reason"]
    assert result["symbol_rate_hz"] == pytest.approx(TRUE_SYMBOL_RATE, rel=0.02)


def test_cyclostationary_profile_is_empty_for_a_too_short_capture() -> None:
    """Too short to segment is "no evidence", not a rate of zero."""
    alphas, profile = cyclostationary_profile(
        np.ones(40, dtype=np.complex128), SAMPLE_RATE
    )
    assert alphas.size == 0
    assert profile.size == 0


def test_cyclostationary_profile_null_expectation_is_one() -> None:
    """The profile is normalised so "no cyclic structure" reads as 1.0.

    Every threshold in the estimator is expressed in those units, so if this
    drifts the thresholds silently stop meaning what the docstrings say.
    """
    rng = np.random.default_rng(13)
    n = 16000
    samples = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    _, profile = cyclostationary_profile(samples, SAMPLE_RATE)
    assert profile.size > 100
    assert np.median(profile) == pytest.approx(1.0, abs=0.25)


@pytest.mark.skipif(not AUDIO_DIR.is_dir(), reason="sample_data/ not generated")
def test_cyclostationary_fires_on_real_speech_and_that_is_not_a_bug() -> None:
    """The cyclostationary detector *does* fire on speech, and must keep doing so.

    This test is deliberately inverted from what one would first write. Voiced
    speech is a periodic glottal pulse train, so it genuinely is
    cyclostationary -- at the pitch, 60-400 Hz. The detector reporting a peak
    there is the detector working; the mistake would be to *report that peak as
    a symbol rate*, and that is the fusion's job, not this estimator's. Pinning
    it here stops anyone "fixing" the detector by raising its threshold until
    speech stops firing, which would also stop it finding real low-rate
    signals. See ``test_fusion_declines_on_real_audio`` for the contract that
    actually matters.
    """
    from rf_analyzer.core.io import load_wav

    present = [AUDIO_DIR / name for name in AUDIO_FILES if (AUDIO_DIR / name).is_file()]
    if not present:
        pytest.skip("no bundled audio captures present")
    fired = 0
    for path in present:
        samples, sample_rate = load_wav(str(path))
        flat = np.asarray(samples).ravel()[:16000]
        if cyclostationary_symbol_rate(flat, float(sample_rate))["found"]:
            fired += 1
    assert fired > 0, (
        "no bundled audio produced a cyclostationary peak; either the captures "
        "changed or the detector has been over-damped"
    )


@pytest.mark.skipif(not AUDIO_DIR.is_dir(), reason="sample_data/ not generated")
def test_fusion_declines_on_real_audio() -> None:
    """Real speech must not be reported as a symbol rate. This is the contract."""
    from rf_analyzer.core.io import load_wav

    present = [AUDIO_DIR / name for name in AUDIO_FILES if (AUDIO_DIR / name).is_file()]
    if not present:
        pytest.skip("no bundled audio captures present")
    for path in present:
        samples, sample_rate = load_wav(str(path))
        flat = np.asarray(samples).ravel()[:16000]
        result = estimate_symbol_rate_fused(flat, float(sample_rate))
        assert result["symbol_rate_hz"] == 0.0, f"{path.name}: {result}"
        assert result["confidence"] == 0.0
        assert result["warnings"]


# --------------------------------------------------------------------------- #
# Envelope symbol-rate estimator
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("scheme", ["BPSK", "QPSK", "8PSK", "16-QAM", "64-QAM"])
@pytest.mark.parametrize("snr_db", [None, 20.0, 10.0, 0.0])
def test_envelope_recovers_the_symbol_rate(scheme: str, snr_db: float | None) -> None:
    samples, _ = _synth_long(scheme, snr_db)
    result = envelope_symbol_rate(samples, SAMPLE_RATE, max_rate_hz=200_000.0)
    assert result["found"], f"{scheme} at {snr_db} dB: {result['reason']}"
    assert result["symbol_rate_hz"] == pytest.approx(TRUE_SYMBOL_RATE, rel=0.02)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_envelope_declines_on_awgn(seed: int) -> None:
    """A single periodogram fires on noise about half the time; Welch must not.

    ``dsp.estimate_symbol_rate`` uses one 8192-point periodogram, whose maximum
    sits ~5-6 dB above the median for white noise by construction -- right on
    its 6 dB gate. Welch averaging cuts the per-bin variance to ~1/sqrt(K), so
    the gate is no longer at the noise floor. Five seeds, because the old
    estimator's failure was intermittent and one seed would not catch it.
    """
    rng = np.random.default_rng(seed)
    n = 16000
    samples = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    result = envelope_symbol_rate(samples, SAMPLE_RATE)
    assert not result["found"], f"seed {seed}: {result}"
    assert result["symbol_rate_hz"] == 0.0


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("scheme", ["BPSK", "QPSK", "8PSK", "16-QAM", "64-QAM"])
@pytest.mark.parametrize("snr_db", [None, 20.0, 10.0, 0.0])
def test_fusion_corroborates_a_linear_modulation(
    scheme: str, snr_db: float | None
) -> None:
    """Two independent paths agreeing is what earns a confident number."""
    samples, _ = _synth_long(scheme, snr_db)
    result = estimate_symbol_rate_fused(
        samples, SAMPLE_RATE, bandwidth_estimate=200_000.0
    )
    assert result["method"] == "corroborated", result
    assert result["confidence"] == pytest.approx(CONFIDENCE_CORROBORATED)
    assert result["symbol_rate_hz"] == pytest.approx(TRUE_SYMBOL_RATE, rel=0.02)
    assert result["components"]["envelope"] is not None
    assert result["components"]["cyclostationary"] is not None
    assert result["agreement"] is not None
    assert result["agreement"] < AGREEMENT_TOLERANCE


@pytest.mark.parametrize("label", ["awgn", "tone", "random_walk", "voiced_speech"])
def test_fusion_refuses_to_invent_a_symbol_rate(label: str) -> None:
    """No modulation means no symbol rate -- 0.0, a warning, and no confidence."""
    rng = np.random.default_rng(17)
    n = 16000
    if label == "awgn":
        samples = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    elif label == "tone":
        samples = np.exp(1j * 2 * np.pi * 0.05 * np.arange(n))
    elif label == "random_walk":
        samples = np.cumsum(rng.standard_normal(n)).astype(np.complex128)
    else:
        samples = _voiced_speech_surrogate(n)

    result = estimate_symbol_rate_fused(samples, SAMPLE_RATE)
    assert result["symbol_rate_hz"] == 0.0, result
    assert result["confidence"] == 0.0
    assert result["method"] in ("none", "single", "disagreement")
    assert result["warnings"], "a refusal must say why"


def test_fusion_refuses_when_the_paths_disagree() -> None:
    """Disagreement is evidence that one path is wrong -- so claim nothing.

    This is the rule that stops speech being reported as a symbol rate. Voiced
    speech really is cyclostationary at its pitch, so the cyclostationary path
    is not malfunctioning when it fires; the envelope path simply sees a
    different period, and two estimates that far apart cannot both be right.
    """
    samples = _voiced_speech_surrogate()
    result = estimate_symbol_rate_fused(samples, 16000.0)
    envelope = result["components"]["envelope"]
    cyclostationary = result["components"]["cyclostationary"]
    assert result["symbol_rate_hz"] == 0.0
    if envelope is not None and cyclostationary is not None:
        assert result["method"] == "disagreement"
        assert any("disagree" in warning for warning in result["warnings"])


def test_fusion_refuses_on_a_single_uncorroborated_path() -> None:
    """One detection is a hypothesis, and the report says so rather than guessing."""
    rng = np.random.default_rng(19)
    n = 16000
    samples = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    result = estimate_symbol_rate_fused(samples, SAMPLE_RATE)
    assert result["symbol_rate_hz"] == 0.0
    assert result["confidence"] == 0.0
    assert result["method"] in ("none", "single", "disagreement")


def test_fusion_reports_a_recovered_fsk_period_as_direct() -> None:
    """A recovered period is a measurement, not a detection, and outranks one."""
    samples, _ = wf.synth(
        "2-FSK",
        4000,
        sps=16,
        snr_db=None,
        seed=1,
        sample_rate=SAMPLE_RATE,
        modulation_index=1.0,
    )
    period = estimate_fsk_symbol_period(samples)
    assert period > 1, "the noiseless reference FSK must have a recoverable period"

    result = estimate_symbol_rate_fused(
        samples,
        SAMPLE_RATE,
        bandwidth_estimate=500_000.0,
        fsk_period=period,
        modulation_hint="2-FSK",
    )
    assert result["method"] == "direct"
    assert result["confidence"] == pytest.approx(CONFIDENCE_DIRECT)
    assert result["symbol_rate_hz"] == pytest.approx(SAMPLE_RATE / period, rel=1e-9)


def test_fusion_skips_the_envelope_path_for_fsk() -> None:
    """The |x|^2 line is meaningless for a constant-envelope signal.

    Left in, it reports whatever the noise floor peaks at -- measured at
    395 kHz for a 1 kHz burst.
    """
    samples, _ = wf.synth(
        "2-FSK",
        4000,
        sps=16,
        snr_db=None,
        seed=1,
        sample_rate=SAMPLE_RATE,
        modulation_index=1.0,
    )
    result = estimate_symbol_rate_fused(
        samples, SAMPLE_RATE, bandwidth_estimate=500_000.0, modulation_hint="2-FSK"
    )
    assert result["components"]["envelope"] is None


def test_fusion_works_on_a_burst_in_noise() -> None:
    """A real recording is a burst in noise, not a continuous signal."""
    for snr_db in (20.0, 10.0, 0.0):
        burst, _ = wf.synth(
            "QPSK", 4000, sps=SPS, snr_db=None, seed=4, sample_rate=SAMPLE_RATE
        )
        noisy = wf.add_awgn(burst, snr_db, np.random.default_rng(4))
        padded = np.concatenate(
            [
                0.05
                * (
                    np.random.default_rng(4).standard_normal(4000)
                    + 1j * np.random.default_rng(5).standard_normal(4000)
                ),
                noisy,
                0.05
                * (
                    np.random.default_rng(6).standard_normal(4000)
                    + 1j * np.random.default_rng(7).standard_normal(4000)
                ),
            ]
        )
        result = estimate_symbol_rate_fused(
            padded, SAMPLE_RATE, bandwidth_estimate=200_000.0
        )
        assert result["method"] == "corroborated", (snr_db, result)
        assert result["symbol_rate_hz"] == pytest.approx(TRUE_SYMBOL_RATE, rel=0.02)


def test_fusion_flags_a_symbol_rate_above_the_occupied_bandwidth() -> None:
    """``BW = Rs (1 + beta)`` with ``beta >= 0``, so ``Rs > BW`` is impossible."""
    samples, _ = _synth("QPSK", 10.0)
    result = estimate_symbol_rate_fused(
        samples, SAMPLE_RATE, bandwidth_estimate=40_000.0
    )
    assert any(
        "exceeds the occupied bandwidth" in warning for warning in result["warnings"]
    )


# --------------------------------------------------------------------------- #
# Fused sampling rate
# --------------------------------------------------------------------------- #


def test_sampling_rate_fused_refuses_when_unmeasurable() -> None:
    """No bandwidth and no symbol rate means no sampling rate -- and it says so."""
    rng = np.random.default_rng(23)
    n = 16000
    samples = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    result = estimate_sampling_rate_fused(
        samples, SAMPLE_RATE, bandwidth_estimate=0.0, symbol_rate_hz=0.0
    )
    assert result["sampling_rate_hz"] == 0.0
    assert result["confidence"] == 0.0
    assert result["warnings"]
    assert "not measurable" in result["warnings"][0]


def test_sampling_rate_fused_respects_nyquist() -> None:
    samples, _ = _synth("QPSK", 10.0)
    result = estimate_sampling_rate_fused(
        samples,
        SAMPLE_RATE,
        bandwidth_estimate=200_000.0,
        symbol_rate_hz=TRUE_SYMBOL_RATE,
    )
    assert result["sampling_rate_hz"] >= 2.0 * 200_000.0
    assert result["candidates"]


def test_sampling_rate_fused_warns_when_only_one_constraint_exists() -> None:
    """A feasibility floor is not a measurement, and the report must say which."""
    samples, _ = _synth("QPSK", 10.0)
    result = estimate_sampling_rate_fused(
        samples, SAMPLE_RATE, bandwidth_estimate=200_000.0, symbol_rate_hz=0.0
    )
    assert result["sampling_rate_hz"] > 0.0
    assert any("single constraint" in warning for warning in result["warnings"])
