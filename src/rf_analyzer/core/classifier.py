"""Modulation classification: spectral features + rules + ML head.

The original classifier was Higher-Order Cumulants (HOC) plus an
instantaneous-frequency gate for BPSK/QPSK/8PSK/16-QAM/2-FSK. This module keeps
that path as the honest fallback and adds:

* :func:`extract_features` -- a 13-dimensional spectral/statistical feature
  vector (cumulants C40/C42/C20, instantaneous-frequency variance/kurtosis/
  peak count/robust span, envelope statistics, PAPR, amplitude-moment
  ratios, PSD bandwidth/SNR). Every feature is a plain float; the vector
  order is fixed by :data:`FEATURE_NAMES` and shared with the trained model.
* Rule extensions for ``64-QAM`` (low C40 plus amplitude/PAPR tie-break),
  ``4-FSK`` (multilevel instantaneous frequency), ``AM`` (large envelope
  variance with stable frequency), ``FM`` (varying frequency with constant
  envelope) and ``TONE``/``AUDIO`` rejects.
* An ML head (:func:`classify_with_ml`): a small ReLU multilayer perceptron
  (13 -> 32 -> classes) trained by ``scripts/train_classifier.py`` on seeded
  ``waveform.synth`` captures at 0-20 dB Eb/N0 (plus analog rejects), stored
  in ``ml_model.npz`` next to this file together with the standardisation
  statistics. NumPy-only inference -- no scikit-learn dependency. A linear
  head was tried first and is documented in the training script: it cannot
  carve the bimodal 16-QAM boundary (symbol-spaced vs pulse-shaped captures
  sit in different feature regions) and it underweights the PSD peak
  sharpness that separates a noisy tone from weak FM.

Honesty contracts (do not weaken):

* :func:`classify_modulation` tries the ML head first and falls back to the
  rule/HOC path on *any* failure (missing model file, corrupt weights, bad
  shapes), so a broken model can only cost accuracy, never crash analysis.
* ML softmax probabilities are capped at :data:`ML_MAX_CONFIDENCE` (0.90):
  a softmax is overconfident out of distribution and must never report
  near-certainty.
* An FSK label without independent symbol-period corroboration is capped at
  ``MODULATION_UNCORROBORATED_CONFIDENCE`` (0.35), exactly as the pipeline
  does. A tone has zero frequency variance and audio has no piecewise-
  constant structure, so neither can pass the period test -- this is what
  guarantees "tone/audio never reported as confident FSK".
* ``TONE``/``AM``/``FM``/``AUDIO`` are *reject* labels: the tool cannot
  demodulate them, and the pipeline reports them as low-confidence with an
  explicit warning rather than as digital modes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rf_analyzer.config import MODULATION_UNCORROBORATED_CONFIDENCE

#: Feature vector order. Fixed -- the trained model depends on it.
FEATURE_NAMES = (
    "c40_abs",
    "c42_abs",
    "c20_abs",
    "log_fvar",
    "freq_kurt",
    "freq_nsig",
    "env_norm",
    "papr_db",
    "m4m2",
    "m6m2",
    "freq_prange",
    "psd_snr",
    "psd_bw_norm",
)

#: Softmax confidence is never reported above this. A discriminative model is
#: overconfident far from its training distribution; the cap keeps the report
#: honest.
ML_MAX_CONFIDENCE = 0.90

#: Below this top-1 softmax probability the capture is out of distribution
#: (e.g. real speech against synthetic training audio) and is reported as
#: UNKNOWN rather than as a forced guess.
ML_UNKNOWN_THRESHOLD = 0.40

#: Model artifact written by scripts/train_classifier.py.
MODEL_PATH = Path(__file__).with_name("ml_model.npz")

#: Labels that name analog/unmodulated energy rather than a digital mode the
#: tool can demodulate. The pipeline caps these at low confidence.
ANALOG_REJECT_LABELS = ("AM", "FM", "TONE", "AUDIO")

#: FSK labels: confidence requires symbol-period corroboration.
FSK_LABELS = ("2-FSK", "4-FSK")

#: Historically held 8PSK/64-QAM/4-FSK before those slicers existed.
#: Those modes now have both naive (``core/demod``) and coherent
#: (``core/receiver``) paths, so the set is empty. Kept as a named
#: constant so older callers and report code keep importing cleanly.
UNSUPPORTED_DIGITAL_LABELS: tuple[str, ...] = ()

#: Only a prefix is used for period recovery: the autocorrelation FFT is
#: O(n log n) and a multi-megabyte capture would pay it for nothing, since a
#: genuine symbol structure repeats and shows in any long slice.
PERIOD_PREFIX_SAMPLES = 250_000

#: Lazily-loaded ML head. A dict (not two globals) so the loader mutates
#: state without a `global` statement: {"model": dict | None, "missing": bool}.
_MODEL_STATE: dict = {"model": None, "missing": False}


def compute_cumulants(samples: np.ndarray) -> dict:
    """Compute 2nd and 4th order cumulants for AMC.

    Normalizes input to unit power before computation.

    Returns dict with keys: C20, C21, C40, C42
    """
    x = np.asarray(samples, dtype=np.complex128).ravel()
    if x.size < 4:
        return {"C20": 0.0, "C21": 0.0, "C40": 0.0, "C42": 0.0}

    # Normalize to unit power
    power = float(np.mean(np.abs(x) ** 2))
    if power < 1e-12:
        return {"C20": 0.0, "C21": 0.0, "C40": 0.0, "C42": 0.0}
    x = x / np.sqrt(power)

    # Moments
    M20 = complex(np.mean(x**2))
    M21 = float(np.mean(np.abs(x) ** 2))
    M40 = complex(np.mean(x**4))
    M42 = complex(np.mean(np.abs(x) ** 2 * x**2))

    # Cumulants
    C20 = M20
    C21 = M21
    C40 = M40 - 3.0 * M20**2
    C42 = M42 - abs(M20) ** 2 - 2.0 * M21**2

    return {
        "C20": C20,
        "C21": C21,
        "C40": C40,
        "C42": C42,
    }


def _is_fsk(samples: np.ndarray) -> bool:
    """Detect FSK via low instantaneous-frequency variance."""
    if len(samples) < 4:
        return False
    phase = np.unwrap(np.angle(samples.astype(np.complex128)))
    inst_freq = np.diff(phase)
    return float(np.var(inst_freq)) < 1.0


def _focus_burst(samples: np.ndarray) -> np.ndarray:
    """Return the measured burst view, or the original capture if unavailable.

    A classifier fed an entire recording otherwise lets surrounding noise set
    the cumulants, envelope, and first-PSD-window statistics.  Reusing the
    deterministic envelope-region estimator makes those features describe the
    signal burst itself.  This is a feature-extraction aid only: it does not
    make a region a decode claim, and every uncertainty/corroboration rule in
    the public classifier remains unchanged.
    """
    x = np.asarray(samples, dtype=np.complex128).ravel()
    if x.size < 128:
        return x
    try:
        from rf_analyzer.core.dsp import estimate_burst_region

        region = estimate_burst_region(x)
    except Exception:
        return x
    if not region.get("found", False):
        return x
    start = region.get("start_sample")
    end = region.get("end_sample")
    if start is None or end is None:
        return x
    start = max(0, int(start))
    end = min(x.size, int(end))
    focused = x[start:end]
    return focused if focused.size >= 32 else x


def extract_features(
    samples: np.ndarray, sample_rate: float = 1_000_000.0
) -> dict[str, float]:
    """Extract the finite spectral/statistical feature vector.

    For a capture with a measurable burst, the feature statistics are taken
    over the envelope-detected burst view rather than the surrounding noise.
    The full-capture path is unchanged for continuous signals. Every value is
    a finite float (non-finite inputs become 0.0), so the vector is always
    safe to standardise and feed to the ML head. The PSD pair reuses
    :mod:`rf_analyzer.core.dsp` on that same view.

    Args:
        samples: Complex baseband samples (any length >= 1).
        sample_rate: Sample rate in Hz; only normalises the bandwidth.

    Returns:
        Dict with one entry per name in :data:`FEATURE_NAMES`.
    """
    x = _focus_burst(np.asarray(samples, dtype=np.complex128).ravel())
    n = int(x.size)
    feats: dict[str, float] = dict.fromkeys(FEATURE_NAMES, 0.0)
    if n < 4:
        return feats

    c = compute_cumulants(x)
    feats["c40_abs"] = float(abs(c["C40"]))
    feats["c42_abs"] = float(abs(c["C42"]))
    feats["c20_abs"] = float(abs(c["C20"]))

    phase = np.unwrap(np.angle(x))
    inst_freq = np.diff(phase)
    fvar = float(np.var(inst_freq))
    feats["log_fvar"] = float(np.log10(fvar + 1e-12))
    if inst_freq.size >= 2:
        # Robust deviation span (1st-99th percentile): clean FSK shows its
        # tone spacing here (2-FSK ~0.39 rad/sample at the training setup,
        # 4-FSK ~3x that), a tone shows ~0, FM shows its peak deviation.
        # Percentiles, not min/max, so one noise outlier cannot set it.
        lo, hi = np.percentile(inst_freq, [1.0, 99.0])
        feats["freq_prange"] = float(max(0.0, float(hi) - float(lo)))
    if fvar > 1e-18:
        mean_f = float(np.mean(inst_freq))
        kurt = float(np.mean((inst_freq - mean_f) ** 4) / (fvar**2))
        feats["freq_kurt"] = float(np.clip(kurt, 0.0, 30.0))
    try:
        hist, _ = np.histogram(inst_freq, bins=30)
        peak = float(np.max(hist))
        if peak > 0:
            feats["freq_nsig"] = float(int(np.sum(hist > 0.2 * peak)))
    except Exception:
        feats["freq_nsig"] = 0.0

    env = np.abs(x)
    mean_env = float(np.mean(env))
    std_env = float(np.std(env))
    feats["env_norm"] = float(std_env / mean_env) if mean_env > 1e-12 else 0.0
    mean_sq = float(np.mean(env**2))
    if mean_sq > 1e-30:
        feats["papr_db"] = float(10.0 * np.log10(float(np.max(env**2)) / mean_sq))
        feats["m4m2"] = float(float(np.mean(env**4)) / (mean_sq**2))
        # Sixth-order amplitude moment: separates QAM orders (more rings ->
        # heavier tail) from constant-modulus modes (exactly 1.0). BPSK with
        # ISI sits between, which is what makes it useful rather than just
        # another envelope number.
        feats["m6m2"] = float(float(np.mean(env**6)) / (mean_sq**3))

    try:
        from rf_analyzer.core.dsp import (
            compute_psd,
            estimate_bandwidth,
            estimate_snr,
        )

        freqs, psd_db = compute_psd(x, float(sample_rate))
        feats["psd_snr"] = float(estimate_snr(psd_db))
        bw = float(estimate_bandwidth(freqs, psd_db))
        if bw > 0.0 and sample_rate > 0.0:
            feats["psd_bw_norm"] = float(bw / float(sample_rate))
    except Exception:
        pass

    for key, value in feats.items():
        if not np.isfinite(value):
            feats[key] = 0.0
    return feats


def feature_vector(samples: np.ndarray, sample_rate: float = 1_000_000.0) -> np.ndarray:
    """Feature dict as an ordered float array matching :data:`FEATURE_NAMES`."""
    feats = extract_features(samples, sample_rate)
    return np.array([float(feats[name]) for name in FEATURE_NAMES], dtype=np.float64)


def _fsk_order(nsig: float, fvar: float) -> str:
    """Decide 2-FSK vs 4-FSK from frequency discreteness.

    Measured regimes (Eb/N0, waveform.synth at 8 samples/symbol):

    * ``fvar < 0.11``: only 2-FSK lives this low (clean 0.034, 20 dB 0.044,
      15 dB 0.066; 4-FSK never drops below 0.16).
    * Clean 4-FSK concentrates into ~7 bins (``nsig <= 8``); noise smears
      both orders, and at 10 dB the counts are 12-14 vs 18-19.
    * At 5 dB and below the two orders overlap and this guess is unreliable
      (the ML head owns the noisy cases); confidence stays capped anyway.
    """
    if fvar < 0.11:
        return "2-FSK"
    if nsig <= 8:
        return "4-FSK"
    return "4-FSK" if nsig >= 16 else "2-FSK"


def _recovered_period(samples: np.ndarray) -> int:
    """Symbol-period corroboration on a bounded prefix (1 = declined)."""
    try:
        from rf_analyzer.core.dsp import estimate_fsk_symbol_period
    except Exception:
        return 1
    x = np.asarray(samples).ravel()
    if x.size > PERIOD_PREFIX_SAMPLES:
        x = x[:PERIOD_PREFIX_SAMPLES]
    try:
        return int(estimate_fsk_symbol_period(x))
    except Exception:
        return 1


def classify_rules(
    samples: np.ndarray, sample_rate: float = 1_000_000.0
) -> tuple[str, float, list[str]]:
    """Rule/HOC classifier: the fallback when the ML head is unavailable.

    Covers BPSK/QPSK/8PSK/16-QAM/64-QAM/2-FSK/4-FSK plus AM/FM/TONE/AUDIO
    rejects. FSK confidence without a recovered symbol period is capped at
    ``MODULATION_UNCORROBORATED_CONFIDENCE``, so tone/audio can never come
    back as confident FSK from this path either.
    """
    flat = np.asarray(samples).ravel()
    if flat.size < 4:
        return "UNKNOWN", 0.5, ["BPSK", "QPSK", "2-FSK"]

    feats = extract_features(flat, sample_rate)
    c40 = feats["c40_abs"]
    c20 = feats["c20_abs"]
    fvar = 10.0 ** feats["log_fvar"] - 1e-12
    kurt = feats["freq_kurt"]
    nsig = feats["freq_nsig"]
    env_norm = feats["env_norm"]

    # --- Analog rejects first: deterministic, need no model ---
    if fvar < 1e-9:
        # No measurable frequency movement: an unmodulated carrier, or an
        # amplitude-modulated one. The envelope decides.
        if env_norm > 0.2:
            return "AM", 0.70, ["TONE", "FM"]
        return "TONE", 0.90, ["FM", "AM"]
    if env_norm > 0.45 and fvar < 1e-6:
        return "AM", 0.70, ["TONE", "FM"]
    if env_norm < 0.12 and fvar < 0.02 and feats["freq_prange"] < 0.25:
        # Constant envelope, gently wandering frequency, narrow deviation
        # span: frequency modulation rather than frequency-shift keying.
        # Clean 2-FSK already spans ~0.39 rad/sample between its tones, so
        # it does not land here; noisy FSK exceeds the variance gate.
        return "FM", 0.65, ["4-FSK", "2-FSK", "TONE"]

    # --- FSK via independent period evidence ---
    if c40 < 0.35 and kurt < 10.0:
        period = _recovered_period(flat)
        order = _fsk_order(nsig, fvar)
        others = ["4-FSK", "2-FSK"]
        others = [m for m in others if m != order] + ["8PSK", "QPSK"]
        if period > 1:
            return order, 0.80, others[:3]
        if c40 < 0.25 and kurt < 8.0:
            # Uncorroborated: keep the label, cap the confidence. A tone
            # never reaches here (caught above); audio lands here at worst
            # as a capped guess, never as confident FSK.
            capped = min(0.80, MODULATION_UNCORROBORATED_CONFIDENCE)
            return order, capped, others[:3]

    # --- HOC ladder for PSK / QAM ---
    if c20 > 0.4:
        return "BPSK", 0.85, ["QPSK", "2-FSK"]
    if c40 > 1.5:
        return "BPSK", 0.85, ["QPSK", "2-FSK"]
    if c40 > 0.70:
        return "QPSK", 0.80, ["BPSK", "8PSK"]
    if 0.30 <= c40 <= 0.70:
        # 16-QAM vs 64-QAM: adjacent cumulant ranges (clean 0.54 vs 0.47)
        # that overlap under noise and seed variance, so the amplitude
        # distribution breaks the tie and confidence stays modest.
        if c40 > 0.60:
            return "16-QAM", 0.65, ["64-QAM", "QPSK"]
        if c40 < 0.42:
            return "64-QAM", 0.60, ["16-QAM", "8PSK"]
        if feats["papr_db"] > 6.42 or feats["m4m2"] > 1.49:
            return "64-QAM", 0.60, ["16-QAM", "8PSK"]
        return "16-QAM", 0.65, ["64-QAM", "QPSK"]
    if c40 < 0.30:
        return "8PSK", 0.70, ["QPSK", "16-QAM"]

    return "UNKNOWN", 0.50, ["BPSK", "QPSK", "2-FSK"]


def _load_model() -> dict | None:
    """Load and validate the trained ML head (cached; None when unusable)."""
    if _MODEL_STATE["model"] is not None:
        return _MODEL_STATE["model"]
    if _MODEL_STATE["missing"]:
        return None
    try:
        bundle = np.load(str(MODEL_PATH), allow_pickle=True)
        kind = str(bundle.get("kind", "mlp"))
        labels = [str(v) for v in bundle["labels"].tolist()]
        means = np.asarray(bundle["means"], dtype=np.float64).ravel()
        scales = np.asarray(bundle["scales"], dtype=np.float64).ravel()
        names = [str(v) for v in bundle["feature_names"].tolist()]
        if names != list(FEATURE_NAMES) or not labels:
            raise ValueError("ml_model.npz feature names do not match")
        if means.size != len(FEATURE_NAMES) or scales.size != len(FEATURE_NAMES):
            raise ValueError("ml_model.npz standardisation stats mismatch")
        if kind != "mlp":
            raise ValueError(f"unsupported model kind {kind!r}")
        hidden = int(bundle["hidden"])
        w1 = np.asarray(bundle["W1"], dtype=np.float64)
        b1 = np.asarray(bundle["b1"], dtype=np.float64).ravel()
        w2 = np.asarray(bundle["W2"], dtype=np.float64)
        b2 = np.asarray(bundle["b2"], dtype=np.float64).ravel()
        if (
            w1.shape != (len(FEATURE_NAMES), hidden)
            or b1.size != hidden
            or w2.shape != (hidden, len(labels))
            or b2.size != len(labels)
        ):
            raise ValueError("ml_model.npz has an unexpected layout")
        for arr in (w1, b1, w2, b2):
            if not np.all(np.isfinite(arr)):
                raise ValueError("ml_model.npz weights are not finite")
        _MODEL_STATE["model"] = {
            "labels": labels,
            "means": means,
            "scales": np.where(scales > 1e-12, scales, 1.0),
            "W1": w1,
            "b1": b1,
            "W2": w2,
            "b2": b2,
        }
        return _MODEL_STATE["model"]
    except Exception:
        _MODEL_STATE["missing"] = True
        return None


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - float(np.max(logits))
    exp = np.exp(shifted)
    total = float(np.sum(exp))
    if not np.isfinite(total) or total <= 0.0:
        return np.full_like(exp, 1.0 / max(exp.size, 1))
    return exp / total


def classify_with_ml(
    samples: np.ndarray, sample_rate: float = 1_000_000.0
) -> tuple[str, float, list[str]]:
    """Classify with the trained ML head.

    Raises:
        RuntimeError: no usable model (caller falls back to rules).
        ValueError: input too short.
    """
    flat = np.asarray(samples).ravel()
    if flat.size < 4:
        raise ValueError("not enough samples to classify")
    model = _load_model()
    if model is None:
        raise RuntimeError(f"ML model unavailable at {MODEL_PATH}")

    vec = feature_vector(flat, sample_rate)
    standardised = (vec - model["means"]) / model["scales"]
    if not np.all(np.isfinite(standardised)):
        raise RuntimeError("non-finite feature vector")
    hidden = np.maximum(standardised @ model["W1"] + model["b1"], 0.0)
    logits = hidden @ model["W2"] + model["b2"]
    probs = _softmax(logits)
    order = np.argsort(-probs)
    labels: list[str] = model["labels"]
    top = labels[int(order[0])]
    top_prob = float(probs[int(order[0])])
    alternatives = [labels[int(i)] for i in order[1:4]]

    if top_prob < ML_UNKNOWN_THRESHOLD:
        return "UNKNOWN", 0.50, alternatives

    confidence = min(top_prob, ML_MAX_CONFIDENCE)
    period = _recovered_period(flat)

    if top in FSK_LABELS:
        # Same corroboration rule as the fallback path: no period, no
        # confidence. The label is kept (it is still the best guess) but a
        # tone or audio capture can never leave here confident.
        if period <= 1:
            confidence = min(confidence, MODULATION_UNCORROBORATED_CONFIDENCE)
    elif period > 1 and float(vec[0]) < 0.35:
        # Independent period evidence overrules a digital ML label only when
        # the cumulants agree the signal is non-linear (constant envelope).
        # Genuine noisy FSK the ML head misses is recovered here; PSK/QAM
        # never passes the period test (measured), so they are unaffected.
        fvar = 10.0 ** float(vec[3]) - 1e-12
        top = _fsk_order(float(vec[5]), fvar)
        alternatives = (
            ["2-FSK", "4-FSK", "8PSK"] if top == "2-FSK" else ["4-FSK", "2-FSK", "8PSK"]
        )
        confidence = 0.80

    return top, float(confidence), alternatives


def classify_modulation(
    samples: np.ndarray,
    sample_rate: float | None = None,
    use_ml: bool = True,
) -> tuple[str, float, list[str]]:
    """Classify modulation, ML head first with rule/HOC fallback.

    Returns ``(estimated_type, confidence, alternatives)``.

    ``estimated_type`` is one of: ``'BPSK'``, ``'QPSK'``, ``'8PSK'``,
    ``'16-QAM'``, ``'64-QAM'``, ``'2-FSK'``, ``'4-FSK'``, ``'AM'``,
    ``'FM'``, ``'TONE'``, ``'AUDIO'``, ``'UNKNOWN'``.

    ``confidence`` is in ``[0.0, 1.0]``. FSK confidence without a recovered
    symbol period is capped at 0.35; ML confidence is capped at 0.90.

    Args:
        samples: Complex baseband samples.
        sample_rate: Sample rate in Hz for the PSD features (defaults to
            1 MHz; only normalises the bandwidth feature).
        use_ml: When False, skip the ML head and use rules only.
    """
    flat = np.asarray(samples).ravel()
    if flat.size < 4:
        return "UNKNOWN", 0.5, ["BPSK", "QPSK", "2-FSK"]
    rate = float(sample_rate) if sample_rate else 1_000_000.0

    if use_ml:
        try:
            return classify_with_ml(flat, rate)
        except Exception:
            pass

    try:
        return classify_rules(flat, rate)
    except Exception:
        pass

    # --- Legacy HOC ladder: last resort, never raises ---
    if _is_fsk(flat):
        return "2-FSK", 0.75, ["BPSK", "QPSK"]

    c = compute_cumulants(flat)
    c40_abs = abs(c["C40"])
    c42_abs = abs(c["C42"])

    if c40_abs > 1.5:
        return "BPSK", 0.85, ["QPSK", "2-FSK"]
    if 0.8 < c40_abs <= 1.5:
        return "QPSK", 0.80, ["BPSK", "8PSK"]
    if c40_abs < 0.3:
        if c42_abs < 0.5:
            return "8PSK", 0.70, ["QPSK", "16-QAM"]
        return "16-QAM", 0.65, ["8PSK", "QPSK"]

    return "UNKNOWN", 0.50, ["BPSK", "QPSK", "2-FSK"]


__all__ = [
    "ANALOG_REJECT_LABELS",
    "FEATURE_NAMES",
    "FSK_LABELS",
    "ML_MAX_CONFIDENCE",
    "ML_UNKNOWN_THRESHOLD",
    "MODEL_PATH",
    "UNSUPPORTED_DIGITAL_LABELS",
    "classify_modulation",
    "classify_rules",
    "classify_with_ml",
    "compute_cumulants",
    "extract_features",
    "feature_vector",
]
