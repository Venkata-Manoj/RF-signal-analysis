"""Train the classifier ML head (a small NumPy-only ReLU MLP).

Dataset (seeded, reproducible):

* Digital modes ``BPSK/QPSK/8PSK/16-QAM/64-QAM/2-FSK/4-FSK`` from
  :func:`rf_analyzer.core.waveform.synth` at Eb/N0 0/5/10/15/20 dB plus
  clean, 12 training seeds. ``BPSK/QPSK/16-QAM`` additionally get 1-sample-
  per-symbol captures (the ``framing.modulate`` regime the pipeline's
  existing tests use) for half the seeds, so the head stays accurate on
  both oversampled and symbol-spaced inputs.
* Short digital bursts in AWGN, with burst lengths spanning 512--6000
  samples, surrounding pads spanning 0--8192 samples, and SNR from 6--20 dB.
  These are generated with :func:`rf_analyzer.core.waveform.burst_in_noise`
  using seeds disjoint from the full-capture and held-out sets.  A short
  burst occupies only a small fraction of a noisy recording, so training on
  full captures alone teaches the head to call that realistic case AUDIO.
* Analog rejects ``AM/FM/TONE/AUDIO`` from the seeded generators below at
  the same SNR grid (SNR stands in for Eb/N0 where bits are meaningless).
* Every capture is round-tripped through a real file -- alternating ``.iq``
  (via :func:`rf_analyzer.core.io.load_iq`) and ``.wav`` (via
  :func:`rf_analyzer.core.io.load_wav`) -- so the trained features are the
  ``.iq`` + ``.wav`` spectral features the pipeline actually sees, not
  in-memory arrays.

Model: a small ReLU multilayer perceptron (13 -> 64 -> classes) trained
with full-batch Adam and L2 regularisation. Full-capture and burst rows are
weighted separately so the larger augmentation does not erase the original
full-capture boundaries. Deliberately NumPy-only:
scikit-learn is not a project dependency (``requirements.txt`` has no
sklearn) and adding one for a 13-feature head would trade a hard install
requirement for nothing.

A single-layer softmax was tried first and kept as a cautionary result: it
stops at ~90 % train / ~77 % held-out because two boundaries are not
linear. 16-QAM is bimodal (symbol-spaced captures sit at C40 ~ 0.68 with a
peaky spectrum, pulse-shaped ones at ~0.55 with a flat one), and a noisy
tone vs weak FM vs weak 2-FSK differ only in PSD peak sharpness given the
noise floor -- both need the hidden layer. Nothing here claims more than a
small head over honest features -- confidence is capped at inference (see
``core/classifier.py``).

Held-out checks: fresh full-capture seeds (100-102) at 0--20 dB and a
separate burst-in-noise matrix (fresh seeds 3000-3001) must both classify
above 90 %. Tone/audio must never come back as confident FSK. The script
exits non-zero when any bar is missed.

Writes ``src/rf_analyzer/core/ml_model.npz`` (weights, feature means and
scales, label order, feature names).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rf_analyzer.core import waveform as waveform_mod
from rf_analyzer.core.classifier import (
    FEATURE_NAMES,
    FSK_LABELS,
    MODEL_PATH,
    feature_vector,
)
from rf_analyzer.core.io import load_iq, load_wav

SAMPLE_RATE = 1_000_000.0
TRAIN_SNR_DB = (0.0, 5.0, 10.0, 15.0, 20.0, None)
TRAIN_SEEDS = tuple(range(12))
HELDOUT_SEEDS = (100, 101, 102)
#: Burst lengths are *sample* counts, not bit counts.  896 is included
#: explicitly because it is the short-BPSK regression that motivated this
#: augmentation; 3000 and 6000 keep the range broad rather than tuning only
#: to the smallest case.
BURST_LENGTHS_SAMPLES = (512, 896, 3000, 6000)
#: Include both modest and large surrounding-noise regions.  A zero pad is
#: useful as the control, while 8192 is long enough to make the burst a
#: minority of the whole capture.
BURST_PADS_SAMPLES = (0, 256, 800, 2048, 8192)
BURST_SNR_DB = (6.0, 10.0, 15.0, 20.0)
BURST_TRAIN_SEEDS = (2000, 2001)
BURST_HELDOUT_SEEDS = (3000, 3001)
#: The burst matrix is intentionally larger than the original full-capture
#: matrix.  Give each full-capture row two units of loss so the augmentation
#: teaches the missing regime without moving the established 16-QAM boundary
#: wholesale into the burst cluster.
FULL_CAPTURE_WEIGHT = 2.0
BURST_WEIGHT = 1.0
MLP_HIDDEN = 64
MLP_ITERS = 6000
#: Extra noise realisations where FSK is weakest: constant-envelope modes
#: under noise share (fvar, kurtosis, envelope) statistics with FM, so the
#: head needs dense sampling there to stay calibrated instead of guessing
#: FM at 0.7+ confidence.
FSK_LOW_SNR = (0.0, 2.5, 5.0)
FSK_EXTRA_SEEDS = tuple(range(1000, 1004))
N_BITS = 6000
N_ANALOG = 48_000
BURST_MIN_ACCURACY = 0.90

DIGITAL_SCHEMES = (
    "BPSK",
    "QPSK",
    "8PSK",
    "16-QAM",
    "64-QAM",
    "2-FSK",
    "4-FSK",
)
ANALOG_KINDS = ("AM", "FM", "TONE", "AUDIO")
ALL_LABELS = tuple(list(DIGITAL_SCHEMES) + list(ANALOG_KINDS))

#: 1-sample-per-symbol augmentation covers the framing.modulate regime for
#: the modes the original transmit path emits. FSK is always oversampled and
#: 8PSK/64-QAM have no symbol-spaced transmit path, so they train oversampled only.
ONE_SPS_SCHEMES = ("BPSK", "QPSK", "16-QAM")


def synth_analog(
    kind: str, n_samples: int, seed: int, snr_db: float | None
) -> np.ndarray:
    """Seeded analog-reject generator at unit average power.

    * TONE: unmodulated carrier at a random offset.
    * AM: carrier RAM-modulated by one sinusoidal message (depth 0.5-0.9).
    * FM: carrier phase-modulated by one sinusoid (beta 2-6).
    * AUDIO: five real sinusoids (100 Hz - 3.5 kHz, speech-band-ish) with
      independent slow amplitude wobble plus breath noise, real-only like a
      mono ``.wav`` speech recording loads.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples, dtype=np.float64) / SAMPLE_RATE
    if kind == "TONE":
        fc = float(rng.uniform(-20_000.0, 20_000.0))
        wave = np.exp(1j * 2.0 * np.pi * fc * t)
    elif kind == "AM":
        fc = float(rng.uniform(-15_000.0, 15_000.0))
        fm = float(rng.uniform(500.0, 2000.0))
        depth = float(rng.uniform(0.5, 0.9))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        wave = (1.0 + depth * np.sin(2.0 * np.pi * fm * t + phase)) * np.exp(
            1j * 2.0 * np.pi * fc * t
        )
    elif kind == "FM":
        fc = float(rng.uniform(-15_000.0, 15_000.0))
        fm = float(rng.uniform(800.0, 2500.0))
        # Beta >= 3 keeps the peak deviation (>= 2.4 kHz) measurable: below
        # that the modulation hides under the training noise floor and the
        # capture is a tone for every honest purpose (held-out still sweeps
        # the ambiguous corner, and the report caps what the head claims).
        beta = float(rng.uniform(3.0, 8.0))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        wave = np.exp(
            1j * (2.0 * np.pi * fc * t + beta * np.sin(2.0 * np.pi * fm * t + phase))
        )
    elif kind == "AUDIO":
        wave = np.zeros(n_samples, dtype=np.float64)
        for _ in range(5):
            freq = float(rng.uniform(100.0, 3500.0))
            amp = float(rng.uniform(0.3, 1.0))
            wobble_f = float(rng.uniform(3.0, 9.0))
            tone_phase = float(rng.uniform(0.0, 2.0 * np.pi))
            wave = wave + amp * (
                1.0 + 0.4 * np.sin(2.0 * np.pi * wobble_f * t)
            ) * np.sin(2.0 * np.pi * freq * t + tone_phase)
        wave = wave + 0.05 * rng.standard_normal(n_samples)
        wave = wave.astype(np.complex128)  # real-only, like mono .wav
    else:
        raise ValueError(f"unknown analog kind {kind!r}")

    power = float(np.mean(np.abs(wave) ** 2))
    if power > 0.0:
        wave = wave / np.sqrt(power)
    if snr_db is not None:
        wave = waveform_mod.add_awgn(wave, float(snr_db), rng)
    return np.asarray(wave, dtype=np.complex64)


def synth_one_sps(
    scheme: str, n_bits: int, seed: int, ebn0_db: float | None
) -> np.ndarray:
    """Symbol-spaced capture (framing.modulate regime) plus AWGN at Es/N0."""
    rng = np.random.default_rng(seed)
    width = waveform_mod.bits_per_symbol(scheme)
    n_symbols = max(1, int(n_bits) // width)
    bits = rng.integers(0, 2, size=n_symbols * width, dtype=np.uint8)
    symbols = waveform_mod.bits_to_symbols(bits, scheme)
    if ebn0_db is not None:
        esn0_db = waveform_mod.ebn0_to_esn0(ebn0_db, width)
        symbols = waveform_mod.add_awgn(symbols, esn0_db, rng)
    return np.asarray(symbols, dtype=np.complex64)


def _burst_n_bits(scheme: str, n_samples: int) -> int:
    """Convert a requested clean burst length in samples to waveform bits.

    ``waveform.synth`` takes a bit count.  Its linear schemes emit eight
    samples per symbol, while its FSK schemes emit ``bits_per_symbol``
    bits per symbol at the same samples-per-symbol rate.  Keeping this
    conversion in one place prevents a nominal "896-sample" augmentation
    from silently becoming a several-thousand-sample burst for QAM/FSK.
    """
    width = waveform_mod.bits_per_symbol(scheme)
    sps = waveform_mod.DEFAULT_SPS
    return max(width, round(float(n_samples) * width / sps))


def _synth_burst(
    scheme: str,
    n_samples: int,
    pad: int,
    snr_db: float,
    seed: int,
) -> np.ndarray:
    """Generate one deterministic digital burst surrounded by complex AWGN."""
    raw, _, _ = waveform_mod.burst_in_noise(
        scheme,
        _burst_n_bits(scheme, n_samples),
        snr_db=snr_db,
        sps=waveform_mod.DEFAULT_SPS,
        lead=pad,
        tail=pad,
        seed=seed,
    )
    return np.asarray(raw, dtype=np.complex64)


def _roundtrip(
    samples: np.ndarray, counter: int, workdir: Path
) -> tuple[np.ndarray, str]:
    """Write to .iq/.wav alternately and reload, exercising both io paths."""
    if counter % 2 == 0:
        path = workdir / f"cap_{counter}.iq"
        np.asarray(samples, dtype=np.complex64).tofile(path)
        return load_iq(path, dtype="complex64"), "iq"
    path = workdir / f"cap_{counter}.wav"
    stereo = np.column_stack([np.real(samples), np.imag(samples)]).astype(np.float32)
    sf.write(str(path), stereo, int(SAMPLE_RATE), subtype="FLOAT")
    back, _ = load_wav(path)
    return back, "wav"


def build_burst_dataset(
    seeds: tuple[int, ...] = BURST_TRAIN_SEEDS,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int]]:
    """Build the deterministic short-burst-in-noise feature matrix.

    The clean burst is placed between equal lead/tail noise regions.  The
    requested lengths are sample counts and cover 512--6000 samples; the
    surrounding pads and SNR values deliberately include both ends of the
    operating range so the classifier cannot key only on a particular
    signal-to-noise ratio or padding length.
    """
    rows: list[np.ndarray] = []
    labels: list[str] = []
    io_mix = {"iq": 0, "wav": 0}
    counter = 0
    with tempfile.TemporaryDirectory(prefix="rf_train_burst_") as tmp:
        workdir = Path(tmp)
        for label in DIGITAL_SCHEMES:
            for n_samples in BURST_LENGTHS_SAMPLES:
                for pad in BURST_PADS_SAMPLES:
                    for snr in BURST_SNR_DB:
                        for seed in seeds:
                            raw = _synth_burst(label, n_samples, pad, snr, seed)
                            back, kind = _roundtrip(raw, counter, workdir)
                            io_mix[kind] += 1
                            counter += 1
                            rows.append(feature_vector(back, SAMPLE_RATE))
                            labels.append(label)
    return np.stack(rows), np.array(labels), list(ALL_LABELS), io_mix


def build_dataset(
    seeds: tuple[int, ...],
    one_sps: bool = True,
    *,
    include_bursts: bool = False,
    burst_seeds: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int]]:
    """Generate (X, y) with one row of features per seeded capture.

    ``include_bursts`` is deliberately opt-in so held-out full-capture
    evaluation remains independent of the burst augmentation.  The burst
    rows use their own disjoint seed set by default.
    """
    rows: list[np.ndarray] = []
    labels: list[str] = []
    io_mix = {"iq": 0, "wav": 0}
    counter = 0
    with tempfile.TemporaryDirectory(prefix="rf_train_") as tmp:
        workdir = Path(tmp)
        for label in ALL_LABELS:
            for snr in TRAIN_SNR_DB:
                for seed in seeds:
                    if label in DIGITAL_SCHEMES:
                        if one_sps and label in ONE_SPS_SCHEMES and seed % 2 == 1:
                            raw = synth_one_sps(label, N_BITS, seed, snr)
                        else:
                            kw = {} if snr is None else {"ebn0_db": snr}
                            raw, _ = waveform_mod.synth(label, N_BITS, seed=seed, **kw)
                    else:
                        raw = synth_analog(
                            label, N_ANALOG, seed * 31 + ALL_LABELS.index(label), snr
                        )
                    back, kind = _roundtrip(raw, counter, workdir)
                    io_mix[kind] += 1
                    counter += 1
                    rows.append(feature_vector(back, SAMPLE_RATE))
                    labels.append(label)
        # Extra low-SNR FSK realisations (see FSK_LOW_SNR note above).
        if seeds == TRAIN_SEEDS:
            for label in ("2-FSK", "4-FSK"):
                for snr in FSK_LOW_SNR:
                    for seed in FSK_EXTRA_SEEDS:
                        raw, _ = waveform_mod.synth(
                            label, N_BITS, seed=seed, ebn0_db=snr
                        )
                        back, kind = _roundtrip(raw, counter, workdir)
                        io_mix[kind] += 1
                        counter += 1
                        rows.append(feature_vector(back, SAMPLE_RATE))
                        labels.append(label)
        if include_bursts:
            burst_seed_set = BURST_TRAIN_SEEDS if burst_seeds is None else burst_seeds
            burst_x, burst_y, _, burst_io = build_burst_dataset(burst_seed_set)
            rows.extend(burst_x)
            labels.extend(burst_y.tolist())
            for kind in io_mix:
                io_mix[kind] += burst_io[kind]
    return np.stack(rows), np.array(labels), list(ALL_LABELS), io_mix


def train_mlp(
    x: np.ndarray,
    y_idx: np.ndarray,
    n_classes: int,
    hidden: int = 32,
    seed: int = 0,
    iters: int = 3000,
    lr: float = 0.02,
    l2: float = 1e-4,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Full-batch Adam for a ReLU MLP (D -> hidden -> classes).

    ``sample_weight`` lets the two capture regimes share one head without
    letting the much larger burst matrix erase the original full-capture
    decision boundaries.  The weights affect the loss only; feature
    standardisation is still computed from the unweighted training matrix.
    """
    rng = np.random.default_rng(seed)
    n, d = x.shape
    if sample_weight is None:
        weights = np.ones(n, dtype=np.float64)
    else:
        weights = np.asarray(sample_weight, dtype=np.float64).ravel()
        if weights.size != n or np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError(
                "sample_weight must contain one positive finite value per row"
            )
    weight_sum = float(np.sum(weights))
    W1 = rng.standard_normal((d, hidden)) * np.sqrt(2.0 / d)
    b1 = np.zeros(hidden)
    W2 = rng.standard_normal((hidden, n_classes)) * np.sqrt(1.0 / hidden)
    b2 = np.zeros(n_classes)
    Y = np.zeros((n, n_classes))
    Y[np.arange(n), y_idx] = 1.0
    params = [W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    b1m, b2m = 0.9, 0.999
    eps = 1e-8
    for t in range(1, iters + 1):
        pre = x @ W1 + b1
        H = np.maximum(pre, 0.0)
        logits = H @ W2 + b2
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        P = exp / exp.sum(axis=1, keepdims=True)
        dlog = (P - Y) * (weights / weight_sum)[:, None]
        dW2 = H.T @ dlog + l2 * W2
        db2 = dlog.sum(axis=0)
        dH = dlog @ W2.T
        dH[pre <= 0.0] = 0.0
        dW1 = x.T @ dH + l2 * W1
        db1 = dH.sum(axis=0)
        for i, g in enumerate((dW1, db1, dW2, db2)):
            m[i] = b1m * m[i] + (1.0 - b1m) * g
            v[i] = b2m * v[i] + (1.0 - b2m) * (g * g)
            mh = m[i] / (1.0 - b1m**t)
            vh = v[i] / (1.0 - b2m**t)
            params[i] -= lr * mh / (np.sqrt(vh) + eps)
        W1, b1, W2, b2 = params
    return W1, b1, W2, b2


def predict_proba(
    x: np.ndarray, means: np.ndarray, scales: np.ndarray, *model: np.ndarray
) -> np.ndarray:
    """MLP softmax probabilities for rows of x."""
    W1, b1, W2, b2 = model
    xs = (x - means) / scales
    hidden = np.maximum(xs @ W1 + b1, 0.0)
    logits = hidden @ W2 + b2
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def accuracy_table(
    x: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    *model: np.ndarray,
    class_labels: list[str] | None = None,
) -> tuple[float, dict[str, float]]:
    """Overall + per-class accuracy from a fitted (means, scales, *weights).

    ``labels`` always describes the model's output-index order.  When a
    held-out matrix contains only digital rows, ``class_labels`` limits the
    reported per-class table without accidentally remapping output indices.
    """
    means, scales, *weights = model
    proba = predict_proba(x, means, scales, *weights)
    pred = np.array([labels[i] for i in np.argmax(proba, axis=1)])
    report_labels = labels if class_labels is None else class_labels
    per_class = {lab: float(np.mean(pred[y == lab] == lab)) for lab in report_labels}
    return float(np.mean(pred == y)), per_class


def _print_accuracy_table(
    title: str,
    x: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    *model: np.ndarray,
    class_labels: list[str] | None = None,
) -> tuple[float, dict[str, float]]:
    """Print and return one labelled held-out accuracy table."""
    overall, per_class = accuracy_table(x, y, labels, *model, class_labels=class_labels)
    report_labels = labels if class_labels is None else class_labels
    print(title)
    print(f"  {'class':10s} {'acc':>6s}  n")
    for lab in report_labels:
        n = int(np.sum(y == lab))
        print(f"  {lab:10s} {per_class[lab] * 100:5.1f}%  {n}")
    print(f"  {'OVERALL':10s} {overall * 100:5.1f}%  {len(y)}")
    return overall, per_class


def main() -> int:
    print("Building seeded training set (.iq + .wav round-trips) ...")
    X_full, y_full, labels, full_io = build_dataset(TRAIN_SEEDS)
    X_burst, y_burst, _, burst_train_io = build_burst_dataset()
    X = np.vstack((X_full, X_burst))
    y = np.concatenate((y_full, y_burst))
    io_mix = {kind: full_io[kind] + burst_train_io[kind] for kind in full_io}
    sample_weight = np.concatenate(
        (
            np.full(X_full.shape[0], FULL_CAPTURE_WEIGHT, dtype=np.float64),
            np.full(X_burst.shape[0], BURST_WEIGHT, dtype=np.float64),
        )
    )
    print(
        f"  rows={X.shape[0]} dims={X.shape[1]} io_mix={io_mix} "
        f"full_rows={X_full.shape[0]} burst_rows={X_burst.shape[0]}"
    )
    label_index = {lab: i for i, lab in enumerate(labels)}
    y_idx = np.array([label_index[v] for v in y])

    # Standardise with the same regime weights used by the loss.  Otherwise
    # the much larger burst matrix shifts every mean/scale toward noise and
    # quietly moves the full-capture decision boundaries.
    means = np.average(X, axis=0, weights=sample_weight)
    variance = np.average((X - means) ** 2, axis=0, weights=sample_weight)
    scales = np.sqrt(variance)
    scales = np.where(scales > 1e-12, scales, 1.0)
    Xs = (X - means) / scales

    print(f"Training NumPy MLP head (13 -> {MLP_HIDDEN} -> classes) ...")
    W1, b1, W2, b2 = train_mlp(
        Xs,
        y_idx,
        len(labels),
        hidden=MLP_HIDDEN,
        iters=MLP_ITERS,
        sample_weight=sample_weight,
    )
    train_acc, _ = accuracy_table(X, y, labels, means, scales, W1, b1, W2, b2)
    print(f"  train accuracy: {train_acc * 100:.1f}%")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        MODEL_PATH,
        kind=np.array("mlp"),
        hidden=np.array(MLP_HIDDEN),
        W1=W1,
        b1=b1,
        W2=W2,
        b2=b2,
        means=means,
        scales=scales,
        labels=np.array(labels),
        feature_names=np.array(list(FEATURE_NAMES)),
    )
    print(f"  model saved to {MODEL_PATH}")

    # Full-capture held-out accuracy.  Report the digital subset separately:
    # analog rejects are useful sanity checks, but they are not evidence that
    # the digital head works at the requested 0--20 dB operating range.
    print("Held-out full-capture check (fresh seeds 100-102) ...")
    Xh, yh, _, _ = build_dataset(HELDOUT_SEEDS)
    overall, _ = _print_accuracy_table(
        "  all labels", Xh, yh, labels, means, scales, W1, b1, W2, b2
    )
    digital_mask = np.isin(yh, DIGITAL_SCHEMES)
    digital_overall, _ = _print_accuracy_table(
        "  digital 0--20 dB",
        Xh[digital_mask],
        yh[digital_mask],
        labels,
        means,
        scales,
        W1,
        b1,
        W2,
        b2,
        class_labels=list(DIGITAL_SCHEMES),
    )

    # Separate burst matrix: no training seeds and no full-capture rows are
    # used here.  This is the regression test for the short-burst/AUDIO gap.
    print("Held-out burst-in-noise check (fresh seeds 3000-3001) ...")
    Xb, yb, _, burst_io = build_burst_dataset(BURST_HELDOUT_SEEDS)
    burst_overall, _ = _print_accuracy_table(
        "  digital bursts",
        Xb,
        yb,
        labels,
        means,
        scales,
        W1,
        b1,
        W2,
        b2,
        class_labels=list(DIGITAL_SCHEMES),
    )
    print(
        f"  lengths={BURST_LENGTHS_SAMPLES} pads={BURST_PADS_SAMPLES} "
        f"snr_db={BURST_SNR_DB} io_mix={burst_io}"
    )

    # Tone/audio must never be confident FSK: check via the public
    # classifier (ML head + corroboration caps), not raw softmax output.
    from rf_analyzer.core.classifier import classify_modulation

    ok = True
    sr = SAMPLE_RATE
    n = N_ANALOG
    t = np.arange(n, dtype=np.float64) / sr
    tone = np.exp(1j * 2.0 * np.pi * 10_000.0 * t).astype(np.complex64)
    audio = synth_analog("AUDIO", n, 777, 10.0)
    for name, sig in (("tone", tone), ("audio-10dB", audio)):
        lab, conf, _ = classify_modulation(sig, sample_rate=sr)
        flag = "OK " if not (lab in FSK_LABELS and conf > 0.5) else "FAIL"
        print(f"  reject {name:10s}: {lab:8s} conf={conf:.2f} [{flag}]")
        if lab in FSK_LABELS and conf > 0.5:
            print(f"  FAIL: {name} reported as confident FSK")
            ok = False
    if overall < BURST_MIN_ACCURACY:
        print(f"  FAIL: held-out all-label accuracy {overall:.3f} < 0.90")
        ok = False
    if digital_overall < BURST_MIN_ACCURACY:
        print(f"  FAIL: held-out digital accuracy {digital_overall:.3f} < 0.90")
        ok = False
    if burst_overall < BURST_MIN_ACCURACY:
        print(f"  FAIL: held-out burst accuracy {burst_overall:.3f} < 0.90")
        ok = False
    print("  RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
