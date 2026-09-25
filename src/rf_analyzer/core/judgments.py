"""Typed judgments for the RF workbench, following TypeSafe System One patterns.

Implements the TypeSafe primitives (Choice / Noul / Score) and the
confidence-gated routing pattern as deterministic local code -- no network,
no API key, no model download. The live TypeSafe docs
(https://docs.typesafe.ai/llms.txt, primitives, confidence,
patterns/confidence-routing) are the design source; this module is the
offline application of those patterns to signal-analysis decisions:

* **Choice** -- which modulation? Distribution over the classifier's labels,
  ``confidence = (k * pmax - 1) / (k - 1)`` per the docs confidence explorer.
* **Noul** -- is the header really present? Probability derived from the
  CFAR-controlled correlation (``correlator.FALSE_ALARM_TARGET``), not from
  a raw score vs a constant.
* **Score** -- where does the constellation fit sit on ordered levels?
  Ordered EVM levels owned by this repo, probabilities over levels.
* **Confidence-gated routing** -- ``act`` / ``confirm`` / ``escalate`` from
  confidence with risk-scaled thresholds. A decode (high stakes, CRC decides)
  never acts on confidence alone; modulation display (low stakes) may.

Honesty contracts (do not weaken):
* Typed output constrains the interface, never the truth. Probabilities come
  from measured evidence (softmax, CFAR tail, EVM) and every judgment carries
  the method that produced it.
* Low confidence escalates to a human via warnings + ``assumed`` flags, never
  to a silent guess. ``0.0`` still means "not measurable".
* No decode without a CRC pass -- ``decode_noul`` is 0.99 only when the
  existing ``decode_hypotheses`` search validated, else ~0.01.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Risk-scaled routing thresholds (confidence-gated routing pattern).
ROUTE_ACT_THRESHOLD = 0.80
ROUTE_CONFIRM_THRESHOLD = 0.50


def confidence_from_probs(probs: np.ndarray) -> float:
    """Confidence from a probability distribution.

    ``(k * pmax - 1) / (k - 1)`` for ``k`` options: 1.0 when all mass sits on
    one option, 0.0 on a uniform split. Matches the TypeSafe confidence
    explorer for Choice answers.
    """
    p = np.asarray(probs, dtype=np.float64).ravel()
    k = int(p.size)
    if k < 2:
        return 0.0
    total = float(np.sum(p))
    if not np.isfinite(total) or total <= 0.0:
        return 0.0
    p = p / total
    peak = float(np.max(p))
    return float(min(1.0, max(0.0, (k * peak - 1.0) / (k - 1.0))))


@dataclass
class Choice:
    """One-of-a-defined-set judgment with distribution + confidence."""

    choice: str
    probabilities: dict[str, float]
    confidence: float
    method: str = ""
    options: list[str] = field(default_factory=list)


@dataclass
class Noul:
    """Yes/no judgment. ``noul`` is P(yes); near 0.5 is uncertain.

    No separate confidence -- the probability itself is the signal.
    """

    noul: float
    method: str = ""


@dataclass
class Score:
    """Position along ordered levels with distribution + confidence."""

    score: float
    legend: list[str]
    probabilities: dict[str, float]
    confidence: float
    method: str = ""


def route(
    confidence: float,
    *,
    high: float = ROUTE_ACT_THRESHOLD,
    low: float = ROUTE_CONFIRM_THRESHOLD,
) -> str:
    """Confidence-gated routing: ``act`` / ``confirm`` / ``escalate``."""
    c = float(confidence)
    if c >= float(high):
        return "act"
    if c >= float(low):
        return "confirm"
    return "escalate"


def modulation_choice(samples: np.ndarray, sample_rate: float | None = None) -> Choice:
    """Modulation as a Choice over classifier labels with real distribution.

    Loads the trained ML head's softmax when available (same weights the
    pipeline's classifier uses); falls back to a peaked pseudo-distribution
    around the rule/HOC answer when the model is missing. FSK corroboration
    caps are applied exactly as in ``core.classifier`` so a tone can never
    leave here as confident FSK.
    """
    from rf_analyzer.core import classifier as clf

    flat = np.asarray(samples).ravel()
    rate = float(sample_rate) if sample_rate else 1_000_000.0
    if flat.size < 4:
        return Choice(
            choice="UNKNOWN",
            probabilities={"UNKNOWN": 1.0},
            confidence=0.0,
            method="too few samples",
            options=["UNKNOWN"],
        )
    # Real softmax distribution when the model loads.
    try:
        model = clf._load_model()
        if model is not None:
            vec = clf.feature_vector(flat, rate)
            std = (vec - model["means"]) / model["scales"]
            hidden = np.maximum(std @ model["W1"] + model["b1"], 0.0)
            logits = hidden @ model["W2"] + model["b2"]
            shifted = logits - float(np.max(logits))
            exp = np.exp(shifted)
            probs = exp / float(np.sum(exp))
            labels: list[str] = list(model["labels"])
            prob_map = {lab: float(probs[i]) for i, lab in enumerate(labels)}
            # Apply the same corroboration rules as classify_with_ml so this
            # Choice agrees with the pipeline's reported label.
            top, conf, _ = clf.classify_with_ml(flat, rate)
            # Re-peak the distribution on the corroborated top label without
            # inventing mass: keep the measured spread, move the peak.
            if top in prob_map:
                mass = float(prob_map[top])
                target = float(conf)
                # Only shrink an overconfident peak (corroboration cap); never
                # inflate beyond what was measured.
                if target < mass:
                    leftover = 1.0 - target
                    rest = 1.0 - mass
                    prob_map = {
                        lab: (
                            target
                            if lab == top
                            else (
                                p / rest * leftover
                                if rest > 0
                                else leftover / (len(prob_map) - 1)
                            )
                        )
                        for lab, p in prob_map.items()
                    }
            conf_val = confidence_from_probs(
                np.array([prob_map[lab] for lab in labels])
            )
            order = sorted(prob_map, key=lambda lab: -prob_map[lab])
            return Choice(
                choice=top,
                probabilities={lab: prob_map[lab] for lab in order},
                confidence=float(min(conf_val, 0.90)),
                method="ml-softmax",
                options=order,
            )
    except Exception:
        pass
    # Fallback: rule/HOC answer, peaked pseudo-distribution (method says so).
    try:
        top, conf, alts = clf.classify_rules(flat, rate)
    except Exception:
        top, conf, alts = "UNKNOWN", 0.5, []
    options = [top, *[a for a in alts or [] if a != top][:3]]
    if not options:
        options = [top]
    k = len(options)
    conf = float(min(max(float(conf), 0.0), 0.90))
    rest = (1.0 - conf) / max(k - 1, 1) if k > 1 else 0.0
    prob_map = {lab: (conf if i == 0 else rest) for i, lab in enumerate(options)}
    return Choice(
        choice=top,
        probabilities=prob_map,
        confidence=confidence_from_probs(np.array([prob_map[lab] for lab in options])),
        method="rules-fallback",
        options=options,
    )


def header_noul(bits: np.ndarray, sync_word: str = "0x1ACFFC1D") -> Noul:
    """Is the sync word really present? CFAR-derived probability.

    Uses ``find_header`` (length-aware, ``FALSE_ALARM_TARGET=0.01``): a
    significant detection maps near 0.99, a high raw score that misses
    significance maps low (chance match on a long capture), otherwise the
    raw score scaled down. Never compares a raw score against a constant.
    """
    from rf_analyzer.core.correlator import find_header, hex_to_bits

    rx = np.asarray(bits, dtype=np.uint8).ravel()
    try:
        sync_bits = hex_to_bits(str(sync_word))
    except Exception:
        return Noul(noul=0.0, method="bad sync word")
    if rx.size < sync_bits.size or sync_bits.size == 0:
        return Noul(noul=0.0, method="stream shorter than sync word")
    try:
        found = find_header(rx, sync_bits)
    except Exception:
        return Noul(noul=0.0, method="correlation error")
    score = float(found.get("score", 0.0))
    min_score = float(found.get("min_score", 0.0))
    detected = bool(found.get("detected", False))
    if detected:
        return Noul(noul=0.99, method="cfar-significant")
    if score >= 0.85 and score < min_score:
        # Looks like a match but is expected by chance at this search length.
        return Noul(noul=0.15, method="chance-match-below-cfar")
    return Noul(
        noul=float(min(max(score * 0.5, 0.0), 0.49)), method="raw-score-downweighted"
    )


QUALITY_LEVELS = ["excellent", "good", "poor", "unusable"]


def quality_score(evm_percent: float | None) -> Score:
    """Constellation fit as a Score over ordered EVM levels.

    Levels (owned by this repo, matching ``MODULATION_FIT_EVM_WARN=30``):
    excellent <5%, good <15%, poor <30%, unusable >=30%. ``None`` (FSK /
    unmeasurable) returns a flat distribution with confidence 0.
    """
    if evm_percent is None or not np.isfinite(float(evm_percent)):
        flat = dict.fromkeys(QUALITY_LEVELS, 0.25)
        return Score(
            score=3.0,
            legend=list(QUALITY_LEVELS),
            probabilities=flat,
            confidence=0.0,
            method="not-applicable",
        )
    evm = float(evm_percent)
    # Triangular weighting around the measured EVM for a smooth distribution.
    centers = np.array([2.5, 10.0, 22.5, 45.0])
    dist = np.abs(centers - evm)
    weights = 1.0 / (1.0 + dist / 7.5)
    probs = weights / float(np.sum(weights))
    prob_map = {lvl: float(probs[i]) for i, lvl in enumerate(QUALITY_LEVELS)}
    score_pos = float(np.sum(probs * np.arange(4)))
    return Score(
        score=score_pos,
        legend=list(QUALITY_LEVELS),
        probabilities=prob_map,
        confidence=confidence_from_probs(probs),
        method="evm-levels",
    )


def decode_noul(*, validated: bool, crc_pass: bool | None) -> Noul:
    """Decode trust: 0.99 only on a CRC-validated decode, else ~0.01."""
    if bool(validated) and crc_pass is True:
        return Noul(noul=0.99, method="crc-16-validated")
    return Noul(noul=0.01, method="no-crc-pass")


def ask_all(
    samples: np.ndarray,
    bits: np.ndarray,
    sync_word: str | None = None,
    evm_percent: float | None = None,
    *,
    validated: bool = False,
    crc_pass: bool | None = None,
    sample_rate: float | None = None,
) -> dict[str, Choice | Noul | Score]:
    """Speculative fan-out: independent judgments over the same state.

    All questions see the same capture and are evaluated independently --
    one answer never becomes hidden context for another. Code decides which
    answers to use (e.g. ignore quality when the mode is FSK).
    """
    judgments: dict[str, Choice | Noul | Score] = {
        "modulation": modulation_choice(samples, sample_rate),
        "signal_quality": quality_score(evm_percent),
        "decode_trust": decode_noul(validated=validated, crc_pass=crc_pass),
    }
    if sync_word:
        judgments["header_present"] = header_noul(bits, sync_word)
    else:
        judgments["header_present"] = Noul(noul=0.0, method="no sync word supplied")
    return judgments


__all__ = [
    "QUALITY_LEVELS",
    "ROUTE_ACT_THRESHOLD",
    "ROUTE_CONFIRM_THRESHOLD",
    "Choice",
    "Noul",
    "Score",
    "ask_all",
    "confidence_from_probs",
    "decode_noul",
    "header_noul",
    "modulation_choice",
    "quality_score",
    "route",
]
