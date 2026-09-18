"""A framed burst inside noise -- the realistic capture -- and what it costs.

Every other coded-capture test in this suite writes the frame as the *whole*
file. A real recording is a burst surrounded by noise, and this file pins the
two ways that difference changes the answer. Both are known MVP limitations
rather than desired behaviour: the tests exist so the behaviour is explicit and
so that fixing either one fails loudly here instead of silently elsewhere.

Neither test claims the tool is wrong to be cautious. The point is that the
limitation is *measured and stated*, not discovered later by a user.
"""

from __future__ import annotations

import numpy as np
import pytest

from rf_analyzer.core.framing import build_frame, modulate
from rf_analyzer.pipeline import analyze_file

SAMPLE_RATE = 100_000
SYNC_WORD = "0x1ACFFC1D"
MESSAGE = b"SIH26147 burst-in-noise characterisation payload."

BUDGET = {"decode_time_budget_s": 120.0, "decode_max_bits": 200_000}


def _burst_in_noise(path, pad: int, amplitude: float = 0.05):
    """Write ``noise | frame | noise`` and return the transmitted frame bits."""
    frame = build_frame(MESSAGE, fec="conv", interleaver="block")
    core = modulate(frame["bits"], "BPSK", sample_rate=SAMPLE_RATE)

    rng = np.random.default_rng(7)
    if pad:
        lead = (rng.standard_normal(pad) + 1j * rng.standard_normal(pad)) * amplitude
        trail = (rng.standard_normal(pad) + 1j * rng.standard_normal(pad)) * amplitude
        samples = np.concatenate([lead, core, trail])
    else:
        samples = core

    samples.astype(np.complex64).tofile(path)
    return frame


def _analyze(path):
    return analyze_file(
        {
            "file_path": str(path),
            "sample_rate": SAMPLE_RATE,
            "iq_format": "complex64",
            "modulation": "auto",
            "sync_word": SYNC_WORD,
            **BUDGET,
        }
    )


def test_the_coded_region_must_end_the_capture_to_decode(tmp_path):
    """Known limitation: the decode search assumes the capture ends at the frame.

    ``decode_hypotheses`` takes a ``frame_bits`` argument precisely for this, but
    the pipeline has no way to know the burst length from the signal alone, so it
    passes nothing and the search assumes a single-burst capture. A trailing tail
    therefore breaks the CRC-16 anchor even when everything else is perfect.

    Note how strong the rest of the evidence is here: the modulation is correct,
    the sync word is found at exactly the right offset, and the correlation score
    is a perfect 1.0. Only the decode fails. That is why this is worth stating
    rather than leaving to be discovered -- it looks like a decoding bug but is
    really a missing frame-length estimate (a V2 item).
    """
    path = tmp_path / "burst_padded.iq"
    _burst_in_noise(path, pad=1_600)
    report = _analyze(path)

    correlation = report["correlation"]
    assert correlation["detected"] is True
    assert correlation["score"] == 1.0
    assert correlation["header_offset"] == 1_600

    assert report["modulation"]["estimated_type"] == "BPSK"
    assert report["payload"]["decoded"]["available"] is False

    # The search is not out of budget -- it simply cannot anchor the CRC.
    assert report["payload"]["decode_search"]["budget_exhausted"] is False


def test_capping_the_decode_prefix_restores_the_decode(tmp_path):
    """The workaround for the limitation above, and the evidence for its cause.

    Capping ``decode_max_bits`` to just past the frame excludes the noise tail
    from the search, so the CRC anchor is correct again and the payload decodes.
    That is what makes the missing frame-length estimate the cause rather than
    the noise itself.
    """
    path = tmp_path / "burst_capped.iq"
    frame = _burst_in_noise(path, pad=1_600)

    report = analyze_file(
        {
            "file_path": str(path),
            "sample_rate": SAMPLE_RATE,
            "iq_format": "complex64",
            "modulation": "auto",
            "sync_word": SYNC_WORD,
            "decode_time_budget_s": 120.0,
            "decode_max_bits": int(frame["bits"].size),
        }
    )

    decoded = report["payload"]["decoded"]
    assert decoded["available"] is True
    assert decoded["crc_pass"] is True
    assert bytes.fromhex(decoded["hex"]) == MESSAGE


def test_the_modulation_label_depends_on_how_much_noise_surrounds_it(tmp_path):
    """Known limitation: the classifier is not invariant to the surrounding noise.

    The transmitted burst is byte-identical in every case -- the only variable is
    how many noise samples sit either side of it. Sweeping that padding changes
    the reported modulation, at 0.7-0.85 confidence, with nothing in the report
    warning that the label is unreliable.

    The assertion is deliberately about the *instability* rather than about any
    one padding value. Which pads flip depends on the frame's length and content
    (a different message flips at different pads), so pinning a single value
    would be a brittle test of an accident. What is stable is that the same
    signal cannot be classified consistently.

    This is the more consequential of the two limitations, because real captures
    are bursts in noise. Unlike the 2-FSK case there is no cheap independent
    corroboration to lean on: EVM cannot separate these, because a sparser
    constellation always fits a denser one's points.
    """
    labels = {}
    for pad in (0, 200, 400, 800, 1_600, 4_000):
        path = tmp_path / f"burst_pad{pad}.iq"
        _burst_in_noise(path, pad=pad)
        labels[pad] = _analyze(path)["modulation"]["estimated_type"]

    # Control: with no padding the burst is labelled correctly...
    assert labels[0] == "BPSK"

    # ...but adding noise to the *same* burst changes the answer.
    assert any(label != "BPSK" for label in labels.values()), labels


def test_a_mislabelled_burst_can_lose_its_header(tmp_path):
    """Why the instability above matters: the wrong demod can hide the frame.

    A wrong label does not *always* hide the header -- 8PSK still correlates the
    sync word here, because a BPSK signal sits on a subset of the 8PSK points and
    the decision device happens to recover the right bits. But when it does hide
    it, the report shows no detection at all, so a user sees "no frame found"
    rather than "the modulation is uncertain", which points at the wrong problem.

    Two things are asserted, both of which do hold: a mislabelled burst never
    produces a verified payload, and at least one padding value loses the header
    outright.
    """
    lost = []
    for pad in (0, 200, 400, 800, 1_600, 4_000):
        path = tmp_path / f"hdr_pad{pad}.iq"
        _burst_in_noise(path, pad=pad)
        report = _analyze(path)

        label = report["modulation"]["estimated_type"]
        if label == "BPSK":
            continue

        # A wrong label must never yield a verified payload.
        assert report["payload"]["decoded"]["available"] is False, (pad, label)
        if not report["correlation"]["detected"]:
            lost.append((pad, label))

    assert lost, "expected at least one padding value to hide the header"


def test_the_unpadded_burst_is_the_control(tmp_path):
    """Without padding the same burst classifies and decodes correctly.

    This is the control that keeps the two tests above honest: they are about the
    surrounding noise, not about the frame being undecodable in principle.
    """
    path = tmp_path / "burst_bare.iq"
    _burst_in_noise(path, pad=0)
    report = _analyze(path)

    assert report["modulation"]["estimated_type"] == "BPSK"
    assert report["correlation"]["detected"] is True
    assert report["correlation"]["score"] == 1.0

    decoded = report["payload"]["decoded"]
    assert decoded["available"] is True
    assert bytes.fromhex(decoded["hex"]) == MESSAGE
