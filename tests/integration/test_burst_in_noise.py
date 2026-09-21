"""A framed burst inside noise -- the realistic capture -- and what it costs.

Every other coded-capture test in this suite writes the frame as the *whole*
file. A real recording is a burst surrounded by noise, and that difference broke
three things. All three are fixed, and this file is the record of each.

1. **The modulation label.** The classifier works from whole-capture statistics,
   which are not invariant to how much noise surrounds the burst, so a clean BPSK
   burst was labelled QPSK or 8PSK. A wrong label hides the sync word, so the
   report said "no frame found" and pointed the user at a framing bug that was
   really a classification one. ``pipeline._corroborate_modulation`` now lets the
   header search arbitrate: when the chosen demodulation sees a hint of the sync
   word but cannot confirm it, the other candidates are tried.

2. **A CRC-verified payload with one extra trailing byte.** The CRC anchor could
   land on a documented deterministic false positive. Fixed in ``fec``; see
   ``tests/unit/test_fec_codecs.py`` for the mechanism.

3. **A noise tail hiding the frame.** The CRC anchor is searched near the end of
   the stream, so a frame followed by a noise tail was never found even when
   everything else was perfect. ``signal.burst`` now measures where the burst
   ends and bounds the search with it.

The honesty invariant runs through all of them: a reported decode is exactly the
message, never the message plus something. Failing to decode is acceptable;
reporting a wrong payload is not.
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

#: Padding values used throughout. The classifier's answer flips between these
#: (BPSK / QPSK / 8PSK) even though the burst is byte-identical, which is the
#: whole point -- and why assertions here target the invariants rather than any
#: single padding value.
PADS = (0, 200, 400, 800, 1_600, 4_000)


def _burst_in_noise(path, pad: int, amplitude: float = 0.05):
    """Write ``noise | frame | noise`` and return the transmitted frame."""
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


def _analyze(path, **overrides):
    request = {
        "file_path": str(path),
        "sample_rate": SAMPLE_RATE,
        "iq_format": "complex64",
        "modulation": "auto",
        "sync_word": SYNC_WORD,
        **BUDGET,
    }
    request.update(overrides)
    return analyze_file(request)


def test_a_wrong_label_is_revised_when_it_hides_the_header(tmp_path):
    """The header search must be allowed to overrule the classifier.

    At this padding the classifier calls a clean BPSK burst QPSK, and the QPSK
    demodulation cannot find the sync word at all. Without corroboration the
    report would claim no frame exists.
    """
    path = tmp_path / "revised.iq"
    _burst_in_noise(path, pad=400)
    report = _analyze(path)

    modulation = report["modulation"]
    assert modulation["revised_from"] == "QPSK"
    assert modulation["estimated_type"] == "BPSK"
    assert modulation["confidence"] > 0.5, "a header match is real evidence"

    assert report["correlation"]["detected"] is True
    assert report["correlation"]["header_offset"] == 400

    decoded = report["payload"]["decoded"]
    assert decoded["available"] is True
    assert bytes.fromhex(decoded["hex"]) == MESSAGE

    # The revision must be explained, never silent.
    assert any("revised" in w for w in report["warnings"])


def test_an_unsupported_label_reports_the_modulation_actually_used(tmp_path):
    """The report must not name a modulation the bit stream never came from.

    At this padding the classifier emits 8PSK. There is no 8PSK demodulator, so
    the pipeline reads the bits as BPSK -- and BPSK is the truth. Reporting
    "8PSK" would be a wrong extracted parameter, which is the one thing this
    tool exists to get right, so the label follows what was actually used and
    the classifier's answer is kept in `revised_from`.
    """
    path = tmp_path / "unsupported.iq"
    _burst_in_noise(path, pad=800)
    report = _analyze(path)

    modulation = report["modulation"]
    assert modulation["revised_from"] == "8PSK"
    assert modulation["estimated_type"] == "BPSK"
    assert any("Unsupported modulation" in w for w in report["warnings"])


@pytest.mark.parametrize("pad", PADS)
def test_the_reported_modulation_is_bpsk_at_every_padding(tmp_path, pad):
    """The label must be invariant to how much noise surrounds the burst.

    This is the property that was broken: the same byte-identical burst came
    back as BPSK, QPSK or 8PSK depending only on the padding.
    """
    path = tmp_path / f"invariant_{pad}.iq"
    _burst_in_noise(path, pad=pad)
    assert _analyze(path)["modulation"]["estimated_type"] == "BPSK"


@pytest.mark.parametrize("pad", [200, 400])
def test_an_explicit_request_is_never_revised(tmp_path, pad):
    """Corroboration only ever revises an *estimate*, never the user's choice.

    Asking for QPSK on a BPSK burst must report QPSK and fail to find the
    header, rather than quietly overriding the caller.
    """
    path = tmp_path / f"explicit_{pad}.iq"
    _burst_in_noise(path, pad=pad)
    report = _analyze(path, modulation="QPSK")

    assert report["modulation"]["estimated_type"] == "QPSK"
    assert report["modulation"]["revised_from"] is None
    assert report["correlation"]["detected"] is False


def test_no_revision_without_a_sync_word(tmp_path):
    """With nothing to correlate against there is nothing to arbitrate.

    The revision is only as good as the header evidence behind it, so it must
    not happen when no sync word is available.
    """
    path = tmp_path / "no_sync.iq"
    _burst_in_noise(path, pad=400)
    report = _analyze(path, sync_word=None)

    assert report["modulation"]["revised_from"] is None


def test_the_unpadded_burst_is_the_control(tmp_path):
    """Without padding the same burst classifies and decodes correctly.

    Keeps the tests above honest: they are about the surrounding noise, not about
    the frame being undecodable in principle.
    """
    path = tmp_path / "bare.iq"
    _burst_in_noise(path, pad=0)
    report = _analyze(path)

    assert report["modulation"]["estimated_type"] == "BPSK"
    assert report["modulation"]["revised_from"] is None
    assert report["correlation"]["detected"] is True
    assert report["correlation"]["score"] == 1.0

    decoded = report["payload"]["decoded"]
    assert decoded["available"] is True
    assert bytes.fromhex(decoded["hex"]) == MESSAGE


# --------------------------------------------------------------------------- #
# The honesty invariant, over every padding.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("pad", PADS)
def test_a_reported_decode_is_always_exactly_the_message(tmp_path, pad):
    """A verified payload must be the message -- never the message plus a byte.

    ``crc_pass`` is the tool's licence to claim a decode, so a payload that
    passes the CRC but carries a stray byte is worse than a failed decode: it
    looks authoritative and is wrong. That happened at pad=400, where the CRC
    framing anchored on a deterministic one-byte-too-long collision (the byte
    after the frame is 0x00, so ``crc16(M || C_hi) == C_lo << 8`` validates as
    ``[M][C_hi]`` with ``[C_lo][0x00]``).

    Asserting the *exact* payload rather than a prefix is the point: the buggy
    output started with the right bytes.
    """
    path = tmp_path / f"exact_{pad}.iq"
    _burst_in_noise(path, pad=pad)
    decoded = _analyze(path)["payload"]["decoded"]

    if not decoded["available"]:
        # Failing to decode is acceptable; reporting a wrong payload is not.
        return
    assert decoded["crc_pass"] is True
    assert bytes.fromhex(decoded["hex"]) == MESSAGE


# --------------------------------------------------------------------------- #
# The burst estimate is what makes a real capture decodable.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("pad", [800, 1_600, 4_000])
def test_a_noise_tail_no_longer_blocks_the_decode(tmp_path, pad):
    """A frame followed by noise must still decode.

    This used to fail, and the failure looked like a decoding bug: the modulation
    was right, the sync word was found at exactly the right offset with score 1.0,
    and ``budget_exhausted`` was False -- only the decode failed. The real cause
    was that the CRC anchor is searched near the *end of the stream*, so a noise
    tail put the anchor on noise instead of on the frame.

    ``signal.burst`` now measures where the burst ends, and that length bounds the
    search. These paddings are exactly the ones that used to fail.
    """
    path = tmp_path / f"tail_{pad}.iq"
    frame = _burst_in_noise(path, pad=pad)
    report = _analyze(path)

    assert report["modulation"]["estimated_type"] == "BPSK"
    assert report["correlation"]["header_offset"] == pad

    burst = report["signal"]["burst"]
    assert burst["found"] is True
    assert burst["snr_db"] > 20.0

    # The estimator's error is one-sided by construction: a block counts as burst
    # when its mean power clears the threshold, so a block holding even a little
    # burst is included. Never early, and never more than one window late.
    n_frame = int(frame["bits"].size)
    assert pad + n_frame <= burst["end_sample"] <= pad + n_frame + burst["window"]

    decoded = report["payload"]["decoded"]
    assert decoded["available"] is True
    assert decoded["crc_pass"] is True
    assert bytes.fromhex(decoded["hex"]) == MESSAGE

    # The search must have been bounded by the estimate, not by the capture end.
    assert report["payload"]["decode_search"]["frame_bits"] is not None
    assert any("burst region ends at sample" in w for w in report["warnings"])


def test_an_explicit_frame_bits_still_wins(tmp_path):
    """An explicit hint is the caller's, not ours to override."""
    path = tmp_path / "explicit_hint.iq"
    frame = _burst_in_noise(path, pad=1_600)
    coded_bits = int(frame["bits"].size) - 32  # minus the sync word

    report = _analyze(path, frame_bits=coded_bits)

    assert report["payload"]["decode_search"]["frame_bits"] == coded_bits
    decoded = report["payload"]["decoded"]
    assert decoded["available"] is True
    assert bytes.fromhex(decoded["hex"]) == MESSAGE


def test_a_noise_gap_is_required_to_measure_a_burst(tmp_path):
    """A capture with no noise gap has no burst to find, so nothing changes.

    The bare-frame captures fill their file, and the real-world recordings are
    continuous signals. Neither has a burst, and both must keep decoding exactly
    as they did before this estimate existed -- so no hint is derived and the
    search falls back to assuming the capture ends at the frame.
    """
    path = tmp_path / "continuous.iq"
    _burst_in_noise(path, pad=0)
    report = _analyze(path)

    burst = report["signal"]["burst"]
    assert burst["found"] is False
    assert burst["end_sample"] is None
    assert report["payload"]["decode_search"]["frame_bits"] is None
    assert bytes.fromhex(report["payload"]["decoded"]["hex"]) == MESSAGE


def test_a_measured_burst_is_never_evidence_on_its_own(tmp_path):
    """The estimate narrows a search; the CRC-16 still decides.

    A burst of random bits in noise is measured -- a burst genuinely is there, so
    the search does get bounded by it -- and the report still claims no payload,
    because no CRC-16 validates. Without this, a region estimate would be a way to
    produce a decode that is not there.
    """
    path = tmp_path / "random_burst.iq"
    rng = np.random.default_rng(3)
    core = ((rng.integers(0, 2, size=2_000) * 2.0) - 1.0).astype(np.complex64)
    lead = (rng.standard_normal(800) + 1j * rng.standard_normal(800)) * 0.05
    trail = (rng.standard_normal(800) + 1j * rng.standard_normal(800)) * 0.05
    np.concatenate([lead, core, trail]).astype(np.complex64).tofile(path)

    report = _analyze(path)

    assert report["signal"]["burst"]["found"] is True
    assert report["payload"]["decoded"]["available"] is False
