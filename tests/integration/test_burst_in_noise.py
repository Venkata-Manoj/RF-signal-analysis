"""A framed burst inside noise -- the realistic capture -- and what it costs.

Every other coded-capture test in this suite writes the frame as the *whole*
file. A real recording is a burst surrounded by noise, and that difference used
to break two things: the modulation label, and the decode.

The label half is now fixed. The classifier works from whole-capture statistics,
which are not invariant to how much noise surrounds the burst, so a clean BPSK
burst was labelled QPSK or 8PSK. A wrong label hides the sync word, so the report
said "no frame found" and pointed the user at a framing bug that was really a
classification one. ``pipeline._corroborate_modulation`` now lets the header
search arbitrate: when the chosen demodulation sees a hint of the sync word but
cannot confirm it, the other candidates are tried.

The decode half is *mostly* fixed: the CRC anchor used to be able to land on a
documented deterministic false positive and report a verified payload carrying
one extra trailing byte, which is the worst failure mode this tool can have. The
remaining decode limitation is different and is pinned at the bottom of this
file: the search assumes the capture ends at the frame boundary.
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
        # Failing to decode is acceptable (see the limitation below); reporting a
        # wrong payload is not.
        return
    assert decoded["crc_pass"] is True
    assert bytes.fromhex(decoded["hex"]) == MESSAGE


# --------------------------------------------------------------------------- #
# Known limitation: the decode still assumes the capture ends at the frame.
# --------------------------------------------------------------------------- #


def test_the_coded_region_must_end_the_capture_to_decode(tmp_path):
    """Known limitation: the decode search assumes the capture ends at the frame.

    ``decode_hypotheses`` takes a ``frame_bits`` argument precisely for this, but
    the pipeline has no way to know the burst length from the signal alone, so it
    passes nothing and the search assumes a single-burst capture. A trailing tail
    therefore breaks the CRC-16 anchor even when everything else is perfect.

    Note how strong the rest of the evidence is: the modulation is right, the
    sync word is found at exactly the right offset, and the score is 1.0. Only
    the decode fails. That is why this is worth stating rather than leaving to be
    discovered -- it looks like a decoding bug but is really a missing
    frame-length estimate (a V2 item).
    """
    path = tmp_path / "padded.iq"
    _burst_in_noise(path, pad=1_600)
    report = _analyze(path)

    assert report["modulation"]["estimated_type"] == "BPSK"
    assert report["correlation"]["detected"] is True
    assert report["correlation"]["score"] == 1.0
    assert report["correlation"]["header_offset"] == 1_600

    assert report["payload"]["decoded"]["available"] is False
    # Not a budget problem -- the search simply cannot anchor the CRC.
    assert report["payload"]["decode_search"]["budget_exhausted"] is False


def test_capping_the_decode_prefix_restores_the_decode(tmp_path):
    """The workaround for the limitation above, and the evidence for its cause.

    Capping ``decode_max_bits`` to just past the frame excludes the noise tail
    from the search, so the CRC anchor is correct again and the payload decodes.
    That is what identifies the missing frame-length estimate as the cause,
    rather than the noise itself.
    """
    path = tmp_path / "capped.iq"
    frame = _burst_in_noise(path, pad=1_600)

    report = _analyze(path, decode_max_bits=int(frame["bits"].size))

    decoded = report["payload"]["decoded"]
    assert decoded["available"] is True
    assert decoded["crc_pass"] is True
    assert bytes.fromhex(decoded["hex"]) == MESSAGE
