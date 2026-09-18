"""Unit tests for the transmit-side frame builder.

:mod:`rf_analyzer.core.framing` is the mirror of the analysis pipeline: it is
what makes requirements (ii)-(v) provable end to end. These tests pin the frame
layout and check that every FEC/interleaver combination the receiver can search
over is actually producible.
"""

from __future__ import annotations

import numpy as np
import pytest

from rf_analyzer.core.deinterleave import search_interleaver
from rf_analyzer.core.fec import (
    bits_to_bytes,
    crc16_check,
    scan_crc16,
    scan_crc16_strict,
)
from rf_analyzer.core.framing import (
    FEC_SCHEMES,
    INTERLEAVER_SCHEMES,
    apply_interleaver,
    build_frame,
    encode_fec,
    modulate,
)
from rf_analyzer.core.payload import slice_payload

MESSAGE = b"framing unit-test message, deliberately over 40 bytes long."


class TestEncodeFec:
    @pytest.mark.parametrize("scheme", FEC_SCHEMES)
    def test_every_scheme_produces_bits(self, scheme):
        bits = encode_fec(MESSAGE, scheme)
        assert bits.size > 0
        assert set(np.unique(bits)).issubset({0, 1})

    def test_uncoded_scheme_is_the_crc_framed_message(self):
        bits = encode_fec(MESSAGE, "none")
        assert bits.size == 8 * (len(MESSAGE) + 2)
        assert scan_crc16_strict(bits_to_bytes(bits))[0] is True

    def test_convolutional_doubles_the_rate_and_adds_the_tail(self):
        bits = encode_fec(MESSAGE, "conv")
        assert bits.size == 2 * (8 * (len(MESSAGE) + 2) + 6)

    def test_rs_appends_parity_symbols(self):
        bits = encode_fec(MESSAGE, "rs", nsym=32)
        assert bits.size == 8 * (len(MESSAGE) + 2 + 32)

    def test_ldpc_pads_to_a_whole_number_of_blocks(self):
        from rf_analyzer.core.fec import LDPC_K, LDPC_N

        bits = encode_fec(MESSAGE, "ldpc")
        assert bits.size % LDPC_N == 0
        framed_bits = 8 * (len(MESSAGE) + 2)
        padded = ((framed_bits + LDPC_K - 1) // LDPC_K) * LDPC_K
        assert bits.size == padded // LDPC_K * LDPC_N

    def test_unknown_scheme_is_rejected(self):
        with pytest.raises(ValueError, match="unknown FEC scheme"):
            encode_fec(MESSAGE, "turbo")


class TestApplyInterleaver:
    @pytest.mark.parametrize("scheme", INTERLEAVER_SCHEMES)
    def test_every_scheme_is_length_preserving(self, scheme):
        bits = np.arange(500, dtype=np.uint8) % 2
        out = apply_interleaver(bits, scheme)
        assert out.size >= bits.size
        assert out.size - bits.size < 512  # padding stays bounded

    def test_none_is_a_pass_through(self):
        bits = np.arange(64, dtype=np.uint8) % 2
        assert np.array_equal(apply_interleaver(bits, "none"), bits)

    def test_unknown_scheme_is_rejected(self):
        with pytest.raises(ValueError, match="unknown interleaver"):
            apply_interleaver(np.zeros(16, dtype=np.uint8), "golden-ratio")


class TestModulate:
    @pytest.mark.parametrize("mode", ["BPSK", "QPSK", "16-QAM"])
    def test_symbol_count_matches_the_spectral_efficiency(self, mode):
        bits = np.arange(256, dtype=np.uint8) % 2
        symbols = modulate(bits, mode)
        expected = {"BPSK": 256, "QPSK": 128, "16-QAM": 64}[mode]
        assert symbols.size == expected

    def test_unknown_modulation_is_rejected(self):
        with pytest.raises(ValueError, match="unsupported modulation"):
            modulate(np.zeros(16, dtype=np.uint8), "OFDM")


class TestBuildFrame:
    @pytest.mark.parametrize("scheme", ["none", "block", "convolutional", "diagonal"])
    @pytest.mark.parametrize("fec", ["none", "conv", "rs", "concatenated"])
    def test_frame_starts_with_the_sync_word(self, fec, scheme):
        frame = build_frame(MESSAGE, fec=fec, interleaver=scheme)
        assert np.array_equal(
            frame["bits"][: frame["sync_bits"].size], frame["sync_bits"]
        )

    def test_frame_bits_describes_the_coded_region(self):
        frame = build_frame(MESSAGE, fec="conv", interleaver="block")
        assert frame["frame_bits"] == frame["coded_bits"].size
        assert frame["bits"].size == frame["frame_bits"] + frame["sync_bits"].size

    def test_layout_records_the_transmitter_settings(self):
        frame = build_frame(MESSAGE, fec="rs", interleaver="diagonal", nsym=32)
        assert frame["layout"]["fec"] == "rs"
        assert frame["layout"]["nsym"] == 32
        assert frame["layout"]["interleaver"] == "diagonal"
        assert frame["layout"]["message_bytes"] == len(MESSAGE)


class TestFrameRoundTrip:
    """The receiver must recover the message without being told the settings."""

    @pytest.mark.parametrize(
        ("fec", "interleaver"),
        [
            ("none", "none"),
            ("conv", "none"),
            ("conv", "convolutional"),
            ("rs", "none"),
            ("rs", "block"),
            ("concatenated", "none"),
            ("concatenated", "diagonal"),
            ("ldpc", "pseudo-random-block"),
        ],
    )
    def test_blind_search_recovers_the_message(self, fec, interleaver):
        frame = build_frame(MESSAGE, fec=fec, interleaver=interleaver)
        sync_len = int(frame["sync_bits"].size)

        result = search_interleaver(
            frame["bits"],
            start_offset=sync_len,
            max_bits=100_000,
            time_budget_s=120.0,
        )

        assert result["validated"] is True, f"{fec}/{interleaver}: no CRC-valid decode"
        assert bytes.fromhex(result["best"]["payload_hex"]) == MESSAGE

    @pytest.mark.parametrize(
        ("fec", "interleaver"),
        [
            ("conv", "convolutional"),
            ("rs", "block"),
            ("concatenated", "diagonal"),
            ("ldpc", "pseudo-random-block"),
        ],
    )
    def test_blind_search_names_the_interleaver(self, fec, interleaver):
        frame = build_frame(MESSAGE, fec=fec, interleaver=interleaver)
        result = search_interleaver(
            frame["bits"],
            start_offset=int(frame["sync_bits"].size),
            max_bits=100_000,
            time_budget_s=120.0,
        )
        assert result["interleaver"] == interleaver

    def test_no_false_positive_on_random_bits(self):
        rng = np.random.default_rng(11)
        noise = rng.integers(0, 2, size=4000, dtype=np.uint8)
        result = search_interleaver(noise, max_bits=100_000, time_budget_s=120.0)
        assert result["validated"] is False

    def test_frame_bits_hint_truncates_trailing_samples(self):
        frame = build_frame(MESSAGE, fec="rs", interleaver="block")
        sync_len = int(frame["sync_bits"].size)
        rng = np.random.default_rng(7)
        # Append a burst of unrelated samples after the frame.
        noisy_tail = rng.integers(0, 2, size=1500, dtype=np.uint8)
        received = np.concatenate([frame["bits"], noisy_tail])

        hinted = search_interleaver(
            received,
            start_offset=sync_len,
            frame_bits=frame["frame_bits"],
            max_bits=100_000,
            time_budget_s=120.0,
        )
        assert hinted["validated"] is True
        assert bytes.fromhex(hinted["best"]["payload_hex"]) == MESSAGE


class TestCrcFramingEdgeCases:
    """Regression guards for the CRC framing scanners."""

    def test_strict_scanner_finds_the_frame_before_the_padding(self):
        from rf_analyzer.core.fec import crc16_append

        framed = crc16_append(MESSAGE)
        buf = framed + b"\x00" * 11  # block-interleaver zero fill
        ok, payload = scan_crc16_strict(buf)
        assert ok is True
        assert payload == MESSAGE

    def test_strict_scanner_does_not_invent_a_payload_from_noise(self):
        rng = np.random.default_rng(3)
        noise = rng.integers(0, 256, size=400, dtype=np.uint8).tobytes()
        assert scan_crc16_strict(noise)[0] is False

    def test_degenerate_longer_framing_is_not_chosen(self):
        """A zero byte after the frame must not yield a one-byte-longer payload.

        For CRC-16/CCITT-FALSE ``crc16(M || C_hi) == C_lo << 8`` always holds, so
        ``[M][C_hi]`` with ``[C_lo][0x00]`` as its CRC validates too. The scanner
        must anchor on the padding-stripped end and return ``M``, not ``M||C_hi``.
        """
        from rf_analyzer.core.fec import crc16_append, crc16_ccitt

        framed = crc16_append(MESSAGE)
        c_lo = framed[-1]
        assert crc16_ccitt(framed[:-1]) == (c_lo << 8), "degenerate case precondition"

        for pad in (b"\x00", b"\x00\x00", b"\x00" * 7):
            ok, payload = scan_crc16_strict(framed + pad)
            assert ok is True
            assert payload == MESSAGE, f"picked the degenerate framing with pad={pad!r}"

    def test_free_scanner_still_finds_a_mid_buffer_frame(self):
        """RS/coded streams put the CRC mid-buffer; the free scan must still work."""
        from rf_analyzer.core.fec import crc16_append

        framed = crc16_append(MESSAGE)
        buf = framed + bytes(range(32))  # parity symbols follow the CRC
        ok, payload = scan_crc16(buf)
        assert ok is True
        assert payload == MESSAGE

    def test_crc16_check_rejects_short_input(self):
        assert crc16_check(b"")[0] is False
        assert crc16_check(b"A")[0] is False


class TestSliceHelpers:
    def test_slice_payload_strips_the_sync_word(self):
        frame = build_frame(MESSAGE, fec="none", interleaver="none")
        payload = slice_payload(frame["bits"], int(frame["sync_bits"].size))

        assert payload.size == frame["frame_bits"]
        ok, recovered = scan_crc16_strict(bits_to_bytes(payload))
        assert ok is True
        assert recovered == MESSAGE
