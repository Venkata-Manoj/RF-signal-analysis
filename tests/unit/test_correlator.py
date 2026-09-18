"""Unit tests for sync-word correlation (info.md §17.4)."""

from __future__ import annotations

import numpy as np
import pytest

from rf_analyzer.core.correlator import (
    find_header,
    hex_to_bits,
    min_matches_for_length,
    sliding_correlate,
)


def test_hex_to_bits():
    # Arrange / Act
    bits = hex_to_bits("0x1", bit_length=4)

    # Assert
    assert np.array_equal(bits, np.array([0, 0, 0, 1], dtype=np.uint8))


def test_sliding_correlate_finds_offset():
    # Arrange: 32-bit sync word — collision in random prefix is ~2^-32.
    # Seeded for determinism (same offset every run).
    np.random.seed(0)
    sync_bits = hex_to_bits("0x1ACFFC1D")
    random_prefix = np.random.randint(0, 2, size=100, dtype=np.uint8)
    random_suffix = np.random.randint(0, 2, size=100, dtype=np.uint8)

    received = np.concatenate([random_prefix, sync_bits, random_suffix])

    # Act
    offset, score = sliding_correlate(received, sync_bits)

    # Assert
    assert offset == 100
    assert score > 0.99


def test_find_header():
    # Arrange
    sync_bits = np.array([1, 1, 0, 0, 1, 0, 1, 0], dtype=np.uint8)
    prefix = np.zeros(50, dtype=np.uint8)
    suffix = np.ones(50, dtype=np.uint8)

    received = np.concatenate([prefix, sync_bits, suffix])

    # Act
    result = find_header(received, sync_bits, threshold=0.9)

    # Assert
    assert result["detected"] is True
    assert result["offset"] == 50


# --------------------------------------------------------------------------- #
# Constant-false-alarm-rate gating
# --------------------------------------------------------------------------- #
def test_min_matches_tightens_with_search_length():
    """The same score is weaker evidence on a longer capture."""
    short = min_matches_for_length(32, 1_001)
    long = min_matches_for_length(32, 499_969)
    assert short == 28
    assert long == 31
    assert long > short


def test_min_matches_meets_the_false_alarm_target():
    from math import comb

    for sync_length, positions in (
        (32, 1_001),
        (32, 100_000),
        (32, 5_000_000),
    ):
        k = min_matches_for_length(sync_length, positions)
        tail = sum(comb(sync_length, j) for j in range(k, sync_length + 1))
        expected = tail / 2.0**sync_length * positions
        assert expected <= 0.01 + 1e-12, (sync_length, positions, k, expected)


def test_min_matches_falls_back_to_an_exact_match_when_unreachable():
    """A 16-bit sync word over 50k positions cannot reach a 0.01 target.

    Even an exact match is expected 0.76 times by chance there, so the honest
    answer is "require an exact match" rather than pretending the target was
    met. The function documents this fallback.
    """
    from math import comb

    assert min_matches_for_length(16, 50_000) == 16

    exact_match_false_alarms = comb(16, 16) / 2.0**16 * 50_000
    assert exact_match_false_alarms > 0.01  # provably unreachable, not a bug


def test_min_matches_requires_an_exact_match_for_a_weak_sync_word():
    """An 8-bit sync word is too weak to tolerate any mismatch over 94 tries."""
    assert min_matches_for_length(8, 94) == 8


def test_min_matches_is_defensive_on_degenerate_input():
    assert min_matches_for_length(0, 100) == 0
    assert min_matches_for_length(32, 0) == 32
    assert min_matches_for_length(32, -5) == 32


def test_find_header_rejects_a_chance_match_on_a_long_capture():
    """Regression: an analog TV capture produced a 0.875 "header".

    At the fixed 0.85 threshold a 32-bit sync word over 500k positions expects
    ~4.8 chance matches, so a best score in the high 0.8s is noise, not a
    header. The exact score is not asserted: a random 500k-bit stream is
    *expected* to contain a 29/32 match somewhere (0.64 occurrences on
    average), which is precisely why the threshold has to account for length.
    """
    rng = np.random.default_rng(11)
    sync_bits = hex_to_bits("0x1ACFFC1D")

    received = rng.integers(0, 2, size=500_000, dtype=np.uint8)
    near = sync_bits.copy()
    near[:4] = 1 - near[:4]  # plant a 28/32 match
    received[222_451 : 222_451 + 32] = near

    result = find_header(received, sync_bits)

    assert result["score"] < 0.95  # only a chance-level match exists
    assert result["min_matches"] == 31
    assert result["min_score"] == pytest.approx(31 / 32)
    assert result["positions_searched"] == 500_000 - 32 + 1
    assert result["detected"] is False


def test_find_header_still_detects_a_real_sync_word_on_a_long_capture():
    rng = np.random.default_rng(12)
    sync_bits = hex_to_bits("0x1ACFFC1D")
    received = rng.integers(0, 2, size=500_000, dtype=np.uint8)
    received[300_000 : 300_000 + 32] = sync_bits

    result = find_header(received, sync_bits)

    assert result["detected"] is True
    assert result["offset"] == 300_000
    assert result["score"] == pytest.approx(1.0)


def test_find_header_tolerates_one_bit_error_on_a_short_capture():
    """Short captures keep the caller's permissive threshold."""
    rng = np.random.default_rng(13)
    sync_bits = hex_to_bits("0x1ACFFC1D")
    received = rng.integers(0, 2, size=1_000, dtype=np.uint8)
    damaged = sync_bits.copy()
    damaged[0] = 1 - damaged[0]
    received[500:532] = damaged

    result = find_header(received, sync_bits)

    assert result["min_matches"] == 28
    assert result["detected"] is True
    assert result["offset"] == 500
