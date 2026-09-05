"""Unit tests for sync-word correlation (info.md §17.4)."""

from __future__ import annotations

import numpy as np

from rf_analyzer.core.correlator import find_header, hex_to_bits, sliding_correlate


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
