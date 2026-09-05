"""Shared pytest fixtures. Full data spec: info.md §15."""

from __future__ import annotations

from pathlib import Path

import pytest

# Canonical synthetic-data sample rate (info.md §15.1).
SAMPLE_RATE = 100_000

# Canonical sync word used by synthetic data + pipeline tests.
SYNC_WORD = "0x1ACFFC1D"


@pytest.fixture
def sample_rate() -> float:
    """Canonical sample rate for synthetic-signal tests."""
    return float(SAMPLE_RATE)


@pytest.fixture
def sync_word() -> str:
    """Canonical sync word hex string."""
    return SYNC_WORD


@pytest.fixture
def sample_data_dir() -> Path:
    """Repo-root sample_data directory (synthetic IQ/WAV artifacts)."""
    # tests/conftest.py -> parents[1] is repo root.
    return Path(__file__).resolve().parents[1] / "sample_data"
