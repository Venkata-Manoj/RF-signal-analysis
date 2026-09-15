"""Tests for the Higher-Order Cumulant (HOC) modulation classifier."""

from __future__ import annotations

import numpy as np

from rf_analyzer.core.classifier import classify_modulation, compute_cumulants


def test_compute_cumulants_keys():
    samples = np.random.randn(500) + 1j * np.random.randn(500)
    c = compute_cumulants(samples)
    assert "C20" in c
    assert "C21" in c
    assert "C40" in c
    assert "C42" in c


def test_classify_bpsk_clean():
    # Clean BPSK has C40 ≈ 2.0
    bits = np.random.randint(0, 2, 1000)
    symbols = (2.0 * bits - 1.0).astype(np.complex64)
    # Add minimal noise to avoid pure real edge cases
    symbols += (
        np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols))
    ) * 0.01
    mod, conf, alts = classify_modulation(symbols)
    assert mod == "BPSK"
    assert conf >= 0.8


def test_classify_qpsk_clean():
    # Clean QPSK has |C40| ≈ 1.0
    bits = np.random.randint(0, 2, 2000)
    i = 2.0 * bits[0::2] - 1.0
    q = 2.0 * bits[1::2] - 1.0
    symbols = ((i + 1j * q) / np.sqrt(2.0)).astype(np.complex64)
    symbols += (
        np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols))
    ) * 0.01
    mod, conf, alts = classify_modulation(symbols)
    assert mod == "QPSK"
    assert conf >= 0.75


def test_classify_short_input():
    mod, conf, alts = classify_modulation(np.array([1.0 + 1j]))
    assert mod == "UNKNOWN"
    assert conf == 0.5
