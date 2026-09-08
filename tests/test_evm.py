"""Unit tests for EVM (Error Vector Magnitude) calculation module."""

import numpy as np
import pytest

from qpsk_src.evm import compute_evm, compute_theoretical_evm_from_snr


def test_compute_evm_perfect_match():
    """Identical TX and RX signals should have near-zero EVM."""
    # 100 QPSK symbols
    tx = np.array([0.707 + 0.707j, -0.707 + 0.707j, -0.707 - 0.707j, 0.707 - 0.707j] * 25)
    rx = tx.copy()

    metrics = compute_evm(rx, tx)
    assert "evm_rms_pct" in metrics
    assert "evm_db" in metrics
    assert "evm_peak_pct" in metrics

    assert metrics["evm_rms_pct"] < 1e-4
    assert metrics["evm_peak_pct"] < 1e-4
    assert metrics["evm_db"] < -100.0


def test_compute_evm_scale_invariance():
    """Scaling RX by an arbitrary positive gain should not alter EVM because of power normalization."""
    tx = np.array([0.707 + 0.707j, -0.707 + 0.707j, -0.707 - 0.707j, 0.707 - 0.707j] * 50)
    noise = (np.random.randn(len(tx)) + 1j * np.random.randn(len(tx))) * 0.05
    rx1 = tx + noise
    rx2 = rx1 * 5.0  # arbitrary amplitude scaling

    m1 = compute_evm(rx1, tx)
    m2 = compute_evm(rx2, tx)

    assert np.isclose(m1["evm_rms_pct"], m2["evm_rms_pct"], rtol=1e-3)
    assert np.isclose(m1["evm_db"], m2["evm_db"], rtol=1e-3)
    assert np.isclose(m1["evm_peak_pct"], m2["evm_peak_pct"], rtol=1e-3)


def test_compute_evm_known_noise():
    """Verify EVM RMS against known noise variance."""
    np.random.seed(42)
    tx = np.array([0.707 + 0.707j, -0.707 + 0.707j, -0.707 - 0.707j, 0.707 - 0.707j] * 1000)
    
    # Add noise with relative standard deviation ~ 0.1 (10% EVM)
    noise = (np.random.normal(0, 0.0707, len(tx)) + 1j * np.random.normal(0, 0.0707, len(tx)))
    rx = tx + noise

    metrics = compute_evm(rx, tx)
    # Expected EVM RMS is approximately 10% (+- 1%)
    assert 8.0 < metrics["evm_rms_pct"] < 12.0
    assert -22.0 < metrics["evm_db"] < -18.0
    assert metrics["evm_peak_pct"] > metrics["evm_rms_pct"]


def test_compute_evm_empty_signals():
    """Empty signals should return NaN without raising errors."""
    empty = np.array([], dtype=np.complex128)
    metrics = compute_evm(empty, empty)
    assert np.isnan(metrics["evm_rms_pct"])
    assert np.isnan(metrics["evm_db"])
    assert np.isnan(metrics["evm_peak_pct"])


def test_compute_theoretical_evm():
    """Theoretical EVM from SNR: EVM = 100 / sqrt(10^(SNR/10))."""
    res = compute_theoretical_evm_from_snr(20.0)
    # SNR = 20 dB -> lin = 100 -> evm_rms_pct = 10%
    assert np.isclose(res["evm_rms_pct"], 10.0, atol=0.1)
    assert np.isclose(res["evm_db"], -20.0, atol=0.1)
