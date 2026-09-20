"""Tests for demodulator module."""

import numpy as np
import pytest
from qpsk_src.demodulator import (
    QPSK_CONSTELLATION,
    modulate_qpsk,
    demodulate_qpsk,
    sync_signals,
    process_signal,
)


def test_modulate_demodulate_roundtrip():
    """Test that modulate_qpsk followed by demodulate_qpsk accurately reconstructs bits."""
    bits = np.random.randint(0, 2, 1000)
    symbols = modulate_qpsk(bits)
    demodulated_bits = demodulate_qpsk(symbols)
    assert np.array_equal(bits, demodulated_bits)


def test_modulate_known_bits():
    """Test known bit mapping to QPSK constellation."""
    bits = np.array([0, 0, 0, 1, 1, 0, 1, 1])
    symbols = modulate_qpsk(bits)
    expected = QPSK_CONSTELLATION
    np.testing.assert_allclose(symbols, expected)


def test_demodulate_known_symbols():
    """Test demodulating constellation points directly."""
    symbols = np.copy(QPSK_CONSTELLATION)
    bits = demodulate_qpsk(symbols)
    expected = np.array([0, 0, 0, 1, 1, 0, 1, 1])
    assert np.array_equal(bits, expected)


def test_demodulate_noisy_equivalence():
    """Verify that demodulate_qpsk matches minimum Euclidean distance for arbitrary noisy samples."""
    np.random.seed(42)
    # Generate 10000 noisy samples across all quadrants
    samples = (np.random.randn(10000) + 1j * np.random.randn(10000)).astype(np.complex128)
    
    # Reference Euclidean distance computation
    distances = np.abs(samples[:, None] - QPSK_CONSTELLATION[None, :])
    indices = np.argmin(distances, axis=1)
    ref_bits = np.zeros(len(indices) * 2, dtype=int)
    ref_bits[0::2] = (indices >> 1) & 1
    ref_bits[1::2] = indices & 1

    actual_bits = demodulate_qpsk(samples)
    assert np.array_equal(actual_bits, ref_bits)



def test_sync_signals_perfect_match():
    """Test sync_signals when rx matches tx with a positive delay."""
    bits = np.random.randint(0, 2, 200)
    tx = modulate_qpsk(bits)
    delay_offset = 15
    # Create rx by prepending zeros (delay = 15)
    rx = np.concatenate([np.zeros(delay_offset, dtype=np.complex128), tx])

    tx_s, rx_s, detected_delay = sync_signals(tx, rx)
    assert detected_delay == delay_offset
    assert len(tx_s) == len(rx_s)
    np.testing.assert_allclose(tx_s, rx_s, atol=1e-5)


def test_process_signal_clean_with_rotation():
    """Test process_signal on a clean signal subject to delay and phase rotation."""
    bits = np.random.randint(0, 2, 400)
    tx_ref = modulate_qpsk(bits)
    rotation = -1j
    delay = 10
    rx_signal = np.concatenate([np.zeros(delay, dtype=np.complex128), tx_ref * (1 / rotation)])

    res = process_signal(tx_ref, rx_signal)
    assert res["ber"] == 0.0
    assert res["detected_delay"] == delay
    assert res["phase_rotation"] == rotation


def test_empty_signals():
    """Test handling of empty inputs in demodulator functions."""
    assert len(modulate_qpsk(np.array([]))) == 0
    assert len(demodulate_qpsk(np.array([]))) == 0
    tx_s, rx_s, delay = sync_signals(np.array([]), np.array([]))
    assert len(tx_s) == 0 and len(rx_s) == 0 and delay == 0


def test_process_signal_with_evm_options():
    """Test process_signal with warm-up skip and best-window search."""
    np.random.seed(42)
    bits = np.random.randint(0, 2, 800)
    tx_ref = modulate_qpsk(bits)
    # Add heavy noise to first 50 symbols, then low noise with a pristine segment
    noise = (np.random.normal(0, 0.05, len(tx_ref)) + 1j * np.random.normal(0, 0.05, len(tx_ref)))
    noise[:50] = 2.0  # heavy warm-up distortion
    noise[150:250] = 0.001  # pristine block
    rx = tx_ref + noise

    res = process_signal(tx_ref, rx, skip_initial_symbols=50, best_window_symbols=100)
    assert res["ber"] < 0.05
    assert res["evm_rms_pct"] < 15.0
    assert res["evm_best_rms_pct"] < res["evm_rms_pct"]
    assert res["evm_delta_rms_pct"] > 0
    assert res["best_window_start"] is not None

