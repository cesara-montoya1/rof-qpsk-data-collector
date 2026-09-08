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
