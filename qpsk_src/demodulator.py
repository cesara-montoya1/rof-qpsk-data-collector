"""QPSK Demodulation, Modulation, Synchronization, and Signal Processing Core."""

from typing import Any, Dict, Tuple
import numpy as np

# QPSK Constellation mapping:
# 00 -> 0.707 + 0.707j
# 01 -> -0.707 + 0.707j
# 10 -> -0.707 - 0.707j
# 11 -> 0.707 - 0.707j
QPSK_CONSTELLATION = np.array(
    [0.707 + 0.707j, -0.707 + 0.707j, -0.707 - 0.707j, 0.707 - 0.707j]
)


def modulate_qpsk(bits: np.ndarray) -> np.ndarray:
    """
    Modulates a 1D binary array into QPSK complex symbols.

    Bit pairs are mapped as follows:
    - 00 -> QPSK_CONSTELLATION[0]
    - 01 -> QPSK_CONSTELLATION[1]
    - 10 -> QPSK_CONSTELLATION[2]
    - 11 -> QPSK_CONSTELLATION[3]

    Args:
        bits: 1D array of binary values (0 or 1).

    Returns:
        1D numpy array of complex QPSK symbols.
    """
    bits_arr = np.asarray(bits, dtype=int)
    if len(bits_arr) == 0:
        return np.array([], dtype=np.complex128)

    remainder = len(bits_arr) % 2
    if remainder != 0:
        bits_arr = np.append(bits_arr, 0)

    reshaped = bits_arr.reshape(-1, 2)
    indices = (reshaped[:, 0] << 1) | reshaped[:, 1]
    return QPSK_CONSTELLATION[indices]


def demodulate_qpsk(signal: np.ndarray) -> np.ndarray:
    """
    Demodulates a 1D array of complex QPSK samples into bits using Minimum Distance decision.

    Args:
        signal: 1D numpy array of complex QPSK samples.

    Returns:
        1D numpy array of bits (0 or 1).
    """
    sig_arr = np.asarray(signal, dtype=np.complex128)
    if len(sig_arr) == 0:
        return np.array([], dtype=int)

    distances = np.abs(sig_arr[:, None] - QPSK_CONSTELLATION[None, :])
    indices = np.argmin(distances, axis=1)

    bits = np.zeros(len(indices) * 2, dtype=int)
    bits[0::2] = (indices >> 1) & 1
    bits[1::2] = indices & 1
    return bits


def sync_signals(
    tx: np.ndarray, rx: np.ndarray, eps: float = 1e-9
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Synchronizes tx and rx signals by cross-correlating I and Q components independently.

    Args:
        tx: 1D complex array representing transmitted reference symbols.
        rx: 1D complex array representing received signal samples.
        eps: Epsilon parameter for numerical stability during normalization.

    Returns:
        Tuple of (tx_sync, rx_sync, delay) where tx_sync and rx_sync are aligned slices.
    """
    if len(tx) == 0 or len(rx) == 0:
        return np.array([], dtype=np.complex128), np.array([], dtype=np.complex128), 0

    tx_i = (tx.real - np.mean(tx.real)) / (np.std(tx.real) + eps)
    tx_q = (tx.imag - np.mean(tx.imag)) / (np.std(tx.imag) + eps)
    rx_i = (rx.real - np.mean(rx.real)) / (np.std(rx.real) + eps)
    rx_q = (rx.imag - np.mean(rx.imag)) / (np.std(rx.imag) + eps)

    corr_ii = np.correlate(rx_i, tx_i, mode="full")
    corr_qq = np.correlate(rx_q, tx_q, mode="full")
    total_corr = corr_ii + corr_qq

    lags = np.arange(-len(tx) + 1, len(rx))
    delay = int(lags[np.argmax(total_corr)])

    if delay < 0:
        rx_sync = rx[0 : len(tx) + delay]
        tx_sync = tx[-delay : -delay + len(rx_sync)]
    else:
        rx_sync = rx[delay : delay + len(tx)]
        tx_sync = tx[: len(rx_sync)]

    return tx_sync, rx_sync, delay


def process_signal(tx_ref: np.ndarray, rx_signal: np.ndarray) -> Dict[str, Any]:
    """
    Sweeps phase rotations (1, 1j, -1, -1j) to find the best phase alignment
    producing the minimum Bit Error Rate (BER) and detected delay.

    Args:
        tx_ref: 1D complex array of transmitted reference symbols.
        rx_signal: 1D complex array of received signal samples.

    Returns:
        Dict containing:
            - 'ber': Minimum BER achieved (float)
            - 'detected_delay': Estimated delay in samples (int)
            - 'phase_rotation': Phase rotation factor that produced minimum BER (complex)
    """
    rotations = [1, 1j, -1, -1j]
    best_ber = 1.0
    best_delay = 0
    best_rotation = 1

    for rot in rotations:
        rx_rotated = rx_signal * rot
        tx_s, rx_s, delay = sync_signals(tx_ref, rx_rotated)

        b_ref = demodulate_qpsk(tx_s)
        b_rx = demodulate_qpsk(rx_s)

        n = min(len(b_ref), len(b_rx))
        if n == 0:
            continue

        ber = float(np.mean(b_ref[:n] != b_rx[:n]))
        if ber < best_ber:
            best_ber = ber
            best_delay = delay
            best_rotation = rot

    return {
        "ber": best_ber,
        "detected_delay": int(best_delay),
        "phase_rotation": complex(best_rotation),
    }
