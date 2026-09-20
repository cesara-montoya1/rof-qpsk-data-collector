"""QPSK Demodulation, Modulation, Synchronization, and Signal Processing Core."""

from typing import Any, Dict, Optional, Tuple
import numpy as np

from .evm import compute_evm

# QPSK Constellation mapping:
# 00 -> 0.707 + 0.707j
# 01 -> -0.707 + 0.707j
# 10 -> -0.707 - 0.707j
# 11 -> 0.707 - 0.707j
INV_SQRT2 = float(1.0 / np.sqrt(2.0))
QPSK_CONSTELLATION = np.array(
    [
        INV_SQRT2 + 1j * INV_SQRT2,
        -INV_SQRT2 + 1j * INV_SQRT2,
        -INV_SQRT2 - 1j * INV_SQRT2,
        INV_SQRT2 - 1j * INV_SQRT2,
    ],
    dtype=np.complex128,
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
    Demodulates a 1D array of complex QPSK samples into bits using optimal quadrant decision.

    Bit mapping matches QPSK constellation:
    - Q1 (I >= 0, Q >= 0) -> 00
    - Q2 (I < 0,  Q >= 0) -> 01
    - Q3 (I < 0,  Q < 0)  -> 10
    - Q4 (I >= 0, Q < 0)  -> 11

    Args:
        signal: 1D numpy array of complex QPSK samples.

    Returns:
        1D numpy array of bits (0 or 1).
    """
    sig_arr = np.asarray(signal)
    if len(sig_arr) == 0:
        return np.array([], dtype=int)

    real_neg = sig_arr.real < 0
    imag_neg = sig_arr.imag < 0

    bits = np.empty(len(sig_arr) * 2, dtype=int)
    bits[0::2] = imag_neg
    bits[1::2] = imag_neg ^ real_neg
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


def sync_signals_complex(
    tx: np.ndarray, rx: np.ndarray, eps: float = 1e-9
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Synchronizes tx and rx signals using fast FFT complex cross-correlation magnitude.
    Delay estimation is robust against carrier phase offset and runs in O(N log N).
    """
    n_tx, n_rx = len(tx), len(rx)
    if n_tx == 0 or n_rx == 0:
        return np.array([], dtype=np.complex128), np.array([], dtype=np.complex128), 0

    tx_norm = (tx - np.mean(tx)) / (np.std(tx) + eps)
    rx_norm = (rx - np.mean(rx)) / (np.std(rx) + eps)

    # FFT-based cross correlation O(N log N)
    conv_len = n_rx + n_tx - 1
    fft_len = 1 << conv_len.bit_length()

    fft_rx = np.fft.fft(rx_norm, fft_len)
    fft_tx = np.fft.fft(tx_norm, fft_len)
    corr_fft = np.fft.ifft(fft_rx * np.conj(fft_tx))

    # Unwrap circular lags: negative lags [- (n_tx-1) .. -1] then positive [0 .. n_rx-1]
    corr_full = np.concatenate([corr_fft[fft_len - (n_tx - 1) :], corr_fft[:n_rx]])
    total_corr = np.abs(corr_full)

    lags = np.arange(-n_tx + 1, n_rx)
    delay = int(lags[np.argmax(total_corr)])

    if delay < 0:
        rx_sync = rx[0 : n_tx + delay]
        tx_sync = tx[-delay : -delay + len(rx_sync)]
    else:
        rx_sync = rx[delay : delay + n_tx]
        tx_sync = tx[: len(rx_sync)]

    return tx_sync, rx_sync, delay


def process_signal(
    tx_ref: np.ndarray,
    rx_signal: np.ndarray,
    skip_initial_symbols: int = 0,
    best_window_symbols: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Finds the optimal delay and phase rotation (1, 1j, -1, -1j) producing
    the minimum Bit Error Rate (BER), and computes Error Vector Magnitude (EVM).

    Args:
        tx_ref: 1D complex array of transmitted reference symbols.
        rx_signal: 1D complex array of received signal samples.
        skip_initial_symbols: Number of initial symbols to ignore to allow equalizer/PLL/AGC warm-up.
        best_window_symbols: Optional contiguous block size (M) to find the lowest-noise EVM window.

    Returns:
        Dict containing:
            - 'ber': Minimum BER achieved (float)
            - 'detected_delay': Estimated delay in samples (int)
            - 'phase_rotation': Phase rotation factor that produced minimum BER (complex)
            - 'evm_rms_pct': RMS EVM in percent (%)
            - 'evm_db': RMS EVM in dB
            - 'evm_peak_pct': Peak EVM in percent (%)
            - 'evm_best_rms_pct': Best-window RMS EVM (%) or None
            - 'evm_best_db': Best-window EVM in dB or None
            - 'evm_delta_rms_pct': Delta EVM (%) or None
            - 'evm_delta_db': Delta EVM in dB or None
            - 'best_window_start': Start index of best window or None
    """
    rotations = [1, 1j, -1, -1j]
    best_ber = 1.0
    best_delay = 0
    best_rotation = 1
    best_tx_s = None
    best_rx_s = None

    if len(tx_ref) == 0 or len(rx_signal) == 0:
        return {
            "ber": 1.0,
            "detected_delay": 0,
            "phase_rotation": complex(1),
            "evm_rms_pct": float("nan"),
            "evm_db": float("nan"),
            "evm_peak_pct": float("nan"),
            "evm_best_rms_pct": None if best_window_symbols is None else float("nan"),
            "evm_best_db": None if best_window_symbols is None else float("nan"),
            "evm_delta_rms_pct": None if best_window_symbols is None else float("nan"),
            "evm_delta_db": None if best_window_symbols is None else float("nan"),
            "best_window_start": None,
        }

    tx_s, rx_s, delay = sync_signals_complex(tx_ref, rx_signal)
    b_ref = demodulate_qpsk(tx_s)
    n_ref = len(b_ref)
    bit_skip = 2 * max(0, int(skip_initial_symbols))

    for rot in rotations:
        rx_rotated = rx_s * rot
        b_rx = demodulate_qpsk(rx_rotated)

        n = min(n_ref, len(b_rx))
        if n <= bit_skip:
            continue

        ber = float(np.mean(b_ref[bit_skip:n] != b_rx[bit_skip:n]))
        if ber < best_ber:
            best_ber = ber
            best_delay = delay
            best_rotation = rot
            best_tx_s = tx_s
            best_rx_s = rx_rotated

    if best_tx_s is not None and best_rx_s is not None and len(best_tx_s) > 0 and len(best_rx_s) > 0:
        n_sym = min(len(best_tx_s), len(best_rx_s))
        evm_res = compute_evm(
            best_rx_s[:n_sym],
            best_tx_s[:n_sym],
            skip_initial_symbols=skip_initial_symbols,
            best_window_symbols=best_window_symbols,
        )
    else:
        evm_res = {
            "evm_rms_pct": float("nan"),
            "evm_db": float("nan"),
            "evm_peak_pct": float("nan"),
            "evm_best_rms_pct": None if best_window_symbols is None else float("nan"),
            "evm_best_db": None if best_window_symbols is None else float("nan"),
            "evm_delta_rms_pct": None if best_window_symbols is None else float("nan"),
            "evm_delta_db": None if best_window_symbols is None else float("nan"),
            "best_window_start": None,
        }

    return {
        "ber": best_ber,
        "detected_delay": int(best_delay),
        "phase_rotation": complex(best_rotation),
        "evm_rms_pct": evm_res["evm_rms_pct"],
        "evm_db": evm_res["evm_db"],
        "evm_peak_pct": evm_res["evm_peak_pct"],
        "evm_best_rms_pct": evm_res.get("evm_best_rms_pct"),
        "evm_best_db": evm_res.get("evm_best_db"),
        "evm_delta_rms_pct": evm_res.get("evm_delta_rms_pct"),
        "evm_delta_db": evm_res.get("evm_delta_db"),
        "best_window_start": evm_res.get("best_window_start"),
    }

