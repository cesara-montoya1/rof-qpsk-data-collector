"""EVM (Error Vector Magnitude) calculation module for modulated optical signals."""

from typing import Any, Dict, Union
import numpy as np


def compute_evm(
    rx_synced: np.ndarray,
    tx_synced: np.ndarray,
    eps: float = 1e-12,
    skip_initial_symbols: int = 0,
    best_window_symbols: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Computes Error Vector Magnitude (EVM) metrics between synchronized received
    and ideal reference symbol sequences.

    Formula:
        Scale normalization:
            P_tx = mean(|tx|^2)
            P_rx = mean(|rx|^2)
            scale = sqrt(P_tx / (P_rx + eps))
            rx_norm = rx * scale

        Error vector:
            e = rx_norm - tx

        EVM RMS (%):
            EVM_rms = sqrt( mean(|e|^2) / (P_ref + eps) ) * 100

        EVM (dB):
            EVM_db = 20 * log10( EVM_rms / 100 + eps )

        Peak EVM (%):
            EVM_peak = ( max(|e|) / sqrt(P_ref + eps) ) * 100

    Args:
        rx_synced: 1D complex array of received symbols (time- and phase-aligned).
        tx_synced: 1D complex array of transmitted reference symbols.
        eps: Small constant for numerical stability to prevent division/log by zero.
        skip_initial_symbols: Number of initial symbols to ignore to allow equalizer/PLL/AGC warm-up.
        best_window_symbols: Optional contiguous block size (M) to find the lowest-noise EVM window.

    Returns:
        Dict containing:
            - 'evm_rms_pct': RMS EVM as percentage (%) over evaluated symbols
            - 'evm_db': RMS EVM in decibels (dB)
            - 'evm_peak_pct': Peak EVM as percentage (%)
            - 'evm_best_rms_pct': Best-window RMS EVM (%) if best_window_symbols specified, else None
            - 'evm_best_db': Best-window EVM in dB if best_window_symbols specified, else None
            - 'evm_delta_rms_pct': Difference (overall - best) quantifying noise burst impact
            - 'evm_delta_db': Difference in dB (overall - best)
            - 'best_window_start': Starting index of best window, or None
            - 'evaluated_symbols': Number of symbols evaluated
    """
    n = min(len(rx_synced), len(tx_synced))
    if n == 0 or skip_initial_symbols >= n:
        return {
            "evm_rms_pct": float("nan"),
            "evm_db": float("nan"),
            "evm_peak_pct": float("nan"),
            "evm_best_rms_pct": None if best_window_symbols is None else float("nan"),
            "evm_best_db": None if best_window_symbols is None else float("nan"),
            "evm_delta_rms_pct": None if best_window_symbols is None else float("nan"),
            "evm_delta_db": None if best_window_symbols is None else float("nan"),
            "best_window_start": None,
            "evaluated_symbols": 0,
        }

    # Discard warm-up symbols
    start_idx = max(0, int(skip_initial_symbols))
    rx_s = np.asarray(rx_synced[start_idx:n], dtype=np.complex128)
    tx_s = np.asarray(tx_synced[start_idx:n], dtype=np.complex128)
    evaluated_count = len(rx_s)

    if evaluated_count == 0:
        return {
            "evm_rms_pct": float("nan"),
            "evm_db": float("nan"),
            "evm_peak_pct": float("nan"),
            "evm_best_rms_pct": None if best_window_symbols is None else float("nan"),
            "evm_best_db": None if best_window_symbols is None else float("nan"),
            "evm_delta_rms_pct": None if best_window_symbols is None else float("nan"),
            "evm_delta_db": None if best_window_symbols is None else float("nan"),
            "best_window_start": None,
            "evaluated_symbols": 0,
        }

    # Power scale normalization
    tx_power = float(np.mean(np.abs(tx_s) ** 2))
    rx_power = float(np.mean(np.abs(rx_s) ** 2))
    scale = np.sqrt(tx_power / (rx_power + eps))
    rx_norm = rx_s * scale

    # Error vectors
    error_vectors = rx_norm - tx_s
    err_sq = np.abs(error_vectors) ** 2

    # Reference power
    p_ref = tx_power if tx_power > eps else float(np.mean(np.abs(tx_s) ** 2)) + eps

    # EVM RMS (%)
    mean_err_power = float(np.mean(err_sq))
    evm_rms_pct = float(np.sqrt(mean_err_power / p_ref) * 100.0)

    # EVM (dB)
    evm_rms_frac = evm_rms_pct / 100.0
    evm_db = float(20.0 * np.log10(evm_rms_frac + eps))

    # Peak EVM (%)
    max_err = float(np.max(np.abs(error_vectors)))
    evm_peak_pct = float((max_err / np.sqrt(p_ref)) * 100.0)

    # Best-window analysis for transient noise detection
    evm_best_rms_pct = None
    evm_best_db = None
    evm_delta_rms_pct = None
    evm_delta_db = None
    best_window_start = None

    if best_window_symbols is not None:
        m = int(best_window_symbols)
        if m > 0 and m <= evaluated_count:
            # Fast vectorized moving sum via convolution
            window_sums = np.convolve(err_sq, np.ones(m, dtype=np.float64), mode="valid")
            best_window_start = int(np.argmin(window_sums))
            best_mean_err = float(window_sums[best_window_start] / m)
            evm_best_rms_pct = float(np.sqrt(best_mean_err / p_ref) * 100.0)
            evm_best_db = float(20.0 * np.log10(evm_best_rms_pct / 100.0 + eps))
            evm_delta_rms_pct = float(evm_rms_pct - evm_best_rms_pct)
            evm_delta_db = float(evm_db - evm_best_db)
        elif m > evaluated_count and evaluated_count > 0:
            # Window size exceeds signal length: fallback to entire evaluated signal
            evm_best_rms_pct = evm_rms_pct
            evm_best_db = evm_db
            evm_delta_rms_pct = 0.0
            evm_delta_db = 0.0
            best_window_start = 0

    return {
        "evm_rms_pct": evm_rms_pct,
        "evm_db": evm_db,
        "evm_peak_pct": evm_peak_pct,
        "evm_best_rms_pct": evm_best_rms_pct,
        "evm_best_db": evm_best_db,
        "evm_delta_rms_pct": evm_delta_rms_pct,
        "evm_delta_db": evm_delta_db,
        "best_window_start": best_window_start,
        "evaluated_symbols": evaluated_count,
    }


def compute_theoretical_evm_from_snr(
    snr_db: Union[float, np.ndarray]
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calculates theoretical EVM from SNR assuming AWGN channel.

    EVM_rms (%) = 100 / sqrt(10^(SNR_db / 10))
    EVM (dB) = -SNR_db

    Args:
        snr_db: SNR in decibels (scalar or numpy array).

    Returns:
        Dict with 'evm_rms_pct' and 'evm_db'.
    """
    snr_arr = np.asarray(snr_db, dtype=float)
    snr_lin = 10.0 ** (snr_arr / 10.0)
    evm_rms_pct = 100.0 / np.sqrt(snr_lin)
    evm_db = -snr_arr

    if np.ndim(snr_db) == 0:
        return {"evm_rms_pct": float(evm_rms_pct), "evm_db": float(evm_db)}
    return {"evm_rms_pct": evm_rms_pct, "evm_db": evm_db}
