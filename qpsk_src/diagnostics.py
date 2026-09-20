"""Advanced channel and signal diagnostics module for optical/wireless QPSK links.

Provides specialized representations to evaluate channel physics:
- Polar EVM decomposition (Phase noise EVM_theta vs Amplitude noise EVM_r)
- Error Vector Power Spectral Density (EVM PSD vs frequency)
- 2D Constellation density heatmaps (KDE/histogram)
- Empirical BER vs EVM with theoretical AWGN bound
- Time-domain bit error burstiness profiles
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from .demodulator import QPSK_CONSTELLATION, demodulate_qpsk, sync_signals_complex
from .evm import compute_evm


def compute_polar_evm(
    rx_norm: np.ndarray,
    tx: np.ndarray,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Decomposes EVM into orthogonal polar coordinates:
    - Radial / Amplitude error (EVM_r): Photodiode thermal noise, ASE, attenuation.
    - Tangential / Phase error (EVM_theta): Laser linewidth, phase noise, USRP PLL jitter.

    Args:
        rx_norm: 1D complex array of power-normalized received symbols.
        tx: 1D complex array of ideal reference symbols.
        eps: Numerical stability constant.

    Returns:
        Dict with 'evm_r_pct', 'evm_theta_pct', 'phase_jitter_deg', 'phase_amplitude_ratio_db'.
    """
    n = min(len(rx_norm), len(tx))
    if n == 0:
        return {
            "evm_r_pct": float("nan"),
            "evm_theta_pct": float("nan"),
            "phase_jitter_deg": float("nan"),
            "phase_amplitude_ratio_db": float("nan"),
        }

    rx_s = np.asarray(rx_norm[:n], dtype=np.complex128)
    tx_s = np.asarray(tx[:n], dtype=np.complex128)

    r_rx = np.abs(rx_s)
    r_tx = np.abs(tx_s)
    e_r = r_rx - r_tx
    p_ref = float(np.mean(r_tx**2)) + eps

    # Radial EVM (%)
    evm_r_pct = float(np.sqrt(np.mean(e_r**2) / p_ref) * 100.0)

    # Phase difference in radians: delta_theta = angle(rx * conj(tx))
    delta_theta = np.angle(rx_s * np.conj(tx_s))
    evm_theta_pct = float(np.sqrt(np.mean(delta_theta**2)) * 100.0)
    phase_jitter_deg = float(np.std(delta_theta) * (180.0 / np.pi))

    # Phase-to-amplitude noise ratio in dB
    panr_db = float(10.0 * np.log10((evm_theta_pct**2 + eps) / (evm_r_pct**2 + eps)))

    return {
        "evm_r_pct": evm_r_pct,
        "evm_theta_pct": evm_theta_pct,
        "phase_jitter_deg": phase_jitter_deg,
        "phase_amplitude_ratio_db": panr_db,
    }


def plot_ber_vs_evm_theoretical(
    df,
    output_file: Union[Path, str],
    title: Optional[str] = None,
) -> Path:
    """
    Plots empirical BER vs EVM RMS (%) against the theoretical AWGN QPSK bound.
    Reveals whether link errors are governed by Gaussian noise or phase slips / burst ISI.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.special import erfc

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    # Theoretical curve: BER = 0.5 * erfc( 1 / (sqrt(2) * (EVM/100)) )
    evm_grid = np.linspace(4.0, 45.0, 300)
    ber_theory = 0.5 * erfc(1.0 / (np.sqrt(2.0) * (evm_grid / 100.0)))

    ax.plot(
        evm_grid,
        ber_theory,
        "k--",
        linewidth=2.2,
        label="Theoretical AWGN Limit",
        zorder=2,
    )

    if "distance_km" in df.columns and "evm_rms_pct" in df.columns and "ber" in df.columns:
        valid_df = df.dropna(subset=["evm_rms_pct", "ber", "distance_km"])
        distances = sorted(valid_df["distance_km"].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(distances), 1)))
        markers = ["o", "s", "^", "d", "v", "<", ">", "p"]

        for idx, dist in enumerate(distances):
            dist_df = (
                valid_df[valid_df["distance_km"] == dist]
                .groupby("evm_rms_pct")["ber"]
                .mean()
                .reset_index()
                .sort_values(by="evm_rms_pct")
            )
            if dist_df.empty:
                continue

            dist_label = f"{dist:.1f} km" if dist != int(dist) else f"{int(dist)} km"
            ax.plot(
                dist_df["evm_rms_pct"],
                dist_df["ber"],
                marker=markers[idx % len(markers)],
                color=colors[idx % len(colors)],
                linewidth=1.8,
                markersize=6,
                label=f"Empirical ({dist_label})",
                zorder=3,
            )

    ax.set_yscale("log", nonpositive="clip")
    ax.set_ylim(1e-6, 1.0)
    ax.set_xlim(5.0, 40.0)
    ax.set_xlabel("EVM RMS (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Bit Error Rate (BER)", fontsize=11, fontweight="bold")
    ax.set_title(
        title or "BER vs EVM (Empirical vs Theoretical AWGN)",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close(fig)
    return output_file


def plot_error_vector_spectrum(
    rx_norm: np.ndarray,
    tx: np.ndarray,
    output_file: Union[Path, str],
    n_fft: int = 1024,
    title: Optional[str] = None,
) -> Path:
    """
    Computes and plots the Power Spectral Density (PSD) of the complex error vector.
    Identifies chromatic dispersion ripples, USRP filter roll-off, or RF cable reflections.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import welch

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    n = min(len(rx_norm), len(tx))
    error_vector = rx_norm[:n] - tx[:n]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    if n < 32:
        ax.text(0.5, 0.5, "Insufficient signal length for PSD", ha="center", va="center")
        plt.savefig(output_file)
        plt.close(fig)
        return output_file

    nperseg = min(n_fft, n)
    freqs, psd = welch(
        error_vector,
        nperseg=nperseg,
        return_onesided=False,
        scaling="density",
    )

    # Shift frequencies so 0 is in center: [-0.5, 0.5] cycles/sample
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)
    psd_db = 10.0 * np.log10(psd + 1e-15)

    ax.plot(freqs, psd_db, color="#1f77b4", linewidth=1.8, label="Error Vector PSD")
    mean_floor = float(np.mean(psd_db))
    ax.axhline(
        mean_floor,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"Mean Noise Floor ({mean_floor:.1f} dB)",
    )

    ax.set_xlabel("Normalized Frequency (f / fs)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Power Spectral Density (dB/Hz)", fontsize=11, fontweight="bold")
    ax.set_title(
        title or "Error Vector Power Spectral Density (Noise Spectrum)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close(fig)
    return output_file


def plot_constellation_density(
    rx_norm: np.ndarray,
    output_file: Union[Path, str],
    bins: int = 150,
    title: Optional[str] = None,
) -> Path:
    """
    Renders a 2D density heatmap (histogram) of the constellation.
    Highlights non-linear gain compression, DC leakage at origin, and I/Q imbalance.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)

    if len(rx_norm) == 0:
        ax.text(0.5, 0.5, "Empty signal", ha="center", va="center")
        plt.savefig(output_file)
        plt.close(fig)
        return output_file

    h, xedges, yedges = np.histogram2d(
        rx_norm.real,
        rx_norm.imag,
        bins=bins,
        range=[[-1.8, 1.8], [-1.8, 1.8]],
    )

    im = ax.imshow(
        np.log1p(h.T),
        origin="lower",
        extent=[-1.8, 1.8, -1.8, 1.8],
        cmap="inferno",
        aspect="equal",
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Log Symbol Density", fontsize=10)

    # Ideal constellation overlay
    ax.scatter(
        QPSK_CONSTELLATION.real,
        QPSK_CONSTELLATION.imag,
        s=80,
        marker="x",
        color="cyan",
        linewidth=2,
        label="Ideal QPSK",
        zorder=5,
    )

    ax.set_title(
        title or "QPSK Constellation 2D Density Heatmap",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel("In-Phase (I)", fontweight="bold")
    ax.set_ylabel("Quadrature (Q)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3, color="white")
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close(fig)
    return output_file


def plot_polar_evm(
    rx_norm: np.ndarray,
    tx: np.ndarray,
    output_file: Union[Path, str],
    title: Optional[str] = None,
) -> Path:
    """
    Generates a polar decomposition diagram contrasting Phase Jitter vs Amplitude Noise.
    Identifies whether laser linewidth or optical attenuation is the dominant impairment.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    metrics = compute_polar_evm(rx_norm, tx)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=150)

    n = min(len(rx_norm), len(tx))
    rx_s = rx_norm[:n]
    tx_s = tx[:n]
    e_r = np.abs(rx_s) - np.abs(tx_s)
    delta_deg = np.angle(rx_s * np.conj(tx_s)) * (180.0 / np.pi)

    # Subsample if large
    pts_idx = np.linspace(0, n - 1, min(n, 20000), dtype=int) if n > 0 else np.array([], dtype=int)
    if len(pts_idx) > 0:
        ax1.scatter(e_r[pts_idx], delta_deg[pts_idx], s=2, alpha=0.3, color="#1f77b4")
    
    max_e_r = max(float(np.max(np.abs(e_r))) if n > 0 else 0.1, 0.1)
    max_deg = max(float(np.max(np.abs(delta_deg))) if n > 0 else 5.0, 5.0)
    ax1.set_xlim(-1.2 * max_e_r, 1.2 * max_e_r)
    ax1.set_ylim(-1.2 * max_deg, 1.2 * max_deg)

    ax1.set_xlabel("Amplitude Error e_r", fontweight="bold")
    ax1.set_ylabel("Phase Error Δθ (degrees)", fontweight="bold")
    ax1.set_title("Polar Error Dispersion (e_r vs Δθ)", fontweight="bold", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Bar breakdown
    categories = ["Amplitude EVM (EVM_r)", "Phase EVM (EVM_θ)"]
    v_r = metrics["evm_r_pct"] if not np.isnan(metrics["evm_r_pct"]) else 0.0
    v_th = metrics["evm_theta_pct"] if not np.isnan(metrics["evm_theta_pct"]) else 0.0
    values = [v_r, v_th]
    colors = ["#1f77b4", "#ff7f0e"]

    bars = ax2.bar(categories, values, color=colors, alpha=0.85, width=0.5)
    max_val = max(max(values), 1.0)
    ax2.set_ylim(0, max_val * 1.3)

    for b in bars:
        h = b.get_height()
        ax2.text(
            b.get_x() + b.get_width() / 2.0,
            h + (max_val * 0.03),
            f"{h:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    panr = metrics["phase_amplitude_ratio_db"]
    panr_str = f"{panr:+.1f} dB" if not np.isnan(panr) else "0.0 dB"
    dominant = "Laser Phase Noise" if panr > 0 else "Optical / Thermal Loss"
    ax2.set_ylabel("EVM Component (%)", fontweight="bold")
    ax2.set_title(
        f"Impairment Breakdown | Dominant: {dominant}\n(Phase-to-Amp Ratio: {panr_str})",
        fontweight="bold",
        fontsize=11,
    )
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")

    main_title = title or "EVM Polar Decomposition (Phase vs Amplitude)"
    fig.suptitle(main_title, fontsize=13, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_file)
    plt.close(fig)
    return output_file


def plot_error_burst_profile(
    rx_bits: np.ndarray,
    tx_bits: np.ndarray,
    output_file: Union[Path, str],
    title: Optional[str] = None,
) -> Path:
    """
    Plots cumulative error accumulation over the bit sequence.
    Linear slope = Stationary Gaussian noise.
    Abrupt vertical steps = Fiber vibration, acoustic transient, or USRP buffer underflow.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    n = min(len(rx_bits), len(tx_bits))
    error_mask = (rx_bits[:n] != tx_bits[:n]).astype(int)
    cum_err = np.cumsum(error_mask)
    total_errs = int(cum_err[-1]) if n > 0 else 0
    ber = total_errs / n if n > 0 else 0.0

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    ax.plot(cum_err, color="#d62728", linewidth=1.8, label=f"Measured Errors (BER: {ber:.2e})")

    # Ideal uniform accumulation line
    if n > 0 and total_errs > 0:
        ref_line = np.linspace(0, total_errs, n)
        ax.plot(ref_line, "k--", linewidth=1.2, label="Uniform Stationary Noise Reference")

    ax.set_xlabel("Bit Index", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cumulative Bit Errors", fontsize=11, fontweight="bold")
    ax.set_title(
        title or "Bit Error Accumulation Profile (Burstiness Analysis)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close(fig)
    return output_file


def generate_dataset_diagnostics(
    csv_path: Union[Path, str],
    output_dir: Optional[Union[Path, str]] = None,
    subfolder: str = "diagnostics",
) -> List[Path]:
    """
    Generates dataset-wide channel diagnostic figures (e.g. BER vs EVM with theoretical bound).

    By default, saves into a dedicated subfolder ('diagnostics/') to avoid cluttering
    standard result folders. Can be configured to save directly in the parent directory
    by passing subfolder=''.

    Args:
        csv_path: Path to dataset results CSV file.
        output_dir: Base directory (defaults to <csv_dir>/plots).
        subfolder: Subdirectory name for extra plots (default: 'diagnostics').

    Returns:
        List of Paths to generated PNG files.
    """
    import pandas as pd

    csv_path = Path(csv_path)
    base_dir = Path(output_dir) if output_dir is not None else csv_path.parent / "plots"
    target_dir = base_dir / subfolder if subfolder else base_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    generated = []

    if "ber" in df.columns and "evm_rms_pct" in df.columns:
        p = plot_ber_vs_evm_theoretical(
            df=df,
            output_file=target_dir / "ber_vs_evm_theoretical.png",
        )
        generated.append(p)

    return generated


def generate_signal_diagnostics(
    rx_signal: np.ndarray,
    tx_ref: np.ndarray,
    output_dir: Optional[Union[Path, str]] = None,
    subfolder: str = "diagnostics",
    skip_initial_symbols: int = 0,
    best_window_symbols: Optional[int] = None,
) -> List[Path]:
    """
    Generates single-signal diagnostic figures:
    - Polar EVM breakdown (Phase vs Amplitude)
    - Error Vector Spectrum (PSD)
    - Constellation 2D Density Heatmap
    - Error Burst Profile

    By default, saves into a dedicated subfolder ('diagnostics/') to avoid cluttering
    standard result folders. Can be configured to save directly in the parent directory
    by passing subfolder=''.

    Args:
        rx_signal: 1D complex array of raw received samples.
        tx_ref: 1D complex array of transmitted reference symbols.
        output_dir: Base directory (defaults to ./plots).
        subfolder: Subdirectory name for extra plots (default: 'diagnostics').
        skip_initial_symbols: Initial warm-up symbols to ignore.
        best_window_symbols: Best block window size.

    Returns:
        List of Paths to generated PNG files.
    """
    base_dir = Path(output_dir) if output_dir is not None else Path(".") / "plots"
    target_dir = base_dir / subfolder if subfolder else base_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    generated = []

    # Synchronize and rotate
    tx_s, rx_s, _ = sync_signals_complex(tx_ref, rx_signal)
    if len(tx_s) == 0 or len(rx_s) == 0:
        return generated

    rotations = [1, 1j, -1, -1j]
    best_rot = 1
    best_ber = 1.0
    b_ref = demodulate_qpsk(tx_s)
    bit_skip = 2 * max(0, int(skip_initial_symbols))

    for rot in rotations:
        rx_rot = rx_s * rot
        b_rx = demodulate_qpsk(rx_rot)
        n = min(len(b_ref), len(b_rx))
        if n <= bit_skip:
            continue
        ber = float(np.mean(b_ref[bit_skip:n] != b_rx[bit_skip:n]))
        if ber < best_ber:
            best_ber = ber
            best_rot = rot

    rx_aligned = rx_s * best_rot
    n_sym = min(len(tx_s), len(rx_aligned))
    start_idx = max(0, int(skip_initial_symbols))
    if start_idx >= n_sym:
        start_idx = 0

    tx_eval = tx_s[start_idx:n_sym]
    rx_eval = rx_aligned[start_idx:n_sym]

    # Power scale normalization
    scale = np.sqrt(np.mean(np.abs(tx_eval)**2) / (np.mean(np.abs(rx_eval)**2) + 1e-12))
    rx_norm = rx_eval * scale

    # 1. Constellation 2D Density
    p1 = plot_constellation_density(
        rx_norm=rx_norm,
        output_file=target_dir / "constellation_density.png",
    )
    generated.append(p1)

    # 2. Polar EVM Breakdown
    p2 = plot_polar_evm(
        rx_norm=rx_norm,
        tx=tx_eval,
        output_file=target_dir / "polar_evm_breakdown.png",
    )
    generated.append(p2)

    # 3. Error Vector Spectrum (PSD)
    p3 = plot_error_vector_spectrum(
        rx_norm=rx_norm,
        tx=tx_eval,
        output_file=target_dir / "error_vector_spectrum.png",
    )
    generated.append(p3)

    # 4. Error Burst Profile
    rx_bits = demodulate_qpsk(rx_norm)
    tx_bits = demodulate_qpsk(tx_eval)
    p4 = plot_error_burst_profile(
        rx_bits=rx_bits,
        tx_bits=tx_bits,
        output_file=target_dir / "error_burst_profile.png",
    )
    generated.append(p4)

    return generated
