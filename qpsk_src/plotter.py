"""Plotting module for QPSK signal demodulation results.

Generates BER vs SNR and BER vs OSNR curves grouped by distance,
with options for mean lines and min-max shaded envelopes.
"""

from pathlib import Path
from typing import List, Optional, Union


def load_and_prepare_data(csv_path: Union[Path, str]):
    """Loads CSV results and ensures required numerical types."""
    import pandas as pd

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "distance_km" not in df.columns:
        raise ValueError("CSV missing required 'distance_km' column.")

    df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce")

    for col in ["ber", "evm_db", "evm_rms_pct", "evm_peak_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Round SNR and OSNR for clean grouping if present
    if "snr_db" in df.columns:
        df["snr_db"] = pd.to_numeric(df["snr_db"], errors="coerce").round(2)
    if "osnr_db" in df.columns:
        df["osnr_db"] = pd.to_numeric(df["osnr_db"], errors="coerce").round(2)

    return df.dropna(subset=["distance_km"])


def _aggregate_curve_data(
    dist_df,
    x_col: str,
    y_col: str,
    snr_bin_width: Optional[float] = None,
):
    """
    Aggregates metric values by X coordinate.
    If x_col is 'snr_db', continuous measurement jitter across repeated captures is smoothed by:
    1) Grouping by 'osnr_db' if available (nominal optical setpoint), computing mean SNR and mean metric.
    2) Binning by snr_bin_width (default 0.5 dB if osnr_db is unavailable or snr_bin_width is explicitly passed).
    Returns a DataFrame with columns: ['x_val', 'mean_val', 'min_val', 'max_val', 'count'] sorted by 'x_val'.
    """
    if x_col == "snr_db":
        if snr_bin_width is not None:
            bin_w = float(snr_bin_width)
            binned = (dist_df[x_col] / bin_w).round() * bin_w
            grouped = dist_df.groupby(binned)
        elif "osnr_db" in dist_df.columns and dist_df["osnr_db"].dropna().nunique() > 1:
            grouped = dist_df.groupby("osnr_db")
        else:
            bin_w = 0.5
            binned = (dist_df[x_col] / bin_w).round() * bin_w
            grouped = dist_df.groupby(binned)

        stats = (
            grouped.agg(
                x_val=(x_col, "mean"),
                mean_val=(y_col, "mean"),
                min_val=(y_col, "min"),
                max_val=(y_col, "max"),
                count=(y_col, "count"),
            )
            .reset_index(drop=True)
            .sort_values(by="x_val")
        )
    else:
        stats = (
            dist_df.groupby(x_col)
            .agg(
                x_val=(x_col, "mean"),
                mean_val=(y_col, "mean"),
                min_val=(y_col, "min"),
                max_val=(y_col, "max"),
                count=(y_col, "count"),
            )
            .reset_index(drop=True)
            .sort_values(by="x_val")
        )

    return stats


def _plot_metric_curve(
    df,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    output_file: Path,
    shaded: bool = False,
    log_y: bool = False,
    snr_bin_width: Optional[float] = None,
) -> Path:
    """Generic helper to plot a metric vs X (SNR/OSNR) for each distance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if (
        x_col not in df.columns
        or y_col not in df.columns
        or df[x_col].dropna().empty
        or df[y_col].dropna().empty
    ):
        return output_file

    valid_df = df.dropna(subset=[x_col, y_col])
    if valid_df.empty:
        return output_file

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    distances = sorted(valid_df["distance_km"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(distances), 1)))
    markers = ["o", "s", "^", "d", "v", "<", ">", "p", "*"]

    for idx, dist in enumerate(distances):
        dist_df = valid_df[valid_df["distance_km"] == dist]
        if dist_df.empty:
            continue

        stats = _aggregate_curve_data(
            dist_df=dist_df,
            x_col=x_col,
            y_col=y_col,
            snr_bin_width=snr_bin_width,
        )

        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        label = f"{dist:.1f} km" if dist != int(dist) else f"{int(dist)} km"

        ax.plot(
            stats["x_val"],
            stats["mean_val"],
            label=label,
            color=color,
            marker=marker,
            linewidth=2,
            markersize=6,
        )

        if shaded:
            ax.fill_between(
                stats["x_val"],
                stats["min_val"],
                stats["max_val"],
                color=color,
                alpha=0.25,
            )

    if log_y:
        ax.set_yscale("log", nonpositive="clip")

    ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(title="Distance", fontsize=10, title_fontsize=11, loc="best")

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file)
    plt.close(fig)

    return output_file


def plot_ber_curve(
    df,
    x_col: str,
    x_label: str,
    title: str,
    output_file: Path,
    shaded: bool = False,
    snr_bin_width: Optional[float] = None,
) -> Path:
    """Plots BER vs Metric (SNR or OSNR) for each distance."""
    has_positive = (df["ber"] > 0).any() if "ber" in df.columns else False
    return _plot_metric_curve(
        df=df,
        x_col=x_col,
        y_col="ber",
        x_label=x_label,
        y_label="Bit Error Rate (BER)",
        title=title,
        output_file=output_file,
        shaded=shaded,
        log_y=has_positive,
        snr_bin_width=snr_bin_width,
    )



def generate_ber_plots(
    csv_path: Union[Path, str],
    output_dir: Optional[Union[Path, str]] = None,
    snr_bin_width: Optional[float] = None,
) -> List[Path]:
    """Generates all 4 BER plot variations from a results CSV.

    Args:
        csv_path: Path to the input CSV file.
        output_dir: Path to directory where plots will be saved. Defaults to <csv_dir>/plots/.
        snr_bin_width: Optional bin width in dB for averaging SNR measurements.

    Returns:
        List of Paths to generated PNG files.
    """
    csv_path = Path(csv_path)
    df = load_and_prepare_data(csv_path)

    if output_dir is None:
        output_dir = csv_path.parent / "plots"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    # 1. BER vs SNR (Mean only)
    if "snr_db" in df.columns:
        p1 = plot_ber_curve(
            df=df,
            x_col="snr_db",
            x_label="SNR (dB)",
            title="BER vs SNR (Mean)",
            output_file=output_dir / "ber_vs_snr_mean.png",
            shaded=False,
            snr_bin_width=snr_bin_width,
        )
        generated_files.append(p1)

        # 2. BER vs SNR (Shaded Min-Max)
        p2 = plot_ber_curve(
            df=df,
            x_col="snr_db",
            x_label="SNR (dB)",
            title="BER vs SNR (Mean with Min-Max Range)",
            output_file=output_dir / "ber_vs_snr_shaded.png",
            shaded=True,
            snr_bin_width=snr_bin_width,
        )
        generated_files.append(p2)

    # 3. BER vs OSNR (Mean only)
    if "osnr_db" in df.columns:
        p3 = plot_ber_curve(
            df=df,
            x_col="osnr_db",
            x_label="OSNR (dB)",
            title="BER vs OSNR (Mean)",
            output_file=output_dir / "ber_vs_osnr_mean.png",
            shaded=False,
        )
        generated_files.append(p3)

        # 4. BER vs OSNR (Shaded Min-Max)
        p4 = plot_ber_curve(
            df=df,
            x_col="osnr_db",
            x_label="OSNR (dB)",
            title="BER vs OSNR (Mean with Min-Max Range)",
            output_file=output_dir / "ber_vs_osnr_shaded.png",
            shaded=True,
        )
        generated_files.append(p4)

    return generated_files


def plot_evm_curve(
    df,
    x_col: str,
    x_label: str,
    y_col: str,
    y_label: str,
    title: str,
    output_file: Path,
    shaded: bool = False,
    snr_bin_width: Optional[float] = None,
) -> Path:
    """Plots EVM vs Metric (SNR or OSNR) for each distance."""
    return _plot_metric_curve(
        df=df,
        x_col=x_col,
        y_col=y_col,
        x_label=x_label,
        y_label=y_label,
        title=title,
        output_file=output_file,
        shaded=shaded,
        log_y=False,
        snr_bin_width=snr_bin_width,
    )


def generate_evm_plots(
    csv_path: Union[Path, str],
    output_dir: Optional[Union[Path, str]] = None,
    snr_bin_width: Optional[float] = None,
) -> List[Path]:
    """Generates EVM plot variations (dB and RMS %) from a results CSV.

    Args:
        csv_path: Path to the input CSV file.
        output_dir: Path to directory where plots will be saved. Defaults to <csv_dir>/plots/.
        snr_bin_width: Optional bin width in dB for averaging SNR measurements.

    Returns:
        List of Paths to generated PNG files.
    """
    csv_path = Path(csv_path)
    df = load_and_prepare_data(csv_path)

    if output_dir is None:
        output_dir = csv_path.parent / "plots"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    # EVM (dB) vs SNR
    if "snr_db" in df.columns and "evm_db" in df.columns and not df["evm_db"].dropna().empty:
        p1 = plot_evm_curve(
            df=df,
            x_col="snr_db",
            x_label="SNR (dB)",
            y_col="evm_db",
            y_label="EVM (dB)",
            title="EVM vs SNR (Mean)",
            output_file=output_dir / "evm_vs_snr_mean.png",
            shaded=False,
            snr_bin_width=snr_bin_width,
        )
        generated_files.append(p1)

        p2 = plot_evm_curve(
            df=df,
            x_col="snr_db",
            x_label="SNR (dB)",
            y_col="evm_db",
            y_label="EVM (dB)",
            title="EVM vs SNR (Mean with Min-Max Range)",
            output_file=output_dir / "evm_vs_snr_shaded.png",
            shaded=True,
            snr_bin_width=snr_bin_width,
        )
        generated_files.append(p2)

    # EVM (dB) vs OSNR
    if "osnr_db" in df.columns and "evm_db" in df.columns and not df["evm_db"].dropna().empty:
        p3 = plot_evm_curve(
            df=df,
            x_col="osnr_db",
            x_label="OSNR (dB)",
            y_col="evm_db",
            y_label="EVM (dB)",
            title="EVM vs OSNR (Mean)",
            output_file=output_dir / "evm_vs_osnr_mean.png",
            shaded=False,
        )
        generated_files.append(p3)

        p4 = plot_evm_curve(
            df=df,
            x_col="osnr_db",
            x_label="OSNR (dB)",
            y_col="evm_db",
            y_label="EVM (dB)",
            title="EVM vs OSNR (Mean with Min-Max Range)",
            output_file=output_dir / "evm_vs_osnr_shaded.png",
            shaded=True,
        )
        generated_files.append(p4)

    # EVM RMS (%) vs SNR & OSNR
    if "snr_db" in df.columns and "evm_rms_pct" in df.columns and not df["evm_rms_pct"].dropna().empty:
        p5 = plot_evm_curve(
            df=df,
            x_col="snr_db",
            x_label="SNR (dB)",
            y_col="evm_rms_pct",
            y_label="EVM RMS (%)",
            title="EVM RMS (%) vs SNR (Mean with Range)",
            output_file=output_dir / "evm_pct_vs_snr_shaded.png",
            shaded=True,
            snr_bin_width=snr_bin_width,
        )
        generated_files.append(p5)

    if "osnr_db" in df.columns and "evm_rms_pct" in df.columns and not df["evm_rms_pct"].dropna().empty:
        p6 = plot_evm_curve(
            df=df,
            x_col="osnr_db",
            x_label="OSNR (dB)",
            y_col="evm_rms_pct",
            y_label="EVM RMS (%)",
            title="EVM RMS (%) vs OSNR (Mean with Range)",
            output_file=output_dir / "evm_pct_vs_osnr_shaded.png",
            shaded=True,
        )
        generated_files.append(p6)

    return generated_files


def generate_all_plots(
    csv_path: Union[Path, str],
    output_dir: Optional[Union[Path, str]] = None,
    snr_bin_width: Optional[float] = None,
) -> List[Path]:
    """Generates both BER and EVM plot figures.

    Args:
        csv_path: Path to the input CSV file.
        output_dir: Path to directory where plots will be saved.
        snr_bin_width: Optional bin width in dB for averaging SNR measurements.

    Returns:
        List of Paths to generated PNG files.
    """
    all_plots = []
    ber_plots = generate_ber_plots(csv_path=csv_path, output_dir=output_dir, snr_bin_width=snr_bin_width)
    all_plots.extend(ber_plots)

    evm_plots = generate_evm_plots(csv_path=csv_path, output_dir=output_dir, snr_bin_width=snr_bin_width)
    all_plots.extend(evm_plots)

    return all_plots


def plot_constellation(
    rx_signal: np.ndarray,
    tx_ref: np.ndarray,
    output_file: Union[Path, str],
    skip_initial_symbols: int = 0,
    best_window_symbols: Optional[int] = None,
    title: Optional[str] = None,
    max_scatter_points: int = 50000,
) -> Path:
    """
    Plots QPSK constellation diagram showing EVM metrics, with optional warm-up symbol
    exclusion and best-window contrast.

    Args:
        rx_signal: Received complex samples (1D complex array).
        tx_ref: Transmitted reference symbols (1D complex array).
        output_file: Path to save the generated constellation image.
        skip_initial_symbols: Initial symbols to skip (warm-up / settling time).
        best_window_symbols: Block size (M) to search for cleanest EVM window.
        title: Optional plot title override.
        max_scatter_points: Subsample limit for scatter plotting to keep rendering fast.

    Returns:
        Path to output PNG image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from .demodulator import QPSK_CONSTELLATION, demodulate_qpsk, sync_signals_complex
    from .evm import compute_evm

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if len(rx_signal) == 0 or len(tx_ref) == 0:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        ax.text(0.5, 0.5, "Empty Signal Data", ha="center", va="center")
        plt.savefig(output_file)
        plt.close(fig)
        return output_file

    # Time synchronization
    tx_s, rx_s, _ = sync_signals_complex(tx_ref, rx_signal)
    if len(tx_s) == 0 or len(rx_s) == 0:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        ax.text(0.5, 0.5, "Synchronization Failed", ha="center", va="center")
        plt.savefig(output_file)
        plt.close(fig)
        return output_file

    # Phase rotation resolution
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
    scale = np.sqrt(np.mean(np.abs(tx_eval) ** 2) / (np.mean(np.abs(rx_eval) ** 2) + 1e-12))
    rx_norm = rx_eval * scale

    # Compute EVM
    evm_res = compute_evm(
        rx_norm,
        tx_eval,
        skip_initial_symbols=0,
        best_window_symbols=best_window_symbols,
    )

    evm_rms = evm_res["evm_rms_pct"]
    evm_db = evm_res["evm_db"]

    has_best_window = (
        best_window_symbols is not None
        and evm_res.get("evm_best_rms_pct") is not None
        and evm_res.get("best_window_start") is not None
    )

    if has_best_window:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), dpi=150)

        # Left panel: Evaluated (Full / Warm-up skipped) Constellation
        pts = rx_norm
        if len(pts) > max_scatter_points:
            subsample_idx = np.linspace(0, len(pts) - 1, max_scatter_points, dtype=int)
            pts = pts[subsample_idx]

        ax1.scatter(pts.real, pts.imag, s=2, alpha=0.3, color="#1f77b4", label="RX Symbols")
        ax1.scatter(
            QPSK_CONSTELLATION.real,
            QPSK_CONSTELLATION.imag,
            s=80,
            marker="x",
            color="red",
            linewidth=2,
            label="Ideal QPSK",
            zorder=5,
        )
        ax1.set_title(
            f"Overall Constellation (N={len(rx_norm)})\nEVM RMS: {evm_rms:.2f}% ({evm_db:.2f} dB)",
            fontsize=11,
            fontweight="bold",
        )
        ax1.set_xlabel("In-Phase (I)", fontweight="bold")
        ax1.set_ylabel("Quadrature (Q)", fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.axis("equal")
        ax1.set_xlim(-1.8, 1.8)
        ax1.set_ylim(-1.8, 1.8)
        ax1.legend(loc="upper right", fontsize=9)

        # Right panel: Best Window Constellation
        w_start = int(evm_res["best_window_start"])
        w_len = int(best_window_symbols)
        best_pts = rx_norm[w_start : min(w_start + w_len, len(rx_norm))]
        if len(best_pts) > max_scatter_points:
            subsample_idx = np.linspace(0, len(best_pts) - 1, max_scatter_points, dtype=int)
            best_pts = best_pts[subsample_idx]

        best_rms = evm_res["evm_best_rms_pct"]
        best_db = evm_res["evm_best_db"]
        delta_rms = evm_res["evm_delta_rms_pct"]

        ax2.scatter(best_pts.real, best_pts.imag, s=3, alpha=0.4, color="#2ca02c", label=f"Best {w_len} Block")
        ax2.scatter(
            QPSK_CONSTELLATION.real,
            QPSK_CONSTELLATION.imag,
            s=80,
            marker="x",
            color="red",
            linewidth=2,
            label="Ideal QPSK",
            zorder=5,
        )
        ax2.set_title(
            f"Best Window ({w_len} syms, idx {w_start}-{w_start+w_len})\nEVM: {best_rms:.2f}% ({best_db:.2f} dB) | ΔEVM: {delta_rms:+.2f}%",
            fontsize=11,
            fontweight="bold",
        )
        ax2.set_xlabel("In-Phase (I)", fontweight="bold")
        ax2.set_ylabel("Quadrature (Q)", fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.axis("equal")
        ax2.set_xlim(-1.8, 1.8)
        ax2.set_ylim(-1.8, 1.8)
        ax2.legend(loc="upper right", fontsize=9)

        main_title = title or "QPSK Constellation & Noise Transient Analysis"
        fig.suptitle(main_title, fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
        pts = rx_norm
        if len(pts) > max_scatter_points:
            subsample_idx = np.linspace(0, len(pts) - 1, max_scatter_points, dtype=int)
            pts = pts[subsample_idx]

        ax.scatter(pts.real, pts.imag, s=2, alpha=0.3, color="#1f77b4", label="RX Symbols")
        ax.scatter(
            QPSK_CONSTELLATION.real,
            QPSK_CONSTELLATION.imag,
            s=80,
            marker="x",
            color="red",
            linewidth=2,
            label="Ideal QPSK",
            zorder=5,
        )
        main_title = title or f"QPSK Constellation Diagram\nEVM RMS: {evm_rms:.2f}% ({evm_db:.2f} dB)"
        ax.set_title(main_title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("In-Phase (I)", fontweight="bold")
        ax.set_ylabel("Quadrature (Q)", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.axis("equal")
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.legend(loc="upper right", fontsize=9)
        plt.tight_layout()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file)
    plt.close(fig)
    return output_file


