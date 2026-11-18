"""Plotting module for 16-QAM signal demodulation results.

Generates BER vs SNR, BER vs OSNR, EVM curves, and 16-QAM constellation diagrams.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np


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
    If x_col is 'snr_db', groups by 'osnr_db' if available or bins by snr_bin_width.
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
    """Generates all 4 BER plot variations from a results CSV."""
    csv_path = Path(csv_path)
    df = load_and_prepare_data(csv_path)

    if output_dir is None:
        output_dir = csv_path.parent / "plots"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    if "snr_db" in df.columns:
        generated_files.append(
            plot_ber_curve(
                df=df,
                x_col="snr_db",
                x_label="SNR (dB)",
                title="BER vs SNR (Mean)",
                output_file=output_dir / "ber_vs_snr_mean.png",
                shaded=False,
                snr_bin_width=snr_bin_width,
            )
        )
        generated_files.append(
            plot_ber_curve(
                df=df,
                x_col="snr_db",
                x_label="SNR (dB)",
                title="BER vs SNR (Mean with Min-Max Range)",
                output_file=output_dir / "ber_vs_snr_shaded.png",
                shaded=True,
                snr_bin_width=snr_bin_width,
            )
        )

    if "osnr_db" in df.columns:
        generated_files.append(
            plot_ber_curve(
                df=df,
                x_col="osnr_db",
                x_label="OSNR (dB)",
                title="BER vs OSNR (Mean)",
                output_file=output_dir / "ber_vs_osnr_mean.png",
                shaded=False,
            )
        )
        generated_files.append(
            plot_ber_curve(
                df=df,
                x_col="osnr_db",
                x_label="OSNR (dB)",
                title="BER vs OSNR (Mean with Min-Max Range)",
                output_file=output_dir / "ber_vs_osnr_shaded.png",
                shaded=True,
            )
        )

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
    """Generates EVM plot variations (dB and RMS %) from a results CSV."""
    csv_path = Path(csv_path)
    df = load_and_prepare_data(csv_path)

    if output_dir is None:
        output_dir = csv_path.parent / "plots"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    if "snr_db" in df.columns and "evm_db" in df.columns and not df["evm_db"].dropna().empty:
        generated_files.append(
            plot_evm_curve(
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
        )
        generated_files.append(
            plot_evm_curve(
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
        )

    if "osnr_db" in df.columns and "evm_db" in df.columns and not df["evm_db"].dropna().empty:
        generated_files.append(
            plot_evm_curve(
                df=df,
                x_col="osnr_db",
                x_label="OSNR (dB)",
                y_col="evm_db",
                y_label="EVM (dB)",
                title="EVM vs OSNR (Mean)",
                output_file=output_dir / "evm_vs_osnr_mean.png",
                shaded=False,
            )
        )
        generated_files.append(
            plot_evm_curve(
                df=df,
                x_col="osnr_db",
                x_label="OSNR (dB)",
                y_col="evm_db",
                y_label="EVM (dB)",
                title="EVM vs OSNR (Mean with Min-Max Range)",
                output_file=output_dir / "evm_vs_osnr_shaded.png",
                shaded=True,
            )
        )

    if "snr_db" in df.columns and "evm_rms_pct" in df.columns and not df["evm_rms_pct"].dropna().empty:
        generated_files.append(
            plot_evm_curve(
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
        )

    if "osnr_db" in df.columns and "evm_rms_pct" in df.columns and not df["evm_rms_pct"].dropna().empty:
        generated_files.append(
            plot_evm_curve(
                df=df,
                x_col="osnr_db",
                x_label="OSNR (dB)",
                y_col="evm_rms_pct",
                y_label="EVM RMS (%)",
                title="EVM RMS (%) vs OSNR (Mean with Range)",
                output_file=output_dir / "evm_pct_vs_osnr_shaded.png",
                shaded=True,
            )
        )

    return generated_files


def generate_all_plots(
    csv_path: Union[Path, str],
    output_dir: Optional[Union[Path, str]] = None,
    snr_bin_width: Optional[float] = None,
) -> List[Path]:
    """Generates both BER and EVM plot figures."""
    all_plots = []
    all_plots.extend(
        generate_ber_plots(csv_path=csv_path, output_dir=output_dir, snr_bin_width=snr_bin_width)
    )
    all_plots.extend(
        generate_evm_plots(csv_path=csv_path, output_dir=output_dir, snr_bin_width=snr_bin_width)
    )
    return all_plots


def plot_constellation(
    rx_signal: np.ndarray,
    tx_ref: np.ndarray,
    output_file: Optional[Union[Path, str]] = None,
    skip_initial_symbols: int = 0,
    best_window_symbols: Optional[int] = None,
    title: Optional[str] = None,
    max_scatter_points: int = 50000,
    show: bool = False,
    return_metrics: bool = False,
) -> Union[Path, Tuple[Path, dict]]:
    """
    Plots 16-QAM constellation diagram showing BER and EVM metrics, with optional
    warm-up symbol exclusion and best-window contrast.
    """
    import os
    import tempfile
    import matplotlib

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if not (show and has_display):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .demodulator import (
        QAM16_CONSTELLATION,
        demodulate_16qam,
        normalize_rx_to_tx,
        sync_signals_complex,
    )
    from .evm import compute_evm

    if output_file is None:
        output_file = Path(tempfile.gettempdir()) / "16qam_constellation_check.png"
    else:
        output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    empty_metrics = {
        "ber": 1.0,
        "bit_errors": 0,
        "total_bits": 0,
        "evm_rms_pct": 100.0,
        "evm_db": 0.0,
        "evm_peak_pct": 100.0,
        "best_rotation": "0°",
        "detected_delay": 0,
        "output_file": output_file,
    }

    if len(rx_signal) == 0 or len(tx_ref) == 0:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        ax.text(0.5, 0.5, "Empty Signal Data", ha="center", va="center")
        plt.savefig(output_file)
        plt.close(fig)
        return (output_file, empty_metrics) if return_metrics else output_file

    tx_s, rx_s, detected_delay = sync_signals_complex(tx_ref, rx_signal)
    if len(tx_s) == 0 or len(rx_s) == 0:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        ax.text(0.5, 0.5, "Synchronization Failed", ha="center", va="center")
        plt.savefig(output_file)
        plt.close(fig)
        return (output_file, empty_metrics) if return_metrics else output_file

    rx_s_norm = normalize_rx_to_tx(rx_s, tx_s, skip_initial_symbols=skip_initial_symbols)
    rotations = [1, 1j, -1, -1j]
    rot_labels = {1: "0°", 1j: "90°", -1: "180°", -1j: "270°"}
    best_rot = 1
    best_ber = 1.0
    n_bit_errors = 0
    n_bits_eval = 0
    b_ref = demodulate_16qam(tx_s)
    bit_skip = 4 * max(0, int(skip_initial_symbols))

    for rot in rotations:
        rx_rot = rx_s_norm * rot
        b_rx = demodulate_16qam(rx_rot)
        n = min(len(b_ref), len(b_rx))
        if n <= bit_skip:
            continue
        errors = int(np.sum(b_ref[bit_skip:n] != b_rx[bit_skip:n]))
        total_eval = int(n - bit_skip)
        ber = float(errors / total_eval)
        if ber < best_ber:
            best_ber = ber
            best_rot = rot
            n_bit_errors = errors
            n_bits_eval = total_eval

    rx_aligned = rx_s_norm * best_rot
    n_sym = min(len(tx_s), len(rx_aligned))
    start_idx = max(0, int(skip_initial_symbols))
    if start_idx >= n_sym:
        start_idx = 0

    tx_eval = tx_s[start_idx:n_sym]
    rx_norm = normalize_rx_to_tx(rx_aligned[start_idx:n_sym], tx_eval)

    evm_res = compute_evm(
        rx_norm,
        tx_eval,
        skip_initial_symbols=0,
        best_window_symbols=best_window_symbols,
    )

    evm_rms = float(evm_res["evm_rms_pct"])
    evm_db = float(evm_res["evm_db"])
    evm_peak = float(evm_res.get("evm_peak_pct", 0.0))

    rot_str = rot_labels.get(best_rot, str(best_rot))
    if best_ber == 0.0:
        ber_str = "0.000000 (0 errors)"
    elif best_ber < 1e-4:
        ber_str = f"{best_ber:.2e} ({n_bit_errors} err)"
    else:
        ber_str = f"{best_ber:.6f} ({n_bit_errors} err)"

    info_box_text = (
        f"BER:      {ber_str}\n"
        f"EVM RMS:  {evm_rms:.2f}% ({evm_db:.2f} dB)\n"
        f"EVM Peak: {evm_peak:.2f}%\n"
        f"Rot: {rot_str} | Delay: {int(detected_delay):,}"
    )

    has_best_window = (
        best_window_symbols is not None
        and evm_res.get("evm_best_rms_pct") is not None
        and evm_res.get("best_window_start") is not None
    )

    if has_best_window:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), dpi=150)

        pts = rx_norm
        if len(pts) > max_scatter_points:
            subsample_idx = np.linspace(0, len(pts) - 1, max_scatter_points, dtype=int)
            pts = pts[subsample_idx]

        ax1.scatter(pts.real, pts.imag, s=2, alpha=0.3, color="#1f77b4", label="RX Symbols")
        ax1.scatter(
            QAM16_CONSTELLATION.real,
            QAM16_CONSTELLATION.imag,
            s=80,
            marker="x",
            color="red",
            linewidth=2,
            label="Ideal 16-QAM",
            zorder=5,
        )
        ax1.set_title(
            f"Overall Constellation (N={len(rx_norm)})\nBER: {ber_str} | EVM: {evm_rms:.2f}% ({evm_db:.2f} dB)",
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
        ax1.text(
            0.03,
            0.97,
            info_box_text,
            transform=ax1.transAxes,
            fontsize=8.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.88, edgecolor="#999999"),
            family="monospace",
        )

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
            QAM16_CONSTELLATION.real,
            QAM16_CONSTELLATION.imag,
            s=80,
            marker="x",
            color="red",
            linewidth=2,
            label="Ideal 16-QAM",
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

        main_title = title or f"16-QAM Constellation & Noise Transient Analysis (BER: {ber_str})"
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
            QAM16_CONSTELLATION.real,
            QAM16_CONSTELLATION.imag,
            s=80,
            marker="x",
            color="red",
            linewidth=2,
            label="Ideal 16-QAM",
            zorder=5,
        )
        main_title = title or f"16-QAM Constellation Diagram\nBER: {ber_str} | EVM RMS: {evm_rms:.2f}% ({evm_db:.2f} dB)"
        ax.set_title(main_title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("In-Phase (I)", fontweight="bold")
        ax.set_ylabel("Quadrature (Q)", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.axis("equal")
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.legend(loc="upper right", fontsize=9)
        ax.text(
            0.03,
            0.97,
            info_box_text,
            transform=ax.transAxes,
            fontsize=8.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.88, edgecolor="#999999"),
            family="monospace",
        )
        plt.tight_layout()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file)
    if show and has_display and matplotlib.get_backend().lower() != "agg":
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)

    metrics_dict = {
        "ber": best_ber,
        "bit_errors": n_bit_errors,
        "total_bits": n_bits_eval,
        "evm_rms_pct": evm_rms,
        "evm_db": evm_db,
        "evm_peak_pct": evm_peak,
        "best_rotation": rot_str,
        "detected_delay": int(detected_delay),
        "output_file": output_file,
    }

    if return_metrics:
        return output_file, metrics_dict
    return output_file
