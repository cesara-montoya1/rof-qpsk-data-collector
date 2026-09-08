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


def plot_ber_curve(
    df,
    x_col: str,
    x_label: str,
    title: str,
    output_file: Path,
    shaded: bool = False,
) -> Path:
    """Plots BER vs Metric (SNR or OSNR) for each distance.

    Args:
        df: Processed pandas DataFrame.
        x_col: Column name for X-axis ('snr_db' or 'osnr_db').
        x_label: Label for X-axis.
        title: Plot title.
        output_file: Output PNG filepath.
        shaded: If True, draws shaded min-max envelope around the mean.

    Returns:
        Path to saved PNG image.
    """
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for headless execution
    import matplotlib.pyplot as plt
    import numpy as np

    if x_col not in df.columns or df[x_col].dropna().empty:
        print(f"Skipping plot {output_file.name}: Column '{x_col}' has no valid data.")
        return output_file

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    distances = sorted(df["distance_km"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(distances), 1)))
    markers = ["o", "s", "^", "d", "v", "<", ">", "p", "*"]

    for idx, dist in enumerate(distances):
        dist_df = df[df["distance_km"] == dist]
        if dist_df.empty:
            continue

        # Group by metric and calculate stats
        stats = (
            dist_df.groupby(x_col)["ber"]
            .agg(mean_ber="mean", min_ber="min", max_ber="max", count="count")
            .reset_index()
            .sort_values(by=x_col)
        )

        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        label = f"{dist:.1f} km" if dist != int(dist) else f"{int(dist)} km"

        # Plot mean line
        ax.plot(
            stats[x_col],
            stats["mean_ber"],
            label=label,
            color=color,
            marker=marker,
            linewidth=2,
            markersize=6,
        )

        # Draw shaded envelope if requested
        if shaded:
            ax.fill_between(
                stats[x_col],
                stats["min_ber"],
                stats["max_ber"],
                color=color,
                alpha=0.25,
            )

    has_positive = (df["ber"] > 0).any() if "ber" in df.columns else False
    if has_positive:
        ax.set_yscale("log", nonpositive="clip")
    ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
    ax.set_ylabel("Bit Error Rate (BER)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(title="Distance", fontsize=10, title_fontsize=11, loc="best")

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file)
    plt.close(fig)

    return output_file


def generate_ber_plots(
    csv_path: Union[Path, str], output_dir: Optional[Union[Path, str]] = None
) -> List[Path]:
    """Generates all 4 BER plot variations from a results CSV.

    Args:
        csv_path: Path to the input CSV file.
        output_dir: Path to directory where plots will be saved. Defaults to <csv_dir>/plots/.

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
) -> Path:
    """Plots EVM vs Metric (SNR or OSNR) for each distance.

    Args:
        df: Processed pandas DataFrame.
        x_col: Column name for X-axis ('snr_db' or 'osnr_db').
        x_label: Label for X-axis.
        y_col: Column name for Y-axis ('evm_db' or 'evm_rms_pct').
        y_label: Label for Y-axis.
        title: Plot title.
        output_file: Output PNG filepath.
        shaded: If True, draws shaded min-max envelope around the mean.

    Returns:
        Path to saved PNG image.
    """
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

        stats = (
            dist_df.groupby(x_col)[y_col]
            .agg(mean_val="mean", min_val="min", max_val="max", count="count")
            .reset_index()
            .sort_values(by=x_col)
        )

        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        label = f"{dist:.1f} km" if dist != int(dist) else f"{int(dist)} km"

        ax.plot(
            stats[x_col],
            stats["mean_val"],
            label=label,
            color=color,
            marker=marker,
            linewidth=2,
            markersize=6,
        )

        if shaded:
            ax.fill_between(
                stats[x_col],
                stats["min_val"],
                stats["max_val"],
                color=color,
                alpha=0.25,
            )

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


def generate_evm_plots(
    csv_path: Union[Path, str], output_dir: Optional[Union[Path, str]] = None
) -> List[Path]:
    """Generates EVM plot variations (dB and RMS %) from a results CSV.

    Args:
        csv_path: Path to the input CSV file.
        output_dir: Path to directory where plots will be saved. Defaults to <csv_dir>/plots/.

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
    csv_path: Union[Path, str], output_dir: Optional[Union[Path, str]] = None
) -> List[Path]:
    """Generates both BER and EVM plot figures.

    Args:
        csv_path: Path to the input CSV file.
        output_dir: Path to directory where plots will be saved.

    Returns:
        List of Paths to generated PNG files.
    """
    all_plots = []
    ber_plots = generate_ber_plots(csv_path=csv_path, output_dir=output_dir)
    all_plots.extend(ber_plots)

    evm_plots = generate_evm_plots(csv_path=csv_path, output_dir=output_dir)
    all_plots.extend(evm_plots)

    return all_plots

