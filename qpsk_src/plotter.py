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
    required_cols = {"distance_km", "ber"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV missing required columns: {required_cols - set(df.columns)}")

    df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce")
    df["ber"] = pd.to_numeric(df["ber"], errors="coerce")

    # Round SNR and OSNR for clean grouping if present
    if "snr_db" in df.columns:
        df["snr_db"] = pd.to_numeric(df["snr_db"], errors="coerce").round(2)
    if "osnr_db" in df.columns:
        df["osnr_db"] = pd.to_numeric(df["osnr_db"], errors="coerce").round(2)

    return df.dropna(subset=["ber", "distance_km"])


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

    ax.set_yscale("log")
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
