#!/usr/bin/env python3
"""
Script to generate comparative plots between:
- rof_uspl_NLNA_snr_1p0_650MHz_ch24
- rof_uspl_NLNA_snr_1p0_650MHz_ch25

Plots are saved to: results/hanna_maestria/plots_uspl/
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "hanna_maestria"
CH24_CSV = RESULTS_DIR / "rof_uspl_NLNA_snr_1p0_650MHz_ch24" / "results_rof_uspl_NLNA_snr_1p0_650MHz_ch24.csv"
CH25_CSV = RESULTS_DIR / "rof_uspl_NLNA_snr_1p0_650MHz_ch25" / "results_rof_uspl_NLNA_snr_1p0_650MHz_ch25.csv"
CH26_CSV = RESULTS_DIR / "rof_uspl_NLNA_snr_7p3_650MHz_ch26" / "results_rof_uspl_NLNA_snr_7p3_650MHz_ch26.csv"
OUTPUT_DIR = RESULTS_DIR / "plots_uspl"


def load_datasets():
    if not CH24_CSV.exists():
        raise FileNotFoundError(f"Missing CH24 CSV: {CH24_CSV}")
    if not CH25_CSV.exists():
        raise FileNotFoundError(f"Missing CH25 CSV: {CH25_CSV}")
    if not CH26_CSV.exists():
        raise FileNotFoundError(f"Missing CH26 CSV: {CH26_CSV}")

    df24 = pd.read_csv(CH24_CSV)
    df25 = pd.read_csv(CH25_CSV)
    df26 = pd.read_csv(CH26_CSV)

    df24["channel"] = "CH24"
    df25["channel"] = "CH25"
    df26["channel"] = "CH26"

    for df in [df24, df25, df26]:
        df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce")
        df["osnr_db"] = pd.to_numeric(df["osnr_db"], errors="coerce").round(2)
        df["snr_db"] = pd.to_numeric(df["snr_db"], errors="coerce").round(2)
        df["ber"] = pd.to_numeric(df["ber"], errors="coerce")
        df["evm_db"] = pd.to_numeric(df["evm_db"], errors="coerce")
        df["evm_rms_pct"] = pd.to_numeric(df["evm_rms_pct"], errors="coerce")
        df["evm_peak_pct"] = pd.to_numeric(df["evm_peak_pct"], errors="coerce")

    # Combine
    combined_df = pd.concat([df24, df25, df26], ignore_index=True)
    return df24, df25, df26, combined_df


def get_series_list(df24, df25, df26):
    """
    Returns a list of series configurations:
    (name, df, color, marker, linestyle)
    """
    series = []
    # CH24 0km (B2B)
    s1 = df24[df24["distance_km"] == 0.0]
    if not s1.empty:
        series.append({
            "label": "CH24 - 0 km (B2B)",
            "df": s1,
            "color": "#1f77b4",  # Blue
            "marker": "o",
            "linestyle": "-",
        })

    # CH24 20km
    s2 = df24[df24["distance_km"] == 20.0]
    if not s2.empty:
        series.append({
            "label": "CH24 - 20 km",
            "df": s2,
            "color": "#ff7f0e",  # Orange
            "marker": "s",
            "linestyle": "--",
        })

    # CH25 0km (B2B)
    s3 = df25[df25["distance_km"] == 0.0]
    if not s3.empty:
        series.append({
            "label": "CH25 - 0 km (B2B)",
            "df": s3,
            "color": "#2ca02c",  # Green
            "marker": "^",
            "linestyle": "-",
        })

    # CH25 20km
    s4 = df25[df25["distance_km"] == 20.0]
    if not s4.empty:
        series.append({
            "label": "CH25 - 20 km",
            "df": s4,
            "color": "#d62728",  # Red
            "marker": "v",
            "linestyle": "--",
        })

    # CH26 0km (B2B)
    s5 = df26[df26["distance_km"] == 0.0]
    if not s5.empty:
        series.append({
            "label": "CH26 - 0 km (B2B)",
            "df": s5,
            "color": "#9467bd",  # Purple
            "marker": "D",
            "linestyle": "-",
        })

    # CH26 20km
    s6 = df26[df26["distance_km"] == 20.0]
    if not s6.empty:
        series.append({
            "label": "CH26 - 20 km",
            "df": s6,
            "color": "#8c564b",  # Brown
            "marker": "X",
            "linestyle": "--",
        })

    return series


def aggregate_by_optic_seed(sub_df, y_col):
    """
    Groups by osnr_db (representing Optic Seed in dB) and computes stats for y_col.
    """
    stats = sub_df.groupby("osnr_db").agg(
        optic_seed_mean=("osnr_db", "mean"),
        snr_mean=("snr_db", "mean"),
        y_mean=(y_col, "mean"),
        y_min=(y_col, "min"),
        y_max=(y_col, "max"),
        y_std=(y_col, "std"),
        count=(y_col, "count"),
    ).reset_index(drop=True).sort_values(by="optic_seed_mean")
    return stats


def plot_single_comparison(
    series_list,
    x_metric: str,
    y_metric: str,
    x_label: str,
    y_label: str,
    title: str,
    output_path: Path,
    log_y: bool = False,
    shaded: bool = True,
    y_lim: tuple = None,
):
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    for item in series_list:
        sub_df = item["df"]
        stats = aggregate_by_optic_seed(sub_df, y_metric)

        x_vals = stats["optic_seed_mean"] if x_metric == "osnr_db" else stats["snr_mean"]
        y_vals = stats["y_mean"]
        y_min = stats["y_min"]
        y_max = stats["y_max"]

        if log_y:
            # Handle zeros for log scale
            y_vals = np.maximum(y_vals, 1e-6)
            y_min = np.maximum(y_min, 1e-6)
            y_max = np.maximum(y_max, 1e-6)

        # Plot line & points
        ax.plot(
            x_vals,
            y_vals,
            label=item["label"],
            color=item["color"],
            marker=item["marker"],
            linestyle=item["linestyle"],
            linewidth=2.2,
            markersize=6.5,
            alpha=0.95,
            zorder=3,
        )

        if shaded:
            ax.fill_between(
                x_vals,
                y_min,
                y_max,
                color=item["color"],
                alpha=0.18,
                zorder=2,
            )

    if log_y:
        ax.set_yscale("log")
        if y_lim is not None:
            ax.set_ylim(bottom=y_lim[0], top=y_lim[1])
        else:
            ax.set_ylim(bottom=1e-6, top=1.0)
        ax.axhline(1e-6, color="gray", linestyle=":", alpha=0.6, label="Floor (Zero errors)", zorder=1)

    ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.legend(title="Canal / Distancia", fontsize=9.5, title_fontsize=10.5, loc="best", frameon=True, framealpha=0.9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"✓ Saved: {output_path}")


def plot_dashboard(series_list, output_path: Path):
    """
    Generates a 1x2 comprehensive publication dashboard comparing:
    (a) EVM (dB) vs Optic Seed (dB)
    (b) BER vs Optic Seed (dB) [Log Scale]
    """
    fig, (ax_evm, ax_ber) = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)

    for item in series_list:
        sub_df = item["df"]

        # 1. EVM dB vs Optic Seed
        stats_evm_db = aggregate_by_optic_seed(sub_df, "evm_db")
        ax_evm.plot(
            stats_evm_db["optic_seed_mean"],
            stats_evm_db["y_mean"],
            label=item["label"],
            color=item["color"],
            marker=item["marker"],
            linestyle=item["linestyle"],
            linewidth=2.0,
            markersize=6,
        )
        ax_evm.fill_between(
            stats_evm_db["optic_seed_mean"],
            stats_evm_db["y_min"],
            stats_evm_db["y_max"],
            color=item["color"],
            alpha=0.15,
        )

        # 2. BER vs Optic Seed
        stats_ber = aggregate_by_optic_seed(sub_df, "ber")
        ber_y = np.maximum(stats_ber["y_mean"], 1e-6)
        ber_min = np.maximum(stats_ber["y_min"], 1e-6)
        ber_max = np.maximum(stats_ber["y_max"], 1e-6)
        ax_ber.plot(
            stats_ber["optic_seed_mean"],
            ber_y,
            label=item["label"],
            color=item["color"],
            marker=item["marker"],
            linestyle=item["linestyle"],
            linewidth=2.0,
            markersize=6,
            zorder=3,
        )
        ax_ber.fill_between(
            stats_ber["optic_seed_mean"],
            ber_min,
            ber_max,
            color=item["color"],
            alpha=0.15,
            zorder=2,
        )

    # Panel 1: EVM dB vs Optic Seed
    ax_evm.set_xlabel("Optic Seed (dB)", fontsize=12, fontweight="bold")
    ax_evm.set_ylabel("EVM (dB)", fontsize=12, fontweight="bold")
    ax_evm.set_title("(a) EVM (dB) vs Optic Seed (dB)", fontsize=13, fontweight="bold")
    ax_evm.grid(True, linestyle="--", alpha=0.7)
    ax_evm.legend(fontsize=9, loc="best")

    # Panel 2: BER vs Optic Seed
    ax_ber.set_yscale("log")
    ax_ber.set_ylim(bottom=7e-7, top=1.0)
    ax_ber.set_xlabel("Optic Seed (dB)", fontsize=12, fontweight="bold")
    ax_ber.set_ylabel("Bit Error Rate (BER)", fontsize=12, fontweight="bold")
    ax_ber.set_title("(b) BER vs Optic Seed (dB) [Log Scale]", fontsize=13, fontweight="bold")
    ax_ber.axhline(1e-6, color="gray", linestyle=":", alpha=0.6, label="Floor (BER = 0)", zorder=1)
    ax_ber.grid(True, which="both", linestyle="--", alpha=0.7)
    ax_ber.legend(fontsize=9, loc="best")

    fig.suptitle("Comparativa RoF USPL: CH24 vs CH25 vs CH26 (NLNA, 650 MHz)", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    alt_dashboard = output_path.parent / "uspl_ch24_vs_ch25_vs_ch26_dashboard.png"
    plt.savefig(alt_dashboard, dpi=300)
    plt.close(fig)
    print(f"✓ Saved Dashboard: {output_path}")
    print(f"✓ Saved Dashboard (alias): {alt_dashboard}")


def main():
    print("Loading datasets...")
    df24, df25, df26, _ = load_datasets()
    series_list = get_series_list(df24, df25, df26)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating comparative plots in: {OUTPUT_DIR}")

    # Remove obsolete plots no longer requested
    obsolete_files = [
        OUTPUT_DIR / "evm_db_vs_snr.png",
        OUTPUT_DIR / "evm_pct_vs_snr.png",
        OUTPUT_DIR / "evm_pct_vs_osnr.png",
        OUTPUT_DIR / "ber_vs_snr.png",
    ]
    for obs in obsolete_files:
        if obs.exists():
            obs.unlink()
            print(f"Removed obsolete plot: {obs.name}")

    # 1. EVM (dB) vs Optic Seed (dB)
    plot_single_comparison(
        series_list,
        x_metric="osnr_db",
        y_metric="evm_db",
        x_label="Optic Seed (dB)",
        y_label="EVM (dB)",
        title="EVM (dB) vs Optic Seed (dB) — Comparativa CH24 vs CH25 vs CH26 (USPL)",
        output_path=OUTPUT_DIR / "evm_db_vs_optic_seed.png",
        shaded=True,
    )
    # Also save as evm_db_vs_osnr.png for backward compatibility
    plot_single_comparison(
        series_list,
        x_metric="osnr_db",
        y_metric="evm_db",
        x_label="Optic Seed (dB)",
        y_label="EVM (dB)",
        title="EVM (dB) vs Optic Seed (dB) — Comparativa CH24 vs CH25 vs CH26 (USPL)",
        output_path=OUTPUT_DIR / "evm_db_vs_osnr.png",
        shaded=True,
    )

    # 2. BER vs Optic Seed (dB)
    plot_single_comparison(
        series_list,
        x_metric="osnr_db",
        y_metric="ber",
        x_label="Optic Seed (dB)",
        y_label="Bit Error Rate (BER)",
        title="BER vs Optic Seed (dB) — Comparativa CH24 vs CH25 vs CH26 (USPL)",
        output_path=OUTPUT_DIR / "ber_vs_optic_seed.png",
        log_y=True,
        shaded=True,
        y_lim=(7e-7, 1.0),
    )
    # Also save as ber_vs_osnr.png for backward compatibility
    plot_single_comparison(
        series_list,
        x_metric="osnr_db",
        y_metric="ber",
        x_label="Optic Seed (dB)",
        y_label="Bit Error Rate (BER)",
        title="BER vs Optic Seed (dB) — Comparativa CH24 vs CH25 vs CH26 (USPL)",
        output_path=OUTPUT_DIR / "ber_vs_osnr.png",
        log_y=True,
        shaded=True,
        y_lim=(7e-7, 1.0),
    )

    # 3. Dashboard 1x2: EVM (dB) vs Optic Seed & BER vs Optic Seed
    plot_dashboard(series_list, OUTPUT_DIR / "uspl_ch24_vs_ch25_dashboard.png")

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()

