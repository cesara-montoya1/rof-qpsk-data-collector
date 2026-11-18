#!/usr/bin/env python3
"""
Script to generate comparative plots between:
- rof_uspl_NLNA_snr_0p5_370mA_650_Hz_ch_24
- rof_uspl_NLNA_snr_0p5_370mA_650_Hz_ch_25

Plots are saved to:
- results/hanna_maestria/plots_uspl_snr_0p5/
- results/hanna_maestria/plots_uspl/
"""

import shutil
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
CH24_CSV = RESULTS_DIR / "rof_uspl_NLNA_snr_0p5_370mA_650_Hz_ch_24" / "results_rof_uspl_NLNA_snr_0p5_370mA_650_Hz_ch_24.csv"
CH25_CSV = RESULTS_DIR / "rof_uspl_NLNA_snr_0p5_370mA_650_Hz_ch_25" / "results_rof_uspl_NLNA_snr_0p5_370mA_650_Hz_ch_25.csv"
OUTPUT_DIR = RESULTS_DIR / "plots_uspl_snr_0p5"
PLOTS_USPL_DIR = RESULTS_DIR / "plots_uspl"


def load_datasets():
    if not CH24_CSV.exists():
        raise FileNotFoundError(f"Missing CH24 CSV: {CH24_CSV}")
    if not CH25_CSV.exists():
        raise FileNotFoundError(f"Missing CH25 CSV: {CH25_CSV}")

    df24 = pd.read_csv(CH24_CSV)
    df25 = pd.read_csv(CH25_CSV)

    df24["channel"] = "CH24"
    df25["channel"] = "CH25"

    for df in [df24, df25]:
        df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce")
        df["osnr_db"] = pd.to_numeric(df["osnr_db"], errors="coerce").round(2)
        df["snr_db"] = pd.to_numeric(df["snr_db"], errors="coerce").round(2)
        df["ber"] = pd.to_numeric(df["ber"], errors="coerce")
        df["evm_db"] = pd.to_numeric(df["evm_db"], errors="coerce")
        df["evm_rms_pct"] = pd.to_numeric(df["evm_rms_pct"], errors="coerce")
        df["evm_peak_pct"] = pd.to_numeric(df["evm_peak_pct"], errors="coerce")

    return df24, df25


def get_series_list(df24, df25):
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

    return series


def aggregate_by_optic_seed(sub_df, y_col):
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


def aggregate_by_snr(sub_df, y_col):
    stats = sub_df.groupby("osnr_db").agg(
        snr_mean=("snr_db", "mean"),
        y_mean=(y_col, "mean"),
        y_min=(y_col, "min"),
        y_max=(y_col, "max"),
        y_std=(y_col, "std"),
        count=(y_col, "count"),
    ).reset_index(drop=True).sort_values(by="snr_mean")
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
        if x_metric == "osnr_db":
            stats = aggregate_by_optic_seed(sub_df, y_metric)
            x_vals = stats["optic_seed_mean"]
        else:
            stats = aggregate_by_snr(sub_df, y_metric)
            x_vals = stats["snr_mean"]

        y_vals = stats["y_mean"]
        y_min = stats["y_min"]
        y_max = stats["y_max"]

        if log_y:
            y_vals = np.maximum(y_vals, 1e-6)
            y_min = np.maximum(y_min, 1e-6)
            y_max = np.maximum(y_max, 1e-6)

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
            ax.set_ylim(bottom=7e-7, top=1.0)
        ax.axhline(1e-6, color="gray", linestyle=":", alpha=0.6, label="Floor (BER = 0)", zorder=1)

    ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.legend(title="Canal / Distancia", fontsize=9.5, title_fontsize=10.5, loc="best", frameon=True, framealpha=0.9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"✓ Guardado: {output_path.name}")


def plot_dashboard_1x2(series_list, output_path: Path):
    """
    Dashboard 1x2 de publicación:
    (a) EVM (dB) vs Optic Seed (dB)
    (b) BER vs Optic Seed (dB) [Escala Logarítmica]
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
            zorder=3,
        )
        ax_evm.fill_between(
            stats_evm_db["optic_seed_mean"],
            stats_evm_db["y_min"],
            stats_evm_db["y_max"],
            color=item["color"],
            alpha=0.15,
            zorder=2,
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
    ax_evm.set_xlabel("Optic Seed / OSNR (dB)", fontsize=12, fontweight="bold")
    ax_evm.set_ylabel("EVM (dB)", fontsize=12, fontweight="bold")
    ax_evm.set_title("(a) EVM (dB) vs Optic Seed (dB)", fontsize=13, fontweight="bold")
    ax_evm.grid(True, linestyle="--", alpha=0.7)
    ax_evm.legend(title="Canal / Distancia", fontsize=9.5, title_fontsize=10.5, loc="best", frameon=True, framealpha=0.9)

    # Panel 2: BER vs Optic Seed
    ax_ber.set_yscale("log")
    ax_ber.set_ylim(bottom=7e-7, top=1.0)
    ax_ber.set_xlabel("Optic Seed / OSNR (dB)", fontsize=12, fontweight="bold")
    ax_ber.set_ylabel("Bit Error Rate (BER)", fontsize=12, fontweight="bold")
    ax_ber.set_title("(b) BER vs Optic Seed (dB) [Log Scale]", fontsize=13, fontweight="bold")
    ax_ber.axhline(1e-6, color="gray", linestyle=":", alpha=0.6, label="Floor (BER = 0)", zorder=1)
    ax_ber.grid(True, which="both", linestyle="--", alpha=0.7)
    ax_ber.legend(title="Canal / Distancia", fontsize=9.5, title_fontsize=10.5, loc="best", frameon=True, framealpha=0.9)

    fig.suptitle("Comparativa RoF USPL: CH24 vs CH25 (NLNA, SNR 0.5 dB, 370 mA, 650 MHz)", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"✓ Guardado Dashboard 1x2: {output_path.name}")


def plot_dashboard_2x2(series_list, output_path: Path):
    """
    Dashboard 2x2 completo:
    (a) EVM (dB) vs Optic Seed
    (b) BER vs Optic Seed
    (c) EVM (dB) vs SNR
    (d) BER vs SNR
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    (ax_evm_osnr, ax_ber_osnr), (ax_evm_snr, ax_ber_snr) = axes

    for item in series_list:
        sub_df = item["df"]

        # --- Fila 1: vs Optic Seed ---
        stats_evm_osnr = aggregate_by_optic_seed(sub_df, "evm_db")
        ax_evm_osnr.plot(
            stats_evm_osnr["optic_seed_mean"], stats_evm_osnr["y_mean"],
            label=item["label"], color=item["color"], marker=item["marker"],
            linestyle=item["linestyle"], linewidth=2.0, markersize=5.5, zorder=3
        )
        ax_evm_osnr.fill_between(
            stats_evm_osnr["optic_seed_mean"], stats_evm_osnr["y_min"], stats_evm_osnr["y_max"],
            color=item["color"], alpha=0.15, zorder=2
        )

        stats_ber_osnr = aggregate_by_optic_seed(sub_df, "ber")
        ber_y = np.maximum(stats_ber_osnr["y_mean"], 1e-6)
        ber_min = np.maximum(stats_ber_osnr["y_min"], 1e-6)
        ber_max = np.maximum(stats_ber_osnr["y_max"], 1e-6)
        ax_ber_osnr.plot(
            stats_ber_osnr["optic_seed_mean"], ber_y,
            label=item["label"], color=item["color"], marker=item["marker"],
            linestyle=item["linestyle"], linewidth=2.0, markersize=5.5, zorder=3
        )
        ax_ber_osnr.fill_between(
            stats_ber_osnr["optic_seed_mean"], ber_min, ber_max,
            color=item["color"], alpha=0.15, zorder=2
        )

        # --- Fila 2: vs SNR ---
        stats_evm_snr = aggregate_by_snr(sub_df, "evm_db")
        ax_evm_snr.plot(
            stats_evm_snr["snr_mean"], stats_evm_snr["y_mean"],
            label=item["label"], color=item["color"], marker=item["marker"],
            linestyle=item["linestyle"], linewidth=2.0, markersize=5.5, zorder=3
        )
        ax_evm_snr.fill_between(
            stats_evm_snr["snr_mean"], stats_evm_snr["y_min"], stats_evm_snr["y_max"],
            color=item["color"], alpha=0.15, zorder=2
        )

        stats_ber_snr = aggregate_by_snr(sub_df, "ber")
        ber_y_snr = np.maximum(stats_ber_snr["y_mean"], 1e-6)
        ber_min_snr = np.maximum(stats_ber_snr["y_min"], 1e-6)
        ber_max_snr = np.maximum(stats_ber_snr["y_max"], 1e-6)
        ax_ber_snr.plot(
            stats_ber_snr["snr_mean"], ber_y_snr,
            label=item["label"], color=item["color"], marker=item["marker"],
            linestyle=item["linestyle"], linewidth=2.0, markersize=5.5, zorder=3
        )
        ax_ber_snr.fill_between(
            stats_ber_snr["snr_mean"], ber_min_snr, ber_max_snr,
            color=item["color"], alpha=0.15, zorder=2
        )

    # Configuración de ejes
    # (a)
    ax_evm_osnr.set_xlabel("Optic Seed / OSNR (dB)", fontsize=11, fontweight="bold")
    ax_evm_osnr.set_ylabel("EVM (dB)", fontsize=11, fontweight="bold")
    ax_evm_osnr.set_title("(a) EVM (dB) vs Optic Seed (dB)", fontsize=12, fontweight="bold")
    ax_evm_osnr.grid(True, linestyle="--", alpha=0.7)
    ax_evm_osnr.legend(fontsize=9, loc="best", framealpha=0.9)

    # (b)
    ax_ber_osnr.set_yscale("log")
    ax_ber_osnr.set_ylim(bottom=7e-7, top=1.0)
    ax_ber_osnr.set_xlabel("Optic Seed / OSNR (dB)", fontsize=11, fontweight="bold")
    ax_ber_osnr.set_ylabel("Bit Error Rate (BER)", fontsize=11, fontweight="bold")
    ax_ber_osnr.set_title("(b) BER vs Optic Seed (dB) [Log Scale]", fontsize=12, fontweight="bold")
    ax_ber_osnr.axhline(1e-6, color="gray", linestyle=":", alpha=0.6, label="Floor (BER = 0)", zorder=1)
    ax_ber_osnr.grid(True, which="both", linestyle="--", alpha=0.7)
    ax_ber_osnr.legend(fontsize=9, loc="best", framealpha=0.9)

    # (c)
    ax_evm_snr.set_xlabel("SNR Eléctrico (dB)", fontsize=11, fontweight="bold")
    ax_evm_snr.set_ylabel("EVM (dB)", fontsize=11, fontweight="bold")
    ax_evm_snr.set_title("(c) EVM (dB) vs SNR Eléctrico (dB)", fontsize=12, fontweight="bold")
    ax_evm_snr.grid(True, linestyle="--", alpha=0.7)
    ax_evm_snr.legend(fontsize=9, loc="best", framealpha=0.9)

    # (d)
    ax_ber_snr.set_yscale("log")
    ax_ber_snr.set_ylim(bottom=7e-7, top=1.0)
    ax_ber_snr.set_xlabel("SNR Eléctrico (dB)", fontsize=11, fontweight="bold")
    ax_ber_snr.set_ylabel("Bit Error Rate (BER)", fontsize=11, fontweight="bold")
    ax_ber_snr.set_title("(d) BER vs SNR Eléctrico (dB) [Log Scale]", fontsize=12, fontweight="bold")
    ax_ber_snr.axhline(1e-6, color="gray", linestyle=":", alpha=0.6, label="Floor (BER = 0)", zorder=1)
    ax_ber_snr.grid(True, which="both", linestyle="--", alpha=0.7)
    ax_ber_snr.legend(fontsize=9, loc="best", framealpha=0.9)

    fig.suptitle("Dashboard Comparativo Completo: CH24 vs CH25 (USPL, SNR 0.5 dB, 370 mA, 650 MHz)", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"✓ Guardado Dashboard 2x2: {output_path.name}")


def main():
    print("Cargando datasets de CH24 y CH25...")
    df24, df25 = load_datasets()
    series_list = get_series_list(df24, df25)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_USPL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generando figuras comparativas en: {OUTPUT_DIR}")

    # 1. EVM (dB) vs Optic Seed
    plot_single_comparison(
        series_list,
        x_metric="osnr_db",
        y_metric="evm_db",
        x_label="Optic Seed / OSNR (dB)",
        y_label="EVM (dB)",
        title="EVM (dB) vs Optic Seed (dB) — CH24 vs CH25 (USPL)",
        output_path=OUTPUT_DIR / "comparativa_evm_db_vs_optic_seed.png",
        shaded=True,
    )

    # 2. EVM (%) vs Optic Seed
    plot_single_comparison(
        series_list,
        x_metric="osnr_db",
        y_metric="evm_rms_pct",
        x_label="Optic Seed / OSNR (dB)",
        y_label="EVM RMS (%)",
        title="EVM RMS (%) vs Optic Seed (dB) — CH24 vs CH25 (USPL)",
        output_path=OUTPUT_DIR / "comparativa_evm_pct_vs_optic_seed.png",
        shaded=True,
    )

    # 3. BER vs Optic Seed
    plot_single_comparison(
        series_list,
        x_metric="osnr_db",
        y_metric="ber",
        x_label="Optic Seed / OSNR (dB)",
        y_label="Bit Error Rate (BER)",
        title="BER vs Optic Seed (dB) — CH24 vs CH25 (USPL)",
        output_path=OUTPUT_DIR / "comparativa_ber_vs_optic_seed.png",
        log_y=True,
        shaded=True,
        y_lim=(7e-7, 1.0),
    )

    # 4. EVM (dB) vs SNR
    plot_single_comparison(
        series_list,
        x_metric="snr_db",
        y_metric="evm_db",
        x_label="SNR Eléctrico (dB)",
        y_label="EVM (dB)",
        title="EVM (dB) vs SNR Eléctrico (dB) — CH24 vs CH25 (USPL)",
        output_path=OUTPUT_DIR / "comparativa_evm_db_vs_snr.png",
        shaded=True,
    )

    # 5. BER vs SNR
    plot_single_comparison(
        series_list,
        x_metric="snr_db",
        y_metric="ber",
        x_label="SNR Eléctrico (dB)",
        y_label="Bit Error Rate (BER)",
        title="BER vs SNR Eléctrico (dB) — CH24 vs CH25 (USPL)",
        output_path=OUTPUT_DIR / "comparativa_ber_vs_snr.png",
        log_y=True,
        shaded=True,
        y_lim=(7e-7, 1.0),
    )

    # 6. Dashboards
    dash_1x2_path = OUTPUT_DIR / "uspl_0p5_ch24_vs_ch25_dashboard.png"
    plot_dashboard_1x2(series_list, dash_1x2_path)
    
    # Copiar también en plots_uspl/ para compatibilidad
    shutil.copy(dash_1x2_path, PLOTS_USPL_DIR / "uspl_0p5_ch24_vs_ch25_dashboard.png")
    print(f"✓ Copiado a: {PLOTS_USPL_DIR / 'uspl_0p5_ch24_vs_ch25_dashboard.png'}")

    dash_2x2_path = OUTPUT_DIR / "uspl_0p5_ch24_vs_ch25_dashboard_2x2.png"
    plot_dashboard_2x2(series_list, dash_2x2_path)
    shutil.copy(dash_2x2_path, PLOTS_USPL_DIR / "uspl_0p5_ch24_vs_ch25_dashboard_2x2.png")

    print("\n¡Todas las gráficas comparativas generadas con éxito!")


if __name__ == "__main__":
    main()
