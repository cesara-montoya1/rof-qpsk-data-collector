#!/usr/bin/env python3
"""
Script to generate comparative plots between all 4 channels:
- rof_uspl_NLNA_snr_0p5_370mA_650_Hz_ch_24
- rof_uspl_NLNA_snr_0p5_370mA_650_Hz_ch_25
- rof_uspl_NLNA_snr_0p5_370mA_650_Hz_ch_26
- rof_uspl_NLNA_snr_0p5_370mA_650_Hz_ch_28

Metrics:
1. EVM RMS (%) vs Optic Seed (dB)
2. EVM (dB) vs Optic Seed (dB)
3. Measured BER vs Optic Seed (dB)
4. Inferred BER (from EVM RMS) vs Optic Seed (dB)

Outputs saved to:
- results/hanna_maestria/plots_uspl_snr_0p5/
- results/hanna_maestria/plots_uspl/
"""

import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import erfc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "hanna_maestria"

CH24_CSV = RESULTS_DIR / "rof_uspl_NLNA_snr_0p5_370mA_650_Hz_ch_24" / "results_rof_uspl_NLNA_snr_0p5_370mA_650_Hz_ch_24.csv"
CH25_CSV = RESULTS_DIR / "rof_uspl_NLNA_snr_0p5_370mA_650_Hz_ch_25" / "results_rof_uspl_NLNA_snr_0p5_370mA_650_Hz_ch_25.csv"
CH26_CSV = RESULTS_DIR / "rof_uspl_NLNA_snr_0p5_370mA_650_Hz_ch_26" / "results_rof_uspl_NLNA_snr_0p5_370mA_650_Hz_ch_26.csv"
CH28_CSV = RESULTS_DIR / "rof_uspl_NLNA_snr_0p5_370mA_650_Hz_ch_28" / "results_rof_uspl_NLNA_snr_0p5_370mA_650_Hz_ch_28.csv"

OUTPUT_DIR = RESULTS_DIR / "plots_uspl_snr_0p5"
PLOTS_USPL_DIR = RESULTS_DIR / "plots_uspl"


def compute_inferred_ber(evm_pct: pd.Series | np.ndarray) -> np.ndarray:
    """
    Computes analytical BER for QPSK from EVM RMS (%):
    EVM_ratio = EVM_pct / 100.0
    BER = 0.5 * erfc(1.0 / (sqrt(2.0) * EVM_ratio))
    """
    evm_ratio = np.asarray(evm_pct, dtype=float) / 100.0
    # Guard against zero or negative values
    evm_ratio = np.maximum(evm_ratio, 1e-4)
    return 0.5 * erfc(1.0 / (np.sqrt(2.0) * evm_ratio))


def load_datasets():
    datasets = [
        ("CH24", CH24_CSV),
        ("CH25", CH25_CSV),
        ("CH26", CH26_CSV),
        ("CH28", CH28_CSV),
    ]
    dfs = {}
    for name, p in datasets:
        if not p.exists():
            raise FileNotFoundError(f"Missing {name} CSV: {p}")
        df = pd.read_csv(p)
        df["channel"] = name
        df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce")
        df["osnr_db"] = pd.to_numeric(df["osnr_db"], errors="coerce").round(2)
        df["snr_db"] = pd.to_numeric(df["snr_db"], errors="coerce").round(2)
        df["ber"] = pd.to_numeric(df["ber"], errors="coerce")
        df["evm_db"] = pd.to_numeric(df["evm_db"], errors="coerce")
        df["evm_rms_pct"] = pd.to_numeric(df["evm_rms_pct"], errors="coerce")
        df["evm_peak_pct"] = pd.to_numeric(df["evm_peak_pct"], errors="coerce")
        df["ber_inferred"] = compute_inferred_ber(df["evm_rms_pct"])
        dfs[name] = df

    return dfs["CH24"], dfs["CH25"], dfs["CH26"], dfs["CH28"]


def get_series_list(df24, df25, df26, df28):
    """
    Returns configured series for high-contrast multi-channel comparison.
    """
    series = []

    # CH24 0km (B2B) & 20km
    s1 = df24[df24["distance_km"] == 0.0]
    if not s1.empty:
        series.append({
            "label": "CH24 - 0 km (B2B)",
            "df": s1,
            "color": "#1f77b4",  # Blue
            "marker": "o",
            "linestyle": "-",
        })
    s2 = df24[df24["distance_km"] == 20.0]
    if not s2.empty:
        series.append({
            "label": "CH24 - 20 km",
            "df": s2,
            "color": "#ff7f0e",  # Orange
            "marker": "s",
            "linestyle": "--",
        })

    # CH25 0km (B2B) & 20km
    s3 = df25[df25["distance_km"] == 0.0]
    if not s3.empty:
        series.append({
            "label": "CH25 - 0 km (B2B)",
            "df": s3,
            "color": "#2ca02c",  # Green
            "marker": "^",
            "linestyle": "-",
        })
    s4 = df25[df25["distance_km"] == 20.0]
    if not s4.empty:
        series.append({
            "label": "CH25 - 20 km",
            "df": s4,
            "color": "#d62728",  # Red
            "marker": "v",
            "linestyle": "--",
        })

    # CH26 0km (B2B) & 20km
    s5 = df26[df26["distance_km"] == 0.0]
    if not s5.empty:
        series.append({
            "label": "CH26 - 0 km (B2B)",
            "df": s5,
            "color": "#9467bd",  # Purple
            "marker": "D",
            "linestyle": "-",
        })
    s6 = df26[df26["distance_km"] == 20.0]
    if not s6.empty:
        series.append({
            "label": "CH26 - 20 km",
            "df": s6,
            "color": "#8c564b",  # Brown
            "marker": "X",
            "linestyle": "--",
        })

    # CH28 20km
    s7 = df28[df28["distance_km"] == 20.0]
    if not s7.empty:
        series.append({
            "label": "CH28 - 20 km",
            "df": s7,
            "color": "#17becf",  # Cyan / Teal
            "marker": "P",
            "linestyle": "--",
        })
    # Also include 0km for CH28 if present
    s8 = df28[df28["distance_km"] == 0.0]
    if not s8.empty:
        series.append({
            "label": "CH28 - 0 km (B2B)",
            "df": s8,
            "color": "#e377c2",  # Pink
            "marker": "p",
            "linestyle": "-",
        })

    return series


def aggregate_by_optic_seed(sub_df, y_col):
    stats = sub_df.groupby("osnr_db").agg(
        optic_seed_mean=("osnr_db", "mean"),
        y_mean=(y_col, "mean"),
        y_min=(y_col, "min"),
        y_max=(y_col, "max"),
        y_std=(y_col, "std"),
        count=(y_col, "count"),
    ).reset_index(drop=True).sort_values(by="optic_seed_mean")
    return stats


def plot_single_comparison(
    series_list,
    y_metric: str,
    y_label: str,
    title: str,
    output_path: Path,
    log_y: bool = False,
    shaded: bool = True,
    y_lim: tuple = None,
):
    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=300)

    for item in series_list:
        sub_df = item["df"]
        stats = aggregate_by_optic_seed(sub_df, y_metric)
        x_vals = stats["optic_seed_mean"]
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
                alpha=0.16,
                zorder=2,
            )

    if log_y:
        ax.set_yscale("log")
        if y_lim is not None:
            ax.set_ylim(bottom=y_lim[0], top=y_lim[1])
        else:
            ax.set_ylim(bottom=7e-7, top=1.0)
        ax.axhline(1e-6, color="gray", linestyle=":", alpha=0.65, label="Piso (BER = 0)", zorder=1)

    ax.set_xlabel("Optic Seed (dB)", fontsize=12, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.legend(title="Canal / Distancia", fontsize=9.5, title_fontsize=10.5, loc="best", frameon=True, framealpha=0.92)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"✓ Guardado: {output_path.name}")


def plot_dashboard_1x3(series_list, output_path: Path):
    """
    Dashboard 1x3 para publicación:
    (a) EVM RMS (%) vs Optic Seed (dB)
    (b) BER Inferida (desde EVM) vs Optic Seed (dB) [Log Scale]
    (c) BER Medida vs Optic Seed (dB) [Log Scale]
    """
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.2), dpi=300)
    ax_evm, ax_ber_inf, ax_ber_meas = axes

    for item in series_list:
        sub_df = item["df"]

        # 1. EVM RMS (%)
        stats_evm = aggregate_by_optic_seed(sub_df, "evm_rms_pct")
        ax_evm.plot(
            stats_evm["optic_seed_mean"], stats_evm["y_mean"],
            label=item["label"], color=item["color"], marker=item["marker"],
            linestyle=item["linestyle"], linewidth=2.0, markersize=5.5, zorder=3
        )
        ax_evm.fill_between(
            stats_evm["optic_seed_mean"], stats_evm["y_min"], stats_evm["y_max"],
            color=item["color"], alpha=0.15, zorder=2
        )

        # 2. BER Inferida
        stats_ber_inf = aggregate_by_optic_seed(sub_df, "ber_inferred")
        ber_inf_y = np.maximum(stats_ber_inf["y_mean"], 1e-6)
        ber_inf_min = np.maximum(stats_ber_inf["y_min"], 1e-6)
        ber_inf_max = np.maximum(stats_ber_inf["y_max"], 1e-6)
        ax_ber_inf.plot(
            stats_ber_inf["optic_seed_mean"], ber_inf_y,
            label=item["label"], color=item["color"], marker=item["marker"],
            linestyle=item["linestyle"], linewidth=2.0, markersize=5.5, zorder=3
        )
        ax_ber_inf.fill_between(
            stats_ber_inf["optic_seed_mean"], ber_inf_min, ber_inf_max,
            color=item["color"], alpha=0.15, zorder=2
        )

        # 3. BER Medida
        stats_ber_meas = aggregate_by_optic_seed(sub_df, "ber")
        ber_m_y = np.maximum(stats_ber_meas["y_mean"], 1e-6)
        ber_m_min = np.maximum(stats_ber_meas["y_min"], 1e-6)
        ber_m_max = np.maximum(stats_ber_meas["y_max"], 1e-6)
        ax_ber_meas.plot(
            stats_ber_meas["optic_seed_mean"], ber_m_y,
            label=item["label"], color=item["color"], marker=item["marker"],
            linestyle=item["linestyle"], linewidth=2.0, markersize=5.5, zorder=3
        )
        ax_ber_meas.fill_between(
            stats_ber_meas["optic_seed_mean"], ber_m_min, ber_m_max,
            color=item["color"], alpha=0.15, zorder=2
        )

    # Panel 1: EVM (%)
    ax_evm.set_xlabel("Optic Seed (dB)", fontsize=11, fontweight="bold")
    ax_evm.set_ylabel("EVM RMS (%)", fontsize=11, fontweight="bold")
    ax_evm.set_title("(a) EVM RMS (%) vs Optic Seed", fontsize=12, fontweight="bold")
    ax_evm.grid(True, linestyle="--", alpha=0.7)
    ax_evm.legend(fontsize=8.5, loc="best", framealpha=0.9)

    # Panel 2: BER Inferida
    ax_ber_inf.set_yscale("log")
    ax_ber_inf.set_ylim(bottom=7e-7, top=1.0)
    ax_ber_inf.set_xlabel("Optic Seed (dB)", fontsize=11, fontweight="bold")
    ax_ber_inf.set_ylabel("BER Inferida (desde EVM)", fontsize=11, fontweight="bold")
    ax_ber_inf.set_title(r"(b) BER Inferida: $\frac{1}{2}\mathrm{erfc}\left(\frac{1}{\sqrt{2}\mathrm{EVM}}\right)$", fontsize=12, fontweight="bold")
    ax_ber_inf.axhline(1e-6, color="gray", linestyle=":", alpha=0.6, label="Piso (BER = 0)", zorder=1)
    ax_ber_inf.grid(True, which="both", linestyle="--", alpha=0.7)
    ax_ber_inf.legend(fontsize=8.5, loc="best", framealpha=0.9)

    # Panel 3: BER Medida
    ax_ber_meas.set_yscale("log")
    ax_ber_meas.set_ylim(bottom=7e-7, top=1.0)
    ax_ber_meas.set_xlabel("Optic Seed (dB)", fontsize=11, fontweight="bold")
    ax_ber_meas.set_ylabel("BER Medida (Conteo de Errores)", fontsize=11, fontweight="bold")
    ax_ber_meas.set_title("(c) BER Medida vs Optic Seed", fontsize=12, fontweight="bold")
    ax_ber_meas.axhline(1e-6, color="gray", linestyle=":", alpha=0.6, label="Piso (BER = 0)", zorder=1)
    ax_ber_meas.grid(True, which="both", linestyle="--", alpha=0.7)
    ax_ber_meas.legend(fontsize=8.5, loc="best", framealpha=0.9)

    fig.suptitle("Comparativa RoF USPL: CH24, CH25, CH26 y CH28 (NLNA, SNR 0.5 dB, 370 mA, 650 MHz)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"✓ Guardado Dashboard 1x3: {output_path.name}")


def plot_dashboard_1x2(series_list, output_path: Path):
    """
    Dashboard 1x2 estándar:
    (a) EVM RMS (%) vs Optic Seed
    (b) BER vs Optic Seed
    """
    fig, (ax_evm, ax_ber) = plt.subplots(1, 2, figsize=(16, 6.2), dpi=300)

    for item in series_list:
        sub_df = item["df"]

        # EVM RMS %
        stats_evm = aggregate_by_optic_seed(sub_df, "evm_rms_pct")
        ax_evm.plot(
            stats_evm["optic_seed_mean"], stats_evm["y_mean"],
            label=item["label"], color=item["color"], marker=item["marker"],
            linestyle=item["linestyle"], linewidth=2.0, markersize=5.5, zorder=3
        )
        ax_evm.fill_between(
            stats_evm["optic_seed_mean"], stats_evm["y_min"], stats_evm["y_max"],
            color=item["color"], alpha=0.15, zorder=2
        )

        # BER Medida
        stats_ber = aggregate_by_optic_seed(sub_df, "ber")
        ber_y = np.maximum(stats_ber["y_mean"], 1e-6)
        ber_min = np.maximum(stats_ber["y_min"], 1e-6)
        ber_max = np.maximum(stats_ber["y_max"], 1e-6)
        ax_ber.plot(
            stats_ber["optic_seed_mean"], ber_y,
            label=item["label"], color=item["color"], marker=item["marker"],
            linestyle=item["linestyle"], linewidth=2.0, markersize=5.5, zorder=3
        )
        ax_ber.fill_between(
            stats_ber["optic_seed_mean"], ber_min, ber_max,
            color=item["color"], alpha=0.15, zorder=2
        )

    ax_evm.set_xlabel("Optic Seed (dB)", fontsize=11, fontweight="bold")
    ax_evm.set_ylabel("EVM RMS (%)", fontsize=11, fontweight="bold")
    ax_evm.set_title("(a) EVM RMS (%) vs Optic Seed", fontsize=12, fontweight="bold")
    ax_evm.grid(True, linestyle="--", alpha=0.7)
    ax_evm.legend(title="Canal / Distancia", fontsize=9, title_fontsize=9.5, loc="best", framealpha=0.9)

    ax_ber.set_yscale("log")
    ax_ber.set_ylim(bottom=7e-7, top=1.0)
    ax_ber.set_xlabel("Optic Seed (dB)", fontsize=11, fontweight="bold")
    ax_ber.set_ylabel("Bit Error Rate (BER)", fontsize=11, fontweight="bold")
    ax_ber.set_title("(b) BER vs Optic Seed (dB) [Log Scale]", fontsize=12, fontweight="bold")
    ax_ber.axhline(1e-6, color="gray", linestyle=":", alpha=0.6, label="Piso (BER = 0)", zorder=1)
    ax_ber.grid(True, which="both", linestyle="--", alpha=0.7)
    ax_ber.legend(title="Canal / Distancia", fontsize=9, title_fontsize=9.5, loc="best", framealpha=0.9)

    fig.suptitle("Comparativa RoF USPL: CH24, CH25, CH26 y CH28 (NLNA, SNR 0.5 dB, 370 mA, 650 MHz)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"✓ Guardado Dashboard 1x2: {output_path.name}")


def main():
    print("Cargando datasets de CH24, CH25, CH26 y CH28 (SNR 0.5 dB, 370 mA)...")
    df24, df25, df26, df28 = load_datasets()
    series_list = get_series_list(df24, df25, df26, df28)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_USPL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generando figuras comparativas en: {OUTPUT_DIR}")

    # 1. EVM RMS (%) vs Optic Seed
    plot_single_comparison(
        series_list,
        y_metric="evm_rms_pct",
        y_label="EVM RMS (%)",
        title="EVM RMS (%) vs Optic Seed (dB) — CH24, CH25, CH26 y CH28 (USPL 0.5 dB)",
        output_path=OUTPUT_DIR / "comparativa_evm_pct_vs_optic_seed.png",
        shaded=True,
    )

    # 2. EVM (dB) vs Optic Seed
    plot_single_comparison(
        series_list,
        y_metric="evm_db",
        y_label="EVM (dB)",
        title="EVM (dB) vs Optic Seed (dB) — CH24, CH25, CH26 y CH28 (USPL 0.5 dB)",
        output_path=OUTPUT_DIR / "comparativa_evm_db_vs_optic_seed.png",
        shaded=True,
    )

    # 3. BER Medida vs Optic Seed
    plot_single_comparison(
        series_list,
        y_metric="ber",
        y_label="Bit Error Rate (BER)",
        title="BER Medida vs Optic Seed (dB) — CH24, CH25, CH26 y CH28 (USPL 0.5 dB)",
        output_path=OUTPUT_DIR / "comparativa_ber_vs_optic_seed.png",
        log_y=True,
        shaded=True,
        y_lim=(7e-7, 1.0),
    )

    # 4. BER Inferida vs Optic Seed
    plot_single_comparison(
        series_list,
        y_metric="ber_inferred",
        y_label=r"BER Inferida: $\frac{1}{2}\mathrm{erfc}\left(\frac{1}{\sqrt{2}\mathrm{EVM}}\right)$",
        title="BER Inferida (desde EVM) vs Optic Seed (dB) — CH24, CH25, CH26 y CH28",
        output_path=OUTPUT_DIR / "comparativa_ber_inferida_vs_optic_seed.png",
        log_y=True,
        shaded=True,
        y_lim=(7e-7, 1.0),
    )

    # 5. Dashboards
    dash_1x3_path = OUTPUT_DIR / "uspl_0p5_ch24_to_ch28_dashboard_1x3.png"
    plot_dashboard_1x3(series_list, dash_1x3_path)
    shutil.copy(dash_1x3_path, PLOTS_USPL_DIR / dash_1x3_path.name)

    dash_1x2_path = OUTPUT_DIR / "uspl_0p5_ch24_to_ch28_dashboard.png"
    plot_dashboard_1x2(series_list, dash_1x2_path)
    shutil.copy(dash_1x2_path, PLOTS_USPL_DIR / dash_1x2_path.name)

    dash_2x2_path = OUTPUT_DIR / "uspl_0p5_ch24_to_ch28_dashboard_2x2.png"
    plot_dashboard_2x2(series_list, dash_2x2_path)
    shutil.copy(dash_2x2_path, PLOTS_USPL_DIR / dash_2x2_path.name)

    print("\n¡Todas las gráficas comparativas generadas con éxito!")


def plot_dashboard_2x2(series_list, output_path: Path):
    """
    Dashboard 2x2 completo para publicación:
    (a) EVM RMS (%) vs Optic Seed (dB)
    (b) EVM (dB) vs Optic Seed (dB)
    (c) BER Inferida (desde EVM) vs Optic Seed (dB) [Log Scale]
    (d) BER Medida vs Optic Seed (dB) [Log Scale]
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=300)
    (ax_evm_pct, ax_evm_db), (ax_ber_inf, ax_ber_meas) = axes

    for item in series_list:
        sub_df = item["df"]

        # (a) EVM RMS (%)
        stats_evm_pct = aggregate_by_optic_seed(sub_df, "evm_rms_pct")
        ax_evm_pct.plot(
            stats_evm_pct["optic_seed_mean"], stats_evm_pct["y_mean"],
            label=item["label"], color=item["color"], marker=item["marker"],
            linestyle=item["linestyle"], linewidth=2.0, markersize=5.5, zorder=3
        )
        ax_evm_pct.fill_between(
            stats_evm_pct["optic_seed_mean"], stats_evm_pct["y_min"], stats_evm_pct["y_max"],
            color=item["color"], alpha=0.15, zorder=2
        )

        # (b) EVM (dB)
        stats_evm_db = aggregate_by_optic_seed(sub_df, "evm_db")
        ax_evm_db.plot(
            stats_evm_db["optic_seed_mean"], stats_evm_db["y_mean"],
            label=item["label"], color=item["color"], marker=item["marker"],
            linestyle=item["linestyle"], linewidth=2.0, markersize=5.5, zorder=3
        )
        ax_evm_db.fill_between(
            stats_evm_db["optic_seed_mean"], stats_evm_db["y_min"], stats_evm_db["y_max"],
            color=item["color"], alpha=0.15, zorder=2
        )

        # (c) BER Inferida
        stats_ber_inf = aggregate_by_optic_seed(sub_df, "ber_inferred")
        ber_inf_y = np.maximum(stats_ber_inf["y_mean"], 1e-6)
        ber_inf_min = np.maximum(stats_ber_inf["y_min"], 1e-6)
        ber_inf_max = np.maximum(stats_ber_inf["y_max"], 1e-6)
        ax_ber_inf.plot(
            stats_ber_inf["optic_seed_mean"], ber_inf_y,
            label=item["label"], color=item["color"], marker=item["marker"],
            linestyle=item["linestyle"], linewidth=2.0, markersize=5.5, zorder=3
        )
        ax_ber_inf.fill_between(
            stats_ber_inf["optic_seed_mean"], ber_inf_min, ber_inf_max,
            color=item["color"], alpha=0.15, zorder=2
        )

        # (d) BER Medida
        stats_ber_meas = aggregate_by_optic_seed(sub_df, "ber")
        ber_m_y = np.maximum(stats_ber_meas["y_mean"], 1e-6)
        ber_m_min = np.maximum(stats_ber_meas["y_min"], 1e-6)
        ber_m_max = np.maximum(stats_ber_meas["y_max"], 1e-6)
        ax_ber_meas.plot(
            stats_ber_meas["optic_seed_mean"], ber_m_y,
            label=item["label"], color=item["color"], marker=item["marker"],
            linestyle=item["linestyle"], linewidth=2.0, markersize=5.5, zorder=3
        )
        ax_ber_meas.fill_between(
            stats_ber_meas["optic_seed_mean"], ber_m_min, ber_m_max,
            color=item["color"], alpha=0.15, zorder=2
        )

    # (a)
    ax_evm_pct.set_xlabel("Optic Seed (dB)", fontsize=11, fontweight="bold")
    ax_evm_pct.set_ylabel("EVM RMS (%)", fontsize=11, fontweight="bold")
    ax_evm_pct.set_title("(a) EVM RMS (%) vs Optic Seed", fontsize=12, fontweight="bold")
    ax_evm_pct.grid(True, linestyle="--", alpha=0.7)
    ax_evm_pct.legend(fontsize=9, loc="best", framealpha=0.9)

    # (b)
    ax_evm_db.set_xlabel("Optic Seed (dB)", fontsize=11, fontweight="bold")
    ax_evm_db.set_ylabel("EVM (dB)", fontsize=11, fontweight="bold")
    ax_evm_db.set_title("(b) EVM (dB) vs Optic Seed", fontsize=12, fontweight="bold")
    ax_evm_db.grid(True, linestyle="--", alpha=0.7)
    ax_evm_db.legend(fontsize=9, loc="best", framealpha=0.9)

    # (c)
    ax_ber_inf.set_yscale("log")
    ax_ber_inf.set_ylim(bottom=7e-7, top=1.0)
    ax_ber_inf.set_xlabel("Optic Seed (dB)", fontsize=11, fontweight="bold")
    ax_ber_inf.set_ylabel("BER Inferida (desde EVM)", fontsize=11, fontweight="bold")
    ax_ber_inf.set_title(r"(c) BER Inferida: $\frac{1}{2}\mathrm{erfc}\left(\frac{1}{\sqrt{2}\mathrm{EVM}}\right)$", fontsize=12, fontweight="bold")
    ax_ber_inf.axhline(1e-6, color="gray", linestyle=":", alpha=0.6, label="Piso (BER = 0)", zorder=1)
    ax_ber_inf.grid(True, which="both", linestyle="--", alpha=0.7)
    ax_ber_inf.legend(fontsize=9, loc="best", framealpha=0.9)

    # (d)
    ax_ber_meas.set_yscale("log")
    ax_ber_meas.set_ylim(bottom=7e-7, top=1.0)
    ax_ber_meas.set_xlabel("Optic Seed (dB)", fontsize=11, fontweight="bold")
    ax_ber_meas.set_ylabel("BER Medida", fontsize=11, fontweight="bold")
    ax_ber_meas.set_title("(d) BER Medida vs Optic Seed", fontsize=12, fontweight="bold")
    ax_ber_meas.axhline(1e-6, color="gray", linestyle=":", alpha=0.6, label="Piso (BER = 0)", zorder=1)
    ax_ber_meas.grid(True, which="both", linestyle="--", alpha=0.7)
    ax_ber_meas.legend(fontsize=9, loc="best", framealpha=0.9)

    fig.suptitle("Dashboard Comparativo Completo: CH24, CH25, CH26 y CH28 (NLNA, SNR 0.5 dB, 370 mA, 650 MHz)", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"✓ Guardado Dashboard 2x2: {output_path.name}")


if __name__ == "__main__":
    main()
