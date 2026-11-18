#!/usr/bin/env python3
"""
Optical Spectra Plotting Utility.

Parses and visualizes optical spectrum data recorded from Optical Spectrum
Analyzers (OSA) in CSV format (e.g. WaveData20070101_*.csv).

Generates:
1. Individual high-resolution dual-scale figures (dBm and linear mW) for each CSV file.
2. An overlay comparative plot of all spectra in dBm.
3. A multi-panel summary dashboard for quick visual inspection of all measurements.

Follows KISS, DRY, and modular code decoupling principles.
"""

import argparse
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("plot_spectra")


@dataclass
class TraceData:
    """Represents a single optical spectrum trace."""
    name: str
    wavelength_nm: np.ndarray
    power_w: np.ndarray
    power_mw: np.ndarray
    power_dbm: np.ndarray
    peak_wavelength_nm: float
    peak_power_mw: float
    peak_power_dbm: float


@dataclass
class SpectrumRecord:
    """Represents an OSA measurement file containing one or more traces."""
    file_path: Path
    filename: str
    date_time: str
    center_wavelength_nm: float
    span_wavelength_nm: float
    resolution_nm: float
    actual_resolution_nm: float
    traces: List[TraceData]


def watts_to_dbm(power_w: np.ndarray, noise_floor_dbm: float = -80.0) -> np.ndarray:
    """
    Converts optical power from Watts to dBm safely, clipping at noise_floor_dbm.
    P_dBm = 10 * log10(P_mW) = 10 * log10(P_W * 1000)
    """
    power_mw = power_w * 1000.0
    floor_mw = 10.0 ** (noise_floor_dbm / 10.0)
    clipped_mw = np.maximum(power_mw, floor_mw)
    return 10.0 * np.log10(clipped_mw)


def parse_spectrum_csv(
    file_path: Path,
    noise_floor_dbm: float = -80.0,
    exclude_traces: Optional[Set[str]] = None,
    include_traces: Optional[Set[str]] = None,
) -> SpectrumRecord:
    """
    Parses an OSA CSV file supporting both single-trace and multi-trace layouts.
    Filters out redundant duplicated traces.
    """
    file_path = Path(file_path)
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        lines = [line.strip() for line in f.readlines()]

    # Extract metadata from header
    meta: Dict[str, str] = {}
    for line in lines[:40]:
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) >= 2:
            meta[parts[0]] = parts[1]

    date_time = meta.get("Date/Time", "Unknown")
    try:
        center_nm = float(meta.get("Center Wavelength", 1550.0))
    except ValueError:
        center_nm = 1550.0

    try:
        span_nm = float(meta.get("Span Wavelength", 50.0))
    except ValueError:
        span_nm = 50.0

    try:
        res_nm = float(meta.get("Resolution", 0.1))
    except ValueError:
        res_nm = 0.1

    try:
        act_res_nm = float(meta.get("Actual Resolution", res_nm))
    except ValueError:
        act_res_nm = res_nm

    # Check if this is a multi-trace file (contains header starting with Wavelength(...)
    header_idx: Optional[int] = None
    for idx, line in enumerate(lines):
        if line.startswith("Wavelength("):
            header_idx = idx
            break

    traces: List[TraceData] = []

    if header_idx is not None:
        # Multi-trace table
        csv_data = "\n".join(lines[header_idx:])
        df = pd.read_csv(io.StringIO(csv_data))
        level_cols = [c for c in df.columns if "Level" in c]

        seen_levels: List[np.ndarray] = []
        for l_col in level_cols:
            trace_name = l_col.replace("Level(", "").replace(")", "").strip()
            if exclude_traces and trace_name.upper() in exclude_traces:
                continue
            if include_traces and trace_name.upper() not in include_traces:
                continue
            w_col = f"Wavelength({trace_name})"
            if w_col not in df.columns:
                continue

            w_vals = pd.to_numeric(df[w_col], errors="coerce").dropna().values
            p_vals_w = pd.to_numeric(df[l_col], errors="coerce").dropna().values

            if len(w_vals) == 0 or len(p_vals_w) == 0:
                continue

            # Ensure equal lengths
            min_len = min(len(w_vals), len(p_vals_w))
            w_vals = w_vals[:min_len]
            p_vals_w = p_vals_w[:min_len]

            # Check for duplicate traces (OSA copies active trace into empty trace slots)
            is_duplicate = any(
                np.allclose(p_vals_w, prev_p, atol=1e-12, rtol=1e-5)
                for prev_p in seen_levels
            )
            if is_duplicate:
                continue

            seen_levels.append(p_vals_w)

            p_vals_mw = p_vals_w * 1000.0
            p_vals_dbm = watts_to_dbm(p_vals_w, noise_floor_dbm=noise_floor_dbm)

            peak_idx = int(np.argmax(p_vals_w))
            peak_w = float(w_vals[peak_idx])
            peak_mw = float(p_vals_mw[peak_idx])
            peak_dbm = float(p_vals_dbm[peak_idx])

            traces.append(
                TraceData(
                    name=trace_name,
                    wavelength_nm=w_vals,
                    power_w=p_vals_w,
                    power_mw=p_vals_mw,
                    power_dbm=p_vals_dbm,
                    peak_wavelength_nm=peak_w,
                    peak_power_mw=peak_mw,
                    peak_power_dbm=peak_dbm,
                )
            )
    else:
        # Single-trace file
        # Determine trace name from header if present
        trace_name = "A"
        for line in lines[:35]:
            if line.startswith("Trace,"):
                parts = line.split(",")
                if len(parts) >= 2 and parts[1].strip():
                    trace_name = parts[1].strip()

        # Find first line of numerical data
        data_start_idx = 0
        for idx, line in enumerate(lines):
            parts = line.split(",")
            if len(parts) >= 2 and parts[0].replace(".", "").isdigit():
                data_start_idx = idx
                break

        csv_data = "\n".join(lines[data_start_idx:])
        df = pd.read_csv(
            io.StringIO(csv_data),
            header=None,
            names=["Wavelength", "Level"],
            usecols=[0, 1],
        )

        w_vals = pd.to_numeric(df["Wavelength"], errors="coerce").dropna().values
        p_vals_w = pd.to_numeric(df["Level"], errors="coerce").dropna().values

        min_len = min(len(w_vals), len(p_vals_w))
        w_vals = w_vals[:min_len]
        p_vals_w = p_vals_w[:min_len]

        p_vals_mw = p_vals_w * 1000.0
        p_vals_dbm = watts_to_dbm(p_vals_w, noise_floor_dbm=noise_floor_dbm)

        peak_idx = int(np.argmax(p_vals_w))
        peak_w = float(w_vals[peak_idx])
        peak_mw = float(p_vals_mw[peak_idx])
        peak_dbm = float(p_vals_dbm[peak_idx])

        traces.append(
            TraceData(
                name=trace_name,
                wavelength_nm=w_vals,
                power_w=p_vals_w,
                power_mw=p_vals_mw,
                power_dbm=p_vals_dbm,
                peak_wavelength_nm=peak_w,
                peak_power_mw=peak_mw,
                peak_power_dbm=peak_dbm,
            )
        )

    return SpectrumRecord(
        file_path=file_path,
        filename=file_path.name,
        date_time=date_time,
        center_wavelength_nm=center_nm,
        span_wavelength_nm=span_nm,
        resolution_nm=res_nm,
        actual_resolution_nm=act_res_nm,
        traces=traces,
    )


# Color palette for consistent trace identification
TRACE_COLORS = {
    "A": "#1f77b4",  # Blue
    "B": "#ff7f0e",  # Orange
    "C": "#2ca02c",  # Green
    "D": "#d62728",  # Red
    "E": "#9467bd",  # Purple
    "F": "#8c564b",  # Brown
    "G": "#e377c2",  # Pink
    "H": "#7f7f7f",  # Gray
    "I": "#bcbd22",  # Olive
    "J": "#17becf",  # Cyan
}

# Default file-specific trace exclusions (empty by default; use --exclude-traces if needed)
FILE_EXCLUDED_TRACES: Dict[str, Set[str]] = {}


def format_spectrum_label(stem: str) -> str:
    """Formats a human-readable clean label from filename stem."""
    import re
    m = re.search(r"Canal_(\d+)(?:_(\d+km|[a-zA-Z0-9]+))?", stem, re.IGNORECASE)
    if m:
        ch = m.group(1)
        dist = m.group(2)
        if dist:
            return f"Canal {ch} ({dist})"
        return f"Canal {ch}"
    if stem.startswith("WaveData"):
        return f"WaveData {stem[-3:]}"
    return stem


def plot_individual_spectrum(
    spec: SpectrumRecord,
    output_dir: Path,
    dpi: int = 300,
) -> Path:
    """
    Plots an individual optical spectrum file in a 2-panel figure:
    Top: Optical Power Spectrum [dBm] (logarithmic scale)
    Bottom: Optical Power Spectrum [mW] (linear scale)
    """
    fig, (ax_dbm, ax_mw) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"hspace": 0.25}
    )

    for trace in spec.traces:
        color = TRACE_COLORS.get(trace.name.upper(), "#333333")
        label = (
            f"Trace {trace.name} (Peak: {trace.peak_power_dbm:.2f} dBm @ {trace.peak_wavelength_nm:.2f} nm)"
        )

        # Panel 1: dBm
        ax_dbm.plot(
            trace.wavelength_nm,
            trace.power_dbm,
            label=label,
            color=color,
            linewidth=1.4,
        )
        # Mark peak
        ax_dbm.plot(
            trace.peak_wavelength_nm,
            trace.peak_power_dbm,
            marker="o",
            markersize=6,
            color=color,
            markeredgecolor="black",
            markeredgewidth=0.8,
        )
        offset_x = -70 if trace.peak_wavelength_nm >= spec.center_wavelength_nm else 15
        ax_dbm.annotate(
            f"{trace.peak_power_dbm:.1f} dBm\n{trace.peak_wavelength_nm:.2f} nm",
            xy=(trace.peak_wavelength_nm, trace.peak_power_dbm),
            xytext=(offset_x, 8),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.85),
        )

        # Panel 2: mW
        label_mw = f"Trace {trace.name} (Peak: {trace.peak_power_mw:.3f} mW)"
        ax_mw.plot(
            trace.wavelength_nm,
            trace.power_mw,
            label=label_mw,
            color=color,
            linewidth=1.4,
        )
        ax_mw.plot(
            trace.peak_wavelength_nm,
            trace.peak_power_mw,
            marker="o",
            markersize=6,
            color=color,
            markeredgecolor="black",
            markeredgewidth=0.8,
        )

    # Subplot 1 formatting (dBm) with headroom for legend and peak tags
    max_dbm = max(t.peak_power_dbm for t in spec.traces)
    min_dbm = min(t.power_dbm.min() for t in spec.traces)
    ax_dbm.set_ylim(bottom=min_dbm - 4, top=max_dbm + 14)
    ax_dbm.set_ylabel("Optical Power (dBm)", fontsize=11, fontweight="bold")
    ax_dbm.set_title(
        f"Optical Spectrum: {spec.filename}\n"
        f"Span: {spec.span_wavelength_nm:.0f} nm | Center: {spec.center_wavelength_nm:.2f} nm | "
        f"Res: {spec.resolution_nm:.2f} nm (Actual: {spec.actual_resolution_nm:.3f} nm) | Date: {spec.date_time}",
        fontsize=11,
        fontweight="bold",
    )
    ax_dbm.grid(True, linestyle="--", alpha=0.5)
    ax_dbm.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # Subplot 2 formatting (mW)
    max_mw = max(t.peak_power_mw for t in spec.traces)
    ax_mw.set_ylim(bottom=-0.03 * max_mw, top=max_mw * 1.22)
    ax_mw.set_xlabel("Wavelength (nm)", fontsize=11, fontweight="bold")
    ax_mw.set_ylabel("Optical Power (mW)", fontsize=11, fontweight="bold")
    ax_mw.grid(True, linestyle="--", alpha=0.5)
    ax_mw.legend(loc="upper right", fontsize=9, framealpha=0.9)

    out_name = f"{spec.file_path.stem}_spectrum.png"
    out_path = output_dir / out_name
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"✓ Saved individual plot: {out_path}")
    return out_path


def plot_all_spectra_overlay(
    records: List[SpectrumRecord],
    output_dir: Path,
    pump_current: Optional[str] = None,
    dpi: int = 300,
) -> Path:
    """
    Generates a consolidated overlay comparison of all spectra across all CSV files.
    Plots Optical Power Spectrum (dBm) vs Wavelength (nm) on the same axes.
    Applies baseline cleaning at -45 dBm to eliminate detector zero-crossing noise spikes.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    all_traces: List[Tuple[SpectrumRecord, TraceData, int, int]] = []
    for f_idx, spec in enumerate(records):
        for t_idx, trace in enumerate(spec.traces):
            all_traces.append((spec, trace, f_idx, t_idx))

    colors = plt.cm.tab20(np.linspace(0, 1, len(all_traces)))

    for i, (spec, trace, f_idx, t_idx) in enumerate(all_traces):
        clean_name = format_spectrum_label(spec.file_path.stem)
        label = f"{clean_name} [{trace.name}] (Peak: {trace.peak_power_dbm:.1f} dBm @ {trace.peak_wavelength_nm:.1f} nm)"
        ls = "-" if t_idx == 0 else "--"
        # Clean detector noise spikes below -45 dBm
        p_clean = np.maximum(trace.power_dbm, -45.0)
        ax.plot(
            trace.wavelength_nm,
            p_clean,
            label=label,
            color=colors[i],
            linestyle=ls,
            linewidth=1.5,
            alpha=0.9,
        )

    title_suffix = f" — Bombeo: {pump_current} mA" if pump_current else ""
    ax.set_title(
        f"Comparación de Espectros Ópticos Superpuestos (Todas las Trazas){title_suffix}\n"
        "Optical Power (dBm) vs Wavelength (nm)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel(r"Longitud de Onda $\lambda$ (nm)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Potencia Óptica (dBm)", fontsize=12, fontweight="bold")
    ax.set_xlim(1500, 1615)
    ax.set_ylim(-46, 18)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8.5,
        framealpha=0.95,
        title="Espectros & Trazas",
        title_fontsize=9.5,
    )
    fig.tight_layout()

    out_path = output_dir / "all_spectra_overlay.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    # Also save with Spanish explicit alias
    fig.savefig(output_dir / "espectros_superpuestos_overlay.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"✓ Saved overlay plot: {out_path} and espectros_superpuestos_overlay.png")
    return out_path


def plot_main_spectra_overlay(
    records: List[SpectrumRecord],
    output_dir: Path,
    pump_current: Optional[str] = None,
    dpi: int = 300,
) -> Path:
    """
    Plots the primary optical spectrum of each CSV file superimposed
    on the exact same axis in dBm for a crystal-clear direct 1-to-1 comparison.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    cmap10 = plt.get_cmap("tab10")

    for idx, spec in enumerate(records):
        clean_name = format_spectrum_label(spec.file_path.stem)
        # Choose the trace with highest peak power as the primary spectrum
        main_trace = max(spec.traces, key=lambda t: t.peak_power_dbm)
        label = (
            f"{clean_name} [{main_trace.name}] "
            f"(Pico: {main_trace.peak_power_dbm:.1f} dBm @ {main_trace.peak_wavelength_nm:.1f} nm)"
        )
        p_clean = np.maximum(main_trace.power_dbm, -45.0)
        ax.plot(
            main_trace.wavelength_nm,
            p_clean,
            label=label,
            color=cmap10(idx % 10),
            linewidth=1.6,
            alpha=0.9,
        )

    title_suffix = f" — Bombeo: {pump_current} mA" if pump_current else ""
    ax.set_title(
        f"Comparación de Espectros Principales Superpuestos (1 por Archivo CSV){title_suffix}\n"
        "Optical Power (dBm) vs Wavelength (nm)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel(r"Longitud de Onda $\lambda$ (nm)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Potencia Óptica (dBm)", fontsize=12, fontweight="bold")
    ax.set_xlim(1500, 1615)
    ax.set_ylim(-46, 18)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=9.0,
        framealpha=0.95,
        title=f"Archivos CSV ({len(records)} Mediciones)",
        title_fontsize=10.0,
    )
    fig.tight_layout()

    out_path = output_dir / "espectros_principales_superpuestos.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"✓ Saved main spectra overlay: {out_path}")
    return out_path


def plot_waterfall_spectra(
    records: List[SpectrumRecord],
    output_dir: Path,
    offset_step_db: float = 10.0,
    pump_current: Optional[str] = None,
    dpi: int = 300,
) -> Path:
    """
    Plots spectra with a vertical offset (Waterfall), literally placing them 'one above the other'
    to compare shapes, widths, and peak wavelengths without line occlusion.
    """
    fig, ax = plt.subplots(figsize=(14, 11))

    all_traces: List[Tuple[SpectrumRecord, TraceData, int, int]] = []
    for f_idx, spec in enumerate(records):
        for t_idx, trace in enumerate(spec.traces):
            all_traces.append((spec, trace, f_idx, t_idx))

    colors = plt.cm.tab20(np.linspace(0, 1, len(all_traces)))
    max_peak_shifted = -999.0

    for i, (spec, trace, f_idx, t_idx) in enumerate(all_traces):
        clean_name = format_spectrum_label(spec.file_path.stem)
        offset = i * offset_step_db
        p_clean = np.maximum(trace.power_dbm, -45.0)
        y_shifted = p_clean + offset
        label = f"{clean_name} [{trace.name}] (+{offset:.0f} dB)"
        ax.plot(
            trace.wavelength_nm,
            y_shifted,
            label=label,
            color=colors[i],
            linewidth=1.4,
            alpha=0.95,
        )

        # Baseline indicator line
        ax.axhline(
            -45.0 + offset,
            xmin=0.03,
            xmax=0.97,
            color=colors[i],
            linestyle=":",
            alpha=0.25,
            linewidth=0.8,
        )

        # Peak marker
        pk_w = trace.peak_wavelength_nm
        pk_y = trace.peak_power_dbm + offset
        if pk_y > max_peak_shifted:
            max_peak_shifted = pk_y
        ax.plot(pk_w, pk_y, "o", color=colors[i], markersize=4)

        # Name tag on the right side
        tag = f"{clean_name} [{trace.name}]"
        tag_x = trace.wavelength_nm[-1] + 0.8
        ax.text(
            tag_x,
            y_shifted[-1],
            tag,
            color=colors[i],
            fontsize=8,
            fontweight="bold",
            va="center",
        )

    title_suffix = f" — Bombeo: {pump_current} mA" if pump_current else ""
    ax.set_title(
        f"Comparación de Espectros Ópticos con Desplazamiento Vertical (Waterfall){title_suffix}\n"
        f"Visualización 'Una sobre la otra' con Offset de +{offset_step_db:.0f} dB por Traza",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel(r"Longitud de Onda $\lambda$ (nm)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Potencia Óptica (dBm) + Offset Vertical (dB)", fontsize=12, fontweight="bold")
    ax.set_xlim(1500, 1630)
    ax.set_ylim(-52, max(max_peak_shifted + 15, len(all_traces) * offset_step_db + 15))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=8.5,
        framealpha=0.95,
        title="Trazas (+Offset)",
        title_fontsize=9.5,
    )
    fig.tight_layout()

    out_path = output_dir / "espectros_waterfall_offset.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"✓ Saved waterfall plot: {out_path}")
    return out_path


def plot_stacked_vertical_spectra(
    records: List[SpectrumRecord],
    output_dir: Path,
    pump_current: Optional[str] = None,
    dpi: int = 300,
) -> Path:
    """
    Plots all CSV files in a single column of vertically stacked subplots sharing the X-axis.
    Allows exact vertical alignment comparison across wavelengths.
    """
    n_files = len(records)
    fig, axes = plt.subplots(
        n_files, 1, figsize=(12, 1.8 * n_files), sharex=True, gridspec_kw={"hspace": 0.08}
    )
    if n_files == 1:
        axes = [axes]

    cmap10 = plt.get_cmap("tab10")

    for idx, (spec, ax) in enumerate(zip(records, axes)):
        clean_name = format_spectrum_label(spec.file_path.stem)
        for t_idx, trace in enumerate(spec.traces):
            ls = "-" if t_idx == 0 else "--"
            p_clean = np.maximum(trace.power_dbm, -45.0)
            ax.plot(
                trace.wavelength_nm,
                p_clean,
                color=cmap10(idx % 10),
                linestyle=ls,
                linewidth=1.3,
                label=f"{trace.name} ({trace.peak_power_dbm:.1f} dBm)",
            )
            ax.plot(trace.peak_wavelength_nm, trace.peak_power_dbm, "o", color=cmap10(idx % 10), markersize=3.5)

        ax.set_ylabel(clean_name, fontsize=8.5, fontweight="bold", rotation=0, labelpad=35, va="center")
        ax.set_ylim(-46, 18)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", fontsize=7.5, framealpha=0.85)
        ax.tick_params(axis="both", labelsize=8)

    axes[-1].set_xlabel(r"Longitud de Onda $\lambda$ (nm)", fontsize=11, fontweight="bold")
    title_suffix = f" — Bombeo: {pump_current} mA" if pump_current else ""
    fig.suptitle(
        f"Espectros Ópticos Apilados Verticalmente (Alineación Directa de $\\lambda$){title_suffix}",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    out_path = output_dir / "espectros_apilados_vertical.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"✓ Saved stacked vertical plot: {out_path}")
    return out_path


def plot_spectra_dashboard(
    records: List[SpectrumRecord],
    output_dir: Path,
    pump_current: Optional[str] = None,
    dpi: int = 300,
) -> Path:
    """
    Creates a multi-panel dashboard displaying all spectra side-by-side in a 2-column grid.
    """
    n_records = len(records)
    cols = 2
    rows = (n_records + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols, figsize=(16, 3.2 * rows), sharex=False, sharey=False
    )
    axes = np.array(axes).flatten()

    for idx, (spec, ax) in enumerate(zip(records, axes)):
        clean_name = format_spectrum_label(spec.file_path.stem)
        for trace in spec.traces:
            color = TRACE_COLORS.get(trace.name.upper(), "#1f77b4")
            label = f"Tr {trace.name}: {trace.peak_power_dbm:.1f} dBm"
            ax.plot(
                trace.wavelength_nm,
                trace.power_dbm,
                label=label,
                color=color,
                linewidth=1.2,
            )
            ax.plot(
                trace.peak_wavelength_nm,
                trace.peak_power_dbm,
                marker="o",
                markersize=4,
                color=color,
            )

        ax.margins(y=0.15)
        ax.set_title(
            f"{clean_name} ({spec.filename} | Res: {spec.resolution_nm:.2f} nm)",
            fontsize=9.5,
            fontweight="bold",
        )
        ax.set_xlabel(r"$\lambda$ (nm)", fontsize=8.5)
        ax.set_ylabel("Power (dBm)", fontsize=8.5)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=7.5, framealpha=0.8)
        ax.tick_params(axis="both", labelsize=8)

    # Hide any unused subplots
    for ax in axes[n_records:]:
        ax.axis("off")

    title_suffix = f" — Bombeo: {pump_current} mA" if pump_current else ""
    fig.suptitle(
        f"Optical Spectra Dashboard — All Measurements (dBm vs Wavelength){title_suffix}",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    out_path = output_dir / "spectra_dashboard.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"✓ Saved dashboard: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot optical spectra from OSA CSV files in espectros folder."
    )
    default_input = Path("/export2/hanna.colorado/lafi_shared/espectros")
    default_output = default_input / "plots"

    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        default=default_input,
        help=f"Directory containing CSV files (default: {default_input})",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=default_output,
        help=f"Directory to save plots (default: {default_output})",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default=None,
        help="Specific CSV file name or path to plot (default: all files in input-dir)",
    )
    parser.add_argument(
        "--pump-current",
        type=str,
        default=None,
        help="Pump current in mA (e.g. '370' or '350') to display in titles and labels",
    )
    parser.add_argument(
        "--exclude-traces",
        type=str,
        default=None,
        help="Comma-separated list of trace names to exclude (e.g. 'A')",
    )
    parser.add_argument(
        "--include-traces",
        type=str,
        default=None,
        help="Comma-separated list of trace names to include (e.g. 'B,C')",
    )
    parser.add_argument(
        "--noise-floor",
        type=float,
        default=-80.0,
        help="Noise floor in dBm for clipping detector zeros/negatives (default: -80 dBm)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI resolution (default: 300)",
    )

    args = parser.parse_args()

    input_dir: Path = args.input_dir.resolve()
    output_dir: Path = args.output_dir.resolve()

    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.file:
        candidate = Path(args.file.strip())
        if candidate.is_absolute() and candidate.exists():
            target = candidate
        elif (input_dir / candidate).exists():
            target = input_dir / candidate
        elif candidate.exists():
            target = candidate.resolve()
        else:
            logger.error(f"Specified CSV file not found: {args.file} (checked {candidate} and {input_dir / candidate})")
            sys.exit(1)
        csv_files = [target]
    else:
        csv_files = sorted(list(input_dir.glob("*.csv")))
    if not csv_files:
        logger.error(f"No CSV files found in {input_dir}")
        sys.exit(1)

    logger.info(f"Found {len(csv_files)} CSV files in {input_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Parse all CSV files
    records: List[SpectrumRecord] = []
    for csv_file in csv_files:
        try:
            # Filter traces (by file rule or CLI arguments)
            excl_set = set(FILE_EXCLUDED_TRACES.get(csv_file.name, set()))
            if args.exclude_traces:
                excl_set.update(x.strip().upper() for x in args.exclude_traces.split(",") if x.strip())

            incl_set = None
            if args.include_traces:
                incl_set = {x.strip().upper() for x in args.include_traces.split(",") if x.strip()}

            record = parse_spectrum_csv(
                csv_file,
                noise_floor_dbm=args.noise_floor,
                exclude_traces=excl_set,
                include_traces=incl_set,
            )

            if not record.traces:
                logger.warning(f"No traces remaining for {csv_file.name} after filtering. Skipping.")
                continue

            records.append(record)
            trace_desc = ", ".join(
                [f"{t.name} (peak: {t.peak_power_dbm:.2f} dBm @ {t.peak_wavelength_nm:.2f} nm)" for t in record.traces]
            )
            logger.info(f"Loaded {csv_file.name}: {len(record.traces)} trace(s) [{trace_desc}]")
        except Exception as e:
            logger.error(f"Failed to parse {csv_file}: {e}", exc_info=True)

    if not records:
        logger.error("No valid spectrum records parsed.")
        sys.exit(1)

    # 1. Individual plots for each CSV
    logger.info("Generating individual spectrum plots...")
    for rec in records:
        plot_individual_spectrum(rec, output_dir=output_dir, dpi=args.dpi)

    if len(records) > 1:
        # 2. Overlay comparison plot (All traces superposed in a single plot)
        logger.info("Generating consolidated overlay plot (all traces)...")
        plot_all_spectra_overlay(records, output_dir=output_dir, pump_current=args.pump_current, dpi=args.dpi)

        # 3. Main spectra comparison plot (1 primary spectrum per CSV file superposed)
        logger.info("Generating main spectra comparison plot (1 per CSV)...")
        plot_main_spectra_overlay(records, output_dir=output_dir, pump_current=args.pump_current, dpi=args.dpi)

        # 4. Waterfall plot (Vertically shifted 'one above the other' with +10 dB offset)
        logger.info("Generating waterfall offset plot ('one above the other')...")
        plot_waterfall_spectra(records, output_dir=output_dir, offset_step_db=10.0, pump_current=args.pump_current, dpi=args.dpi)

        # 5. Stacked vertical subplots (Aligned column sharing X-axis)
        logger.info("Generating vertically stacked subplots...")
        plot_stacked_vertical_spectra(records, output_dir=output_dir, pump_current=args.pump_current, dpi=args.dpi)

        # 6. Summary Dashboard (5x2 grid)
        logger.info("Generating spectra dashboard (5x2 grid)...")
        plot_spectra_dashboard(records, output_dir=output_dir, pump_current=args.pump_current, dpi=args.dpi)
    else:
        logger.info("Single file processed: skipped multi-spectra comparison plots.")

    logger.info("=" * 60)
    logger.info(f"All optical spectra successfully generated in: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    import sys
    main()
