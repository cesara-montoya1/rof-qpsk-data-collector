"""Unit tests for QPSK results plotter module."""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from qpsk_src.plotter import generate_ber_plots, load_and_prepare_data, plot_ber_curve


@pytest.fixture
def sample_csv_path(tmp_path: Path) -> Path:
    """Creates a temporary sample results CSV with 10 repetitions per SNR/OSNR."""
    rows = []
    distances = [0.0, 2.0, 20.0]
    snr_values = [3.0, 6.0, 9.0, 12.0]
    osnr_values = [30.0, 33.0, 36.0, 39.0]

    for d in distances:
        for snr, osnr in zip(snr_values, osnr_values):
            base_ber = 0.5 * np.exp(-snr / 3.0) + (d * 0.005)
            # Create 10 repetitions per condition
            for i in range(10):
                noise = np.random.uniform(-0.01, 0.01)
                ber_val = max(1e-5, base_ber + noise)
                rows.append({
                    "launch_power_dbm": 0.0,
                    "distance_km": d,
                    "osnr_db": osnr,
                    "snr_db": snr,
                    "freq_mhz": 650.0,
                    "bitrate_mbps": 2.0,
                    "filename": f"test_d{d}_s{snr}_{i}.complex64",
                    "ber": ber_val,
                    "evm_rms_pct": 10.0 + noise * 10,
                    "evm_db": -20.0 + noise * 5,
                    "evm_peak_pct": 15.0 + noise * 10,
                    "detected_delay": 100,
                    "npz_source": f"data_{int(d)}km.npz",
                })

    csv_file = tmp_path / "results_test_dataset.csv"
    pd.DataFrame(rows).to_csv(csv_file, index=False)
    return csv_file


def test_load_and_prepare_data(sample_csv_path: Path) -> None:
    df = load_and_prepare_data(sample_csv_path)
    assert not df.empty
    assert "distance_km" in df.columns
    assert "ber" in df.columns
    assert "evm_db" in df.columns
    assert "evm_rms_pct" in df.columns
    assert set(df["distance_km"].unique()) == {0.0, 2.0, 20.0}


def test_generate_ber_plots(sample_csv_path: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "plots"
    plots = generate_ber_plots(sample_csv_path, output_dir=out_dir)

    assert len(plots) == 4
    for p in plots:
        assert p.exists()
        assert p.stat().st_size > 0
        assert p.suffix == ".png"

    plot_names = [p.name for p in plots]
    assert "ber_vs_snr_mean.png" in plot_names
    assert "ber_vs_snr_shaded.png" in plot_names
    assert "ber_vs_osnr_mean.png" in plot_names
    assert "ber_vs_osnr_shaded.png" in plot_names


def test_generate_evm_plots(sample_csv_path: Path, tmp_path: Path) -> None:
    from qpsk_src.plotter import generate_all_plots, generate_evm_plots

    out_dir = tmp_path / "plots_evm"
    plots = generate_evm_plots(sample_csv_path, output_dir=out_dir)

    assert len(plots) >= 4
    for p in plots:
        assert p.exists()
        assert p.stat().st_size > 0
        assert p.suffix == ".png"

    plot_names = [p.name for p in plots]
    assert "evm_vs_snr_mean.png" in plot_names
    assert "evm_vs_snr_shaded.png" in plot_names
    assert "evm_vs_osnr_mean.png" in plot_names
    assert "evm_vs_osnr_shaded.png" in plot_names


def test_generate_all_plots(sample_csv_path: Path, tmp_path: Path) -> None:
    from qpsk_src.plotter import generate_all_plots

    out_dir = tmp_path / "plots_all"
    plots = generate_all_plots(sample_csv_path, output_dir=out_dir)

    # At least 4 BER plots + 4-6 EVM plots
    assert len(plots) >= 8
    for p in plots:
        assert p.exists()
        assert p.stat().st_size > 0


def test_load_data_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_and_prepare_data("/nonexistent/results.csv")


def test_ber_vs_snr_averaging_with_noisy_snr(tmp_path: Path) -> None:
    """Ensure noisy continuous SNR values from repeated captures are aggregated/averaged cleanly."""
    rows = []
    # 10 repetitions per condition with slight continuous SNR jitter (e.g. 10.02, 9.98, 10.05...)
    np.random.seed(42)
    nominal_snrs = [5.0, 10.0, 15.0]
    for nom_snr in nominal_snrs:
        for i in range(10):
            snr_jitter = float(np.random.uniform(-0.2, 0.2))
            ber_val = float(0.1 * np.exp(-nom_snr / 4.0) + np.random.uniform(-0.002, 0.002))
            rows.append({
                "distance_km": 0.0,
                "osnr_db": 30.0 + nom_snr,
                "snr_db": nom_snr + snr_jitter,
                "ber": max(1e-5, ber_val),
                "evm_rms_pct": 15.0 - nom_snr * 0.5,
                "evm_db": -18.0 - nom_snr * 0.5,
                "filename": f"test_snr_{nom_snr}_{i}.complex64",
            })

    noisy_csv = tmp_path / "noisy_snr_results.csv"
    pd.DataFrame(rows).to_csv(noisy_csv, index=False)

    df = load_and_prepare_data(noisy_csv)
    out_file = tmp_path / "ber_vs_snr_averaged.png"

    from qpsk_src.plotter import _aggregate_curve_data

    # Verify that the 30 continuous noisy measurements are aggregated into exactly 3 points
    stats_osnr = _aggregate_curve_data(df, x_col="snr_db", y_col="ber")
    assert len(stats_osnr) == 3
    assert all(stats_osnr["count"] == 10)
    assert np.allclose(stats_osnr["x_val"], [5.0, 10.0, 15.0], atol=0.25)

    # Verify explicit bin width aggregation when osnr is not present
    df_no_osnr = df.drop(columns=["osnr_db"])
    stats_binned = _aggregate_curve_data(df_no_osnr, x_col="snr_db", y_col="ber", snr_bin_width=1.0)
    assert len(stats_binned) == 3
    assert all(stats_binned["count"] == 10)

    # Plotting should group by condition/bin and produce an averaged curve
    p = plot_ber_curve(
        df=df,
        x_col="snr_db",
        x_label="SNR (dB)",
        title="BER vs SNR Averaged",
        output_file=out_file,
        shaded=True,
    )
    assert p.exists()
    assert p.stat().st_size > 0


def test_plot_constellation(tmp_path: Path) -> None:
    """Test generating a constellation plot with EVM and best-window contrast."""
    from qpsk_src.demodulator import modulate_qpsk
    from qpsk_src.plotter import plot_constellation

    np.random.seed(42)
    tx_bits = np.random.randint(0, 2, 2000)
    tx_symbols = modulate_qpsk(tx_bits)

    # First 100 symbols have warm-up noise, remainder is cleaner, with a very clean block in middle
    noise = (np.random.normal(0, 0.1, len(tx_symbols)) + 1j * np.random.normal(0, 0.1, len(tx_symbols)))
    noise[:100] *= 4.0
    noise[500:700] *= 0.2
    rx_signal = tx_symbols + noise

    out_png = tmp_path / "test_constellation.png"
    p = plot_constellation(
        rx_signal=rx_signal,
        tx_ref=tx_symbols,
        output_file=out_png,
        skip_initial_symbols=100,
        best_window_symbols=200,
    )

    assert p.exists()
    assert p.stat().st_size > 0

