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
