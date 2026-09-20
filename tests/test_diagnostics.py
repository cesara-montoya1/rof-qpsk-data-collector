"""Unit tests for advanced channel diagnostics module."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from qpsk_src.demodulator import QPSK_CONSTELLATION, modulate_qpsk
from qpsk_src.diagnostics import (
    compute_polar_evm,
    generate_dataset_diagnostics,
    generate_signal_diagnostics,
    plot_ber_vs_evm_theoretical,
    plot_constellation_density,
    plot_error_burst_profile,
    plot_error_vector_spectrum,
    plot_polar_evm,
)


@pytest.fixture
def sample_qpsk_data():
    """Generates synthetic TX and RX symbols with known impairment."""
    np.random.seed(42)
    bits = np.random.randint(0, 2, 4000)
    tx = modulate_qpsk(bits)

    # Add both amplitude jitter and small phase noise
    amp_noise = np.random.normal(0, 0.05, len(tx))
    phase_noise = np.random.normal(0, 0.08, len(tx))
    rx = (np.abs(tx) + amp_noise) * np.exp(1j * (np.angle(tx) + phase_noise))
    return tx, rx, bits


def test_compute_polar_evm_mechanisms():
    """Verify that compute_polar_evm separates amplitude noise from phase noise."""
    np.random.seed(42)
    tx = np.tile(QPSK_CONSTELLATION, 500)

    # 1. Pure phase noise
    phase_jitter = np.random.normal(0, 0.1, len(tx))
    rx_phase = tx * np.exp(1j * phase_jitter)
    m_phase = compute_polar_evm(rx_phase, tx)

    assert m_phase["evm_theta_pct"] > m_phase["evm_r_pct"]
    assert m_phase["phase_jitter_deg"] > 4.0
    assert m_phase["phase_amplitude_ratio_db"] > 0

    # 2. Pure amplitude noise
    amp_jitter = np.random.normal(0, 0.1, len(tx))
    rx_amp = tx * (1.0 + amp_jitter)
    m_amp = compute_polar_evm(rx_amp, tx)

    assert m_amp["evm_r_pct"] > m_amp["evm_theta_pct"]
    assert m_amp["phase_amplitude_ratio_db"] < 0


def test_plot_ber_vs_evm_theoretical(tmp_path: Path):
    """Test generating empirical BER vs EVM plot with theoretical AWGN curve."""
    df = pd.DataFrame({
        "distance_km": [0.0, 0.0, 2.0, 2.0, 20.0, 20.0],
        "evm_rms_pct": [10.0, 15.0, 12.0, 18.0, 20.0, 28.0],
        "ber": [1e-4, 5e-3, 2e-4, 1e-2, 5e-3, 4e-2],
    })
    out_file = tmp_path / "ber_vs_evm_test.png"
    p = plot_ber_vs_evm_theoretical(df, out_file)

    assert p.exists()
    assert p.stat().st_size > 0


def test_plot_error_vector_spectrum(sample_qpsk_data, tmp_path: Path):
    """Test error vector power spectral density (PSD) plotting."""
    tx, rx, _ = sample_qpsk_data
    out_file = tmp_path / "spectrum_test.png"
    p = plot_error_vector_spectrum(rx, tx, out_file)

    assert p.exists()
    assert p.stat().st_size > 0


def test_plot_constellation_density(sample_qpsk_data, tmp_path: Path):
    """Test 2D constellation density heatmap plotting."""
    _, rx, _ = sample_qpsk_data
    out_file = tmp_path / "density_test.png"
    p = plot_constellation_density(rx, out_file)

    assert p.exists()
    assert p.stat().st_size > 0


def test_plot_polar_evm(sample_qpsk_data, tmp_path: Path):
    """Test polar EVM phase vs amplitude breakdown plotting."""
    tx, rx, _ = sample_qpsk_data
    out_file = tmp_path / "polar_evm_test.png"
    p = plot_polar_evm(rx, tx, out_file)

    assert p.exists()
    assert p.stat().st_size > 0


def test_plot_error_burst_profile(tmp_path: Path):
    """Test cumulative error distribution plotting."""
    np.random.seed(42)
    tx_bits = np.random.randint(0, 2, 2000)
    rx_bits = tx_bits.copy()
    # Insert burst of errors at index 800-850
    rx_bits[800:850] = 1 - rx_bits[800:850]

    out_file = tmp_path / "burst_test.png"
    p = plot_error_burst_profile(rx_bits, tx_bits, out_file)

    assert p.exists()
    assert p.stat().st_size > 0


def test_generate_dataset_diagnostics_configurable_destination(tmp_path: Path):
    """
    Verify that dataset diagnostics save to subfolder 'diagnostics/' by default,
    and directly into the target directory when subfolder='' is configured.
    """
    df = pd.DataFrame({
        "distance_km": [0.0, 10.0],
        "evm_rms_pct": [12.0, 22.0],
        "ber": [1e-4, 1e-2],
    })
    csv_file = tmp_path / "results_test.csv"
    df.to_csv(csv_file, index=False)

    # 1. Default subfolder='diagnostics'
    plots_default = generate_dataset_diagnostics(csv_file, output_dir=tmp_path / "plots")
    assert len(plots_default) > 0
    for p in plots_default:
        assert p.parent.name == "diagnostics"
        assert p.exists()

    # 2. Configurable: subfolder='' saves directly in main plots directory
    plots_flattened = generate_dataset_diagnostics(
        csv_file, output_dir=tmp_path / "plots", subfolder=""
    )
    assert len(plots_flattened) > 0
    for p in plots_flattened:
        assert p.parent.name == "plots"
        assert p.exists()


def test_generate_signal_diagnostics_configurable_destination(sample_qpsk_data, tmp_path: Path):
    """
    Verify that signal diagnostics save to subfolder 'diagnostics/' by default,
    and directly into the target directory when subfolder='' is configured.
    """
    tx, rx, _ = sample_qpsk_data

    # 1. Default subfolder='diagnostics'
    plots_default = generate_signal_diagnostics(rx, tx, output_dir=tmp_path / "out")
    assert len(plots_default) >= 4
    for p in plots_default:
        assert p.parent.name == "diagnostics"
        assert p.exists()

    # 2. Configurable: subfolder='' saves directly in main output directory
    plots_custom = generate_signal_diagnostics(
        rx, tx, output_dir=tmp_path / "out_flat", subfolder=""
    )
    assert len(plots_custom) >= 4
    for p in plots_custom:
        assert p.parent.name == "out_flat"
        assert p.exists()
