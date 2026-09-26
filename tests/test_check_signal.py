"""Tests for quick transmission check (constellation, BER, EVM) functionality."""

from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from qpsk_src.cli import main
from qpsk_src.demodulator import modulate_qpsk
from qpsk_src.plotter import plot_constellation


@pytest.fixture
def dummy_tx_rx_data(tmp_path: Path):
    """Generates synthetic TX reference file and matching RX complex64 file."""
    np.random.seed(42)
    tx_bits = np.random.randint(0, 2, 2000)
    tx_symbols = modulate_qpsk(tx_bits)

    # Save tx.txt
    tx_txt_path = tmp_path / "tx.txt"
    tx_txt_path.write_text("".join(str(b) for b in tx_bits), encoding="utf-8")

    # Add small noise to create realistic RX signal
    noise = (np.random.normal(0, 0.05, len(tx_symbols)) + 1j * np.random.normal(0, 0.05, len(tx_symbols)))
    rx_signal = (tx_symbols + noise).astype(np.complex64)

    rx_file_path = tmp_path / "sample_test.complex64"
    rx_signal.tofile(rx_file_path)

    return tx_txt_path, rx_file_path, tx_symbols, rx_signal


def test_plot_constellation_return_metrics(dummy_tx_rx_data, tmp_path: Path):
    """Test that plot_constellation returns metrics dictionary when return_metrics=True."""
    _, _, tx_symbols, rx_signal = dummy_tx_rx_data
    out_file = tmp_path / "custom_out.png"

    result = plot_constellation(
        rx_signal=rx_signal,
        tx_ref=tx_symbols,
        output_file=out_file,
        return_metrics=True,
        show=False,
    )

    assert isinstance(result, tuple)
    out_path, metrics = result
    assert out_path == out_file
    assert out_path.exists()
    assert "ber" in metrics
    assert "evm_rms_pct" in metrics
    assert "evm_db" in metrics
    assert "evm_peak_pct" in metrics
    assert "best_rotation" in metrics
    assert "detected_delay" in metrics
    assert metrics["ber"] == 0.0  # clean signal with tiny noise should be 0 errors
    assert metrics["evm_rms_pct"] > 0.0


def test_plot_constellation_default_tmp_path(dummy_tx_rx_data):
    """Test that plot_constellation defaults to /tmp when output_file is None."""
    _, _, tx_symbols, rx_signal = dummy_tx_rx_data

    out_path = plot_constellation(
        rx_signal=rx_signal,
        tx_ref=tx_symbols,
        output_file=None,
        show=False,
    )

    assert Path(tempfile.gettempdir()) in out_path.parents or out_path.parent == Path(tempfile.gettempdir())
    assert out_path.exists()


def test_cli_constellation_positional_and_flags(dummy_tx_rx_data, tmp_path: Path, monkeypatch, capsys):
    """Test running CLI constellation with positional argument and saving to /tmp by default."""
    tx_txt_path, rx_file_path, _, _ = dummy_tx_rx_data

    test_args = [
        "cli.py",
        "constellation",
        str(rx_file_path),
        "--tx-ref",
        str(tx_txt_path),
        "--no-show",
    ]
    monkeypatch.setattr("sys.argv", test_args)

    main()

    captured = capsys.readouterr().out
    assert "REVISIÓN RÁPIDA DE TRANSMISIÓN QPSK" in captured
    assert "sample_test.complex64" in captured
    assert "BER:" in captured
    assert "EVM (RMS):" in captured
    assert "Gráfica temporal:" in captured or "/tmp" in captured


def test_cli_constellation_custom_output(dummy_tx_rx_data, tmp_path: Path, monkeypatch, capsys):
    """Test running CLI constellation with custom -o output."""
    tx_txt_path, rx_file_path, _, _ = dummy_tx_rx_data
    custom_png = tmp_path / "saved_report.png"

    test_args = [
        "cli.py",
        "constellation",
        "-r",
        str(rx_file_path),
        "--tx-ref",
        str(tx_txt_path),
        "-o",
        str(custom_png),
        "--no-show",
    ]
    monkeypatch.setattr("sys.argv", test_args)

    main()

    assert custom_png.exists()
    assert custom_png.stat().st_size > 0
    captured = capsys.readouterr().out
    assert "Gráfica guardada en:" in captured
