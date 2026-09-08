"""Tests for pipeline orchestrator and CLI."""

import csv
import sys
from pathlib import Path
import numpy as np
import pytest

from qpsk_src.demodulator import modulate_qpsk
from qpsk_src.pipeline import load_tx_reference, run_pipeline
from qpsk_src.cli import main as cli_main


def test_load_tx_reference(tmp_path: Path):
    tx_file = tmp_path / "tx.txt"
    tx_file.write_text("00011011\n")
    ref_symbols = load_tx_reference(tx_file)
    assert len(ref_symbols) == 4
    bits = np.array([0, 0, 0, 1, 1, 0, 1, 1])
    expected = modulate_qpsk(bits)
    np.testing.assert_allclose(ref_symbols, expected)


def test_load_tx_reference_not_found():
    with pytest.raises(FileNotFoundError):
        load_tx_reference("non_existent_file.txt")


def test_run_pipeline_end_to_end(tmp_path: Path):
    # 1. Create reference TX bits file
    bits = np.tile([0, 0, 0, 1, 1, 0, 1, 1], 50)
    tx_file = tmp_path / "tx.txt"
    tx_file.write_text("".join(map(str, bits)))

    # 2. Create mock dataset directory structure
    dataset_dir = tmp_path / "test_dataset"
    dist_dir = dataset_dir / "0km"
    dist_dir.mkdir(parents=True)

    # 3. Create a mock signal file
    sig_symbols = modulate_qpsk(bits)
    sig_file = dist_dir / "rof_0dBm_0km_osnr32p82dB_650mhz_2mbps_snr5p50dB.complex64"
    sig_symbols.astype(np.complex64).tofile(sig_file)

    # 4. Run pipeline
    output_csv = run_pipeline(dataset_dir, tx_file)

    assert output_csv.exists()
    assert output_csv.name == "results_test_dataset.csv"

    # 5. Inspect generated CSV
    with open(output_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    row = rows[0]
    assert row["filename"] == "rof_0dBm_0km_osnr32p82dB_650mhz_2mbps_snr5p50dB"
    assert float(row["ber"]) == 0.0
    assert float(row["osnr_db"]) == 32.82
    assert float(row["snr_db"]) == 5.50

    # 6. Test resumption / idempotency (re-running should not duplicate results)
    run_pipeline(dataset_dir, tx_file)
    with open(output_csv, "r", encoding="utf-8") as f:
        rows_after = list(csv.DictReader(f))
    assert len(rows_after) == 1


def test_cli_commands(tmp_path: Path, monkeypatch):
    bits = np.tile([0, 0, 0, 1, 1, 0, 1, 1], 20)
    tx_file = tmp_path / "tx.txt"
    tx_file.write_text("".join(map(str, bits)))

    dataset_dir = tmp_path / "cli_dataset"
    dist_dir = dataset_dir / "2km"
    dist_dir.mkdir(parents=True)

    sig_symbols = modulate_qpsk(bits)
    sig_file = dist_dir / "rof_0dBm_2km_osnr35p0dB_650mhz_2mbps_snr10p0dB.complex64"
    sig_symbols.astype(np.complex64).tofile(sig_file)

    # Test CLI validate
    monkeypatch.setattr(
        sys, "argv", ["cli.py", "validate", "--dataset-dir", str(dataset_dir)]
    )
    cli_main()

    # Test CLI compress
    monkeypatch.setattr(
        sys, "argv", ["cli.py", "compress", "--dataset-dir", str(dataset_dir)]
    )
    cli_main()
    assert (dist_dir / "data_2km.npz").exists()

    # Test CLI run
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.py",
            "run",
            "--dataset-dir",
            str(dataset_dir),
            "--tx-ref",
            str(tx_file),
        ],
    )
    cli_main()
    assert (dataset_dir / "results_cli_dataset.csv").exists()
