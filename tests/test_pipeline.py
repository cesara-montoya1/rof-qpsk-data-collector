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
    assert "evm_rms_pct" in row
    assert float(row["evm_rms_pct"]) < 1e-3
    assert "evm_db" in row
    assert "evm_peak_pct" in row

    # 6. Test resumption / idempotency (re-running should not duplicate results)
    run_pipeline(dataset_dir, tx_file)
    with open(output_csv, "r", encoding="utf-8") as f:
        rows_after = list(csv.DictReader(f))
    assert len(rows_after) == 1
    assert "evm_rms_pct" in rows_after[0]


def test_pipeline_backfill_evm_for_existing_ber(tmp_path: Path):
    """
    If a CSV already exists with BER but missing EVM columns,
    re-running the pipeline should compute EVM and update existing rows without duplicating.
    """
    bits = np.tile([0, 0, 0, 1, 1, 0, 1, 1], 40)
    tx_file = tmp_path / "tx.txt"
    tx_file.write_text("".join(map(str, bits)))

    dataset_dir = tmp_path / "backfill_dataset"
    dist_dir = dataset_dir / "0km"
    dist_dir.mkdir(parents=True)

    sig_symbols = modulate_qpsk(bits)
    filename = "rof_0dBm_0km_osnr30p0dB_650mhz_2mbps_snr10p0dB"
    sig_file = dist_dir / f"{filename}.complex64"
    sig_symbols.astype(np.complex64).tofile(sig_file)

    # Pre-create output CSV with ONLY BER (no EVM fields)
    csv_file = dataset_dir / "results_backfill_dataset.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "launch_power_dbm",
                "distance_km",
                "osnr_db",
                "snr_db",
                "freq_mhz",
                "bitrate_mbps",
                "filename",
                "ber",
                "detected_delay",
                "npz_source",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "launch_power_dbm": "0.0",
            "distance_km": "0.0",
            "osnr_db": "30.0",
            "snr_db": "10.0",
            "freq_mhz": "650",
            "bitrate_mbps": "2",
            "filename": filename,
            "ber": "0.0",
            "detected_delay": "0",
            "npz_source": "data_0km.npz",
        })

    # Run pipeline on the dataset with existing CSV
    out_csv = run_pipeline(dataset_dir, tx_file, output_csv_path=csv_file)

    with open(out_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Should NOT have duplicated rows
    assert len(rows) == 1
    updated_row = rows[0]
    assert updated_row["filename"] == filename
    assert updated_row["ber"] == "0.0"
    # EVM should now be calculated and filled in
    assert "evm_rms_pct" in updated_row
    assert float(updated_row["evm_rms_pct"]) < 1e-3
    assert "evm_db" in updated_row
    assert "evm_peak_pct" in updated_row


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
    csv_res = dataset_dir / "results_cli_dataset.csv"
    assert csv_res.exists()

    # Test CLI plot
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.py",
            "plot",
            "--csv-path",
            str(csv_res),
            "--metric",
            "all",
        ],
    )
    cli_main()
    plots_dir = dataset_dir / "plots"
    assert (plots_dir / "ber_vs_snr_mean.png").exists()
    assert (plots_dir / "evm_vs_snr_mean.png").exists()
