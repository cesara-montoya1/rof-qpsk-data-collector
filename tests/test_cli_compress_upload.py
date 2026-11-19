"""Tests for CLI compress command with --upload flag across common, QPSK, and 16-QAM CLIs."""

from pathlib import Path
from unittest.mock import patch
import numpy as np

from common.cli import main as common_main
from qpsk_src.cli import main as qpsk_main


def test_common_cli_compress_with_upload_dry_run(tmp_path: Path, capsys):
    dataset_dir = tmp_path / "rof_qpsk" / "uspl" / "test_exp"
    dist_dir = dataset_dir / "0km"
    dist_dir.mkdir(parents=True)

    raw_file = dist_dir / "signal.complex64"
    np.array([1 + 1j, -1 - 1j], dtype=np.complex64).tofile(raw_file)

    with (
        patch("common.uploader.detect_active_wifi_ssid", return_value=None),
        patch("common.uploader.resolve_ssh_user", return_value="camontoya"),
    ):
        common_main(
            [
                "compress",
                "-d",
                str(dataset_dir),
                "--capture-mode",
                "usrp_rx_raw",
                "--upload",
                "--dry-run",
            ]
        )

    captured = capsys.readouterr().out
    expected_npz = dist_dir / "usrp_rx_raw_rof_qpsk_uspl_test_exp_0km.npz"
    assert expected_npz.exists()
    assert "DRY RUN" in captured
    assert "rof_qpsk/uspl/test_exp" in captured


def test_qpsk_cli_compress_delegates_to_common(tmp_path: Path, monkeypatch, capsys):
    dataset_dir = tmp_path / "rof_qpsk" / "dfb" / "exp_cli"
    dist_dir = dataset_dir / "0km"
    dist_dir.mkdir(parents=True)

    raw_file = dist_dir / "signal.complex64"
    np.array([1 + 1j], dtype=np.complex64).tofile(raw_file)

    monkeypatch.setattr(
        "sys.argv",
        ["cli.py", "compress", "-d", str(dataset_dir), "--upload", "--dry-run"],
    )

    with (
        patch("common.uploader.detect_active_wifi_ssid", return_value=None),
        patch("common.uploader.resolve_ssh_user", return_value="camontoya"),
    ):
        qpsk_main()

    captured = capsys.readouterr().out
    assert "usrp_rx_gui_rof_qpsk_dfb_exp_cli_0km.npz" in captured
    assert "DRY RUN" in captured
