"""Tests for CLI compress command with --upload flag across common, QPSK, and 16-QAM CLIs."""

from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest

from common.cli import main as common_main
from qpsk_src.cli import main as qpsk_main


def test_common_cli_compress_with_upload_dry_run(tmp_path: Path, capsys):
    dataset_dir = tmp_path / "rof_qpsk" / "uspl" / "test_exp"
    dist_dir = dataset_dir / "0km"
    dist_dir.mkdir(parents=True)

    raw_file = dist_dir / "signal.complex64"
    np.array([1 + 1j, -1 - 1j], dtype=np.complex64).tofile(raw_file)

    key_file = tmp_path / "id_test"
    key_file.write_text("dummy\n", encoding="utf-8")

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
                "-i",
                str(key_file),
                "--dry-run",
            ]
        )

    captured = capsys.readouterr().out
    expected_npz = dist_dir / "usrp_rx_raw_rof_qpsk_uspl_test_exp_0km.npz"
    assert expected_npz.exists()
    assert "DRY RUN" in captured
    assert "rof_qpsk/uspl/test_exp" in captured


def test_common_cli_upload_requires_identity_file(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPTNET_GITA_KEY", raising=False)
    dataset_dir = tmp_path / "exp_no_key"
    (dataset_dir / "0km").mkdir(parents=True)

    with pytest.raises(SystemExit) as exc_info:
        common_main(["upload", "-d", str(dataset_dir), "--dry-run"])
    assert exc_info.value.code == 1


def test_qpsk_cli_compress_delegates_to_common(tmp_path: Path, monkeypatch, capsys):
    dataset_dir = tmp_path / "rof_qpsk" / "dfb" / "exp_cli"
    dist_dir = dataset_dir / "0km"
    dist_dir.mkdir(parents=True)

    raw_file = dist_dir / "signal.complex64"
    np.array([1 + 1j], dtype=np.complex64).tofile(raw_file)

    key_file = tmp_path / "id_test"
    key_file.write_text("dummy\n", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        ["cli.py", "compress", "-d", str(dataset_dir), "--upload", "-i", str(key_file), "--dry-run"],
    )

    with (
        patch("common.uploader.detect_active_wifi_ssid", return_value=None),
        patch("common.uploader.resolve_ssh_user", return_value="camontoya"),
    ):
        qpsk_main()

    captured = capsys.readouterr().out
    assert "usrp_rx_gui_rof_qpsk_dfb_exp_cli_0km.npz" in captured
    assert "DRY RUN" in captured
