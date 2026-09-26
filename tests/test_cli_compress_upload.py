"""Tests for CLI compress command with --upload flag."""

from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest

from qpsk_src.cli import main


def test_cli_compress_with_upload_dry_run(tmp_path: Path, monkeypatch, capsys):
    """Test CLI compress with --upload and --dry-run."""
    # Create fake valid results directory inside optnet_ml/results
    dataset_dir = tmp_path / "optnet_ml" / "results" / "test_exp"
    dist_dir = dataset_dir / "0km"
    dist_dir.mkdir(parents=True)

    # Create dummy complex64 file
    raw_file = dist_dir / "signal.complex64"
    data = np.array([1 + 1j, -1 - 1j], dtype=np.complex64)
    data.tofile(raw_file)

    test_args = [
        "cli.py",
        "compress",
        "-d",
        str(dataset_dir),
        "--upload",
        "--dry-run",
    ]
    monkeypatch.setattr("sys.argv", test_args)

    main()

    captured = capsys.readouterr().out
    assert "0km" in captured
    assert "DRY RUN" in captured


def test_cli_compress_with_upload_outside_rejected(tmp_path: Path, monkeypatch):
    """Test CLI compress rejects paths outside optnet_ml/results when --upload is set."""
    external_dir = tmp_path / "external_exp"
    dist_dir = external_dir / "0km"
    dist_dir.mkdir(parents=True)

    # Create dummy file so compress has something to work with
    raw_file = dist_dir / "signal.complex64"
    data = np.array([1 + 1j], dtype=np.complex64)
    data.tofile(raw_file)

    test_args = [
        "cli.py",
        "compress",
        "-d",
        str(external_dir),
        "--upload",
    ]
    monkeypatch.setattr("sys.argv", test_args)

    with pytest.raises((ValueError, SystemExit)):
        main()
