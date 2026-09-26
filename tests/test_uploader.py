"""Tests for dataset uploader and path validation module."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from qpsk_src.uploader import trigger_gdrive_sync, validate_results_path


def test_validate_results_path_valid(tmp_path: Path):
    """Test that a directory inside optnet_ml/results is accepted."""
    results_dir = tmp_path / "optnet_ml" / "results" / "hanna_maestria" / "exp1"
    results_dir.mkdir(parents=True)
    resolved = validate_results_path(results_dir)
    assert resolved == results_dir.resolve()


def test_validate_results_path_outside_rejected(tmp_path: Path):
    """Test that directories outside optnet_ml/results raise ValueError."""
    external_dir = tmp_path / "Muestras_Laboratorio" / "exp1"
    external_dir.mkdir(parents=True)
    with pytest.raises(ValueError, match="optnet_ml/results"):
        validate_results_path(external_dir)


def test_validate_results_path_nonexistent(tmp_path: Path):
    """Test that non-existent directory raises FileNotFoundError."""
    missing_dir = tmp_path / "optnet_ml" / "results" / "nonexistent"
    with pytest.raises(FileNotFoundError):
        validate_results_path(missing_dir)


def test_trigger_gdrive_sync_dry_run():
    """Test that dry_run mode does not invoke external commands."""
    success, msg = trigger_gdrive_sync(dry_run=True)
    assert success is True
    assert "DRY RUN" in msg


@patch("qpsk_src.uploader.subprocess.run")
def test_trigger_gdrive_sync_systemd_success(mock_run: MagicMock):
    """Test successful systemd bisync trigger."""
    # First call: check service status (returncode 3 = inactive but loaded)
    # Second call: start service (returncode 0 = success)
    mock_run.side_effect = [
        MagicMock(returncode=3, stdout="inactive\n", stderr=""),
        MagicMock(returncode=0, stdout="", stderr=""),
    ]
    success, msg = trigger_gdrive_sync(dry_run=False)
    assert success is True
    assert "exitosa" in msg.lower() or "success" in msg.lower()


@patch("qpsk_src.uploader.subprocess.run")
def test_trigger_gdrive_sync_systemd_failure(mock_run: MagicMock):
    """Test handling of systemd trigger failure."""
    mock_run.side_effect = [
        MagicMock(returncode=3, stdout="inactive\n", stderr=""),
        MagicMock(returncode=1, stdout="", stderr="Job failed"),
    ]
    success, msg = trigger_gdrive_sync(dry_run=False)
    assert success is False
    assert "error" in msg.lower()
