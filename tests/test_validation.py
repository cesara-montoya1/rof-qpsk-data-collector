import pytest
from pathlib import Path
import sys

# Ensure optnet_ml directory is in sys.path so qpsk_src can be imported
optnet_ml_dir = Path(__file__).resolve().parent.parent
if str(optnet_ml_dir) not in sys.path:
    sys.path.insert(0, str(optnet_ml_dir))

from qpsk_src.validation import validate_dataset_dir


def test_validate_dataset_dir_valid(tmp_path):
    d0 = tmp_path / "0km"
    d20 = tmp_path / "20km"
    d0p5 = tmp_path / "0p5km"
    d0.mkdir()
    d20.mkdir()
    d0p5.mkdir()

    distance_folders = validate_dataset_dir(tmp_path)
    assert len(distance_folders) == 3
    assert all(isinstance(p, Path) for p in distance_folders)
    folder_names = [f.name for f in distance_folders]
    assert folder_names == ["0km", "0p5km", "20km"]


def test_validate_dataset_dir_invalid(tmp_path):
    (tmp_path / "invalid_subfolder").mkdir()
    with pytest.raises(ValueError, match="No valid distance folders"):
        validate_dataset_dir(tmp_path)


def test_validate_dataset_dir_file_as_dir(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello")
    with pytest.raises((ValueError, NotADirectoryError)):
        validate_dataset_dir(file_path)


def test_validate_dataset_dir_nonexistent(tmp_path):
    non_existent = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        validate_dataset_dir(non_existent)
