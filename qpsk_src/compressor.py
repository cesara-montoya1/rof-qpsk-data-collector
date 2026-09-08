"""Distance folder compressor module for QPSK signal dataset."""

from pathlib import Path
from typing import Optional
import numpy as np


def compress_distance_folder(distance_path: Path | str) -> Optional[Path]:
    """
    Scans distance directory for .complex64 files, compresses them into
    a single data_<folder_name>.npz file inside the distance directory, and returns its Path.

    If no .complex64 files are present, returns the Path to an existing .npz file
    if found, otherwise returns None.

    Args:
        distance_path: Path to the distance directory.

    Returns:
        Path to the compressed .npz file, or None if no data is available.

    Raises:
        FileNotFoundError: If distance_path does not exist.
        ValueError: If distance_path is not a directory.
    """
    path = Path(distance_path)
    if not path.exists():
        raise FileNotFoundError(f"Distance directory does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Distance path is not a directory: {path}")

    folder_name = path.name
    target_npz_path = path / f"data_{folder_name}.npz"

    complex_files = sorted(path.glob("*.complex64"))

    if complex_files:
        data_dict = {}
        for file_path in complex_files:
            signal_data = np.fromfile(file_path, dtype=np.complex64)
            data_dict[file_path.stem] = signal_data

        np.savez_compressed(target_npz_path, **data_dict)
        return target_npz_path

    if target_npz_path.exists():
        return target_npz_path

    existing_npzs = sorted(path.glob("*.npz"))
    if existing_npzs:
        return existing_npzs[0]

    return None
