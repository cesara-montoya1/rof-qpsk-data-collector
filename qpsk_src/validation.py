import re
from pathlib import Path
from typing import List

DISTANCE_PATTERN = re.compile(r"^(\d+p?\d*)km$", re.IGNORECASE)


def validate_dataset_dir(dataset_dir: Path | str) -> List[Path]:
    """
    Validates that dataset_dir exists and contains at least one valid distance folder (e.g., 0km, 20km, 0p5km).
    Returns a sorted list of matching distance folder Paths.
    
    Raises:
        FileNotFoundError: If dataset_dir does not exist.
        ValueError: If dataset_dir is not a directory or contains no valid distance subdirectories.
    """
    path = Path(dataset_dir)
    if not path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Dataset path is not a directory: {path}")

    matching_folders = [
        p for p in path.iterdir()
        if p.is_dir() and DISTANCE_PATTERN.match(p.name)
    ]

    if not matching_folders:
        raise ValueError(
            f"No valid distance folders (e.g. 0km, 2km, 20km) found directly inside: {path}"
        )

    return sorted(matching_folders, key=lambda p: p.name)
