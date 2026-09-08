"""Safe storage and rotation module for processing results."""

import csv
import shutil
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union


def load_existing_results(
    csv_path: Union[Path, str]
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """
    Loads existing processing results from a CSV file.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        Tuple containing:
            - List of result dictionaries.
            - Set of processed filenames.
    """
    path = Path(csv_path)
    if not path.exists() or path.stat().st_size == 0:
        return [], set()

    results_list: List[Dict[str, Any]] = []
    processed_files: Set[str] = set()

    with open(path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_dict = dict(row)
            results_list.append(row_dict)
            if "filename" in row_dict and row_dict["filename"]:
                processed_files.add(row_dict["filename"])

    return results_list, processed_files


def save_results_safely(
    csv_path: Union[Path, str], results_list: List[Dict[str, Any]]
) -> None:
    """
    Safely saves results to a CSV file using temporary write and atomic replacement.
    Creates a backup file (.csv.bak) if target CSV already exists.

    Args:
        csv_path: Target CSV file path.
        results_list: List of result dictionaries to save.
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    bak_path = path.with_name(path.name + ".bak")
    tmp_path = path.with_name(path.name + ".tmp")

    # Step 1: Create backup if target file already exists
    if path.exists():
        shutil.copy2(path, bak_path)

    # Step 2: Write results to temporary file
    if results_list:
        fieldnames: List[str] = []
        for res in results_list:
            for k in res.keys():
                if k not in fieldnames:
                    fieldnames.append(k)

        with open(tmp_path, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_list)
    else:
        tmp_path.write_text("")

    # Step 3: Atomically replace target file with tmp file
    tmp_path.replace(path)
