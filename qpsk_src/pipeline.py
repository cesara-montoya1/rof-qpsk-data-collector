"""Pipeline orchestrator for QPSK signal demodulation and results logging."""

from pathlib import Path
from typing import Optional
import numpy as np

from .compressor import compress_distance_folder
from .demodulator import modulate_qpsk, process_signal
from .metadata import parse_filename_metadata
from .storage import load_existing_results, save_results_safely
from .validation import validate_dataset_dir


def load_tx_reference(tx_ref_path: Path | str) -> np.ndarray:
    """
    Loads TX reference bit string from file and converts it into complex QPSK symbols.

    Args:
        tx_ref_path: Path to the reference TX file (e.g. tx.txt).

    Returns:
        1D numpy array of complex QPSK symbols.

    Raises:
        FileNotFoundError: If tx_ref_path does not exist.
    """
    path = Path(tx_ref_path)
    if not path.exists():
        raise FileNotFoundError(f"TX reference file does not exist: {path}")

    bit_str = path.read_text(encoding="utf-8").replace("\n", "").strip()
    bits = np.array([int(b) for b in bit_str if b in ("0", "1")], dtype=np.uint8)
    return modulate_qpsk(bits)


def run_pipeline(
    dataset_dir: Path | str,
    tx_ref_path: Path | str,
    output_csv_path: Optional[Path | str] = None,
) -> Path:
    """
    Runs the complete QPSK processing and demodulation pipeline on a dataset directory.

    Steps:
    1. Validates dataset_dir using validate_dataset_dir.
    2. Resolves output_csv_path (defaults to <dataset_dir>/results_<dataset_dir_name>.csv).
    3. Loads tx_ref via load_tx_reference.
    4. Reads existing results via load_existing_results.
    5. Iterates through distance folders, compresses them via compress_distance_folder,
       loads signals from NPZ, runs process_signal, extracts metadata via parse_filename_metadata,
       appends to results list, and saves via save_results_safely.

    Args:
        dataset_dir: Path to dataset directory.
        tx_ref_path: Path to reference TX bit file.
        output_csv_path: Optional path for output CSV file.

    Returns:
        Path to the output CSV file.
    """
    dataset_path = Path(dataset_dir)
    distance_folders = validate_dataset_dir(dataset_path)

    if output_csv_path is None:
        output_path = dataset_path / f"results_{dataset_path.name}.csv"
    else:
        output_path = Path(output_csv_path)

    tx_ref = load_tx_reference(tx_ref_path)
    results_list, processed_files = load_existing_results(output_path)

    for dist_folder in distance_folders:
        npz_path = compress_distance_folder(dist_folder)
        if npz_path is None or not npz_path.exists():
            continue

        with np.load(npz_path, allow_pickle=True) as npz_data:
            for filename in npz_data.files:
                if filename in processed_files:
                    continue

                rx_signal = npz_data[filename]
                sig_res = process_signal(tx_ref, rx_signal)
                meta = parse_filename_metadata(filename)

                record = {
                    "launch_power_dbm": meta["launch_power_dbm"],
                    "distance_km": meta["distance_km"],
                    "osnr_db": meta["osnr_db"],
                    "snr_db": meta["snr_db"],
                    "freq_mhz": meta["freq_mhz"],
                    "bitrate_mbps": meta["bitrate_mbps"],
                    "filename": filename,
                    "ber": sig_res["ber"],
                    "detected_delay": sig_res["detected_delay"],
                    "npz_source": str(npz_path.name),
                }
                results_list.append(record)
                processed_files.add(filename)

        save_results_safely(output_path, results_list)

    return output_path
