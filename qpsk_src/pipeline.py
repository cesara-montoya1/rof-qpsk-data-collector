"""Pipeline orchestrator for QPSK signal demodulation and results logging."""

from pathlib import Path
from typing import Any, Dict, Optional
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
    results_list, _ = load_existing_results(output_path)
    # Index existing records by filename for in-place updating and fast lookup
    existing_records_by_name: dict[str, dict[str, Any]] = {
        r["filename"]: r for r in results_list if "filename" in r and r["filename"]
    }

    def is_fully_processed(rec: Optional[dict[str, Any]]) -> bool:
        if not rec:
            return False
        ber_val = rec.get("ber")
        evm_val = rec.get("evm_rms_pct")
        if ber_val is None or str(ber_val).strip().lower() in ("", "nan", "none"):
            return False
        if evm_val is None or str(evm_val).strip().lower() in ("", "nan", "none"):
            return False
        return True

    for dist_folder in distance_folders:
        npz_path = compress_distance_folder(dist_folder)
        if npz_path is None or not npz_path.exists():
            continue

        with np.load(npz_path, allow_pickle=True) as npz_data:
            for filename in npz_data.files:
                existing_rec = existing_records_by_name.get(filename)

                # If the record is already fully processed (both BER and EVM present), skip it
                if is_fully_processed(existing_rec):
                    continue

                rx_signal = npz_data[filename]
                sig_res = process_signal(tx_ref, rx_signal)

                if existing_rec is not None:
                    # Update existing record in-place with EVM metrics (and any missing BER/delay)
                    existing_rec["evm_rms_pct"] = sig_res["evm_rms_pct"]
                    existing_rec["evm_db"] = sig_res["evm_db"]
                    existing_rec["evm_peak_pct"] = sig_res["evm_peak_pct"]
                    if existing_rec.get("ber") in (None, "", "nan", "None"):
                        existing_rec["ber"] = sig_res["ber"]
                    if existing_rec.get("detected_delay") in (None, "", "nan", "None"):
                        existing_rec["detected_delay"] = sig_res["detected_delay"]
                else:
                    # Brand new record: extract metadata and append
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
                        "evm_rms_pct": sig_res["evm_rms_pct"],
                        "evm_db": sig_res["evm_db"],
                        "evm_peak_pct": sig_res["evm_peak_pct"],
                        "npz_source": str(npz_path.name),
                    }
                    results_list.append(record)
                    existing_records_by_name[filename] = record

        save_results_safely(output_path, results_list)

    return output_path
