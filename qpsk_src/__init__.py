"""QPSK Signal Processing and Demodulation Pipeline Package."""

from .validation import validate_dataset_dir
from .metadata import extract_value, parse_filename_metadata
from .compressor import compress_distance_folder
from .storage import load_existing_results, save_results_safely
from .demodulator import (
    QPSK_CONSTELLATION,
    modulate_qpsk,
    demodulate_qpsk,
    sync_signals,
    process_signal,
)
from .evm import compute_evm, compute_theoretical_evm_from_snr
from .pipeline import load_tx_reference, run_pipeline
from .plotter import generate_all_plots, generate_ber_plots, generate_evm_plots

__all__ = [
    "validate_dataset_dir",
    "extract_value",
    "parse_filename_metadata",
    "compress_distance_folder",
    "load_existing_results",
    "save_results_safely",
    "QPSK_CONSTELLATION",
    "modulate_qpsk",
    "demodulate_qpsk",
    "sync_signals",
    "process_signal",
    "compute_evm",
    "compute_theoretical_evm_from_snr",
    "load_tx_reference",
    "run_pipeline",
    "generate_ber_plots",
    "generate_evm_plots",
    "generate_all_plots",
]
