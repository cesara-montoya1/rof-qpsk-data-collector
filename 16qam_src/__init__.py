"""16-QAM Signal Processing and Demodulation Pipeline Package."""

from .validation import validate_dataset_dir
from .metadata import extract_value, parse_filename_metadata
from .compressor import compress_distance_folder
from .storage import load_existing_results, save_results_safely
from .demodulator import (
    QAM16_CONSTELLATION,
    modulate_16qam,
    demodulate_16qam,
    sync_signals,
    sync_signals_complex,
    process_signal,
)
from .evm import compute_evm, compute_theoretical_evm_from_snr
from .pipeline import load_tx_reference, run_pipeline
from .plotter import (
    generate_all_plots,
    generate_ber_plots,
    generate_evm_plots,
    plot_ber_curve,
    plot_constellation,
    plot_evm_curve,
)
from .diagnostics import (
    compute_polar_evm,
    generate_dataset_diagnostics,
    generate_signal_diagnostics,
    plot_ber_vs_evm_theoretical,
    plot_constellation_density,
    plot_error_burst_profile,
    plot_error_vector_spectrum,
    plot_polar_evm,
)

__all__ = [
    "validate_dataset_dir",
    "extract_value",
    "parse_filename_metadata",
    "compress_distance_folder",
    "load_existing_results",
    "save_results_safely",
    "QAM16_CONSTELLATION",
    "modulate_16qam",
    "demodulate_16qam",
    "sync_signals",
    "sync_signals_complex",
    "process_signal",
    "compute_evm",
    "compute_theoretical_evm_from_snr",
    "load_tx_reference",
    "run_pipeline",
    "generate_ber_plots",
    "generate_evm_plots",
    "generate_all_plots",
    "plot_ber_curve",
    "plot_evm_curve",
    "plot_constellation",
    "compute_polar_evm",
    "generate_dataset_diagnostics",
    "generate_signal_diagnostics",
    "plot_ber_vs_evm_theoretical",
    "plot_constellation_density",
    "plot_error_burst_profile",
    "plot_error_vector_spectrum",
    "plot_polar_evm",
]
