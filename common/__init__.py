"""Shared modulation-agnostic utilities for OptNET ML (QPSK, 16-QAM, and future pipelines)."""

from .compressor import (
    DEFAULT_CAPTURE_MODE,
    KNOWN_CAPTURE_MODES,
    build_npz_filename,
    compress_distance_folder,
    infer_scenario_slug,
)
from .evm import compute_evm, compute_theoretical_evm_from_snr
from .generate_tx import POLYNOMIALS, generate_prbs_lfsr
from .metadata import extract_value, parse_filename_metadata
from .runner import reset_usrp_usb, run_grc_capture, run_grc_capture_with_retry
from .storage import load_existing_results, save_results_safely
from .uploader import (
    BLOCKED_WIFI_SSIDS,
    DEFAULT_GITA_HOST,
    DEFAULT_GITA_PORT,
    DEFAULT_GITA_REMOTE_ROOT,
    build_ssh_transport_args,
    check_gita_network_access,
    detect_active_wifi_ssid,
    has_active_ethernet,
    infer_remote_relative_path,
    resolve_ssh_user,
    upload_dataset_to_gita,
    validate_dataset_upload_path,
)
from .validation import DISTANCE_PATTERN, validate_dataset_dir

__all__ = [
    "BLOCKED_WIFI_SSIDS",
    "DEFAULT_CAPTURE_MODE",
    "DEFAULT_GITA_HOST",
    "DEFAULT_GITA_PORT",
    "DEFAULT_GITA_REMOTE_ROOT",
    "DISTANCE_PATTERN",
    "KNOWN_CAPTURE_MODES",
    "POLYNOMIALS",
    "build_npz_filename",
    "build_ssh_transport_args",
    "check_gita_network_access",
    "compress_distance_folder",
    "compute_evm",
    "compute_theoretical_evm_from_snr",
    "detect_active_wifi_ssid",
    "extract_value",
    "generate_prbs_lfsr",
    "has_active_ethernet",
    "infer_remote_relative_path",
    "infer_scenario_slug",
    "load_existing_results",
    "parse_filename_metadata",
    "reset_usrp_usb",
    "resolve_ssh_user",
    "run_grc_capture",
    "run_grc_capture_with_retry",
    "save_results_safely",
    "upload_dataset_to_gita",
    "validate_dataset_dir",
    "validate_dataset_upload_path",
]
