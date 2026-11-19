"""Backward-compatible re-export shim for shared NPZ compressor."""

from common.compressor import (
    DEFAULT_CAPTURE_MODE,
    KNOWN_CAPTURE_MODES,
    build_npz_filename,
    compress_distance_folder,
    infer_scenario_slug,
)

__all__ = [
    "DEFAULT_CAPTURE_MODE",
    "KNOWN_CAPTURE_MODES",
    "build_npz_filename",
    "compress_distance_folder",
    "infer_scenario_slug",
]
