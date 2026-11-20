"""Backward-compatible re-export shim for shared validation utilities."""

from common.validation import DISTANCE_PATTERN, validate_dataset_dir

__all__ = ["DISTANCE_PATTERN", "validate_dataset_dir"]
