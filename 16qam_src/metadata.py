"""Backward-compatible re-export shim for shared metadata parser."""

from common.metadata import extract_value, parse_filename_metadata

__all__ = ["extract_value", "parse_filename_metadata"]
