"""Backward-compatible re-export shim for shared safe CSV storage."""

from common.storage import load_existing_results, save_results_safely

__all__ = ["load_existing_results", "save_results_safely"]
