"""Backward-compatible re-export shim for shared USRP GRC runner."""

import common.runner as _common_runner
from common.runner import (
    USBDEVFS_RESET,
    USRP_VENDOR_IDS,
    reset_usrp_usb,
    run_grc_capture,
)


def run_grc_capture_with_retry(*args, **kwargs):
    """Delegates to common.runner while honoring local monkeypatches on this module."""
    orig_capture = _common_runner.run_grc_capture
    orig_reset = _common_runner.reset_usrp_usb
    _common_runner.run_grc_capture = run_grc_capture
    _common_runner.reset_usrp_usb = reset_usrp_usb
    try:
        return _common_runner.run_grc_capture_with_retry(*args, **kwargs)
    finally:
        _common_runner.run_grc_capture = orig_capture
        _common_runner.reset_usrp_usb = orig_reset


__all__ = [
    "USBDEVFS_RESET",
    "USRP_VENDOR_IDS",
    "reset_usrp_usb",
    "run_grc_capture",
    "run_grc_capture_with_retry",
]
