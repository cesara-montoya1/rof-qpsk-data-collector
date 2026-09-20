"""Unit tests for USRP runner teardown, error recovery, and auto-retry."""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# Ensure repo root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from qpsk_src.src.runner import (
    reset_usrp_usb,
    run_grc_capture,
    run_grc_capture_with_retry,
)


def test_reset_usrp_usb_safe_when_no_usrp():
    """When no USRP is connected, reset_usrp_usb should return False without raising errors."""
    with patch("glob.glob", return_value=[]):
        res = reset_usrp_usb()
        assert res is False


def test_reset_usrp_usb_triggers_ioctl_when_device_present(tmp_path):
    """When a device with vendor 2500 is found, ioctl USBDEVFS_RESET should be called."""
    mock_vendor = tmp_path / "idVendor"
    mock_vendor.write_text("2500\n")
    mock_bus = tmp_path / "busnum"
    mock_bus.write_text("1\n")
    mock_dev = tmp_path / "devnum"
    mock_dev.write_text("2\n")

    fake_usb_node = tmp_path / "fake_usb"
    fake_usb_node.write_bytes(b"")

    orig_open = open
    def side_effect_open(path, *args, **kwargs):
        if str(path).startswith("/dev/bus/usb"):
            return orig_open(fake_usb_node, *args, **kwargs)
        return orig_open(path, *args, **kwargs)

    with patch("glob.glob", return_value=[str(mock_vendor)]), \
         patch("os.path.exists", return_value=True), \
         patch("builtins.open", side_effect=side_effect_open), \
         patch("fcntl.ioctl") as mock_ioctl:

        res = reset_usrp_usb()
        assert res is True
        assert mock_ioctl.called



def test_run_grc_capture_synchronous_teardown():
    """Ensure process is completely waited on during teardown, leaving no zombie/unreleased state."""
    # Run a simple python one-liner that emits 2 SNR messages then sleeps
    cmd = [
        sys.executable,
        "-c",
        (
            "import sys, time; "
            "print('message_debug :info: Message: 15.5', flush=True); "
            "print('message_debug :info: Message: 16.0', flush=True); "
            "time.sleep(10)"
        ),
    ]
    samples, device_error = run_grc_capture(cmd, num_samples=2, inactivity_timeout=0.2)
    assert len(samples) == 2
    assert samples == [15.5, 16.0]
    assert device_error is False


def test_run_grc_capture_detects_usrp_lookup_error():
    """When GRC exits with 'No devices found', it should detect the device error."""
    cmd = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "print('RuntimeError: LookupError: KeyError: No devices found for -----> Empty Device Address', flush=True); "
            "sys.exit(1)"
        ),
    ]
    samples, device_error = run_grc_capture(cmd, num_samples=5, inactivity_timeout=0.2)
    assert len(samples) == 0
    assert device_error is True


def test_run_grc_capture_with_retry_recovers():
    """When attempt 1 encounters a device error, it resets USB and succeeds on attempt 2."""
    attempt = 0

    def mock_capture(command, num_samples, **kwargs):
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            return [], True  # Failed with device error
        return [18.2, 18.5], False  # Succeeded

    with patch("qpsk_src.src.runner.run_grc_capture", side_effect=mock_capture), \
         patch("qpsk_src.src.runner.reset_usrp_usb", return_value=True), \
         patch("time.sleep", return_value=None):
        
        samples = run_grc_capture_with_retry(
            command=["dummy"],
            num_samples=2,
            max_retries=3,
            cooldown_seconds=0.01,
        )
        assert len(samples) == 2
        assert samples == [18.2, 18.5]
        assert attempt == 2
