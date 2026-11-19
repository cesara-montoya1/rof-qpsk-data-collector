"""Shared GRC Execution Runner with Synchronous Teardown and Automatic USRP Recovery."""

from __future__ import annotations

import fcntl
import glob
import logging
import os
import re
import subprocess
import time
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Linux ioctl code for USB device reset (equivalent to unplugging and replugging)
USBDEVFS_RESET = ord("U") << 8 | 20  # 21780

# Known USRP Vendor IDs (hex strings):
# 2500 = Ettus Research (B200, B210, B200mini, etc.)
# 3923 = National Instruments (NI USRP series)
USRP_VENDOR_IDS = ("2500", "3923")


def reset_usrp_usb() -> bool:
    """
    Performs a software-driven USB port reset on any connected Ettus/NI USRP device.
    Emulates physically unplugging and replugging the USB cable.

    Returns:
        True if at least one USRP USB port was found and reset, False otherwise.
    """
    reset_performed = False

    for vendor_file in glob.glob("/sys/bus/usb/devices/*/idVendor"):
        try:
            with open(vendor_file, "r", encoding="utf-8") as vf:
                vendor = vf.read().strip().lower()

            if vendor in USRP_VENDOR_IDS:
                dev_dir = os.path.dirname(vendor_file)
                bus_file = os.path.join(dev_dir, "busnum")
                dev_file = os.path.join(dev_dir, "devnum")

                if os.path.exists(bus_file) and os.path.exists(dev_file):
                    with open(bus_file, "r", encoding="utf-8") as bf:
                        bus = int(bf.read().strip())
                    with open(dev_file, "r", encoding="utf-8") as df:
                        dev = int(df.read().strip())

                    usb_path = f"/dev/bus/usb/{bus:03d}/{dev:03d}"
                    if os.path.exists(usb_path):
                        logger.info(
                            "Found USRP USB device at %s. Initiating software port reset...",
                            usb_path,
                        )
                        with open(usb_path, "wb") as dev_fh:
                            fcntl.ioctl(dev_fh, USBDEVFS_RESET, 0)
                        logger.info("USRP USB port successfully reset by software.")
                        reset_performed = True
        except Exception as exc:
            logger.warning("Could not reset USB device at %s: %s", vendor_file, exc)

    if not reset_performed:
        logger.info("No connected USRP USB devices detected for reset.")

    return reset_performed


def run_grc_capture(
    command: List[str], num_samples: int, inactivity_timeout: float = 0.5
) -> Tuple[List[float], bool]:
    """
    Executes the GRC script, logs all terminal output, and collects SNR samples.
    Performs synchronous process termination to ensure USB/UHD locks are freed.

    Returns:
        Tuple of (snr_buffer, device_error_detected).
    """
    snr_buffer: List[float] = []
    device_error_detected = False

    snr_pattern = re.compile(r"message_debug :info: Message: ([-+]?\d*\.\d+|\d+)")
    device_error_pattern = re.compile(
        r"(LookupError|No devices found|KeyError.*Device|RuntimeError.*UHD)",
        re.IGNORECASE,
    )

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    fd = process.stdout.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

    last_output_time = time.time()
    first_sample_received = False

    try:
        while len(snr_buffer) < num_samples:
            try:
                line = process.stdout.readline()
                if line:
                    clean_line = line.strip()
                    if clean_line:
                        logger.info("[GRC] %s", clean_line)
                    last_output_time = time.time()

                    if device_error_pattern.search(clean_line):
                        device_error_detected = True

                    match = snr_pattern.search(line)
                    if match:
                        val = float(match.group(1))
                        snr_buffer.append(val)
                        if not first_sample_received:
                            logger.info("First SNR sample captured. Heartbeat monitor active.")
                            first_sample_received = True
                else:
                    if process.poll() is not None:
                        if process.poll() != 0 and not first_sample_received:
                            device_error_detected = True
                        break

                    if first_sample_received:
                        if (time.time() - last_output_time) > inactivity_timeout:
                            logger.warning(
                                "Inactivity timeout (%ss) reached.", inactivity_timeout
                            )
                            break
                    time.sleep(0.01)

            except (IOError, TypeError):
                if first_sample_received and (time.time() - last_output_time) > inactivity_timeout:
                    break
                time.sleep(0.01)

    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                logger.warning("GRC process did not exit within 3s; sending SIGKILL...")
                process.kill()
                process.wait(timeout=2.0)
            logger.info("GRC process cleanly terminated.")

    return snr_buffer, device_error_detected


def run_grc_capture_with_retry(
    command: List[str],
    num_samples: int,
    inactivity_timeout: float = 0.5,
    max_retries: int = 3,
    cooldown_seconds: float = 2.0,
) -> List[float]:
    """Runs GRC capture with automatic retry and software USB reset on USRP device errors."""
    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            logger.info("Retrying GRC capture: attempt %d of %d...", attempt, max_retries)

        snr_buffer, device_error = run_grc_capture(
            command=command,
            num_samples=num_samples,
            inactivity_timeout=inactivity_timeout,
        )

        if snr_buffer:
            if attempt > 1:
                logger.info("Capture recovered successfully on attempt %d.", attempt)
            return snr_buffer

        if device_error and attempt < max_retries:
            logger.warning(
                "USRP device error detected on attempt %d/%d. Triggering software USB reset...",
                attempt,
                max_retries,
            )
            reset_usrp_usb()
            time.sleep(cooldown_seconds)
        elif attempt < max_retries:
            logger.warning(
                "Capture produced no samples on attempt %d/%d. Cooling down %ss...",
                attempt,
                max_retries,
                cooldown_seconds,
            )
            time.sleep(cooldown_seconds)

    logger.error("All %d capture attempts failed.", max_retries)
    return []
