import subprocess
import re
import time
import os
import fcntl
import logging

logger = logging.getLogger(__name__)


def run_grc_capture(command, num_samples, inactivity_timeout=0.5):
    """
    Executes the GRC script, logs all terminal output, and collects SNR samples.
    """
    snr_buffer = []
    # Regex to capture the SNR value
    snr_pattern = re.compile(r"message_debug :info: Message: ([-+]?\d*\.\d+|\d+)")

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        text=True,
        bufsize=1,
    )

    # Set non-blocking mode for stdout
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
                        # TEE: Log the raw output so it appears in terminal and file
                        logger.info(f"[GRC] {clean_line}")
                    # Reset timer on any output
                    last_output_time = time.time()

                    # Parse for SNR
                    match = snr_pattern.search(line)
                    if match:
                        val = float(match.group(1))
                        snr_buffer.append(val)
                        if not first_sample_received:
                            logger.info(
                                "First SNR sample captured. Heartbeat monitor active."
                            )
                            first_sample_received = True

                else:
                    # Timeout logic
                    if first_sample_received:
                        if (time.time() - last_output_time) > inactivity_timeout:
                            logger.warning(
                                f"Inactivity timeout ({inactivity_timeout}s) reached."
                            )
                            break
                    time.sleep(0.01)

            except (IOError, TypeError):
                if (
                    first_sample_received
                    and (time.time() - last_output_time) > inactivity_timeout
                ):
                    break
                time.sleep(0.01)
    finally:
        if process.poll() is None:
            process.terminate()
            logger.info("GRC process terminated.")

    return snr_buffer
