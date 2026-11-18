"""Output file naming and collision avoidance for 16-QAM RoF captures."""

import logging
import os
import shutil

logger = logging.getLogger(__name__)


def get_unique_filepath(out_dir, params, avg_snr):
    """
    Generates a unique filename including OSNR, Frequency, and Mbps for 16-QAM (4 bits/sym).
    Mbps is calculated from max_sample_rate, samp_rate_div, and samp_sym.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    osnr_p = f"{params['osnr']:.2f}".replace(".", "p")
    freq_mhz = int(params["freq"] / 1e6)

    # Mbps Calculation: (max_samp_rate / div / samp_sym) * 4 bits/symbol
    max_samp_rate = params.get("max_sample_rate", 15e6) / 1e6
    mbps_val = (max_samp_rate / (params["samp_rate_div"] * params["samp_sym"])) * 4
    mbps_p = f"{mbps_val:.2f}".replace(".", "p")

    adjustments = [0.0, 0.01, -0.01, 0.02, -0.02, 0.03, -0.03]

    for adj in adjustments:
        current_snr = avg_snr + adj
        snr_p = f"{current_snr:.2f}".replace(".", "p")

        filename = (
            f"rof_{params['power']}dBm_{params['dist']}km_"
            f"osnr{osnr_p}dB_{freq_mhz}mhz_{mbps_p}mbps_snr{snr_p}dB.complex64"
        )
        full_path = os.path.join(out_dir, filename)

        if not os.path.exists(full_path):
            if adj != 0:
                logger.warning(f"Collision avoidance: SNR adjusted by {adj:+.2f} dB")
            return full_path

        logger.warning(f"File exists: {filename}. Retrying with adjustment...")

    return None


def move_output_file(src, dest):
    """Moves capture file from temporary path to final destination."""
    if os.path.exists(src):
        shutil.move(src, dest)
        logger.info(f"File successfully moved to: {dest}")
        return True
    logger.error(f"Source file not found: {src}")
    return False
