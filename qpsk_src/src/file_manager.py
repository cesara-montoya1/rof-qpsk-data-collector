import os
import shutil
import logging

logger = logging.getLogger(__name__)


def get_unique_filepath(out_dir, params, avg_snr):
    """
    Generates a unique filename including OSNR, Frequency, and Mbps.
    Mbps is calculated from samp_rate_div and samp_sym.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 1. Basic formatting
    osnr_p = f"{params['osnr']:.2f}".replace(".", "p")
    freq_mhz = int(params["freq"] / 1e6)

    # 2. Mbps Calculation: (15MHz / div / samp_sym) * 2 bits
    max_samp_rate = 15.0
    mbps_val = (max_samp_rate / (params["samp_rate_div"] * params["samp_sym"])) * 2
    mbps_p = f"{mbps_val:.2f}".replace(".", "p")

    adjustments = [0.0, 0.01, -0.01, 0.02, -0.02, 0.03, -0.03]

    for adj in adjustments:
        current_snr = avg_snr + adj
        snr_p = f"{current_snr:.2f}".replace(".", "p")

        # Comprehensive filename: rof_0dBm_20km_osnr25p50dB_650mhz_1p88mbps_snr22p40dB.complex64
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
    if os.path.exists(src):
        shutil.move(src, dest)
        logger.info(f"File successfully moved to: {dest}")
        return True
    logger.error(f"Source file not found: {src}")
    return False
