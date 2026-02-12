import argparse
import logging
import numpy as np
import sys
import os
from experiment_runner import run_grc_capture
from file_manager import get_unique_filepath, move_output_file

# Configuration for Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("../logs/experiment.log"),
    ],
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RoF Automated Experiment Orchestrator"
    )

    # Metadata
    parser.add_argument("--osnr", type=float, required=True, help="OSNR in dB")
    parser.add_argument(
        "--distance", type=int, required=True, help="Fiber distance in km"
    )
    parser.add_argument(
        "--power", type=int, required=True, help="Optical launch power in dBm"
    )
    parser.add_argument(
        "--samples", type=int, default=30, help="Samples for SNR averaging"
    )

    # Logic Flags
    parser.add_argument(
        "--move",
        type=str,
        default="",
        help="Destination directory for the renamed file",
    )

    # GRC / Hardware Parameters
    # We take MHz from CLI, but we will send Hz to GRC
    parser.add_argument(
        "--freq", type=float, required=True, help="Center frequency in MHz"
    )
    parser.add_argument(
        "--samp-rate-div", type=int, default=1, help="Sampling rate decimation factor"
    )
    parser.add_argument("--samp-sym", type=int, default=16, help="Samples per symbol")
    parser.add_argument(
        "--file-rx",
        type=str,
        default="../data/qpsk.complex64",
        help="Temporary capture file",
    )
    parser.add_argument(
        "--zmq-addr", type=str, default="tcp://0.0.0.0:18305", help="ZMQ source address"
    )

    args = parser.parse_args()

    # 1. Prepare GRC Command
    grc_path = os.path.join("..", "grc", "qpsk_rx.py")

    # Convert MHz to Hz for the SDR/GRC block
    freq_hz = args.freq * 1e6

    cmd = [
        "python3",
        grc_path,
        "--freq",
        str(freq_hz),
        "--file-rx",
        args.file_rx,  # Unified name
        "--samp-rate-div",
        str(args.samp_rate_div),
        "--samp-sym",
        str(args.samp_sym),
        "--zmq-addr",
        args.zmq_addr,
    ]

    logger.info(f"Starting Experiment | Freq: {args.freq} MHz | P: {args.power} dBm")

    # 2. Execute Capture
    snr_samples = run_grc_capture(cmd, args.samples)

    if snr_samples:
        avg_snr = np.mean(snr_samples)
        logger.info(f"Capture Finished. Avg SNR: {avg_snr:.2f} dB")

        # 3. File Management (Triggered if --move is provided)
        if args.move:
            params = {
                "osnr": args.osnr,
                "dist": args.distance,
                "power": args.power,
                "freq": freq_hz,
                "samp_rate_div": args.samp_rate_div,
                "samp_sym": args.samp_sym,
            }
            final_path = get_unique_filepath(args.move, params, avg_snr)

            if final_path:
                # We move the file from its temporary location to the final named path
                move_output_file(args.file_rx, final_path)
        else:
            logger.info(f"Move not requested. Raw file remains at: {args.file_rx}")
    else:
        logger.error("Capture failed: No SNR samples collected.")
