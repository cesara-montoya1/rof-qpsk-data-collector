"""RoF Automated Experiment Orchestrator with USRP Auto-Recovery."""

import argparse
import logging
import os
import sys
import time
import numpy as np

try:
    from .runner import run_grc_capture_with_retry
    from .file_manager import get_unique_filepath, move_output_file
except ImportError:
    from runner import run_grc_capture_with_retry
    from file_manager import get_unique_filepath, move_output_file

# Ensure logs directory exists before configuring FileHandler
os.makedirs("./logs", exist_ok=True)

# Configuration for Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("./logs/experiment.log"),
    ],
)
logger = logging.getLogger(__name__)


def capture_single(args, freq_hz: float, grc_path: str) -> bool:
    """Executes a single capture iteration with recovery and saves results."""
    cmd = [
        "python3",
        grc_path,
        "--freq",
        str(freq_hz),
        "--file-rx",
        args.file_rx,
        "--samp-rate-div",
        str(args.samp_rate_div),
        "--samp-sym",
        str(args.samp_sym),
        "--zmq-addr",
        args.zmq_addr,
    ]
    if getattr(args, "usrp_args", ""):
        cmd.extend(["--args", args.usrp_args])

    logger.info(f"Executing Capture | Freq: {args.freq} MHz | P: {args.power} dBm")

    snr_samples = run_grc_capture_with_retry(
        command=cmd,
        num_samples=args.samples,
        max_retries=args.retries,
        cooldown_seconds=args.cooldown,
    )

    if snr_samples:
        avg_snr = float(np.mean(snr_samples))
        logger.info(f"Capture Finished. Avg SNR: {avg_snr:.2f} dB")

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
                move_output_file(args.file_rx, final_path)
        else:
            logger.info(f"Move not requested. Raw file remains at: {args.file_rx}")
        return True
    else:
        logger.error("Capture failed: No SNR samples collected after retries.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="RoF Automated Experiment Orchestrator with USRP Auto-Recovery"
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
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of consecutive captures to take (default: 1)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Max retries with software USB reset on USRP device error (default: 3)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=2.0,
        help="Cooldown in seconds between captures and retries (default: 2.0)",
    )

    # GRC / Hardware Parameters
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
        default="./data/qpsk.complex64",
        help="Temporary capture file",
    )
    parser.add_argument(
        "--zmq-addr", type=str, default="tcp://0.0.0.0:18305", help="ZMQ source address"
    )
    parser.add_argument(
        "--usrp-args", type=str, default="", help="Optional device args for USRP (e.g. type=b200)"
    )

    args = parser.parse_args()

    # Locate qpsk_rx.py robustly
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "..", "grc", "qpsk_rx.py"),
        os.path.join(script_dir, "grc", "qpsk_rx.py"),
        os.path.join(".", "qpsk_src", "grc", "qpsk_rx.py"),
        os.path.join(".", "grc", "qpsk_rx.py"),
    ]
    grc_path = next((c for c in candidates if os.path.exists(c)), candidates[0])

    freq_hz = args.freq * 1e6

    successes = 0
    for i in range(1, args.repeat + 1):
        if args.repeat > 1:
            logger.info(f"--- Running capture sequence {i}/{args.repeat} ---")
        ok = capture_single(args, freq_hz, grc_path)
        if ok:
            successes += 1
        if i < args.repeat:
            logger.info(f"Waiting {args.cooldown}s cooldown before next sequence...")
            time.sleep(args.cooldown)

    logger.info(f"Finished sequence. Completed {successes}/{args.repeat} captures.")


if __name__ == "__main__":
    main()
