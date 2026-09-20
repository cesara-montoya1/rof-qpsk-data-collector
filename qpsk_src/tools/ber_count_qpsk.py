import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

# Allow imports from project root
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from qpsk_src.demodulator import (
    QPSK_CONSTELLATION,
    demodulate_qpsk,
    modulate_qpsk,
    sync_signals,
)



def main():
    parser = argparse.ArgumentParser(
        description="QPSK BER and Error Distribution Analysis."
    )
    parser.add_argument("--tx", required=True, help="Path to TX bits file (.txt)")
    parser.add_argument(
        "--rx", required=True, help="Path to RX samples file (.complex64)"
    )
    args = parser.parse_args()

    # 1. Load Data
    with open(args.tx, "r") as f:
        bit_str = f.read().replace("\n", "").strip()
    tx_bits = np.array([int(b) for b in bit_str])

    tx_symbols = modulate_qpsk(tx_bits)
    rx_signal = np.fromfile(args.rx, dtype=np.complex64)

    # 2. Phase Rotation Analysis
    rotations = [1, 1j, -1, -1j]
    labels = ["0°", "90°", "180°", "270°"]
    results = []
    best_ber = 1.0
    best_idx = 0

    print(f"{'Rotation':<10} | {'Delay':<10} | {'BER':<10}")
    print("-" * 35)

    for i, rot in enumerate(rotations):
        tx_s, rx_s, delay = sync_signals(tx_symbols, rx_signal * rot)
        rx_bits = demodulate_qpsk(rx_s)


        # Calculate errors and cumulative sum
        sync_tx_bits = tx_bits[: len(rx_bits)]
        error_mask = (sync_tx_bits != rx_bits).astype(int)
        cum_errors = np.cumsum(error_mask)

        ber = cum_errors[-1] / len(rx_bits) if len(rx_bits) > 0 else 1.0

        results.append(
            {
                "label": labels[i],
                "corr": corr_mag,
                "ber": ber,
                "cum_err": cum_errors,
                "rx_sync": rx_s,
            }
        )

        if ber < best_ber:
            best_ber = ber
            best_idx = i
        print(f"{labels[i]:<10} | {delay:<10} | {ber:.6f}")

    # 3. Enhanced Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Correlation
    for i, res in enumerate(results):
        alpha = 1.0 if i == best_idx else 0.3
        ax1.plot(res["corr"], label=res["label"], alpha=alpha)
    ax1.set_title("Synchronization (Correlation)")
    ax1.set_xlabel("Lag Index")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Cumulative Errors
    for i, res in enumerate(results):
        alpha = 1.0 if i == best_idx else 0.3
        ax2.plot(
            res["cum_err"], label=f"{res['label']} (BER: {res['ber']:.2e})", alpha=alpha
        )
    ax2.set_title("Error Accumulation (Burst vs Random)")
    ax2.set_xlabel("Bit Index")
    ax2.set_ylabel("Total Errors")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Constellation
    best_rx = results[best_idx]["rx_sync"]
    ax3.scatter(best_rx.real, best_rx.imag, s=1, alpha=0.2, label="RX Samples")
    ax3.scatter(
        QPSK_CONSTELLATION.real,
        QPSK_CONSTELLATION.imag,
        marker="x",
        color="red",
        s=40,
        label="Ideal",
    )
    ax3.set_title(f"Constellation ({labels[best_idx]} Rotation)")
    ax3.axis("equal")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("qpsk_analysis_report.png")
    print("\nReport saved as: qpsk_analysis_report.png")


if __name__ == "__main__":
    main()
