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
    sync_signals_complex,
)
from qpsk_src.evm import compute_evm


def main():
    parser = argparse.ArgumentParser(
        description="QPSK BER and Error Distribution Analysis."
    )
    parser.add_argument("--tx", required=True, help="Path to TX bits file (.txt)")
    parser.add_argument(
        "--rx", required=True, help="Path to RX samples file (.complex64)"
    )
    parser.add_argument(
        "--skip-initial-symbols",
        type=int,
        default=0,
        help="Number of initial symbols to skip for warm-up (default: 0).",
    )
    parser.add_argument(
        "--best-window-symbols",
        type=int,
        default=None,
        help="Block size (M) for best-case EVM window (default: None).",
    )
    parser.add_argument(
        "--output",
        default="qpsk_analysis_report.png",
        help="Path to save analysis report image.",
    )
    args = parser.parse_args()

    # 1. Load Data
    with open(args.tx, "r") as f:
        bit_str = f.read().replace("\n", "").strip()
    tx_bits = np.array([int(b) for b in bit_str if b in ("0", "1")])

    tx_symbols = modulate_qpsk(tx_bits)
    rx_signal = np.fromfile(args.rx, dtype=np.complex64)

    # 2. Phase Rotation Analysis
    rotations = [1, 1j, -1, -1j]
    labels = ["0°", "90°", "180°", "270°"]
    results = []
    best_ber = 1.0
    best_idx = 0
    bit_skip = 2 * max(0, int(args.skip_initial_symbols))

    print(f"{'Rotation':<10} | {'Delay':<10} | {'BER':<10}")
    print("-" * 35)

    for i, rot in enumerate(rotations):
        tx_s, rx_s, delay = sync_signals_complex(tx_symbols, rx_signal * rot)
        b_ref = demodulate_qpsk(tx_s)
        b_rx = demodulate_qpsk(rx_s)

        n = min(len(b_ref), len(b_rx))
        if n <= bit_skip:
            ber = 1.0
            cum_errors = np.zeros(1)
        else:
            sync_tx_bits = b_ref[bit_skip:n]
            sync_rx_bits = b_rx[bit_skip:n]
            error_mask = (sync_tx_bits != sync_rx_bits).astype(int)
            cum_errors = np.cumsum(error_mask)
            ber = float(cum_errors[-1] / len(sync_rx_bits))

        results.append(
            {
                "label": labels[i],
                "ber": ber,
                "cum_err": cum_errors,
                "rx_sync": rx_s,
                "tx_sync": tx_s,
            }
        )

        if ber < best_ber:
            best_ber = ber
            best_idx = i
        print(f"{labels[i]:<10} | {delay:<10} | {ber:.6f}")

    best_res = results[best_idx]
    best_rx = best_res["rx_sync"]
    best_tx = best_res["tx_sync"]

    # Compute EVM for best rotation
    evm_metrics = compute_evm(
        best_rx,
        best_tx,
        skip_initial_symbols=args.skip_initial_symbols,
        best_window_symbols=args.best_window_symbols,
    )

    print("-" * 35)
    print(f"Optimal Rotation: {labels[best_idx]}")
    print(f"BER: {best_ber:.6e}")
    print(f"EVM RMS: {evm_metrics['evm_rms_pct']:.2f}% ({evm_metrics['evm_db']:.2f} dB)")
    if evm_metrics.get("evm_best_rms_pct") is not None:
        print(
            f"Best-Window EVM ({args.best_window_symbols} syms): "
            f"{evm_metrics['evm_best_rms_pct']:.2f}% ({evm_metrics['evm_best_db']:.2f} dB)"
        )
        print(f"Noise Burst Impact (ΔEVM): {evm_metrics['evm_delta_rms_pct']:+.2f}%")

    # 3. Enhanced Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Error rate bar chart / summary
    rot_names = [r["label"] for r in results]
    bers = [r["ber"] for r in results]
    colors = ["#2ca02c" if i == best_idx else "#7f7f7f" for i in range(len(results))]
    ax1.bar(rot_names, bers, color=colors, alpha=0.8)
    ax1.set_title("BER across 90° Rotations")
    ax1.set_ylabel("Bit Error Rate (BER)")
    ax1.grid(True, alpha=0.3, axis="y")

    # Panel 2: Cumulative Errors
    for i, res in enumerate(results):
        alpha = 1.0 if i == best_idx else 0.3
        ax2.plot(
            res["cum_err"], label=f"{res['label']} (BER: {res['ber']:.2e})", alpha=alpha
        )
    ax2.set_title("Error Accumulation (Burst vs Random)")
    ax2.set_xlabel("Evaluated Bit Index")
    ax2.set_ylabel("Total Bit Errors")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Constellation
    start_s = max(0, int(args.skip_initial_symbols))
    rx_disp = best_rx[start_s:]
    scale = np.sqrt(np.mean(np.abs(best_tx[start_s:]) ** 2) / (np.mean(np.abs(rx_disp) ** 2) + 1e-12))
    rx_disp_norm = rx_disp * scale

    ax3.scatter(rx_disp_norm.real, rx_disp_norm.imag, s=2, alpha=0.3, label="RX Samples")
    ax3.scatter(
        QPSK_CONSTELLATION.real,
        QPSK_CONSTELLATION.imag,
        marker="x",
        color="red",
        s=80,
        linewidth=2,
        label="Ideal",
        zorder=5,
    )
    evm_str = f"EVM: {evm_metrics['evm_rms_pct']:.2f}% ({evm_metrics['evm_db']:.2f} dB)"
    ax3.set_title(f"Constellation ({labels[best_idx]})\n{evm_str}")
    ax3.axis("equal")
    ax3.set_xlim(-1.8, 1.8)
    ax3.set_ylim(-1.8, 1.8)
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output)
    plt.close(fig)
    print(f"\nReport saved as: {args.output}")


if __name__ == "__main__":
    main()
