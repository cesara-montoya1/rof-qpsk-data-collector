import numpy as np
import matplotlib.pyplot as plt
import argparse


def sync_and_get_corr(tx_symbols, rx_signal):
    """
    Synchronizes signals by cross-correlating I and Q components.
    Returns synchronized segments, the delay, and the correlation magnitude.
    """
    eps = 1e-9
    tx_i = (tx_symbols.real - np.mean(tx_symbols.real)) / (
        np.std(tx_symbols.real) + eps
    )
    tx_q = (tx_symbols.imag - np.mean(tx_symbols.imag)) / (
        np.std(tx_symbols.imag) + eps
    )
    rx_i = (rx_signal.real - np.mean(rx_signal.real)) / (np.std(rx_signal.real) + eps)
    rx_q = (rx_signal.imag - np.mean(rx_signal.imag)) / (np.std(rx_signal.imag) + eps)

    corr_ii = np.correlate(rx_i, tx_i, mode="full")
    corr_qq = np.correlate(rx_q, tx_q, mode="full")
    total_corr = np.abs(corr_ii + corr_qq)

    lags = np.arange(-len(tx_symbols) + 1, len(rx_signal))
    delay = lags[np.argmax(total_corr)]

    if delay < 0:
        rx_sync = rx_signal[0 : len(tx_symbols) + delay]
        tx_sync = tx_symbols[-delay : -delay + len(rx_sync)]
    else:
        rx_sync = rx_signal[delay : delay + len(tx_symbols)]
        tx_sync = tx_symbols[: len(rx_sync)]

    return tx_sync, rx_sync, delay, total_corr


def demodulate_qpsk(rx_samples, constellation):
    """
    Demodulates QPSK samples to bits using Minimum Distance decision.
    """
    indices = np.argmin(np.abs(rx_samples[:, None] - constellation[None, :]), axis=1)
    bits = np.zeros(len(indices) * 2, dtype=int)
    bits[0::2] = (indices >> 1) & 1
    bits[1::2] = indices & 1
    return bits


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

    constellation = np.array(
        [0.707 + 0.707j, -0.707 + 0.707j, -0.707 - 0.707j, 0.707 - 0.707j]
    )
    tx_sym_indices = (tx_bits[0::2] << 1) | tx_bits[1::2]
    tx_symbols = constellation[tx_sym_indices]

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
        _, rx_s, delay, corr_mag = sync_and_get_corr(tx_symbols, rx_signal * rot)
        rx_bits = demodulate_qpsk(rx_s, constellation)

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
        constellation.real,
        constellation.imag,
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
