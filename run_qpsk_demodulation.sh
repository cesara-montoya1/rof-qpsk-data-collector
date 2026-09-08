#!/bin/bash
# ==============================================================================
# OptNET ML: QPSK Signal Demodulation Wrapper (Environment-Agnostic)
# ==============================================================================
#
# Usage (Local):
#   ./run_qpsk_demodulation.sh <dataset_directory> [tx_ref_file]
#
# Usage (GITA via srun interactive):
#   srun --partition=full-gpu ./run_qpsk_demodulation.sh <dataset_directory> [tx_ref_file]
#
# Usage (GITA via sbatch background):
#   sbatch --partition=full-gpu ./run_qpsk_demodulation.sh <dataset_directory> [tx_ref_file]
# ==============================================================================

set -e
umask 002

DATASET_DIR="$1"
TX_REF="${2:-qpsk_src/data/tx.txt}"

if [ -z "$DATASET_DIR" ]; then
    echo "Error: Missing dataset directory."
    echo "Usage: $0 <dataset_directory> [tx_ref_file]"
    exit 1
fi

PIXI_BIN="$(which pixi 2>/dev/null || echo "$HOME/.pixi/bin/pixi")"

if ! command -v "$PIXI_BIN" &>/dev/null; then
    echo "Error: pixi binary not found. Please install pixi or add it to PATH."
    exit 1
fi

exec "$PIXI_BIN" run demodulate-qpsk --dataset-dir "$DATASET_DIR" --tx-ref "$TX_REF"
