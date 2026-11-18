#!/bin/bash
# ==============================================================================
# OptNET ML: Optical Spectra Plotting Wrapper (Environment-Agnostic)
# ==============================================================================
#
# Usage (Local):
#   ./run_plot_spectra.sh [input_dir] [output_dir] [pump_current]
#
# Usage (GITA via srun interactive):
#   srun --partition=full-gpu ./run_plot_spectra.sh [input_dir] [output_dir] [pump_current]
# ==============================================================================

set -e
umask 002

INPUT_DIR="${1:-results/hanna_maestria/espectros}"
OUTPUT_DIR="${2:-${INPUT_DIR}/plots}"
PUMP_CURRENT="${3:-370}"

PIXI_BIN="$(which pixi 2>/dev/null || echo "$HOME/.pixi/bin/pixi")"

if ! command -v "$PIXI_BIN" &>/dev/null; then
    echo "Error: pixi binary not found. Please install pixi or add it to PATH."
    exit 1
fi

CMD=("$PIXI_BIN" run python utils/plot_spectra.py -i "$INPUT_DIR" -o "$OUTPUT_DIR")
if [ -n "$PUMP_CURRENT" ]; then
    CMD+=(--pump-current "$PUMP_CURRENT")
fi

exec "${CMD[@]}"
