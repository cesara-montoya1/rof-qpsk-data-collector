# 16-QAM Signal Processing & RoF Automation Package (`16qam_src`)

Part of **[OptNET ML](../README.md)** · Maintained by the **LAFI / SOFA Research Group**

This package provides GNU Radio Companion (GRC) 3.10 flowgraphs, high-throughput burst capture orchestration, offline FFT-synchronized 16-QAM demodulation, and physical-layer diagnostics for Radio-over-Fiber (RoF) experiments using dual NI USRP-2904 radios (separate TX and RX). Modulation-agnostic utilities live in [`common/`](../common/) and are re-exported here for backward compatibility.

---

## Architecture Overview

### 1. GNU Radio Flowgraphs (`16qam_src/grc/`)
- [`16qam_tx.grc`](grc/16qam_tx.grc) / [`16qam_tx.py`](grc/16qam_tx.py): Interactive Qt GUI transmitter (`15 MS/s` streaming rate, `sps=16`, live spectrum sink).
- [`16qam_tx_burst.grc`](grc/16qam_tx_burst.grc) / [`16qam_tx_burst.py`](grc/16qam_tx_burst.py): Headless high-rate continuous burst transmitter (`61.44 MS/s` 1-channel rate, `sps=4`, `--gain` CLI parameter).
- [`16qam_rx.grc`](grc/16qam_rx.grc) / [`16qam_rx.py`](grc/16qam_rx.py): Interactive Qt GUI receiver (`15 MS/s`) with Gardner symbol sync, 15-tap CMA linear equalizer, decision-directed carrier/phase recovery, and SNR probe.
- [`16qam_rx_burst_test.grc`](grc/16qam_rx_burst_test.grc) / [`16qam_rx_burst_test.py`](grc/16qam_rx_burst_test.py): Live Qt GUI alignment/diagnostics viewer at `61.44 MS/s` (`sps=4`, file sink bypassed).
- [`16qam_rx_burst.grc`](grc/16qam_rx_burst.grc) / [`16qam_rx_burst.py`](grc/16qam_rx_burst.py): Headless burst capture (`61.44 MS/s`, `sps=4`, auto-terminates after skipping `n_skip` warm-up symbols and saving `n_save` symbols).
- [`16qam_rx_burst_raw.grc`](grc/16qam_rx_burst_raw.grc) / [`16qam_rx_burst_raw.py`](grc/16qam_rx_burst_raw.py): Raw IQ burst capture for 100% offline DSP (`--capture-mode usrp_rx_raw`).

### 2. NI USRP-2904 Data Rates (16-QAM = 4 bits/symbol)
- **Continuous Streaming (`15 MS/s` max base rate):**
  - `sps = 16` → `0.9375 MSym/s` → **`3.75 Mbps`**
  - `sps = 4` → `3.75 MSym/s` → **`15.0 Mbps`**
  - `sps = 1` → `15.0 MSym/s` → **`60.0 Mbps`** (unfiltered Nyquist limit)
- **Single-Channel Headless Burst (`61.44 MS/s` peak 1-CH rate):**
  - `sps = 4` → `15.36 MSym/s` → **`61.44 Mbps`** raw bit rate

### 3. Offline Python DSP & Diagnostics (`16qam_src/`)
- `demodulator.py`: Gray-coded unit-average-power 16-QAM (`1/sqrt(10)` grid, thresholds at `0` and `±2/sqrt(10)`), FFT complex cross-correlation delay sync, scale-normalized 4-rotation BER search, and RMS/peak/best-window EVM.
- `pipeline.py`: Batch dataset processing into structured CSV telemetry; tiles reference bitstreams cleanly by multiples of 4 when `len(tx_bits) % 4 != 0`.
- `diagnostics.py`: Theoretical Gray 16-QAM AWGN BER vs. EVM bound, 2D density heatmaps, polar EVM breakdown, error PSD, and temporal burst profiles.

---

## CLI Usage

```bash
# Offline 16-QAM demodulation + EVM analysis on a dataset directory
pixi run demodulate-16qam \
    --dataset-dir /export2/camontoya/sofa_datasets/rof_16qam/dfb/<experiment> \
    --skip-initial-symbols 1000 \
    --best-window-symbols 5000

# Quick check on a single .complex64 capture
pixi run check-signal-16qam path/to/capture.complex64 \
    --skip-initial-symbols 1000 \
    --best-window-symbols 5000 \
    --no-show

# Diagnostics and plotting
pixi run python3 -m 16qam_src.cli diagnostics --csv-path ./results_16qam.csv --rx path/to/capture.complex64
pixi run python3 -m 16qam_src.cli plot --csv-path ./results_16qam.csv --output-dir ./plots_16qam

# Automated headless burst capture with USRP USB recovery
python3 -m 16qam_src.src.main \
    --osnr 28 --distance 25 --power 0 --freq 650 \
    --samp-sym 4 --max-sample-rate 61.44 \
    --grc-script 16qam_src/grc/16qam_rx_burst.py \
    --move ./rof_16qam/dfb/my_experiment/25km

# Compress (raw mode) and upload to GITA with your personal SSH key
pixi run compress-and-upload \
    -d ./rof_16qam/dfb/my_experiment \
    --capture-mode usrp_rx_raw \
    -i ~/.ssh/id_ed25519_gita
```

For dataset naming standards, SSH setup, and GITA upload troubleshooting, see [`docs/gita_datasets_workflow.md`](../docs/gita_datasets_workflow.md).
