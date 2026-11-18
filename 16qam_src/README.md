# 16-QAM Signal Processing & RoF Automation Package

Software suite and GNU Radio Companion (GRC) 3.10 flowgraphs for generating, capturing, demodulating, analyzing, and cataloging **16-QAM** signals in Radio-over-Fiber (RoF) physical-layer experiments using dual NI USRP-2904 radios (separate TX and RX).

---

## Architecture Overview

1. **GNU Radio Flowgraphs (`16qam_src/grc/`):**
   - [`16qam_tx.grc`](grc/16qam_tx.grc) / [`16qam_tx.py`](grc/16qam_tx.py): Interactive Qt GUI transmitter (`15 MS/s` default streaming rate, `sps=16`, live spectrum sink, optional ZMQ/channel simulation blocks).
   - [`16qam_tx_burst.grc`](grc/16qam_tx_burst.grc) / [`16qam_tx_burst.py`](grc/16qam_tx_burst.py): Headless high-rate continuous burst transmitter (`61.44 MS/s` default 1-channel rate, `sps=4`, `--gain` CLI parameter, no GUI overhead).
   - [`16qam_rx.grc`](grc/16qam_rx.grc) / [`16qam_rx.py`](grc/16qam_rx.py): Interactive Qt GUI receiver (`15 MS/s` default streaming rate) with Gardner timing recovery (`digital_symbol_sync_xx`), 15-tap CMA linear equalizer (`digital_linear_equalizer`), decision-directed carrier/phase recovery (`digital_constellation_receiver_cb`), and decision-directed SNR probe (`digital_probe_mpsk_snr_est_c`).
   - [`16qam_rx_burst_test.grc`](grc/16qam_rx_burst_test.grc) / [`16qam_rx_burst_test.py`](grc/16qam_rx_burst_test.py): Live Qt GUI alignment/diagnostics viewer at `61.44 MS/s` (`sps=4`, `1/15 s` refresh rate) with file capture disabled (`skiphead`/`head`/`file_sink` bypassed).
   - [`16qam_rx_burst.grc`](grc/16qam_rx_burst.grc) / [`16qam_rx_burst.py`](grc/16qam_rx_burst.py): Headless high-throughput burst capture (`61.44 MS/s`, `sps=4`, `--gain` parameter, auto-terminates after skipping `n_skip` warm-up symbols and saving `n_save` symbols while logging decision-directed SNR to stdout).

2. **NI USRP-2904 Data Rates (16-QAM = 4 bits/symbol):**
   - **Continuous Streaming Mode (`15 MS/s` max base rate):**
     - `sps = 16` -> `0.9375 MSym/s` -> **`3.75 Mbps`**
     - `sps = 4` -> `3.75 MSym/s` -> **`15.0 Mbps`**
     - `sps = 1` -> `15.0 MSym/s` -> **`60.0 Mbps`** (Nyquist unfiltered limit)
   - **Single-Channel Burst Mode (`61.44 MS/s` peak 1-CH rate, headless):**
     - `sps = 4` -> `15.36 MSym/s` -> **`61.44 Mbps`** raw bit rate.

3. **Offline Python DSP & Diagnostics (`16qam_src/`):**
   - `demodulator.py`: Gray-coded unit-average-power 16-QAM modulator/demodulator (`INV_SQRT10 = 1/sqrt(10)`, thresholds `0` and `+/- 2/sqrt(10)`), FFT complex cross-correlation delay sync (`sync_signals_complex`), scale-normalized 4-rotation BER search (`process_signal`), RMS/peak EVM, and best-window EVM extraction.
   - `pipeline.py`: Batch `.complex64` dataset processing into structured CSV telemetry (`osnr`, `distance`, `power`, `freq`, `ber`, `evm_rms_pct`, `evm_db`, `evm_peak_pct`, etc.). Note that `load_tx_reference` tiles the reference bitstream cleanly by multiples of 4 when `len(tx_bits) % 4 != 0` so all 4 symbol phases remain consistent across wrap-arounds.
   - `diagnostics.py`: Theoretical Gray 16-QAM AWGN BER vs. EVM curve (`0.375*erfc(x) + 0.25*erfc(3*x) - 0.125*erfc(5*x)` with `x = 1/(sqrt(10)*EVM)`), 2D density constellation heatmaps, polar EVM scatter plots, PSD, and temporal burst EVM profiles.

---

## CLI Usage

```bash
# Run offline 16-QAM demodulation + EVM analysis on a dataset directory
pixi run demodulate-16qam --dataset-dir ./dataset --output-csv ./results_16qam.csv --skip-initial-symbols 1000 --best-window-symbols 5000

# Inspect a single captured .complex64 file (constellation + 16-QAM EVM/BER metrics)
pixi run check-signal-16qam path/to/capture.complex64 --skip-initial-symbols 1000 --best-window-symbols 5000 --no-show

# Full CLI subcommands
python3 -m 16qam_src.cli --help
python3 -m 16qam_src.cli run --dataset-dir ./dataset
python3 -m 16qam_src.cli diagnostics --csv-path ./results_16qam.csv --rx path/to/capture.complex64
python3 -m 16qam_src.cli plot --csv-path ./results_16qam.csv --output-dir ./plots_16qam
python3 -m 16qam_src.cli validate --dataset-dir ./dataset

# Automated burst capture with USRP recovery
python3 -m 16qam_src.src.main --osnr 28 --distance 25 --power 0 --freq 650 \
    --samp-sym 4 --max-sample-rate 61.44 \
    --grc-script 16qam_src/grc/16qam_rx_burst.py \
    --move ./captures_16qam
```
