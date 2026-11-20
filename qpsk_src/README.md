# QPSK Signal Processing & RoF Automation Package (`qpsk_src`)

Part of **[OptNET ML](../README.md)** · Maintained by the **LAFI / SOFA Research Group**

This package provides QPSK-specific GNU Radio Companion (GRC) flowgraphs, live USRP capture orchestration, offline FFT-synchronized demodulation, BER/EVM plotting, and channel diagnostics. Modulation-agnostic utilities (`compressor`, `uploader`, `validation`, `metadata`, `storage`, `evm`, `runner`, `generate_tx`) live in [`common/`](../common/) and are re-exported here for backward compatibility.

---

## Package Layout

```text
qpsk_src/
├── cli.py              # QPSK CLI entrypoint (run, plot, diagnostics, constellation, compress, validate)
├── demodulator.py      # Gray QPSK constellation, FFT delay sync, 4-rotation BER search
├── pipeline.py         # Batch .npz/.complex64 demodulation with incremental CSV backfill
├── plotter.py          # BER/EVM vs. SNR/OSNR curves and constellation plots
├── diagnostics.py      # Polar EVM, error PSD, 2D density, burst profile, AWGN bound
├── data/tx.txt         # Reference PRBS transmission bitstream
├── grc/                # GNU Radio Companion flowgraphs (.grc) and compiled .py scripts
├── src/                # Live USRP capture orchestrator (main.py, file_manager.py)
└── tools/              # Standalone single-file analyzer (ber_count_qpsk.py)
```

---

## Quick Usage

```bash
# Generate a PRBS17 transmission reference bitstream
python3 -m common.generate_tx --order 17 --output qpsk_src/data/tx.txt

# Automated capture (10 repetitions with USRP USB recovery)
python3 -m qpsk_src.src.main \
    --osnr 32.5 --distance 20 --power 0 --freq 650 \
    --repeat 10 --cooldown 2.0 --retries 3 \
    --move ./rof_qpsk/dfb/my_experiment/20km

# Quick single-file constellation + BER/EVM check
pixi run check-signal path/to/capture.complex64

# Batch demodulation and plotting
pixi run demodulate-qpsk -d /export2/camontoya/sofa_datasets/rof_qpsk/uspl/<experiment>
pixi run plot-results -d /export2/camontoya/sofa_datasets/rof_qpsk/uspl/<experiment>

# Standalone exploratory report on a single capture
python3 qpsk_src/tools/ber_count_qpsk.py \
    --tx qpsk_src/data/tx.txt \
    --rx path/to/capture.complex64 \
    --skip-initial-symbols 500 \
    --best-window-symbols 5000 \
    --output /tmp/qpsk_report.png
```

---

## Capture Filename Convention

`rof_{Power}dBm_{Dist}km_osnr{OSNR}dB_{Freq}mhz_{Mbps}mbps_snr{MeasuredSNR}dB.complex64`

If a capture produces an identical measured SNR to an existing file in the same directory, `file_manager.py` applies a small collision-avoidance suffix so no measurement is overwritten.

---

## Capture Parameters (`qpsk_src.src.main`)

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--freq` | **Required** | RF carrier frequency in MHz (e.g., `650`, `900`, `1200`). |
| `--osnr` | `0.0` | Nominal OSNR setpoint in dB. |
| `--distance` | `0.0` | Optical fiber length in km. |
| `--power` | `0.0` | Optical launch power in dBm. |
| `--repeat` | `1` | Number of consecutive captures per run. |
| `--cooldown` | `2.0` | Pause in seconds between captures. |
| `--retries` | `3` | Maximum recovery retries on USRP `LookupError`. |
| `--usrp-args` | `""` | Additional UHD device arguments (e.g., `serial=...`). |
| `--samp-rate-div` | `1` | Decimation factor for the 15 MHz base clock. |
| `--samp-sym` | `16` | Samples per symbol (SPS). |
| `--zmq-addr` | `tcp://0.0.0.0:18305` | ZMQ endpoint for SNR heartbeat messages. |
| `--move` | `""` | Destination folder for renamed captures. |

For dataset compression, self-contained NPZ naming, and GITA uploads, see [`docs/gita_datasets_workflow.md`](../docs/gita_datasets_workflow.md).
