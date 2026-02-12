# RoF QPSK Data Collector

This repository contains a modular Python tool designed to automate data collection from GNU Radio Companion (GRC) flowgraphs in Radio over Fiber (RoF) experiments.

## Features
* **Non-blocking Execution**: Runs `qpsk_rx.py` as a subprocess.
* **Smart Timeout**: Features a 0.5s inactivity heartbeat that activates only after the first sample is received.
* **Tee Logging**: Captures and redirects all GRC terminal output to local logs for debugging.
* **Collision Avoidance**: Automatically adjusts SNR in filenames (±0.03 dB) to prevent overwriting existing data.

## Usage
```bash
python main.py --osnr 25.5 --distance 20 --power 0 --out_dir ./results --samples 50
