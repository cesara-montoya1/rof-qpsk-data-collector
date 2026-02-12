# RoF QPSK Data Collector

A modular Python automation framework designed for **Radio-over-Fiber (RoF)** experimental data acquisition. This tool orchestrates GNU Radio flowgraphs to capture QPSK modulated signals, calculates real-time SNR, and manages experimental datasets with high precision.

## 📁 Repository Structure

```text
rof-qpsk-data-collector/
├── data/               # Static data (tx.txt) and result exports
├── grc/                # GNU Radio Companion flowgraphs and compiled .py scripts
├── src/                # Core automation logic
│   ├── main.py         # Main entry point (Orchestrator)
│   ├── runner.py       # GRC process management & Heartbeat logic
│   └── file_manager.py # Dynamic naming
└── tools/              # Post-processing & Utility scripts
    ├── ber_count.py    # QPSK BER Analysis
    └── generate_tx.py  # Dynamic PRBS LFSR Generator

```

---

## 🚀 Getting Started

### 1. Prerequisites

* **GNU Radio 3.8+** (with ZMQ blocks)
* **Python 3.11+**

### 2. Generate Transmission Data

Before running an experiment, generate a PRBS sequence. For example, a PRBS17 (131,071 bits):

```bash
python tools/generate_tx.py --order 17 --output data/tx.txt

```

### 3. Running an Experiment

The orchestrator manages the GRC execution, monitors SNR via ZMQ, and renames the output based on physical parameters.

```bash
python src/main.py \
    --osnr 25.5 \
    --distance 20 \
    --power 0 \
    --freq 650 \
    --samp-rate-div 1 \
    --samp-sym 16 \
    --move ../data/results

```

**NOTE:** If you're in a laboratory with USRP hardware, manually open the GRC flowgraphs and disable the ZMQ blocks while enabling the USRP blocks.

---

## 📊 Automated Filename Convention

The system uses a strict naming convention to ensure datasets are self-describing and collision-resistant:

`rof_{Power}dBm_{Dist}km_osnr{OSNR}dB_{Freq}mhz_{Mbps}mbps_snr{MeasuredSNR}dB.complex64`

* **Mbps Calculation:** Based on a 15 MHz base clock.
* **Collision Avoidance:** If a file with the same measured SNR exists, the script applies a tiny SNR jitter (±dB) to the filename to prevent overwriting.

---

## 🔍 Analysis Tools

### BER & Phase Analysis

Since carrier recovery can lock in four different phases (0°, 90°, 180°, 270°), the analysis tool correlates all four rotations to find the best match.

```bash
python tools/ber_count.py --tx data/tx.txt --rx data/your_file.complex64

```

**Features:**

* **Correlation Map:** Identifies synchronization peaks.
* **Cumulative Error Plot:** Distinguishes between **Random Noise** (linear slope) and **Burst Errors** (vertical steps).
* **Constellation Diagram:** Visualizes signal quality for the best phase rotation.

---

## ⚙️ Configuration

| Parameter | Default | Description |
| --- | --- | --- |
| `--freq` | **Required** | Center Frequency in MHz. |
| `--samp-rate-div` | `1` | Decimation factor for the 15MHz clock. |
| `--samp-sym` | `16` | Samples per symbol (SPS). |
| `--zmq-addr` | `tcp://0.0.0.0:18305` | ZMQ Source for SNR messages. |
| `--move` | `""` | Destination folder. If empty, file is not renamed/moved. |

---

### Contribution & License

This project is designed for laboratory research. Feel free to open an issue for new modulation support.
