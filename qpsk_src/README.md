# RoF QPSK Data Collector

A modular Python automation framework designed for **Radio-over-Fiber (RoF)** experimental data acquisition. This tool orchestrates GNU Radio flowgraphs to capture QPSK modulated signals, calculates real-time SNR, and manages experimental datasets with high precision.

---

## 📁 Repository Structure

```text
qpsk_src/
├── data/               # Static reference data (tx.txt)
├── grc/                # GNU Radio Companion flowgraphs and compiled .py scripts
├── src/                # Core automation logic
│   ├── main.py         # Main orchestrator (--repeat, --retries, --cooldown)
│   ├── runner.py       # GRC process management, synchronous teardown & USB reset
│   └── file_manager.py # Naming convention and collision avoidance
└── tools/              # Post-processing & utility scripts
    ├── ber_count_qpsk.py # Single-file QPSK BER/EVM and constellation analyzer
    └── generate_tx.py    # PRBS generator using streaming binary chunks
```

---

## 🚀 Getting Started

### 1. Prerequisites

* **GNU Radio 3.8+** (with ZMQ and UHD blocks)
* **Python 3.11+**
* Connected USRP device (Ettus Research B200/B210/B200mini, NI USRP series)

### 2. Generate Transmission Reference Bits

Generate a PRBS sequence using the streaming generator (e.g. PRBS17 = 131,071 bits):

```bash
python3 tools/generate_tx.py --order 17 --output data/tx.txt
```

### 3. Running an Experiment (Automated Capture with Auto-Recovery)

Instead of using error-prone bash loops that cause USB collisions and `No devices found` errors, use the built-in `--repeat` and `--cooldown` parameters:

```bash
# Capture 10 consecutive files at 650 MHz, OSNR 32.5 dB, 20 km distance:
python3 src/main.py \
    --osnr 32.5 \
    --distance 20 \
    --power 0 \
    --freq 650 \
    --repeat 10 \
    --cooldown 2.0 \
    --retries 3 \
    --move ../data/results
```

#### 🛡️ USRP Hardware Teardown & Auto-Recovery:
- **`--repeat N`**: Executes $N$ captures sequentially without exiting the Python process.
- **`--cooldown T`**: Waits $T$ seconds (recommended: 2.0s) between iterations to guarantee `libusb` and the Linux kernel cleanly release the USRP device handle.
- **`--retries R`**: If a transient `RuntimeError: LookupError: KeyError: No devices found for -----> Empty Device Address` occurs, the runner initiates a **software USB bus reset** via `ioctl(USBDEVFS_RESET)` on the USRP port, emulating a physical disconnect/reconnect automatically.
- **`--usrp-args "..."`**: Allows specifying explicit USRP device parameters (e.g., `--usrp-args "type=b200"` or serial number).

---

## 📊 Automated Filename Convention

The system uses a strict naming convention to ensure datasets are self-describing and collision-resistant:

`rof_{Power}dBm_{Dist}km_osnr{OSNR}dB_{Freq}mhz_{Mbps}mbps_snr{MeasuredSNR}dB.complex64`

* **Mbps Calculation:** Based on base clock and samples per symbol.
* **Collision Avoidance:** If a file with the exact same measured SNR exists, the script applies a micro-jitter to the filename to prevent accidental overwrites.

---

## 🔍 Single-File Exploratory Analysis (`tools/ber_count_qpsk.py`)

To inspect a single captured signal file without running the full dataset pipeline:

```bash
python3 tools/ber_count_qpsk.py \
    --tx data/tx.txt \
    --rx ../data/results/your_file.complex64 \
    --skip-initial-symbols 500 \
    --best-window-symbols 5000 \
    --output report.png
```

**Output Panels:**
1. **BER across 90° Rotations:** Visualizes bit error rates for all 4 possible carrier phase locks ($0^\circ, 90^\circ, 180^\circ, 270^\circ$).
2. **Cumulative Error Plot:** Distinguishes between **Uniform Random Noise** (constant diagonal slope) and **Burst Errors** (sharp vertical steps).
3. **Constellation Diagram:** Displays synchronized received symbols against ideal QPSK coordinates, reporting both RMS EVM and best-window EVM.

---

## ⚙️ Configuration Parameters

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--freq` | **Required** | Center frequency in MHz (e.g. 650, 900, 1200). |
| `--osnr` | `0.0` | Optical Signal-to-Noise Ratio (dB) setpoint. |
| `--distance`| `0.0` | Optical fiber length in km. |
| `--power` | `0.0` | Launch power in dBm. |
| `--repeat` | `1` | Number of consecutive captures to perform. |
| `--cooldown`| `2.0` | Cooldown pause (seconds) between captures. |
| `--retries` | `3` | Max recovery retries upon USRP `LookupError`. |
| `--usrp-args`| `""` | Extra device arguments passed to UHD (e.g. `serial=...`). |
| `--samp-rate-div` | `1` | Decimation factor for the 15 MHz clock. |
| `--samp-sym` | `16` | Samples per symbol (SPS). |
| `--zmq-addr` | `tcp://0.0.0.0:18305` | ZMQ endpoint for SNR heartbeat messages. |
| `--move` | `""` | Destination folder for renamed files. |
