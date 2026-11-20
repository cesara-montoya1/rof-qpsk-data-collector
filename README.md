# OptNET ML

Repository: <https://github.com/cesara-montoya1/rof-qpsk-data-collector>  
Maintained by: **LAFI / SOFA Research Group**, Universidad de Antioquia

With **OptNET ML**, you can acquire, validate, compress, upload, demodulate, and diagnose experimental **Radio-over-Fiber (RoF)** optical signals modulated in **Quadrature Phase-Shift Keying (QPSK)** and **16-Quadrature Amplitude Modulation (16-QAM)**. It replaces ad-hoc capture scripts and manual spreadsheets with a reproducible Python and [Pixi](https://pixi.sh/) pipeline spanning Universal Software Radio Peripheral (USRP) receivers in the laboratory and High-Performance Computing (HPC) processing on the GITA cluster.

> **Scope:** OptNET ML focuses on physical-layer signal acquisition, offline digital signal processing (DSP), channel diagnostics, and centralized dataset management. It is not an optical fiber propagation simulator.

---

## Table of Contents

- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Usage & Core Workflows](#usage--core-workflows)
- [Repository Architecture](#repository-architecture)
- [Running Tests](#running-tests)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Credits & License](#credits--license)

---

## Key Features

- **Modulation-agnostic core (`common/`):** Shared dataset validation, filename metadata parsing, Error Vector Magnitude (EVM) computation, atomic CSV storage with automatic `.csv.bak` rotation, PRBS generation, and USRP USB auto-recovery reused cleanly by both QPSK and 16-QAM packages.
- **Automated USRP capture with USB recovery:** Runs GNU Radio Companion (GRC) receivers sequentially with cooldown windows and software USB port resets (`ioctl(USBDEVFS_RESET)`) when Ettus/NI hardware throws transient `LookupError` failures.
- **Self-contained dataset compression:** Packs raw `.complex64` captures into self-describing NumPy archives (`<capture_mode>_<technology>_<source>_<scenario>_<distance>.npz`) supporting `usrp_rx_gui`, `usrp_rx_headless`, and `usrp_rx_raw` capture modes.
- **Attributed GITA uploads over SSH/rsync:** Transfers datasets to the shared `sofa` UNIX group on GITA with mandatory personal SSH identity (`-i` / `OPTNET_GITA_KEY`), automatic `upload_log.jsonl` auditing, and campus Wi-Fi port-block detection (`VisitanteUdeA` / `SoyUdeA`).
- **Offline DSP & physical diagnostics:** FFT-accelerated time/phase synchronization, 4-rotation Bit Error Rate (BER) search, RMS/peak/best-window EVM, polar amplitude-vs-phase breakdown, error Power Spectral Density (PSD), 2D constellation density, and burst error profiles.

---

## Prerequisites

- **Linux (x86_64)** with [Pixi](https://pixi.sh/) installed (manages Python `>=3.10`, NumPy, Pandas, Matplotlib, Seaborn, tqdm, and Pytest).
- **For live USRP signal capture only:** GNU Radio 3.10+ with UHD and ZMQ blocks, plus a connected Ettus/NI USRP device.
- **For dataset uploads to GITA:** `rsync`, `ssh`, membership in the GITA `sofa` group, a personal SSH private key, a private `Host gita` alias in `~/.ssh/config`, and a **wired Ethernet** connection on campus.

---

## Quick Start

Follow these steps to clone the repository, install dependencies, and verify your environment for the first time:

```bash
# 1. Clone outside any cloud-synced LAFI folder
git clone git@github.com:cesara-montoya1/rof-qpsk-data-collector.git ~/Development/optnet_ml
cd ~/Development/optnet_ml

# 2. Install the reproducible Pixi environment
pixi install

# 3. Verify your installation by running the test suite
pixi run test
```

Once `pixi run test` finishes with all tests passing, your local environment is ready for offline demodulation, diagnostics, compression, and dataset uploads.

---

## Usage & Core Workflows

### 1. Inspect a Single Capture Instantly

Verify constellation lock, delay, BER, and EVM on a single `.complex64` file before running a full batch:

```bash
# QPSK capture (saves temporary PNG to /tmp/ unless -o is provided)
pixi run check-signal path/to/capture.complex64

# 16-QAM capture (headless mode with warm-up skip and best-window EVM)
pixi run check-signal-16qam path/to/capture.complex64 \
    --skip-initial-symbols 1000 \
    --best-window-symbols 5000 \
    --no-show \
    -o /tmp/check_16qam.png
```

### 2. Validate, Compress, and Upload a Dataset to GITA

Organize captures by technology and optical setup (`rof_qpsk/uspl/<experiment>/0km/`, `20km/`, etc.), then run a dry-run followed by the upload using your personal SSH key:

```bash
# Validate distance folder layout
pixi run validate-dataset -d rof_qpsk/uspl/nlna_step0p5db_370ma_650mhz_ch24

# Preview compression and rsync target without transferring
pixi run compress-and-upload \
    -d rof_qpsk/uspl/nlna_step0p5db_370ma_650mhz_ch24 \
    -i ~/.ssh/id_ed25519_gita \
    --dry-run

# Compress to self-contained .npz and upload (.npz + .complex64) to GITA
pixi run compress-and-upload \
    -d rof_qpsk/uspl/nlna_step0p5db_370ma_650mhz_ch24 \
    -i ~/.ssh/id_ed25519_gita

# On a shared lab PC, pass your GITA username explicitly for attribution
pixi run compress-and-upload \
    -d rof_qpsk/uspl/nlna_step0p5db_370ma_650mhz_ch24 \
    --ssh-user <your_gita_user> \
    -i ~/.ssh/<your_key>
```

> **Campus Network Note:** `VisitanteUdeA` and `SoyUdeA` Wi-Fi networks block GITA's SSH port. The uploader automatically detects them and aborts early; switch to wired **Ethernet** before uploading.

### 3. Batch Demodulation and Plotting

```bash
# Demodulate a QPSK dataset (writes results_<experiment>.csv with incremental backfill)
pixi run demodulate-qpsk -d /export2/camontoya/sofa_datasets/rof_qpsk/uspl/<experiment>

# Demodulate a 16-QAM dataset
pixi run demodulate-16qam -d /export2/camontoya/sofa_datasets/rof_16qam/dfb/<experiment>

# Generate BER and EVM curves vs. SNR/OSNR
pixi run plot-results -d /export2/camontoya/sofa_datasets/rof_qpsk/uspl/<experiment>

# Run QPSK demodulation on GITA via SLURM
srun --partition=full-gpu ./run_qpsk_demodulation.sh \
    /export2/camontoya/sofa_datasets/rof_qpsk/uspl/<experiment>
```

### Pixi Task Summary

| Task | Command | Purpose |
| :--- | :--- | :--- |
| `check-signal` | `pixi run check-signal <file.complex64>` | Quick QPSK constellation, BER, and EVM check. |
| `check-signal-16qam` | `pixi run check-signal-16qam <file.complex64>` | Quick 16-QAM constellation, BER, and EVM check. |
| `validate-dataset` | `pixi run validate-dataset -d <dir>` | Verify distance subfolder structure (`0km`, `2km`, `20km`, ...). |
| `compress-dataset` | `pixi run compress-dataset -d <dir>` | Compress `.complex64` files into self-contained `.npz` archives locally. |
| `compress-and-upload` | `pixi run compress-and-upload -d <dir> -i <key>` | Compress and upload `.npz` + `.complex64` to GITA over SSH/rsync. |
| `upload-dataset` | `pixi run upload-dataset -d <dir> -i <key>` | Upload an already-compressed dataset directly to GITA. |
| `demodulate-qpsk` | `pixi run demodulate-qpsk -d <dir>` | Run full QPSK demodulation pipeline and write CSV metrics. |
| `demodulate-16qam` | `pixi run demodulate-16qam -d <dir>` | Run full 16-QAM demodulation pipeline and write CSV metrics. |
| `plot-results` | `pixi run plot-results -d <dir>` | Generate BER/EVM curves with dispersion shading. |
| `test` | `pixi run test` | Execute the full Pytest unit and integration test suite. |

---

## Repository Architecture

```text
optnet_ml/
├── README.md                  # Project overview and quick start
├── pixi.toml                  # Reproducible dependencies and CLI tasks
├── run_qpsk_demodulation.sh   # Local and SLURM wrapper for GITA execution
├── common/                    # Modulation-agnostic shared package
│   ├── cli.py                 # Unified CLI (validate, compress, upload)
│   ├── compressor.py          # Self-contained NPZ archive builder
│   ├── uploader.py            # GITA rsync/SSH uploader, Wi-Fi guard & audit log
│   ├── validation.py          # Distance folder validation and numeric sorting
│   ├── metadata.py            # Regex metadata parser for signal filenames
│   ├── storage.py             # Atomic CSV persistence with .bak rotation
│   ├── evm.py                 # RMS, dB, peak, and best-window EVM calculations
│   ├── runner.py              # GRC execution runner with USRP USB ioctl reset
│   └── generate_tx.py         # Maximal-length PRBS LFSR sequence generator
├── qpsk_src/                  # QPSK demodulator, pipeline, plots, diagnostics & GRC flowgraphs
├── 16qam_src/                 # 16-QAM demodulator, pipeline, plots, diagnostics & GRC flowgraphs
├── tools/                     # Standalone offline DSP utilities
├── utils/                     # Auxiliary plotting and conversion scripts
├── tests/                     # Automated unit and GNU Radio integration tests
└── docs/                      # Detailed workflow, hardware, metrics, and diagnostics guides
```

---

## Running Tests

Run the full test suite from the repository root:

```bash
pixi run test
```

The suite validates constellation geometry, modulation/demodulation round-trips, EVM math, atomic CSV backfill, self-contained NPZ compression, GITA uploader Wi-Fi/identity checks, USRP USB recovery, and end-to-end GNU Radio 16-QAM simulation (when GNU Radio is installed on the host).

---

## Documentation

- **[Dataset Naming, Compression, and GITA Upload Workflow](docs/gita_datasets_workflow.md)** — Private SSH setup, `sofa` group permissions, self-contained NPZ naming rules, shared lab PC upload examples, `upload_log.jsonl` auditing, and `rclone-bisync` maintenance.
- **[Experiment README & Experimental Design Template](docs/experiment_readme_template.md)** — Standard per-campaign README format with Mermaid optical/RF bench schematic, operationalized variables, bias control, replication, and uncertainty reporting.
- **[Advanced Channel Diagnostics Guide](docs/channel_diagnostics.md)** — Physical interpretation of BER vs. theoretical AWGN EVM, polar amplitude/phase EVM decomposition, error PSD, 2D constellation density, and error burst profiles.
- **[CSV Metrics & Backfill Reference](docs/csv_metrics_reference.md)** — Exact schema, units, and incremental backfill semantics for `results_*.csv`.
- **[USRP Capture Hardware & Auto-Recovery](docs/usrp_capture_hardware.md)** — Automated multi-capture loops (`--repeat`, `--cooldown`) and software USB reset via `ioctl(USBDEVFS_RESET)`.
- **Package-specific guides:** [`qpsk_src/README.md`](qpsk_src/README.md) · [`16qam_src/README.md`](16qam_src/README.md)

---

## Contributing

1. Create a topic branch from `main` using `feature/<name>`, `fix/<name>`, or `refactor/<name>`.
2. Keep modulation-agnostic logic inside `common/` and modulation-specific DSP inside `qpsk_src/` or `16qam_src/`.
3. Never commit private hostnames, IP addresses, SSH ports, private keys, or raw dataset binaries (`.complex64`, `.npz`).
4. Ensure `pixi run test` passes cleanly before opening a pull request.

---

## Credits & License

Developed by the **LAFI / SOFA Research Group** at Universidad de Antioquia.  
Licensed under the **MIT License** — see [`qpsk_src/LICENSE`](qpsk_src/LICENSE) and [`16qam_src/LICENSE`](16qam_src/LICENSE).
