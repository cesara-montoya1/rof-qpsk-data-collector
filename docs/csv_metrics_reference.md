# CSV Metrics & Incremental Backfill Reference

This reference documents every column produced in `results_<experiment>.csv` by the offline QPSK and 16-QAM demodulation pipelines (`qpsk_src.pipeline` and `16qam_src.pipeline`).

---

## Output CSV Schema

| Column | Unit / Type | Description |
| :--- | :--- | :--- |
| `launch_power_dbm` | dBm (`float`) | Optical launch power at the transmitter parsed from filename. |
| `distance_km` | km (`float`) | Optical fiber span length (sorted numerically: `0`, `0.5`, `2`, `20`, `40`). |
| `osnr_db` | dB (`float`) | Nominal Optical Signal-to-Noise Ratio setpoint (or auxiliary seed tone where documented). |
| `snr_db` | dB (`float`) | Electrical RF Signal-to-Noise Ratio measured by the USRP receiver. |
| `freq_mhz` | MHz (`float`) | RF carrier frequency. |
| `bitrate_mbps` | Mbps (`float`) | Nominal transmission bit rate. |
| `filename` | `str` | Signal key/stem inside the source `.npz` (without extension). |
| `ber` | `float` | Bit Error Rate after optimal constellation phase rotation ($0^\circ, 90^\circ, 180^\circ, 270^\circ$). |
| `detected_delay` | `int` | Estimated symbol alignment lag against the reference TX sequence. |
| `evm_rms_pct` | % (`float`) | Root-Mean-Square Error Vector Magnitude over all evaluated symbols. |
| `evm_db` | dB (`float`) | RMS EVM in decibels: $20 \log_{10}(\text{EVM}_{\text{rms}} / 100)$. |
| `evm_peak_pct` | % (`float`) | Peak normalized error vector magnitude across evaluated symbols. |
| `evm_best_rms_pct` | % (`float`) | *(Optional)* Lowest RMS EVM across any contiguous block of $M$ symbols (`--best-window-symbols`). |
| `evm_best_db` | dB (`float`) | *(Optional)* Best-window EVM in decibels. |
| `evm_delta_rms_pct` | % (`float`) | *(Optional)* $\Delta\text{EVM} = \text{EVM}_{\text{rms}} - \text{EVM}_{\text{best}}$ (quantifies transient burst impact). |
| `evm_delta_db` | dB (`float`) | *(Optional)* Difference in dB between overall and best-window EVM. |
| `npz_source` | `str` | Filename of the compressed `.npz` archive containing the processed capture. |

---

## Incremental Resumption & Atomic Storage (`common.storage`)

- **Skip completed records:** If a capture already has valid `ber` and `evm_rms_pct` (and `evm_best_rms_pct` when `--best-window-symbols` is requested), the pipeline skips recomputation.
- **In-place backfill:** If an older CSV contains BER measurements without EVM columns, the pipeline detects incomplete rows, computes missing EVM metrics, and updates those rows in place without duplicating entries.
- **Crash-safe writes:** `common.storage.save_results_safely()` backs up any existing CSV to `<name>.csv.bak`, writes to `<name>.csv.tmp`, and atomically replaces the target file.
