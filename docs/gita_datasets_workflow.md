# Dataset Naming, Compression, and GITA Upload Workflow

Complete operational guide for managing experimental Radio-over-Fiber (RoF) datasets across **GitHub**, **GITA (`sofa` group)**, and **Google Drive LAFI**.

---

## 1. Storage Architecture (Where Everything Lives)

| Asset Type | Location | Policy |
| :--- | :--- | :--- |
| **Source code (`optnet_ml`)** | GitHub (`rof-qpsk-data-collector`), cloned locally outside Google Drive (e.g., `~/Projects/optnet_ml`) | Never store `.pixi/` or raw datasets inside Google Drive. |
| **Heavy datasets (`.npz`, `.complex64`) & spectra** | GITA: `/export2/camontoya/sofa_datasets/` | Shared via UNIX group **`sofa`** (`chmod 2770`, SGID). Preserves both compressed `.npz` and raw `.complex64`. |
| **Lab docs, characterizations & lightweight results** | Google Drive LAFI (`caracterizaciones/`, `documentos/`, `misc/`, `resultados/`) | Synced automatically via `rclone-bisync`. `filters-lafi.txt` strictly blocks `*.npz`, `*.complex64`, and `/espectros/**`. |

---

## 2. One-Time SSH Setup (Private Per Researcher)

Server domain/IP and SSH port are kept private in each researcher's local `~/.ssh/config` (never committed to Git).

1. **Generate a personal SSH key (if you do not have one yet):**
   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_gita -C "your.name@udea.edu.co"
   ```
2. **Add the private `gita` alias to `~/.ssh/config`** (request the actual hostname and port from the lab coordinator):
   ```sshconfig
   Host gita
       HostName <GITA_HOST>
       Port <GITA_SSH_PORT>
       User <your_gita_username>
       IdentityFile ~/.ssh/id_ed25519_gita
       IdentitiesOnly yes
   ```
3. **Optional — set your default key via environment variable** (in `~/.bashrc` or `~/.zshrc`) so you do not have to pass `-i` on every upload from your personal laptop:
   ```bash
   export OPTNET_GITA_KEY="$HOME/.ssh/id_ed25519_gita"
   ```
   *(On shared lab PCs, do **not** export a shared key — always pass `-i` and `--ssh-user` explicitly so file ownership and `upload_log.jsonl` attribute every upload to the exact researcher.)*

---

## 3. Directory Structure & Self-Contained NPZ Naming

Datasets are grouped by **technology** and **optical setup** (lowercase `snake_case`, no spaces or accents):

```text
/export2/camontoya/sofa_datasets/
├── rof_qpsk/
│   ├── dfb/
│   │   ├── baseline_650mhz/
│   │   ├── snr18db_650mhz/
│   │   ├── dh_step0p4db_650mhz/
│   │   └── dh_step0p4db_900mhz/
│   └── uspl/
│       ├── baseline_650mhz/
│       ├── baseline_650mhz_calibrated_osnr/
│       ├── dh_650mhz/
│       ├── prbs23_650mhz/
│       ├── nlna_step0p5db_370ma_650mhz_ch24/
│       └── nlna_step1p0db_650mhz_ch24/
├── rof_16qam/
├── espectros/
├── archived_history/
└── upload_log.jsonl
```

### Self-Contained `.npz` Filename Format

```text
<capture_mode>_<technology>_<source>_<scenario>_<distance>.npz
```

Supported `--capture-mode` prefixes:
- `usrp_rx_gui` *(default)*: Captures generated with the GNU Radio GUI receiver flowgraph.
- `usrp_rx_headless`: Captures generated with the headless GNU Radio flowgraph.
- `usrp_rx_raw`: Raw IQ captures without GNU Radio DSP, intended for offline DSP pipelines.

**Example:**
- Folder: `rof_qpsk/uspl/nlna_step0p5db_370ma_650mhz_ch24/0km/`
- Generated archive: `usrp_rx_gui_rof_qpsk_uspl_nlna_step0p5db_370ma_650mhz_ch24_0km.npz`
- Results CSV (at experiment root): `results_nlna_step0p5db_370ma_650mhz_ch24.csv`

*(Internal keys inside `.npz` archives and the `filename` column of CSVs preserve the original capture stem so `common.metadata.parse_filename_metadata` works without changes.)*

---

## 4. Common CLI Examples (`pixi run`)

All dataset validation, compression, and upload tasks live in the shared `common` package and work identically for **QPSK**, **16-QAM**, and future modulation schemes.

### 4.1 Validate Distance Folder Structure
```bash
pixi run validate-dataset -d rof_qpsk/uspl/nlna_step0p5db_370ma_650mhz_ch24
```

### 4.2 Compress Only (Local, No Upload)
```bash
# Default capture mode (usrp_rx_gui):
pixi run compress-dataset -d rof_qpsk/uspl/nlna_step0p5db_370ma_650mhz_ch24

# Raw IQ captures for offline DSP:
pixi run compress-dataset -d rof_16qam/dfb/test_650mhz --capture-mode usrp_rx_raw

# Force re-compression even if an up-to-date .npz already exists:
pixi run compress-dataset -d rof_qpsk/dfb/baseline_650mhz --force
```

### 4.3 Compress and Upload to GITA (`rsync` over SSH)
> **Network Warning:** Campus Wi-Fi networks **`VisitanteUdeA`** and **`SoyUdeA`** block GITA's SSH port. The uploader automatically detects these SSIDs and aborts with a warning unless wired Ethernet is active. Connect via **Ethernet cable** before uploading.

```bash
# 1. Dry-run first (verifies naming, SSH identity, and exact remote path without transferring):
pixi run compress-and-upload \
    -d rof_qpsk/uspl/nlna_step0p5db_370ma_650mhz_ch24 \
    -i ~/.ssh/id_ed25519_gita \
    --dry-run

# 2. Standard upload from your personal laptop (uploads both .npz and .complex64):
pixi run compress-and-upload \
    -d rof_qpsk/uspl/nlna_step0p5db_370ma_650mhz_ch24 \
    -i ~/.ssh/id_ed25519_gita

# 3. Upload from a shared lab PC (explicit user + personal SSH key for attribution):
pixi run compress-and-upload \
    -d rof_qpsk/uspl/nlna_step0p5db_370ma_650mhz_ch24 \
    --ssh-user hanna.colorado \
    -i ~/.ssh/id_ed25519_hanna

# 4. Upload a flat local folder into a specific technology path on GITA:
pixi run compress-and-upload \
    -d ~/captures/nlna_step0p5db_370ma_650mhz_ch29 \
    --tech-subdir rof_qpsk/uspl \
    -i ~/.ssh/id_ed25519_gita

# 5. Upload compressed .npz and CSVs only (skip raw .complex64 on slow links):
pixi run compress-and-upload \
    -d rof_qpsk/uspl/nlna_step0p5db_370ma_650mhz_ch24 \
    -i ~/.ssh/id_ed25519_gita \
    --npz-only
```

### 4.4 Upload an Already-Compressed Dataset
```bash
pixi run upload-dataset \
    -d rof_qpsk/uspl/nlna_step0p5db_370ma_650mhz_ch24 \
    -i ~/.ssh/id_ed25519_gita
```

### 4.5 Demodulate Directly on GITA & Copy Lightweight CSV to Google Drive
```bash
# On GITA (SLURM interactive or batch):
srun --partition=full-gpu ./run_qpsk_demodulation.sh \
    /export2/camontoya/sofa_datasets/rof_qpsk/uspl/nlna_step0p5db_370ma_650mhz_ch24

# For 16-QAM:
pixi run demodulate-16qam -d /export2/camontoya/sofa_datasets/rof_16qam/dfb/<experiment>
```
Once `results_<experiment>.csv` and `plots/` are generated, copy **only** those lightweight files into `GoogleDriveLAFI/resultados/<technology>/<setup>/<experiment>/` if quick access via Drive is needed.

---

## 5. Checking Upload Attribution & Group Permissions on GITA

Every successful upload sets group-writable SGID permissions (`--chmod=Dg+s,ug+rwX,o-rwx`) and appends an audit record to `/export2/camontoya/sofa_datasets/upload_log.jsonl`.

```bash
# View recent uploads (who uploaded what, from which host, and file counts):
ssh gita "tail -n 10 /export2/camontoya/sofa_datasets/upload_log.jsonl"

# Inspect filesystem owner & group (owner = GITA user who published; group = sofa):
ssh gita "ls -la /export2/camontoya/sofa_datasets/rof_qpsk/uspl/"
```

---

## 6. Updating Lab PCs (`rclone-bisync`)

To keep Google Drive lightweight on laboratory machines (`lab18305pc2`, etc.):
1. Copy `GoogleDriveLAFI/misc/filters.txt` to `~/.config/rclone/filters-lafi.txt`.
2. Run a one-time baseline resync after major remote cleanups:
   ```bash
   rclone bisync ~/Cloud/GoogleDriveLAFI lafi:/ \
       --filter-from ~/.config/rclone/filters-lafi.txt \
       --compare modtime,size --modify-window 2s \
       --drive-export-formats link.html --fix-case \
       --resync --resilient --recover
   ```

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| `Se requiere especificar tu archivo de identidad SSH personal...` | Neither `-i/--identity` nor `OPTNET_GITA_KEY` was provided. | Pass `-i ~/.ssh/<your_key>` (or export `OPTNET_GITA_KEY`). |
| `ADVERTENCIA DE RED: Estás conectado a la red Wi-Fi 'VisitanteUdeA' / 'SoyUdeA'` | Campus Wi-Fi blocks GITA's SSH port. | Plug in wired **Ethernet** cable (or pass `--ignore-wifi-check` only if routing through an active VPN). |
| `Permission denied` on GITA folder | User not yet in group `sofa` or old session cached groups. | Run `id` on GITA to confirm `sofa` is listed; log out and back in if recently added. |
