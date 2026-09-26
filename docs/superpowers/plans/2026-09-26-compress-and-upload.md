# Dataset Compression & Google Drive Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide an automated, fool-proof CLI workflow (`pixi run compress-and-upload -d <path>`) that compresses raw `.complex64` signal captures into `.npz` archives and safely triggers Google Drive synchronization via `rclone bisync` without upload conflicts.

**Architecture:** A dedicated module (`qpsk_src/uploader.py`) validates that the target experiment directory strictly resides within `optnet_ml/results/`, and triggers the host system's `rclone-bisync` service. The CLI (`qpsk_src/cli.py`) integrates this under `compress --upload` and exposes a top-level Pixi task (`compress-and-upload`).

**Tech Stack:** Python 3.10+, Pixi, systemd / rclone, pytest.

---

### Task 1: Results Path Validator and Bisync Trigger Module

**Files:**
- Create: `optnet_ml/qpsk_src/uploader.py`
- Create: `optnet_ml/tests/test_uploader.py`

- [ ] **Step 1: Write the failing unit tests for `uploader.py`**

Create `optnet_ml/tests/test_uploader.py`:
```python
"""Tests for dataset uploader and path validation module."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from qpsk_src.uploader import trigger_gdrive_sync, validate_results_path


def test_validate_results_path_valid(tmp_path: Path):
    """Test that a directory inside optnet_ml/results is accepted."""
    results_dir = tmp_path / "optnet_ml" / "results" / "hanna_maestria" / "exp1"
    results_dir.mkdir(parents=True)
    resolved = validate_results_path(results_dir)
    assert resolved == results_dir.resolve()


def test_validate_results_path_outside_rejected(tmp_path: Path):
    """Test that directories outside optnet_ml/results raise ValueError."""
    external_dir = tmp_path / "Muestras_Laboratorio" / "exp1"
    external_dir.mkdir(parents=True)
    with pytest.raises(ValueError, match="optnet_ml/results"):
        validate_results_path(external_dir)


def test_validate_results_path_nonexistent(tmp_path: Path):
    """Test that non-existent directory raises FileNotFoundError."""
    missing_dir = tmp_path / "optnet_ml" / "results" / "nonexistent"
    with pytest.raises(FileNotFoundError):
        validate_results_path(missing_dir)


def test_trigger_gdrive_sync_dry_run():
    """Test that dry_run mode does not invoke external commands."""
    success, msg = trigger_gdrive_sync(dry_run=True)
    assert success is True
    assert "DRY RUN" in msg


@patch("subprocess.run")
def test_trigger_gdrive_sync_systemd_success(mock_run: MagicMock):
    """Test successful systemd bisync trigger."""
    # First call: check service exists (returncode 0)
    # Second call: start service (returncode 0)
    mock_run.side_effect = [
        MagicMock(returncode=0, stdout="active\n", stderr=""),
        MagicMock(returncode=0, stdout="", stderr=""),
    ]
    success, msg = trigger_gdrive_sync(dry_run=False)
    assert success is True
    assert "exitosa" in msg.lower() or "success" in msg.lower()


@patch("subprocess.run")
def test_trigger_gdrive_sync_systemd_failure(mock_run: MagicMock):
    """Test handling of systemd trigger failure."""
    mock_run.side_effect = [
        MagicMock(returncode=0, stdout="active\n", stderr=""),
        MagicMock(returncode=1, stdout="", stderr="Job failed"),
    ]
    success, msg = trigger_gdrive_sync(dry_run=False)
    assert success is False
    assert "error" in msg.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /home/cesar/Cloud/GoogleDriveLAFI/optnet_ml && pytest tests/test_uploader.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'qpsk_src.uploader'`

- [ ] **Step 3: Write minimal implementation in `uploader.py`**

Create `optnet_ml/qpsk_src/uploader.py`:
```python
"""Dataset path validator and Google Drive synchronization trigger."""

from pathlib import Path
import shutil
import subprocess
from typing import Tuple


def validate_results_path(dataset_path: Path | str) -> Path:
    """
    Validates that a dataset directory exists, is a directory, and is located
    strictly inside the repository's 'optnet_ml/results' tree.

    Args:
        dataset_path: Relative or absolute path to the dataset directory.

    Returns:
        Resolved absolute Path.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If the path is not a directory or is outside optnet_ml/results.
    """
    path = Path(dataset_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"El directorio no existe: {path}")
    if not path.is_dir():
        raise ValueError(f"La ruta no es un directorio: {path}")

    # Check if 'optnet_ml/results' or 'results' is in the path ancestry
    parts = path.parts
    is_valid = False
    for i in range(len(parts) - 1):
        if parts[i] == "optnet_ml" and parts[i + 1] == "results":
            is_valid = True
            break
        if parts[i] == "results" and i > 0 and parts[i - 1] in ("optnet_ml", "lafi_shared", "GDrive"):
            is_valid = True
            break

    if not is_valid:
        # Fallback check if the name of path or parents match optnet_ml/results
        path_str = str(path)
        if "optnet_ml/results" in path_str or "GDrive/optnet_ml/results" in path_str:
            is_valid = True

    if not is_valid:
        raise ValueError(
            f"Ubicación inválida: '{path}'.\n"
            "Por estándar del laboratorio, la carpeta del experimento DEBE estar dentro "
            "del árbol de Google Drive en 'optnet_ml/results/...'.\n"
            "Por favor mueve o copia tus datos a 'optnet_ml/results/<tu_experimento>' antes de continuar."
        )

    return path


def trigger_gdrive_sync(timeout_seconds: int = 600, dry_run: bool = False) -> Tuple[bool, str]:
    """
    Triggers the system rclone-bisync service to synchronize pending .npz archives
    with Google Drive safely without timestamp or cache collision.

    Args:
        timeout_seconds: Maximum time to wait for the sync service to finish.
        dry_run: If True, simulates the action without triggering network transfer.

    Returns:
        Tuple of (success: bool, detail_message: str).
    """
    if dry_run:
        return True, "[DRY RUN] Simulación exitosa: rclone-bisync habría sido ejecutado."

    # 1. Try systemd user service: rclone-bisync@gdrive-lafi.service
    candidate_services = [
        "rclone-bisync@gdrive-lafi.service",
        "rclone-bisync.service",
    ]

    systemctl_bin = shutil.which("systemctl")
    if systemctl_bin:
        for service in candidate_services:
            # Check if unit is loaded / recognized
            check_res = subprocess.run(
                [systemctl_bin, "--user", "status", service],
                capture_output=True,
                text=True,
            )
            # returncode 0 (active) or 3 (inactive/dead) means service is loaded
            if check_res.returncode in (0, 3):
                # Trigger service and wait
                start_res = subprocess.run(
                    [systemctl_bin, "--user", "start", service],
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                )
                if start_res.returncode == 0:
                    return True, f"Sincronización completada exitosamente vía {service}."
                else:
                    return False, f"Error ejecutando {service}: {start_res.stderr.strip()}"

    # 2. Fallback to standalone script if present
    home = Path.home()
    script_candidates = [
        home / ".local" / "bin" / "rclone-sync.sh",
        home / "GDrive" / "misc" / "rclone-sync.sh",
    ]
    for script in script_candidates:
        if script.exists() and script.is_file():
            res = subprocess.run(
                [str(script)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            if res.returncode == 0:
                return True, f"Sincronización completada exitosamente vía {script.name}."
            else:
                return False, f"Error ejecutando {script.name}: {res.stderr.strip()}"

    return False, (
        "No se encontró un servicio systemd ni script de sincronización rclone activo en este equipo. "
        "Asegúrate de que rclone-bisync esté configurado."
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd /home/cesar/Cloud/GoogleDriveLAFI/optnet_ml && pytest tests/test_uploader.py -v
```
Expected: PASS with 5 passed tests.

- [ ] **Step 5: Commit changes**

Run:
```bash
cd /home/cesar/Cloud/GoogleDriveLAFI/optnet_ml && git add qpsk_src/uploader.py tests/test_uploader.py && git commit -m "feat(uploader): add results path validation and bisync trigger"
```

---

### Task 2: CLI Integration for `compress --upload`

**Files:**
- Modify: `optnet_ml/qpsk_src/cli.py:194-206,316-333`
- Create: `optnet_ml/tests/test_cli_compress_upload.py`

- [ ] **Step 1: Write failing CLI integration test**

Create `optnet_ml/tests/test_cli_compress_upload.py`:
```python
"""Tests for CLI compress command with --upload flag."""

from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest

from qpsk_src.cli import main


def test_cli_compress_with_upload_dry_run(tmp_path: Path, monkeypatch, capsys):
    """Test CLI compress with --upload and --dry-run."""
    # Create fake valid results directory
    dataset_dir = tmp_path / "optnet_ml" / "results" / "test_exp"
    dist_dir = dataset_dir / "0km"
    dist_dir.mkdir(parents=True)

    # Create dummy complex64 file
    raw_file = dist_dir / "signal.complex64"
    data = np.array([1 + 1j, -1 - 1j], dtype=np.complex64)
    data.tofile(raw_file)

    test_args = [
        "cli.py",
        "compress",
        "-d",
        str(dataset_dir),
        "--upload",
        "--dry-run",
    ]
    monkeypatch.setattr("sys.argv", test_args)

    main()

    captured = capsys.readouterr().out
    assert "Compressed" in captured or "0km" in captured
    assert "DRY RUN" in captured


def test_cli_compress_with_upload_outside_rejected(tmp_path: Path, monkeypatch):
    """Test CLI compress rejects paths outside optnet_ml/results when --upload is set."""
    external_dir = tmp_path / "external_exp" / "0km"
    external_dir.mkdir(parents=True)

    test_args = [
        "cli.py",
        "compress",
        "-d",
        str(external_dir),
        "--upload",
    ]
    monkeypatch.setattr("sys.argv", test_args)

    with pytest.raises(ValueError, match="optnet_ml/results"):
        main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /home/cesar/Cloud/GoogleDriveLAFI/optnet_ml && pytest tests/test_cli_compress_upload.py -v
```
Expected: FAIL with `unrecognized arguments: --upload --dry-run`

- [ ] **Step 3: Modify `optnet_ml/qpsk_src/cli.py`**

In `optnet_ml/qpsk_src/cli.py`:
1. Add `--upload` and `--dry-run` flags to `compress_parser`.
2. In the `command == "compress"` block:
   - If `args.upload` is set, validate path using `validate_results_path(args.dataset_dir)`.
   - Run compression logic.
   - If `args.upload` is set, invoke `trigger_gdrive_sync(dry_run=args.dry_run)`.
   - Print summary.

Update lines in `optnet_ml/qpsk_src/cli.py`:
```python
    # Subcommand: compress
    compress_parser = subparsers.add_parser(
        "compress", help="Compress raw complex64 files into NPZ archives."
    )
    compress_parser.add_argument(
        "--dataset-dir",
        "-d",
        required=True,
        type=str,
        help="Path to dataset directory or distance folder to compress.",
    )
    compress_parser.add_argument(
        "--upload",
        action="store_true",
        help="Trigger immediate Google Drive synchronization after compression.",
    )
    compress_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate compression and upload without modifying remote files.",
    )
```

And in the execution branch:
```python
    elif args.command == "compress":
        from .uploader import trigger_gdrive_sync, validate_results_path

        raw_path = Path(args.dataset_dir)
        if args.upload:
            path = validate_results_path(raw_path)
        else:
            path = raw_path

        compressed_count = 0
        try:
            distance_folders = validate_dataset_dir(path)
            for folder in distance_folders:
                npz = compress_distance_folder(folder)
                if npz:
                    compressed_count += 1
                    print(f"Compressed {folder.name} -> {npz}")
                else:
                    print(f"No files compressed in {folder.name}")
        except (ValueError, FileNotFoundError):
            npz = compress_distance_folder(path)
            if npz:
                compressed_count += 1
                print(f"Compressed {path.name} -> {npz}")
            else:
                print(f"No files to compress in {path}")

        if args.upload:
            print("\nIniciando sincronización con Google Drive (lafi:)...")
            success, sync_msg = trigger_gdrive_sync(dry_run=args.dry_run)
            print(sync_msg)
            if not success:
                raise SystemExit(1)
            print("\n¡Proceso completado exitosamente!")
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd /home/cesar/Cloud/GoogleDriveLAFI/optnet_ml && pytest tests/test_cli_compress_upload.py -v
```
Expected: PASS with 2 passed tests.

- [ ] **Step 5: Commit changes**

Run:
```bash
cd /home/cesar/Cloud/GoogleDriveLAFI/optnet_ml && git add qpsk_src/cli.py tests/test_cli_compress_upload.py && git commit -m "feat(cli): add --upload and --dry-run flags to compress subcommand"
```

---

### Task 3: Pixi Task Definition and Researcher Guide

**Files:**
- Modify: `optnet_ml/pixi.toml`
- Modify: `optnet_ml/README.md`

- [ ] **Step 1: Add Pixi task `compress-and-upload` in `pixi.toml`**

Add to `[tasks]` section of `optnet_ml/pixi.toml`:
```toml
compress-and-upload = "python3 -m qpsk_src.cli compress --upload"
```

- [ ] **Step 2: Add Researcher Workflow Guide to `optnet_ml/README.md`**

Add a clear section in `optnet_ml/README.md`:
```markdown
## Flujo de Captura y Subida de Datos (Hanna & David)

Para no saturar la cuota de almacenamiento de Google Drive (15 GB) con archivos `.complex64` sin comprimir:

1. **Ubicación de los Datos:**
   Guarda siempre la carpeta de tu experimento dentro de `optnet_ml/results/hanna_maestria/` (o la subcarpeta de resultados correspondiente dentro del árbol de Google Drive).
   ```text
   optnet_ml/results/hanna_maestria/tu_experimento/
   ├── 0km/
   │   └── *.complex64
   └── 20km/
       └── *.complex64
   ```

2. **Comprimir y Subir a Google Drive (Un solo comando):**
   Una vez terminada la toma de datos, abre la terminal en la carpeta `optnet_ml` y corre:
   ```bash
   pixi run compress-and-upload -d results/hanna_maestria/tu_experimento
   ```

3. **¿Qué hace este comando automáticamente?**
   - Comprime los `.complex64` en cada subcarpeta de distancia generando un archivo `data_<distancia>.npz`.
   - Dispara la sincronización oficial con Google Drive vía `rclone-bisync`.
   - Sube únicamente los archivos `.npz` y archivos de resultados (`.csv`, etc.). **Nunca** subirá archivos `.complex64`.
   - Notifica en pantalla cuando la subida esté completa.
```

- [ ] **Step 3: Run full test suite and verify CLI help**

Run:
```bash
cd /home/cesar/Cloud/GoogleDriveLAFI/optnet_ml && pixi run test
cd /home/cesar/Cloud/GoogleDriveLAFI/optnet_ml && pixi run compress-and-upload --help
```
Expected:
- All unit tests pass cleanly.
- `compress-and-upload --help` displays usage with `--upload` and `--dry-run`.

- [ ] **Step 4: Commit changes**

Run:
```bash
cd /home/cesar/Cloud/GoogleDriveLAFI/optnet_ml && git add pixi.toml README.md && git commit -m "docs: add compress-and-upload pixi task and researcher workflow guide"
```
