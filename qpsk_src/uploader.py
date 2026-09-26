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

    # Check if 'optnet_ml/results' appears in the path ancestry
    path_str = str(path)
    if "optnet_ml/results" not in path_str:
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

    # 1. Try systemd user service candidates
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
