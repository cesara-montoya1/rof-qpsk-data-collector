"""Shared dataset uploader to GITA over SSH/rsync with Wi-Fi guard and user attribution."""

from __future__ import annotations

from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
from typing import Optional, Tuple

from .compressor import DEFAULT_CAPTURE_MODE

BLOCKED_WIFI_SSIDS = ("VisitanteUdeA", "SoyUdeA")
DEFAULT_GITA_HOST = "gita"
DEFAULT_GITA_REMOTE_ROOT = "/export2/camontoya/sofa_datasets"

_TECH_DIR_RE = re.compile(r"^rof_[a-z0-9]+$", re.IGNORECASE)


def validate_dataset_upload_path(dataset_path: Path | str) -> Path:
    """
    Validates that dataset_path exists and is a directory ready for upload to GITA.

    Args:
        dataset_path: Path to the local dataset directory.

    Returns:
        Resolved absolute Path.

    Raises:
        FileNotFoundError: If dataset_path does not exist.
        ValueError: If dataset_path is not a directory.
    """
    path = Path(dataset_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Dataset path is not a directory: {path}")
    return path


def detect_active_wifi_ssid() -> Optional[str]:
    """
    Detects the currently connected Wi-Fi SSID on Linux using nmcli or iwgetid.

    Returns:
        Active SSID string if connected to Wi-Fi, otherwise None.
    """
    nmcli_bin = shutil.which("nmcli")
    if nmcli_bin:
        try:
            res = subprocess.run(
                [nmcli_bin, "-t", "-f", "ACTIVE,SSID", "dev", "wifi"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    if line.startswith("yes:"):
                        ssid = line.split(":", 1)[1].strip()
                        if ssid:
                            return ssid
        except (OSError, subprocess.SubprocessError):
            pass

    iwgetid_bin = shutil.which("iwgetid")
    if iwgetid_bin:
        try:
            res = subprocess.run(
                [iwgetid_bin, "-r"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0:
                ssid = res.stdout.strip()
                if ssid:
                    return ssid
        except (OSError, subprocess.SubprocessError):
            pass

    return None


def has_active_ethernet() -> bool:
    """Returns True if an active wired Ethernet connection is reported by nmcli."""
    nmcli_bin = shutil.which("nmcli")
    if not nmcli_bin:
        return False
    try:
        res = subprocess.run(
            [nmcli_bin, "-t", "-f", "TYPE,STATE", "dev"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            for line in res.stdout.splitlines():
                parts = line.strip().split(":")
                if len(parts) >= 2 and parts[0] == "ethernet" and parts[1] == "connected":
                    return True
    except (OSError, subprocess.SubprocessError):
        pass
    return False


def check_gita_network_access(
    ignore_wifi_check: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    Checks whether the host is on a campus Wi-Fi network known to block GITA's SSH port.

    Returns:
        Tuple of (can_proceed, warning_message).
        When connected to VisitanteUdeA or SoyUdeA without active Ethernet:
        - returns (False, warning_message) unless ignore_wifi_check=True.
        When connected to those SSIDs WITH active Ethernet (or ignore_wifi_check=True):
        - returns (True, warning_message) so the warning is still displayed.
    """
    ssid = detect_active_wifi_ssid()
    if not ssid:
        return True, None

    blocked_lower = {name.lower() for name in BLOCKED_WIFI_SSIDS}
    if ssid.lower() not in blocked_lower:
        return True, None

    warning_msg = (
        f"⚠️ ADVERTENCIA DE RED: Estás conectado a la red Wi-Fi '{ssid}'. "
        "Las redes VisitanteUdeA y SoyUdeA bloquean el puerto SSH de acceso al servidor GITA. "
        "Conéctate por cable Ethernet para subir datos a GITA."
    )

    if ignore_wifi_check or has_active_ethernet():
        return True, warning_msg

    return False, warning_msg


def resolve_ssh_user(ssh_host: str, ssh_user: Optional[str] = None) -> str:
    """
    Resolves the GITA SSH username for per-user file ownership and audit logging.
    Priority: explicit ssh_user -> OPTNET_GITA_USER env -> `ssh -G <host>` -> local user.
    """
    if ssh_user:
        return ssh_user.strip()

    env_user = os.environ.get("OPTNET_GITA_USER", "").strip()
    if env_user:
        return env_user

    ssh_bin = shutil.which("ssh")
    if ssh_bin:
        try:
            res = subprocess.run(
                [ssh_bin, "-G", ssh_host],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    if line.lower().startswith("user "):
                        parsed = line.split(None, 1)[1].strip()
                        if parsed:
                            return parsed
        except (OSError, subprocess.SubprocessError):
            pass

    return getpass.getuser()


def infer_remote_relative_path(
    dataset_path: Path, tech_subdir: Optional[str] = None
) -> str:
    """
    Computes the target relative path under GITA's sofa_datasets root.

    - If tech_subdir is provided (e.g., ``rof_qpsk/uspl``), returns ``<tech_subdir>/<dataset.name>``.
    - Otherwise, if a technology directory (e.g., ``rof_qpsk`` / ``rof_16qam``) appears
      in dataset_path's ancestry, preserves the hierarchy starting from that directory.
    - Otherwise falls back to ``dataset_path.name``.
    """
    resolved = dataset_path.resolve()
    if tech_subdir:
        clean_sub = tech_subdir.strip().strip("/")
        return f"{clean_sub}/{resolved.name}" if clean_sub else resolved.name

    parts = resolved.parts
    for idx in range(len(parts) - 1, -1, -1):
        if _TECH_DIR_RE.match(parts[idx]):
            return "/".join(parts[idx:])

    return resolved.name


def resolve_identity_file(identity_file: Optional[Path | str] = None) -> Path:
    """
    Resolves and validates the mandatory per-user SSH identity file (-i / OPTNET_GITA_KEY).

    Raises:
        ValueError: If no identity file is supplied via CLI or OPTNET_GITA_KEY.
        FileNotFoundError: If the specified identity file does not exist or is not a file.
    """
    raw = identity_file or os.environ.get("OPTNET_GITA_KEY")
    if not raw or not str(raw).strip():
        raise ValueError(
            "Se requiere especificar tu archivo de identidad SSH personal mediante "
            "-i/--identity o la variable de entorno OPTNET_GITA_KEY para identificar "
            "quién publica los datos en GITA."
        )
    key_path = Path(str(raw).strip()).expanduser().resolve()
    if not key_path.exists() or not key_path.is_file():
        raise FileNotFoundError(f"Archivo de identidad SSH no encontrado: {key_path}")
    return key_path


def build_ssh_transport_args(
    identity_file: Path | str,
    connect_timeout: int = 15,
) -> list[str]:
    """Builds the SSH command tokens used by rsync -e and remote mkdir/log commands."""
    key_path = Path(identity_file).expanduser()
    return [
        "ssh",
        "-o",
        f"ConnectTimeout={connect_timeout}",
        "-i",
        str(key_path),
        "-o",
        "IdentitiesOnly=yes",
    ]


def _collect_dataset_stats(dataset_path: Path, include_raw: bool) -> dict:
    npz_files = sorted(str(p.relative_to(dataset_path)) for p in dataset_path.rglob("*.npz"))
    c64_files = (
        sorted(dataset_path.rglob("*.complex64")) if include_raw else []
    )
    total_bytes = 0
    for f in dataset_path.rglob("*"):
        if not f.is_file():
            continue
        if not include_raw and f.suffix.lower() == ".complex64":
            continue
        try:
            total_bytes += f.stat().st_size
        except OSError:
            pass

    return {
        "npz_count": len(npz_files),
        "npz_files": npz_files,
        "complex64_count": len(c64_files),
        "total_bytes": total_bytes,
    }


def _append_remote_upload_log(
    ssh_cmd: list[str],
    ssh_target: str,
    remote_root: str,
    record: dict,
) -> Tuple[bool, str]:
    """Appends one JSON line to <remote_root>/upload_log.jsonl on GITA."""
    log_path = f"{remote_root.rstrip('/')}/upload_log.jsonl"
    json_line = json.dumps(record, ensure_ascii=False)
    remote_cmd = (
        f"umask 002 && printf '%s\\n' {shlex.quote(json_line)} >> {shlex.quote(log_path)}"
    )
    res = subprocess.run(
        [*ssh_cmd, ssh_target, remote_cmd],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if res.returncode == 0:
        return True, f"Registro de auditoría añadido en {log_path}"
    return False, f"No se pudo actualizar upload_log.jsonl: {res.stderr.strip()}"


def upload_dataset_to_gita(
    dataset_path: Path | str,
    ssh_host: Optional[str] = None,
    ssh_user: Optional[str] = None,
    identity_file: Optional[Path | str] = None,
    remote_root: Optional[str] = None,
    tech_subdir: Optional[str] = None,
    researcher: Optional[str] = None,
    capture_mode: str = DEFAULT_CAPTURE_MODE,
    include_raw: bool = True,
    ignore_wifi_check: bool = False,
    dry_run: bool = False,
    timeout_seconds: int = 3600,
) -> Tuple[bool, str]:
    """
    Uploads a dataset directory to GITA using rsync over SSH with group `sofa`
    permissions (SGID + g+rwX), Wi-Fi port-block guard, and per-user attribution.

    Args:
        dataset_path: Local dataset directory to upload.
        ssh_host: SSH host alias from ~/.ssh/config (default: OPTNET_GITA_HOST or 'gita').
        ssh_user: Optional explicit GITA SSH username (overrides ~/.ssh/config User).
        identity_file: Optional path to SSH private key (-i) for per-user identity on shared PCs.
        remote_root: Remote base directory on GITA (default: /export2/camontoya/sofa_datasets).
        tech_subdir: Optional remote subdirectory by technology/setup (e.g. 'rof_qpsk/uspl').
        researcher: Optional researcher label stored in upload_log.jsonl.
        capture_mode: Acquisition mode metadata recorded in upload_log.jsonl.
        include_raw: If True (default), upload both .npz and .complex64 files; if False, exclude .complex64.
        ignore_wifi_check: If True, warn on VisitanteUdeA/SoyUdeA but do not abort.
        dry_run: If True, simulate rsync transfer without modifying remote files.
        timeout_seconds: Timeout for the rsync transfer.

    Returns:
        Tuple of (success, detail_message).
    """
    local_path = validate_dataset_upload_path(dataset_path)

    try:
        resolved_identity = resolve_identity_file(identity_file)
    except (ValueError, FileNotFoundError) as exc:
        return False, str(exc)

    can_proceed, wifi_warning = check_gita_network_access(
        ignore_wifi_check=ignore_wifi_check
    )
    if wifi_warning:
        print(wifi_warning)
    if not can_proceed and not dry_run:
        return False, (
            f"{wifi_warning}\n"
            "Subida cancelada para evitar timeout. Conéctate por cable Ethernet "
            "(o usa --ignore-wifi-check si tienes VPN/túnel activo)."
        )

    rsync_bin = shutil.which("rsync")
    if not rsync_bin and not dry_run:
        return False, "No se encontró 'rsync' instalado en este equipo."

    resolved_host = (
        ssh_host or os.environ.get("OPTNET_GITA_HOST") or DEFAULT_GITA_HOST
    ).strip()
    resolved_user = resolve_ssh_user(resolved_host, ssh_user=ssh_user)
    resolved_root = (
        remote_root
        or os.environ.get("OPTNET_GITA_REMOTE_ROOT")
        or os.environ.get("OPTNET_GITA_DEST")
        or DEFAULT_GITA_REMOTE_ROOT
    ).rstrip("/")

    ssh_target = f"{ssh_user.strip()}@{resolved_host}" if ssh_user else resolved_host
    rel_dest = infer_remote_relative_path(local_path, tech_subdir=tech_subdir)
    remote_dir = f"{resolved_root}/{rel_dest}"

    ssh_cmd = build_ssh_transport_args(identity_file=resolved_identity)
    ssh_cmd_str = " ".join(shlex.quote(tok) for tok in ssh_cmd)

    rsync_cmd = [
        rsync_bin or "rsync",
        "-avz",
        "--partial",
        "--progress",
        "--chmod=Dg+s,ug+rwX,o-rwx",
        "-e",
        ssh_cmd_str,
    ]
    if not include_raw:
        rsync_cmd.extend(["--exclude=*.complex64", "--exclude=*.COMPLEX64"])
    if dry_run:
        rsync_cmd.append("--dry-run")

    rsync_cmd.extend([f"{local_path}/", f"{ssh_target}:{remote_dir}/"])

    if dry_run:
        return True, (
            f"[DRY RUN] Usuario SSH: {resolved_user} | Identidad: {resolved_identity} | Destino: {ssh_target}:{remote_dir}/\n"
            f"[DRY RUN] Comando: {' '.join(shlex.quote(c) for c in rsync_cmd)}"
        )

    # Ensure remote directory exists with SGID group-writable permissions
    mkdir_res = subprocess.run(
        [
            *ssh_cmd,
            ssh_target,
            f"umask 002 && mkdir -p {shlex.quote(remote_dir)}",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if mkdir_res.returncode != 0:
        err = mkdir_res.stderr.strip()
        hint = (
            "\nRecuerda: las redes Wi-Fi VisitanteUdeA y SoyUdeA bloquean el puerto SSH "
            "de acceso a GITA; conéctate por cable Ethernet."
        )
        return False, f"Error conectando con GITA ({ssh_target}): {err}{hint}"

    transfer_res = subprocess.run(
        rsync_cmd,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    if transfer_res.returncode != 0:
        err = transfer_res.stderr.strip()
        hint = (
            "\nVerifica conexión por cable Ethernet (VisitanteUdeA/SoyUdeA bloquean GITA) "
            "y tus credenciales SSH (--ssh-user / -i)."
        )
        return False, f"Error ejecutando rsync hacia GITA: {err}{hint}"

    stats = _collect_dataset_stats(local_path, include_raw=include_raw)
    audit_record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ssh_user": resolved_user,
        "researcher": researcher or os.environ.get("OPTNET_RESEARCHER") or resolved_user,
        "source_host": socket.gethostname(),
        "remote_rel_path": rel_dest,
        "capture_mode": capture_mode,
        "include_raw": include_raw,
        **stats,
    }
    log_ok, log_msg = _append_remote_upload_log(
        ssh_cmd=ssh_cmd,
        ssh_target=ssh_target,
        remote_root=resolved_root,
        record=audit_record,
    )

    summary = (
        f"Subida a GITA completada exitosamente por '{resolved_user}' -> "
        f"{ssh_target}:{remote_dir}/ ({stats['npz_count']} .npz, "
        f"{stats['complex64_count']} .complex64)."
    )
    if not log_ok:
        summary = f"{summary}\n⚠️ {log_msg}"
    return True, summary


# Legacy aliases retained for backward compatibility with existing callers
validate_results_path = validate_dataset_upload_path
