"""Unified modulation-agnostic CLI for dataset validation, compression, and GITA upload."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .compressor import (
    DEFAULT_CAPTURE_MODE,
    KNOWN_CAPTURE_MODES,
    compress_distance_folder,
)
from .uploader import (
    DEFAULT_GITA_HOST,
    DEFAULT_GITA_REMOTE_ROOT,
    upload_dataset_to_gita,
)
from .validation import validate_dataset_dir


def add_upload_arguments(parser: argparse.ArgumentParser) -> None:
    """Adds GITA rsync-over-SSH upload flags to an argparse subparser."""
    parser.add_argument(
        "--ssh-host",
        type=str,
        default=None,
        help=f"SSH host alias in ~/.ssh/config (default: $OPTNET_GITA_HOST or '{DEFAULT_GITA_HOST}').",
    )
    parser.add_argument(
        "--ssh-user",
        type=str,
        default=None,
        help="Explicit GITA SSH username for per-user attribution on shared lab PCs.",
    )
    parser.add_argument(
        "-i",
        "--identity",
        type=str,
        default=None,
        help="Required path to your personal SSH private key file (or set OPTNET_GITA_KEY).",
    )
    parser.add_argument(
        "--remote-root",
        type=str,
        default=None,
        help=f"Remote base directory on GITA (default: '{DEFAULT_GITA_REMOTE_ROOT}').",
    )
    parser.add_argument(
        "--tech-subdir",
        type=str,
        default=None,
        help="Technology/setup relative path on GITA (e.g., 'rof_qpsk/uspl' or 'rof_16qam/dfb').",
    )
    parser.add_argument(
        "--researcher",
        type=str,
        default=None,
        help="Optional researcher name recorded in GITA's upload_log.jsonl.",
    )
    parser.add_argument(
        "--npz-only",
        action="store_true",
        help="Upload only compressed .npz/CSV files (exclude raw .complex64).",
    )
    parser.add_argument(
        "--ignore-wifi-check",
        action="store_true",
        help="Warn on VisitanteUdeA/SoyUdeA Wi-Fi without aborting (e.g., when using VPN).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate compression/upload without writing or transferring remote files.",
    )


def handle_compress_command(args: argparse.Namespace) -> None:
    """Shared implementation of the `compress` subcommand across common/QPSK/16-QAM CLIs."""
    path = Path(args.dataset_dir)
    capture_mode = getattr(args, "capture_mode", DEFAULT_CAPTURE_MODE)
    scenario_name = getattr(args, "scenario_name", None)
    force = getattr(args, "force", False)

    try:
        distance_folders = validate_dataset_dir(path)
        for folder in distance_folders:
            npz = compress_distance_folder(
                folder,
                capture_mode=capture_mode,
                scenario_name=scenario_name,
                force=force,
            )
            if npz:
                print(f"Compressed {folder.name} -> {npz}")
            else:
                print(f"No files compressed in {folder.name}")
    except (ValueError, FileNotFoundError):
        npz = compress_distance_folder(
            path,
            capture_mode=capture_mode,
            scenario_name=scenario_name,
            force=force,
        )
        if npz:
            print(f"Compressed {path.name} -> {npz}")
        else:
            print(f"No files to compress in {path}")

    if getattr(args, "upload", False):
        print("\nIniciando subida hacia GITA vía rsync/SSH...")
        success, msg = upload_dataset_to_gita(
            dataset_path=path,
            ssh_host=getattr(args, "ssh_host", None),
            ssh_user=getattr(args, "ssh_user", None),
            identity_file=getattr(args, "identity", None),
            remote_root=getattr(args, "remote_root", None),
            tech_subdir=getattr(args, "tech_subdir", None),
            researcher=getattr(args, "researcher", None),
            capture_mode=capture_mode,
            include_raw=not getattr(args, "npz_only", False),
            ignore_wifi_check=getattr(args, "ignore_wifi_check", False),
            dry_run=getattr(args, "dry_run", False),
        )
        print(msg)
        if not success:
            raise SystemExit(1)
        print("\n¡Proceso completado exitosamente!")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="OptNET ML Shared Dataset CLI (validate, compress, upload to GITA)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # compress
    compress_parser = subparsers.add_parser(
        "compress",
        help="Compress raw .complex64 files into self-contained .npz archives (optionally upload to GITA).",
    )
    compress_parser.add_argument(
        "--dataset-dir",
        "-d",
        required=True,
        type=str,
        help="Path to dataset directory or single distance folder.",
    )
    compress_parser.add_argument(
        "--capture-mode",
        type=str,
        default=DEFAULT_CAPTURE_MODE,
        help=f"Capture mode prefix for NPZ names (presets: {', '.join(KNOWN_CAPTURE_MODES)}; default: {DEFAULT_CAPTURE_MODE}).",
    )
    compress_parser.add_argument(
        "--scenario-name",
        type=str,
        default=None,
        help="Explicit scenario slug override for NPZ filename.",
    )
    compress_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-compression even if target NPZ is already up to date.",
    )
    compress_parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload dataset directly to GITA via rsync over SSH after compression.",
    )
    add_upload_arguments(compress_parser)

    # upload
    upload_parser = subparsers.add_parser(
        "upload",
        help="Upload a dataset directory directly to GITA via rsync over SSH.",
    )
    upload_parser.add_argument(
        "--dataset-dir",
        "-d",
        required=True,
        type=str,
        help="Path to dataset directory to upload.",
    )
    upload_parser.add_argument(
        "--capture-mode",
        type=str,
        default=DEFAULT_CAPTURE_MODE,
        help=f"Capture mode metadata recorded in upload_log.jsonl (default: {DEFAULT_CAPTURE_MODE}).",
    )
    add_upload_arguments(upload_parser)

    # validate
    validate_parser = subparsers.add_parser(
        "validate", help="Validate dataset directory distance structure."
    )
    validate_parser.add_argument(
        "--dataset-dir",
        "-d",
        required=True,
        type=str,
        help="Path to dataset directory to validate.",
    )

    args = parser.parse_args(argv)

    if args.command == "compress":
        handle_compress_command(args)
    elif args.command == "upload":
        success, msg = upload_dataset_to_gita(
            dataset_path=args.dataset_dir,
            ssh_host=args.ssh_host,
            ssh_user=args.ssh_user,
            identity_file=args.identity,
            remote_root=args.remote_root,
            tech_subdir=args.tech_subdir,
            researcher=args.researcher,
            capture_mode=args.capture_mode,
            include_raw=not args.npz_only,
            ignore_wifi_check=args.ignore_wifi_check,
            dry_run=args.dry_run,
        )
        print(msg)
        if not success:
            raise SystemExit(1)
    elif args.command == "validate":
        folders = validate_dataset_dir(args.dataset_dir)
        print(f"Dataset directory '{args.dataset_dir}' is valid.")
        print("Distance subfolders found:")
        for folder in folders:
            print(f"  - {folder.name}")


if __name__ == "__main__":
    main()
