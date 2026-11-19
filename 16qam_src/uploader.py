"""Backward-compatible re-export shim for shared GITA uploader."""

from common.uploader import (
    BLOCKED_WIFI_SSIDS,
    DEFAULT_GITA_HOST,
    DEFAULT_GITA_PORT,
    DEFAULT_GITA_REMOTE_ROOT,
    build_ssh_transport_args,
    check_gita_network_access,
    detect_active_wifi_ssid,
    has_active_ethernet,
    infer_remote_relative_path,
    resolve_ssh_user,
    upload_dataset_to_gita,
    validate_dataset_upload_path,
    validate_results_path,
)

__all__ = [
    "BLOCKED_WIFI_SSIDS",
    "DEFAULT_GITA_HOST",
    "DEFAULT_GITA_PORT",
    "DEFAULT_GITA_REMOTE_ROOT",
    "build_ssh_transport_args",
    "check_gita_network_access",
    "detect_active_wifi_ssid",
    "has_active_ethernet",
    "infer_remote_relative_path",
    "resolve_ssh_user",
    "upload_dataset_to_gita",
    "validate_dataset_upload_path",
    "validate_results_path",
]
