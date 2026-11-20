"""Shared distance folder compressor with self-contained NPZ naming."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional
import numpy as np

DEFAULT_CAPTURE_MODE = "usrp_rx_gui"
KNOWN_CAPTURE_MODES = ("usrp_rx_gui", "usrp_rx_headless", "usrp_rx_raw")

_TECH_DIR_RE = re.compile(r"^rof_[a-z0-9]+$", re.IGNORECASE)
_ROOT_STOP_NAMES = {
    "datasets",
    "sofa_datasets",
    "results",
    "resultados",
    "hanna_maestria",
    "optnet_ml",
    "lafi_shared",
    "googledrivelafi",
}


def _slugify_segment(text: str) -> str:
    """Normalizes a path segment into lowercase snake_case suitable for filenames."""
    cleaned = text.strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    return cleaned.strip("_")


def infer_scenario_slug(distance_path: Path) -> str:
    """
    Infers a descriptive, self-contained scenario slug from the directory ancestry
    of a distance folder (e.g. ``.../rof_qpsk/uspl/nlna_step0p5db_370ma_650mhz_ch24/0km``
    -> ``rof_qpsk_uspl_nlna_step0p5db_370ma_650mhz_ch24``).

    For flat directories without a technology prefix, uses the immediate parent directory name.
    """
    parent = distance_path.resolve().parent
    parts = parent.parts

    # Prefer starting from a technology folder (e.g., rof_qpsk, rof_16qam) if present
    for idx in range(len(parts) - 1, -1, -1):
        if _TECH_DIR_RE.match(parts[idx]):
            segments = [_slugify_segment(p) for p in parts[idx:] if _slugify_segment(p)]
            if segments:
                return "_".join(segments)

    # Otherwise collect up to 2 meaningful non-root parent segments if available,
    # defaulting to the immediate parent folder name.
    parent_slug = _slugify_segment(parent.name)
    if parent_slug and parent_slug not in _ROOT_STOP_NAMES:
        return parent_slug

    return "dataset"


def build_npz_filename(
    distance_path: Path | str,
    capture_mode: str = DEFAULT_CAPTURE_MODE,
    scenario_name: Optional[str] = None,
) -> str:
    """
    Constructs a self-contained NPZ filename embedding capture mode, parent scenario,
    and distance folder name.

    Example:
        ``usrp_rx_gui_rof_qpsk_uspl_nlna_step0p5db_370ma_650mhz_ch24_0km.npz``
    """
    path = Path(distance_path)
    mode_slug = _slugify_segment(capture_mode)
    if not mode_slug:
        raise ValueError(f"Invalid capture_mode: '{capture_mode}'")

    dist_slug = _slugify_segment(path.name)
    if not dist_slug:
        raise ValueError(f"Invalid distance folder name: '{path.name}'")

    if scenario_name is not None:
        scenario_slug = _slugify_segment(scenario_name)
    else:
        scenario_slug = infer_scenario_slug(path)

    if scenario_slug:
        return f"{mode_slug}_{scenario_slug}_{dist_slug}.npz"
    return f"{mode_slug}_{dist_slug}.npz"


def compress_distance_folder(
    distance_path: Path | str,
    capture_mode: str = DEFAULT_CAPTURE_MODE,
    scenario_name: Optional[str] = None,
    force: bool = False,
) -> Optional[Path]:
    """
    Scans a distance directory for ``.complex64`` files, compresses them into a
    self-contained NPZ archive inside the distance directory, and returns its Path.

    If no ``.complex64`` files are present, returns the expected self-contained NPZ
    if present, or falls back to the first existing ``*.npz`` archive for backward
    compatibility with legacy datasets.

    Args:
        distance_path: Path to the distance directory (e.g., ``.../0km``).
        capture_mode: Acquisition/pipeline prefix (default: ``usrp_rx_gui``;
            future presets: ``usrp_rx_headless``, ``usrp_rx_raw``).
        scenario_name: Optional explicit scenario slug override.
        force: If True, always re-compress even when target NPZ is newer than sources.

    Returns:
        Path to the compressed NPZ file, or None if no signal data exists.

    Raises:
        FileNotFoundError: If distance_path does not exist.
        ValueError: If distance_path is not a directory.
    """
    path = Path(distance_path)
    if not path.exists():
        raise FileNotFoundError(f"Distance directory does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Distance path is not a directory: {path}")

    target_npz_name = build_npz_filename(
        path, capture_mode=capture_mode, scenario_name=scenario_name
    )
    target_npz_path = path / target_npz_name

    complex_files = sorted(path.glob("*.complex64"))

    if complex_files:
        if target_npz_path.exists() and not force:
            npz_mtime = target_npz_path.stat().st_mtime_ns
            if all(f.stat().st_mtime_ns <= npz_mtime for f in complex_files):
                return target_npz_path

        data_dict = {}
        for file_path in complex_files:
            signal_data = np.fromfile(file_path, dtype=np.complex64)
            data_dict[file_path.stem] = signal_data

        np.savez_compressed(target_npz_path, **data_dict)
        return target_npz_path

    if target_npz_path.exists():
        return target_npz_path

    existing_npzs = sorted(path.glob("*.npz"))
    if existing_npzs:
        return existing_npzs[0]

    return None
