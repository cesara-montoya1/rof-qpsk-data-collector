"""Signal metadata parser for QPSK signal filenames."""

import re
from pathlib import Path
from typing import Any, Callable, Dict, Union


def extract_value(
    pattern: str,
    text: str,
    conversion_func: Callable[[str], Any] = float,
    default: Any = ...,
) -> Any:
    """
    Extracts a value matching a regex pattern from text, replacing 'p'/'P' with '.',
    and converts it with conversion_func.

    Args:
        pattern: Regex pattern containing a capture group or full match.
        text: String containing metadata (e.g., filename).
        conversion_func: Conversion function to apply to extracted string (default float).
        default: Default value if pattern is not found. If omitted, raises ValueError.

    Returns:
        Converted value or default.

    Raises:
        ValueError: If pattern is not found in text and no default is provided.
    """
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        if default is not ...:
            return default
        raise ValueError(f"Pattern '{pattern}' not found in text: '{text}'")

    matched_str = match.group(1) if match.groups() else match.group(0)
    cleaned_str = matched_str.replace("p", ".").replace("P", ".")
    return conversion_func(cleaned_str)


def parse_filename_metadata(
    filename: Union[Path, str], fallback_on_missing: bool = False
) -> Dict[str, Any]:
    """
    Extracts metadata dictionary from a signal filename.

    Extracted fields:
        - launch_power_dbm (float)
        - distance_km (float)
        - osnr_db (float)
        - snr_db (float)
        - freq_mhz (float)
        - bitrate_mbps (float)

    Args:
        filename: Path or filename string.
        fallback_on_missing: If True, missing fields return float('nan') instead of raising ValueError.

    Returns:
        Dict[str, Any] containing metadata values.
    """
    name = Path(filename).name
    default_val = float("nan") if fallback_on_missing else ...

    return {
        "launch_power_dbm": extract_value(r"(-?\d+(?:[pP]\d+)?)dBm", name, default=default_val),
        "distance_km": extract_value(r"(\d+(?:[pP]\d+)?)km", name, default=default_val),
        "osnr_db": extract_value(r"osnr(\d+(?:[pP\.]\d+)?)dB", name, default=default_val),
        "snr_db": extract_value(r"(?<![oO])snr(\d+(?:[pP\.]\d+)?)dB", name, default=default_val),
        "freq_mhz": extract_value(r"(\d+(?:[pP]\d+)?)mhz", name, default=default_val),
        "bitrate_mbps": extract_value(r"(\d+(?:[pP]\d+)?)mbps", name, default=default_val),
    }

