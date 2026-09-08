import pytest
from pathlib import Path
import sys

# Ensure optnet_ml directory is in sys.path
optnet_ml_dir = Path(__file__).resolve().parent.parent
if str(optnet_ml_dir) not in sys.path:
    sys.path.insert(0, str(optnet_ml_dir))

from qpsk_src.metadata import extract_value, parse_filename_metadata


def test_parse_filename_metadata_example():
    filename = "rof_dfb_0dBm_20km_osnr32p50dB_900mhz_1p88mbps_snr11p40dB.complex64"
    metadata = parse_filename_metadata(filename)

    assert metadata["launch_power_dbm"] == pytest.approx(0.0)
    assert metadata["distance_km"] == pytest.approx(20.0)
    assert metadata["osnr_db"] == pytest.approx(32.50)
    assert metadata["snr_db"] == pytest.approx(11.40)
    assert metadata["freq_mhz"] == pytest.approx(900.0)
    assert metadata["bitrate_mbps"] == pytest.approx(1.88)


def test_parse_filename_metadata_variations():
    filename_with_path = "/path/to/data/rof_0dBm_0km_osnr32p82dB_650mhz_2mbps_snr3p38dB.complex64"
    metadata = parse_filename_metadata(filename_with_path)

    assert metadata["launch_power_dbm"] == pytest.approx(0.0)
    assert metadata["distance_km"] == pytest.approx(0.0)
    assert metadata["osnr_db"] == pytest.approx(32.82)
    assert metadata["snr_db"] == pytest.approx(3.38)
    assert metadata["freq_mhz"] == pytest.approx(650.0)
    assert metadata["bitrate_mbps"] == pytest.approx(2.0)

    filename_negative_pwr = "rof_dfb_-5dBm_0p5km_osnr15p0dB_1200mhz_10mbps_snr8p5dB.complex64"
    metadata_neg = parse_filename_metadata(filename_negative_pwr)
    assert metadata_neg["launch_power_dbm"] == pytest.approx(-5.0)
    assert metadata_neg["distance_km"] == pytest.approx(0.5)


def test_extract_value_custom_and_errors():
    assert extract_value(r"(\d+)km", "20km", int) == 20
    assert extract_value(r"(\d+(?:p\d+)?)km", "0p5km", float) == pytest.approx(0.5)

    with pytest.raises(ValueError, match="Pattern '.*' not found"):
        extract_value(r"(\d+)GHz", "900MHz")
