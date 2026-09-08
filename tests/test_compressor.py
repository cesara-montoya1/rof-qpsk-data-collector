import pytest
from pathlib import Path
import numpy as np
import sys

# Ensure optnet_ml directory is in sys.path
optnet_ml_dir = Path(__file__).resolve().parent.parent
if str(optnet_ml_dir) not in sys.path:
    sys.path.insert(0, str(optnet_ml_dir))

from qpsk_src.compressor import compress_distance_folder


def test_compress_distance_folder_with_complex64(tmp_path):
    dist_dir = tmp_path / "20km"
    dist_dir.mkdir()

    sig1 = np.array([1 + 2j, 3 - 4j, 5 + 6j], dtype=np.complex64)
    sig2 = np.array([0 + 1j, -1 + 0j, 2 - 2j], dtype=np.complex64)

    file1 = dist_dir / "rof_0dBm_20km_osnr30dB_900mhz_2mbps_snr10dB.complex64"
    file2 = dist_dir / "rof_0dBm_20km_osnr31dB_900mhz_2mbps_snr11dB.complex64"

    sig1.tofile(file1)
    sig2.tofile(file2)

    result_npz = compress_distance_folder(dist_dir)

    assert result_npz == dist_dir / "data_20km.npz"
    assert result_npz.exists()

    with np.load(result_npz) as npz_data:
        assert set(npz_data.files) == {file1.stem, file2.stem}
        np.testing.assert_array_equal(npz_data[file1.stem], sig1)
        np.testing.assert_array_equal(npz_data[file2.stem], sig2)


def test_compress_distance_folder_existing_npz_no_complex(tmp_path):
    dist_dir = tmp_path / "10km"
    dist_dir.mkdir()

    target_npz = dist_dir / "data_10km.npz"
    np.savez_compressed(target_npz, sample=np.array([1, 2, 3]))

    result_npz = compress_distance_folder(dist_dir)
    assert result_npz == target_npz


def test_compress_distance_folder_empty(tmp_path):
    dist_dir = tmp_path / "0km"
    dist_dir.mkdir()

    result = compress_distance_folder(dist_dir)
    assert result is None


def test_compress_distance_folder_invalid_path(tmp_path):
    non_existent = tmp_path / "non_existent"
    with pytest.raises(FileNotFoundError):
        compress_distance_folder(non_existent)

    sample_file = tmp_path / "file.txt"
    sample_file.write_text("dummy")
    with pytest.raises(ValueError):
        compress_distance_folder(sample_file)
