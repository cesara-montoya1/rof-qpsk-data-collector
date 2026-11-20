"""Tests for self-contained NPZ compression and backward compatibility."""

from pathlib import Path
import sys
import numpy as np
import pytest

optnet_ml_dir = Path(__file__).resolve().parent.parent
if str(optnet_ml_dir) not in sys.path:
    sys.path.insert(0, str(optnet_ml_dir))

from common.compressor import build_npz_filename, compress_distance_folder, infer_scenario_slug
from qpsk_src.compressor import compress_distance_folder as qpsk_compress


def test_build_npz_filename_with_technology_ancestry(tmp_path: Path):
    dist_dir = tmp_path / "sofa_datasets" / "rof_qpsk" / "uspl" / "nlna_step0p5db_370ma_650mhz_ch24" / "0km"
    dist_dir.mkdir(parents=True)

    slug = infer_scenario_slug(dist_dir)
    assert slug == "rof_qpsk_uspl_nlna_step0p5db_370ma_650mhz_ch24"

    fname = build_npz_filename(dist_dir)
    assert fname == "usrp_rx_gui_rof_qpsk_uspl_nlna_step0p5db_370ma_650mhz_ch24_0km.npz"

    raw_fname = build_npz_filename(dist_dir, capture_mode="usrp_rx_raw")
    assert raw_fname == "usrp_rx_raw_rof_qpsk_uspl_nlna_step0p5db_370ma_650mhz_ch24_0km.npz"


def test_compress_distance_folder_with_complex64(tmp_path: Path):
    exp_dir = tmp_path / "rof_qpsk" / "dfb" / "baseline_650mhz"
    dist_dir = exp_dir / "20km"
    dist_dir.mkdir(parents=True)

    sig1 = np.array([1 + 2j, 3 - 4j, 5 + 6j], dtype=np.complex64)
    sig2 = np.array([0 + 1j, -1 + 0j, 2 - 2j], dtype=np.complex64)

    file1 = dist_dir / "rof_0dBm_20km_osnr30dB_900mhz_2mbps_snr10dB.complex64"
    file2 = dist_dir / "rof_0dBm_20km_osnr31dB_900mhz_2mbps_snr11dB.complex64"

    sig1.tofile(file1)
    sig2.tofile(file2)

    result_npz = compress_distance_folder(dist_dir)

    assert result_npz == dist_dir / "usrp_rx_gui_rof_qpsk_dfb_baseline_650mhz_20km.npz"
    assert result_npz.exists()

    with np.load(result_npz) as npz_data:
        assert set(npz_data.files) == {file1.stem, file2.stem}
        np.testing.assert_array_equal(npz_data[file1.stem], sig1)
        np.testing.assert_array_equal(npz_data[file2.stem], sig2)


def test_compress_distance_folder_skips_when_npz_is_newer(tmp_path: Path):
    dist_dir = tmp_path / "exp_test" / "5km"
    dist_dir.mkdir(parents=True)
    file1 = dist_dir / "sig1.complex64"
    file1.write_bytes(b"\x00" * 16)

    npz1 = qpsk_compress(dist_dir)
    assert npz1 is not None
    mtime1 = npz1.stat().st_mtime_ns

    npz2 = qpsk_compress(dist_dir)
    assert npz2 == npz1
    assert npz2.stat().st_mtime_ns == mtime1


def test_compress_distance_folder_legacy_npz_fallback(tmp_path: Path):
    dist_dir = tmp_path / "10km"
    dist_dir.mkdir()

    legacy_npz = dist_dir / "data_10km.npz"
    np.savez_compressed(legacy_npz, sample=np.array([1, 2, 3]))

    result_npz = compress_distance_folder(dist_dir)
    assert result_npz == legacy_npz


def test_compress_distance_folder_empty(tmp_path: Path):
    dist_dir = tmp_path / "0km"
    dist_dir.mkdir()

    result = compress_distance_folder(dist_dir)
    assert result is None


def test_compress_distance_folder_invalid_path(tmp_path: Path):
    non_existent = tmp_path / "non_existent"
    with pytest.raises(FileNotFoundError):
        compress_distance_folder(non_existent)

    sample_file = tmp_path / "file.txt"
    sample_file.write_text("dummy", encoding="utf-8")
    with pytest.raises(ValueError):
        compress_distance_folder(sample_file)
