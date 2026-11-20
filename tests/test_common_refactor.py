"""Tests confirming shared modules in common/ are cleanly re-exported by qpsk_src and 16qam_src."""

import importlib

import common.compressor
import common.evm
import common.generate_tx
import common.metadata
import common.runner
import common.storage
import common.uploader
import common.validation


def test_qpsk_and_16qam_share_common_implementations():
    qpsk_pkg = importlib.import_module("qpsk_src")
    qam_pkg = importlib.import_module("16qam_src")

    assert qpsk_pkg.validate_dataset_dir is common.validation.validate_dataset_dir
    assert qam_pkg.validate_dataset_dir is common.validation.validate_dataset_dir

    assert qpsk_pkg.parse_filename_metadata is common.metadata.parse_filename_metadata
    assert qam_pkg.parse_filename_metadata is common.metadata.parse_filename_metadata

    assert qpsk_pkg.load_existing_results is common.storage.load_existing_results
    assert qam_pkg.load_existing_results is common.storage.load_existing_results

    assert qpsk_pkg.compute_evm is common.evm.compute_evm
    assert qam_pkg.compute_evm is common.evm.compute_evm

    assert qpsk_pkg.compress_distance_folder is common.compressor.compress_distance_folder
    assert qam_pkg.compress_distance_folder is common.compressor.compress_distance_folder

    qpsk_up = importlib.import_module("qpsk_src.uploader")
    qam_up = importlib.import_module("16qam_src.uploader")
    assert qpsk_up.upload_dataset_to_gita is common.uploader.upload_dataset_to_gita
    assert qam_up.upload_dataset_to_gita is common.uploader.upload_dataset_to_gita

    qpsk_gen = importlib.import_module("qpsk_src.tools.generate_tx")
    qam_gen = importlib.import_module("16qam_src.tools.generate_tx")
    assert qpsk_gen.generate_prbs_lfsr is common.generate_tx.generate_prbs_lfsr
    assert qam_gen.generate_prbs_lfsr is common.generate_tx.generate_prbs_lfsr
