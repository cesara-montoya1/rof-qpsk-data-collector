import pytest
from pathlib import Path
import sys

# Ensure optnet_ml directory is in sys.path
optnet_ml_dir = Path(__file__).resolve().parent.parent
if str(optnet_ml_dir) not in sys.path:
    sys.path.insert(0, str(optnet_ml_dir))

from qpsk_src.storage import load_existing_results, save_results_safely


def test_load_existing_results_nonexistent(tmp_path):
    csv_file = tmp_path / "non_existent.csv"
    results, processed = load_existing_results(csv_file)
    assert results == []
    assert processed == set()

    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("")
    results_empty, processed_empty = load_existing_results(empty_file)
    assert results_empty == []
    assert processed_empty == set()


def test_save_results_safely_first_write(tmp_path):
    csv_file = tmp_path / "results.csv"
    bak_file = tmp_path / "results.csv.bak"
    tmp_file = tmp_path / "results.csv.tmp"

    data = [
        {"filename": "sig1.complex64", "ber": "0.01", "osnr_db": "30.0"},
        {"filename": "sig2.complex64", "ber": "0.02", "osnr_db": "32.0"},
    ]

    save_results_safely(csv_file, data)

    assert csv_file.exists()
    assert not bak_file.exists()
    assert not tmp_file.exists()

    loaded_results, processed = load_existing_results(csv_file)
    assert len(loaded_results) == 2
    assert processed == {"sig1.complex64", "sig2.complex64"}
    assert loaded_results[0]["filename"] == "sig1.complex64"
    assert loaded_results[0]["ber"] == "0.01"


def test_save_results_safely_second_write_backup_created(tmp_path):
    csv_file = tmp_path / "results.csv"
    bak_file = tmp_path / "results.csv.bak"

    initial_data = [
        {"filename": "sig1.complex64", "ber": "0.01"},
    ]
    save_results_safely(csv_file, initial_data)
    assert not bak_file.exists()

    updated_data = [
        {"filename": "sig1.complex64", "ber": "0.01"},
        {"filename": "sig2.complex64", "ber": "0.005"},
    ]
    save_results_safely(csv_file, updated_data)

    assert bak_file.exists()
    bak_results, bak_processed = load_existing_results(bak_file)
    assert len(bak_results) == 1
    assert bak_processed == {"sig1.complex64"}

    curr_results, curr_processed = load_existing_results(csv_file)
    assert len(curr_results) == 2
    assert curr_processed == {"sig1.complex64", "sig2.complex64"}
