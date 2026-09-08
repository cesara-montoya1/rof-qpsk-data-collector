#!/usr/bin/env python3
"""
Convert all .complex64 files in the current directory to a single .npz file.
Each file is stored as a separate array in the npz archive.
"""

import glob
import numpy as np
import os
from pathlib import Path


def load_file(filepath):
    """Load a file as a numpy array."""
    try:
        data = np.fromfile(filepath, dtype=np.complex64)
        return data
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")


def main():
    # Find all .complex64 files in current directory
    files = sorted(glob.glob("*.complex64"))

    if not files:
        print("No .complex64 files found in current directory")
        return

    print(f"Found {len(files)} .complex64 files")

    # Dictionary to store all arrays
    data_dict = {}

    # Load each bin file
    for file in files:
        # Use filename (without extension) as key
        key = Path(file).stem
        print(f"Loading {file}...")

        try:
            data = load_file(file)
            data_dict[key] = data
            print(
                f"  Loaded {len(data)} values, shape: {data.shape}, dtype: {data.dtype}"
            )
        except Exception as e:
            print(f"  Error loading {file}: {e}")

    # Save to npz file
    output_file = "all_data.npz"
    print(f"\nSaving to {output_file}...")
    np.savez_compressed(output_file, **data_dict)

    # Print summary
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Done! Saved {len(data_dict)} arrays to {output_file}")
    print(f"Output file size: {file_size_mb:.2f} MB")
    print("\nTo load the data:")
    print("  data = np.load({output_file})")
    print("  # Access individual arrays: data['array_name']")
    print("  # List all arrays: list(data.keys())")


if __name__ == "__main__":
    main()
