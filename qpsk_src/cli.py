"""Command-line interface for QPSK signal processing pipeline."""

import argparse
from pathlib import Path

from .compressor import compress_distance_folder
from .pipeline import run_pipeline
from .plotter import generate_ber_plots
from .validation import validate_dataset_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QPSK Signal Processing and Demodulation Pipeline CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: run
    run_parser = subparsers.add_parser("run", help="Run the QPSK demodulation pipeline.")
    run_parser.add_argument(
        "--dataset-dir",
        "-d",
        required=True,
        type=str,
        help="Path to dataset directory containing distance folders.",
    )
    run_parser.add_argument(
        "--tx-ref",
        type=str,
        default="qpsk_src/data/tx.txt",
        help="Path to reference TX bit file (default: qpsk_src/data/tx.txt).",
    )
    run_parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to output CSV file (default: <dataset_dir>/results_<dataset_name>.csv).",
    )

    # Subcommand: plot
    plot_parser = subparsers.add_parser(
        "plot", help="Generate BER vs SNR and BER vs OSNR plot figures."
    )
    plot_parser.add_argument(
        "--csv-path",
        "-c",
        type=str,
        default=None,
        help="Path to input results CSV file.",
    )
    plot_parser.add_argument(
        "--dataset-dir",
        "-d",
        type=str,
        default=None,
        help="Path to dataset directory (auto-detects results_*.csv inside).",
    )
    plot_parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Directory to save output PNG plots (default: <csv_dir>/plots/).",
    )

    # Subcommand: compress
    compress_parser = subparsers.add_parser(
        "compress", help="Compress raw complex64 files into NPZ archives."
    )
    compress_parser.add_argument(
        "--dataset-dir",
        "-d",
        required=True,
        type=str,
        help="Path to dataset directory or distance folder to compress.",
    )

    # Subcommand: validate
    validate_parser = subparsers.add_parser(
        "validate", help="Validate dataset directory structure."
    )
    validate_parser.add_argument(
        "--dataset-dir",
        "-d",
        required=True,
        type=str,
        help="Path to dataset directory to validate.",
    )

    args = parser.parse_args()

    if args.command == "run":
        output_csv = run_pipeline(
            dataset_dir=args.dataset_dir,
            tx_ref_path=args.tx_ref,
            output_csv_path=args.output_csv,
        )
        print(f"Pipeline executed successfully. Output saved to: {output_csv}")

    elif args.command == "plot":
        csv_file = None
        if args.csv_path:
            csv_file = Path(args.csv_path)
        elif args.dataset_dir:
            ds_dir = Path(args.dataset_dir)
            matches = list(ds_dir.glob("results_*.csv")) + list(ds_dir.glob("*.csv"))
            if matches:
                csv_file = matches[0]
            else:
                raise FileNotFoundError(f"No results CSV file found in: {ds_dir}")
        else:
            raise ValueError("Must specify either --csv-path or --dataset-dir to generate plots.")

        plots = generate_ber_plots(csv_path=csv_file, output_dir=args.output_dir)
        print(f"Generated {len(plots)} plot figures from {csv_file}:")
        for p in plots:
            print(f"  - {p}")

    elif args.command == "compress":
        path = Path(args.dataset_dir)
        try:
            distance_folders = validate_dataset_dir(path)
            for folder in distance_folders:
                npz = compress_distance_folder(folder)
                if npz:
                    print(f"Compressed {folder.name} -> {npz}")
                else:
                    print(f"No files compressed in {folder.name}")
        except (ValueError, FileNotFoundError):
            npz = compress_distance_folder(path)
            if npz:
                print(f"Compressed {path.name} -> {npz}")
            else:
                print(f"No files to compress in {path}")

    elif args.command == "validate":
        folders = validate_dataset_dir(args.dataset_dir)
        print(f"Dataset directory '{args.dataset_dir}' is valid.")
        print("Distance subfolders found:")
        for f in folders:
            print(f"  - {f.name}")


if __name__ == "__main__":
    main()
