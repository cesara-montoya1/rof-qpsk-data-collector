"""Command-line interface for 16-QAM signal processing pipeline."""

import argparse
from pathlib import Path
import numpy as np

from .compressor import compress_distance_folder
from .pipeline import load_tx_reference, run_pipeline
from .plotter import (
    generate_all_plots,
    generate_ber_plots,
    generate_evm_plots,
    plot_constellation,
)
from .validation import validate_dataset_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="16-QAM Signal Processing and Demodulation Pipeline CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: run
    run_parser = subparsers.add_parser("run", help="Run the 16-QAM demodulation pipeline.")
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
        default="16qam_src/data/tx.txt",
        help="Path to reference TX bit file (default: 16qam_src/data/tx.txt).",
    )
    run_parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to output CSV file (default: <dataset_dir>/results_<dataset_name>.csv).",
    )
    run_parser.add_argument(
        "--skip-initial-symbols",
        type=int,
        default=0,
        help="Number of initial symbols to skip for warm-up / settling time (default: 0).",
    )
    run_parser.add_argument(
        "--best-window-symbols",
        type=int,
        default=None,
        help="Contiguous symbol block size to find best-case EVM window (default: None).",
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
    plot_parser.add_argument(
        "--metric",
        "-m",
        type=str,
        choices=["all", "ber", "evm"],
        default="all",
        help="Metric to plot: 'all' (default), 'ber', or 'evm'.",
    )
    plot_parser.add_argument(
        "--snr-bin-width",
        type=float,
        default=None,
        help="Bin width in dB to aggregate SNR measurements in BER/EVM vs SNR plots (default: auto).",
    )
    plot_parser.add_argument(
        "--include-diagnostics",
        action="store_true",
        help="Also generate advanced channel diagnostic plots (e.g. BER vs EVM theoretical bound).",
    )
    plot_parser.add_argument(
        "--diagnostics-subfolder",
        type=str,
        default="diagnostics",
        help="Subdirectory for extra diagnostic plots (default: 'diagnostics'). Set to '' to save directly in the regular plots folder.",
    )

    # Subcommand: diagnostics
    diag_parser = subparsers.add_parser(
        "diagnostics", help="Generate advanced channel diagnostics (polar EVM, spectrum, density, BER vs EVM)."
    )
    diag_parser.add_argument(
        "--csv-path",
        "-c",
        type=str,
        default=None,
        help="Path to results CSV file for dataset diagnostics.",
    )
    diag_parser.add_argument(
        "--rx",
        type=str,
        default=None,
        help="Path to raw RX complex64 file for single-signal diagnostics.",
    )
    diag_parser.add_argument(
        "--tx-ref",
        type=str,
        default="16qam_src/data/tx.txt",
        help="Path to reference TX bit file (default: 16qam_src/data/tx.txt).",
    )
    diag_parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Directory to save diagnostic plots (default: <data_dir>/plots).",
    )
    diag_parser.add_argument(
        "--subfolder",
        type=str,
        default="diagnostics",
        help="Subfolder inside output-dir (default: 'diagnostics'). Pass '' to save directly in the main folder.",
    )
    diag_parser.add_argument(
        "--skip-initial-symbols",
        type=int,
        default=0,
        help="Number of initial symbols to skip for warm-up (default: 0).",
    )
    diag_parser.add_argument(
        "--best-window-symbols",
        type=int,
        default=None,
        help="Block size (M) for best-case EVM window (default: None).",
    )

    # Subcommand: constellation
    constellation_parser = subparsers.add_parser(
        "constellation", help="Quick transmission check: plot constellation, BER and EVM for a single .complex64 file."
    )
    constellation_parser.add_argument(
        "rx_file",
        nargs="?",
        default=None,
        type=str,
        help="Path to raw RX complex64 file (positional argument).",
    )
    constellation_parser.add_argument(
        "--rx",
        "-r",
        type=str,
        default=None,
        help="Path to raw RX complex64 file.",
    )
    constellation_parser.add_argument(
        "--tx-ref",
        "-t",
        type=str,
        default="16qam_src/data/tx.txt",
        help="Path to reference TX bit file (default: 16qam_src/data/tx.txt).",
    )
    constellation_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to save output constellation plot PNG (default: None, saves temporarily to /tmp/ to keep workspace clean).",
    )
    constellation_parser.add_argument(
        "--skip-initial-symbols",
        type=int,
        default=0,
        help="Number of initial symbols to skip for warm-up / settling time (default: 0).",
    )
    constellation_parser.add_argument(
        "--best-window-symbols",
        type=int,
        default=None,
        help="Contiguous symbol block size to find best-case EVM window (default: None).",
    )
    constellation_parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display interactive plot window (headless or batch mode).",
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
    compress_parser.add_argument(
        "--upload",
        action="store_true",
        help="Trigger immediate Google Drive synchronization after compression.",
    )
    compress_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate compression and upload without modifying remote files.",
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
            skip_initial_symbols=args.skip_initial_symbols,
            best_window_symbols=args.best_window_symbols,
        )
        print(f"Pipeline executed successfully. Output saved to: {output_csv}")

    elif args.command == "plot":
        from .diagnostics import generate_dataset_diagnostics

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

        if args.metric == "ber":
            plots = generate_ber_plots(
                csv_path=csv_file, output_dir=args.output_dir, snr_bin_width=args.snr_bin_width
            )
        elif args.metric == "evm":
            plots = generate_evm_plots(
                csv_path=csv_file, output_dir=args.output_dir, snr_bin_width=args.snr_bin_width
            )
        else:
            plots = generate_all_plots(
                csv_path=csv_file, output_dir=args.output_dir, snr_bin_width=args.snr_bin_width
            )

        if args.include_diagnostics:
            diag_plots = generate_dataset_diagnostics(
                csv_path=csv_file,
                output_dir=args.output_dir,
                subfolder=args.diagnostics_subfolder,
            )
            plots.extend(diag_plots)

        print(f"Generated {len(plots)} plot figures ({args.metric}) from {csv_file}:")
        for p in plots:
            print(f"  - {p}")

    elif args.command == "diagnostics":
        from .diagnostics import generate_dataset_diagnostics, generate_signal_diagnostics

        generated_plots = []
        if args.csv_path:
            generated_plots.extend(
                generate_dataset_diagnostics(
                    csv_path=args.csv_path,
                    output_dir=args.output_dir,
                    subfolder=args.subfolder,
                )
            )

        if args.rx:
            tx_symbols = load_tx_reference(args.tx_ref)
            rx_samples = np.fromfile(args.rx, dtype=np.complex64)
            generated_plots.extend(
                generate_signal_diagnostics(
                    rx_signal=rx_samples,
                    tx_ref=tx_symbols,
                    output_dir=args.output_dir,
                    subfolder=args.subfolder,
                    skip_initial_symbols=args.skip_initial_symbols,
                    best_window_symbols=args.best_window_symbols,
                )
            )

        if not args.csv_path and not args.rx:
            raise ValueError("Must provide either --csv-path or --rx to run diagnostics.")

        dest_desc = f"subfolder '{args.subfolder}'" if args.subfolder else "main directory"
        print(f"Generated {len(generated_plots)} channel diagnostic plots in {dest_desc}:")
        for p in generated_plots:
            print(f"  - {p}")

    elif args.command == "constellation":
        rx_target = args.rx_file or args.rx
        if not rx_target:
            constellation_parser.error(
                "Debe especificar la ruta al archivo .complex64 (ej: pixi run check-signal-16qam muestra.complex64)"
            )

        rx_path = Path(rx_target)
        if not rx_path.exists():
            raise FileNotFoundError(f"Archivo de muestra no encontrado: {rx_path}")

        tx_ref_path = Path(args.tx_ref)
        if not tx_ref_path.exists():
            candidate = Path(__file__).resolve().parent / "data" / "tx.txt"
            if candidate.exists():
                tx_ref_path = candidate
            else:
                raise FileNotFoundError(f"Archivo de referencia TX no encontrado: {tx_ref_path}")

        tx_symbols = load_tx_reference(tx_ref_path)
        rx_samples = np.fromfile(rx_path, dtype=np.complex64)

        is_temp_output = args.output is None
        if is_temp_output:
            import tempfile

            out_file = Path(tempfile.gettempdir()) / f"16qam_constellation_{rx_path.stem}.png"
        else:
            out_file = Path(args.output)

        out_path, metrics = plot_constellation(
            rx_signal=rx_samples,
            tx_ref=tx_symbols,
            output_file=out_file,
            skip_initial_symbols=args.skip_initial_symbols,
            best_window_symbols=args.best_window_symbols,
            show=not args.no_show,
            return_metrics=True,
        )

        status_tag = "SIN ERRORES ✅" if metrics["ber"] == 0.0 else f"BER: {metrics['ber']:.2e}"
        n_samples = len(rx_samples)
        n_symbols_eval = metrics["total_bits"] // 4 if metrics["total_bits"] > 0 else n_samples
        print("\n" + "=" * 60)
        print("         REVISIÓN RÁPIDA DE TRANSMISIÓN 16-QAM         ")
        print("=" * 60)
        print(f"Archivo:        {rx_path.name}")
        print(f"Muestras:       {n_samples:,} ({n_symbols_eval:,} símbolos evaluados)")
        print(f"Retardo:        {metrics['detected_delay']:,} muestras | Rotación: {metrics['best_rotation']}")
        print("-" * 60)
        print(f"BER:            {metrics['ber']:.6f} ({metrics['bit_errors']:,} / {metrics['total_bits']:,} bits) [{status_tag}]")
        print(f"EVM (RMS):      {metrics['evm_rms_pct']:.2f} % ({metrics['evm_db']:.2f} dB)")
        print(f"EVM (Pico):     {metrics['evm_peak_pct']:.2f} %")
        print("-" * 60)
        if is_temp_output:
            print(f"Gráfica temporal: {out_path}")
            print("Tip: Para guardarla permanentemente use: -o mi_grafica.png")
        else:
            print(f"Gráfica guardada en: {out_path}")
        print("=" * 60 + "\n")

    elif args.command == "compress":
        from .uploader import trigger_gdrive_sync, validate_results_path

        raw_path = Path(args.dataset_dir)
        if args.upload:
            path = validate_results_path(raw_path)
        else:
            path = raw_path

        compressed_count = 0
        try:
            distance_folders = validate_dataset_dir(path)
            for folder in distance_folders:
                npz = compress_distance_folder(folder)
                if npz:
                    compressed_count += 1
                    print(f"Compressed {folder.name} -> {npz}")
                else:
                    print(f"No files compressed in {folder.name}")
        except (ValueError, FileNotFoundError):
            npz = compress_distance_folder(path)
            if npz:
                compressed_count += 1
                print(f"Compressed {path.name} -> {npz}")
            else:
                print(f"No files to compress in {path}")

        if args.upload:
            print("\nIniciando sincronización con Google Drive (lafi:)...")
            success, sync_msg = trigger_gdrive_sync(dry_run=args.dry_run)
            print(sync_msg)
            if not success:
                raise SystemExit(1)
            print("\n¡Proceso completado exitosamente!")

    elif args.command == "validate":
        folders = validate_dataset_dir(args.dataset_dir)
        print(f"Dataset directory '{args.dataset_dir}' is valid.")
        print("Distance subfolders found:")
        for f in folders:
            print(f"  - {f.name}")


if __name__ == "__main__":
    main()
