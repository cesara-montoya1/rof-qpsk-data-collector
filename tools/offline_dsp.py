"""100% Offline DSP pipeline for raw I/Q captures (QPSK and 16-QAM).

Reads a raw complex64 baseband I/Q file captured directly from USRP (e.g. via
qpsk_rx_v2_burst_raw.py or 16qam_rx_burst_raw.py), runs matched filtering,
AGC, Gardner timing recovery, CMA equalization, and carrier/phase recovery
offline in GNU Radio, and evaluates BER/EVM using qpsk_src or 16qam_src.
"""

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from gnuradio import analog, blocks, digital, filter as gr_filter, gr
from gnuradio.filter import firdes


def _build_constellation(mod: str):
    if mod == "qpsk":
        pts = [0.707 + 0.707j, -0.707 + 0.707j, -0.707 - 0.707j, 0.707 - 0.707j]
        return digital.constellation_rect(
            pts, [0, 1, 2, 3], 4, 2, 2, 1, 1
        ).base()
    if mod == "16qam":
        demod_16 = importlib.import_module("16qam_src.demodulator")
        pts = [complex(z) for z in demod_16.QAM16_CONSTELLATION]
        return digital.constellation_calcdist(
            pts, list(range(16)), 4, 1, digital.constellation.POWER_NORMALIZATION
        ).base()
    raise ValueError(f"Unsupported modulation: {mod}")


def recover_symbols_offline(
    raw_iq_path: Path | str,
    mod: str,
    samp_rate: float = 61.44e6,
    samp_sym: int = 4,
    beta: float = 0.35,
) -> np.ndarray:
    """Runs offline GNU Radio DSP chain on a raw complex64 I/Q file."""
    const_obj = _build_constellation(mod)
    rrc_taps = firdes.root_raised_cosine(
        1.0, samp_rate, samp_rate / samp_sym, beta, 11 * samp_sym
    )

    raw_iq = np.fromfile(raw_iq_path, dtype=np.complex64)
    if len(raw_iq) == 0:
        return np.array([], dtype=np.complex64)
    # Remove DC offset and normalize global RMS to avoid AGC startup transients
    raw_iq = raw_iq - np.mean(raw_iq)
    rms = float(np.sqrt(np.mean(np.abs(raw_iq) ** 2)))
    if rms > 1e-12:
        raw_iq = (raw_iq / rms).astype(np.complex64)

    tb = gr.top_block()
    src = blocks.vector_source_c(raw_iq.tolist(), False)
    rx_filt = gr_filter.freq_xlating_fir_filter_ccf(1, rrc_taps, 0, samp_rate)
    agc = analog.agc2_cc(1e-3, 1e-3, 1.0, 1.0)
    symsync = digital.symbol_sync_cc(
        digital.TED_GARDNER,
        samp_sym,
        2 * np.pi / 200,
        1.0,
        1.0,
        0.05,
        2,
        const_obj,
        digital.IR_PFB_NO_MF,
        64,
        [],
    )
    eq_alg = digital.adaptive_algorithm_cma(const_obj, 0.0001, 1).base()
    eq = digital.linear_equalizer(15, 2, eq_alg, True, [], "corr_est")
    snk = blocks.vector_sink_c()

    tb.connect(src, rx_filt, agc, symsync, eq)

    if mod == "qpsk":
        costas = digital.costas_loop_cc(2 * np.pi / 100, 4, False)
        tb.connect(eq, costas, snk)
    else:
        crec = digital.constellation_receiver_cb(const_obj, 2 * np.pi / 200, -0.25, 0.25)
        ns0 = blocks.null_sink(gr.sizeof_char)
        ns_f = blocks.null_sink(gr.sizeof_float)
        tb.connect(eq, crec)
        tb.connect((crec, 0), ns0)
        tb.connect((crec, 1), (ns_f, 0))
        tb.connect((crec, 2), (ns_f, 1))
        tb.connect((crec, 3), (ns_f, 2))
        tb.connect((crec, 4), snk)

    tb.run()
    return np.array(snk.data(), dtype=np.complex64)


def run_offline_dsp(
    raw_iq_path: Path | str,
    mod: str,
    tx_ref_path: Optional[Path | str] = None,
    samp_rate: float = 61.44e6,
    samp_sym: int = 4,
    beta: float = 0.35,
    skip_initial_symbols: int = 1000,
    best_window_symbols: Optional[int] = None,
    output_symbols_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Recovers symbols offline and evaluates BER/EVM against TX reference."""
    symbols = recover_symbols_offline(
        raw_iq_path=raw_iq_path,
        mod=mod,
        samp_rate=samp_rate,
        samp_sym=samp_sym,
        beta=beta,
    )
    if output_symbols_path is not None:
        out_p = Path(output_symbols_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        symbols.tofile(out_p)

    pkg = "qpsk_src" if mod == "qpsk" else "16qam_src"
    pipeline_mod = importlib.import_module(f"{pkg}.pipeline")
    demod_mod = importlib.import_module(f"{pkg}.demodulator")

    ref_path = Path(tx_ref_path) if tx_ref_path else Path(pkg) / "data" / "tx.txt"
    tx_ref = pipeline_mod.load_tx_reference(ref_path)
    metrics = demod_mod.process_signal(
        tx_ref,
        symbols,
        skip_initial_symbols=skip_initial_symbols,
        best_window_symbols=best_window_symbols,
    )
    metrics["num_recovered_symbols"] = int(len(symbols))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="100% Offline DSP for raw USRP I/Q captures (QPSK / 16-QAM)."
    )
    parser.add_argument("--mod", choices=["qpsk", "16qam"], required=True)
    parser.add_argument("--input", "-i", required=True, help="Input raw .complex64 file")
    parser.add_argument("--tx-ref", default=None, help="TX reference bit file")
    parser.add_argument("--samp-rate", type=float, default=61.44e6)
    parser.add_argument("--samp-sym", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.35)
    parser.add_argument("--skip-initial-symbols", type=int, default=1000)
    parser.add_argument("--best-window-symbols", type=int, default=None)
    parser.add_argument("--output-symbols", "-o", default=None)

    args = parser.parse_args()
    res = run_offline_dsp(
        raw_iq_path=args.input,
        mod=args.mod,
        tx_ref_path=args.tx_ref,
        samp_rate=args.samp_rate,
        samp_sym=args.samp_sym,
        beta=args.beta,
        skip_initial_symbols=args.skip_initial_symbols,
        best_window_symbols=args.best_window_symbols,
        output_symbols_path=args.output_symbols,
    )
    print(f"[{args.mod.upper()} OFFLINE DSP] Symbols: {res['num_recovered_symbols']}")
    print(f"  BER:          {res['ber']:.6f}")
    print(f"  EVM RMS (%):  {res['evm_rms_pct']:.2f}%")
    print(f"  EVM (dB):     {res['evm_db']:.2f} dB")
    print(f"  Delay:        {res['detected_delay']} | Rotation: {res['phase_rotation']}")


if __name__ == "__main__":
    main()
