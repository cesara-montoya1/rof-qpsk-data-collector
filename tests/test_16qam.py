"""Comprehensive unit and GNU Radio integration tests for the 16qam_src package."""

import csv
import importlib
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest

demod = importlib.import_module("16qam_src.demodulator")
pipeline = importlib.import_module("16qam_src.pipeline")
plotter = importlib.import_module("16qam_src.plotter")
diagnostics = importlib.import_module("16qam_src.diagnostics")
file_manager = importlib.import_module("16qam_src.src.file_manager")


def test_constellation_properties():
    """Verify unit average power, 16 unique points, 90-deg symmetry, and Gray coding."""
    pts = demod.QAM16_CONSTELLATION
    assert len(pts) == 16
    assert len(np.unique(np.round(pts, 8))) == 16
    assert np.isclose(np.mean(np.abs(pts) ** 2), 1.0, atol=1e-9)

    # 90-degree rotational closure
    rot90 = pts * 1j
    for z in rot90:
        assert np.min(np.abs(pts - z)) < 1e-9

    # Gray-coding check: nearest neighbors (minimum Euclidean distance) differ by 1 bit
    d_min = 2.0 * demod.INV_SQRT10
    for i in range(16):
        for j in range(i + 1, 16):
            if np.isclose(abs(pts[i] - pts[j]), d_min, atol=1e-9):
                xor_bits = i ^ j
                assert bin(xor_bits).count("1") == 1


def test_modulate_demodulate_roundtrip_and_edge_cases():
    """Verify 16-QAM Gray round-trip, bit padding, and empty input handling."""
    rng = np.random.default_rng(123)
    bits = rng.integers(0, 2, size=4000, dtype=int)
    syms = demod.modulate_16qam(bits)
    assert len(syms) == 1000
    assert np.array_equal(demod.demodulate_16qam(syms), bits)

    # Non-multiple of 4 bits pads with zeros
    odd_bits = np.array([1, 0, 1], dtype=int)
    padded_syms = demod.modulate_16qam(odd_bits)
    assert len(padded_syms) == 1
    assert np.array_equal(demod.demodulate_16qam(padded_syms), np.array([1, 0, 1, 0]))

    # Empty inputs
    assert len(demod.modulate_16qam(np.array([], dtype=int))) == 0
    assert len(demod.demodulate_16qam(np.array([], dtype=np.complex128))) == 0


def test_threshold_demodulator_matches_euclidean_distance():
    """Verify axis threshold demodulator matches exact nearest-neighbor symbol decision."""
    grid = np.linspace(-1.4, 1.4, 29)
    # Exclude exact boundaries (0 and +/- 2/sqrt(10)) where tie-breaking can differ
    samples = np.array(
        [complex(x + 1e-4, y - 1e-4) for x in grid for y in grid], dtype=np.complex128
    )
    threshold_bits = demod.demodulate_16qam(samples).reshape(-1, 4)

    dists = np.abs(samples[:, None] - demod.QAM16_CONSTELLATION[None, :])
    nearest_idx = np.argmin(dists, axis=1)
    euclid_bits = np.column_stack(
        [
            (nearest_idx >> 3) & 1,
            (nearest_idx >> 2) & 1,
            (nearest_idx >> 1) & 1,
            nearest_idx & 1,
        ]
    )
    assert np.array_equal(threshold_bits, euclid_bits)


@pytest.mark.parametrize("rotation", [1, 1j, -1, -1j])
@pytest.mark.parametrize("scale", [0.2, 5.0])
def test_process_signal_delay_rotations_scale_and_windowing(rotation, scale):
    """Verify delay detection, 4-fold rotation recovery, scaling normalization, and EVM."""
    rng = np.random.default_rng(7)
    tx_bits = rng.integers(0, 2, size=8000, dtype=int)
    tx_ref = demod.modulate_16qam(tx_bits)

    delay = 37
    noise = 0.015 * (rng.normal(size=len(tx_ref)) + 1j * rng.normal(size=len(tx_ref)))
    rx_clean = np.concatenate(
        [np.zeros(delay, dtype=np.complex128), tx_ref * rotation + noise]
    ) * scale
    # Inject noisy warm-up symbols before valid window
    rx_clean[:25] = 2.5 + 2.5j

    res = demod.process_signal(
        tx_ref,
        rx_clean,
        skip_initial_symbols=100,
        best_window_symbols=500,
    )
    assert res["ber"] == 0.0
    assert res["detected_delay"] == delay
    assert np.isclose(abs(res["phase_rotation"] * rotation - 1.0), 0.0) or np.isclose(
        abs(res["phase_rotation"] - np.conj(rotation)), 0.0
    )
    assert res["evm_rms_pct"] < 4.0
    assert res["evm_best_rms_pct"] is not None
    assert res["evm_best_rms_pct"] <= res["evm_rms_pct"] + 1e-6
    assert res["best_window_start"] is not None


def test_load_tx_reference_multiple_and_non_multiple_of_4(tmp_path):
    """Verify TX reference loading handles both mod-4==0 and PRBS lengths not divisible by 4."""
    # Length divisible by 4 (64 bits -> 16 symbols)
    p_even = tmp_path / "tx_even.txt"
    p_even.write_text("0110" * 16, encoding="utf-8")
    syms_even = pipeline.load_tx_reference(p_even)
    assert len(syms_even) == 16

    # Prime length not divisible by 4 (e.g. 127 bits PRBS-7 -> gcd(127, 4)=1 -> 4x tiled -> 127 symbols)
    rng = np.random.default_rng(99)
    prbs_bits = "".join(str(b) for b in rng.integers(0, 2, size=127))
    p_odd = tmp_path / "tx_odd.txt"
    p_odd.write_text(prbs_bits, encoding="utf-8")
    syms_odd = pipeline.load_tx_reference(p_odd)
    assert len(syms_odd) == 127

    # All 4 wrap-around bit alignments (offsets 0..3 into repeated bitstream) must appear verbatim
    repeated_bits = (prbs_bits * 8)[: len(prbs_bits) * 4]
    decoded_ref_bits = "".join(str(b) for b in demod.demodulate_16qam(syms_odd))
    assert decoded_ref_bits == (prbs_bits * 4)


def test_run_pipeline_synthetic_dataset(tmp_path):
    """Verify full offline 16-QAM pipeline executes end-to-end and logs expected CSV metrics."""
    rng = np.random.default_rng(21)
    bits = rng.integers(0, 2, size=4096, dtype=int)
    tx_path = tmp_path / "tx.txt"
    tx_path.write_text("".join(str(b) for b in bits), encoding="utf-8")

    tx_syms = demod.modulate_16qam(bits)
    rx_stream = np.tile(tx_syms, 3).astype(np.complex64)
    rx_stream = (rx_stream * (0.4 * 1j)).astype(np.complex64)

    dist_dir = tmp_path / "dataset" / "20km"
    dist_dir.mkdir(parents=True)
    rx_file = dist_dir / "rof_0dBm_20km_osnr28p00dB_650mhz_61p44mbps_snr24p50dB.complex64"
    rx_stream.tofile(rx_file)

    csv_out = pipeline.run_pipeline(
        dataset_dir=tmp_path / "dataset",
        tx_ref_path=tx_path,
        output_csv_path=tmp_path / "results.csv",
        skip_initial_symbols=50,
        best_window_symbols=200,
    )
    assert csv_out.exists()
    with open(csv_out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert float(row["distance_km"]) == 20.0
    assert float(row["bitrate_mbps"]) == 61.44
    assert float(row["ber"]) == 0.0
    assert float(row["evm_rms_pct"]) < 1.0


def test_plotter_and_diagnostics_outputs(tmp_path):
    """Verify 16-QAM constellation plotter and advanced diagnostics produce valid outputs."""
    import pandas as pd

    rng = np.random.default_rng(33)
    bits = rng.integers(0, 2, size=2400, dtype=int)
    tx_syms = demod.modulate_16qam(bits)
    noise = 0.03 * (rng.normal(size=len(tx_syms)) + 1j * rng.normal(size=len(tx_syms)))
    rx_syms = (tx_syms + noise).astype(np.complex64)

    png_const, metrics = plotter.plot_constellation(
        rx_signal=rx_syms,
        tx_ref=tx_syms,
        output_file=tmp_path / "constellation.png",
        skip_initial_symbols=50,
        best_window_symbols=200,
        return_metrics=True,
    )
    assert png_const.exists() and png_const.stat().st_size > 0
    assert metrics["ber"] == 0.0
    assert metrics["evm_rms_pct"] < 8.0

    polar = diagnostics.compute_polar_evm(rx_syms, tx_syms)
    assert polar["evm_r_pct"] > 0.0
    assert polar["evm_theta_pct"] > 0.0

    png_density = diagnostics.plot_constellation_density(rx_syms, tmp_path / "density.png")
    assert png_density.exists() and png_density.stat().st_size > 0

    df = pd.DataFrame(
        {
            "distance_km": [0.0, 20.0, 20.0],
            "evm_rms_pct": [4.5, 8.0, 14.0],
            "ber": [1e-5, 2e-4, 1e-2],
        }
    )
    png_theo = diagnostics.plot_ber_vs_evm_theoretical(df, tmp_path / "ber_vs_evm.png")
    assert png_theo.exists() and png_theo.stat().st_size > 0


def test_file_manager_4bits_per_symbol_mbps(tmp_path):
    """Verify 16-QAM filename Mbps calculation uses 4 bits/symbol."""
    params_burst = {
        "osnr": 28.5,
        "dist": 25,
        "power": 0,
        "freq": 650e6,
        "samp_rate_div": 1,
        "samp_sym": 4,
        "max_sample_rate": 61.44e6,
    }
    fp_burst = file_manager.get_unique_filepath(str(tmp_path), params_burst, 22.10)
    assert Path(fp_burst).name == "rof_0dBm_25km_osnr28p50dB_650mhz_61p44mbps_snr22p10dB.complex64"

    params_stream = {
        "osnr": 20.0,
        "dist": 0,
        "power": -3,
        "freq": 650e6,
        "samp_rate_div": 1,
        "samp_sym": 16,
        "max_sample_rate": 15e6,
    }
    fp_stream = file_manager.get_unique_filepath(str(tmp_path), params_stream, 18.0)
    assert Path(fp_stream).name == "rof_-3dBm_0km_osnr20p00dB_650mhz_3p75mbps_snr18p00dB.complex64"


def test_cli_help_and_constellation(tmp_path):
    """Verify 16qam_src CLI entrypoints work cleanly including constellation check."""
    res_help = subprocess.run(
        [sys.executable, "-m", "16qam_src.cli", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "16-QAM" in res_help.stdout

    rng = np.random.default_rng(55)
    bits = rng.integers(0, 2, size=1600, dtype=int)
    tx_path = tmp_path / "tx.txt"
    tx_path.write_text("".join(str(b) for b in bits), encoding="utf-8")
    rx_path = tmp_path / "rx.complex64"
    demod.modulate_16qam(bits).astype(np.complex64).tofile(rx_path)
    out_png = tmp_path / "cli_const.png"

    res_const = subprocess.run(
        [
            sys.executable,
            "-m",
            "16qam_src.cli",
            "constellation",
            str(rx_path),
            "--tx-ref",
            str(tx_path),
            "--output",
            str(out_png),
            "--no-show",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out_png.exists()
    assert "BER" in res_const.stdout


_GNURADIO_SIM_CODE = """
import importlib, sys
from pathlib import Path
import numpy as np
import pmt
from gnuradio import analog, blocks, channels, digital, filter as gr_filter, gr
from gnuradio.filter import firdes

sys.path.insert(0, str(Path.cwd()))
demod = importlib.import_module("16qam_src.demodulator")
pts = [complex(z) for z in demod.QAM16_CONSTELLATION]
qam16 = digital.constellation_calcdist(
    pts, list(range(16)), 4, 1, digital.constellation.POWER_NORMALIZATION
).base()

tx_str = Path("16qam_src/data/tx.txt").read_text(encoding="utf-8").strip()[:20000]
tx_bits = np.array([int(c) for c in tx_str], dtype=int)
tx_ref = demod.modulate_16qam(tx_bits)

sps = 4
samp_rate = 61.44e6
beta = 0.35

tb = gr.top_block()
raw_bytes = [ord(c) for c in (tx_str * 8)]
src = blocks.vector_source_b(raw_bytes, False)
add_c = blocks.add_const_bb(208)
pack = blocks.pack_k_bits_bb(4)
enc_tx = digital.constellation_encoder_bc(qam16)
tx_rrc = gr_filter.interp_fir_filter_ccf(
    sps, firdes.root_raised_cosine(0.5 * sps, samp_rate, samp_rate / sps, beta, 11 * sps)
)
chan = channels.channel_model(0.01, 0.0001, 1.0, [0.8 * np.exp(1j * 0.5)], 42)
agc = analog.agc2_cc(1e-1, 1e-2, 1.0, 1.0)
rx_filt = gr_filter.freq_xlating_fir_filter_ccf(
    1, firdes.root_raised_cosine(1.0, samp_rate, samp_rate / sps, beta, 11 * sps), 0, samp_rate
)
symsync = digital.symbol_sync_cc(
    digital.TED_GARDNER,
    sps,
    2 * np.pi / 100,
    1.0,
    1.0,
    1.5,
    2,
    qam16,
    digital.IR_PFB_NO_MF,
    64,
    [],
)
eq_alg = digital.adaptive_algorithm_cma(qam16, 0.0001, 1).base()
eq = digital.linear_equalizer(15, 2, eq_alg, True, [], "corr_est")
crec = digital.constellation_receiver_cb(qam16, 2 * np.pi / 200, -0.25, 0.25)
ns = blocks.null_sink(gr.sizeof_float)
enc_rx = digital.constellation_encoder_bc(qam16)
sub = blocks.sub_cc(1)
addc = blocks.add_const_cc(1.0 + 0j)
probe = digital.probe_mpsk_snr_est_c(2, 5000, 0.001)
dbg = blocks.message_debug()
snk = blocks.vector_sink_c()

tb.connect(src, add_c, pack, enc_tx, tx_rrc, chan, agc, rx_filt, symsync, eq, crec)
tb.connect((crec, 1), (ns, 0))
tb.connect((crec, 2), (ns, 1))
tb.connect((crec, 3), (ns, 2))
tb.connect((crec, 4), snk)
tb.connect((crec, 0), enc_rx)
tb.connect((crec, 4), (sub, 0))
tb.connect(enc_rx, (sub, 1))
tb.connect(sub, addc, probe)
tb.msg_connect((probe, "snr"), (dbg, "store"))
tb.run()

rx_syms = np.array(snk.data()[15000:35000], dtype=np.complex64)
res = demod.process_signal(tx_ref, rx_syms, skip_initial_symbols=1000)
assert res["ber"] == 0.0, f"Expected BER==0.0, got {res['ber']}"
assert res["evm_rms_pct"] < 6.0, f"Expected EVM<6%, got {res['evm_rms_pct']}"
assert dbg.num_messages() > 0, "Expected SNR probe messages"
last_snr = pmt.to_python(dbg.get_message(dbg.num_messages() - 1))
assert last_snr > 15.0, f"Expected SNR>15 dB, got {last_snr}"
"""


def test_gnuradio_16qam_end_to_end_simulation():
    """Verify GNU Radio 16-QAM TX->Channel->RX DSP chain matches 16qam_src demodulator with BER==0."""
    try:
        import gnuradio  # noqa: F401
        python_exe = sys.executable
    except ImportError:
        sys_py = Path("/usr/bin/python3")
        if not sys_py.exists():
            pytest.skip("gnuradio not installed")
        probe_res = subprocess.run(
            [str(sys_py), "-c", "import gnuradio"], capture_output=True
        )
        if probe_res.returncode != 0:
            pytest.skip("gnuradio not available in /usr/bin/python3")
        python_exe = str(sys_py)

    proc = subprocess.run(
        [python_exe, "-c", _GNURADIO_SIM_CODE],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"Simulation failed:\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}"
