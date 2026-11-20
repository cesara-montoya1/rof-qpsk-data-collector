# Advanced Channel Diagnostics Guide

This guide explains how to generate and physically interpret the optical and RF channel diagnostic plots built into **OptNET ML** (`qpsk_src` and `16qam_src`).

By default, diagnostic figures are saved into `<output_dir>/plots/diagnostics/` so standard BER/EVM curves remain uncluttered. Pass `--subfolder ""` to write figures directly into the main plots directory.

---

## CLI Commands

```bash
# Dataset-level diagnostics (empirical BER vs. theoretical AWGN EVM bound)
pixi run python3 -m qpsk_src.cli diagnostics --csv-path results_experiment.csv
pixi run python3 -m 16qam_src.cli diagnostics --csv-path results_16qam.csv

# Single-capture diagnostics (polar EVM, error PSD, 2D density, burst profile)
pixi run python3 -m qpsk_src.cli diagnostics \
    --rx path/to/capture.complex64 \
    --skip-initial-symbols 500 \
    --best-window-symbols 5000

# 16-QAM single-capture diagnostics saved directly in output-dir (no subfolder)
pixi run python3 -m 16qam_src.cli diagnostics \
    --rx path/to/capture.complex64 \
    --output-dir /tmp/diag_16qam \
    --subfolder ""
```

You can also append `--include-diagnostics` to the regular `plot` command:

```bash
pixi run plot-results --csv-path results_experiment.csv --include-diagnostics
```

---

## Physical Interpretation of Diagnostic Figures

### 1. `ber_vs_evm_theoretical.png` — Empirical BER vs. AWGN Theoretical Bound
- **Principle:** Compares measured `(BER, EVM)` pairs against the Gray-coded Additive White Gaussian Noise (AWGN) theoretical limit:
  - **QPSK:**
    $$\text{BER}_{\text{QPSK}} = \frac{1}{2}\operatorname{erfc}\!\left(\frac{1}{\sqrt{2}\,\text{EVM}_{\text{rms}}}\right)$$
  - **16-QAM:**
    $$\text{BER}_{\text{16-QAM}} = \frac{3}{8}\operatorname{erfc}(x) + \frac{1}{4}\operatorname{erfc}(3x) - \frac{1}{8}\operatorname{erfc}(5x), \quad x = \frac{1}{\sqrt{10}\,\text{EVM}_{\text{rms}}}$$
- **Interpretation:**
  - Points tracking the dashed curve indicate a linear, memoryless AWGN-limited channel.
  - Points lying significantly above the theoretical curve reveal carrier phase cycle slips (Costas loop unlocks), chromatic dispersion (CD), or inter-symbol interference (ISI).

### 2. `polar_evm_breakdown.png` — Polar Decomposition (Amplitude vs. Phase Error)
- **Principle:** Decomposes complex symbol error vectors into radial amplitude error ($\text{EVM}_r$) and tangential phase error ($\text{EVM}_\theta$).
- **Interpretation:**
  - Dominant $\text{EVM}_\theta$ points to **phase noise** from laser linewidth (e.g., DFB or USPL phase jitter) or USRP local-oscillator instability.
  - Dominant $\text{EVM}_r$ points to **amplitude noise** from low received optical power or Amplified Spontaneous Emission (ASE) from an EDFA.

### 3. `error_vector_spectrum.png` — Error Vector Power Spectral Density (PSD)
- **Principle:** Frequency-domain power spectrum of the complex error sequence $e[k] = r[k] - s[k]$.
- **Interpretation:**
  - A flat spectrum confirms white, uncorrelated noise.
  - High-frequency roll-offs, notches, or periodic ripples expose chromatic dispersion fading or analog baseband filter distortion in the USRP front end.

### 4. `constellation_density.png` — 2D Constellation Density Heatmap
- **Principle:** Logarithmic 2D histogram of synchronized received symbols in the complex $(I, Q)$ plane.
- **Interpretation:**
  - Reveals **nonlinear compression** (outer symbols pulled toward the origin by Mach-Zehnder modulator or RF amplifier saturation), I/Q gain/phase imbalance, or DC carrier leakage at $(0, 0)$.

### 5. `error_burst_profile.png` — Temporal Error Burst Profile
- **Principle:** Cumulative bit errors over symbol index compared against a uniform-error diagonal reference.
- **Interpretation:**
  - A straight diagonal confirms stationary, uniform noise.
  - Sharp vertical jumps indicate **transient bursts** caused by mechanical fiber vibration, polarization transients, or USB packet drops on the USRP host bus.
