# USRP Capture Hardware & Automatic Recovery

This guide documents automated multi-capture execution and USB hardware recovery for Ettus Research and National Instruments (NI) USRP receivers in **OptNET ML**.

The underlying execution runner lives in [`common/runner.py`](../common/runner.py) and is shared by both `qpsk_src` and `16qam_src`.

---

## Automated Multi-Capture Execution

Avoid writing external bash loops around GNU Radio scripts, as rapid restart cycles frequently trigger `libusb` handle collisions and `No devices found` errors. Use the built-in `--repeat`, `--cooldown`, and `--retries` flags instead:

```bash
# QPSK: capture 10 consecutive signals at 650 MHz, 32.5 dB OSNR, 20 km fiber
python3 -m qpsk_src.src.main \
    --osnr 32.5 \
    --distance 20 \
    --power 0 \
    --freq 650 \
    --repeat 10 \
    --cooldown 2.0 \
    --retries 3 \
    --move ./rof_qpsk/dfb/my_experiment/20km

# 16-QAM headless burst capture at 61.44 MS/s (sps=4)
python3 -m 16qam_src.src.main \
    --osnr 28.0 \
    --distance 25 \
    --power 0 \
    --freq 650 \
    --samp-sym 4 \
    --max-sample-rate 61.44 \
    --grc-script 16qam_src/grc/16qam_rx_burst.py \
    --repeat 10 \
    --cooldown 2.0 \
    --retries 3 \
    --move ./rof_16qam/dfb/my_experiment/25km
```

---

## Stability & Hardware Recovery Mechanisms

- **`--repeat N`:** Runs $N$ captures sequentially within a single orchestrated session with synchronous process teardown (`terminate` → `wait` → `kill` fallback) so UHD/USB handles are released deterministically.
- **`--cooldown T`:** Sleeps $T$ seconds (recommended: `2.0`) between captures to let the Linux kernel and `libusb` settle cleanly.
- **`--retries R` (Software USB Reset):** When UHD emits transient device errors (`LookupError`, `No devices found`, `RuntimeError ... UHD`), `common.runner.reset_usrp_usb()` locates connected Ettus (`idVendor=2500`) or NI (`idVendor=3923`) USB devices under `/sys/bus/usb/devices/` and issues a low-level `ioctl(USBDEVFS_RESET)` call. This emulates physically unplugging and replugging the USB cable without manual intervention, then retries the capture automatically.
- **`--usrp-args "..."`:** Passes explicit UHD device arguments (e.g., `"type=b200"` or `"serial=..."`) when multiple USRP units are connected to the same host.
