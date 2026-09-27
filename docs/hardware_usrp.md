# Hardware USRP y Captura Automatizada

Este documento describe la configuración avanzada de hardware y los mecanismos de estabilidad para capturas masivas con la USRP en el laboratorio.

---

## Ejecución de Capturas con `main.py`

Para realizar capturas automáticas de señales sin caídas aleatorias de hardware:

```bash
# Capturar 10 repeticiones automáticas con auto-recuperación ante LookupError:
python3 qpsk_src/src/main.py \
    --osnr 32.5 \
    --distance 20 \
    --power 0 \
    --freq 650 \
    --repeat 10 \
    --cooldown 2.0 \
    --retries 3 \
    --move ../data/results
```

---

## Características de Estabilidad y Hardware

- **`--repeat N`**: Ejecuta las $N$ capturas en el mismo comando sin necesidad de escribir bucles `for` en bash.
- **`--cooldown T`**: Pausa de $T$ segundos entre capturas para permitir que el kernel de Linux y `libusb` liberen los descriptores de la USRP de forma síncrona.
- **Auto-Recuperación de USRP (`--retries`)**: 
  - Si la USRP arroja el error efímero:
    `RuntimeError: LookupError: KeyError: No devices found for -----> Empty Device Address`
  - El runner realiza un **reset por software del puerto USB** mediante llamadas de bajo nivel `ioctl(USBDEVFS_RESET)` al hardware de Ettus/NI.
  - Esto emula una desconexión y reconexión física del cable USB sin requerir intervención humana, reintentando la captura automáticamente.
