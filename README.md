# OptNET ML: Optical Signal Processing and Machine Learning Suite

Framework en Python para adquisición, procesamiento, demodulación y análisis de señales ópticas **Radio-over-Fiber (RoF)** con modulación **QPSK**.

El sistema integra algoritmos de sincronización temporal y de fase acelerados con FFT, cálculo de **BER** y **EVM**, compresión masiva (`.complex64` a `.npz`), sincronización oficial con Google Drive, y generación automatizada de curvas analíticas.

---

## ⚡ Comandos Rápidos (Pixi Tasks)

Todas las operaciones se ejecutan mediante [Pixi](https://pixi.sh/) para garantizar reproducibilidad exacta sin conflictos de dependencias:

| Tarea | Comando | Descripción |
| :--- | :--- | :--- |
| **Diagnóstico Rápido** | `pixi run check-signal <archivo.complex64>` | Calcula BER, EVM (%) y visualiza la constelación de una sola muestra en segundos (guarda temporal en `/tmp/`). |
| **Comprimir y Subir a GITA** | `pixi run compress-and-upload -d <carpeta_experimento>` | Comprime las subcarpetas a `.npz` autocontenidos (`usrp_rx_gui_<escenario>_<dist>.npz`) y sube `.npz` + `.complex64` a GITA por `rsync`/SSH (con registro de auditoría y alerta en Wi-Fi `VisitanteUdeA`/`SoyUdeA`). |
| **Comprimir Solo** | `pixi run compress-dataset -d <carpeta_experimento>` | Comprime los `.complex64` locales a `.npz` autocontenidos sin transferir a GITA. |
| **Subir a GITA** | `pixi run upload-dataset -d <carpeta_experimento>` | Sube un dataset ya comprimido al servidor GITA (`/export2/camontoya/sofa_datasets`, grupo `sofa`). |
| **Demodular Dataset** | `pixi run demodulate-qpsk -d <carpeta_experimento>` | Ejecuta el pipeline completo de demodulación y genera el `results_*.csv`. |
| **Generar Gráficas** | `pixi run plot-results -c <ruta_a_resultados.csv>` | Genera curvas de BER y EVM vs SNR/OSNR promedio con sombreado de dispersión. |
| **Validar Estructura** | `pixi run validate-dataset -d <carpeta_experimento>` | Comprueba que la carpeta del experimento tenga la estructura correcta (`0km/`, `20km/`, etc.). |
| **Ejecutar Tests** | `pixi run test` | Corre la suite completa de pruebas unitarias (Pytest). |

---

## 🔬 Flujo de Trabajo en el Laboratorio (Hanna & David)

Sigue estos 4 pasos durante las jornadas de medición:

### 1. Guardar las Mediciones con Estructura por Tecnología
Organiza las muestras localmente siguiendo el esquema por tecnología y montaje:
```text
rof_qpsk/uspl/nlna_step0p5db_370ma_650mhz_ch24/
├── 0km/
│   └── *.complex64
└── 20km/
    └── *.complex64
```

### 2. Verificar la Transmisión al Instante
Antes de tomar muchas capturas o procesar todo el dataset, prueba una muestra para comprobar que la señal esté bien alineada:
```bash
pixi run check-signal rof_qpsk/uspl/mi_experimento/0km/muestra.complex64
```
- **En pantalla:** Abre una ventana interactiva con el diagrama de constelación, cruces ideales, **BER** y **EVM (%)**.
- **Limpio:** La gráfica se almacena en `/tmp/` para no ensuciar el repositorio. Si deseas guardarla permanentemente, agrega `-o mi_grafica.png`.

### 3. Comprimir y Subir a GITA (`sofa_datasets`)
> ⚠️ **Importante (Red UdeA):** Las redes Wi-Fi **VisitanteUdeA** y **SoyUdeA** bloquean el puerto SSH de acceso al servidor GITA. Conéctate por **cable Ethernet** antes de subir datos.

La subida a GITA **exige siempre tu archivo de identidad SSH personal** (`-i/--identity` o la variable de entorno `OPTNET_GITA_KEY`) para mantener trazabilidad individual sobre quién publica cada dataset (el dominio y puerto de GITA se configuran exclusivamente de forma privada en tu `~/.ssh/config` bajo el alias `Host gita`):
```bash
# Subir con tu llave SSH personal (puedes exportar OPTNET_GITA_KEY=~/.ssh/tu_llave para omitir -i):
pixi run compress-and-upload -d rof_qpsk/uspl/mi_experimento -i ~/.ssh/id_ed25519_gita

# Desde un PC compartido del laboratorio (indicando además tu usuario de GITA):
pixi run compress-and-upload -d rof_qpsk/uspl/mi_experimento \
    --ssh-user hanna.colorado -i ~/.ssh/id_ed25519_hanna

# Opcional: especificar modo de captura (usrp_rx_gui | usrp_rx_headless | usrp_rx_raw) o subir solo .npz:
pixi run compress-and-upload -d rof_16qam/dfb/mi_experimento -i ~/.ssh/id_ed25519_gita --capture-mode usrp_rx_raw --npz-only
```
- Genera archivos autocontenidos `usrp_rx_gui_<tecnologia>_<montaje>_<escenario>_<distancia>.npz`.
- Transfiere vía `rsync` sobre SSH preservando el usuario dueño (quién publicó) y permisos del grupo `sofa` (`2770`).
- Registra automáticamente cada publicación en `/export2/camontoya/sofa_datasets/upload_log.jsonl`.
- Los archivos `.npz` y `.complex64` **NO se suben a Google Drive** (Google Drive conserva únicamente documentos, caracterizaciones y los `.csv`/gráficas ligeros).

### 4. Demodular y Graficar Resultados
```bash
# Procesar demodulación QPSK (calcula BER, EVM y genera el CSV):
pixi run demodulate-qpsk -d /export2/camontoya/sofa_datasets/rof_qpsk/uspl/<experimento>

# Procesar demodulación 16-QAM:
pixi run demodulate-16qam -d /export2/camontoya/sofa_datasets/rof_16qam/dfb/<experimento>

# Graficar curvas BER y EVM vs (O)SNR en carpeta plots/:
pixi run plot-results -d /export2/camontoya/sofa_datasets/rof_qpsk/uspl/<experimento>
```

---

## 📁 Estructura del Repositorio

```text
optnet_ml/
├── README.md                # Esta guía de usuario
├── pixi.toml                # Entorno y tareas reproducibles
├── run_qpsk_demodulation.sh # Wrapper para ejecución local y SLURM en GITA
│
├── common/                  # Utilidades compartidas (agnósticas a la modulación)
│   ├── cli.py               # CLI unificada (validate, compress, upload)
│   ├── compressor.py        # Compresión a .npz autocontenidos (usrp_rx_gui/headless/raw)
│   ├── uploader.py          # Subida a GITA por rsync/SSH (guardia Wi-Fi UdeA + -i + upload_log)
│   ├── validation.py        # Validación de estructura de distancias (0km, 2km, 20km, ...)
│   ├── metadata.py          # Parser de metadatos desde nombres de captura
│   ├── storage.py           # Lectura y escritura atómica segura de CSVs
│   ├── evm.py               # Cálculo de EVM (RMS %, dB, Peak, mejor ventana)
│   ├── runner.py            # Ejecutor GRC con auto-recuperación USB de USRP
│   └── generate_tx.py       # Generador PRBS LFSR
│
├── qpsk_src/                # Pipeline, demodulador, gráficas, diagnósticos y GRC de QPSK
├── 16qam_src/               # Pipeline, demodulador, gráficas, diagnósticos y GRC de 16-QAM
├── tests/                   # Suite de pruebas automatizadas (Pytest)
└── docs/                    # Documentación técnica y guías de flujo de datos
```

---

## 📚 Documentación Técnica Adicional

Para usuarios avanzados, investigadores que requieran interpretar anomalías físicas o detalles de hardware:

- 🚀 **[Flujo de Datasets, Normalización y Subida a GITA](docs/gita_datasets_workflow.md)**: Guía paso a paso con ejemplos completos de configuración SSH privada, nombres `.npz` autocontenidos, subida con `-i` desde PC personal o compartido, `upload_log.jsonl` y actualización de `rclone-bisync`.
- 📊 **[Guía de Diagnósticos Avanzados de Canal](docs/diagnosticos_avanzados.md)**: Análisis polar de EVM (ruido de fase vs ASE), densidad 2D, espectro de error (PSD) y detección de ráfagas transitorias.
- 📋 **[Referencia de Métricas y Archivo CSV](docs/referencia_metricas_csv.md)**: Descripción detallada de cada columna del CSV, cálculo de EVM en dB/RMS, y reanudación inteligente (*backfill*).
- 📻 **[Hardware USRP y Capturas Automatizadas](docs/hardware_usrp.md)**: Configuración del runner, parámetros de captura (`--repeat`, `--cooldown`), y auto-recuperación de USRP mediante reinicio por software del bus USB (`ioctl`).
