# OptNET ML: Optical Signal Processing and Machine Learning Suite

Framework modular en Python para adquisición, procesamiento, demodulación, validación y análisis de señales ópticas moduladas en configuraciones de **Radio-over-Fiber (RoF)** con esquemas **QPSK**.

El sistema integra algoritmos de sincronización temporal y de fase, cálculo automatizado de **BER** (Bit Error Rate) y **EVM** (Error Vector Magnitude), compresión de señales (`.complex64` a `.npz`), generación de curvas analíticas y pruebas automatizadas.

---

## 📁 Estructura del Repositorio

```text
optnet_ml/
├── README.md                # Documentación general del repositorio
├── pixi.toml                # Definición del entorno y tareas reproducibles Pixi
├── pixi.lock                # Bloqueo determinista de dependencias
├── run_qpsk_demodulation.sh # Script ejecutable wrapper (compatible con local y SLURM en GITA)
├── filters_gdrive.txt       # Reglas de exclusión para sincronización con rclone
│
├── qpsk_src/                # Módulo principal de procesamiento
│   ├── __init__.py          # Exportaciones principales del paquete
│   ├── cli.py               # CLI unificada (run, plot, compress, validate)
│   ├── pipeline.py          # Orquestador del flujo de procesamiento y guardado
│   ├── demodulator.py       # Modulación, sincronización y barrido de fase
│   ├── evm.py               # Cálculo de EVM (RMS %, dB y Peak %)
│   ├── validation.py        # Validación estructural de datasets y señales
│   ├── metadata.py          # Extractor de parámetros físicos desde nombres de archivo
│   ├── storage.py           # Gestor de lectura/escritura segura en CSV (.bak atómico)
│   ├── compressor.py        # Compresión masiva de .complex64 a .npz
│   ├── plotter.py           # Generación de curvas BER vs (O)SNR y EVM vs (O)SNR
│   ├── data/
│   │   └── tx.txt           # Secuencia PRBS transmitida de referencia
│   ├── grc/                 # Flowgraphs de GNU Radio Companion (TX y RX)
│   ├── src/                 # Orquestador original de captura en laboratorio
│   └── tools/               # Generador de TX y herramientas auxiliares
│
├── ml_demodulation/         # Notebooks Jupyter para análisis exploratorio y ML
│   ├── qpsk_demod.py
│   ├── qpsk_demod_CON_ML.ipynb
│   ├── qpsk_demod_2.ipynb
│   └── Prueba para EVM qpsk_demod.ipynb
│
├── tests/                   # Suite completa de pruebas unitarias (pytest)
│   ├── conftest.py
│   ├── test_demodulator.py
│   ├── test_evm.py
│   ├── test_pipeline.py
│   ├── test_plotter.py
│   ├── test_metadata.py
│   ├── test_storage.py
│   ├── test_compressor.py
│   └── test_validation.py
│
├── utils/
│   └── compress_npz.py      # Script utilitario de compresión
│
└── results/                 # (Ignorado en Git) Datasets pesados y resultados
```

---

## ⚙️ Requisitos e Instalación

El proyecto utiliza **[Pixi](https://pixi.sh/)** para garantizar un entorno reproducible en cualquier máquina (Linux x86_64).

1. Instalar Pixi si no está instalado:
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```
2. Las dependencias se instalarán automáticamente en el primer comando que ejecutes con `pixi run`.

---

## 🚀 Guía de Uso

### 1. Ejecución Rápida del Pipeline de Demodulación

Puedes ejecutar el script principal indicando la carpeta del dataset a procesar:

```bash
# Uso local básico
./run_qpsk_demodulation.sh <ruta_del_dataset>

# Especificando un archivo de bits de referencia personalizado
./run_qpsk_demodulation.sh <ruta_del_dataset> <ruta_a_tx.txt>
```

#### En el Clúster GITA (Servidor con SLURM)
```bash
# Modo interactivo
srun --partition=full-gpu ./run_qpsk_demodulation.sh <ruta_del_dataset>

# Modo segundo plano (batch)
sbatch --partition=full-gpu ./run_qpsk_demodulation.sh <ruta_del_dataset>
```

---

### 2. Tareas Disponibles con Pixi

```bash
# 1. Demodular dataset y extraer métricas (BER y EVM)
pixi run demodulate-qpsk --dataset-dir <ruta_dataset>

# 2. Generar todas las curvas analíticas (BER y EVM vs SNR / OSNR)
pixi run plot-results -c <ruta_al_csv_generado>

# 3. Validar la estructura del dataset
pixi run validate-dataset -d <ruta_dataset>

# 4. Comprimir archivos crudos .complex64 a .npz
pixi run compress-dataset -d <ruta_dataset>

# 5. Ejecutar todas las pruebas unitarias
pixi run test
```

---

### 3. CLI Unificada (`qpsk_src.cli`)

Puedes invocar la CLI directamente con `python3 -m qpsk_src.cli`:

```bash
# Procesar dataset
pixi run python3 -m qpsk_src.cli run -d <ruta_dataset> [--output-csv <salida.csv>]

# Generar gráficas específicas:
# Ambas métricas (BER y EVM)
pixi run python3 -m qpsk_src.cli plot -c <salida.csv> --metric all

# Solo curvas de EVM
pixi run python3 -m qpsk_src.cli plot -c <salida.csv> --metric evm

# Solo curvas de BER
pixi run python3 -m qpsk_src.cli plot -c <salida.csv> --metric ber
```

---

## 📊 Métricas Calculadas y Reanudación Inteligente

El pipeline procesa cada archivo de señal y registra en el archivo CSV:
- **Parámetros físicos extraídos:** potencia de lanzamiento (`launch_power_dbm`), distancia (`distance_km`), OSNR (`osnr_db`), SNR (`snr_db`), frecuencia (`freq_mhz`), tasa de bits (`bitrate_mbps`).
- **Métricas de demodulación:**
  - `ber`: Tasa de error binario tras rotación óptima de constelación ($1, j, -1, -j$).
  - `detected_delay`: Retardo temporal estimado en muestras.
  - `evm_rms_pct`: Error Vector Magnitude RMS en porcentaje ($\%$).
  - `evm_db`: Error Vector Magnitude en decibeles ($\text{dB}$).
  - `evm_peak_pct`: Error Vector Magnitude pico en porcentaje ($\%$).

### 💡 Reanudación Inteligente (Backfill)
- **Si el dataset ya fue procesado y el CSV tiene los datos completos:** El pipeline lo detecta y no repite el cómputo.
- **Si el CSV ya contiene mediciones de BER pero le falta el EVM:** El pipeline detecta los registros incompletos, calcula únicamente el EVM y **actualiza el CSV en su lugar (in-place)** sin generar filas duplicadas.

---

## 📈 Gráficas Generadas

El generador de gráficas guarda en la carpeta `plots/` las siguientes curvas agrupadas por distancia:
- **`ber_vs_snr_mean.png` / `ber_vs_snr_shaded.png`**: BER vs SNR (dB) (Media y rango mín-máx sombreado).
- **`ber_vs_osnr_mean.png` / `ber_vs_osnr_shaded.png`**: BER vs OSNR (dB).
- **`evm_vs_snr_mean.png` / `evm_vs_snr_shaded.png`**: EVM (dB) vs SNR (dB).
- **`evm_vs_osnr_mean.png` / `evm_vs_osnr_shaded.png`**: EVM (dB) vs OSNR (dB).
- **`evm_pct_vs_snr_shaded.png` / `evm_pct_vs_osnr_shaded.png`**: EVM RMS (%) vs (O)SNR.

---

## 🧪 Pruebas Unitarias

El proyecto cuenta con una cobertura completa mediante `pytest`:
```bash
pixi run test
```
Verifica validación de datasets, extracción de metadatos, algoritmos de demodulación, compresión `.npz`, cálculo y propiedades de escala del EVM, persistencia atómica y generación de gráficas.
