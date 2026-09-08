# OptNET ML: Optical Signal Processing and Machine Learning Suite

Framework modular para adquisición, demodulación, validación y análisis de señales ópticas moduladas (Radio-over-Fiber / QPSK).

## 🚀 Estructura del Proyecto

```text
optnet_ml/
├── pixi.toml                # Gestión de entorno y tareas Pixi
├── pixi.lock                # Bloqueo reproducible de dependencias
├── run_qpsk_demodulation.sh # Script de ejecución rápida
├── ml_demodulation/         # Notebooks de análisis y demodulación ML
├── qpsk_src/                # Módulo central de procesamiento y herramientas
│   ├── cli.py               # Interfaz de línea de comandos unificada
│   ├── pipeline.py          # Orquestador del pipeline de demodulación
│   ├── demodulator.py       # Algoritmos de demodulación QPSK / CMA / Costas
│   ├── validation.py        # Validación de integridad de señales y datos
│   ├── metadata.py          # Extracción y estandarización de metadatos
│   ├── storage.py           # Gestor de persistencia en CSV
│   ├── compressor.py        # Compresión de señales .complex64 a .npz
│   ├── plotter.py           # Generación de curvas BER vs SNR / OSNR
│   ├── grc/                 # Flowgraphs de GNU Radio (TX y RX)
│   ├── src/                 # Scripts originales del orquestador de captura
│   └── tools/               # Generador de TX y cálculo de BER
├── tests/                   # Suite de pruebas unitarias automatizadas (pytest)
└── utils/                   # Scripts auxiliares (compresión y utilidades)
```

## 🛠️ Comandos Disponibles (Pixi)

El proyecto utiliza **Pixi** para la gestión reproducible del entorno:

```bash
# Ejecutar pipeline completo de demodulación
pixi run demodulate-qpsk

# Generar curvas BER vs SNR / OSNR
pixi run plot-results

# Validar integridad del dataset
pixi run validate-dataset

# Comprimir datasets de .complex64 a .npz
pixi run compress-dataset

# Ejecutar suite de pruebas
pixi run test
```

Para detalles del recolector experimental en GNU Radio, consulta [qpsk_src/README.md](qpsk_src/README.md).
