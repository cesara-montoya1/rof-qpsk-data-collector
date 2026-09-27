# OptNET ML: Optical Signal Processing and Machine Learning Suite

Framework en Python para adquisición, procesamiento, demodulación y análisis de señales ópticas **Radio-over-Fiber (RoF)** con modulación **QPSK**.

El sistema integra algoritmos de sincronización temporal y de fase acelerados con FFT, cálculo de **BER** y **EVM**, compresión masiva (`.complex64` a `.npz`), sincronización oficial con Google Drive, y generación automatizada de curvas analíticas.

---

## ⚡ Comandos Rápidos (Pixi Tasks)

Todas las operaciones se ejecutan mediante [Pixi](https://pixi.sh/) para garantizar reproducibilidad exacta sin conflictos de dependencias:

| Tarea | Comando | Descripción |
| :--- | :--- | :--- |
| **Diagnóstico Rápido** | `pixi run check-signal <archivo.complex64>` | Calcula BER, EVM (%) y visualiza la constelación de una sola muestra en segundos (guarda temporal en `/tmp/`). |
| **Comprimir y Subir** | `pixi run compress-and-upload -d <carpeta_experimento>` | Comprime las subcarpetas a `.npz` y sincroniza inmediatamente con Google Drive (nunca sube `.complex64`). |
| **Comprimir Solo** | `pixi run compress-dataset -d <carpeta_experimento>` | Comprime los `.complex64` locales a `.npz` sin sincronizar a la nube. |
| **Demodular Dataset** | `pixi run demodulate-qpsk -d <carpeta_experimento>` | Ejecuta el pipeline completo de demodulación y genera el `results_*.csv`. |
| **Generar Gráficas** | `pixi run plot-results -c <ruta_a_resultados.csv>` | Genera curvas de BER y EVM vs SNR/OSNR promedio con sombreado de dispersión. |
| **Validar Estructura** | `pixi run validate-dataset -d <carpeta_experimento>` | Comprueba que la carpeta del experimento tenga la estructura correcta (`0km/`, `20km/`, etc.). |
| **Ejecutar Tests** | `pixi run test` | Corre la suite completa de pruebas unitarias (Pytest). |

---

## 🔬 Flujo de Trabajo en el Laboratorio (Hanna & David)

Sigue estos 4 pasos durante las jornadas de medición:

### 1. Guardar las Mediciones en el Lugar Correcto
Guarda siempre las muestras dentro del árbol de Google Drive en `optnet_ml/results/hanna_maestria/`:
```text
optnet_ml/results/hanna_maestria/tu_experimento/
├── 0km/
│   └── *.complex64
└── 20km/
    └── *.complex64
```

### 2. Verificar la Transmisión al Instante
Antes de tomar muchas capturas o procesar todo el dataset, prueba una muestra para comprobar que la señal esté bien alineada:
```bash
pixi run check-signal results/hanna_maestria/tu_experimento/0km/muestra.complex64
```
- **En pantalla:** Abre una ventana interactiva con el diagrama de constelación, cruces ideales, **BER** y **EVM (%)**.
- **Limpio:** La gráfica se almacena en `/tmp/` para no ensuciar el repositorio. Si deseas guardarla permanentemente, agrega `-o mi_grafica.png`.

### 3. Comprimir y Subir a Google Drive
Al terminar la jornada de medición, comprime y sincroniza en un solo comando:
```bash
pixi run compress-and-upload -d results/hanna_maestria/tu_experimento
```
- Convierte todos los `.complex64` en `data_<distancia>.npz`.
- Dispara la sincronización oficial con Google Drive vía `rclone-bisync`.
- Protege la cuota de la nube asegurando que **nunca** se suban archivos pesados `.complex64`.

### 4. Demodular y Graficar Resultados
```bash
# Procesar demodulación (calcula BER, EVM y genera el CSV):
pixi run demodulate-qpsk -d results/hanna_maestria/tu_experimento

# Graficar curvas BER y EVM vs (O)SNR en carpeta plots/:
pixi run plot-results -d results/hanna_maestria/tu_experimento
```

---

## 📁 Estructura del Repositorio

```text
optnet_ml/
├── README.md                # Esta guía de usuario
├── pixi.toml                # Entorno y tareas reproducibles
├── run_qpsk_demodulation.sh # Wrapper para ejecución local y SLURM en GITA
│
├── qpsk_src/                # Código fuente del pipeline
│   ├── cli.py               # CLI unificada (run, plot, constellation, compress, etc.)
│   ├── pipeline.py          # Orquestador de demodulación y guardado seguro
│   ├── demodulator.py       # Sincronización FFT y demodulación QPSK
│   ├── evm.py               # Cálculo de EVM (RMS %, dB, Peak)
│   ├── uploader.py          # Validación de rutas y sincronización a Google Drive
│   ├── compressor.py        # Compresión masiva a archivos .npz
│   ├── plotter.py           # Generador de gráficas de resultados y constelación
│   └── data/tx.txt          # Secuencia PRBS transmitida de referencia
│
├── tests/                   # Suite de pruebas automatizadas
├── docs/                    # Documentación técnica avanzada
└── results/                 # Datasets y resultados experimentales
```

---

## 📚 Documentación Técnica Adicional

Para usuarios avanzados, investigadores que requieran interpretar anomalías físicas o detalles de hardware:

- 📊 **[Guía de Diagnósticos Avanzados de Canal](docs/diagnosticos_avanzados.md)**: Análisis polar de EVM (ruido de fase vs ASE), densidad 2D, espectro de error (PSD) y detección de ráfagas transitorias.
- 📋 **[Referencia de Métricas y Archivo CSV](docs/referencia_metricas_csv.md)**: Descripción detallada de cada columna del CSV, cálculo de EVM en dB/RMS, y reanudación inteligente (*backfill*).
- 📻 **[Hardware USRP y Capturas Automatizadas](docs/hardware_usrp.md)**: Configuración del runner, parámetros de captura (`--repeat`, `--cooldown`), y auto-recuperación de USRP mediante reinicio por software del bus USB (`ioctl`).
