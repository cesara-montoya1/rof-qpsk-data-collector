# OptNET ML: Optical Signal Processing and Machine Learning Suite

Framework modular en Python para adquisición, procesamiento, demodulación, validación y análisis de señales ópticas moduladas en configuraciones de **Radio-over-Fiber (RoF)** con esquemas **QPSK**.

El sistema integra algoritmos de sincronización temporal y de fase ultrarrápidos (acelerados con FFT), cálculo automatizado de **BER** (Bit Error Rate) y **EVM** (Error Vector Magnitude), análisis de transitorios (warm-up y ventana del mejor bloque), compresión masiva (`.complex64` a `.npz`), generación de curvas analíticas promediadas, herramientas de diagnóstico de canal y pruebas automatizadas.

---

## 📁 Estructura del Repositorio

```text
optnet_ml/
├── README.md                # Documentación general del repositorio y guía de usuario
├── pixi.toml                # Definición del entorno y tareas reproducibles Pixi
├── pixi.lock                # Bloqueo determinista de dependencias
├── run_qpsk_demodulation.sh # Script ejecutable wrapper (compatible con local y SLURM en GITA)
├── filters_gdrive.txt       # Reglas de exclusión para sincronización con rclone
│
├── qpsk_src/                # Módulo principal de procesamiento y adquisición
│   ├── __init__.py          # Exportaciones públicas del paquete
│   ├── cli.py               # CLI unificada (run, plot, constellation, diagnostics, compress, validate)
│   ├── pipeline.py          # Orquestador del flujo de procesamiento, caché y guardado
│   ├── demodulator.py       # Sincronización FFT O(N log N), demodulación y rotación óptima
│   ├── evm.py               # Cálculo de EVM (RMS %, dB, Peak, warm-up skip y ventana del mejor bloque)
│   ├── diagnostics.py       # Diagnósticos avanzados de canal (polar, PSD, densidad 2D, ráfagas, AWGN)
│   ├── plotter.py           # Curvas promediadas BER/(O)SNR, EVM/(O)SNR y diagramas de constelación
│   ├── validation.py        # Validación estructural y ordenamiento numérico de distancias
│   ├── metadata.py          # Extractor de parámetros físicos desde nombres de archivo
│   ├── storage.py           # Gestor de persistencia atómica en CSV (.bak de seguridad)
│   ├── compressor.py        # Compresión masiva condicional de .complex64 a .npz
│   ├── data/
│   │   └── tx.txt           # Secuencia PRBS transmitida de referencia (trackeada en Git)
│   ├── grc/                 # Flowgraphs de GNU Radio Companion (TX y RX)
│   ├── src/                 # Captura automatizada con recuperación de USRP y cooldown
│   │   ├── main.py          # Orquestador de captura (--repeat, --retries, --cooldown)
│   │   ├── runner.py        # Control de subprocesos y reset por software del bus USB
│   │   └── file_manager.py  # Nomenclatura normalizada y anti-colisión
│   └── tools/               # Generador de secuencias PRBS y análisis exploratorio
│       ├── generate_tx.py   # Generador PRBS por streaming a chunks binarios
│       └── ber_count_qpsk.py# Analizador rápido individual de BER/EVM/Constelación
│
├── ml_demodulation/         # Notebooks Jupyter para análisis exploratorio y ML
│   ├── qpsk_demod.py
│   ├── qpsk_demod_CON_ML.ipynb
│   ├── qpsk_demod_2.ipynb
│   └── Prueba para EVM qpsk_demod.ipynb
│
├── tests/                   # Suite de pruebas unitarias (100% passing)
│   ├── conftest.py
│   ├── test_demodulator.py
│   ├── test_diagnostics.py
│   ├── test_evm.py
│   ├── test_pipeline.py
│   ├── test_plotter.py
│   ├── test_metadata.py
│   ├── test_storage.py
│   ├── test_compressor.py
│   ├── test_usrp_runner.py
│   └── test_validation.py
│
└── results/                 # (Ignorado en Git) Datasets pesados y resultados
```

---

## ⚙️ Requisitos e Instalación

El proyecto utiliza **[Pixi](https://pixi.sh/)** para garantizar reproducibilidad exacta y cero conflictos de dependencias en Linux x86_64:

```bash
# 1. Instalar Pixi (si no está instalado en el sistema)
curl -fsSL https://pixi.sh/install.sh | bash

# 2. Verificar el entorno ejecutando los tests
pixi run test
```

Las dependencias (NumPy, SciPy, Pandas, Matplotlib, Pytest) se aíslan y administran dentro de `.pixi/` sin requerir permisos de superusuario.

---

## 🔬 Adquisición en Laboratorio (Captura con USRP)

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

### 🛡️ Novedades de Hardware y Estabilidad:
- **`--repeat N`**: Ejecuta las $N$ capturas en el mismo comando sin necesidad de escribir bucles `for` en bash.
- **`--cooldown T`**: Pausa de $T$ segundos entre capturas para permitir que el kernel de Linux y `libusb` liberen los descriptores de la USRP de forma síncrona.
- **Auto-Recuperación de USRP (`--retries`)**: Si la USRP arroja el error efímero `RuntimeError: LookupError: KeyError: No devices found for -----> Empty Device Address`, el runner realiza un **reset por software del puerto USB** mediante llamadas `ioctl(USBDEVFS_RESET)` al hardware de Ettus/NI, emulando una desconexión y reconexión física del cable sin intervención humana.

---

## 🚀 Procesamiento y Demodulación de Datasets

### 1. Ejecución Rápida mediante Wrapper Shell
```bash
# Uso local básico
./run_qpsk_demodulation.sh <ruta_del_dataset>

# Especificando archivo TX de referencia
./run_qpsk_demodulation.sh <ruta_del_dataset> <ruta_a_tx.txt>

# En el servidor GITA mediante SLURM:
srun --partition=full-gpu ./run_qpsk_demodulation.sh <ruta_del_dataset>
```

### 2. Uso mediante Pixi Tasks
```bash
# Procesar dataset completo
pixi run demodulate-qpsk --dataset-dir <ruta_dataset>

# Generar gráficas básicas de resultados
pixi run plot-results -c <ruta_al_csv_generado>
```

### 3. CLI Unificada Avanzada (`qpsk_src.cli`)

#### A. Demodulación con Descarte de Transitorio y Ventana Óptima (`run`)
```bash
pixi run python3 -m qpsk_src.cli run \
    -d /ruta/al/dataset \
    --skip-initial-symbols 500 \
    --best-window-symbols 5000 \
    --output-csv /ruta/al/dataset/resultados.csv
```
- `--skip-initial-symbols N`: Descarta los primeros $N$ símbolos sincronizados para permitir que el lazo de portadora (Costas/PLL) y el AGC alcancen régimen estacionario.
- `--best-window-symbols M`: Busca el bloque contiguo de tamaño $M$ con menor error para cuantificar el impacto de ráfagas transitorias de ruido ($\Delta\text{EVM}$).

#### B. Generación de Gráficas de Resultados (`plot`)
```bash
# Genera curvas promediadas de BER y EVM vs SNR y OSNR
pixi run python3 -m qpsk_src.cli plot -c /ruta/al/resultados.csv --metric all

# Configurar ancho de bin de SNR (para suavizar fluctuaciones térmicas de la USRP):
pixi run python3 -m qpsk_src.cli plot -c /ruta/al/resultados.csv --snr-bin-width 0.5

# Incluir diagnósticos de canal adicionales (en subcarpeta plots/diagnostics/):
pixi run python3 -m qpsk_src.cli plot -c /ruta/al/resultados.csv --include-diagnostics
```

#### C. Gráfico de Constelación con Contraste (`constellation`)
```bash
pixi run python3 -m qpsk_src.cli constellation \
    --rx /ruta/a/senal.complex64 \
    --tx-ref qpsk_src/data/tx.txt \
    --skip-initial-symbols 500 \
    --best-window-symbols 5000 \
    -o constelacion.png
```
Genera una comparativa de dos paneles: constelación completa en régimen estacionario vs constelación del mejor bloque de $M$ símbolos, con indicación de $\Delta\text{EVM}$ y superposición de los 4 símbolos ideales de QPSK.

---

## 📊 Métricas Registradas en el CSV

El pipeline extrae y almacena las siguientes columnas en el archivo `.csv` de resultados:

| Columna | Unidad / Tipo | Descripción |
| :--- | :--- | :--- |
| `launch_power_dbm` | dBm (float) | Potencia de lanzamiento óptico en el transmisor. |
| `distance_km` | km (float) | Longitud del tramo de fibra óptica (ordenado numéricamente: 0, 0.5, 2, 10, 20 km). |
| `osnr_db` | dB (float) | Relación señal a ruido óptica nominal fijada en laboratorio. |
| `snr_db` | dB (float) | Relación señal a ruido eléctrica de RF medida por la USRP. |
| `freq_mhz` | MHz (float) | Frecuencia de portadora de RF. |
| `bitrate_mbps` | Mbps (float) | Tasa de transmisión de datos. |
| `filename` | str | Nombre del archivo de señal procesado. |
| `ber` | float | Bit Error Rate tras rotación óptima de constelación ($0^\circ, 90^\circ, 180^\circ, 270^\circ$). |
| `detected_delay` | int | Retardo temporal estimado en muestras respecto a la referencia TX. |
| `evm_rms_pct` | % (float) | Error Vector Magnitude RMS en porcentaje sobre los símbolos evaluados. |
| `evm_db` | dB (float) | Error Vector Magnitude RMS en decibeles ($20 \log_{10}(\text{EVM}_{\text{lin}})$). |
| `evm_peak_pct` | % (float) | Error Vector Magnitude pico (error vectorial máximo normalizado). |
| `evm_best_rms_pct` | % (float) | *(Opcional)* EVM RMS del mejor bloque contiguo de $M$ símbolos. |
| `evm_best_db` | dB (float) | *(Opcional)* EVM en dB del mejor bloque contiguo de $M$ símbolos. |
| `evm_delta_rms_pct` | % (float) | *(Opcional)* $\Delta\text{EVM} = \text{EVM}_{\text{total}} - \text{EVM}_{\text{mejor}}$ (indicador de ráfagas de ruido). |
| `npz_source` | str | Nombre del archivo `.npz` comprimido de origen. |

### 💡 Reanudación Inteligente (Backfill)
- **Datasets ya procesados:** Si un archivo ya tiene registros completos en el CSV, el pipeline no repite el cómputo.
- **Actualización sin duplicados:** Si el CSV ya existía con mediciones de BER previas pero carecía de columnas de EVM, el pipeline detecta los registros incompletos, computa únicamente el EVM y actualiza el archivo CSV **in-place** sin alterar los identificadores existentes.

---

## 📈 Gráficas Básicas (Carpeta `plots/`)

El generador guarda por defecto en `plots/` las curvas promedio con bandas de dispersión mín-máx:
- **`ber_vs_snr_mean.png` / `ber_vs_snr_shaded.png`**: BER vs SNR (dB) promedio agrupado por distancia.
- **`ber_vs_osnr_mean.png` / `ber_vs_osnr_shaded.png`**: BER vs OSNR (dB) promedio agrupado por distancia.
- **`evm_vs_snr_mean.png` / `evm_vs_snr_shaded.png`**: EVM (dB) vs SNR (dB).
- **`evm_vs_osnr_mean.png` / `evm_vs_osnr_shaded.png`**: EVM (dB) vs OSNR (dB).
- **`evm_pct_vs_snr_shaded.png` / `evm_pct_vs_osnr_shaded.png`**: EVM RMS (%) vs (O)SNR.

> [!NOTE]
> **Suavizado de SNR:** En BER vs SNR y EVM vs SNR, el motor agrupa las capturas por escalón de OSNR o mediante ventanas discretas (`--snr-bin-width 0.5`), eliminando la oscilación errática punto a punto provocada por el jitter térmico de la USRP.

---

## 🔬 Diagnósticos Avanzados de Canal (`plots/diagnostics/`)

Para no generar ruido visual a los usuarios habituales del repositorio, todas las figuras de diagnóstico avanzado se guardan por defecto en la subcarpeta aislada:  
📁 `<directorio>/plots/diagnostics/`

*(Configurable: Si deseas que queden en el directorio raíz de plots, utiliza `--subfolder ""`).*

### Comandos de Diagnóstico:
```bash
# Diagnóstico a nivel de dataset (BER empírico vs límite teórico de AWGN):
pixi run python3 -m qpsk_src.cli diagnostics --csv-path resultados.csv

# Diagnóstico a nivel de señal individual (polar, PSD, densidad 2D, ráfagas):
pixi run python3 -m qpsk_src.cli diagnostics \
    --rx senal.complex64 \
    --skip-initial-symbols 500 \
    --best-window-symbols 5000

# Guardar directamente en la carpeta principal sin subcarpeta:
pixi run python3 -m qpsk_src.cli diagnostics --rx senal.complex64 --subfolder ""
```

### Guía de Interpretación Física de las Gráficas:

1. **`ber_vs_evm_theoretical.png` (BER Empírico vs Límite AWGN)**
   - *Fundamento:* Compara las parejas (BER, EVM) medidas frente al límite teórico de QPSK: $\text{BER} = \frac{1}{2} \text{erfc}\left(\frac{1}{\sqrt{2} \cdot \text{EVM}_{\text{rms}}}\right)$.
   - *Interpretación:* Si los datos se alinean sobre la curva teórica discontinua, el canal es estrictamente aditivo gaussiano lineal. Si caen por encima, revela desenganches de fase del lazo Costas (*cycle slips*) o interferencia entre símbolos (ISI).

2. **`polar_evm_breakdown.png` (Descomposición Polar: Fase vs Amplitud)**
   - *Fundamento:* Descompone el error del vector en componente radial ($EVM_r$) y tangencial/de fase ($EVM_\theta$).
   - *Interpretación:* Si domina $EVM_\theta$, la degradación proviene del **ruido de fase / ancho de línea del láser DFB** o del oscilador local de la USRP. Si domina $EVM_r$, la limitación es la **potencia óptica recibida / ruido ASE del EDFA**.

3. **`error_vector_spectrum.png` (Densidad Espectral de Potencia del Error - PSD)**
   - *Fundamento:* Espectro en frecuencia del vector de error complejo $e[k] = rx - tx$.
   - *Interpretación:* Un espectro plano certifica ruido blanco. Caídas en alta frecuencia o lóbulos periódicos delatan distorsión por **dispersión cromática (CD)** en la fibra o la función de transferencia de los filtros pasa-bajos de la USRP.

4. **`constellation_density.png` (Mapa de Calor 2D de la Constelación)**
   - *Fundamento:* Histograma bidimensional en escala logarítmica de la densidad de símbolos recibidos en el plano complejo $(I, Q)$.
   - *Interpretación:* Detecta **compresión no lineal** (si los centroides se comprimen hacia el origen por saturación del modulador Mach-Zehnder o amplificador RF), desbalance de ganancia/fase I/Q o fuga de portadora en $(0,0)$.

5. **`error_burst_profile.png` (Perfil Temporal de Ráfagas de Error)**
   - *Fundamento:* Acumulación de errores de bit en el tiempo vs referencia uniforme.
   - *Interpretación:* Una línea recta diagonal indica ruido uniforme y estacionario. Saltos verticales abruptos revelan **vibraciones mecánicas en la fibra** o pérdidas puntuales de paquetes en el bus USB de la USRP.

---

## 🧪 Pruebas Automatizadas

La suite incluye 60 pruebas unitarias que validan la integridad de todo el sistema:
```bash
pixi run test
```
Cubre algoritmos FFT, demodulación y sincronización QPSK, compresión `.npz`, cálculo y propiedades de escala del EVM, ordenamiento numérico de distancias, auto-recuperación de USRP, y generación determinista de todas las curvas y diagnósticos.
