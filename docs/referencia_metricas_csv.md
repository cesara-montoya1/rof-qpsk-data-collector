# Referencia de Métricas y Archivo CSV

Este documento detalla todas las columnas generadas en el archivo `.csv` de resultados de demodulación en `optnet_ml`.

---

## Columnas del CSV de Resultados

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
| `evm_delta_rms_pct` | % (float) | *(Opcional)* $\Delta\text{EVM} = \text{EVM}_{\text{total}} - \text{EVM}_{\text{mejor}}$ (indicador de ráfagas transitorias de ruido). |
| `npz_source` | str | Nombre del archivo `.npz` comprimido de origen. |

---

## Reanudación Inteligente (Backfill)

- **Datasets ya procesados:** Si un archivo ya tiene registros completos en el CSV, el pipeline no repite el cómputo y pasa al siguiente.
- **Actualización sin duplicados:** Si el CSV ya existía con mediciones de BER previas pero carecía de columnas de EVM, el pipeline detecta los registros incompletos, computa únicamente el EVM y actualiza el archivo CSV **in-place** de forma atómica (generando respaldo `.csv.bak`) sin alterar los identificadores existentes.
