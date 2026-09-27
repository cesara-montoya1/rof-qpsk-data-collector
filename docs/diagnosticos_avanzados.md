# Diagnósticos Avanzados de Canal (QPSK)

Este documento detalla las herramientas de diagnóstico físico y espectral del canal óptico integradas en `optnet_ml`.

Para no saturar las carpetas principales ni generar ruido visual en el laboratorio, todas las figuras de diagnóstico avanzado se guardan por defecto en:
📁 `<directorio>/plots/diagnostics/`

*(Configurable: Si deseas que queden en el directorio raíz de plots, utiliza `--subfolder ""`).*

---

## Comandos de Diagnóstico

```bash
# Diagnóstico a nivel de dataset completo (BER empírico vs límite teórico de AWGN):
pixi run python3 -m qpsk_src.cli diagnostics --csv-path resultados.csv

# Diagnóstico a nivel de señal individual (polar, PSD, densidad 2D, ráfagas):
pixi run python3 -m qpsk_src.cli diagnostics \
    --rx senal.complex64 \
    --skip-initial-symbols 500 \
    --best-window-symbols 5000

# Guardar directamente en la carpeta principal de plots sin subcarpeta:
pixi run python3 -m qpsk_src.cli diagnostics --rx senal.complex64 --subfolder ""
```

---

## Guía de Interpretación Física de las Gráficas

### 1. `ber_vs_evm_theoretical.png` (BER Empírico vs Límite AWGN)
- **Fundamento:** Compara las parejas (BER, EVM) medidas frente al límite teórico de QPSK:
  $$\text{BER} = \frac{1}{2} \text{erfc}\left(\frac{1}{\sqrt{2} \cdot \text{EVM}_{\text{rms}}}\right)$$
- **Interpretación:** 
  - Si los datos se alinean sobre la curva teórica discontinua, el canal es estrictamente lineal aditivo gaussiano (AWGN).
  - Si los puntos caen significativamente por encima de la curva, delata desenganches de fase del lazo Costas (*cycle slips*), distorsión por dispersión cromática (CD) o interferencia entre símbolos (ISI).

### 2. `polar_evm_breakdown.png` (Descomposición Polar: Fase vs Amplitud)
- **Fundamento:** Descompone el error del vector en componente radial ($EVM_r$) y tangencial/de fase ($EVM_\theta$).
- **Interpretación:** 
  - Si domina $EVM_\theta$, la degradación proviene principalmente del **ruido de fase / ancho de línea del láser DFB** o del oscilador local (LO) de la USRP.
  - Si domina $EVM_r$, la limitación principal es la **potencia óptica recibida / ruido de emisión espontánea amplificada (ASE) del EDFA**.

### 3. `error_vector_spectrum.png` (Densidad Espectral de Potencia del Error - PSD)
- **Fundamento:** Espectro en frecuencia del vector de error complejo $e[k] = r[k] - s[k]$.
- **Interpretación:** 
  - Un espectro plano certifica ruido blanco gaussiano sin memoria.
  - Caídas en alta frecuencia o lóbulos periódicos delatan distorsión por **dispersión cromática (CD)** en la fibra o la función de transferencia de los filtros pasabajos analógicos de la USRP.

### 4. `constellation_density.png` (Mapa de Calor 2D de la Constelación)
- **Fundamento:** Histograma bidimensional en escala logarítmica de la densidad de símbolos recibidos en el plano complejo $(I, Q)$.
- **Interpretación:** 
  - Detecta **compresión no lineal** (si los centroides se comprimen hacia el origen por saturación del modulador Mach-Zehnder o amplificador RF).
  - Permite identificar desbalance de ganancia/fase I/Q o fuga de portadora en $(0,0)$.

### 5. `error_burst_profile.png` (Perfil Temporal de Ráfagas de Error)
- **Fundamento:** Acumulación acumulada de errores de bit en el tiempo vs una referencia uniforme ideal.
- **Interpretación:** 
  - Una línea recta diagonal indica ruido uniforme y estacionario.
  - Saltos verticales abruptos revelan **vibraciones mecánicas transitorias en la fibra** o pérdidas puntuales de paquetes en el bus USB de la USRP.
