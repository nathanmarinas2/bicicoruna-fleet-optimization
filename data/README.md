# 📂 Estructura de Datos del Proyecto

Este directorio contiene los datasets utilizados para el análisis, predicción y optimización del sistema BiciCoruña.

Para mantener el repositorio ligero y profesional, **solo se incluyen los datos fuente indispensables y resultados ligeros**. Los archivos intermedios pesados o binarios grandes deben ser regenerados ejecutando el pipeline.

> 🛠️ **Ingeniería de Datos Propia**
>
> Este dataset **NO es un recurso público descargado**. Fue construido desde cero mediante un proceso de ingeniería de datos propia:
> 1. **Monitorización continua** de la API en tiempo real.
> 2. **Consolidación** de miles de snapshots de estado del sistema.
> 3. **Limpieza y estructuración** para crear una serie temporal histórica que no existía previamente.
>
> Este esfuerzo convierte datos volátiles en tiempo real en un activo persistente de valor analítico.

## 🗂️ Descripción de Carpetas

### 1. `coruna/` (Fuente de Verdad)
Contiene los datos crudos originales del sistema BiciCoruña. **Estos archivos son necesarios para ejecutar cualquier análisis.**
- `tracking_data.csv`: Registro histórico de movimientos y estado de las estaciones.
- `sistema.csv`: Metadatos de las estaciones (ubicación, capacidad, ID).

### 2. `external/` (Datos Auxiliares)
Datos complementarios obtenidos de fuentes externas (AEMET, Concello da Coruña, Calendarios laborales).
- `*.geojson`: Límites administrativos (distritos/barrios).
- `demografia_barrios.csv`: Datos sociodemográficos para análisis de equidad.
- `historico_usuarios_*.csv`: Series temporales históricas para proyecciones a largo plazo.

> 💡 **Nota:** Si faltan archivos aquí, ejecuta: `python src/utils/download_datos_externos.py`

### 3. `processed/` (Resultados)
Contiene los *insights* procesados y resultados finales de los modelos.
- `estaciones_clusters.csv`: Segmentación de estaciones (Residencial, Trabajo, Ocio...).
- `ranking_nuevas_paradas.csv`: Listado priorizado de ubicaciones óptimas para nuevas estaciones.
- `proyeccion_usuarios_*.csv`: Predicciones de demanda a futuro (2025-2026).

> ⚠️ **Archivos Ignorados:** Los modelos entrenados (`.pkl`), predicciones masivas (`coruna_predictions.csv`) y datos de entrenamiento intermedios (`.parquet`) están excluidos del control de versiones por su tamaño. Para generarlos, ejecuta el pipeline completo.

### 4. `raw/` (Transfer Learning - Ignorado)
Directorio reservado para datasets crudos de otros sistemas (Madrid, Washington DC, Barcelona) utilizados para el *Transfer Learning*.
- Estos archivos **NO se incluyen** en el repositorio debido a su gran tamaño (>500MB).
- El sistema es capaz de entrenar solo con los datos de Coruña si estos no están presentes, aunque con menor capacidad de generalización.

---

## Reproducibilidad
Para regenerar todos los archivos procesados a partir de los datos fuente:

```bash
# 1. Entrenar modelo de producción
python src/models/classifier_final.py

# 2. Generar análisis y visualizaciones
python src/evaluation/analisis_bicicoruna.py
python src/evaluation/analisis_codo.py
python src/evaluation/optimizacion_flota.py
```
