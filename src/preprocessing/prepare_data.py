"""
=============================================================================
🔧 PREPROCESAMIENTO Y FEATURE ENGINEERING
=============================================================================
Este script:
1. Carga todos los datasets (DC, Barcelona, Madrid)
2. Unifica el formato
3. Crea features optimizadas para transfer learning
4. Guarda el dataset procesado

Features diseñadas para TRANSFERIR bien a Coruña:
- Todas relativas (porcentajes, no valores absolutos)
- Codificación cíclica para tiempo
- Normalizadas por dataset

Ejecutar: python src/preprocessing/prepare_data.py
=============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.utils.config import load_config

# Directorio base
BASE_DIR = Path(__file__).parent.parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

# Asegurar que existe el directorio de salida
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


def cyclic_encode(value, max_value):
    """Codificación cíclica para features temporales."""
    sin_val = np.sin(2 * np.pi * value / max_value)
    cos_val = np.cos(2 * np.pi * value / max_value)
    return sin_val, cos_val


def load_dc_dataset():
    """
    Carga el dataset de Washington DC.
    Este dataset tiene formato horario, muy limpio.
    """
    print("\n📂 Cargando Washington DC...")
    
    dc_dir = DATA_RAW / "dc"
    
    # Buscar archivos
    hour_file = dc_dir / "hour.csv"
    
    if not hour_file.exists():
        print("   ❌ No encontrado. Ejecuta primero download_datasets.py")
        return None
    
    df = pd.read_csv(hour_file)
    print(f"   ✅ Cargado: {len(df):,} filas")
    
    # Renombrar y procesar
    df = df.rename(columns={
        'dteday': 'date',
        'hr': 'hour',
        'temp': 'temp_normalized',  # Ya viene normalizado 0-1
        'hum': 'humidity',
        'windspeed': 'wind_normalized',
        'cnt': 'total_rentals',
        'casual': 'casual_rentals',
        'registered': 'registered_rentals',
        'weathersit': 'weather_code',
        'holiday': 'is_holiday',
        'workingday': 'is_workingday',
        'weekday': 'day_of_week'
    })
    
    # Crear timestamp
    df['date'] = pd.to_datetime(df['date'])
    df['timestamp'] = df.apply(
        lambda x: x['date'] + pd.Timedelta(hours=x['hour']), 
        axis=1
    )
    
    # Features
    df['city'] = 'washington_dc'
    df['is_weekend'] = df['day_of_week'].isin([0, 6]).astype(int)
    df['is_rush_hour'] = df['hour'].apply(
        lambda h: 1 if (7 <= h <= 9) or (17 <= h <= 19) else 0
    )
    
    # El dataset de DC no tiene ocupación por estación, 
    # pero podemos simular ocupación relativa
    max_rentals = df['total_rentals'].max()
    df['occupancy_proxy'] = df['total_rentals'] / max_rentals
    
    # Lluvia
    df['is_raining'] = (df['weather_code'] >= 3).astype(int)
    
    # Temperatura real (el dataset la tiene normalizada)
    # Rango original: -8°C a 39°C (según documentación)
    df['temperature'] = df['temp_normalized'] * 47 - 8
    
    print(f"   📊 Rango fechas: {df['date'].min()} a {df['date'].max()}")
    
    return df


def load_barcelona_dataset():
    """
    Carga el dataset de Barcelona (Bicing).
    Formato esperado: Parquet con disponibilidad por estación.
    """
    print("\n📂 Cargando Barcelona...")
    
    bcn_dir = DATA_RAW / "barcelona"
    
    # Buscar archivos parquet o csv
    parquet_files = list(bcn_dir.glob("*.parquet"))
    csv_files = list(bcn_dir.glob("*.csv"))
    
    if not parquet_files and not csv_files:
        print("   ❌ No encontrado. Descarga desde Kaggle.")
        print("   📋 URL: https://www.kaggle.com/datasets/sgonzalezq/bcn-bike-sharing-dataset-bicing-stations")
        return None
    
    # Cargar (preferir parquet por eficiencia)
    if parquet_files:
        print(f"   Encontrados {len(parquet_files)} archivos Parquet")
        # Tomar muestra si es muy grande
        df = pd.read_parquet(parquet_files[0])
        
        # Si es muy grande, hacer sampling
        if len(df) > 10_000_000:
            print(f"   ⚠️ Dataset muy grande ({len(df):,} filas). Haciendo sampling al 5%...")
            df = df.sample(frac=0.05, random_state=42)
    else:
        print(f"   Encontrados {len(csv_files)} archivos CSV")
        # Cargar solo el primero/más grande
        df = pd.read_csv(csv_files[0], nrows=5_000_000)  # Limitar filas
    
    print(f"   ✅ Cargado: {len(df):,} filas")
    
    # El formato de Kaggle de Barcelona suele tener:
    # station_id, timestamp, num_bikes_available, num_docks_available, etc.
    
    # Detectar columnas y adaptar
    df.columns = df.columns.str.lower()
    
    # Renombrar según lo que exista
    rename_map = {
        'station_id': 'station_id',
        'num_bikes_available': 'bikes_available',
        'num_docks_available': 'docks_available',
        'num_bikes_available_types.mechanical': 'bikes_mechanical',
        'num_bikes_available_types.ebike': 'bikes_electric',
    }
    
    for old, new in rename_map.items():
        if old in df.columns:
            df = df.rename(columns={old: new})
    
    # Timestamp
    time_cols = ['timestamp', 'last_reported', 'date', 'datetime']
    for col in time_cols:
        if col in df.columns:
            df['timestamp'] = pd.to_datetime(df[col])
            break
    
    # Calcular ocupación
    if 'bikes_available' in df.columns and 'docks_available' in df.columns:
        df['capacity'] = df['bikes_available'] + df['docks_available']
        df['capacity'] = df['capacity'].replace(0, np.nan)
        df['occupancy'] = df['bikes_available'] / df['capacity']
        df['occupancy'] = df['occupancy'].fillna(0)
    
    # Features temporales
    if 'timestamp' in df.columns:
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_rush_hour'] = df['hour'].apply(
            lambda h: 1 if (7 <= h <= 9) or (14 <= h <= 16) or (18 <= h <= 20) else 0
        )
    
    df['city'] = 'barcelona'
    
    print(f"   📊 Columnas disponibles: {list(df.columns)[:10]}...")
    
    return df


def load_madrid_dataset():
    """
    Carga el dataset de Madrid (BiciMAD).
    """
    print("\n📂 Cargando Madrid...")
    
    mad_dir = DATA_RAW / "madrid"
    
    # Buscar archivos
    json_files = list(mad_dir.glob("*.json"))
    csv_files = list(mad_dir.glob("*.csv"))
    
    if not json_files and not csv_files:
        print("   ❌ No encontrado. Descarga desde EMT o Kaggle.")
        return None
    
    # Cargar
    if csv_files:
        df = pd.read_csv(csv_files[0], nrows=2_000_000)
    elif json_files:
        df = pd.read_json(json_files[0], lines=True, nrows=2_000_000)
    
    print(f"   ✅ Cargado: {len(df):,} filas")
    
    df['city'] = 'madrid'
    
    return df


def create_transfer_features(df):
    """
    Crea features optimizadas para transfer learning.
    Todas las features son RELATIVAS y NORMALIZADAS.
    """
    print("\n🔧 Creando features para transfer learning...")
    
    features = pd.DataFrame()
    
    # Identificadores
    features['city'] = df.get('city', 'unknown')
    features['timestamp'] = df.get('timestamp', pd.NaT)
    
    # =========== FEATURES TEMPORALES (CÍCLICAS) ===========
    if 'hour' in df.columns:
        features['hour'] = df['hour']
        features['hour_sin'], features['hour_cos'] = cyclic_encode(df['hour'], 24)
    
    if 'day_of_week' in df.columns:
        features['day_of_week'] = df['day_of_week']
        features['dow_sin'], features['dow_cos'] = cyclic_encode(df['day_of_week'], 7)
    
    # =========== FEATURES BINARIAS ===========
    features['is_weekend'] = df.get('is_weekend', 0)
    features['is_rush_hour'] = df.get('is_rush_hour', 0)
    features['is_holiday'] = df.get('is_holiday', 0)
    features['is_raining'] = df.get('is_raining', 0)
    
    # =========== FEATURES NORMALIZADAS ===========
    # Ocupación (ya es 0-1)
    if 'occupancy' in df.columns:
        features['occupancy'] = df['occupancy'].clip(0, 1)
    elif 'occupancy_proxy' in df.columns:
        features['occupancy'] = df['occupancy_proxy'].clip(0, 1)
    
    # Temperatura normalizada (z-score por ciudad)
    if 'temperature' in df.columns:
        temp_mean = df.groupby('city')['temperature'].transform('mean')
        temp_std = df.groupby('city')['temperature'].transform('std').replace(0, 1)
        features['temp_zscore'] = (df['temperature'] - temp_mean) / temp_std
    elif 'temp_normalized' in df.columns:
        features['temp_zscore'] = (df['temp_normalized'] - 0.5) * 2  # Centrar en 0
    
    # =========== TARGET ===========
    # Predecimos CAMBIO de ocupación en los próximos 30 MINUTOS
    # Esto es mucho más predecible que cambios instantáneos
    
    if 'occupancy' in features.columns:
        city = df.get('city', 'unknown').iloc[0] if len(df) > 0 else 'unknown'
        
        # Determinar cuántas observaciones = 30 minutos
        # Barcelona: cada 4 min → shift(8) = 32 min
        # DC: cada hora → shift(1) = 60 min (usamos 1 porque no tenemos más granularidad)
        if 'station_id' in df.columns:
            # Datos por estación (Barcelona, Coruña)
            cfg = load_config()
            # Para datos cada 4-5 min, shift(6) ≈ 30 min
            HORIZON_SHIFTS = cfg["horizon_shifts"]
            
            # Ordenar por estación y tiempo
            df_sorted = df.sort_values(['station_id', 'timestamp']).reset_index(drop=True)
            features = features.reindex(df_sorted.index)
            
            # Calcular ocupación futura
            future_occupancy = df_sorted.groupby('station_id')['occupancy'].shift(-HORIZON_SHIFTS)
            features['delta_30min'] = future_occupancy - features['occupancy']
            features['delta_next'] = features['delta_30min']  # Compatibilidad
            
            print(f"   📊 Target: cambio en ~30 min (shift={HORIZON_SHIFTS})")
        else:
            # Datos agregados (DC) - ya son por hora
            features['delta_next'] = features['occupancy'].shift(-1) - features['occupancy']
            print(f"   📊 Target: cambio en 1 hora")
        
        # Clasificación: ¿subirá o bajará significativamente?
        features['will_increase'] = (features['delta_next'] > 0.10).astype(int)  # >10%
        features['will_decrease'] = (features['delta_next'] < -0.10).astype(int) # <-10%
    
    # Limpiar NaN
    features = features.dropna(subset=['hour', 'occupancy'])
    
    print(f"   ✅ Features creadas: {list(features.columns)}")
    print(f"   📊 Filas finales: {len(features):,}")
    
    return features


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║  🔧 PREPROCESAMIENTO PARA TRANSFER LEARNING               ║
    ║                                                           ║
    ║  Unificando datasets para preentrenamiento                ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    all_data = []
    
    # Cargar cada dataset
    dc_df = load_dc_dataset()
    if dc_df is not None:
        dc_features = create_transfer_features(dc_df)
        all_data.append(dc_features)
        print(f"   → DC: {len(dc_features):,} filas procesadas")
    
    bcn_df = load_barcelona_dataset()
    if bcn_df is not None:
        bcn_features = create_transfer_features(bcn_df)
        all_data.append(bcn_features)
        print(f"   → Barcelona: {len(bcn_features):,} filas procesadas")
    
    mad_df = load_madrid_dataset()
    if mad_df is not None:
        mad_features = create_transfer_features(mad_df)
        all_data.append(mad_features)
        print(f"   → Madrid: {len(mad_features):,} filas procesadas")
    
    if not all_data:
        print("\n❌ No se encontraron datasets. Ejecuta download_datasets.py primero.")
        return
    
    # Unificar
    print("\n📦 Unificando datasets...")
    unified = pd.concat(all_data, ignore_index=True)
    
    # Guardar
    output_file = DATA_PROCESSED / "unified_pretrain_data.parquet"
    unified.to_parquet(output_file, index=False)
    
    print(f"\n✅ Dataset unificado guardado en: {output_file}")
    print(f"   📊 Total filas: {len(unified):,}")
    print(f"   📊 Ciudades: {unified['city'].unique()}")
    print(f"   📊 Features: {list(unified.columns)}")
    
    # Estadísticas
    print("\n📈 Estadísticas por ciudad:")
    print(unified.groupby('city').agg({
        'occupancy': ['mean', 'std', 'count'],
        'is_rush_hour': 'mean',
        'is_weekend': 'mean'
    }).to_string())
    
    print("\n" + "="*60)
    print("✅ Preprocesamiento completado!")
    print("="*60)
    print(f"""
    Siguiente paso - Entrenar modelo:
    
    python src/models/train_pretrain.py
    """)


if __name__ == "__main__":
    main()
