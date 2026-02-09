# -*- coding: utf-8 -*-
"""
=============================================================================
FINE-TUNING CON DATOS DE CORUNA
=============================================================================
Adapta el modelo preentrenado a los datos específicos de Coruña.

Estrategia:
- Cargar modelo preentrenado
- Continuar entrenamiento con datos de Coruña
- Learning rate muy bajo para no "olvidar" lo aprendido
- Early stopping agresivo

Ejecutar: python src/models/finetune_coruna.py
=============================================================================
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.utils.config import load_config

# Directorios
BASE_DIR = Path(__file__).parent.parent.parent
DATA_CORUNA = BASE_DIR / "data" / "coruna"
MODELS_DIR = BASE_DIR / "models"
DATA_PROCESSED = BASE_DIR / "data" / "processed"


def load_coruna_data():
    """
    Carga los datos de Coruña exportados desde Google Sheets.
    
    Formato esperado (CSV exportado de tu sheet "Estaciones"):
    Timestamp, ID, Nombre, Lat, Lon, Capacidad, Bicis, Docks, 
    Ocupación, Vacía, Llena, Delta, Hora, Día, Finde, Festivo, Punta, Turismo
    """
    print("\n📂 Buscando datos de Coruña...")
    
    # Buscar archivos CSV en la carpeta de Coruña
    csv_files = list(DATA_CORUNA.glob("*.csv"))
    
    if not csv_files:
        print(f"""
    ❌ No se encontraron datos de Coruña.
    
    📋 Instrucciones:
    1. Ve a tu Google Sheet "Bicicoruna_tracking"
    2. Selecciona la hoja "Estaciones"
    3. Archivo → Descargar → Valores separados por comas (.csv)
    4. Guarda el archivo en: {DATA_CORUNA}/
    5. Ejecuta este script de nuevo
        """)
        return None
    
    # Cargar el archivo más reciente
    csv_file = max(csv_files, key=lambda x: x.stat().st_mtime)
    print(f"   📄 Cargando: {csv_file.name}")
    
    # Intentar detectar el separador (coma o punto y coma)
    with open(csv_file, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        separator = ';' if ';' in first_line else ','
    
    df = pd.read_csv(csv_file, sep=separator, decimal=',')
    print(f"   ✅ Cargado: {len(df):,} filas")
    print(f"   📊 Columnas: {list(df.columns)}")
    
    return df


def prepare_coruna_features(df):
    """
    Prepara las features de Coruña en el mismo formato que el preentrenamiento.
    """
    print("\n🔧 Preparando features de Coruña...")
    
    # Normalizar nombres de columnas
    df.columns = df.columns.str.strip().str.lower()
    
    # Mapeo de columnas del Google Sheet
    column_map = {
        'timestamp': 'timestamp',
        'id': 'station_id',
        'nombre': 'station_name',
        'capacidad': 'capacity',
        'bicis': 'bikes_available',
        'docks': 'docks_available',
        'ocupación': 'occupancy_raw',
        'hora': 'hour',
        'día': 'day_of_week',
        'finde': 'is_weekend',
        'festivo': 'is_holiday',
        'punta': 'is_rush_hour',
        'turismo': 'is_tourist_season',
        'delta': 'delta_bikes'
    }
    
    # Renombrar columnas que existan
    for old, new in column_map.items():
        if old in df.columns:
            df = df.rename(columns={old: new})
    
    # Timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calcular ocupación correctamente (0-1)
    if 'occupancy_raw' in df.columns:
        # La ocupación del sheet viene multiplicada por 1000
        df['occupancy'] = df['occupancy_raw'] / 1000.0
        df['occupancy'] = df['occupancy'].clip(0, 1)
    elif 'bikes_available' in df.columns and 'capacity' in df.columns:
        df['occupancy'] = df['bikes_available'] / df['capacity'].replace(0, np.nan)
        df['occupancy'] = df['occupancy'].fillna(0).clip(0, 1)
    
    # Features cíclicas
    if 'hour' in df.columns:
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    if 'day_of_week' in df.columns:
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Binarias
    for col in ['is_weekend', 'is_rush_hour', 'is_holiday']:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    # is_raining no viene en los datos (por ahora asumimos 0)
    df['is_raining'] = 0  # TODO: añadir cuando tengamos datos de lluvia
    
    cfg = load_config()
    horizon_shifts = cfg["horizon_shifts"]

    # Target: delta de ocupacion en el mismo horizonte que el preentrenamiento
    df = df.sort_values(['station_id', 'timestamp'])
    df['delta_next'] = df.groupby('station_id')['occupancy'].shift(-horizon_shifts) - df['occupancy']
    
    print(f"   ✅ Features creadas")
    print(f"   📊 Estadísticas de ocupación:")
    print(f"      Media: {df['occupancy'].mean():.3f}")
    print(f"      Std:   {df['occupancy'].std():.3f}")
    print(f"      Min:   {df['occupancy'].min():.3f}")
    print(f"      Max:   {df['occupancy'].max():.3f}")
    
    return df


def load_pretrained_model():
    """Carga el modelo preentrenado y sus metadatos."""
    import tempfile
    import shutil
    
    model_path = MODELS_DIR / "pretrained_lgbm.txt"
    metadata_path = MODELS_DIR / "pretrained_metadata.joblib"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontro el modelo preentrenado en {model_path}. "
            "Ejecuta primero: python src/models/train_pretrain.py"
        )
    
    # Workaround: copiar a ruta temporal sin caracteres especiales
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
        tmp_path = tmp.name
    shutil.copy(str(model_path), tmp_path)
    
    model = lgb.Booster(model_file=tmp_path)
    metadata = joblib.load(metadata_path)
    
    # Limpiar archivo temporal
    Path(tmp_path).unlink()
    
    print(f"Modelo preentrenado cargado")
    print(f"   Features: {metadata['feature_cols']}")
    print(f"   Val RMSE original: {metadata['metrics']['Val RMSE']:.4f}")
    
    return model, metadata


def finetune_model(df, pretrained_model, metadata):
    """
    Fine-tuning del modelo con datos de Coruña.
    
    Estrategia:
    - Learning rate muy bajo
    - Pocas iteraciones
    - Early stopping agresivo
    """
    print("\n" + "="*60)
    print("🎯 FINE-TUNING CON DATOS DE CORUÑA")
    print("="*60)
    
    # Obtener columnas de features del modelo preentrenado
    feature_cols = metadata['feature_cols']
    
    # Verificar que tenemos todas las features
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️ Features faltantes: {missing_cols}")
        # Crear columnas faltantes con valores por defecto
        for col in missing_cols:
            df[col] = 0
    
    # Preparar X e y
    valid_mask = df[feature_cols + ['delta_next']].notna().all(axis=1)
    df_clean = df[valid_mask].copy()
    
    X = df_clean[feature_cols]
    y = df_clean['delta_next']
    
    print(f"📊 Datos de Coruña: {len(X):,} filas")
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"   Train: {len(X_train):,} | Validation: {len(X_val):,}")
    
    # Parámetros de fine-tuning (más conservadores)
    finetune_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 6,
        'min_data_in_leaf': 50,  # Menos restrictivo para pocos datos
        'lambda_l1': 0.2,
        'lambda_l2': 0.2,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'learning_rate': 0.01,  # MUY bajo para fine-tuning
        'n_jobs': -1,
        'verbose': -1,
        'seed': 42,
    }
    
    # Datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Fine-tuning: continuar desde el modelo preentrenado
    print("\n🚀 Fine-tuning...")
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=30),
        lgb.log_evaluation(period=50)
    ]
    
    # Entrenar modelo nuevo (LightGBM no soporta continuar entrenamiento directo,
    # pero los pesos están "inspirados" por la estructura aprendida)
    finetuned_model = lgb.train(
        finetune_params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=callbacks,
        init_model=pretrained_model  # Inicializar desde preentrenado
    )
    
    # Métricas
    print("\n📈 RESULTADOS FINE-TUNING:")
    
    y_pred_train = finetuned_model.predict(X_train)
    y_pred_val = finetuned_model.predict(X_val)
    
    metrics = {
        'Train RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'Val RMSE': np.sqrt(mean_squared_error(y_val, y_pred_val)),
        'Train MAE': mean_absolute_error(y_train, y_pred_train),
        'Val MAE': mean_absolute_error(y_val, y_pred_val),
        'Train R²': r2_score(y_train, y_pred_train),
        'Val R²': r2_score(y_val, y_pred_val),
    }
    
    for name, value in metrics.items():
        print(f"   {name}: {value:.4f}")
    
    # Comparar con preentrenamiento
    print("\n📊 COMPARACIÓN:")
    print(f"   Preentrenado Val RMSE: {metadata['metrics']['Val RMSE']:.4f}")
    print(f"   Fine-tuned Val RMSE:   {metrics['Val RMSE']:.4f}")
    
    improvement = (metadata['metrics']['Val RMSE'] - metrics['Val RMSE']) / metadata['metrics']['Val RMSE'] * 100
    if improvement > 0:
        print(f"   ✅ Mejora: {improvement:.1f}%")
    else:
        print(f"   ⚠️ Cambio: {improvement:.1f}%")
    
    return finetuned_model, metrics


def predict_for_dashboard(model, df, feature_cols):
    """
    Genera predicciones para usar en el dashboard.
    """
    print("\n📊 Generando predicciones para dashboard...")
    
    # Preparar datos
    df_pred = df.dropna(subset=feature_cols + ['timestamp', 'station_id']).copy()
    
    X = df_pred[feature_cols]
    df_pred['predicted_delta'] = model.predict(X)
    
    # Clasificar predicciones
    df_pred['trend'] = pd.cut(
        df_pred['predicted_delta'],
        bins=[-np.inf, -0.1, -0.02, 0.02, 0.1, np.inf],
        labels=['bajada_fuerte', 'bajada', 'estable', 'subida', 'subida_fuerte']
    )
    
    # Guardar para dashboard
    output_file = DATA_PROCESSED / "coruna_predictions.csv"
    df_pred.to_csv(output_file, index=False)
    print(f"   💾 Predicciones guardadas en: {output_file}")
    
    return df_pred


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║  🎯 FINE-TUNING PARA CORUÑA                               ║
    ║                                                           ║
    ║  Adaptando el modelo preentrenado a BiciCoruña            ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Cargar datos de Coruña
    df_raw = load_coruna_data()
    if df_raw is None:
        return
    
    # Preparar features
    df = prepare_coruna_features(df_raw)
    
    # Cargar modelo preentrenado
    pretrained_model, metadata = load_pretrained_model()
    
    # Fine-tuning
    finetuned_model, metrics = finetune_model(df, pretrained_model, metadata)
    
    # Guardar modelo fine-tuned (workaround para ruta con ñ)
    import tempfile
    import shutil
    model_path = MODELS_DIR / "coruna_finetuned_lgbm.txt"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
        tmp_path = tmp.name
    finetuned_model.save_model(tmp_path)
    shutil.move(tmp_path, str(model_path))
    print(f"\nModelo fine-tuned guardado en: {model_path}")
    
    # Guardar metadatos
    finetuned_metadata = {
        'feature_cols': metadata['feature_cols'],
        'metrics': metrics,
        'pretrained_metrics': metadata['metrics'],
        'trained_at': pd.Timestamp.now().isoformat()
    }
    joblib.dump(finetuned_metadata, MODELS_DIR / "coruna_finetuned_metadata.joblib")
    
    # Generar predicciones para dashboard
    predict_for_dashboard(finetuned_model, df, metadata['feature_cols'])
    
    print("\n" + "="*60)
    print("✅ FINE-TUNING COMPLETADO")
    print("="*60)
    print(f"""
    📦 Archivos generados:
       • {model_path}
       • {MODELS_DIR}/coruna_finetuned_metadata.joblib
       • {DATA_PROCESSED}/coruna_predictions.csv
    
    🚀 Siguiente paso - Dashboard:
       
       python dashboard/app.py
    """)


if __name__ == "__main__":
    main()
