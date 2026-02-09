"""
=============================================================================
🧠 ENTRENAMIENTO DEL MODELO BASE (PRETRAINING)
=============================================================================
Entrena un modelo LightGBM con los datos de DC, Barcelona y Madrid
para luego hacer fine-tuning con Coruña.

Características:
- LightGBM optimizado para transfer learning
- Hiperparámetros que favorecen generalización
- Guarda el modelo para posterior fine-tuning

Ejecutar: python src/models/train_pretrain.py
=============================================================================
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Directorios
BASE_DIR = Path(__file__).parent.parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_pretrain_data():
    """Carga el dataset unificado para preentrenamiento."""
    data_file = DATA_PROCESSED / "unified_pretrain_data.parquet"
    
    if not data_file.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en {data_file}. "
            "Ejecuta primero: python src/preprocessing/prepare_data.py"
        )
    
    df = pd.read_parquet(data_file)
    print(f"📂 Cargado dataset: {len(df):,} filas")
    return df


def prepare_features(df):
    """
    Prepara X e y para el entrenamiento.
    Features seleccionadas para máxima transferibilidad.
    """
    # Features CORE que usaremos (deben existir en TODAS las ciudades)
    core_features = [
        'hour_sin', 'hour_cos',           # Hora (cíclico)
        'dow_sin', 'dow_cos',             # Día de semana (cíclico)
        'is_weekend',                      # Fin de semana
        'is_rush_hour',                    # Hora punta
        'is_raining',                      # Lluvia
        'occupancy',                       # Ocupación actual
    ]
    
    # Target: predecir cambio de ocupación
    target_col = 'delta_next'
    
    if target_col not in df.columns:
        raise ValueError(f"No se encontró la columna target '{target_col}'")
    
    # Filtrar solo las features core que existen
    available_cols = [col for col in core_features if col in df.columns]
    
    # IMPORTANTE: Solo añadir temp_zscore si está disponible para TODAS las filas
    # (no queremos perder millones de filas de Barcelona por falta de temperatura)
    if 'temp_zscore' in df.columns:
        temp_coverage = df['temp_zscore'].notna().mean()
        if temp_coverage > 0.9:  # Solo si >90% de filas tienen temperatura
            available_cols.append('temp_zscore')
            print(f"   ✅ temp_zscore incluida ({temp_coverage*100:.1f}% cobertura)")
        else:
            print(f"   ⚠️ temp_zscore excluida (solo {temp_coverage*100:.1f}% cobertura)")
    
    print(f"📊 Features a usar: {available_cols}")
    
    # Filtrar filas válidas (solo para las features core, NO temperatura opcional)
    valid_mask = df[available_cols + [target_col]].notna().all(axis=1)
    df_clean = df[valid_mask].copy()
    
    X = df_clean[available_cols]
    y = df_clean[target_col]
    
    # Info de ciudades para análisis
    cities = df_clean['city'] if 'city' in df_clean.columns else None
    
    print(f"📊 Datos de entrenamiento: {len(X):,} filas, {len(available_cols)} features")
    
    # Mostrar distribución por ciudad
    if cities is not None:
        print("📊 Distribución por ciudad:")
        for city in cities.unique():
            count = (cities == city).sum()
            print(f"   {city}: {count:,} filas")
    
    return X, y, cities, available_cols


def get_lgbm_params():
    """
    Hiperparámetros de LightGBM optimizados para transfer learning.
    
    Filosofía:
    - Árboles poco profundos → capturan patrones generales
    - Regularización alta → evita sobreajuste
    - Learning rate bajo → aprende lentamente pero bien
    """
    params = {
        # Estructura del árbol
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        
        # Tamaño del modelo (moderado para transfer learning)
        'num_leaves': 31,              # No muy grande para generalizar
        'max_depth': 6,                # Profundidad limitada
        'min_data_in_leaf': 100,       # Mínimo de datos por hoja (evita sobreajuste)
        
        # Regularización (importante para transfer learning)
        'lambda_l1': 0.1,              # L1 regularization
        'lambda_l2': 0.1,              # L2 regularization
        'feature_fraction': 0.8,       # Usar 80% de features por árbol
        'bagging_fraction': 0.8,       # Usar 80% de datos por árbol
        'bagging_freq': 5,             # Frecuencia de bagging
        
        # Learning rate bajo para aprender bien
        'learning_rate': 0.05,
        
        # Performance
        'n_jobs': -1,                  # Usar todos los cores
        'verbose': -1,
        'seed': 42,
    }
    
    return params


def train_model(X, y, cities=None):
    """
    Entrena el modelo LightGBM con validación.
    """
    print("\n" + "="*60)
    print("🧠 ENTRENANDO MODELO BASE")
    print("="*60)
    
    # Split estratificado por ciudad si disponible
    if cities is not None:
        # Validar con datos de todas las ciudades
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    print(f"📊 Train: {len(X_train):,} | Validation: {len(X_val):,}")
    
    # Parámetros
    params = get_lgbm_params()
    print(f"\n⚙️ Parámetros LightGBM:")
    for k, v in params.items():
        print(f"   {k}: {v}")
    
    # Crear datasets de LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Entrenar con early stopping
    print(f"\n🚀 Entrenando...")
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=callbacks
    )
    
    # Métricas
    print("\n📈 RESULTADOS:")
    
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
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
    
    # Feature importance
    print("\n📊 Importancia de Features:")
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    for _, row in importance.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.2f}")
    
    return model, metrics, importance


def evaluate_by_city(model, X, y, cities):
    """Evalúa el modelo separadamente por cada ciudad."""
    if cities is None:
        return
    
    print("\n📊 EVALUACIÓN POR CIUDAD:")
    print("-" * 40)
    
    for city in cities.unique():
        mask = cities == city
        X_city = X[mask]
        y_city = y[mask]
        
        if len(X_city) < 100:
            continue
        
        y_pred = model.predict(X_city)
        rmse = np.sqrt(mean_squared_error(y_city, y_pred))
        mae = mean_absolute_error(y_city, y_pred)
        r2 = r2_score(y_city, y_pred)
        
        print(f"\n   {city.upper()}:")
        print(f"      RMSE: {rmse:.4f}")
        print(f"      MAE:  {mae:.4f}")
        print(f"      R²:   {r2:.4f}")
        print(f"      N:    {len(X_city):,}")


def save_model(model, feature_cols, metrics):
    """Guarda el modelo y metadatos para fine-tuning."""
    import tempfile
    import shutil
    
    # Guardar modelo - workaround para caracteres especiales en Windows
    model_path = MODELS_DIR / "pretrained_lgbm.txt"
    
    # Guardar primero en carpeta temporal (sin caracteres especiales)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
        tmp_path = tmp.name
    
    model.save_model(tmp_path)
    
    # Mover a destino final
    shutil.move(tmp_path, str(model_path))
    print(f"\n💾 Modelo guardado en: {model_path}")
    
    # Guardar metadatos
    metadata = {
        'feature_cols': feature_cols,
        'metrics': metrics,
        'params': get_lgbm_params(),
        'trained_at': pd.Timestamp.now().isoformat()
    }
    
    metadata_path = MODELS_DIR / "pretrained_metadata.joblib"
    joblib.dump(metadata, metadata_path)
    print(f"💾 Metadatos guardados en: {metadata_path}")
    
    return model_path, metadata_path


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║  🧠 PREENTRENAMIENTO DEL MODELO BASE                      ║
    ║                                                           ║
    ║  LightGBM optimizado para transfer learning               ║
    ║  Datos: DC + Barcelona + Madrid                           ║
    ║  Target: Predecir cambio de ocupación                     ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Cargar datos
    df = load_pretrain_data()
    
    # Preparar features
    X, y, cities, feature_cols = prepare_features(df)
    
    # Entrenar
    model, metrics, importance = train_model(X, y, cities)
    
    # Evaluar por ciudad
    evaluate_by_city(model, X, y, cities)
    
    # Guardar
    model_path, metadata_path = save_model(model, feature_cols, metrics)
    
    print("\n" + "="*60)
    print("✅ PREENTRENAMIENTO COMPLETADO")
    print("="*60)
    print(f"""
    📦 Archivos generados:
       • {model_path}
       • {metadata_path}
    
    🚀 Siguiente paso - Fine-tuning con Coruña:
       
       1. Exporta tus datos de Google Sheets a CSV
       2. Guárdalos en: data/coruna/tracking_data.csv
       3. Ejecuta: python src/models/finetune_coruna.py
    """)


if __name__ == "__main__":
    main()
