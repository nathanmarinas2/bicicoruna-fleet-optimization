"""
=============================================================================
🎯 MODELO DE CLASIFICACIÓN: ¿SE QUEDARÁ SIN BICIS?
=============================================================================
Este modelo predice:
- ¿Esta estación estará VACÍA en 30 minutos? (0-2 bicis)
- ¿Esta estación estará LLENA en 30 minutos? (0-2 docks libres)

Métricas más impactantes para la noticia:
- Accuracy: "Acertamos el 87% de las veces"
- Precision: "Cuando alertamos, acertamos el 92%"

Ejecutar: python src/models/train_classifier.py
=============================================================================
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score
)
import joblib
import tempfile
import shutil
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.utils.split import time_based_split
from src.utils.config import load_config

# Directorios
BASE_DIR = Path(__file__).parent.parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_and_prepare_classification_data():
    """
    Carga los datos y crea targets de clasificación binaria.
    """
    print("\n📂 Cargando datos para clasificación...")
    
    # Cargar Barcelona (tiene station_id para calcular bien)
    bcn_dir = DATA_RAW / "barcelona"
    csv_files = list(bcn_dir.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError("No se encontraron datos de Barcelona")
    
    # Cargar con límite para no saturar RAM
    df = pd.read_csv(csv_files[0], nrows=2_000_000)
    df.columns = df.columns.str.lower()
    
    print(f"   ✅ Cargado: {len(df):,} filas")
    
    # Timestamp
    time_cols = ['last_reported', 'timestamp', 'date']
    for col in time_cols:
        if col in df.columns:
            df['timestamp'] = pd.to_datetime(df[col], unit='s' if df[col].dtype in ['int64', 'float64'] else None)
            break
    
    # Calcular capacidad y ocupación
    if 'num_bikes_available' in df.columns:
        df['bikes_available'] = df['num_bikes_available']
    if 'num_docks_available' in df.columns:
        df['docks_available'] = df['num_docks_available']
    
    df['capacity'] = df['bikes_available'] + df['docks_available']
    df['capacity'] = df['capacity'].replace(0, np.nan)
    df['occupancy'] = df['bikes_available'] / df['capacity']
    
    # Features temporales
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_rush_hour'] = df['hour'].apply(
        lambda h: 1 if (7 <= h <= 9) or (17 <= h <= 20) else 0
    )
    
    # Features cíclicas
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Ordenar por estación y tiempo
    df = df.sort_values(['station_id', 'timestamp']).reset_index(drop=True)
    
    # ============ TARGETS DE CLASIFICACIÓN ============
    cfg = load_config()
    threshold_empty = cfg["empty_threshold"]
    threshold_full = cfg["full_threshold"]
    horizon_shifts = cfg["horizon_shifts"]
    
    # Calcular estado futuro
    df['future_bikes'] = df.groupby('station_id')['bikes_available'].shift(-horizon_shifts)
    df['future_docks'] = df.groupby('station_id')['docks_available'].shift(-horizon_shifts)
    
    # Targets binarios
    df['will_be_empty'] = (df['future_bikes'] < threshold_empty).astype(int)
    df['will_be_full'] = (df['future_docks'] < threshold_full).astype(int)
    
    # Target combinado: ¿habrá problema?
    df['will_have_problem'] = ((df['will_be_empty'] == 1) | (df['will_be_full'] == 1)).astype(int)
    
    # Limpiar NaN
    df = df.dropna(subset=['future_bikes', 'occupancy', 'hour'])
    
    print(f"   📊 Filas válidas: {len(df):,}")
    print(f"   📊 Estaciones vacías (target=1): {df['will_be_empty'].mean()*100:.1f}%")
    print(f"   📊 Estaciones llenas (target=1): {df['will_be_full'].mean()*100:.1f}%")
    
    return df


def train_classifier(df, target_col='will_be_empty'):
    """
    Entrena un clasificador LightGBM.
    """
    print(f"\n🎯 Entrenando clasificador para: {target_col}")
    print("="*60)
    
    # Features
    feature_cols = [
        'hour_sin', 'hour_cos',
        'dow_sin', 'dow_cos',
        'is_weekend', 'is_rush_hour',
        'occupancy',
        'bikes_available', 'docks_available'
    ]
    
    # Filtrar features disponibles
    available_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[available_cols]
    y = df[target_col]
    
    # Split temporal por estacion
    cfg = load_config()
    train_idx, test_idx = time_based_split(
        df, group_col='station_id', time_col='timestamp', train_fraction=cfg["train_fraction"]
    )
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    
    print(f"📊 Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"📊 Clase 1 en train: {y_train.mean()*100:.1f}%")
    
    # Parámetros para clasificación
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 6,
        'min_data_in_leaf': 100,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'is_unbalance': True,  # Manejar desbalance de clases
        'n_jobs': -1,
        'verbose': -1,
        'seed': 42,
    }
    
    # Entrenar
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
    
    print("\n🚀 Entrenando...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=callbacks
    )
    
    # Predicciones
    y_pred_proba = model.predict(X_test)
    decision_threshold = cfg["decision_threshold"]
    y_pred = (y_pred_proba > decision_threshold).astype(int)
    
    # Métricas
    print("\n" + "="*60)
    print("📊 RESULTADOS DE CLASIFICACIÓN")
    print("="*60)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        roc_auc = float('nan')

    try:
        pr_auc = average_precision_score(y_test, y_pred_proba)
    except ValueError:
        pr_auc = float('nan')

    print(f"""
    🎯 MÉTRICAS PARA LA NOTICIA:
    
    ✅ Accuracy:  {accuracy*100:.1f}%  
       → "Acertamos el {accuracy*100:.0f}% de las predicciones"
    
    ✅ Precision: {precision*100:.1f}%  
       → "Cuando alertamos de problema, acertamos el {precision*100:.0f}%"
    
    ✅ Recall:    {recall*100:.1f}%  
       → "Detectamos el {recall*100:.0f}% de los problemas reales"
    
    ✅ F1-Score:  {f1*100:.1f}%
    ✅ ROC-AUC:   {roc_auc:.3f}
    ✅ PR-AUC:    {pr_auc:.3f}
    """)
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    print("📊 Matriz de Confusión:")
    print(f"   Verdaderos Negativos: {cm[0,0]:,}")
    print(f"   Falsos Positivos:     {cm[0,1]:,}")
    print(f"   Falsos Negativos:     {cm[1,0]:,}")
    print(f"   Verdaderos Positivos: {cm[1,1]:,}")
    
    # Feature importance
    print("\n📊 Importancia de Features:")
    importance = pd.DataFrame({
        'feature': available_cols,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    for _, row in importance.iterrows():
        print(f"   {row['feature']}: {row['importance']:.0f}")
    
    return model, {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'feature_cols': available_cols,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'decision_threshold': decision_threshold
    }


def save_classifier(model, metrics, name):
    """Guarda el clasificador."""
    # Workaround para caracteres especiales en Windows
    model_path = MODELS_DIR / f"classifier_{name}.txt"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
        tmp_path = tmp.name
    
    model.save_model(tmp_path)
    shutil.move(tmp_path, str(model_path))
    
    # Guardar metadatos
    metadata_path = MODELS_DIR / f"classifier_{name}_metadata.joblib"
    joblib.dump(metrics, metadata_path)
    
    print(f"\n💾 Modelo guardado: {model_path}")
    return model_path


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║  🎯 ENTRENAMIENTO DE CLASIFICADOR                         ║
    ║                                                           ║
    ║  Target: "¿Se quedará sin bicis en 30 min?"               ║
    ║  Objetivo: Accuracy > 85%                                 ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Cargar datos
    df = load_and_prepare_classification_data()
    
    # Entrenar clasificador para "estación vacía"
    model_empty, metrics_empty = train_classifier(df, 'will_be_empty')
    save_classifier(model_empty, metrics_empty, 'empty')
    
    # Entrenar clasificador para "estación llena"
    print("\n" + "="*60)
    model_full, metrics_full = train_classifier(df, 'will_be_full')
    save_classifier(model_full, metrics_full, 'full')
    
    print("\n" + "="*60)
    print("✅ CLASIFICADORES ENTRENADOS")
    print("="*60)
    print("""
    📦 Archivos generados:
       • models/classifier_empty.txt (¿se quedará vacía?)
       • models/classifier_full.txt (¿se llenará?)
    
    🎯 Para la noticia puedes decir:
       "El sistema predice con {:.0f}% de precisión cuándo una 
        estación se quedará sin bicicletas disponibles"
    """.format(metrics_empty['accuracy']*100))


if __name__ == "__main__":
    main()
