# -*- coding: utf-8 -*-
"""
CLASIFICADOR FINAL: Transfer Learning + Features Mejoradas
===========================================================
Combina:
1. Preentrenamiento con Barcelona (para aprender patrones de vaciado)
2. Fine-tuning con Coruna + features de clima y tendencia
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
import tempfile
import shutil
import warnings
warnings.filterwarnings('ignore')

from src.utils.config import load_config
from src.utils.split import time_based_split

BASE_DIR = Path(__file__).parent.parent.parent
DATA_CORUNA = BASE_DIR / "data" / "coruna"
DATA_BCN = BASE_DIR / "data" / "raw" / "barcelona"
MODELS_DIR = BASE_DIR / "models"


def prepare_coruna_improved():
    """Prepara Coruna con features mejoradas."""
    df = pd.read_csv(DATA_CORUNA / "tracking_data.csv")
    df.columns = df.columns.str.lower()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['id', 'timestamp']).reset_index(drop=True)
    
    # Renombrar
    df = df.rename(columns={
        'bicis': 'bikes_available', 'docks': 'docks_available',
        'hora': 'hour', 'finde': 'is_weekend', 'punta': 'is_rush_hour',
        'temp': 'temperature', 'lluvia': 'rain', 'viento': 'wind'
    })
    
    # Features basicas
    df['occupancy'] = (df['bikes_available'] / df['capacidad'].replace(0, np.nan)).fillna(0).clip(0,1)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Features clima
    df['temp_norm'] = (df['temperature'] - df['temperature'].min()) / (df['temperature'].max() - df['temperature'].min() + 0.001) if 'temperature' in df.columns else 0.5
    df['is_raining'] = (df['rain'] > 0).astype(int) if 'rain' in df.columns else 0
    df['wind_norm'] = df['wind'] / (df['wind'].max() + 0.001) if 'wind' in df.columns else 0
    
    # Features tendencia
    df['bikes_lag_3'] = df.groupby('id')['bikes_available'].shift(3)
    df['trend_15min'] = ((df['bikes_available'] - df['bikes_lag_3']) / (df['capacidad'] + 0.001)).fillna(0)
    df['velocity'] = (df['bikes_available'] - df.groupby('id')['bikes_available'].shift(1)).fillna(0)
    
    # Target
    cfg = load_config()
    df['future_bikes'] = df.groupby('id')['bikes_available'].shift(-cfg["horizon_shifts"])
    df['will_be_empty'] = (df['future_bikes'] < cfg["empty_threshold"]).astype(int)
    
    return df.dropna(subset=['future_bikes', 'trend_15min'])


def prepare_barcelona():
    """Prepara Barcelona para preentrenamiento."""
    csv_files = list(DATA_BCN.glob("*.csv"))
    if not csv_files:
        return None
    
    df = pd.read_csv(csv_files[0], nrows=500000)
    df.columns = df.columns.str.lower()
    
    if 'last_reported' in df.columns:
        df['timestamp'] = pd.to_datetime(df['last_reported'], unit='s')
    df = df.sort_values(['station_id', 'timestamp']).reset_index(drop=True)
    
    # Features
    df['hour'] = df['timestamp'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['is_weekend'] = df['timestamp'].dt.dayofweek.isin([5,6]).astype(int)
    df['is_rush_hour'] = df['hour'].apply(lambda h: 1 if 7<=h<=9 or 17<=h<=20 else 0)
    
    df['bikes_available'] = df.get('num_bikes_available', df.get('bikes_available', 0))
    df['docks_available'] = df.get('num_docks_available', df.get('docks_available', 0))
    df['capacity'] = df['bikes_available'] + df['docks_available']
    df['occupancy'] = (df['bikes_available'] / df['capacity'].replace(0, np.nan)).fillna(0)
    
    # Features adicionales (con valores por defecto para BCN - no tiene clima)
    df['temp_norm'] = 0.5
    df['is_raining'] = 0
    df['wind_norm'] = 0
    
    # Tendencia
    df['bikes_lag_3'] = df.groupby('station_id')['bikes_available'].shift(3)
    df['trend_15min'] = ((df['bikes_available'] - df['bikes_lag_3']) / (df['capacity'] + 0.001)).fillna(0)
    df['velocity'] = (df['bikes_available'] - df.groupby('station_id')['bikes_available'].shift(1)).fillna(0)
    
    # Target
    cfg = load_config()
    df['future_bikes'] = df.groupby('station_id')['bikes_available'].shift(-cfg["horizon_shifts"])
    df['will_be_empty'] = (df['future_bikes'] < cfg["empty_threshold"]).astype(int)
    
    return df.dropna(subset=['future_bikes', 'trend_15min'])


def main():
    print("="*60)
    print("CLASIFICADOR FINAL: Transfer + Features Mejoradas")
    print("="*60)
    
    # Features a usar
    features = [
        'hour_sin', 'hour_cos', 'is_weekend', 'is_rush_hour',
        'occupancy', 'bikes_available', 'docks_available',
        'temp_norm', 'is_raining', 'wind_norm',
        'trend_15min', 'velocity'
    ]
    
    # Preparar datos
    print("\n1. Preparando datos de Coruna...")
    df_coruna = prepare_coruna_improved()
    print(f"   Coruna: {len(df_coruna):,} filas, {df_coruna['will_be_empty'].mean()*100:.1f}% vacias")
    
    print("\n2. Preparando datos de Barcelona...")
    df_bcn = prepare_barcelona()
    if df_bcn is not None:
        print(f"   Barcelona: {len(df_bcn):,} filas, {df_bcn['will_be_empty'].mean()*100:.1f}% vacias")
    
    # Split Coruna
    X_coruna = df_coruna[features]
    y_coruna = df_coruna['will_be_empty']
    cfg = load_config()
    train_idx, test_idx = time_based_split(
        df_coruna, group_col='id', time_col='timestamp', train_fraction=cfg["train_fraction"]
    )
    X_train, X_test = X_coruna.loc[train_idx], X_coruna.loc[test_idx]
    y_train, y_test = y_coruna.loc[train_idx], y_coruna.loc[test_idx]
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 6,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'n_jobs': -1,
        'verbose': -1,
        'seed': 42,
        'is_unbalance': True
    }
    
    # ========== PREENTRENAMIENTO CON BARCELONA ==========
    print("\n3. Preentrenamiento con Barcelona...")
    X_bcn = df_bcn[features]
    y_bcn = df_bcn['will_be_empty']
    
    pretrain_data = lgb.Dataset(X_bcn, label=y_bcn)
    pretrained = lgb.train(params, pretrain_data, num_boost_round=150)
    print("   Preentrenamiento completado")
    
    # ========== FINE-TUNING CON CORUNA ==========
    print("\n4. Fine-tuning con Coruna + features mejoradas...")
    
    params_ft = params.copy()
    params_ft['learning_rate'] = 0.01  # Mas bajo para fine-tuning
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_test, label=y_test)
    
    model = lgb.train(
        params_ft,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        init_model=pretrained,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    
    # ========== EVALUACION ==========
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > cfg["decision_threshold"]).astype(int)
    
    print("\n" + "="*60)
    print("RESULTADOS FINALES")
    print("="*60)
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1': f1_score(y_test, y_pred, zero_division=0)
    }
    
    for k, v in metrics.items():
        print(f"   {k}: {v*100:.1f}%")
    
    # Comparar con antes
    print("\n" + "-"*40)
    print("COMPARACION vs modelo anterior (Transfer sin features extra)")
    print("-"*40)
    print(f"{'Metrica':<12} | {'Antes':>10} | {'Ahora':>10} | {'Mejora':>10}")
    print("-"*50)
    
    antes = {'Accuracy': 94.6, 'Precision': 68.1, 'Recall': 50.0, 'F1': 57.7}
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1']:
        v1 = antes[metric]
        v2 = metrics[metric] * 100
        diff = v2 - v1
        sign = "+" if diff > 0 else ""
        print(f"{metric:<12} | {v1:>9.1f}% | {v2:>9.1f}% | {sign}{diff:>8.1f}%")
    
    # Feature importance
    print("\n" + "-"*40)
    print("Importancia de features:")
    print("-"*40)
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    for _, row in importance.iterrows():
        bar = "#" * int(row['importance'] / importance['importance'].max() * 20)
        print(f"   {row['feature']:<20} {bar}")
    
    # Guardar modelo
    model_path = MODELS_DIR / "classifier_final.txt"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
        tmp_path = tmp.name
    model.save_model(tmp_path)
    shutil.move(tmp_path, str(model_path))
    print(f"\nModelo guardado: {model_path}")


if __name__ == "__main__":
    main()
