# -*- coding: utf-8 -*-
"""
COMPARACION: Umbral Fijo vs Umbral Porcentual
==============================================
Compara el rendimiento del modelo usando:
1. Umbral fijo (< 5 bicis)
2. Umbral porcentual (< 20% de capacidad)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.utils.config import load_config
from src.utils.split import time_based_split

BASE_DIR = Path(__file__).parent.parent.parent
DATA_CORUNA = BASE_DIR / "data" / "coruna"


def prepare_data_fixed(threshold_empty=5):
    """Prepara datos con umbral FIJO."""
    df = pd.read_csv(DATA_CORUNA / "tracking_data.csv")
    df.columns = df.columns.str.lower()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['id', 'timestamp']).reset_index(drop=True)
    
    df = df.rename(columns={'bicis': 'bikes_available', 'docks': 'docks_available',
                            'hora': 'hour', 'finde': 'is_weekend', 'punta': 'is_rush_hour'})
    
    df['occupancy'] = (df['bikes_available'] / df['capacidad'].replace(0, np.nan)).fillna(0).clip(0,1)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    cfg = load_config()
    df['future_bikes'] = df.groupby('id')['bikes_available'].shift(-cfg["horizon_shifts"])
    
    # UMBRAL FIJO
    df['will_be_empty'] = (df['future_bikes'] < threshold_empty).astype(int)
    
    return df.dropna(subset=['future_bikes'])


def prepare_data_percentage(threshold_pct=0.20):
    """Prepara datos con umbral PORCENTUAL."""
    df = pd.read_csv(DATA_CORUNA / "tracking_data.csv")
    df.columns = df.columns.str.lower()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['id', 'timestamp']).reset_index(drop=True)
    
    df = df.rename(columns={'bicis': 'bikes_available', 'docks': 'docks_available',
                            'hora': 'hour', 'finde': 'is_weekend', 'punta': 'is_rush_hour'})
    
    df['occupancy'] = (df['bikes_available'] / df['capacidad'].replace(0, np.nan)).fillna(0).clip(0,1)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    cfg = load_config()
    df['future_bikes'] = df.groupby('id')['bikes_available'].shift(-cfg["horizon_shifts"])
    
    # UMBRAL PORCENTUAL: vacia si < X% de su capacidad
    df['threshold_for_station'] = df['capacidad'] * threshold_pct
    df['will_be_empty'] = (df['future_bikes'] < df['threshold_for_station']).astype(int)
    
    return df.dropna(subset=['future_bikes'])


def train_and_evaluate(df, name):
    """Entrena modelo y devuelve metricas."""
    features = ['hour_sin', 'hour_cos', 'is_weekend', 'is_rush_hour',
                'occupancy', 'bikes_available', 'docks_available']
    
    cfg = load_config()
    
    X = df[features]
    y = df['will_be_empty']
    
    train_idx, test_idx = time_based_split(
        df, group_col='id', time_col='timestamp', train_fraction=cfg["train_fraction"]
    )
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    
    params = {'objective': 'binary', 'metric': 'binary_logloss', 'num_leaves': 31,
              'learning_rate': 0.05, 'verbose': -1, 'seed': 42, 'is_unbalance': True}
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_test, label=y_test)
    
    model = lgb.train(params, train_data, num_boost_round=300, valid_sets=[val_data],
                      callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    
    # Buscar mejor umbral de decision
    y_proba = model.predict(X_test)
    
    best_f1 = 0
    best_thresh = 0.5
    best_metrics = {}
    
    for thresh in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_proba > thresh).astype(int)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_metrics = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, zero_division=0),
                'Recall': recall_score(y_test, y_pred, zero_division=0),
                'F1': f1
            }
    
    pct_empty = y.mean() * 100
    
    return {
        'name': name,
        'pct_empty': pct_empty,
        'best_threshold': best_thresh,
        'metrics': best_metrics
    }


def main():
    print("="*65)
    print("COMPARACION: Umbral Fijo vs Umbral Porcentual")
    print("="*65)
    
    results = []
    
    # 1. UMBRAL FIJO (< 5 bicis)
    print("\n[1/2] Entrenando con UMBRAL FIJO (< 5 bicis)...")
    df_fixed = prepare_data_fixed(threshold_empty=5)
    result_fixed = train_and_evaluate(df_fixed, "Fijo (< 5 bicis)")
    results.append(result_fixed)
    
    # 2. UMBRAL PORCENTUAL (< 20% capacidad)
    print("[2/2] Entrenando con UMBRAL PORCENTUAL (< 20% capacidad)...")
    df_pct = prepare_data_percentage(threshold_pct=0.20)
    result_pct = train_and_evaluate(df_pct, "Porcentual (< 20%)")
    results.append(result_pct)
    
    # RESULTADOS
    print("\n" + "="*65)
    print("RESULTADOS")
    print("="*65)
    
    print(f"\n{'Metodo':<22} | {'% Vacias':<10} | {'Umbral':<8} | {'Prec':<8} | {'Recall':<8} | {'F1':<8}")
    print("-"*65)
    
    for r in results:
        m = r['metrics']
        print(f"{r['name']:<22} | {r['pct_empty']:>8.1f}% | {r['best_threshold']:>8.2f} | {m['Precision']*100:>6.1f}% | {m['Recall']*100:>6.1f}% | {m['F1']*100:>6.1f}%")
    
    # GANADOR
    print("\n" + "="*65)
    winner = max(results, key=lambda x: x['metrics']['F1'])
    print(f"GANADOR: {winner['name']} (F1: {winner['metrics']['F1']*100:.1f}%)")
    print("="*65)
    
    # Diferencia
    diff = results[1]['metrics']['F1'] - results[0]['metrics']['F1']
    if diff > 0:
        print(f"\nEl umbral PORCENTUAL mejora el F1 en +{diff*100:.1f}%")
    elif diff < 0:
        print(f"\nEl umbral FIJO es mejor por +{abs(diff)*100:.1f}%")
    else:
        print("\nAmbos metodos tienen el mismo rendimiento")


if __name__ == "__main__":
    main()
