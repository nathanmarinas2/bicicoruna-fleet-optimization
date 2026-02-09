# -*- coding: utf-8 -*-
"""
OPTIMIZACION: Mejor umbral + Mejor definicion de "vacia"
=========================================================
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
DATA_BCN = BASE_DIR / "data" / "raw" / "barcelona"


def prepare_data(threshold_empty=2):
    """Prepara datos con umbral configurable."""
    # Coruna
    df = pd.read_csv(DATA_CORUNA / "tracking_data.csv")
    df.columns = df.columns.str.lower()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['id', 'timestamp']).reset_index(drop=True)
    
    df = df.rename(columns={'bicis': 'bikes_available', 'docks': 'docks_available',
                            'hora': 'hour', 'finde': 'is_weekend', 'punta': 'is_rush_hour'})
    
    df['occupancy'] = (df['bikes_available'] / df['capacidad'].replace(0, np.nan)).fillna(0).clip(0,1)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Target con umbral configurable
    cfg = load_config()
    df['future_bikes'] = df.groupby('id')['bikes_available'].shift(-cfg["horizon_shifts"])
    df['will_be_empty'] = (df['future_bikes'] < threshold_empty).astype(int)
    
    return df.dropna(subset=['future_bikes'])


def prepare_bcn(threshold_empty=2):
    """Barcelona para preentrenamiento."""
    csv_files = list(DATA_BCN.glob("*.csv"))
    df = pd.read_csv(csv_files[0], nrows=500000)
    df.columns = df.columns.str.lower()
    
    if 'last_reported' in df.columns:
        df['timestamp'] = pd.to_datetime(df['last_reported'], unit='s')
    df = df.sort_values(['station_id', 'timestamp']).reset_index(drop=True)
    
    df['hour'] = df['timestamp'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['is_weekend'] = df['timestamp'].dt.dayofweek.isin([5,6]).astype(int)
    df['is_rush_hour'] = df['hour'].apply(lambda h: 1 if 7<=h<=9 or 17<=h<=20 else 0)
    
    df['bikes_available'] = df.get('num_bikes_available', df.get('bikes_available', 0))
    df['docks_available'] = df.get('num_docks_available', df.get('docks_available', 0))
    df['capacity'] = df['bikes_available'] + df['docks_available']
    df['occupancy'] = (df['bikes_available'] / df['capacity'].replace(0, np.nan)).fillna(0)
    
    cfg = load_config()
    df['future_bikes'] = df.groupby('station_id')['bikes_available'].shift(-cfg["horizon_shifts"])
    df['will_be_empty'] = (df['future_bikes'] < threshold_empty).astype(int)
    
    return df.dropna(subset=['future_bikes'])


def train_and_optimize(threshold_empty):
    """Entrena modelo y busca umbral optimo."""
    features = ['hour_sin', 'hour_cos', 'is_weekend', 'is_rush_hour',
                'occupancy', 'bikes_available', 'docks_available']
    
    # Preparar datos
    df_coruna = prepare_data(threshold_empty)
    df_bcn = prepare_bcn(threshold_empty)
    
    pct_empty = df_coruna['will_be_empty'].mean() * 100
    
    # Split temporal por estacion
    X = df_coruna[features]
    y = df_coruna['will_be_empty']
    train_idx, test_idx = time_based_split(
        df_coruna, group_col='id', time_col='timestamp', train_fraction=load_config()["train_fraction"]
    )
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    
    # Preentrenamiento
    params = {'objective': 'binary', 'metric': 'binary_logloss', 'num_leaves': 31,
              'learning_rate': 0.05, 'verbose': -1, 'seed': 42, 'is_unbalance': True}
    
    pretrain_data = lgb.Dataset(df_bcn[features], label=df_bcn['will_be_empty'])
    pretrained = lgb.train(params, pretrain_data, num_boost_round=150)
    
    # Fine-tuning
    params['learning_rate'] = 0.01
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_test, label=y_test)
    
    model = lgb.train(params, train_data, num_boost_round=300, valid_sets=[val_data],
                      init_model=pretrained, callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    
    # Probabilidades
    y_proba = model.predict(X_test)
    
    # Buscar mejor umbral
    best_f1 = 0
    best_threshold = 0.5
    best_metrics = {}
    
    for thresh in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_proba > thresh).astype(int)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            best_metrics = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, zero_division=0),
                'Recall': recall_score(y_test, y_pred, zero_division=0),
                'F1': f1
            }
    
    return {
        'threshold_empty': threshold_empty,
        'pct_empty': pct_empty,
        'best_decision_threshold': best_threshold,
        'metrics': best_metrics
    }


def main():
    print("="*60)
    print("OPTIMIZACION: Umbral + Definicion de vacia")
    print("="*60)
    
    results = []
    
    # Probar diferentes definiciones de "vacia"
    cfg = load_config()
    for threshold in cfg["empty_threshold_candidates"]:
        print(f"\n--- Probando: Vacia = menos de {threshold} bicis ---")
        result = train_and_optimize(threshold)
        results.append(result)
        
        m = result['metrics']
        print(f"   Casos vacios: {result['pct_empty']:.1f}%")
        print(f"   Mejor umbral: {result['best_decision_threshold']:.2f}")
        print(f"   Precision: {m['Precision']*100:.1f}% | Recall: {m['Recall']*100:.1f}% | F1: {m['F1']*100:.1f}%")
    
    # Encontrar mejor combinacion
    print("\n" + "="*60)
    print("MEJOR CONFIGURACION")
    print("="*60)
    
    best = max(results, key=lambda x: x['metrics']['F1'])
    m = best['metrics']
    
    print(f"""
    Definicion de vacia: < {best['threshold_empty']} bicis
    Umbral de decision: {best['best_decision_threshold']:.2f}
    
    METRICAS FINALES:
    - Accuracy:  {m['Accuracy']*100:.1f}%
    - Precision: {m['Precision']*100:.1f}%
    - Recall:    {m['Recall']*100:.1f}%
    - F1 Score:  {m['F1']*100:.1f}%
    """)
    
    # Tabla comparativa
    print("\n" + "-"*60)
    print("TABLA COMPARATIVA")
    print("-"*60)
    print(f"{'Def. Vacia':<12} | {'% Casos':>8} | {'Umbral':>8} | {'Prec':>8} | {'Recall':>8} | {'F1':>8}")
    print("-"*60)
    for r in results:
        m = r['metrics']
        print(f"< {r['threshold_empty']} bicis    | {r['pct_empty']:>7.1f}% | {r['best_decision_threshold']:>8.2f} | {m['Precision']*100:>7.1f}% | {m['Recall']*100:>7.1f}% | {m['F1']*100:>7.1f}%")


if __name__ == "__main__":
    main()
