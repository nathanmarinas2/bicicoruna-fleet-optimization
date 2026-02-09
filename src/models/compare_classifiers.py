# -*- coding: utf-8 -*-
"""
Comparacion: Transfer Learning vs Entrenamiento desde cero
===========================================================
Compara:
1. Modelo preentrenado (BCN+DC) + Fine-tuning con Coruna
2. Modelo entrenado SOLO con Coruna

Esto demuestra el valor del transfer learning.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.utils.config import load_config
from src.utils.split import time_based_split

BASE_DIR = Path(__file__).parent.parent.parent
DATA_CORUNA = BASE_DIR / "data" / "coruna"


def load_coruna_data():
    """Carga y prepara datos de Coruna para clasificacion."""
    df = pd.read_csv(DATA_CORUNA / "tracking_data.csv")
    df.columns = df.columns.str.lower()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Renombrar columnas
    df = df.rename(columns={
        'bicis': 'bikes_available',
        'docks': 'docks_available',
        'hora': 'hour',
        'finde': 'is_weekend',
        'punta': 'is_rush_hour'
    })
    
    # Calcular ocupacion
    df['occupancy'] = df['bikes_available'] / df['capacidad'].replace(0, np.nan)
    df['occupancy'] = df['occupancy'].fillna(0).clip(0, 1)
    
    # Features ciclicas
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Ordenar y calcular targets
    df = df.sort_values(['id', 'timestamp']).reset_index(drop=True)
    
    cfg = load_config()
    threshold_empty = cfg["empty_threshold"]
    horizon_shifts = cfg["horizon_shifts"]

    df['future_bikes'] = df.groupby('id')['bikes_available'].shift(-horizon_shifts)
    df['will_be_empty'] = (df['future_bikes'] < threshold_empty).astype(int)
    
    # Limpiar
    df = df.dropna(subset=['future_bikes', 'occupancy'])
    
    print(f"Datos Coruna: {len(df):,} filas")
    print(f"Estaciones vacias (target=1): {df['will_be_empty'].mean()*100:.1f}%")
    
    return df


def evaluate_baseline(df, threshold_empty):
    """Baseline: predict empty if current bikes < threshold."""
    y_true = df['will_be_empty']
    y_pred = (df['bikes_available'] < threshold_empty).astype(int)

    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0)
    }

    return metrics


def train_classifier(X_train, y_train, X_test, y_test, name):
    """Entrena y evalua un clasificador."""
    cfg = load_config()
    decision_threshold = cfg["decision_threshold"]
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_jobs': -1,
        'verbose': -1,
        'seed': 42,
        'is_unbalance': True
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=300,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
    )
    
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > decision_threshold).astype(int)
    
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        roc_auc = float('nan')

    try:
        pr_auc = average_precision_score(y_test, y_pred_proba)
    except ValueError:
        pr_auc = float('nan')

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc,
        'PR-AUC': pr_auc
    }
    
    return model, metrics


def main():
    print("="*60)
    print("COMPARACION: Transfer Learning vs Solo Coruna")
    print("="*60)
    
    # Cargar datos de Coruna
    df = load_coruna_data()

    cfg = load_config()
    
    # Features
    feature_cols = ['hour_sin', 'hour_cos', 'is_weekend', 'is_rush_hour', 
                    'occupancy', 'bikes_available', 'docks_available']
    
    X = df[feature_cols]
    y = df['will_be_empty']
    
    train_idx, test_idx = time_based_split(
        df, group_col='id', time_col='timestamp', train_fraction=cfg["train_fraction"]
    )
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    
    print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")

    print("\n" + "-"*40)
    print("BASELINE: Bikes actuales < umbral")
    print("-"*40)
    baseline_metrics = evaluate_baseline(df.loc[test_idx], cfg["empty_threshold"])
    for k, v in baseline_metrics.items():
        print(f"   {k}: {v*100:.1f}%")
    
    # ========== MODELO 1: Solo Coruna ==========
    print("\n" + "-"*40)
    print("MODELO 1: Solo datos de Coruna")
    print("-"*40)
    
    model_coruna, metrics_coruna = train_classifier(
        X_train, y_train, X_test, y_test, "Solo Coruna"
    )
    
    for k, v in metrics_coruna.items():
        print(f"   {k}: {v*100:.1f}%")
    
    # ========== MODELO 2: Preentrenado + Fine-tuning ==========
    # Para simular transfer learning, entrenamos primero con Barcelona
    # y luego continuamos con Coruna
    print("\n" + "-"*40)
    print("MODELO 2: Preentrenado (BCN) + Fine-tuning (Coruna)")
    print("-"*40)
    
    # Cargar datos de Barcelona para preentrenamiento
    bcn_path = BASE_DIR / "data" / "raw" / "barcelona"
    csv_files = list(bcn_path.glob("*.csv"))
    
    if csv_files:
        # Cargar Barcelona (sample para velocidad)
        df_bcn = pd.read_csv(csv_files[0], nrows=500000)
        df_bcn.columns = df_bcn.columns.str.lower()
        
        # Preparar Barcelona
        if 'last_reported' in df_bcn.columns:
            df_bcn['timestamp'] = pd.to_datetime(df_bcn['last_reported'], unit='s')
        df_bcn['hour'] = df_bcn['timestamp'].dt.hour if 'timestamp' in df_bcn.columns else 12
        df_bcn['hour_sin'] = np.sin(2 * np.pi * df_bcn['hour'] / 24)
        df_bcn['hour_cos'] = np.cos(2 * np.pi * df_bcn['hour'] / 24)
        df_bcn['is_weekend'] = df_bcn['timestamp'].dt.dayofweek.isin([5,6]).astype(int) if 'timestamp' in df_bcn.columns else 0
        df_bcn['is_rush_hour'] = df_bcn['hour'].apply(lambda h: 1 if 7<=h<=9 or 17<=h<=20 else 0)
        
        # Ocupacion
        df_bcn['bikes_available'] = df_bcn.get('num_bikes_available', df_bcn.get('bikes_available', 0))
        df_bcn['docks_available'] = df_bcn.get('num_docks_available', df_bcn.get('docks_available', 0))
        df_bcn['capacity'] = df_bcn['bikes_available'] + df_bcn['docks_available']
        df_bcn['occupancy'] = (df_bcn['bikes_available'] / df_bcn['capacity'].replace(0, np.nan)).fillna(0)
        
        # Target
        df_bcn = df_bcn.sort_values(['station_id', 'timestamp']).reset_index(drop=True)
        df_bcn['future_bikes'] = df_bcn.groupby('station_id')['bikes_available'].shift(-cfg["horizon_shifts"])
        df_bcn['will_be_empty'] = (df_bcn['future_bikes'] < cfg["empty_threshold"]).astype(int)
        df_bcn = df_bcn.dropna(subset=['future_bikes'])
        
        print(f"   Preentrenamiento con {len(df_bcn):,} filas de Barcelona")
        
        # Entrenar con Barcelona
        X_bcn = df_bcn[feature_cols].dropna()
        y_bcn = df_bcn.loc[X_bcn.index, 'will_be_empty']
        
        params_pretrain = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'verbose': -1,
            'seed': 42
        }
        
        pretrain_data = lgb.Dataset(X_bcn, label=y_bcn)
        pretrained = lgb.train(params_pretrain, pretrain_data, num_boost_round=100)
        
        print("   Preentrenamiento completado")
        
        # Fine-tuning con Coruna
        params_finetune = params_pretrain.copy()
        params_finetune['learning_rate'] = 0.01  # Mas bajo para fine-tuning
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_test, label=y_test)
        
        finetuned = lgb.train(
            params_finetune,
            train_data,
            num_boost_round=200,
            valid_sets=[val_data],
            init_model=pretrained,
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
        )
        
        y_pred_ft = (finetuned.predict(X_test) > 0.5).astype(int)
        
        metrics_transfer = {
            'Accuracy': accuracy_score(y_test, y_pred_ft),
            'Precision': precision_score(y_test, y_pred_ft, zero_division=0),
            'Recall': recall_score(y_test, y_pred_ft, zero_division=0),
            'F1': f1_score(y_test, y_pred_ft, zero_division=0)
        }
        
        for k, v in metrics_transfer.items():
            print(f"   {k}: {v*100:.1f}%")
    else:
        print("   No se encontraron datos de Barcelona para preentrenamiento")
        metrics_transfer = metrics_coruna
    
    # ========== COMPARACION ==========
    print("\n" + "="*60)
    print("COMPARACION FINAL")
    print("="*60)
    print(f"\n{'Metrica':<12} | {'Solo Coruna':>12} | {'Transfer':>12} | {'Diferencia':>12}")
    print("-"*55)
    
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1']:
        v1 = metrics_coruna[metric] * 100
        v2 = metrics_transfer[metric] * 100
        diff = v2 - v1
        sign = "+" if diff > 0 else ""
        print(f"{metric:<12} | {v1:>11.1f}% | {v2:>11.1f}% | {sign}{diff:>10.1f}%")
    
    print("\n" + "="*60)
    if metrics_transfer['Accuracy'] > metrics_coruna['Accuracy']:
        print("GANADOR: Transfer Learning (Preentrenado + Fine-tuning)")
    else:
        print("GANADOR: Entrenamiento solo con Coruna")
    print("="*60)


if __name__ == "__main__":
    main()
