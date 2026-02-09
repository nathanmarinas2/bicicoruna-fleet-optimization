# -*- coding: utf-8 -*-
"""
Clasificador MEJORADO con clima + tendencia
============================================
Mejoras:
1. Features de clima (temp, lluvia, viento)
2. Tendencia reciente (cambio en ultimos 15 min)
3. Comparacion antes/despues
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.utils.config import load_config
from src.utils.split import time_based_split

BASE_DIR = Path(__file__).parent.parent.parent
DATA_CORUNA = BASE_DIR / "data" / "coruna"
MODELS_DIR = BASE_DIR / "models"


def load_and_engineer_features():
    """Carga datos y crea features mejoradas."""
    print("Cargando datos de Coruna...")
    df = pd.read_csv(DATA_CORUNA / "tracking_data.csv")
    df.columns = df.columns.str.lower()
    
    # Renombrar columnas
    rename_map = {
        'bicis': 'bikes_available',
        'docks': 'docks_available',
        'hora': 'hour',
        'finde': 'is_weekend',
        'punta': 'is_rush_hour',
        'temp': 'temperature',
        'lluvia': 'rain',
        'viento': 'wind'
    }
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
    
    # Ordenar por estacion y tiempo
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['id', 'timestamp']).reset_index(drop=True)
    
    # ========== FEATURES BASICAS ==========
    df['occupancy'] = df['bikes_available'] / df['capacidad'].replace(0, np.nan)
    df['occupancy'] = df['occupancy'].fillna(0).clip(0, 1)
    
    # Features ciclicas
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # ========== FEATURES DE CLIMA (NUEVAS) ==========
    # Normalizar temperatura (0-1)
    if 'temperature' in df.columns:
        df['temp_norm'] = (df['temperature'] - df['temperature'].min()) / \
                          (df['temperature'].max() - df['temperature'].min() + 0.001)
    else:
        df['temp_norm'] = 0.5
    
    # Lluvia (ya es binario o 0-1)
    if 'rain' in df.columns:
        df['is_raining'] = (df['rain'] > 0).astype(int)
    else:
        df['is_raining'] = 0
    
    # Viento normalizado
    if 'wind' in df.columns:
        df['wind_norm'] = df['wind'] / (df['wind'].max() + 0.001)
    else:
        df['wind_norm'] = 0
    
    # ========== FEATURES DE TENDENCIA (NUEVAS) ==========
    # Cambio en los ultimos 15 min (3 observaciones de 5 min)
    df['bikes_lag_1'] = df.groupby('id')['bikes_available'].shift(1)
    df['bikes_lag_3'] = df.groupby('id')['bikes_available'].shift(3)
    
    # Tendencia: cambio en ultimos 15 min
    df['trend_15min'] = (df['bikes_available'] - df['bikes_lag_3']) / (df['capacidad'] + 0.001)
    df['trend_15min'] = df['trend_15min'].fillna(0)
    
    # Velocidad de cambio (derivada)
    df['velocity'] = df['bikes_available'] - df['bikes_lag_1']
    df['velocity'] = df['velocity'].fillna(0)
    
    # Aceleracion (segunda derivada)
    df['velocity_lag'] = df.groupby('id')['velocity'].shift(1)
    df['acceleration'] = df['velocity'] - df['velocity_lag'].fillna(0)
    
    # ========== TARGET ==========
    cfg = load_config()
    threshold_empty = cfg["empty_threshold"]
    horizon_shifts = cfg["horizon_shifts"]
    
    df['future_bikes'] = df.groupby('id')['bikes_available'].shift(-horizon_shifts)
    df['will_be_empty'] = (df['future_bikes'] < threshold_empty).astype(int)
    
    # Limpiar NaN
    df = df.dropna(subset=['future_bikes', 'occupancy', 'trend_15min'])
    
    print(f"Datos procesados: {len(df):,} filas")
    print(f"Estaciones vacias (target=1): {df['will_be_empty'].mean()*100:.1f}%")
    
    return df


def train_and_compare():
    """Entrena modelo basico vs mejorado y compara."""
    df = load_and_engineer_features()
    
    # Features basicas (antes)
    basic_features = [
        'hour_sin', 'hour_cos', 
        'is_weekend', 'is_rush_hour',
        'occupancy', 
        'bikes_available', 'docks_available'
    ]
    
    # Features mejoradas (ahora)
    improved_features = basic_features + [
        'temp_norm', 'is_raining', 'wind_norm',  # Clima
        'trend_15min', 'velocity', 'acceleration'  # Tendencia
    ]
    
    # Verificar que todas las features existen
    basic_features = [f for f in basic_features if f in df.columns]
    improved_features = [f for f in improved_features if f in df.columns]
    
    print(f"\nFeatures basicas: {len(basic_features)}")
    print(f"Features mejoradas: {len(improved_features)}")
    print(f"Features nuevas: {set(improved_features) - set(basic_features)}")
    
    y = df['will_be_empty']
    
    # Split (mismo para ambos modelos)
    train_idx, test_idx = time_based_split(
        df, group_col='id', time_col='timestamp', train_fraction=cfg["train_fraction"]
    )
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 6,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'n_jobs': -1,
        'verbose': -1,
        'seed': 42,
        'is_unbalance': True
    }
    
    results = {}
    
    for name, features in [('BASICO', basic_features), ('MEJORADO', improved_features)]:
        print(f"\n{'='*50}")
        print(f"Entrenando modelo {name}...")
        print(f"{'='*50}")
        
        X_train = df.loc[train_idx, features]
        X_test = df.loc[test_idx, features]
        y_train = df.loc[train_idx, 'will_be_empty']
        y_test = df.loc[test_idx, 'will_be_empty']
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        y_pred_proba = model.predict(X_test)
        decision_threshold = cfg["decision_threshold"]
        y_pred = (y_pred_proba > decision_threshold).astype(int)
        
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        results[name] = metrics
        
        print(f"\nResultados {name}:")
        for k, v in metrics.items():
            print(f"   {k}: {v*100:.1f}%")
        
        # Feature importance
        if name == 'MEJORADO':
            print(f"\nImportancia de features:")
            importance = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importance(importance_type='gain')
            }).sort_values('importance', ascending=False)
            for _, row in importance.head(8).iterrows():
                print(f"   {row['feature']}: {row['importance']:.0f}")
            
            # Guardar modelo mejorado
            import tempfile
            import shutil
            model_path = MODELS_DIR / "classifier_improved.txt"
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
                tmp_path = tmp.name
            model.save_model(tmp_path)
            shutil.move(tmp_path, str(model_path))
            print(f"\nModelo guardado: {model_path}")
    
    # Comparacion final
    print("\n" + "="*60)
    print("COMPARACION FINAL")
    print("="*60)
    print(f"\n{'Metrica':<12} | {'Basico':>10} | {'Mejorado':>10} | {'Mejora':>10}")
    print("-"*50)
    
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1']:
        v1 = results['BASICO'][metric] * 100
        v2 = results['MEJORADO'][metric] * 100
        diff = v2 - v1
        sign = "+" if diff > 0 else ""
        print(f"{metric:<12} | {v1:>9.1f}% | {v2:>9.1f}% | {sign}{diff:>8.1f}%")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    train_and_compare()
