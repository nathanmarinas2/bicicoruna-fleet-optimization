# -*- coding: utf-8 -*-
"""
OPERATIONAL TOP OFFENDERS & SLA
===============================
Identifica las estaciones más problemáticas y calcula el tiempo de reacción operativa.
1. Top 5 Estaciones con mayor tiempo de inactividad (Stockout).
2. Cálculo del SLA (Ventana de Reacción): Tiempo medio de vaciado (5 -> 0 bicis).
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "coruna"

def load_data():
    df = pd.read_csv(DATA_DIR / "tracking_data.csv")
    df.columns = df.columns.str.lower()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.sort_values(['nombre', 'timestamp'])

def get_top_offenders(df):
    """Identifica estaciones críticas (% tiempo sin servicio)"""
    # Definimos Stockout como < 2 bicis
    df['is_stockout'] = df['bicis'] < 2
    
    # Agrupamos por estación
    stats = df.groupby('nombre').agg(
        total_pings=('timestamp', 'count'),
        stockout_pings=('is_stockout', 'sum')
    )
    
    stats['downtime_pct'] = (stats['stockout_pings'] / stats['total_pings']) * 100
    top_5 = stats.sort_values('downtime_pct', ascending=False).head(5)
    
    print("\n--- TOP 5 ESTACIONES CRÍTICAS (Downtime Risk) ---")
    print(f"{'Estación':<30} | {'% Sin Servicio':<15} | {'Acción Recomendada'}")
    print("-" * 70)
    
    for name, row in top_5.iterrows():
        pct = row['downtime_pct']
        action = "Ampliar Docks (+5)" if pct > 15 else "Rebalanceo Prioritario"
        print(f"{name:<30} | {pct:>14.1f}% | {action}")
        
    return top_5

def calculate_reaction_time(df):
    """Calcula el tiempo medio de vaciado (de 5 a 0 bicis)"""
    # Esto es complejo sin eventos discretos perfectos, pero haremos una aproximación:
    # Filtramos momentos donde bicis <= 5.
    # Calculamos la tasa de caída promedio (Delta negativo) en horas punta (7-10 AM).
    
    morning_rush = df[df['hora'].between(7, 10)]
    # Solo momentos de vaciado (delta < 0)
    depletion_events = morning_rush[morning_rush['delta'] < 0]
    
    if len(depletion_events) == 0:
        print("No hay suficientes datos de vaciado para calcular SLA.")
        return
        
    avg_depletion_rate = depletion_events['delta'].mean() # bicis/5min (negativo)
    bikes_per_min = abs(avg_depletion_rate) / 5
    
    # Tiempo para vaciar 5 bicis
    # T = 5 / (bicis/min)
    time_to_empty = 5 / bikes_per_min
    
    print(f"\n--- SLA OPERATIVO (Tiempo de Reacción) ---")
    print(f"Velocidad de Vaciado (Hora Punta): {bikes_per_min*60:.1f} bicis/hora")
    print(f"Ventana de Alerta (<5 bicis):      {time_to_empty:.0f} minutos")
    print(f"Conclusión: El equipo logístico tiene {time_to_empty:.0f} min para llegar antes del colapso.")

def main():
    print("Analizando KPIs Operativos...")
    df = load_data()
    get_top_offenders(df)
    calculate_reaction_time(df)

if __name__ == "__main__":
    main()
