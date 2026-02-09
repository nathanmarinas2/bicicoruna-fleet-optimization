# -*- coding: utf-8 -*-
"""
OPTIMIZACIÓN DE FLOTA
=====================
Calcula el número óptimo de bicicletas necesario para minimizar roturas de stock (estaciones vacías).
Metodología: Análisis de 'Stress Test' en horas punta.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuración
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "coruna"
REPORT_DIR = BASE_DIR / "reports" / "figures"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Parámetros de Calidad de Servicio
SAFETY_BUFFER = 2  # Queremos que siempre haya al menos 2 bicis por estación
MIN_FLEET_UTILIZATION = 0.5 # Si baja de esto, sobran bicis

def load_data():
    print("Cargando datos de telemetría...")
    df = pd.read_csv(DATA_DIR / "tracking_data.csv")
    df.columns = df.columns.str.lower()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # LIMPIEZA: Eliminar snapshots incompletos (fuga de estaciones)
    counts = df.groupby('timestamp')['nombre'].transform('count')
    df = df[counts >= 75] 
    
    return df

def analyze_fleet_sizing(df):
    # 1. Estimar Flota Total Actual (Máximo de bicis detectadas en sistema a la vez)
    # Agrupamos por timestamp y sumamos bicis
    system_state = df.groupby('timestamp').agg({
        'bicis': 'sum',
        'capacidad': 'sum'
    })
    
    current_fleet_est = int(system_state['bicis'].max())
    total_docks = int(system_state['capacidad'].max())
    
    print(f"\n--- DIAGNÓSTICO DE INFRAESTRUCTURA ---")
    print(f"Capacidad Total (Docks): {total_docks}")
    print(f"Flota Estimada (Bicis):  {current_fleet_est}")
    print(f"Ratio Cobertura:         {current_fleet_est/total_docks:.1%} (Ideal: 40-60%)")
    
    # 2. Calcular Déficit Instantáneo
    # Para cada instante, ¿cuántas bicis faltaban para cumplir el SAFETY_BUFFER en cada estación?
    
    # Pivotar: Index=Timestamp, Columns=Station, Values=Bicis
    pivot_bikes = df.pivot_table(index='timestamp', columns='nombre', values='bicis', aggfunc='sum').fillna(0)
    
    # Matriz de Déficit: max(0, TARGET - Actual)
    deficit_matrix = pd.DataFrame(
        np.maximum(0, SAFETY_BUFFER - pivot_bikes.values),
        index=pivot_bikes.index,
        columns=pivot_bikes.columns
    )
    
    # Déficit total del sistema en cada momento
    system_deficit = deficit_matrix.sum(axis=1)
    
    # El "Worst Case Scenario" (Momento de mayor escasez simultánea)
    max_system_deficit = int(system_deficit.max())
    worst_moment = system_deficit.idxmax()
    
    print(f"\n--- ANÁLISIS DE STRESS (PICO DE DEMANDA) ---")
    print(f"Momento Crítico: {worst_moment}")
    print(f"Déficit Simultáneo: {max_system_deficit} bicicletas")
    print(f"(Bicis necesarias para asegurar {SAFETY_BUFFER} unidades en todas las estaciones)")
    
    # 3. Recomendación
    optimal_fleet = current_fleet_est + max_system_deficit
    buffer_percentage = (optimal_fleet / current_fleet_est - 1) * 100
    
    print(f"\n--- RECOMENDACIÓN FINAL ---")
    print(f"Flota Óptima Calculada: {optimal_fleet} bicicletas")
    print(f"Acción Sugerida: Adquirir {max_system_deficit} unidades (+{buffer_percentage:.1f}%)")
    
    return system_state, system_deficit, optimal_fleet

def plot_supply_demand(system_state, system_deficit, optimal_fleet):
    """Genera gráfico de oferta vs demanda insatisfecha"""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color_avail = '#059669' # Esmerald Green (más oscuro para fondo blanco)
    color_deficit = '#DC2626' # Red (más oscuro para fondo blanco)
    
    # Total Available Bikes
    ax1.set_xlabel('Tiempo')
    ax1.set_ylabel('Bicis Aparcadas (Oferta)', color=color_avail)
    ax1.plot(system_state.index, system_state['bicis'], color=color_avail, label='Oferta: Bicis Aparcadas', linewidth=1.5)
    ax1.tick_params(axis='y', labelcolor=color_avail)
    
    # Deficit
    ax2 = ax1.twinx()
    ax2.set_ylabel('Bicis Faltantes (Estrés)', color=color_deficit)
    ax2.fill_between(system_deficit.index, 0, system_deficit, color=color_deficit, alpha=0.2, label='Demanda: Bicis Faltantes (Estrés)')
    ax2.tick_params(axis='y', labelcolor=color_deficit)
    
    plt.title(f'Optimización de Flota: Análisis de Cobertura Humana', fontsize=14, pad=15)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    out_path = REPORT_DIR / "fleet_optimization.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nGráfico generado: {out_path}")

def main():
    df = load_data()
    state, deficit, opt = analyze_fleet_sizing(df)
    plot_supply_demand(state, deficit, opt)

if __name__ == "__main__":
    main()
