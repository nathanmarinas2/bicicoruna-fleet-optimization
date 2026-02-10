# -*- coding: utf-8 -*-
"""
ANÁLISIS DE CODO (ELBOW METHOD)
===============================
Script dedicado para justificar la elección del número de clusters (K).
Genera:
1. Reporte en consola con Inercia y Reducción %.
2. Gráfico visual 'dashboard/elbow_plot.png'.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Configuración de Rutas
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "coruna"
OUTPUT_DIR = BASE_DIR / "reports" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Carga y prepara la matriz de características (Estación x Hora)."""
    df = pd.read_csv(DATA_DIR / "tracking_data.csv")
    df.columns = df.columns.str.lower()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Preprocesamiento básico
    df = df.rename(columns={'bicis': 'bikes', 'nombre': 'name', 'capacidad': 'capacity', 'hora': 'hour'})
    df['occupancy'] = df['bikes'] / df['capacity'].replace(0, np.nan)
    df['occupancy'] = df['occupancy'].fillna(0)
    
    # Pivotar: Filas=Estaciones, Columnas=Horas (0..23), Valor=Ocupación media
    pivot = df.pivot_table(index='name', columns='hour', values='occupancy', aggfunc='mean').fillna(0)
    return pivot

def calculate_elbow(X_scaled, max_k=10):
    inertias = []
    ks = range(2, max_k + 1)
    
    print(f"{'K':<5} | {'Inercia':<15} | {'Reducción':<15}")
    print("-" * 40)
    
    prev_inertia = None
    
    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia = kmeans.inertia_
        inertias.append(inertia)
        
        reduction = ""
        if prev_inertia:
            pct = (prev_inertia - inertia) / prev_inertia * 100
            reduction = f"-{pct:.1f}%"
            
        print(f"{k:<5} | {inertia:<15.2f} | {reduction:<15}")
        prev_inertia = inertia
        
    return ks, inertias

def plot_elbow(ks, inertias, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(ks, inertias, 'go-', linewidth=2, markersize=8)
    
    # Estética
    plt.title('Método del Codo: Determinación de K Óptimo', fontsize=14)
    plt.xlabel('Número de Clusters (k)', fontsize=12)
    plt.ylabel('Inercia (Distancia intra-cluster)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Anotaciones
    for i, txt in enumerate(inertias):
        plt.annotate(f"{txt:.0f}", (ks[i], inertias[i]), textcoords="offset points", xytext=(0,10), ha='center')
        
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nGráfico guardado en: {output_path}")

def main():
    print("Cargando datos de movilidad...")
    pivot_data = load_data()
    
    print("Normalizando datos...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(pivot_data)
    
    print("\nEjecutando K-Means para k=2..10...")
    ks, inertias = calculate_elbow(X_scaled)
    
    plot_elbow(ks, inertias, OUTPUT_DIR / "elbow_plot.png")

if __name__ == "__main__":
    main()
