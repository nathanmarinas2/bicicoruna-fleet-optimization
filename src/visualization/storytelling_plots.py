# -*- coding: utf-8 -*-
"""
STORYTELLING PLOTS
==================
Genera gráficos estáticos de alto impacto para LinkedIn/README.
Estética: Dark Mode + Cyberpunk Palette.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Config
plt.style.use('dark_background')
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "coruna"
FIG_DIR = BASE_DIR / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Colores coincidentes con el Mapa Premium
CLUSTER_COLORS = ['#06B6D4', '#D946EF', '#FACC15', '#8B5CF6'] # Cyan, Kiwi(Magenta), Yellow, Violet
LABELS = ["0: Morning Destination", "1: Afternoon Hubs", "2: Evening Active", "3: Residential"]

def load_and_cluster():
    df = pd.read_csv(DATA_DIR / "tracking_data.csv")
    df.columns = df.columns.str.lower()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['day_name'] = df['timestamp'].dt.day_name()
    df['hour'] = df['timestamp'].dt.hour
    df = df.rename(columns={'bicis': 'bikes', 'nombre': 'name', 'capacidad': 'capacity'})
    df['occupancy'] = df['bikes'] / df['capacity'].replace(0, np.nan)
    
    # K-Means (Misma lógica que antes)
    pivot = df.pivot_table(index='name', columns='hour', values='occupancy', aggfunc='mean').fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(pivot)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    pivot['cluster'] = kmeans.fit_predict(X_scaled)
    
    return df, pivot # pivot tiene la info del cluster

def plot_cluster_profiles(pivot):
    """Gráfico de Líneas: Perfil Horario Promedio por Cluster (Estilo Clean/White)"""
    with plt.style.context('seaborn-v0_8-whitegrid'):
        plt.figure(figsize=(12, 6))
        
        # Calcular promedio horario por cluster
        for c in range(4):
            subset = pivot[pivot['cluster'] == c].drop('cluster', axis=1)
            mean_profile = subset.mean() * 100 # A porcentaje
            
            plt.plot(mean_profile.index, mean_profile.values, 
                     label=LABELS[c], 
                     color=CLUSTER_COLORS[c], 
                     linewidth=3, 
                     marker='o', markersize=5, alpha=0.9)
        
        plt.title('Ritmos Urbanos: Perfiles de Ocupación por Cluster', fontsize=16, pad=20, color='#1e293b', fontweight='bold')
        plt.xlabel('Hora del Día', fontsize=12, color='#475569')
        plt.ylabel('Ocupación Media (%)', fontsize=12, color='#475569')
        plt.xticks(range(0, 24), color='#475569')
        plt.yticks(color='#475569')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend(frameon=True, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=10)
        
        # Anotaciones Clean
        plt.annotate('Entrada Oficinas', xy=(9, 60), xytext=(11, 75),
                     arrowprops=dict(facecolor='#475569', arrowstyle='->'), color='#475569')
        
        path = FIG_DIR / "cluster_profiles.png"
        plt.savefig(path, dpi=300, bbox_inches='tight') # Fondo blanco por defecto
        print(f"Gráfico generado: {path}")
        plt.close()

def plot_weekly_heatmap(df):
    """Heatmap: Día vs Hora"""
    # Agrupar por Dia y Hora
    # Ordenar dias
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    grouped = df.groupby(['day_name', 'hour'])['occupancy'].mean().unstack() * 100
    grouped = grouped.reindex(days_order)
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(grouped, cmap='magma', annot=False, fmt=".0f", linewidths=.5, linecolor='#0F172A')
    
    plt.title('El Pulso de A Coruña: Intensidad de Uso Semanal', fontsize=16, pad=20, color='white')
    plt.xlabel('Hora del Día', fontsize=12)
    plt.ylabel('', fontsize=12)
    plt.yticks(rotation=0)
    
    path = FIG_DIR / "weekly_heatmap.png"
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='#0F172A')
    print(f"Gráfico generado: {path}")
    plt.close()

def main():
    print("Generando gráficos de Storytelling...")
    df, pivot = load_and_cluster()
    
    plot_cluster_profiles(pivot)
    plot_weekly_heatmap(df)
    print("¡Listo! Imágenes listas para LinkedIn en reports/figures/")

if __name__ == "__main__":
    main()
