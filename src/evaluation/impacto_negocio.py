# -*- coding: utf-8 -*-
"""
IMPACTO DE NEGOCIO Y ROI
========================
Transforma métricas técnicas en KPIs de negocio y mapas de riesgo operativo.
1. Mapa de Calor de Roturas de Stock (Missing Bikes).
2. Estimación de Costes de Oportunidad (Euros Perdidos).
3. Scorecard de Volatilidad por Cluster.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configuración
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "coruna"
FIG_DIR = BASE_DIR / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Suposiciones de Coste (Business Assumptions)
COST_LOST_TRIP = 2.50  # € por viaje perdido (Ticket + Valor Marca)
COST_TRUCK_ROLL = 45.0 # € por salida de camión de rebalanceo
REACTION_TIME_WINDOW = 3 # Horas para considerar "Evento Crítico"

def load_data():
    df = pd.read_csv(DATA_DIR / "tracking_data.csv")
    df.columns = df.columns.str.lower()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_name'] = df['timestamp'].dt.day_name()
    return df

def generate_stockout_heatmap(df):
    """Genera un mapa de calor de eventos de 'Estación Vacía'"""
    # Definimos Rotura de Stock: < 2 bicis (Fallo de Servicio Real, no alerta preventiva)
    df['stockout'] = df['bicis'] < 2
    
    # Agrupamos por Día e Hora, sumando eventos de stockout
    stockouts = df[df['stockout']].groupby(['day_name', 'hour']).size().unstack().fillna(0)
    
    # Ordenar días
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    stockouts = stockouts.reindex(days_order)
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(stockouts, cmap='Reds', annot=False, linewidths=.5, fmt=".0f")
    plt.title('Mapa de Riesgo: Frecuencia de Roturas de Stock (<2 Bicis)', fontsize=14, pad=15)
    plt.ylabel('')
    plt.xlabel('Hora del Día')
    
    out_path = FIG_DIR / "stockout_risk_heatmap.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Mapa de Riesgo generado: {out_path}")
    return stockouts.sum().sum() # Total eventos

def calculate_roi_impact(df, total_stockouts):
    """Calcula el Riesgo de Churn y Costes Operativos"""
    # Tarifas Reales (40€/año). 
    SUBSCRIPTION_PRICE = 40.0
    CHURN_PROBABILITY = 0.02 # Escenario Conservador: 2% de riesgo de baja por frustración
    COST_PER_FAILED_TRIP = SUBSCRIPTION_PRICE * CHURN_PROBABILITY # 0.80€
    
    # Estimamos Demanda Perdida (Calibrado con Datos Oficiales: 145k usos/mes / 30 dias / 18h / 79 estaciones)
    ESTIMATED_DEMAND_PER_HOUR = 3.4 # 3.4 usuarios/hora intentan coger bici en promedio
    hours_of_service_lost = total_stockouts * 5 / 60
    trips_lost = int(hours_of_service_lost * ESTIMATED_DEMAND_PER_HOUR)
    
    # Impacto Económico (Churn Risk)
    churn_risk_value = trips_lost * COST_PER_FAILED_TRIP
    potential_churners = int(trips_lost * CHURN_PROBABILITY)
    
    print(f"\n--- IMPACTO EN NEGOCIO (Semanal) ---")
    print(f"Horas Sin Servicio: {hours_of_service_lost:.1f} h")
    print(f"Experiencias Frustradas: {trips_lost} usuarios")
    print(f"Riesgo de Bajas (Churn): ~{potential_churners} usuarios/semana")
    print(f"Valor en Riesgo (Anual): {churn_risk_value:.2f}€/semana -> {churn_risk_value*52/1000:.1f}k€ (Anual)")
    
    return churn_risk_value

def analyze_cluster_volatility(df):
    """Analiza qué clusters son más inestables (más difíciles de gestionar)"""
    # Recalculamos clusters rápido para tener la etiqueta
    pivot = df.pivot_table(index='nombre', columns='hour', values='bicis', aggfunc='mean').fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(pivot)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Mapeamos nombre -> cluster
    cluster_map = pd.Series(labels, index=pivot.index)
    df['cluster'] = df['nombre'].map(cluster_map)
    
    # Calculamos desviación estándar de bicis por cluster (Volatilidad)
    volatility = df.groupby('cluster')['bicis'].std()
    stockout_rate = df[df['bicis'] < 2].groupby('cluster').size() / df.groupby('cluster').size() * 100
    
    results = pd.DataFrame({
        'Volatilidad (Std Dev)': volatility,
        'Probabilidad Rotura (%)': stockout_rate
    })
    
    print(f"\n--- SCORECARD POR CLUSTER ---")
    print(results.round(2).to_string())
    
    # Gráfico de barras
    plt.figure(figsize=(10, 5))
    colors = ['#06B6D4', '#D946EF', '#FACC15', '#8B5CF6']
    results['Probabilidad Rotura (%)'].plot(kind='bar', color=colors, alpha=0.8)
    plt.title('Riesgo Operativo por Cluster (% Tiempo sin Servicio)', fontsize=14)
    plt.ylabel('% Tiempo sin bicis')
    plt.xlabel('Cluster ID')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    out_path = FIG_DIR / "cluster_risk_scorecard.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Scorecard generado: {out_path}")

def main():
    print("Generando Análisis de Impacto de Negocio...")
    df = load_data()
    total_evts = generate_stockout_heatmap(df)
    calculate_roi_impact(df, total_evts)
    analyze_cluster_volatility(df)

if __name__ == "__main__":
    main()
