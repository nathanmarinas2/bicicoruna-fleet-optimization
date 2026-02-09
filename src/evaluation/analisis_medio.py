# -*- coding: utf-8 -*-
"""
ANALISIS NIVEL MEDIO - Clustering y Patrones Avanzados
=======================================================
1. Clustering de estaciones (Residencial vs Trabajo)
2. Estaciones Gemelas (Correlacion inversa)
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent.parent.parent
DATA_CORUNA = BASE_DIR / "data" / "coruna"


def load_data():
    df = pd.read_csv(DATA_CORUNA / "tracking_data.csv")
    df.columns = df.columns.str.lower()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Renombrar
    df = df.rename(columns={'bicis': 'bikes', 'hora': 'hour', 
                            'nombre': 'name', 'capacidad': 'capacity'})
    
    # Calcular ocupacion normalizada
    df['occupancy'] = df['bikes'] / df['capacity'].replace(0, np.nan)
    df['occupancy'] = df['occupancy'].fillna(0)
    
    return df


def clustering_estaciones(df):
    """Agrupa estaciones por comportamiento horario (Busqueda automatica de K)."""
    print("\n" + "="*60)
    print("1. CLUSTERING DE ESTACIONES (Tipos de Barrio)")
    print("="*60)
    
    # Crear perfil horario para cada estacion (0-23h)
    perfil = df.groupby(['name', 'hour'])['occupancy'].mean().unstack()
    perfil = perfil.fillna(0)
    
    # Normalizar para comparar formas, no magnitudes
    scaler = StandardScaler()
    X = scaler.fit_transform(perfil)
    
    # Busqueda del K optimo usando Silhouette Score
    from sklearn.metrics import silhouette_score
    
    best_k = 2
    best_score = -1
    best_model = None
    
    print("   Buscando el numero optimo de tipos de barrio...")
    for k in range(2, 7):  # Probar de 2 a 6 tipos
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        print(f"   - k={k}: Silhouette Score = {score:.3f}")
        
        if score > best_score:
            best_score = score
            best_k = k
            best_model = kmeans
            
    print(f"\n   RESULTADO MATHEMATICO: Existen {best_k} tipos de comportamiento distintos.")
    print("   " + "-"*40)
    
    # Aplicar mejor modelo
    perfil['cluster'] = best_model.labels_
    
    # Analizar clusters encontrados
    perfiles_medios = perfil.groupby('cluster').mean()
    
    # Identificar tipos basado en picos
    for cluster_id in range(best_k):
        media = perfiles_medios.loc[cluster_id]
        pico_manana = media.loc[7:9].mean()  # 7-9 am
        pico_tarde = media.loc[17:19].mean() # 5-7 pm
        pico_mediodia = media.loc[13:15].mean() # 1-3 pm
        
        # Etiquetado inteligente
        if pico_manana < pico_tarde and pico_manana < pico_mediodia:
             tipo = "RESIDENCIAL (Salen manana)"
        elif pico_manana > pico_tarde:
             tipo = "TRABAJO/OFICINAS (Llegan manana)"
        elif pico_mediodia > pico_manana and pico_mediodia > pico_tarde:
             tipo = "OCIO/HOSTELERIA (Pico mediodia)"
        else:
             tipo = "MIXTO/ROTACION CONSTANTE"
            
        print(f"\n   TIPO {cluster_id + 1}: {tipo}")
        print("   " + "-"*40)
        
        # Caracteristicas
        estaciones = perfil[perfil['cluster'] == cluster_id].index.tolist()
        pct = len(estaciones) / len(perfil) * 100
        print(f"   ({len(estaciones)} estaciones - {pct:.0f}% del sistema)")
        
        # Ejemplos representativos (los mas cercanos al centro del cluster)
        # Simplificacion: mostrar primeros 5
        for est in estaciones[:5]:
            print(f"   - {est[:30]}")
        
        if len(estaciones) > 5:
            print(f"   ... y {len(estaciones)-5} mas")
            
    # Guardar asignacion
    perfil['cluster'].to_csv(BASE_DIR / "data" / "processed" / "estaciones_clusters.csv")
    print(f"\n   Asignacion de clusters guardada en 'data/processed/estaciones_clusters.csv'")


def estaciones_gemelas(df):
    """Encuentra pares de estaciones con comportamiento opuesto."""
    print("\n" + "="*60)
    print("2. ESTACIONES GEMELAS (Correlacion Inversa)")
    print("="*60)
    
    # Pivot table: filas=timestamp, columnas=estacion
    # Necesitamos resamplear a hora para alinear timestamps
    df_hora = df.set_index('timestamp').groupby(['name', pd.Grouper(freq='1h')])['occupancy'].mean().unstack(0)
    
    # Calcular correlacion
    corr = df_hora.corr()
    
    # Encontrar pares con correleacion NEGATIVA fuerte (una sube, otra baja)
    pares = []
    seen = set()
    
    for c1 in corr.columns:
        for c2 in corr.columns:
            if c1 == c2: continue
            
            # Ordenar par para evitar duplicados A-B vs B-A
            par = tuple(sorted([c1, c2]))
            if par in seen: continue
            seen.add(par)
            
            r = corr.loc[c1, c2]
            if r < -0.6:  # Correlacion negativa fuerte
                pares.append((c1, c2, r))
    
    pares.sort(key=lambda x: x[2])  # Mas negativo primero
    
    print("\n   PARES CON COMPORTAMIENTO OPUERTO (Uno se llena, otro se vacia):")
    if pares:
        for c1, c2, r in pares[:10]:
            print(f"   {c1[:20]:<20} <--> {c2[:20]:<20} (Corr: {r:.2f})")
    else:
        print("   No se encontraron pares con correlacion inversa fuerte (<-0.6)")
        
    # Encontrar pares GEMELOS (Correlacion positiva muy alta)
    print("\n   PARES GEMELOS (Se comportan igual):")
    gemelos = []
    seen = set()
    for c1 in corr.columns:
        for c2 in corr.columns:
            if c1 == c2: continue
            par = tuple(sorted([c1, c2]))
            if par in seen: continue
            seen.add(par)
            
            r = corr.loc[c1, c2]
            if r > 0.95:
                gemelos.append((c1, c2, r))
    
    gemelos.sort(key=lambda x: -x[2])
    
    if gemelos:
        for c1, c2, r in gemelos[:10]:
            print(f"   {c1[:20]:<20} <--> {c2[:20]:<20} (Corr: {r:.2f})")


def main():
    print("""
    ========================================================
    ANALISIS ADICIONALES - NIVEL MEDIO
    ========================================================
    """)
    df = load_data()
    clustering_estaciones(df)
    estaciones_gemelas(df)
    print("\n" + "="*60)
    print("ANALISIS COMPLETADO")
    print("="*60)

if __name__ == "__main__":
    main()
