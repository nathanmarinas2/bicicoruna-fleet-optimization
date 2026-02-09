# -*- coding: utf-8 -*-
"""
ANALISIS AVANZADO: Optimización de Clusters (Método del Codo) + Mapa Premium
============================================================================
1. Determina autom. el K óptimo de clusters analizando la curva de inercia.
2. Genera el mapa híbrido con ese K óptimo.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# from kneed import KneeLocator (Usamos implementación propia robusta)

BASE_DIR = Path(__file__).parent.parent.parent
DATA_CORUNA = BASE_DIR / "data" / "coruna"
OUTPUT_DIR = BASE_DIR / "dashboard"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    df = pd.read_csv(DATA_CORUNA / "tracking_data.csv")
    df.columns = df.columns.str.lower()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.rename(columns={'bicis': 'bikes', 'hora': 'hour', 'nombre': 'name', 'capacidad': 'capacity'})
    df['occupancy'] = df['bikes'] / df['capacity'].replace(0, np.nan)
    df['occupancy'] = df['occupancy'].fillna(0)
    return df

def find_optimal_k(X_scaled, max_k=10):
    """Encuentra K óptimo usando el Método del Codo (heurística simple)."""
    inertias = []
    print("\n--- Análisis de Codo (Elbow Method) ---")
    
    # Calculamos inercia para rango de K
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        # print(f"K={k}: Inercia={kmeans.inertia_:.2f}")
    
    # Cálculo geométrico del "codo": distancia a la línea que une primer y último punto
    # (Implementación manual robusta sin dependencias extra)
    x = range(2, max_k + 1)
    y = inertias
    
    # Coordenadas de la línea recta entre primer y último punto
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    
    max_dist = 0
    best_k = 3 # fallback
    
    for i, k in enumerate(x):
        p = np.array([k, y[i]])
        # Distancia de punto p a la linea p1-p2
        dist = np.abs(np.cross(p2-p1, p1-p)) / np.linalg.norm(p2-p1)
        
        if dist > max_dist:
            max_dist = dist
            best_k = k
            
    print(f"--> K Óptimo detectado matemáticamente: {best_k}")
    return best_k

def get_clusters_and_stats(df):
    # 1. Pivotar
    pivot = df.pivot_table(index='name', columns='hour', values='occupancy', aggfunc='mean').fillna(0)
    
    # 2. Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(pivot)
    
    # 3. Optimizar K
    k_optimo = find_optimal_k(X_scaled)
    
    # 4. Clustering Final
    kmeans = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
    pivot['cluster'] = kmeans.fit_predict(X_scaled)
    
    # 5. Stats
    stats = df.groupby('name').agg({
        'lat': 'first', 'lon': 'first', 'capacity': 'first',
        'bikes': 'mean', 'occupancy': 'mean'
    })
    
    return stats.join(pivot[['cluster']]).reset_index(), k_optimo

def generate_map_html(stations, flows, n_clusters):
    center_lat = stations['lat'].mean()
    center_lon = stations['lon'].mean()
    stations_js = stations.to_dict('records')
    coords = stations.set_index('name')[['lat', 'lon']].to_dict('index')
    
    flows_js = []
    for f in flows:
        if f['from'] in coords and f['to'] in coords:
            flows_js.append({
                'from_lat': coords[f['from']]['lat'], 'from_lon': coords[f['from']]['lon'],
                'to_lat': coords[f['to']]['lat'], 'to_lon': coords[f['to']]['lon'],
                'strength': f['strength']
            })

    # Generamos CSS dinámico para n clusters (cíclico si > 6)
    colors = ['#06B6D4', '#D946EF', '#FACC15', '#8B5CF6', '#F97316', '#14B8A6'] # Cyan, Magenta, Yellow, Violet, Orange, Teal
    legend_html = ""
    css_clusters = ""
    
    for i in range(n_clusters):
        c = colors[i % len(colors)]
        css_clusters += f".c{i} {{ border-color: {c}; color: {c}; }}\n"
        legend_html += f'<div class="legend-row"><span class="halo c{i}"></span> Cluster {i+1}</div>\n'

    html = f'''<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>BiciCoruña - Dynamics</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        :root {{ --bg:#0F172A; --glass:rgba(15,23,42,0.9); --text:#F8FAFC; }}
        body {{ margin:0; font-family:'Inter', sans-serif; background:var(--bg); color:var(--text); overflow:hidden; }}
        #map {{ height:100vh; filter:saturate(1.2) brightness(0.9); }}
        .panel {{ position:absolute; z-index:1000; background:var(--glass); padding:15px; border-radius:12px; border:1px solid rgba(255,255,255,0.1); backdrop-filter:blur(10px); }}
        
        .legend-row {{ display:flex; align-items:center; gap:8px; margin-bottom:6px; font-size:12px; color:#cbd5e1; }}
        .dot {{ width:8px; height:8px; border-radius:50%; }}
        .halo {{ width:10px; height:10px; border-radius:50%; border:2px solid; background:transparent; }}
        
        /* Dynamic Cluster Colors */
        {css_clusters}
        
        /* Occupancy */
        .occ-low {{ background: #EF4444; }}
        .occ-med {{ background: #F59E0B; }}
        .occ-high {{ background: #10B981; }}
    </style>
</head>
<body>
    <div id="map"></div>
    
    <div class="panel" style="top:20px; left:20px;">
        <h1 style="margin:0; font-size:18px;">Dinámicas Urbanas</h1>
        <p style="margin:4px 0 0; font-size:12px; opacity:0.7;">K-MEANS (K={n_clusters}) + FLUJOS</p>
    </div>

    <div class="panel" style="bottom:30px; right:20px;">
        <b style="font-size:12px; display:block; margin-bottom:8px; color:#94A3B8;">TIPO DE ESTACIÓN (HALO)</b>
        {legend_html}
        
        <b style="font-size:12px; display:block; margin:16px 0 8px; color:#94A3B8;">OCUPACIÓN (RELLENO)</b>
        <div class="legend-row"><span class="dot occ-low"></span> < 30%</div>
        <div class="legend-row"><span class="dot occ-med"></span> 30-60%</div>
        <div class="legend-row"><span class="dot occ-high"></span> > 60%</div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        const stations = {json.dumps(stations_js)};
        const flows = {json.dumps(flows_js)};
        const clusterColors = {json.dumps([colors[i % len(colors)] for i in range(n_clusters)])};
        
        const map = L.map('map', {{ zoomControl:false, attributionControl:false }}).setView([{center_lat}, {center_lon}], 13);
        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{maxZoom:19}}).addTo(map);

        stations.forEach(s => {{
            let fill = '#F59E0B';
            if (s.avg_occupancy < 0.3) fill = '#EF4444';
            if (s.avg_occupancy > 0.6) fill = '#10B981';
            
            L.circleMarker([s.lat, s.lon], {{
                radius: 6,
                fillColor: fill,
                color: clusterColors[s.cluster],
                weight: 2,
                opacity: 1,
                fillOpacity: 0.85
            }}).bindPopup(`
                <div style="font-family:Inter; padding:5px;">
                    <b>${{s.name}}</b><br>
                    <span style="color:${{clusterColors[s.cluster]}}">● Cluster ${{s.cluster + 1}}</span><br>
                    Ocupación: ${{(s.avg_occupancy*100).toFixed(0)}}%
                </div>
            `).addTo(map);
        }});
        
        flows.forEach(f => {{
            L.polyline([[f.from_lat, f.from_lon], [f.to_lat, f.to_lon]], {{
                color: '#64748B', weight: f.strength * 4, opacity: 0.3, interactive: false
            }}).addTo(map);
        }});
    </script>
</body>
</html>'''
    return html

def infer_flows(df):
    df = df.sort_values(['name', 'timestamp'])
    df['bike_change'] = df.groupby('name')['bikes'].diff()
    pivot = df.pivot_table(index='timestamp', columns='name', values='bike_change', aggfunc='first').fillna(0)
    corr = pivot.corr()
    
    flows = []
    seen = set()
    for c1 in corr.columns:
        for c2 in corr.columns:
            if c1 == c2: continue
            pair = tuple(sorted([c1, c2]))
            if pair in seen: continue
            seen.add(pair)
            if corr.loc[c1, c2] < -0.15:
                flows.append({'from': c1, 'to': c2, 'strength': abs(corr.loc[c1, c2])})
    
    return sorted(flows, key=lambda x: -x['strength'])[:60]

def main():
    print("Iniciando análisis...")
    df = load_data()
    
    data, n_best = get_clusters_and_stats(df)
    flows = infer_flows(df)
    
    html = generate_map_html(data, flows, n_best)
    path = OUTPUT_DIR / "mapa_flujos.html"
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Mapa generado con K={n_best} clusters optimos.")

if __name__ == "__main__":
    main()
