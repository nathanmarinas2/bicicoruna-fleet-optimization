# -*- coding: utf-8 -*-
"""
MAPA DE CALOR: TIEMPO SIN BICIS (PREMIUM EDITION)
=================================================
Genera un dashboard interactivo de alto nivel visual donde cada estación 
se colorea según el riesgo de desabastecimiento (Stockout Risk).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Configuración
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "coruna"
OUTPUT_DIR = BASE_DIR / "dashboard"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Carga los datos de tracking."""
    csv_path = DATA_DIR / "tracking_data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {csv_path}")
        
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()
    
    # Renombrar columnas para consistencia
    rename_map = {
        'bicis': 'bikes',
        'hora': 'hour',
        'nombre': 'name',
        'capacidad': 'capacity'
    }
    df = df.rename(columns=rename_map)
    return df

def calculate_empty_percentage(df):
    """Calcula el % de tiempo sin bicis por estación."""
    stats = df.groupby('name').agg({
        'lat': 'first',
        'lon': 'first',
        'bikes': [
            ('total_obs', 'count'),
            ('empty_obs', lambda x: (x == 0).sum())
        ],
        'capacity': 'first'
    }).reset_index()
    
    stats.columns = ['name', 'lat', 'lon', 'total_obs', 'empty_obs', 'capacity']
    stats['empty_pct'] = (stats['empty_obs'] / stats['total_obs']) * 100
    
    # Añadir métricas de negocio estimadas
    # Asumimos 5 min por observación. 
    # Horas perdidas = observaciones vacías * 5 / 60
    stats['hours_lost'] = stats['empty_obs'] * 5 / 60
    
    return stats.sort_values('empty_pct', ascending=False)

def generate_html_map(stats_df):
    """Genera el HTML del mapa con diseño Premium / Glassmorphism."""
    
    # Datos globales para el panel lateral
    total_stations = len(stats_df)
    critical_stations = len(stats_df[stats_df['empty_pct'] > 10])
    avg_empty_time = stats_df['empty_pct'].mean()
    worst_station = stats_df.iloc[0]
    
    # Top 5 offending stations para la lista
    top_offenders = stats_df.head(5).to_dict('records')
    
    # Centro del mapa
    center_lat = stats_df['lat'].mean()
    center_lon = stats_df['lon'].mean()
    
    # Datos JSON
    stations_data = stats_df.to_dict('records')
    
    # Build list HTML items
    list_items_html = ""
    for s in top_offenders:
        list_items_html += f"""
            <div class="station-item">
                <span class="station-name">{s['name']}</span>
                <span class="station-metric">{s['empty_pct']:.1f}%</span>
            </div>"""

    
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BiciCoruña | Risk Analytics</title>
    
    <!-- Fonts & Leaflet -->
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <!-- Icons -->
    <link href="https://cdn.jsdelivr.net/npm/remixicon@3.5.0/fonts/remixicon.css" rel="stylesheet">

    <style>
        :root {{
            --bg-dark: #09090b;
            --glass-bg: rgba(9, 9, 11, 0.85);
            --glass-border: rgba(255, 255, 255, 0.08);
            --text-main: #ffffff;
            --text-muted: #a1a1aa;
            
            --color-critical: #ef4444; /* Red 500 */
            --color-warning: #f59e0b;  /* Amber 500 */
            --color-safe: #10b981;     /* Emerald 500 */
            --color-accent: #3b82f6;   /* Blue 500 */
        }}
        
        * {{ box-sizing: border-box; }}
        
        body {{ 
            margin: 0; 
            padding: 0; 
            font-family: 'Outfit', sans-serif;
            background: var(--bg-dark);
            color: var(--text-main);
            overflow: hidden;
            width: 100vw;
            height: 100vh;
        }}
        
        #map {{ 
            height: 100%; 
            width: 100%; 
            z-index: 1;
            filter: saturate(1.1) contrast(1.1); /* Pop colors */
        }}
        
        /* --- UI COMPONENTS --- */
        
        .floating-panel {{
            position: absolute;
            top: 24px;
            left: 24px;
            width: 380px;
            z-index: 1000;
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 20px;
            animation: slideIn 0.8s cubic-bezier(0.16, 1, 0.3, 1);
        }}

        @keyframes slideIn {{
            from {{ opacity: 0; transform: translateX(-20px); }}
            to {{ opacity: 1; transform: translateX(0); }}
        }}
        
        .header {{
            display: flex;
            align-items: center;
            gap: 16px;
            border-bottom: 1px solid var(--glass-border);
            padding-bottom: 20px;
        }}
        
        .logo-box {{
            width: 48px;
            height: 48px;
            background: linear-gradient(135deg, var(--color-critical), #b91c1c);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 0 20px rgba(239, 68, 68, 0.3);
        }}
        
        .logo-box i {{
            font-size: 24px;
            color: white;
        }}
        
        .title-group h1 {{
            font-size: 18px;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.02em;
            line-height: 1.2;
        }}
        
        .title-group p {{
            font-size: 13px;
            color: var(--text-muted);
            margin: 4px 0 0 0;
            font-weight: 400;
        }}
        
        /* Stats Grid */
        .stats-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }}
        
        .stat-card {{
            background: rgba(255, 255, 255, 0.03);
            border-radius: 16px;
            padding: 16px;
            border: 1px solid rgba(255,255,255,0.02);
            transition: all 0.2s ease;
        }}
        
        .stat-card:hover {{
            background: rgba(255, 255, 255, 0.06);
            border-color: rgba(255, 255, 255, 0.08);
            transform: translateY(-2px);
        }}
        
        .stat-label {{
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        
        .stat-value {{
            font-size: 24px;
            font-weight: 600;
            letter-spacing: -0.03em;
            color: var(--text-main);
        }}
        
        .text-critical {{ color: var(--color-critical); text-shadow: 0 0 20px rgba(239, 68, 68, 0.3); }}
        .text-warning {{ color: var(--color-warning); }}
        
        /* Top Offenders List */
        .list-section {{
            background: rgba(0,0,0,0.2);
            border-radius: 16px;
            padding: 16px;
            border: 1px solid var(--glass-border);
        }}

        .list-header {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-muted);
            font-weight: 700;
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
        }}
        
        .station-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.04);
        }}
        
        .station-item:last-child {{ border-bottom: none; }}
        
        .station-name {{
            font-size: 13px;
            font-weight: 500;
            color: var(--text-main);
        }}
        
        .station-metric {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            font-weight: 700;
            color: var(--color-critical);
            background: rgba(239, 68, 68, 0.1);
            padding: 4px 8px;
            border-radius: 6px;
        }}

        /* Legend */
        .legend-card {{
            position: absolute;
            bottom: 30px;
            right: 30px;
            width: 280px;
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 20px;
            z-index: 999;
            box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
        }}
        
        .gradient-bar {{
            height: 8px;
            background: linear-gradient(to right, 
                var(--color-safe) 0%, 
                var(--color-warning) 30%, 
                var(--color-critical) 70%,
                #7f1d1d 100%
            );
            border-radius: 4px;
            margin: 12px 0 8px 0;
            position: relative;
        }}
        
        /* Map Markers Glow */
        .leaflet-interactive {{
             transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }}

        .leaflet-popup-content-wrapper {{
            background: rgba(18, 18, 24, 0.95) !important;
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            color: white !important;
            box-shadow: 0 20px 40px -10px rgba(0,0,0,0.6);
            padding: 0;
        }}
        .leaflet-popup-tip {{ background: rgba(18, 18, 24, 0.95) !important; }}
        .leaflet-container a.leaflet-popup-close-button {{
            color: var(--text-muted) !important;
            font-size: 18px !important;
            padding: 8px !important;
        }}
        
    </style>
</head>
<body>

    <div class="floating-panel">
        <div class="header">
            <div class="logo-box">
                <i class="ri-fire-fill"></i>
            </div>
            <div class="title-group">
                <h1>Riesgo de Servicio</h1>
                <p>Análisis de Fallos (Stockouts)</p>
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label"><i class="ri-bar-chart-2-line"></i> Media Red</div>
                <div class="stat-value">{avg_empty_time:.1f}<span style="font-size:14px; color:#666; margin-left:2px">%</span></div>
            </div>
            <div class="stat-card">
                <div class="stat-label"><i class="ri-error-warning-line"></i> Puntos Críticos</div>
                <div class="stat-value text-critical">{critical_stations}</div>
            </div>
            <div class="stat-card" style="grid-column: span 2;">
                <div class="stat-label"><i class="ri-alarm-warning-line"></i> Peor Estación (Max Outage)</div>
                <div class="stat-value text-critical" style="font-size:18px">{worst_station['name']}</div>
                <div style="font-size:12px; color:#666; margin-top:4px; display:flex; gap:6px; align-items:center">
                    <span style="width:6px; height:6px; background:var(--color-critical); border-radius:50%"></span>
                    {worst_station['empty_pct']:.1f}% del tiempo sin bicis
                </div>
            </div>
        </div>
        
        <div class="list-section">
            <div class="list-header">
                <span>Top 5 Críticos</span>
                <i class="ri-arrow-down-line"></i>
            </div>
            {list_items_html}
        </div>
    </div>
    
    <div id="map"></div>
    
    <div class="legend-card">
        <div class="list-header" style="margin-bottom:0">RIESGO (% TIEMPO VACÍA)</div>
        <div class="gradient-bar"></div>
        <div style="display:flex; justify-content:space-between; font-size:11px; color:var(--text-muted); font-weight:500">
            <span>0% (Óptimo)</span>
            <span>10%</span>
            <span>20%+ (Critico)</span>
        </div>
    </div>

    <!-- Scripts -->
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        // Init Map
        const map = L.map('map', {{
            zoomControl: false,
            attributionControl: false
        }}).setView([{center_lat}, {center_lon}], 13);

        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            maxZoom: 19
        }}).addTo(map);

        const stations = {json.dumps(stations_data)};

        function getColor(pct) {{
            if (pct <= 2) return '#10b981'; // Green
            if (pct <= 5) return '#f59e0b'; // Amber
            if (pct <= 10) return '#f97316'; // Orange
            return '#ef4444'; // Red
        }}

        stations.forEach(st => {{
            const color = getColor(st.empty_pct);
            const isCritical = st.empty_pct > 10;
            
            // Marker base radius
            let radius = 4 + (st.empty_pct / 2.5);
            if (radius > 12) radius = 12;
            if (radius < 4) radius = 4;
            
            // Add Marker
            const marker = L.circleMarker([st.lat, st.lon], {{
                radius: radius,
                fillColor: color,
                color: isCritical ? '#fea5a5' : '#ffffff',
                weight: isCritical ? 2 : 1,
                opacity: 1,
                fillOpacity: 0.9,
            }}).addTo(map);
            
            // Rich Popup
            marker.bindPopup(`
                <div style="font-family: 'Outfit'; min-width: 200px; padding: 12px;">
                    <div style="font-size: 10px; text-transform:uppercase; color: #888; font-weight:700; margin-bottom:6px; letter-spacing:0.05em">ESTACIÓN</div>
                    <div style="font-size: 15px; font-weight: 700; color: white; margin-bottom: 16px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom:12px;">${{st.name}}</div>
                    
                    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px;">
                        <div>
                            <div style="font-size:9px; color:#888; font-weight:600; margin-bottom:4px">TIEMPO VACÍA</div>
                            <div style="font-size:18px; font-weight:700; color:${{color}}">${{st.empty_pct.toFixed(1)}}%</div>
                        </div>
                        <div>
                            <div style="font-size:9px; color:#888; font-weight:600; margin-bottom:4px">PÉRDIDA ESTIMADA</div>
                            <div style="font-size:18px; font-weight:700; color:#cbd5e1">${{st.hours_lost.toFixed(1)}}h</div>
                        </div>
                    </div>
                </div>
            `);
        }});
    </script>
</body>
</html>
    """
    return html

def main():
    print("Cargando datos...")
    df = load_data()
    
    print("Calculando métricas de riesgo...")
    stats = calculate_empty_percentage(df)
    
    print("Generando dashboard premium...")
    html_content = generate_html_map(stats)
    
    out_file = OUTPUT_DIR / "mapa_tiempo_sin_bicis.html"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"✅ Dashboard generado exitosamente: {out_file}")

if __name__ == "__main__":
    main()
