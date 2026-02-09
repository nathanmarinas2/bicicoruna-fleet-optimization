# -*- coding: utf-8 -*-
"""
TIMELAPSE SEMANAL PREMIUM - La Respiracion de Coruna
=====================================================
Visualizacion premium de datos de movilidad
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

from src.utils.config import load_config

BASE_DIR = Path(__file__).parent.parent.parent
DATA_CORUNA = BASE_DIR / "data" / "coruna"
OUTPUT_DIR = BASE_DIR / "dashboard"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_CORUNA / "tracking_data.csv")
    df.columns = df.columns.str.lower()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.rename(columns={'bicis': 'bikes', 'hora': 'hour', 
                            'nombre': 'name', 'capacidad': 'capacity'})
    df['occupancy'] = df['bikes'] / df['capacity'].replace(0, np.nan)
    df['occupancy'] = df['occupancy'].fillna(0)
    return df


def prepare_weekly_data(df):
    """Prepara datos por estacion cada 5 minutos."""
    df['ts_5min'] = df['timestamp'].dt.floor('5min')
    
    grouped = df.groupby(['name', 'ts_5min']).agg({
        'lat': 'first',
        'lon': 'first',
        'capacity': 'first',
        'bikes': 'mean',
        'occupancy': 'mean'
    }).reset_index()
    
    timestamps = sorted(grouped['ts_5min'].unique())
    print(f"   Generando frames para {len(timestamps)} timestamps...")
    
    ts_info = {ts: {
        'timestamp': ts.strftime('%Y-%m-%d %H:%M'),
        'day': ts.day_name(),
        'date': ts.strftime('%d/%m'),
        'hour': ts.strftime('%H:%M')
    } for ts in timestamps}
    
    data_by_ts = {}
    for _, row in grouped.iterrows():
        ts = row['ts_5min']
        if ts not in data_by_ts:
            data_by_ts[ts] = []
        
        data_by_ts[ts].append({
            'name': row['name'],
            'lat': float(row['lat']),
            'lon': float(row['lon']),
            'capacity': int(row['capacity']),
            'bikes': float(row['bikes']),
            'occupancy': float(row['occupancy'])
        })
    
    day_es = {
        'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles',
        'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
    }
    
    frames = []
    for ts in timestamps:
        if ts in data_by_ts:
            info = ts_info[ts]
            frames.append({
                'timestamp': info['timestamp'],
                'day': day_es.get(info['day'], info['day']),
                'date': info['date'],
                'hour': info['hour'],
                'stations': data_by_ts[ts]
            })
            
    return frames


def generate_timelapse_html(frames):
    """Genera HTML premium del timelapse."""
    cfg = load_config()
    empty_threshold = cfg["empty_threshold"]
    
    center_lat = np.mean([s['lat'] for s in frames[0]['stations']])
    center_lon = np.mean([s['lon'] for s in frames[0]['stations']])
    n_frames = len(frames)
    
    html = f'''<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BiciCoruña Analytics</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        :root {{
            --bg-primary: #0D0D0F;
            --bg-secondary: #141418;
            --bg-card: rgba(20, 20, 24, 0.9);
            --border: rgba(255, 255, 255, 0.06);
            --text-primary: #FFFFFF;
            --text-secondary: rgba(255, 255, 255, 0.6);
            --text-muted: rgba(255, 255, 255, 0.4);
            --accent: #3B82F6;
            --accent-glow: rgba(59, 130, 246, 0.3);
            --success: #10B981;
            --warning: #F59E0B;
            --danger: #EF4444;
        }}
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{ 
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            overflow: hidden;
            -webkit-font-smoothing: antialiased;
        }}
        
        #map {{ 
            height: 100vh; 
            width: 100%;
            filter: saturate(0.8) brightness(0.95);
        }}
        
        /* Glass card base */
        .glass {{
            background: var(--bg-card);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--border);
            border-radius: 16px;
        }}
        
        /* Header */
        .header {{
            position: absolute;
            top: 24px;
            left: 24px;
            z-index: 1000;
            padding: 20px 28px;
        }}
        
        .header-title {{
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 8px;
        }}
        
        .header-main {{
            font-size: 24px;
            font-weight: 600;
            color: var(--text-primary);
            letter-spacing: -0.5px;
        }}
        
        /* Time display */
        .time-panel {{
            position: absolute;
            top: 24px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 1000;
            padding: 24px 48px;
            text-align: center;
        }}
        
        .time-day {{
            font-size: 13px;
            font-weight: 500;
            color: var(--accent);
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }}
        
        .time-date {{
            font-size: 12px;
            color: var(--text-muted);
            margin-bottom: 12px;
        }}
        
        .time-clock {{
            font-size: 56px;
            font-weight: 300;
            color: var(--text-primary);
            letter-spacing: -2px;
            font-variant-numeric: tabular-nums;
        }}
        
        /* Stats panel */
        .stats {{
            position: absolute;
            top: 24px;
            right: 24px;
            z-index: 1000;
            padding: 20px 24px;
            min-width: 180px;
        }}
        
        .stat {{
            margin-bottom: 16px;
        }}
        
        .stat:last-child {{
            margin-bottom: 0;
        }}
        
        .stat-label {{
            font-size: 11px;
            font-weight: 500;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 4px;
        }}
        
        .stat-value {{
            font-size: 28px;
            font-weight: 600;
            color: var(--text-primary);
            font-variant-numeric: tabular-nums;
        }}
        
        .stat-value.success {{ color: var(--success); }}
        .stat-value.warning {{ color: var(--warning); }}
        .stat-value.danger {{ color: var(--danger); }}
        
        /* Legend */
        .legend {{
            position: absolute;
            bottom: 120px;
            right: 24px;
            z-index: 1000;
            padding: 16px 20px;
        }}
        
        .legend-title {{
            font-size: 11px;
            font-weight: 500;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 12px;
        }}
        
        .legend-items {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 12px;
            color: var(--text-secondary);
        }}
        
        .legend-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }}
        
        /* Controls */
        .controls {{
            position: absolute;
            bottom: 24px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 1000;
            padding: 16px 24px;
            display: flex;
            align-items: center;
            gap: 20px;
            transition: opacity 0.3s ease, transform 0.3s ease;
        }}
        
        .controls.hidden {{
            opacity: 0;
            pointer-events: none;
            transform: translateX(-50%) translateY(20px);
        }}
        
        .btn {{
            background: transparent;
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-family: inherit;
            font-size: 13px;
            font-weight: 500;
            transition: all 0.2s ease;
        }}
        
        .btn:hover {{
            background: rgba(255, 255, 255, 0.05);
            border-color: rgba(255, 255, 255, 0.1);
        }}
        
        .btn-primary {{
            background: var(--accent);
            border-color: var(--accent);
        }}
        
        .btn-primary:hover {{
            background: #2563EB;
            border-color: #2563EB;
        }}
        
        #timeSlider {{
            width: 400px;
            height: 4px;
            -webkit-appearance: none;
            appearance: none;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 2px;
            outline: none;
        }}
        
        #timeSlider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 14px;
            height: 14px;
            background: var(--accent);
            border-radius: 50%;
            cursor: pointer;
            box-shadow: 0 0 10px var(--accent-glow);
        }}
        
        .progress-text {{
            font-size: 12px;
            color: var(--text-muted);
            font-variant-numeric: tabular-nums;
            min-width: 100px;
        }}
        
        /* Toggle controls button */
        .toggle-controls {{
            position: absolute;
            bottom: 24px;
            right: 24px;
            z-index: 1001;
            padding: 10px 16px;
            font-size: 12px;
            transition: all 0.3s ease;
        }}
        
        .toggle-controls.recording {{
            background: var(--danger);
            border-color: var(--danger);
            animation: pulse-recording 1.5s infinite;
        }}
        
        @keyframes pulse-recording {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.7; }}
        }}
        
        /* Leaflet overrides */
        .leaflet-control-attribution {{ display: none !important; }}
    </style>
</head>
<body>
    <div id="map"></div>
    
    <div class="header glass">
        <div class="header-title">BiciCoruña Analytics</div>
        <div class="header-main">Pulso de la Ciudad</div>
    </div>
    
    <div class="time-panel glass">
        <div class="time-day" id="currentDay">Lunes</div>
        <div class="time-date" id="currentDate">27 Enero 2026</div>
        <div class="time-clock" id="currentTime">00:00</div>
    </div>
    
    <div class="stats glass">
        <div class="stat">
            <div class="stat-label">Ocupación Media</div>
            <div class="stat-value" id="avgOccupancy">0%</div>
        </div>
        <div class="stat">
            <div class="stat-label">Estaciones Vacías</div>
            <div class="stat-value" id="emptyStations">0</div>
        </div>
        <div class="stat">
            <div class="stat-label">Bicis Disponibles</div>
            <div class="stat-value" id="totalBikes">0</div>
        </div>
    </div>
    
    <div class="legend glass">
        <div class="legend-title">Disponibilidad</div>
        <div class="legend-items">
            <div class="legend-item">
                <div class="legend-dot" style="background: #EF4444;"></div>
                <span>Baja (&lt;30%)</span>
            </div>
            <div class="legend-item">
                <div class="legend-dot" style="background: #F59E0B;"></div>
                <span>Media (30-60%)</span>
            </div>
            <div class="legend-item">
                <div class="legend-dot" style="background: #10B981;"></div>
                <span>Alta (&gt;60%)</span>
            </div>
        </div>
    </div>
    
    <div class="controls glass" id="controlsPanel">
        <button class="btn btn-primary" id="playBtn">▶ Play</button>
        <input type="range" id="timeSlider" min="0" max="{n_frames - 1}" value="0">
        <button class="btn" id="speedBtn">1×</button>
        <span class="progress-text" id="progress">1 / {n_frames}</span>
    </div>
    
    <button class="btn glass toggle-controls" id="toggleControlsBtn">🎬 Modo Video</button>
    
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        const frames = {json.dumps(frames)};
        const totalFrames = frames.length;
        const EMPTY_THRESHOLD = {empty_threshold};
        
        // Variables for playback control
        let isPlaying = false;
        let playInterval = null;
        let speed = 1;
        let controlsVisible = true;
        
        const map = L.map('map', {{ 
            zoomControl: false,
            attributionControl: false
        }}).setView([{center_lat}, {center_lon}], 13);
        
        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            subdomains: 'abcd',
            maxZoom: 19
        }}).addTo(map);
        
        let markers = [];
        
        function getColor(occupancy) {{
            if (occupancy < 0.3) return '#EF4444';
            else if (occupancy < 0.6) return '#F59E0B';
            else return '#10B981';
        }}
        
        function updateMap(frameIndex) {{
            markers.forEach(m => map.removeLayer(m));
            markers = [];
            
            const frame = frames[frameIndex];
            let totalOccupancy = 0;
            let emptyCount = 0;
            let totalBikes = 0;
            
            frame.stations.forEach(station => {{
                const color = getColor(station.occupancy);
                const radius = Math.max(5, Math.min(16, station.bikes * 1.2));
                
                const marker = L.circleMarker([station.lat, station.lon], {{
                    radius: radius,
                    fillColor: color,
                    color: 'rgba(255,255,255,0.3)',
                    weight: 1,
                    opacity: 1,
                    fillOpacity: 0.85
                }}).addTo(map);
                
                marker.bindPopup(`
                    <div style="font-family: Inter, sans-serif; padding: 4px;">
                        <div style="font-weight: 600; margin-bottom: 4px;">${{station.name}}</div>
                        <div style="color: #888; font-size: 12px;">
                            ${{station.bikes.toFixed(0)}} / ${{station.capacity}} bicis<br>
                            Ocupación: ${{(station.occupancy * 100).toFixed(0)}}%
                        </div>
                    </div>
                `);
                
                markers.push(marker);
                totalOccupancy += station.occupancy;
                totalBikes += station.bikes;
                if (station.bikes < EMPTY_THRESHOLD) emptyCount++;
            }});
            
            const avgOcc = (totalOccupancy / frame.stations.length * 100).toFixed(0);
            const avgEl = document.getElementById('avgOccupancy');
            avgEl.textContent = avgOcc + '%';
            avgEl.className = 'stat-value ' + (avgOcc < 30 ? 'danger' : avgOcc < 50 ? 'warning' : 'success');
            
            document.getElementById('emptyStations').textContent = emptyCount;
            document.getElementById('totalBikes').textContent = Math.round(totalBikes);
            document.getElementById('currentDay').textContent = frame.day;
            document.getElementById('currentDate').textContent = frame.date;
            document.getElementById('currentTime').textContent = frame.hour;
            document.getElementById('timeSlider').value = frameIndex;
            document.getElementById('progress').textContent = (frameIndex + 1) + ' / ' + totalFrames;
        }}
        
        updateMap(0);
        
        // Toggle controls visibility (for video recording)
        function toggleControls() {{
            controlsVisible = !controlsVisible;
            const controlsPanel = document.getElementById('controlsPanel');
            const toggleBtn = document.getElementById('toggleControlsBtn');
            
            if (controlsVisible) {{
                controlsPanel.classList.remove('hidden');
                toggleBtn.textContent = '🎬 Modo Video';
                toggleBtn.classList.remove('recording');
            }} else {{
                controlsPanel.classList.add('hidden');
                toggleBtn.textContent = '⏹ Mostrar Controles';
                toggleBtn.classList.add('recording');
            }}
        }}
        
        document.getElementById('toggleControlsBtn').addEventListener('click', toggleControls);

        document.getElementById('playBtn').addEventListener('click', function() {{
            isPlaying = !isPlaying;
            this.textContent = isPlaying ? '⏸ Pause' : '▶ Play';
            
            if (isPlaying) {{
                playInterval = setInterval(() => {{
                    let idx = parseInt(document.getElementById('timeSlider').value);
                    idx = (idx + 1) % totalFrames;
                    updateMap(idx);
                }}, 300 / speed);
            }} else {{
                clearInterval(playInterval);
            }}
        }});
        
        document.getElementById('timeSlider').addEventListener('input', function() {{
            updateMap(parseInt(this.value));
        }});
        
        document.getElementById('speedBtn').addEventListener('click', function() {{
            if (speed === 1) speed = 2;
            else if (speed === 2) speed = 4;
            else if (speed === 4) speed = 8;
            else if (speed === 8) speed = 16;
            else speed = 1;
            
            this.textContent = speed + '×';
            
            if (isPlaying) {{
                clearInterval(playInterval);
                playInterval = setInterval(() => {{
                    let idx = parseInt(document.getElementById('timeSlider').value);
                    idx = (idx + 1) % totalFrames;
                    updateMap(idx);
                }}, 300 / speed);
            }}
        }});

        // Keyboard shortcuts
        window.addEventListener('keydown', function(e) {{
            if (e.key.toLowerCase() === 'h') {{
                toggleControls();  // H = ocultar solo controles de reproducción
            }}
            if (e.key === ' ') {{  // Space for play/pause
                document.getElementById('playBtn').click();
                e.preventDefault();
            }}
        }});
    </script>
</body>
</html>'''
    
    return html


def main():
    print("""
    ========================================================
    TIMELAPSE PREMIUM - BICICORUNA ANALYTICS
    ========================================================
    """)
    
    df = load_data()
    
    print("1. Preparando datos de la semana (cada 5 min)...")
    frames = prepare_weekly_data(df)
    print(f"   {len(frames)} frames generados")
    print(f"   Desde: {frames[0]['day']} {frames[0]['date']} {frames[0]['hour']}")
    print(f"   Hasta: {frames[-1]['day']} {frames[-1]['date']} {frames[-1]['hour']}")
    
    print("\n2. Generando timelapse HTML premium...")
    html = generate_timelapse_html(frames)
    
    output_path = OUTPUT_DIR / "timelapse_premium.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n   Timelapse guardado en: {output_path}")
    
    print("""
    ========================================================
    TIMELAPSE PREMIUM GENERADO!
    ========================================================
    
    Abre 'dashboard/timelapse_premium.html' en tu navegador.
    
    Velocidades disponibles: 1x, 2x, 4x, 8x, 16x
    """)


if __name__ == "__main__":
    main()
