# -*- coding: utf-8 -*-
"""
ANALISIS NIVEL FACIL - Insights adicionales
============================================
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
DATA_CORUNA = BASE_DIR / "data" / "coruna"


def load_data():
    df = pd.read_csv(DATA_CORUNA / "tracking_data.csv")
    df.columns = df.columns.str.lower()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.rename(columns={'bicis': 'bikes', 'docks': 'docks', 'hora': 'hour',
                            'capacidad': 'capacity', 'nombre': 'name'})
    df['date'] = df['timestamp'].dt.date
    return df


def ranking_estaciones(df):
    """Top 10 estaciones mas y menos usadas."""
    print("\n" + "="*60)
    print("1. RANKING DE ESTACIONES")
    print("="*60)
    
    # Calcular uso: variacion = mas cambios = mas uso
    uso = df.groupby(['id', 'name']).agg({
        'bikes': ['std', 'mean'],
        'capacity': 'first'
    }).reset_index()
    uso.columns = ['id', 'name', 'variacion', 'media', 'capacity']
    uso['uso_relativo'] = uso['variacion'] / uso['capacity'] * 100
    
    print("\n   TOP 10 ESTACIONES MAS ACTIVAS (mayor rotacion):")
    print("   " + "-"*50)
    top = uso.nlargest(10, 'variacion')
    for i, (_, row) in enumerate(top.iterrows(), 1):
        print(f"   {i:2}. {row['name'][:30]:<30} | Variacion: {row['variacion']:.1f} bicis")
    
    print("\n   TOP 10 ESTACIONES MENOS ACTIVAS (menor rotacion):")
    print("   " + "-"*50)
    bottom = uso.nsmallest(10, 'variacion')
    for i, (_, row) in enumerate(bottom.iterrows(), 1):
        print(f"   {i:2}. {row['name'][:30]:<30} | Variacion: {row['variacion']:.1f} bicis")


def patrones_horarios(df):
    """Madrugadores vs nocturnos."""
    print("\n" + "="*60)
    print("2. PATRONES HORARIOS: Madrugadores vs Nocturnos")
    print("="*60)
    
    # Calcular cambio de bicis por hora
    df['bike_change'] = df.groupby('id')['bikes'].diff().abs()
    
    actividad = df.groupby('hour')['bike_change'].mean()
    
    # Clasificar franjas
    madrugada = actividad[0:6].mean()      # 0-6
    manana = actividad[6:12].mean()        # 6-12
    mediodia = actividad[12:15].mean()     # 12-15
    tarde = actividad[15:20].mean()        # 15-20
    noche = actividad[20:24].mean()        # 20-24
    
    franjas = {
        'Madrugada (0-6h)': madrugada,
        'Manana (6-12h)': manana,
        'Mediodia (12-15h)': mediodia,
        'Tarde (15-20h)': tarde,
        'Noche (20-24h)': noche
    }
    
    max_franja = max(franjas, key=franjas.get)
    
    print(f"""
    ACTIVIDAD POR FRANJA HORARIA:
    (cambios medios de bicis por observacion)
    """)
    
    for franja, valor in sorted(franjas.items(), key=lambda x: -x[1] if not pd.isna(x[1]) else 0):
        if pd.isna(valor):
            valor = 0
        bar = "#" * int(valor * 10)
        marker = " <-- PICO" if franja == max_franja else ""
        print(f"   {franja:<20} {bar} {valor:.2f}{marker}")
    
    # Hora pico exacta
    hora_pico = actividad.idxmax()
    print(f"\n   Hora con MAS actividad: {hora_pico}:00")
    print(f"   Hora con MENOS actividad: {actividad.idxmin()}:00")
    
    # Perfil de usuarios
    denom = manana + tarde + noche
    if denom == 0 or pd.isna(denom):
        pct_manana = 0
        pct_tarde = 0
    else:
        pct_manana = (manana / denom) * 100
        pct_tarde = (tarde / denom) * 100
    
    print(f"""
    PERFIL DE USUARIOS:
    - {pct_manana:.0f}% usa BiciCoruna por la MANANA (commuters)
    - {pct_tarde:.0f}% usa BiciCoruna por la TARDE (regreso a casa)
    """)


def eficiencia_sistema(df):
    """Analiza si sobran o faltan bicis."""
    print("\n" + "="*60)
    print("3. EFICIENCIA DEL SISTEMA")
    print("="*60)
    
    # Bicis totales en el sistema por timestamp
    bicis_sistema = df.groupby('timestamp')['bikes'].sum()
    capacidad_total = df.groupby('timestamp')['capacity'].sum().iloc[0]
    
    bicis_media = bicis_sistema.mean()
    bicis_min = bicis_sistema.min()
    bicis_max = bicis_sistema.max()
    
    ocupacion_media = (bicis_media / capacidad_total) * 100
    
    print(f"""
    CAPACIDAD DEL SISTEMA:
    - Capacidad total: {capacidad_total} docks
    - Bicis disponibles (media): {bicis_media:.0f}
    - Bicis disponibles (min): {bicis_min}
    - Bicis disponibles (max): {bicis_max}
    
    OCUPACION GLOBAL: {ocupacion_media:.1f}%
    """)
    
    # Calcular "bicis ideales"
    # Si ocupacion ideal es 50%, necesitamos capacidad/2 bicis
    bicis_ideales = capacidad_total * 0.5
    diferencia = bicis_media - bicis_ideales
    
    if diferencia > 50:
        print(f"   CONCLUSION: Hay {diferencia:.0f} bicis MAS de las ideales")
        print(f"   Recomendacion: El sistema podria funcionar con menos bicis")
    elif diferencia < -50:
        print(f"   CONCLUSION: Faltan {abs(diferencia):.0f} bicis para el optimo")
        print(f"   Recomendacion: Anadir mas bicis al sistema")
    else:
        print(f"   CONCLUSION: El sistema esta BIEN dimensionado")
    
    # Estaciones con problemas de capacidad
    stats = df.groupby('name').agg({
        'bikes': 'mean',
        'capacity': 'first'
    }).reset_index()
    stats['pct'] = stats['bikes'] / stats['capacity'] * 100
    
    print(f"\n   Estaciones con EXCESO de bicis (>70% ocupacion):")
    exceso = stats[stats['pct'] > 70].sort_values('pct', ascending=False)
    for _, row in exceso.head(5).iterrows():
        print(f"   - {row['name'][:30]}: {row['pct']:.0f}% ocupacion")
    
    print(f"\n   Estaciones con DEFICIT de bicis (<30% ocupacion):")
    deficit = stats[stats['pct'] < 30].sort_values('pct')
    for _, row in deficit.head(5).iterrows():
        print(f"   - {row['name'][:30]}: {row['pct']:.0f}% ocupacion")


def dias_atipicos(df):
    """Detecta dias con comportamiento inusual."""
    print("\n" + "="*60)
    print("4. DIAS ATIPICOS")
    print("="*60)
    
    # Actividad por dia
    df['bike_change'] = df.groupby('id')['bikes'].diff().abs()
    actividad_dia = df.groupby('date')['bike_change'].sum()
    
    media = actividad_dia.mean()
    std = actividad_dia.std()
    
    print(f"""
    ACTIVIDAD DIARIA (cambios totales de bicis):
    - Media: {media:.0f} movimientos/dia
    - Desviacion: {std:.0f}
    """)
    
    print("   Actividad por dia:")
    print("   " + "-"*40)
    for date, valor in actividad_dia.items():
        bar = "#" * int(valor / media * 10)
        anomaly = " <-- ATIPICO" if abs(valor - media) > 1.5 * std else ""
        dia_semana = pd.Timestamp(date).day_name()[:3]
        print(f"   {date} ({dia_semana}): {bar} {valor:.0f}{anomaly}")
    
    # Detectar anomalias
    atipicos = actividad_dia[abs(actividad_dia - media) > 1.5 * std]
    if len(atipicos) > 0:
        print(f"\n   DIAS ATIPICOS DETECTADOS: {len(atipicos)}")
        for date, valor in atipicos.items():
            tipo = "MUY ACTIVO" if valor > media else "MUY INACTIVO"
            print(f"   - {date}: {tipo} ({valor:.0f} vs media {media:.0f})")
    else:
        print(f"\n   No se detectaron dias atipicos (todo dentro de lo normal)")


def main():
    print("""
    ========================================================
    ANALISIS ADICIONALES - NIVEL FACIL
    ========================================================
    """)
    
    df = load_data()
    
    ranking_estaciones(df)
    patrones_horarios(df)
    eficiencia_sistema(df)
    dias_atipicos(df)
    
    print("\n" + "="*60)
    print("ANALISIS COMPLETADO")
    print("="*60)


if __name__ == "__main__":
    main()
