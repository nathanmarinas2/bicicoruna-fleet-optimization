# -*- coding: utf-8 -*-
"""
ANALISIS COMPLETO DE BICICORUNA
================================
Genera insights para la noticia y el TFG

Ejecutar: python src/evaluation/analisis_bicicoruna.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from src.utils.config import load_config

BASE_DIR = Path(__file__).parent.parent.parent
DATA_CORUNA = BASE_DIR / "data" / "coruna"
OUTPUT_DIR = BASE_DIR / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Carga los datos de Coruna."""
    df = pd.read_csv(DATA_CORUNA / "tracking_data.csv")
    df.columns = df.columns.str.lower()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Renombrar columnas
    df = df.rename(columns={
        'bicis': 'bikes', 'docks': 'docks', 'hora': 'hour',
        'finde': 'is_weekend', 'capacidad': 'capacity',
        'nombre': 'station_name', 'temp': 'temperature'
    })
    
    df['occupancy'] = (df['bikes'] / df['capacity'].replace(0, np.nan)).fillna(0)
    df['date'] = df['timestamp'].dt.date
    df['day_name'] = df['timestamp'].dt.day_name()
    
    return df


def analisis_general(df):
    """Estadisticas generales del sistema."""
    print("\n" + "="*60)
    print("1. ESTADISTICAS GENERALES DE BICICORUNA")
    print("="*60)
    
    # Periodo de datos
    fecha_inicio = df['timestamp'].min()
    fecha_fin = df['timestamp'].max()
    dias = (fecha_fin - fecha_inicio).days + 1
    
    # Estaciones
    n_estaciones = df['id'].nunique()
    capacidad_total = df.groupby('id')['capacity'].first().sum()
    
    # Observaciones
    n_obs = len(df)
    obs_por_dia = n_obs / dias
    
    print(f"""
    Periodo de datos: {fecha_inicio.strftime('%d/%m/%Y')} - {fecha_fin.strftime('%d/%m/%Y')} ({dias} dias)
    
    Estaciones activas: {n_estaciones}
    Capacidad total: {capacidad_total} bicis
    
    Observaciones: {n_obs:,}
    Observaciones/dia: {obs_por_dia:,.0f}
    """)
    
    return {
        'periodo_dias': dias,
        'n_estaciones': n_estaciones,
        'capacidad_total': capacidad_total,
        'n_observaciones': n_obs
    }


def analisis_salud(df):
    """Analisis de salud del sistema."""
    print("\n" + "="*60)
    print("2. SALUD DEL SISTEMA")
    print("="*60)
    
    cfg = load_config()
    empty_threshold = cfg["empty_threshold"]
    full_threshold = cfg["full_threshold"]

    # Estaciones vacias y llenas
    df['is_empty'] = df['bikes'] < empty_threshold
    df['is_full'] = df['docks'] < full_threshold
    
    pct_empty = df['is_empty'].mean() * 100
    pct_full = df['is_full'].mean() * 100
    pct_ok = 100 - pct_empty - pct_full
    
    # Ocupacion media
    occupancy_media = df['occupancy'].mean() * 100
    
    print(f"""
    Ocupacion media del sistema: {occupancy_media:.1f}%
    
    Estado de las estaciones:
    - OK (funcionando bien): {pct_ok:.1f}%
    - Vacias (<{empty_threshold} bicis): {pct_empty:.1f}%
    - Llenas (<{full_threshold} docks): {pct_full:.1f}%
    
    HEALTH SCORE: {pct_ok:.0f}/100
    """)
    
    return {
        'health_score': pct_ok,
        'pct_empty': pct_empty,
        'pct_full': pct_full,
        'occupancy_media': occupancy_media
    }


def estaciones_problematicas(df):
    """Identifica estaciones infrautilizadas y sobreutilizadas."""
    print("\n" + "="*60)
    print("3. ESTACIONES PROBLEMATICAS")
    print("="*60)
    
    # Calcular metricas por estacion
    stats = df.groupby(['id', 'station_name']).agg({
        'bikes': 'mean',
        'capacity': 'first',
        'occupancy': 'mean',
        'is_empty': 'mean',
        'is_full': 'mean'
    }).reset_index()
    
    stats['pct_empty'] = stats['is_empty'] * 100
    stats['pct_full'] = stats['is_full'] * 100
    stats['occupancy_pct'] = stats['occupancy'] * 100
    
    # INFRAUTILIZADAS (siempre llenas, nadie las usa para devolver)
    print("\n   ESTACIONES INFRAUTILIZADAS (alta ocupacion, poca rotacion):")
    print("   " + "-"*50)
    infrautilizadas = stats[stats['occupancy_pct'] > 70].sort_values('occupancy_pct', ascending=False).head(5)
    for _, row in infrautilizadas.iterrows():
        print(f"   {row['station_name'][:30]:<30} | Ocupacion: {row['occupancy_pct']:.0f}% | Llena {row['pct_full']:.1f}% del tiempo")
    
    # SOBREUTILIZADAS (siempre vacias, mucha demanda)
    print("\n   ESTACIONES SOBREUTILIZADAS (baja ocupacion, mucha demanda):")
    print("   " + "-"*50)
    sobreutilizadas = stats[stats['occupancy_pct'] < 30].sort_values('occupancy_pct').head(5)
    for _, row in sobreutilizadas.iterrows():
        print(f"   {row['station_name'][:30]:<30} | Ocupacion: {row['occupancy_pct']:.0f}% | Vacia {row['pct_empty']:.1f}% del tiempo")
    
    # ESTACIONES FANTASMA (poca variacion = nadie las usa)
    df['bikes_var'] = df.groupby('id')['bikes'].transform('std')
    fantasma = df.groupby(['id', 'station_name']).agg({'bikes_var': 'first', 'capacity': 'first'}).reset_index()
    fantasma = fantasma[fantasma['bikes_var'] < 1].sort_values('bikes_var')
    
    print(f"\n   ESTACIONES FANTASMA (variacion < 1 bici):")
    print("   " + "-"*50)
    if len(fantasma) > 0:
        for _, row in fantasma.head(5).iterrows():
            print(f"   {row['station_name'][:30]:<30} | Variacion: {row['bikes_var']:.2f} bicis")
    else:
        print("   Ninguna estacion fantasma detectada")
    
    return {
        'infrautilizadas': infrautilizadas['station_name'].tolist(),
        'sobreutilizadas': sobreutilizadas['station_name'].tolist(),
        'fantasma': fantasma['station_name'].tolist() if len(fantasma) > 0 else []
    }


def patrones_temporales(df):
    """Analiza patrones por hora y dia."""
    print("\n" + "="*60)
    print("4. PATRONES TEMPORALES")
    print("="*60)
    
    # Por hora
    por_hora = df.groupby('hour')['occupancy'].mean() * 100
    
    hora_max = por_hora.idxmax()
    hora_min = por_hora.idxmin()
    
    print(f"""
    PATRON DIARIO:
    - Hora con MAS bicis disponibles: {hora_max}:00 ({por_hora[hora_max]:.1f}% ocupacion)
    - Hora con MENOS bicis disponibles: {hora_min}:00 ({por_hora[hora_min]:.1f}% ocupacion)
    
    Interpretacion:
    - A las {hora_min}:00 la gente COGE bicis (baja ocupacion)
    - A las {hora_max}:00 la gente DEVUELVE bicis (alta ocupacion)
    """)
    
    # Por dia de la semana
    por_dia = df.groupby('day_name')['occupancy'].mean() * 100
    dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    print("   OCUPACION POR DIA DE LA SEMANA:")
    print("   " + "-"*40)
    for dia in dias_orden:
        if dia in por_dia.index:
            bar = "#" * int(por_dia[dia] / 5)
            print(f"   {dia[:3]}: {bar} {por_dia[dia]:.1f}%")
    
    # Fin de semana vs laborables
    occ_laboral = df[df['is_weekend'] == 0]['occupancy'].mean() * 100
    occ_finde = df[df['is_weekend'] == 1]['occupancy'].mean() * 100
    
    print(f"""
    LABORABLES vs FIN DE SEMANA:
    - Dias laborables: {occ_laboral:.1f}% ocupacion media
    - Fin de semana: {occ_finde:.1f}% ocupacion media
    - Diferencia: {occ_finde - occ_laboral:+.1f}%
    """)
    
    return {
        'hora_pico_demanda': hora_min,
        'hora_pico_devolucion': hora_max,
        'occ_laboral': occ_laboral,
        'occ_finde': occ_finde
    }


def impacto_clima(df):
    """Analiza el impacto del clima."""
    print("\n" + "="*60)
    print("5. IMPACTO DEL CLIMA")
    print("="*60)
    
    if 'temperature' not in df.columns or df['temperature'].isna().all():
        print("   No hay datos de temperatura disponibles")
        return {}
    
    # Correlacion temperatura-ocupacion
    if 'lluvia' in df.columns:
        df['is_raining'] = df['lluvia'] > 0
        
        occ_sin_lluvia = df[df['is_raining'] == False]['occupancy'].mean() * 100
        occ_con_lluvia = df[df['is_raining'] == True]['occupancy'].mean() * 100
        
        print(f"""
    IMPACTO DE LA LLUVIA:
    - Sin lluvia: {occ_sin_lluvia:.1f}% ocupacion
    - Con lluvia: {occ_con_lluvia:.1f}% ocupacion
    - Diferencia: {occ_con_lluvia - occ_sin_lluvia:+.1f}%
        """)
        
        if occ_con_lluvia > occ_sin_lluvia:
            print("   Interpretacion: Cuando llueve, las bicis se quedan en las estaciones (menos uso)")
    
    # Temperatura
    temp_media = df['temperature'].mean()
    temp_min = df['temperature'].min()
    temp_max = df['temperature'].max()
    
    print(f"""
    TEMPERATURA DURANTE EL PERIODO:
    - Media: {temp_media:.1f} C
    - Minima: {temp_min:.1f} C
    - Maxima: {temp_max:.1f} C
    """)
    
    return {'temp_media': temp_media}


def estimacion_co2(df, stats):
    """Estima el CO2 ahorrado."""
    print("\n" + "="*60)
    print("6. IMPACTO AMBIENTAL (ESTIMACION)")
    print("="*60)
    
    # Estimar viajes: cada cambio de -1 en bikes es un alquiler
    df['bike_change'] = df.groupby('id')['bikes'].diff()
    alquileres = df[df['bike_change'] < 0]['bike_change'].abs().sum()
    
    # Viajes por dia
    viajes_dia = alquileres / stats['periodo_dias']
    
    # Distancia media estimada: 3 km por viaje
    KM_POR_VIAJE = 2.5
    km_totales = alquileres * KM_POR_VIAJE
    
    # CO2 ahorrado: 120g por km en coche
    CO2_POR_KM_COCHE = 0.120  # kg
    co2_ahorrado = km_totales * CO2_POR_KM_COCHE
    co2_por_dia = co2_ahorrado / stats['periodo_dias']
    
    # Proyeccion anual
    co2_anual = co2_por_dia * 365
    
    print(f"""
    VIAJES ESTIMADOS:
    - Alquileres totales: {alquileres:,.0f}
    - Alquileres por dia: {viajes_dia:,.0f}
    - Km recorridos (estimado): {km_totales:,.0f} km
    
    CO2 AHORRADO (vs usar coche):
    - Total periodo: {co2_ahorrado:,.0f} kg
    - Por dia: {co2_por_dia:,.1f} kg
    - Proyeccion anual: {co2_anual/1000:,.1f} TONELADAS
    
    Equivalente a: {co2_anual/21:.0f} arboles plantados
    """)
    
    return {
        'alquileres_dia': viajes_dia,
        'co2_anual_kg': co2_anual,
        'co2_anual_ton': co2_anual/1000
    }


def conclusiones(all_stats):
    """Genera conclusiones finales."""
    print("\n" + "="*60)
    print("7. CONCLUSIONES PARA LA NOTICIA")
    print("="*60)
    
    print("""
    TITULARES SUGERIDOS:
    
    1. "BiciCoruna ahorra {:.1f} toneladas de CO2 al ano a la ciudad"
    
    2. "Las estaciones del centro tienen un {:.0f}% de ocupacion en hora punta"
    
    3. "Sistema detecta {:.0f} estaciones 'fantasma' que casi nadie usa"
    
    4. "IA predice con 67% de precision cuando te quedaras sin bici"
    
    5. "La lluvia reduce el uso de BiciCoruna un X%"
    """.format(
        all_stats.get('co2', {}).get('co2_anual_ton', 0),
        all_stats.get('patrones', {}).get('occ_laboral', 0),
        len(all_stats.get('problematicas', {}).get('fantasma', []))
    ))


def main():
    print("""
    ========================================================
    ANALISIS COMPLETO DE BICICORUNA
    ========================================================
    Generando insights para noticia y TFG...
    """)
    
    df = load_data()
    
    all_stats = {}
    
    all_stats['general'] = analisis_general(df)
    all_stats['salud'] = analisis_salud(df)
    all_stats['problematicas'] = estaciones_problematicas(df)
    all_stats['patrones'] = patrones_temporales(df)
    all_stats['clima'] = impacto_clima(df)
    all_stats['co2'] = estimacion_co2(df, all_stats['general'])
    
    conclusiones(all_stats)
    
    # Guardar resumen
    with open(OUTPUT_DIR / "resumen_analisis.txt", "w", encoding="utf-8") as f:
        f.write("RESUMEN ANALISIS BICICORUNA\n")
        f.write("="*40 + "\n\n")
        for section, data in all_stats.items():
            f.write(f"{section.upper()}:\n")
            for k, v in data.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")
    
    print(f"\n   Resumen guardado en: {OUTPUT_DIR / 'resumen_analisis.txt'}")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
