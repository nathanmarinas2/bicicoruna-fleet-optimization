import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "coruna"

def analyze_weather_impact():
    print("Analizando impacto meteorológico...")
    # Cargar datos
    df = pd.read_csv(DATA_DIR / "tracking_data.csv")
    df.columns = df.columns.str.lower()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Asegurarnos de que tenemos columnas de clima
    weather_cols = ['lluvia', 'viento', 'temp', 'clima']
    missing = [c for c in weather_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Faltan columnas meteorológicas: {missing}")
        return

    # 1. Definir Volumen de Uso (Actividad)
    # Delta (valor absoluto) = Movimientos (Alquiler + Devolución)
    # Filtramos filas donde Delta != 0 para contar "Eventos de Movilidad"
    # O mejor: Sumamos abs(Delta) por hora/día para ver intensidad total
    
    # Agrupar por hora para tener unidad de análisis comparable
    # Sumamos abs(delta) -> Total viajes
    # Promediamos lluvia/viento/temp en esa hora
    
    # Asegurar que 'delta' es numérico
    df['activity'] = df['delta'].abs()
    
    hourly = df.groupby(pd.Grouper(key='timestamp', freq='h')).agg({
        'activity': 'sum',
        'lluvia': 'mean',
        'viento': 'mean',
        'temp': 'mean'
    }).dropna()
    
    # 2. Segmentación Lluvia vs Seco
    # Consideramos "Lluvia" si precipitacion > 0.1 mm/h
    dry_hours = hourly[hourly['lluvia'] <= 0.1]
    wet_hours = hourly[hourly['lluvia'] > 0.1]
    
    avg_activity_dry = dry_hours['activity'].mean()
    avg_activity_wet = wet_hours['activity'].mean()
    
    impact_rain = (avg_activity_wet - avg_activity_dry) / avg_activity_dry * 100
    
    print(f"\n--- IMPACTO DE LA LLUVIA ---")
    print(f"Horas Secas Analizadas: {len(dry_hours)}")
    print(f"Horas Lluviosas Analizadas: {len(wet_hours)}")
    print(f"Actividad Media (Viajes/hora) [SECO]:   {avg_activity_dry:.1f}")
    print(f"Actividad Media (Viajes/hora) [LLUVIA]: {avg_activity_wet:.1f}")
    print(f">> IMPACTO: {impact_rain:.1f}% de uso")
    
    # 3. Segmentación Viento
    # Viento Fuerte > 20 km/h (ejemplo)
    calm_hours = hourly[hourly['viento'] <= 20]
    windy_hours = hourly[hourly['viento'] > 20]
    
    if len(windy_hours) > 0:
        avg_act_calm = calm_hours['activity'].mean()
        avg_act_wind = windy_hours['activity'].mean()
        impact_wind = (avg_act_wind - avg_act_calm) / avg_act_calm * 100
        
        print(f"\n--- IMPACTO DEL VIENTO (>20 km/h) ---")
        print(f"Horas Ventosas: {len(windy_hours)}")
        print(f"Actividad Media [CALMA]:  {avg_act_calm:.1f}")
        print(f"Actividad Media [VIENTO]: {avg_act_wind:.1f}")
        print(f">> IMPACTO: {impact_wind:.1f}% de uso")
    else:
        print("\nNo se detectaron periodos de viento fuerte (>20 km/h).")

    # 4. Correlación General
    corr = hourly[['activity', 'lluvia', 'viento', 'temp']].corr()['activity']
    print(f"\n--- CORRELACIONES ---")
    print(corr.drop('activity').to_string())

if __name__ == "__main__":
    analyze_weather_impact()
