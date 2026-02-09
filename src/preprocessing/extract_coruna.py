import pandas as pd
from pathlib import Path

# Rutas
EXCEL_PATH = r'C:\Users\Nathan\Desktop\Universidad\Proyectos\bicicoruña\Bicicoruna_tracking.xlsx'
OUTPUT_DIR = Path(r'C:\Users\Nathan\Desktop\Universidad\Proyectos\bicicoruña\data\coruna')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def prepare_coruna_data():
    print("Reading Excel (this might take a minute)...")
    df = pd.read_excel(EXCEL_PATH, sheet_name='Estaciones')
    
    # Limpieza básica
    # Convertir Timestamp a datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Ordenar por tiempo para que los cálculos de deltas sean correctos
    df = df.sort_values(['ID', 'Timestamp'])
    
    # Guardar en CSV para que el script de fine-tuning lo lea rápido
    output_path = OUTPUT_DIR / 'tracking_data.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✅ ¡Éxito! {len(df):,} filas guardadas en {output_path}")

if __name__ == "__main__":
    prepare_coruna_data()
