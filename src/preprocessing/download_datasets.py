"""
=============================================================================
📥 DESCARGA DE DATASETS PARA PREENTRENAMIENTO
=============================================================================
Datasets:
1. Washington DC - Capital Bikeshare (Kaggle - dataset clásico)
2. Barcelona - Bicing (Kaggle - 250M filas)
3. Madrid - BiciMAD (EMT Open Data)

Ejecutar: python src/preprocessing/download_datasets.py
=============================================================================
"""

import os
import requests
import zipfile
import io
from pathlib import Path

# Directorio base
BASE_DIR = Path(__file__).parent.parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"

def download_dc_dataset():
    """
    Descarga el dataset de Washington DC desde Kaggle.
    Este es el dataset clásico de bike-sharing, muy limpio y documentado.
    
    NOTA: Para Kaggle necesitas configurar las credenciales primero.
    Alternativa: descarga manual desde:
    https://www.kaggle.com/datasets/marklvl/bike-sharing-dataset
    """
    print("\n" + "="*60)
    print("📥 DATASET: Washington DC (Capital Bikeshare)")
    print("="*60)
    
    # URL alternativa (UCI Machine Learning Repository - mismo dataset)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
    
    output_dir = DATA_RAW / "dc"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Descargando desde UCI Repository...")
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        # Extraer ZIP
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(output_dir)
        
        print(f"✅ Descargado en: {output_dir}")
        print(f"   Archivos: {list(output_dir.glob('*.csv'))}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n📋 Descarga manual:")
        print("   1. Ve a: https://www.kaggle.com/datasets/marklvl/bike-sharing-dataset")
        print("   2. Descarga y extrae en: data/raw/dc/")
        return False


def download_barcelona_info():
    """
    Información para descargar Barcelona (Kaggle).
    El dataset es muy grande (varios GB), mejor descarga manual.
    """
    print("\n" + "="*60)
    print("📥 DATASET: Barcelona (Bicing)")
    print("="*60)
    
    output_dir = DATA_RAW / "barcelona"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("""
    ⚠️  El dataset de Barcelona es muy grande (~5GB comprimido).
    
    📋 Descarga manual:
    1. Ve a: https://www.kaggle.com/datasets/jsagrera/bcn-bike-sharing-dataset-bicing-stations
    2. Haz clic en "Download" (necesitas cuenta Kaggle)
    3. Extrae el archivo en: data/raw/barcelona/
    
    Alternativa con kaggle CLI:
    kaggle datasets download -d jsagrera/bcn-bike-sharing-dataset-bicing-stations -p data/raw/barcelona --unzip
    """)
    
    # Crear archivo de instrucciones
    instructions_file = output_dir / "DESCARGAR_AQUI.txt"
    with open(instructions_file, "w", encoding="utf-8") as f:
        f.write("""INSTRUCCIONES PARA DESCARGAR BARCELONA:

1. Ve a: https://www.kaggle.com/datasets/jsagrera/bcn-bike-sharing-dataset-bicing-stations
2. Inicia sesión en Kaggle
3. Haz clic en "Download"
4. Extrae el contenido en esta carpeta

El archivo principal se llama algo como:
- bicing_data.parquet (recomendado, más pequeño)
- o varios archivos CSV

Después de descargar, ejecuta de nuevo el script de preprocesamiento.
""")
    
    print(f"   Instrucciones guardadas en: {instructions_file}")
    return False


def download_madrid_dataset():
    """
    Descarga datos de BiciMAD desde EMT Open Data.
    """
    print("\n" + "="*60)
    print("📥 DATASET: Madrid (BiciMAD)")
    print("="*60)
    
    output_dir = DATA_RAW / "madrid"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # EMT Madrid tiene varios endpoints
    # Vamos a intentar descargar datos de uso
    
    print("""
    📋 Opciones para Madrid:
    
    OPCIÓN 1 - Portal EMT (histórico completo):
    https://opendata.emtmadrid.es/Datos-Estaticos/Datos-historicos-bicimad
    
    OPCIÓN 2 - Kaggle (preprocesado):
    https://www.kaggle.com/datasets/simonm3/bicimad-bases-of-the-public-electric-bicycle-service
    
    Descarga el que prefieras y extrae en: data/raw/madrid/
    """)
    
    # Crear archivo de instrucciones
    instructions_file = output_dir / "DESCARGAR_AQUI.txt"
    with open(instructions_file, "w", encoding="utf-8") as f:
        f.write("""INSTRUCCIONES PARA DESCARGAR MADRID:

OPCIÓN 1 - EMT Open Data (datos oficiales):
1. Ve a: https://opendata.emtmadrid.es/
2. Busca "BiciMAD" en el buscador
3. Descarga los archivos de viajes históricos
4. Extrae en esta carpeta

OPCIÓN 2 - Kaggle (más fácil):
1. Ve a: https://www.kaggle.com/datasets/simonm3/bicimad-bases-of-the-public-electric-bicycle-service
2. Descarga y extrae aquí

Después de descargar, ejecuta de nuevo el script de preprocesamiento.
""")
    
    print(f"   Instrucciones guardadas en: {instructions_file}")
    return False


def check_existing_datasets():
    """Verifica qué datasets ya están descargados."""
    print("\n" + "="*60)
    print("🔍 VERIFICANDO DATASETS EXISTENTES")
    print("="*60)
    
    datasets = {
        "dc": DATA_RAW / "dc",
        "barcelona": DATA_RAW / "barcelona", 
        "madrid": DATA_RAW / "madrid"
    }
    
    status = {}
    for name, path in datasets.items():
        if path.exists():
            files = list(path.glob("*"))
            csv_files = list(path.glob("*.csv"))
            parquet_files = list(path.glob("*.parquet"))
            
            if csv_files or parquet_files:
                status[name] = "✅ Listo"
                print(f"   {name}: ✅ {len(csv_files)} CSV, {len(parquet_files)} Parquet")
            else:
                status[name] = "⚠️ Carpeta vacía"
                print(f"   {name}: ⚠️ Carpeta existe pero sin datos")
        else:
            status[name] = "❌ No descargado"
            print(f"   {name}: ❌ No encontrado")
    
    return status


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║  🚲 DESCARGA DE DATASETS PARA BICIPREDICT                 ║
    ║                                                           ║
    ║  Datasets necesarios:                                     ║
    ║  • Washington DC (automático)                             ║
    ║  • Barcelona - Kaggle (manual, ~5GB)                      ║
    ║  • Madrid - EMT (manual)                                  ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Verificar estado actual
    status = check_existing_datasets()
    
    # Descargar DC (automático)
    if status.get("dc") != "✅ Listo":
        download_dc_dataset()
    
    # Instrucciones para Barcelona
    if status.get("barcelona") != "✅ Listo":
        download_barcelona_info()
    
    # Instrucciones para Madrid
    if status.get("madrid") != "✅ Listo":
        download_madrid_dataset()
    
    print("\n" + "="*60)
    print("📋 RESUMEN")
    print("="*60)
    print("""
    Cuando tengas los datasets descargados, ejecuta:
    
    python src/preprocessing/prepare_data.py
    
    Esto unificará todos los datasets en un formato común
    optimizado para transfer learning.
    """)


if __name__ == "__main__":
    main()
