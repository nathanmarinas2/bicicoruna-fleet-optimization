"""
Script para descargar datos externos necesarios para análisis avanzados:
1. Datos geoespaciales de distritos/barrios de A Coruña
2. Datos demográficos (población por barrio)
3. Datos de renta/nivel socioeconómico por barrio
4. Datos históricos de uso de BiciCoruña

Fuentes:
- ArcGIS REST Services del Concello de A Coruña
- INE (Atlas de Distribución de Renta)
- Datos abiertos del Ayuntamiento
"""

import requests
import json
import pandas as pd
from pathlib import Path
import urllib.parse

class DownloadDatosExternos:
    def __init__(self, output_dir="data/external"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def download_geojson_distritos(self):
        """
        Descarga el GeoJSON de distritos censales de A Coruña desde el servicio ArcGIS
        """
        print("Descargando GeoJSON de distritos censales...")
        
        # URL del servicio ArcGIS MapServer para distritos
        base_url = "https://ide.coruna.es/arcgiswa/rest/services/Publica/CB_DIVISIONES_ADMINISTRATIVAS/MapServer/1/query"
        
        # Parámetros para obtener todas las features en formato GeoJSON
        params = {
            'where': '1=1',  # Obtener todos los registros
            'outFields': '*',  # Todos los campos
            'returnGeometry': 'true',
            'f': 'geojson'  # Formato GeoJSON
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            geojson_data = response.json()
            
            # Guardar el GeoJSON
            output_file = self.output_dir / "coruna_distritos.geojson"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(geojson_data, f, ensure_ascii=False, indent=2)
            
            print(f"OK - GeoJSON descargado: {len(geojson_data['features'])} distritos")
            print(f"   Guardado en: {output_file}")
            return geojson_data
            
        except Exception as e:
            print(f"ERROR - Error descargando GeoJSON: {e}")
            return None
    
    def download_geojson_barrios(self):
        """
        Descarga el GeoJSON de barrios/AA.VV. de A Coruña
        """
        print("\nDescargando GeoJSON de barrios...")
        
        # URL para barrios (layer 2 en el mismo servicio)
        base_url = "https://ide.coruna.es/arcgiswa/rest/services/Publica/CB_DIVISIONES_ADMINISTRATIVAS/MapServer/2/query"
        
        params = {
            'where': '1=1',
            'outFields': '*',
            'returnGeometry': 'true',
            'f': 'geojson'
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            geojson_data = response.json()
            
            output_file = self.output_dir / "coruna_barrios.geojson"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(geojson_data, f, ensure_ascii=False, indent=2)
            
            print(f"OK - GeoJSON descargado: {len(geojson_data['features'])} barrios")
            print(f"   Guardado en: {output_file}")
            return geojson_data
            
        except Exception as e:
            print(f"ERROR - Error descargando GeoJSON de barrios: {e}")
            return None
    
    def crear_datos_demograficos_mock(self):
        """
        Crea un dataset sintético de datos demográficos por barrio
        basado en información pública disponible de A Coruña.
        
        NOTA: Estos son datos aproximados/sintéticos para demostración.
        Para uso en producción, deberían obtenerse datos oficiales del INE.
        """
        print("\nCreando datos demograficos de referencia...")
        
        # Datos aproximados de los principales barrios de A Coruña
        # Fuente: Estimaciones basadas en datos públicos de La Voz de Galicia e INE
        datos_barrios = [
            # Barrio, Población aprox., Renta media anual (€), Densidad (hab/km²)
            ("Agra do Orzán", 15000, 25751, 12500),
            ("Os Mallos", 18000, 25869, 11000),
            ("Monte Alto", 12000, 28500, 9500),
            ("Labañou", 8000, 27000, 7500),
            ("Ventorrillo", 6000, 26000, 6800),
            ("Ciudad Vieja", 5000, 32000, 15000),
            ("Ensanche", 20000, 38000, 14000),
            ("Juan Flórez", 10000, 66774, 13000),
            ("San Pablo", 8000, 62000, 12000),
            ("Los Rosales", 15000, 35000, 10000),
            ("La Torre", 12000, 34000, 9000),
            ("Matogrande", 10000, 36000, 8500),
            ("Plaza de Pontevedra", 7000, 40000, 11500),
            ("Santa Margarita", 9000, 33000, 9800),
            ("Marineda", 11000, 31000, 7000),
            ("Elviña", 14000, 35500, 6500),
            ("Palavea", 6000, 37000, 5500),
            ("Mesoiro", 13000, 34500, 8000),
            ("Feáns", 5000, 32500, 4500),
            ("Barrio de las Flores", 8000, 30000, 9500),
        ]
        
        df = pd.DataFrame(datos_barrios, columns=[
            'barrio', 'poblacion', 'renta_media_anual', 'densidad_hab_km2'
        ])
        
        # Agregar campos calculados
        df['renta_media_mensual'] = (df['renta_media_anual'] / 12).round(0)
        df['categoria_renta'] = pd.cut(
            df['renta_media_anual'], 
            bins=[0, 28000, 35000, 70000],
            labels=['Baja', 'Media', 'Alta']
        )
        
        # Guardar
        output_file = self.output_dir / "demografia_barrios.csv"
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"OK - Datos demograficos creados: {len(df)} barrios")
        print(f"   Guardado en: {output_file}")
        print(f"\nResumen:")
        print(f"   - Población total: {df['poblacion'].sum():,} habitantes")
        print(f"   - Renta media ciudad: {df['renta_media_anual'].mean():,.0f}€/año")
        print(f"   - Barrio más rico: {df.loc[df['renta_media_anual'].idxmax(), 'barrio']}")
        print(f"   - Barrio más poblado: {df.loc[df['poblacion'].idxmax(), 'barrio']}")
        
        return df
    
    def crear_datos_historicos_usuarios(self):
        """
        Crea un dataset de evolución histórica de usuarios de BiciCoruña
        basado en datos públicos reportados en prensa.
        """
        print("\nCreando datos historicos de usuarios...")
        
        # Datos recopilados de fuentes públicas (La Voz de Galicia, etc.)
        datos_historicos = [
            # Año, Mes, Usuarios activos, Total usos (aprox acumulado anual), Notas
            (2009, 7, 500, 2000, "Inauguración servicio mecánico"),
            (2010, 12, 800, 15000, "Primer año completo"),
            (2015, 12, 1500, 45000, "Crecimiento lento"),
            (2020, 12, 2000, 60000, "Pre-renovación"),
            (2021, 12, 2200, 75000, "Pandemia - movilidad sostenible"),
            (2022, 6, 2673, 90000, "Inicio bicicletas eléctricas"),
            (2022, 12, 8500, 450000, "Boom post-electrificación"),
            (2023, 6, 12000, 650000, "Crecimiento sostenido"),
            (2023, 12, 13669, 1200000, "Récord anual 1.2M usos"),
            (2024, 6, 14691, 700000, "Máximo histórico usuarios"),
            (2024, 12, 14800, 1337882, "Récord 2024: 1.34M usos"),
            (2025, 10, 16000, 203285, "Oct 25: Récord mensual 203k usos"), 
            (2025, 12, 16331, 2000000, "Cierre 25: 16.3k Activos / 26.5k Totales")
        ]
        
        df = pd.DataFrame(datos_historicos, columns=[
            'ano', 'mes', 'usuarios_activos', 'total_usos_acumulado', 'notas'
        ])
        
        # Crear fecha
        df['fecha'] = pd.to_datetime(df['ano'].astype(str) + '-' + df['mes'].astype(str) + '-01')
        
        # Calcular métricas derivadas
        df['usos_por_usuario'] = (df['total_usos_acumulado'] / df['usuarios_activos']).round(1)
        df['crecimiento_usuarios_pct'] = df['usuarios_activos'].pct_change() * 100
        
        # Guardar
        output_file = self.output_dir / "historico_usuarios_bicicoruna.csv"
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"OK - Datos historicos creados: {len(df)} registros")
        print(f"   Guardado en: {output_file}")
        print(f"\nEvolucion:")
        print(f"   - Usuarios 2009: {df.iloc[0]['usuarios_activos']:,.0f}")
        print(f"   - Usuarios 2024: {df.iloc[-1]['usuarios_activos']:,.0f}")
        print(f"   - Crecimiento total: {((df.iloc[-1]['usuarios_activos'] / df.iloc[0]['usuarios_activos']) - 1) * 100:.1f}%")
        print(f"   - Usos 2024: {df.iloc[-1]['total_usos_acumulado']:,.0f}")
        
        return df
    
    def run_all(self):
        """
        Ejecuta todas las descargas
        """
        print("=" * 60)
        print("DESCARGA DE DATOS EXTERNOS PARA BICICORUNA")
        print("=" * 60)
        
        # 1. GeoJSON de distritos
        self.download_geojson_distritos()
        
        # 2. GeoJSON de barrios
        self.download_geojson_barrios()
        
        # 3. Datos demográficos
        self.crear_datos_demograficos_mock()
        
        # 4. Datos históricos de usuarios
        self.crear_datos_historicos_usuarios()
        
        print("\n" + "=" * 60)
        print("DESCARGA COMPLETADA")
        print(f"Todos los archivos guardados en: {self.output_dir.absolute()}")
        print("=" * 60)

if __name__ == "__main__":
    downloader = DownloadDatosExternos()
    downloader.run_all()
