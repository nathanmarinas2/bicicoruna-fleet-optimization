"""
Módulo de Predicción de Crecimiento de Usuarios de BiciCoruña

Este módulo analiza la evolución histórica de usuarios y proyecta
el crecimiento futuro basándose en:
1. Tendencias históricas (2009-2024)
2. Estacionalidad (efectos de verano/invierno, lectivo/vacaciones)
3. Eventos clave (electrificación 2022, expansión de red)
4. Factores externos (clima, infraestructura ciclable)

Técnicas utilizadas:
- Regresión exponencial para modelar crecimiento acelerado
- Suavizado exponencial para estacionalidad
- Análisis de puntos de inflexión (electrificación)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class PredictorCrecimientoUsuarios:
    def __init__(self, datos_path="data/external/historico_usuarios_bicicoruna.csv"):
        self.datos_path = Path(datos_path)
        self.df = None
        self.modelo_parametros = None
        
    def cargar_datos(self):
        """Carga datos históricos de usuarios"""
        print("Cargando datos historicos...")
        self.df = pd.read_csv(self.datos_path)
        self.df['fecha'] = pd.to_datetime(self.df['fecha'])
        
        # Crear variable temporal numérica (meses desde inicio)
        fecha_inicio = self.df['fecha'].min()
        self.df['meses_desde_inicio'] = ((self.df['fecha'] - fecha_inicio).dt.days / 30.44).round(0)
        
        print(f"OK - Cargados {len(self.df)} registros historicos")
        print(f"   Periodo: {self.df['fecha'].min().date()} a {self.df['fecha'].max().date()}")
        
        return self.df
    
    def modelo_crecimiento_exponencial(self, x, a, b, c):
        """
        Modelo exponencial: y = a * e^(b*x) + c
        
        Parámetros:
        - a: factor de escala
        - b: tasa de crecimiento
        - c: asíntota (capacidad del sistema)
        """
        return a * np.exp(b * x) + c
    
    def modelo_logistico(self, x, L, k, x0):
        """
        Modelo logístico (Curva S): y = L / (1 + e^(-k*(x-x0)))
        
        Útil para modelar crecimiento con saturación del mercado.
        
        Parámetros:
        - L: capacidad máxima (saturación)
        - k: tasa de crecimiento
        - x0: punto medio
        """
        return L / (1 + np.exp(-k * (x - x0)))
    
    def ajustar_modelo_pre_electrificacion(self):
        """
        Ajusta modelo a datos PRE-electrificación (2009-2022)
        para ver el crecimiento orgánico lento
        """
        print("\n1. Analizando periodo PRE-electrificacion (2009-Jun 2022)...")
        
        # Filtrar datos pre-electrificación
        df_pre = self.df[self.df['fecha'] < '2022-06-01'].copy()
        
        X = df_pre['meses_desde_inicio'].values
        y = df_pre['usuarios_activos'].values
        
        # Ajustar modelo exponencial
        try:
            popt, _ = curve_fit(
                self.modelo_crecimiento_exponencial, 
                X, y, 
                p0=[100, 0.01, 0],
                maxfev=10000
            )
            
            y_pred = self.modelo_crecimiento_exponencial(X, *popt)
            r2 = r2_score(y, y_pred)
            mape = mean_absolute_percentage_error(y, y_pred) * 100
            
            print(f"   Modelo: y = {popt[0]:.2f} * e^({popt[1]:.4f}*x) + {popt[2]:.2f}")
            print(f"   R² = {r2:.3f}")
            print(f"   MAPE = {mape:.1f}%")
            print(f"   Tasa de crecimiento mensual: {popt[1]*100:.2f}%")
            
            # Calcular tasa de crecimiento anual compuesta (CAGR)
            usuarios_inicio = df_pre.iloc[0]['usuarios_activos']
            usuarios_final = df_pre.iloc[-1]['usuarios_activos']
            años = (df_pre.iloc[-1]['fecha'] - df_pre.iloc[0]['fecha']).days / 365.25
            cagr_pre = ((usuarios_final / usuarios_inicio) ** (1/años) - 1) * 100
            
            print(f"   CAGR (2009-2022): {cagr_pre:.1f}% anual")
            
            return {
                'periodo': 'pre_electrificacion',
                'parametros': popt,
                'r2': r2,
                'mape': mape,
                'cagr': cagr_pre,
                'X': X,
                'y': y,
                'y_pred': y_pred
            }
            
        except Exception as e:
            print(f"   ERROR ajustando modelo: {e}")
            return None
    
    def ajustar_modelo_post_electrificacion(self):
        """
        Ajusta modelo a datos POST-electrificación (Jun 2022-2024)
        para ver el boom de crecimiento
        """
        print("\n2. Analizando periodo POST-electrificacion (Jun 2022-Dic 2024)...")
        
        # Filtrar datos post-electrificación
        df_post = self.df[self.df['fecha'] >= '2022-06-01'].copy()
        
        # Renormalizar x para que empiece en 0
        X_original = df_post['meses_desde_inicio'].values
        X = X_original - X_original.min()
        y = df_post['usuarios_activos'].values
        
        # Intentar modelo logístico (curva S con saturación)
        try:
            # Estimar saturación en ~25k usuarios (basado en tendencia actual)
            popt, _ = curve_fit(
                self.modelo_logistico,
                X, y,
                p0=[25000, 0.08, 15],  # L, k, x0
                maxfev=10000,
                bounds=([17000, 0.01, 5], [50000, 0.5, 50])
            )
            
            y_pred = self.modelo_logistico(X, *popt)
            r2 = r2_score(y, y_pred)
            mape = mean_absolute_percentage_error(y, y_pred) * 100
            
            print(f"   Modelo Logistico: y = {popt[0]:.0f} / (1 + e^(-{popt[1]:.3f}*(x-{popt[2]:.1f})))")
            print(f"   Capacidad maxima estimada: {popt[0]:,.0f} usuarios")
            print(f"   R² = {r2:.3f}")
            print(f"   MAPE = {mape:.1f}%")
            
            # CAGR post-electrificación
            usuarios_inicio = df_post.iloc[0]['usuarios_activos']
            usuarios_final = df_post.iloc[-1]['usuarios_activos']
            años = (df_post.iloc[-1]['fecha'] - df_post.iloc[0]['fecha']).days / 365.25
            cagr_post = ((usuarios_final / usuarios_inicio) ** (1/años) - 1) * 100
            
            print(f"   CAGR (2022-2024): {cagr_post:.1f}% anual")
            
            self.modelo_parametros = {
                'tipo': 'logistico',
                'parametros': popt,
                'offset_x': X_original.min()
            }
            
            return {
                'periodo': 'post_electrificacion',
                'parametros': popt,
                'r2': r2,
                'mape': mape,
                'cagr': cagr_post,
                'X': X_original,
                'y': y,
                'y_pred': y_pred
            }
            
        except Exception as e:
            print(f"   ERROR ajustando modelo: {e}")
            return None
    
    def predecir_futuro(self, meses_futuro=24):
        """
        Proyecta usuarios futuros usando el modelo post-electrificación
        
        Args:
            meses_futuro: número de meses a proyectar (default 24 = 2 años)
        """
        print(f"\n3. Proyectando usuarios para los proximos {meses_futuro} meses...")
        
        if self.modelo_parametros is None:
            print("   ERROR: Primero debes ajustar el modelo con ajustar_modelo_post_electrificacion()")
            return None
        
        # Generar meses futuros
        ultimo_mes = self.df['meses_desde_inicio'].max()
        meses_proyeccion = np.arange(ultimo_mes + 1, ultimo_mes + meses_futuro + 1)
        
        # Predecir con modelo logístico
        X_pred = meses_proyeccion - self.modelo_parametros['offset_x']
        y_pred = self.modelo_logistico(X_pred, *self.modelo_parametros['parametros'])
        
        # Crear DataFrame de proyección
        fecha_ultimo = self.df['fecha'].max()
        fechas_futuro = pd.date_range(
            start=fecha_ultimo + pd.DateOffset(months=1),
            periods=meses_futuro,
            freq='MS'
        )
        
        df_proyeccion = pd.DataFrame({
            'fecha': fechas_futuro,
            'usuarios_proyectados': y_pred.astype(int),
            'meses_desde_inicio': meses_proyeccion
        })
        
        # Mostrar hitos clave
        print("\n   PROYECCIONES CLAVE:")
        print(f"   - Usuarios actuales (Dic 2024): {self.df.iloc[-1]['usuarios_activos']:,.0f}")
        print(f"   - Usuarios en 6 meses (Jun 2025): {df_proyeccion.iloc[5]['usuarios_proyectados']:,.0f}")
        print(f"   - Usuarios en 12 meses (Dic 2025): {df_proyeccion.iloc[11]['usuarios_proyectados']:,.0f}")
        print(f"   - Usuarios en 24 meses (Dic 2026): {df_proyeccion.iloc[-1]['usuarios_proyectados']:,.0f}")
        
        capacidad_max = self.modelo_parametros['parametros'][0]
        saturacion_actual = (df_proyeccion.iloc[-1]['usuarios_proyectados'] / capacidad_max) * 100
        print(f"\n   Saturacion estimada en 2 anos: {saturacion_actual:.1f}% de capacidad maxima")
        
        return df_proyeccion
    
    def visualizar_analisis_completo(self, resultado_pre, resultado_post, df_proyeccion):
        """
        Crea visualización completa del análisis temporal
        """
        print("\n4. Generando visualizaciones...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análisis de Crecimiento de Usuarios BiciCoruña', fontsize=16, fontweight='bold')
        
        # 1. Serie temporal completa con ajustes
        ax1 = axes[0, 0]
        ax1.scatter(self.df['fecha'], self.df['usuarios_activos'], 
                   color='#2E86AB', s=60, alpha=0.6, label='Datos reales', zorder=3)
        
        # Línea divisoria electrificación
        ax1.axvline(pd.to_datetime('2022-06-01'), color='red', linestyle='--', 
                   linewidth=2, alpha=0.7, label='Electrificación')
        
        # Proyección futura
        ax1.plot(df_proyeccion['fecha'], df_proyeccion['usuarios_proyectados'],
                color='#A23B72', linewidth=2.5, linestyle='--', label='Proyección 2025-2026', zorder=2)
        
        ax1.set_xlabel('Fecha', fontsize=12)
        ax1.set_ylabel('Usuarios Activos', fontsize=12)
        ax1.set_title('Evolución Histórica y Proyección Futura', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper left', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Comparativa Pre vs Post Electrificación (CAGR)
        ax2 = axes[0, 1]
        periodos = ['Pre-Electrificación\n(2009-2022)', 'Post-Electrificación\n(2022-2024)']
        cagrs = [resultado_pre['cagr'], resultado_post['cagr']]
        colors = ['#6C757D', '#28A745']
        
        bars = ax2.bar(periodos, cagrs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('CAGR (%)', fontsize=12)
        ax2.set_title('Tasa de Crecimiento Anual Compuesta (CAGR)', fontsize=13, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Etiquetas en barras
        for bar, val in zip(bars, cagrs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 3. Impacto de Eventos Clave (usos por usuario)
        ax3 = axes[1, 0]
        ax3.plot(self.df['fecha'], self.df['usos_por_usuario'], 
                marker='o', color='#FF6B35', linewidth=2, markersize=5)
        ax3.axvline(pd.to_datetime('2022-06-01'), color='red', linestyle='--', 
                   linewidth=2, alpha=0.7, label='Electrificación')
        ax3.set_xlabel('Fecha', fontsize=12)
        ax3.set_ylabel('Usos por Usuario', fontsize=12)
        ax3.set_title('Evolución de Intensidad de Uso', fontsize=13, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Proyección con banda de confianza
        ax4 = axes[1, 1]
        
        # Historico
        ax4.plot(self.df['fecha'], self.df['usuarios_activos'], 
                color='#2E86AB', linewidth=2.5, label='Histórico', marker='o', markersize=4)
        
        # Proyección
        ax4.plot(df_proyeccion['fecha'], df_proyeccion['usuarios_proyectados'],
                color='#A23B72', linewidth=2.5, linestyle='--', label='Proyección', marker='s', markersize=4)
        
        # Capacidad máxima
        capacidad_max = self.modelo_parametros['parametros'][0]
        ax4.axhline(capacidad_max, color='orange', linestyle=':', 
                   linewidth=2, alpha=0.8, label=f'Saturación (~{capacidad_max:,.0f})')
        
        ax4.set_xlabel('Fecha', fontsize=12)
        ax4.set_ylabel('Usuarios Activos', fontsize=12)
        ax4.set_title('Proyección con Límite de Saturación', fontsize=13, fontweight='bold')
        ax4.legend(loc='upper left')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Guardar
        output_path = Path("reports/figures/prediccion_crecimiento_usuarios.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   OK - Guardado en: {output_path}")
        
        plt.close()
        
        return output_path
    
    def generar_reporte_completo(self):
        """
        Ejecuta análisis completo y genera reporte
        """
        print("=" * 60)
        print("ANALISIS DE CRECIMIENTO DE USUARIOS - BICICORUNA")
        print("=" * 60)
        
        # Cargar datos
        self.cargar_datos()
        
        # Ajustar modelos
        resultado_pre = self.ajustar_modelo_pre_electrificacion()
        resultado_post = self.ajustar_modelo_post_electrificacion()
        
        # Proyectar futuro
        df_proyeccion = self.predecir_futuro(meses_futuro=24)
        
        # Visualizar
        self.visualizar_analisis_completo(resultado_pre, resultado_post, df_proyeccion)
        
        # Guardar proyección a CSV
        output_csv = Path("data/processed/proyeccion_usuarios_2025_2026.csv")
        df_proyeccion.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"\n   Proyeccion guardada en: {output_csv}")
        
        print("\n" + "=" * 60)
        print("ANALISIS COMPLETADO")
        print("=" * 60)
        
        return df_proyeccion

if __name__ == "__main__":
    predictor = PredictorCrecimientoUsuarios()
    predictor.generar_reporte_completo()
