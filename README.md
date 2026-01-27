# 🚲 BiciCoruña Data Collector

Sistema de recolección de datos del servicio de bicicletas públicas de A Coruña para análisis de movilidad urbana.

## Características

- ✅ Recolección automática cada 5 minutos
- ✅ Integración con clima (OpenMeteo)
- ✅ Detección de patrones temporales (hora punta, festivos)
- ✅ Health Score del sistema en tiempo real
- ✅ Sin necesidad de tener el ordenador encendido (Railway)
- ✅ Datos almacenados en Google Sheets

## Despliegue

Ver [SETUP.md](./SETUP.md) para instrucciones detalladas.

## Datos recogidos

| Campo | Descripción |
|-------|-------------|
| `bikes_available` | Bicis disponibles en la estación |
| `docks_available` | Huecos libres |
| `delta_bikes` | Cambio desde última lectura |
| `temperature` | Temperatura actual |
| `is_raining` | ¿Está lloviendo? |
| `is_rush_hour` | ¿Es hora punta? |
| `health_score` | Salud del sistema (0-100) |

## Autor

Nathan - Proyecto de análisis de movilidad urbana
