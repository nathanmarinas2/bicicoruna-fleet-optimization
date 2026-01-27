# 🚲 BiciCoruña Cloud Collector - Guía de Configuración

## PASO 1: Crear Google Sheet

1. Ve a [Google Sheets](https://sheets.google.com)
2. Crea una nueva hoja llamada "BiciCoruña Data"
3. Copia el ID de la hoja (está en la URL: `https://docs.google.com/spreadsheets/d/ESTE_ES_EL_ID/edit`)

---

## PASO 2: Crear el Webhook (Google Apps Script)

1. En tu Google Sheet, ve a **Extensiones → Apps Script**
2. Borra todo el código y pega esto:

```javascript
function doPost(e) {
  try {
    const data = JSON.parse(e.postData.contents);
    const sheet = SpreadsheetApp.getActiveSpreadsheet();
    
    // Hoja de métricas del sistema
    let systemSheet = sheet.getSheetByName('Sistema');
    if (!systemSheet) {
      systemSheet = sheet.insertSheet('Sistema');
      systemSheet.appendRow([
        'Timestamp', 'Total Estaciones', 'Total Bicis', 'Capacidad Total',
        'Estaciones Vacías', 'Estaciones Llenas', 'Health Score',
        'Temperatura', 'Lloviendo'
      ]);
    }
    
    // Añadir métricas del sistema
    if (data.system) {
      systemSheet.appendRow([
        data.timestamp,
        data.system.total_stations,
        data.system.total_bikes,
        data.system.total_capacity,
        data.system.empty_stations,
        data.system.full_stations,
        data.system.health_score,
        data.system.temperature,
        data.system.is_raining
      ]);
    }
    
    // Hoja de datos por estación
    let stationsSheet = sheet.getSheetByName('Estaciones');
    if (!stationsSheet) {
      stationsSheet = sheet.insertSheet('Estaciones');
      stationsSheet.appendRow([
        'Timestamp', 'ID', 'Nombre', 'Lat', 'Lon', 'Capacidad',
        'Bicis', 'Docks', 'Ocupación', 'Vacía', 'Llena', 'Delta',
        'Hora', 'Día', 'Finde', 'Festivo', 'Punta', 'Turismo',
        'Temp', 'Lluvia', 'Viento', 'Clima'
      ]);
    }
    
    // Añadir datos de estaciones
    if (data.stations && data.stations.length > 0) {
      data.stations.forEach(s => {
        stationsSheet.appendRow([
          s.timestamp, s.station_id, s.station_name, s.lat, s.lon, s.capacity,
          s.bikes_available, s.docks_available, s.occupancy_rate,
          s.is_empty, s.is_full, s.delta_bikes,
          s.hour, s.day_of_week, s.is_weekend, s.is_holiday,
          s.is_rush_hour, s.is_tourist_season,
          s.temperature, s.is_raining, s.wind_speed, s.weather_type
        ]);
      });
    }
    
    return ContentService.createTextOutput(JSON.stringify({success: true}))
      .setMimeType(ContentService.MimeType.JSON);
      
  } catch (error) {
    return ContentService.createTextOutput(JSON.stringify({error: error.message}))
      .setMimeType(ContentService.MimeType.JSON);
  }
}
```

3. Guarda el proyecto (Ctrl+S)
4. **Implementar → Nueva implementación**
5. Tipo: **Aplicación web**
6. Ejecutar como: **Yo**
7. Quién tiene acceso: **Cualquier persona**
8. Haz clic en **Implementar**
9. **COPIA LA URL DEL WEBHOOK** (la necesitarás en el paso 3)

---

## PASO 3: Configurar Railway

1. Sube esta carpeta a un repositorio de GitHub
2. Ve a [Railway](https://railway.app) y crea una cuenta (gratis con GitHub)
3. **New Project → Deploy from GitHub repo**
4. Selecciona tu repositorio
5. Ve a **Variables** y añade:
   - `GOOGLE_SHEET_WEBHOOK` = (la URL que copiaste en el paso 2)
6. Railway desplegará automáticamente

---

## PASO 4: Verificar

1. Espera 5 minutos
2. Ve a tu Google Sheet
3. Deberías ver datos en las pestañas "Sistema" y "Estaciones"

---

## 📊 Qué datos se recogen

### Hoja "Sistema" (1 fila cada 5 min):
- Health Score del sistema (0-100)
- Total de bicis disponibles
- Estaciones vacías/llenas
- Clima

### Hoja "Estaciones" (~30 filas cada 5 min):
- Estado de cada estación
- Cambios (delta) desde la última lectura
- Contexto: hora punta, festivo, clima...

---

## ❓ Problemas comunes

**No llegan datos:**
- Verifica que el webhook esté correctamente copiado
- Comprueba los logs en Railway

**Error de permisos en Apps Script:**
- Asegúrate de que "Quién tiene acceso" sea "Cualquier persona"

**Railway se detiene:**
- En el plan gratuito, Railway se pausa tras inactividad. Para este proyecto, no debería afectar porque el script se mantiene activo.
