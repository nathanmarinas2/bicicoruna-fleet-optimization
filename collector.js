/*
 * =============================================================================
 * 🚲 BICICORUÑA CLOUD COLLECTOR
 * Versión para Railway + Google Sheets
 * =============================================================================
 * 
 * Este script corre en Railway (gratis) y envía datos a Google Sheets.
 * No necesitas tener tu ordenador encendido.
 * 
 * CONFIGURACIÓN:
 * 1. Crear Google Sheet
 * 2. Crear Google Apps Script (ver SETUP.md)
 * 3. Pegar la URL del webhook en GOOGLE_SHEET_WEBHOOK
 * 4. Subir a GitHub y conectar con Railway
 * =============================================================================
 */

// ======================== CONFIGURACIÓN ========================
// IMPORTANTE: Reemplaza esta URL con tu webhook de Google Sheets
const GOOGLE_SHEET_WEBHOOK = process.env.GOOGLE_SHEET_WEBHOOK || 'TU_WEBHOOK_AQUI';

const GBFS_STATION_INFO = 'https://acoruna.publicbikesystem.net/customer/gbfs/v2/es/station_information';
const GBFS_STATION_STATUS = 'https://acoruna.publicbikesystem.net/customer/gbfs/v2/es/station_status';
const WEATHER_API = 'https://api.open-meteo.com/v1/forecast?latitude=43.3623&longitude=-8.4115&current=temperature_2m,precipitation,weather_code,wind_speed_10m';
const POLLING_INTERVAL_MS = 5 * 60 * 1000; // 5 minutos

// ======================== ESTADO ========================
let stationsInfo = {};
let previousStatus = {};
let currentWeather = { temp: null, rain: 0, wind: 0, code: 0 };
let cycleCount = 0;

// ======================== CALENDARIO FESTIVOS ESPAÑA 2026 ========================
const HOLIDAYS_2026 = [
    '2026-01-01', '2026-01-06', '2026-04-02', '2026-04-03',
    '2026-05-01', '2026-05-17', '2026-07-25', '2026-08-15',
    '2026-10-12', '2026-11-01', '2026-12-06', '2026-12-08', '2026-12-25'
];

// ======================== FUNCIONES AUXILIARES ========================

function isHoliday(date) {
    const dateStr = date.toISOString().split('T')[0];
    return HOLIDAYS_2026.includes(dateStr) ? 1 : 0;
}

function isRushHour(hour) {
    return ((hour >= 7 && hour <= 9) || (hour >= 14 && hour <= 16) || (hour >= 18 && hour <= 20)) ? 1 : 0;
}

function isTouristSeason(month) {
    return (month >= 6 && month <= 9) ? 1 : 0;
}

function isServiceHours() {
    const hour = new Date().getHours();
    return hour >= 7 || hour < 1;
}

function getWeatherDescription(code) {
    if (code === 0) return 'clear';
    if (code <= 3) return 'cloudy';
    if (code >= 51 && code <= 67) return 'rain';
    if (code >= 71 && code <= 77) return 'snow';
    if (code >= 80 && code <= 82) return 'showers';
    if (code >= 95) return 'storm';
    return 'other';
}

// ======================== FUNCIONES DE DATOS ========================

async function fetchStationInfo() {
    try {
        const response = await fetch(GBFS_STATION_INFO);
        if (!response.ok) throw new Error('Error fetching station info');
        const data = await response.json();

        const stations = {};
        data.data.stations.forEach(s => {
            stations[s.station_id] = {
                id: s.station_id,
                name: s.name,
                lat: s.lat,
                lon: s.lon,
                capacity: s.capacity || 0
            };
        });

        console.log(`📍 Cargadas ${Object.keys(stations).length} estaciones`);
        return stations;
    } catch (e) {
        console.error('❌ Error cargando estaciones:', e.message);
        return null;
    }
}

async function fetchWeather() {
    try {
        const response = await fetch(WEATHER_API);
        if (!response.ok) return;
        const data = await response.json();

        if (data.current) {
            currentWeather = {
                temp: data.current.temperature_2m || 0,
                rain: data.current.precipitation || 0,
                wind: data.current.wind_speed_10m || 0,
                code: data.current.weather_code || 0
            };
        }
    } catch (e) {
        // Silenciar errores de clima
    }
}

async function sendToGoogleSheets(data) {
    if (GOOGLE_SHEET_WEBHOOK === 'TU_WEBHOOK_AQUI') {
        console.log('⚠️ Webhook no configurado. Datos no enviados.');
        return false;
    }

    try {
        const response = await fetch(GOOGLE_SHEET_WEBHOOK, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (response.ok) {
            return true;
        } else {
            console.error('❌ Error enviando a Google Sheets:', response.status);
            return false;
        }
    } catch (e) {
        console.error('❌ Error de red:', e.message);
        return false;
    }
}

async function collectAndSend() {
    try {
        await fetchWeather();

        const response = await fetch(GBFS_STATION_STATUS);
        if (!response.ok) throw new Error('Error fetching status');
        const data = await response.json();

        const now = new Date();
        const timestamp = now.toISOString();
        const hour = now.getHours();
        const dayOfWeek = now.getDay();
        const month = now.getMonth() + 1;
        const isWeekend = dayOfWeek === 0 || dayOfWeek === 6;
        const holiday = isHoliday(now);
        const rushHour = isRushHour(hour);
        const touristSeason = isTouristSeason(month);
        const isRaining = currentWeather.rain > 0 ? 1 : 0;
        const weatherType = getWeatherDescription(currentWeather.code);

        // Preparar datos para enviar
        const records = [];
        let systemStats = {
            totalBikes: 0,
            totalDocks: 0,
            totalCapacity: 0,
            emptyCount: 0,
            fullCount: 0,
            stationCount: 0
        };

        data.data.stations.forEach(status => {
            const info = stationsInfo[status.station_id];
            if (!info) return;

            const bikes = status.num_bikes_available || 0;
            const docks = status.num_docks_available || 0;
            const capacity = info.capacity || (bikes + docks);
            const occupancyRate = capacity > 0 ? (bikes / capacity) : 0;
            const isEmpty = bikes === 0 ? 1 : 0;
            const isFull = docks === 0 ? 1 : 0;

            const prevBikes = previousStatus[status.station_id] || bikes;
            const deltaBikes = bikes - prevBikes;
            previousStatus[status.station_id] = bikes;

            systemStats.totalBikes += bikes;
            systemStats.totalDocks += docks;
            systemStats.totalCapacity += capacity;
            systemStats.stationCount++;
            if (isEmpty) systemStats.emptyCount++;
            if (isFull) systemStats.fullCount++;

            records.push({
                timestamp,
                station_id: status.station_id,
                station_name: info.name,
                lat: info.lat,
                lon: info.lon,
                capacity,
                bikes_available: bikes,
                docks_available: docks,
                occupancy_rate: occupancyRate.toFixed(3),
                is_empty: isEmpty,
                is_full: isFull,
                delta_bikes: deltaBikes,
                hour,
                day_of_week: dayOfWeek,
                is_weekend: isWeekend ? 1 : 0,
                is_holiday: holiday,
                is_rush_hour: rushHour,
                is_tourist_season: touristSeason,
                temperature: currentWeather.temp,
                is_raining: isRaining,
                wind_speed: currentWeather.wind,
                weather_type: weatherType
            });
        });

        // Calcular health score
        const emptyRate = (systemStats.emptyCount / systemStats.stationCount * 100);
        const fullRate = (systemStats.fullCount / systemStats.stationCount * 100);
        const healthScore = Math.max(0, 100 - (emptyRate * 1.5) - (fullRate * 1.5));

        // Enviar a Google Sheets
        const payload = {
            type: 'bici_data',
            cycle: cycleCount,
            timestamp,
            system: {
                total_stations: systemStats.stationCount,
                total_bikes: systemStats.totalBikes,
                total_capacity: systemStats.totalCapacity,
                empty_stations: systemStats.emptyCount,
                full_stations: systemStats.fullCount,
                health_score: healthScore.toFixed(0),
                temperature: currentWeather.temp,
                is_raining: isRaining
            },
            stations: records
        };

        const sent = await sendToGoogleSheets(payload);

        // Log
        const weatherEmoji = isRaining ? '🌧️' : (currentWeather.temp > 20 ? '☀️' : '⛅');
        console.log(`\n${weatherEmoji} Ciclo ${cycleCount} | ${now.toLocaleTimeString()} | ${currentWeather.temp}°C`);
        console.log(`   📊 Bicis: ${systemStats.totalBikes}/${systemStats.totalCapacity} | Health: ${healthScore.toFixed(0)}/100`);
        console.log(`   📤 Google Sheets: ${sent ? '✅ Enviado' : '❌ Fallo'}`);

        cycleCount++;

    } catch (e) {
        console.error('❌ Error:', e.message);
    }
}

// ======================== MAIN ========================

async function main() {
    console.log('='.repeat(60));
    console.log('🚲 BICICORUÑA CLOUD COLLECTOR');
    console.log('   Ejecutándose en Railway → Google Sheets');
    console.log('='.repeat(60));
    console.log(`⏰ Horario: 07:00 - 01:00`);
    console.log(`📡 Intervalo: ${POLLING_INTERVAL_MS / 1000 / 60} minutos`);
    console.log(`📤 Destino: Google Sheets`);
    console.log('='.repeat(60));

    // Cargar estaciones
    stationsInfo = await fetchStationInfo();
    if (!stationsInfo) {
        console.error('No se pudieron cargar las estaciones. Reintentando en 1 min...');
        setTimeout(main, 60000);
        return;
    }

    // Primera captura
    if (isServiceHours()) {
        await collectAndSend();
    }

    // Bucle principal
    setInterval(async () => {
        if (isServiceHours()) {
            await collectAndSend();
        } else {
            console.log(`\n😴 Fuera de horario (01:00-07:00). Esperando...`);
        }
    }, POLLING_INTERVAL_MS);
}

main().catch(console.error);
