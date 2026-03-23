import json
import time
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/weather", tags=["weather"])

SEOUL_LAT = 37.5665
SEOUL_LON = 126.9780
CACHE_TTL_SECONDS = 3600
_weather_cache = {"expires_at": 0.0, "data": None}


def _fetch_seoul_weather():
    query = urlencode(
        {
            "latitude": SEOUL_LAT,
            "longitude": SEOUL_LON,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code,is_day",
            "daily": "sunset",
            "forecast_days": 1,
            "timezone": "Asia/Seoul",
        }
    )
    url = f"https://api.open-meteo.com/v1/forecast?{query}"
    with urlopen(url, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))

    current = payload.get("current") or {}
    daily = payload.get("daily") or {}
    sunset_raw = (daily.get("sunset") or [None])[0]
    sunset_time = None
    if sunset_raw:
        try:
            sunset_time = datetime.fromisoformat(sunset_raw).strftime("%H:%M")
        except ValueError:
            sunset_time = sunset_raw[-5:]

    return {
        "city": "seoul",
        "temperature_c": current.get("temperature_2m"),
        "humidity_pct": current.get("relative_humidity_2m"),
        "wind_speed_ms": current.get("wind_speed_10m"),
        "weather_code": current.get("weather_code"),
        "is_day": bool(current.get("is_day", 1)),
        "sunset_time": sunset_time,
        "observed_at": current.get("time"),
        "provider": "open-meteo",
    }


@router.get("/seoul")
def get_seoul_weather(force_refresh: bool = False):
    now = time.time()
    cached = _weather_cache["data"]
    if cached and not force_refresh and _weather_cache["expires_at"] > now:
        return cached

    try:
        data = _fetch_seoul_weather()
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        if cached:
            stale = dict(cached)
            stale["stale"] = True
            return stale
        raise HTTPException(status_code=502, detail=f"Weather data fetch failed: {exc}")

    data["fetched_at"] = datetime.now().isoformat(timespec="seconds")
    _weather_cache["data"] = data
    _weather_cache["expires_at"] = now + CACHE_TTL_SECONDS
    return data
