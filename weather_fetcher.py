"""
weather_fetcher.py
------------------
CheckWX integration for aviation-grade weather impact.
"""

from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass

import requests


CHECKWX_BASE_URL = "https://api.checkwx.com"
WEATHER_CACHE_TTL = 900
_MAX_WEATHER_CACHE = 300
_WEATHER_CACHE: dict[str, tuple[float, "WeatherImpact"]] = {}
_CIRCUIT_OPEN_UNTIL: float = 0.0  # timestamp; skip all calls until this time
_CONSECUTIVE_ERRORS: int = 0
_CIRCUIT_ERROR_THRESHOLD: int = 3  # open circuit after this many consecutive failures
_CIRCUIT_COOLDOWN: float = 60.0    # seconds to wait before retrying
_CIRCUIT_LOCK = threading.Lock()   # guards circuit breaker globals across threads


@dataclass(frozen=True)
class WeatherImpact:
    station_icao: str
    flight_category: str
    severity: str
    summary: str
    raw_text: str
    visibility_miles: float | None
    wind_kts: int | None
    gust_kts: int | None
    conditions_text: str
    penalty_minutes: int


def _classify_weather(decoded: dict) -> WeatherImpact:
    icao = str(decoded.get("icao", "N/A"))
    raw_text = str(decoded.get("raw_text", ""))
    flight_category = str(decoded.get("flight_category", "VFR") or "VFR")
    visibility = decoded.get("visibility", {}) or {}
    wind = decoded.get("wind", {}) or {}
    wind_speed = ((wind.get("speed") or {}) or {}).get("kts")
    gust_speed = ((wind.get("gust") or {}) or {}).get("kts")
    conditions = decoded.get("conditions") or []
    condition_texts = ", ".join(item.get("text", "") for item in conditions if item.get("text"))
    visibility_miles = visibility.get("miles")

    severity = "Low"
    penalty_minutes = 0
    drivers: list[str] = []

    category_rank = {"LIFR": 3, "IFR": 2, "MVFR": 1, "VFR": 0}
    if category_rank.get(flight_category, 0) >= 3:
        severity = "Severe"
        penalty_minutes += 14
        drivers.append("very low flight category")
    elif category_rank.get(flight_category, 0) == 2:
        severity = "Severe"
        penalty_minutes += 10
        drivers.append("IFR conditions")
    elif category_rank.get(flight_category, 0) == 1:
        severity = "Moderate"
        penalty_minutes += 5
        drivers.append("marginal weather")

    if visibility_miles is not None and visibility_miles < 2:
        severity = "Severe"
        penalty_minutes += 8
        drivers.append("poor visibility")
    elif visibility_miles is not None and visibility_miles < 5:
        severity = "Moderate" if severity == "Low" else severity
        penalty_minutes += 4
        drivers.append("reduced visibility")

    if gust_speed is not None and gust_speed >= 30:
        severity = "Severe"
        penalty_minutes += 7
        drivers.append("strong gusts")
    elif wind_speed is not None and wind_speed >= 20:
        severity = "Moderate" if severity == "Low" else severity
        penalty_minutes += 4
        drivers.append("strong winds")

    lowered = condition_texts.lower()
    if any(word in lowered for word in ["thunderstorm", "fog", "heavy rain", "ice", "hail"]):
        severity = "Severe"
        penalty_minutes += 10
        drivers.append("active weather")
    elif any(word in lowered for word in ["rain", "mist", "shower"]):
        severity = "Moderate" if severity == "Low" else severity
        penalty_minutes += 4
        drivers.append("precipitation")

    summary = ", ".join(drivers) if drivers else "stable airport weather"
    return WeatherImpact(
        station_icao=icao,
        flight_category=flight_category,
        severity=severity,
        summary=summary,
        raw_text=raw_text,
        visibility_miles=float(visibility_miles) if visibility_miles is not None else None,
        wind_kts=int(wind_speed) if wind_speed is not None else None,
        gust_kts=int(gust_speed) if gust_speed is not None else None,
        conditions_text=condition_texts or "No significant conditions",
        penalty_minutes=min(penalty_minutes, 20),
    )


def get_nearest_airport_weather(lat: float, lon: float, iata_hint: str | None = None) -> WeatherImpact | None:
    global _CIRCUIT_OPEN_UNTIL, _CONSECUTIVE_ERRORS
    api_key = os.getenv("CHECKWX_API_KEY")
    if not api_key:
        return None

    # Circuit breaker: skip when API was recently unreachable (thread-safe read)
    with _CIRCUIT_LOCK:
        if time.time() < _CIRCUIT_OPEN_UNTIL:
            return None

    cache_key = iata_hint if iata_hint else f"{round(float(lat), 2)}:{round(float(lon), 2)}"
    cached = _WEATHER_CACHE.get(cache_key)
    if cached and time.time() - cached[0] < WEATHER_CACHE_TTL:
        return cached[1]

    url = f"{CHECKWX_BASE_URL}/metar/lat/{lat}/lon/{lon}/radius/25/decoded"
    try:
        response = requests.get(
            url,
            headers={"X-API-Key": api_key},
            timeout=(1.5, 2),
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") or []
        if not data:
            return None
        impact = _classify_weather(data[0])
        _WEATHER_CACHE[cache_key] = (time.time(), impact)
        # Cap cache size
        if len(_WEATHER_CACHE) > _MAX_WEATHER_CACHE:
            oldest_key = min(_WEATHER_CACHE, key=lambda k: _WEATHER_CACHE[k][0])
            _WEATHER_CACHE.pop(oldest_key, None)
        with _CIRCUIT_LOCK:
            _CONSECUTIVE_ERRORS = 0  # reset on success
        return impact
    except Exception:
        with _CIRCUIT_LOCK:
            _CONSECUTIVE_ERRORS += 1
            if _CONSECUTIVE_ERRORS >= _CIRCUIT_ERROR_THRESHOLD:
                _CIRCUIT_OPEN_UNTIL = time.time() + _CIRCUIT_COOLDOWN
        return None
