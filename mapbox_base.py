"""
mapbox_base.py
--------------
Minimal, reliable Mapbox GL JS base map for the aviation dashboard.

This module is intentionally narrow in scope:
  - satellite basemap
  - India-focused 2D operations view
  - airport markers with labels
  - aircraft markers using existing SVG silhouettes
  - selected-flight highlight
  - click popups with core flight details

It is designed to be validated in isolation before wiring it into the main app.
"""

from __future__ import annotations

import json
import math as _math
import time as _time

import pandas as pd

from aircraft_icons import _get_svg, get_family
from analytics import compute_fuel_estimate as _compute_fuel_estimate
from data_fetcher import REGION_MAP_SETTINGS, get_airline_name, get_airports_for_region, get_global_airport_lookup
from tracker import FlightTracker as _FlightTracker

_GLOBAL_AIRPORTS = get_global_airport_lookup()
from utils import safe_text as _safe_text, normalize_flight_query as _normalize_flight_query, is_selected_flight as _is_selected_flight


def _haversine_km_py(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Pure-Python haversine for fuel route distance."""
    r = 6371.0
    d_lat = _math.radians(lat2 - lat1)
    d_lon = _math.radians(lon2 - lon1)
    a = (_math.sin(d_lat / 2) ** 2
         + _math.cos(_math.radians(lat1)) * _math.cos(_math.radians(lat2)) * _math.sin(d_lon / 2) ** 2)
    return r * 2 * _math.atan2(_math.sqrt(a), _math.sqrt(1 - a))


INDIA_BOUNDS = [[67.0, 6.0], [98.5, 37.5]]
PRIMARY_HUBS = {"DEL", "BOM", "BLR", "HYD", "MAA", "CCU", "AMD", "COK"}
MODULES = [
    ("live_ops", "Live Ops"),
    ("delay_prediction", "Delay Prediction"),
    ("airport_traffic", "Airport Traffic"),
    ("route_intelligence", "Route Intelligence"),
    ("passenger_view", "Passenger View"),
    ("flight_board", "Flight Board"),
    ("alerts", "Alerts"),
]

# ── Delay signal lookup tables (mirrors analytics.py values) ──────────────────
_AIRLINE_OTP: dict[str, float] = {
    "6E": 0.87, "AI": 0.64, "UK": 0.78, "SG": 0.72, "G8": 0.68,
    "QP": 0.80, "I5": 0.74, "S5": 0.70, "9I": 0.66,
    "EK": 0.85, "EY": 0.82, "QR": 0.84, "SQ": 0.88,
    "BA": 0.76, "LH": 0.78, "TK": 0.77, "FZ": 0.80,
    "WY": 0.79, "TG": 0.75, "CX": 0.82, "VN": 0.73,
    "IR": 0.65, "KL": 0.80, "AF": 0.77, "ET": 0.72,
    "MS": 0.68, "RJ": 0.70, "IX": 0.71, "S2": 0.69,
}
_PEAK_AIRPORTS = {"DEL", "BOM", "BLR", "HYD", "MAA", "CCU", "AMD", "PNQ", "GOI", "COK"}
_MONSOON_AIRPORTS = {"BOM", "GOI", "COK", "CCU", "MAA", "GAU", "IXB", "PNQ", "IXZ", "CNN"}
_SLOT_CAPACITY: dict[str, int] = {
    "DEL": 78, "BOM": 46, "BLR": 40, "MAA": 36,
    "HYD": 32, "CCU": 28, "AMD": 24, "PNQ": 20,
    "COK": 22, "GOI": 18, "GAU": 16,
}
# Widebody/large types = longer taxi + gate turn
_WIDE_BODY_TYPES = {"B744", "B748", "B772", "B773", "B77W", "B77L", "B788", "B789",
                    "B78X", "A332", "A333", "A342", "A343", "A345", "A346",
                    "A358", "A359", "A35K", "A380", "A388", "IL96", "AN24"}
_HOTSPOT_ROUTES: dict[frozenset, int] = {
    frozenset({"DEL", "BOM"}): 8, frozenset({"DEL", "BLR"}): 6,
    frozenset({"BOM", "BLR"}): 5, frozenset({"DEL", "HYD"}): 5,
    frozenset({"DEL", "MAA"}): 5, frozenset({"DEL", "CCU"}): 5,
    frozenset({"BOM", "GOI"}): 4, frozenset({"BLR", "MAA"}): 4,
    frozenset({"DEL", "AMD"}): 4,
}


def _fmt_number(value: object, fallback: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return fallback
        return float(value)
    except Exception:
        return fallback


def _build_flights_json(df: pd.DataFrame, selected_flight: str, schedule_lookup: dict | None = None, *, airports: list | None = None) -> str:
    flights: list[dict] = []
    normalized_selected = _normalize_flight_query(selected_flight)
    _sched = schedule_lookup or {}
    _airports_lookup = {a["iata"]: a for a in airports} if airports else {}
    _global_ap = _GLOBAL_AIRPORTS

    for _row_t in df.itertuples(index=False):
        row = pd.Series(_row_t._asdict())
        lat = row.get("latitude")
        lng = row.get("longitude")
        if pd.isna(lat) or pd.isna(lng):
            continue

        flight_code = _safe_text(row.get("flight_iata"))
        airline_code = _safe_text(row.get("airline_iata"))
        airline_name = get_airline_name(airline_code) if airline_code != "N/A" else "Unknown airline"
        aircraft_icao = _safe_text(row.get("aircraft_icao"))
        altitude_ft = round(_fmt_number(row.get("altitude_ft")))
        speed_kts = round(_fmt_number(row.get("speed_kts")), 1)
        heading = round(_fmt_number(row.get("heading")), 1)
        is_selected = _is_selected_flight(flight_code, normalized_selected)
        color = "#ff6156" if is_selected else "#ffd24a"
        _arr_iata = _safe_text(row.get("arr_iata"))

        # ── Backend fuel estimate (accurate: routing factor + climb/descent + taxi) ──
        _fuel_burn: float | None = None
        _fuel_co2: float | None = None
        _fuel_co2_pax: float | None = None
        _fuel_eff: str | None = None
        if _airports_lookup:
            _dep_iata = _safe_text(row.get("dep_iata"))
            _dep_ap = _airports_lookup.get(_dep_iata)
            _arr_ap = _airports_lookup.get(_arr_iata)
            if _dep_ap and _arr_ap:
                try:
                    _dist_km = _haversine_km_py(
                        _dep_ap["lat"], _dep_ap["lng"], _arr_ap["lat"], _arr_ap["lng"]
                    )
                    _fe = _compute_fuel_estimate(aircraft_icao, _dist_km)
                    _fuel_burn  = _fe.fuel_burn_kg
                    _fuel_co2   = _fe.co2_kg
                    _fuel_co2_pax = _fe.co2_per_pax_kg
                    _fuel_eff   = _fe.efficiency_label
                except Exception:
                    pass

        flights.append(
            {
                "flight": flight_code,
                "airline": airline_name,
                "airline_code": airline_code,
                "dep": _safe_text(row.get("dep_iata")),
                "arr": _safe_text(row.get("arr_iata")),
                "aircraft": aircraft_icao,
                "status": _safe_text(row.get("status"), "en-route").replace("-", " ").title(),
                "registration": _safe_text(row.get("reg_number")),
                "icao24": _safe_text(row.get("hex")),
                "lat": float(lat),
                "lng": float(lng),
                "altitude_ft": altitude_ft,
                "speed_kts": speed_kts,
                "heading": heading,
                "selected": is_selected,
                "color": color,
                "family": get_family(aircraft_icao),
                # ── Destination airport coords (for accurate distance) ─────
                "arr_lat": (_global_ap[_arr_iata]["lat"]
                            if _arr_iata in _global_ap else None),
                "arr_lng": (_global_ap[_arr_iata]["lng"]
                            if _arr_iata in _global_ap else None),
                # ── Scheduled time data (from AirLabs /schedules) ──────────
                "std_ts":     _sched.get(flight_code, {}).get("std_ts"),
                "sta_ts":     _sched.get(flight_code, {}).get("sta_ts"),
                "eta_ts":     _sched.get(flight_code, {}).get("eta_ts"),
                "delayed_min": _sched.get(flight_code, {}).get("delayed_min"),
                # ── Delay prediction signals ──────────────────────────────
                "otp": _AIRLINE_OTP.get(airline_code, -1),
                "is_peak_arr": _arr_iata in _PEAK_AIRPORTS,
                "is_monsoon_arr": _arr_iata in _MONSOON_AIRPORTS,
                "slot_cap": _SLOT_CAPACITY.get(_arr_iata, 0),
                "is_widebody": aircraft_icao.upper() in _WIDE_BODY_TYPES if aircraft_icao != "N/A" else False,
                "hotspot_min": _HOTSPOT_ROUTES.get(frozenset({
                    _safe_text(row.get("dep_iata")), _arr_iata
                }), 0),
                # ── AFRI (Arrival Flow Risk Index) ────────────────────────
                "afri_score": int(row.get("afri_score") or 0),
                "afri_level": _safe_text(row.get("afri_level"), "Normal"),
                "afri_drivers": _safe_text(row.get("afri_drivers"), ""),
                # ── Backend delay prediction ──────────────────────────────
                "pred_delay_min": int(row.get("predicted_delay_min") or 0),
                "pred_delay_risk": _safe_text(row.get("predicted_delay_risk"), "Low"),
                "pred_delay_reason": _safe_text(row.get("predicted_delay_reason"), ""),
                "pred_score": round(_fmt_number(row.get("pred_score")), 1),
                "pred_ml_prob": round(_fmt_number(row.get("pred_ml_prob")), 3),
                "pred_holding": bool(row.get("pred_holding")),
                "pred_go_around": bool(row.get("pred_go_around")),
                "pred_diversion": bool(row.get("pred_diversion")),
                # ── Signal quality ────────────────────────────────────────
                "weather_sev": _safe_text(row.get("weather_severity"), "Low"),
                "stale_min": round(_fmt_number(row.get("stale_minutes")), 1),
                "v_speed": (round(float(row["v_speed"]), 1) if pd.notna(row.get("v_speed")) else None),
                # ── Backend fuel estimate ─────────────────────────────────
                "fuel_burn_kg": _fuel_burn,
                "fuel_co2_kg": _fuel_co2,
                "fuel_co2_pax": _fuel_co2_pax,
                "fuel_eff": _fuel_eff,
            }
        )

    return json.dumps(flights)


def _build_icon_templates_json() -> str:
    families = [
        "b747",
        "a380",
        "a340",
        "b777",
        "a350",
        "a330",
        "b787",
        "b767",
        "b757",
        "a320",
        "b737",
        "e190",
        "crj",
        "atr",
        "concorde",
        "military",
        "default",
    ]
    templates = {family: _get_svg(family, "COLORTOK") for family in families}
    return json.dumps(templates)


def _build_airports_json(region: str) -> str:
    airports = []
    for airport in get_airports_for_region(region):
        airport_copy = dict(airport)
        airport_copy["is_primary_hub"] = airport_copy.get("iata") in PRIMARY_HUBS
        airports.append(airport_copy)
    return json.dumps(airports)


def _build_module_nav_html(active_module: str) -> str:
    buttons = []
    for module_key, label in MODULES:
        active_class = " active" if module_key == active_module else ""
        buttons.append(
            f"<button class='module-btn{active_class}' type='button' data-module-btn data-module='{module_key}'>{label}</button>"
        )
    return "".join(buttons)


def build_flights_json(
    df: "pd.DataFrame",
    selected_flight: str = "",
    schedule_lookup: "dict | None" = None,
    *,
    region: str = "india",
) -> str:
    """Return the flight data as a JSON array string for the live update bridge."""
    airports = list(get_airports_for_region(region))
    return _build_flights_json(df, selected_flight, schedule_lookup or {}, airports=airports)


def generate_mapbox_base_html(
    df: pd.DataFrame,
    mapbox_token: str,
    *,
    selected_flight: str | None = None,
    active_module: str = "live_ops",
    region: str = "india",
    center_lat: float | None = None,
    center_lng: float | None = None,
    zoom_level: int | None = None,
    anomaly_df: "pd.DataFrame | None" = None,
    schedule_lookup: "dict | None" = None,
    airport_metrics: "pd.DataFrame | None" = None,
    schedule_pressure: "dict | None" = None,
    top_airlines: "list | None" = None,
    height: int = 1080,
) -> str:
    """
    Generate a clean Mapbox GL JS base map page.

    Parameters
    ----------
    df:
        Flight dataframe from AirLabs.
    mapbox_token:
        Public Mapbox token (must start with ``pk.``).
    selected_flight:
        Optional flight number to highlight and auto-focus.
    region:
        Region name, currently used for airport markers and default map center.
    """
    settings = REGION_MAP_SETTINGS.get(region, REGION_MAP_SETTINGS["india"])
    lat_center = center_lat if center_lat is not None else settings["center"][0]
    lng_center = center_lng if center_lng is not None else settings["center"][1]
    zoom = zoom_level if zoom_level is not None else settings["zoom"]

    _airports_list = list(get_airports_for_region(region))
    flights_json = _build_flights_json(df, selected_flight or "", schedule_lookup or {}, airports=_airports_list)
    airports_json = json.dumps([{**a, "is_primary_hub": a.get("iata") in PRIMARY_HUBS} for a in _airports_list])
    icon_templates_json = _build_icon_templates_json()
    selected_query = json.dumps(selected_flight or "")
    active_module_json = json.dumps(active_module or "live_ops")
    max_bounds = json.dumps(INDIA_BOUNDS if region == "india" else [[-180, -85], [180, 85]])
    module_nav_html = _build_module_nav_html(active_module)

    # Serialize anomaly data for the Alerts module
    if anomaly_df is not None and not anomaly_df.empty:
        anomalies_json = json.dumps(anomaly_df.to_dict(orient="records"))
    else:
        anomalies_json = "[]"

    # Serialize airport traffic metrics for the Airport Traffic module
    if airport_metrics is not None and not airport_metrics.empty:
        _ap_cols = ["airport_iata", "city", "nearby_count", "inbound_count",
                    "approach_count", "low_altitude_count", "congestion_score", "congestion_level"]
        _ap_rows = airport_metrics[[c for c in _ap_cols if c in airport_metrics.columns]]
        airport_metrics_json = json.dumps(_ap_rows.to_dict(orient="records"))
    else:
        airport_metrics_json = "[]"

    # Serialize schedule pressure dict
    if schedule_pressure:
        schedule_pressure_json = json.dumps(schedule_pressure)
    else:
        schedule_pressure_json = "{}"

    # Serialize top airlines for the Airlines rail card
    top_airlines_json = json.dumps(top_airlines or [])

    # Serialize flight trails (icao24 → positions list) — only flights with ≥2 points
    try:
        _tracker = _FlightTracker()
        _raw_trails = _tracker.get_trails()
        _trails_slim = {
            k: [{"lat": p["lat"], "lng": p["lng"], "alt": p.get("altitude_ft", 0)}
                for p in v.get("positions", [])[-25:]]
            for k, v in _raw_trails.items()
            if len(v.get("positions", [])) >= 2
        }
        trails_json = json.dumps(_trails_slim)
    except Exception:
        trails_json = "{}"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Aviation Mapbox Base</title>
  <link href="https://api.mapbox.com/mapbox-gl-js/v3.4.0/mapbox-gl.css" rel="stylesheet">
  <script src="https://api.mapbox.com/mapbox-gl-js/v3.4.0/mapbox-gl.js" onerror="(function(){{var el=document.getElementById('splash');if(el){{el.style.animation='none';el.classList.add('hidden');el.style.display='none';}}var b=document.getElementById('error-box');if(b){{b.textContent='Failed to load Mapbox library. Check your internet connection.';b.style.display='block';}}}})()"></script>
  <style>
    :root {{
      --bg: #081018;
      --card: rgba(10, 16, 24, 0.90);
      --line: rgba(255,255,255,0.12);
      --text: #f4f7fb;
      --muted: rgba(244,247,251,0.68);
      --accent: #ffd24a;
      --selected: #ff6156;
      --airport: #bcff5c;
      --airport-soft: #9fd86a;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; padding: 0; width: 100%; height: {height}px; min-height: 100vh; overflow: hidden; background: var(--bg); }}
    body {{ font-family: "SF Pro Display", "Segoe UI", Arial, sans-serif; }}
    #stage {{
      position: relative;
      width: 100%;
      height: {height}px;
      overflow: hidden;
      background: #060a10;
    }}
    #map {{ position: absolute; inset: 0; width: 100%; height: 100%; }}
    #map::before {{
      content: "";
      position: absolute;
      inset: 0;
      pointer-events: none;
      z-index: 5;
      border: 1px solid rgba(255,255,255,0.04);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
    }}
    #map::after {{
      content: "";
      position: absolute;
      inset: 0;
      pointer-events: none;
      z-index: 6;
      background:
        radial-gradient(ellipse at top, rgba(0,0,0,0), rgba(0,0,0,0.12) 100%),
        radial-gradient(ellipse at bottom, rgba(0,0,0,0.10), transparent 70%);
    }}
    .mapboxgl-canvas-container canvas {{
      filter: saturate(1.12) contrast(1.05);
    }}
    #splash {{
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      background: linear-gradient(180deg, rgba(6,10,16,0.94), rgba(7,13,20,0.90));
      z-index: 20;
      transition: opacity 280ms ease;
    }}
    #splash.hidden {{
      opacity: 0 !important;
      pointer-events: none !important;
      display: none !important;
    }}
    .splash-card {{
      width: min(420px, 76vw);
      padding: 24px 22px;
      border-radius: 20px;
      border: 1px solid rgba(255,255,255,0.08);
      background: linear-gradient(180deg, rgba(12,18,28,0.94), rgba(10,15,24,0.90));
      box-shadow: 0 24px 60px rgba(0,0,0,0.38);
    }}
    .splash-title {{
      color: var(--text);
      font-size: 34px;
      font-weight: 800;
      letter-spacing: -0.04em;
      margin-bottom: 18px;
    }}
    .skeleton {{
      height: 18px;
      border-radius: 999px;
      margin-top: 14px;
      background: linear-gradient(90deg, rgba(255,255,255,0.05), rgba(255,255,255,0.10), rgba(255,255,255,0.05));
      background-size: 220% 100%;
      animation: shimmer 1.25s linear infinite;
    }}
    .skeleton.small {{ width: 54%; }}
    .skeleton.medium {{ width: 68%; }}
    .skeleton.large {{ width: 82%; }}
    @keyframes shimmer {{
      0% {{ background-position: 200% 0; }}
      100% {{ background-position: -200% 0; }}
    }}
    .mapboxgl-ctrl-logo, .mapboxgl-ctrl-attrib {{ display: none !important; }}
    .mapboxgl-ctrl-group {{
      border: 1px solid rgba(255,255,255,0.10) !important;
      box-shadow: 0 12px 32px rgba(0,0,0,0.28) !important;
      border-radius: 14px !important;
      overflow: hidden;
      background: rgba(10, 16, 24, 0.90) !important;
    }}
    .mapboxgl-ctrl button {{
      background: rgba(10, 16, 24, 0.90) !important;
    }}
    .mapboxgl-ctrl button .mapboxgl-ctrl-icon {{
      filter: invert(1) opacity(0.92);
    }}
    .aircraft-marker {{
      width: 34px;
      height: 34px;
      transform-origin: center center;
      cursor: pointer;
      filter: drop-shadow(0 2px 6px rgba(0,0,0,0.35));
    }}
    .aircraft-marker.selected {{
      width: 42px;
      height: 42px;
      filter: drop-shadow(0 0 14px rgba(255,97,86,0.58));
    }}
    .airport-dot {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: rgba(188,255,92,0.96);
      border: 2px solid rgba(4,8,13,0.90);
      box-shadow: 0 0 0 7px rgba(188,255,92,0.14);
    }}
    .mapboxgl-popup {{
      max-width: 360px !important;
    }}
    .mapboxgl-popup-content {{
      padding: 0 !important;
      background: transparent !important;
      box-shadow: none !important;
      border-radius: 0 !important;
    }}
    .mapboxgl-popup-tip {{
      border-top-color: rgba(7,12,20,0.98) !important;
    }}
    .popup-card {{
      min-width: 252px;
      border-radius: 14px;
      overflow: hidden;
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(7,12,20,0.98);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      box-shadow: 0 28px 64px rgba(0,0,0,0.58), 0 0 0 1px rgba(255,255,255,0.04);
      color: var(--text);
    }}
    .popup-accent-bar {{
      height: 3px;
      background: linear-gradient(90deg, #ffd24a 0%, #ff8c42 100%);
    }}
    .popup-accent-bar.selected {{
      background: linear-gradient(90deg, #ff6156 0%, #ff3357 100%);
    }}
    .popup-head {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      padding: 10px 13px 8px;
      gap: 10px;
    }}
    .popup-flight-num {{
      font-size: 20px;
      font-weight: 900;
      letter-spacing: -0.04em;
      line-height: 1;
      color: #fff;
    }}
    .popup-airline-row {{
      margin-top: 4px;
      font-size: 11px;
      font-weight: 600;
      color: rgba(244,247,251,0.52);
    }}
    .popup-status-pill {{
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 8px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      font-weight: 800;
      white-space: nowrap;
      flex-shrink: 0;
    }}
    .popup-status-pill.enroute {{
      background: rgba(30,180,90,0.18);
      color: #5de888;
      border: 1px solid rgba(50,200,100,0.26);
    }}
    .popup-status-pill.ground {{
      background: rgba(255,200,60,0.14);
      color: #ffd24a;
      border: 1px solid rgba(255,200,60,0.24);
    }}
    .popup-divider {{
      height: 1px;
      background: rgba(255,255,255,0.07);
      margin: 0 13px;
    }}
    .popup-route-row {{
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 8px 13px;
    }}
    .popup-iata {{
      font-size: 20px;
      font-weight: 900;
      letter-spacing: -0.04em;
      line-height: 1;
      color: #f4f8ff;
    }}
    .popup-city {{
      margin-top: 2px;
      font-size: 10px;
      font-weight: 600;
      color: rgba(244,247,251,0.44);
      max-width: 90px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .popup-route-mid {{
      flex: 1;
      display: flex;
      align-items: center;
      gap: 3px;
      justify-content: center;
    }}
    .popup-route-line {{
      flex: 1;
      height: 0;
      border-top: 1.5px dashed rgba(255,210,74,0.34);
    }}
    .popup-plane-icon {{
      color: #ffd24a;
      font-size: 13px;
      flex-shrink: 0;
    }}
    .popup-metrics-row {{
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
    }}
    .popup-metric {{
      padding: 7px 10px;
      border-right: 1px solid rgba(255,255,255,0.06);
    }}
    .popup-metric:last-child {{
      border-right: none;
    }}
    .popup-metric-value {{
      font-size: 13px;
      font-weight: 800;
      color: #eef4ff;
      line-height: 1;
    }}
    .popup-metric-label {{
      margin-top: 4px;
      font-size: 8px;
      font-weight: 700;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: rgba(244,247,251,0.38);
    }}
    .popup-footer {{
      padding: 6px 13px;
      background: rgba(255,255,255,0.022);
      border-top: 1px solid rgba(255,255,255,0.06);
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 10px;
      color: rgba(244,247,251,0.42);
    }}
    .popup-footer-val {{
      color: rgba(244,247,251,0.70);
      font-weight: 700;
    }}
    #error-box {{
      position: absolute;
      left: 20px;
      bottom: 20px;
      z-index: 30;
      display: none;
      max-width: 360px;
      padding: 14px 16px;
      border-radius: 14px;
      background: rgba(89, 17, 20, 0.92);
      color: #ffe7e8;
      border: 1px solid rgba(255,255,255,0.10);
      box-shadow: 0 18px 42px rgba(0,0,0,0.32);
    }}
    #overlay-root {{
      position: absolute;
      inset: 0;
      z-index: 12;
      pointer-events: none;
    }}
    .glass-panel {{
      background: linear-gradient(180deg, rgba(10, 14, 22, 0.92), rgba(8, 12, 18, 0.88));
      border: 1px solid rgba(255,255,255,0.08);
      box-shadow: 0 18px 42px rgba(0,0,0,0.24);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
    }}
    .brand-panel {{
      position: absolute;
      top: 14px;
      left: 14px;
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 288px;
      padding: 12px 16px;
      border-radius: 20px;
      pointer-events: auto;
    }}
    .brand-mark {{
      width: 42px;
      height: 42px;
      border-radius: 14px;
      display: grid;
      place-items: center;
      overflow: hidden;
    }}
    .brand-copy {{
      display: flex;
      flex-direction: column;
      gap: 3px;
    }}
    .brand-title {{
      color: #f5f8fd;
      font-size: 17px;
      font-weight: 850;
      letter-spacing: -0.03em;
      line-height: 1.05;
    }}
    .brand-subtitle {{
      color: rgba(244,247,251,0.48);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }}
    .top-center-stack {{
      position: absolute;
      top: 14px;
      left: 322px;
      right: 382px;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 10px;
      pointer-events: auto;
    }}
    .top-center-stack .search-wrap {{
      width: 100%;
      min-width: 0;
      max-width: 460px;
    }}
    .stats-pill {{
      display: flex;
      align-items: stretch;
      gap: 0;
      padding: 6px 8px;
      border-radius: 18px;
      max-width: 100%;
      overflow: hidden;
      background: linear-gradient(180deg, rgba(10,14,22,0.92), rgba(8,12,18,0.88));
      border: 1px solid rgba(255,255,255,0.08);
      box-shadow: 0 18px 42px rgba(0,0,0,0.24);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
    }}
    .stat-cell {{
      flex: 1 1 78px;
      min-width: 0;
      padding: 6px 10px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      overflow: hidden;
      border-right: 1px solid rgba(255,255,255,0.08);
    }}
    .stat-cell:last-child {{
      border-right: none;
    }}
    .stat-value {{
      color: #f4f7fb;
      font-size: 14px;
      font-weight: 800;
      line-height: 1;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .stat-label {{
      margin-top: 4px;
      color: rgba(244,247,251,0.52);
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      white-space: nowrap;
    }}
    .credit-bar {{
      position: absolute;
      bottom: 10px;
      left: 14px;
      display: flex;
      align-items: center;
      gap: 12px;
      pointer-events: auto;
      z-index: 10;
    }}
    .credit-bar a {{
      color: rgba(244,247,251,0.38);
      font-size: 10px;
      font-weight: 600;
      letter-spacing: 0.06em;
      text-decoration: none;
      text-transform: uppercase;
      transition: color 0.15s;
    }}
    .credit-bar a:hover {{
      color: rgba(244,247,251,0.75);
    }}
    .credit-sep {{
      width: 3px;
      height: 3px;
      border-radius: 50%;
      background: rgba(244,247,251,0.2);
    }}
    .top-actions {{
      position: absolute;
      top: 14px;
      right: 14px;
      display: flex;
      flex-direction: column;
      align-items: flex-end;
      gap: 6px;
      pointer-events: auto;
    }}
    .tools-row-status {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .tools-strip {{
      display: flex;
      align-items: center;
      gap: 6px;
      flex-wrap: wrap;
      max-width: 320px;
      justify-content: flex-end;
    }}
    #freshness-badge {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 6px 11px;
      border-radius: 14px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.02em;
      white-space: nowrap;
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(10,14,22,0.86);
      color: rgba(244,247,251,0.75);
      transition: color 0.4s, border-color 0.4s;
    }}
    #freshness-badge.fresh  {{ color: #4ade80; border-color: rgba(74,222,128,0.30); }}
    #freshness-badge.stale  {{ color: #fbbf24; border-color: rgba(251,191,36,0.30);  }}
    #freshness-badge.offline{{ color: #f87171; border-color: rgba(248,113,113,0.30); }}
    #freshness-dot {{
      width: 6px; height: 6px;
      border-radius: 50%;
      background: currentColor;
      flex-shrink: 0;
    }}
    .tool-btn, .module-btn {{
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 12px;
      background: rgba(10,14,22,0.86);
      color: rgba(244,247,251,0.84);
      padding: 6px 10px;
      font-size: 10.5px;
      font-weight: 700;
      letter-spacing: 0.01em;
      white-space: nowrap;
    }}
    .tool-btn.active, .module-btn.active {{
      color: #d4ff69;
      border-color: rgba(212,255,105,0.24);
      box-shadow: inset 0 0 0 1px rgba(212,255,105,0.12);
    }}
    .module-strip {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: nowrap;
      overflow: auto hidden;
      scrollbar-width: none;
      padding: 6px 8px;
      border-radius: 18px;
      width: max-content;
      max-width: calc(100% - 20px);
      background: linear-gradient(180deg, rgba(10,14,22,0.92), rgba(8,12,18,0.88));
      border: 1px solid rgba(255,255,255,0.08);
      box-shadow: 0 18px 42px rgba(0,0,0,0.24);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
    }}
    .module-strip::-webkit-scrollbar {{
      display: none;
    }}
    .search-wrap {{
      position: relative;
      width: 440px;
      min-width: 220px;
      max-width: 480px;
    }}
    .search-shell {{
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 0 10px 0 12px;
      border-radius: 16px;
      height: 44px;
      width: 100%;
      max-width: 480px;
      background: rgba(14,20,30,0.92);
      border: 1px solid rgba(255,255,255,0.10);
      box-shadow: 0 8px 28px rgba(0,0,0,0.32);
      backdrop-filter: blur(18px);
      -webkit-backdrop-filter: blur(18px);
      transition: border-color 160ms, box-shadow 160ms;
    }}
    .search-shell:focus-within {{
      border-color: rgba(0,229,255,0.42);
      box-shadow: 0 0 0 3px rgba(0,229,255,0.08), 0 8px 28px rgba(0,0,0,0.32);
    }}
    .search-icon {{
      color: rgba(244,247,251,0.36);
      font-size: 15px;
      flex-shrink: 0;
      line-height: 1;
      pointer-events: none;
      transition: color 160ms;
    }}
    .search-shell:focus-within .search-icon {{
      color: rgba(0,229,255,0.72);
    }}
    .search-input {{
      flex: 1;
      border: none;
      outline: none;
      background: transparent;
      color: #f4f7fb;
      font-size: 14px;
      font-weight: 600;
      min-width: 0;
      height: 100%;
    }}
    .search-input::placeholder {{
      color: rgba(244,247,251,0.32);
      font-weight: 500;
    }}
    .search-clear {{
      display: none;
      align-items: center;
      justify-content: center;
      width: 22px;
      height: 22px;
      border-radius: 999px;
      border: none;
      background: rgba(255,255,255,0.10);
      color: rgba(244,247,251,0.56);
      font-size: 13px;
      cursor: pointer;
      flex-shrink: 0;
      line-height: 1;
    }}
    .search-clear.visible {{
      display: flex;
    }}
    .search-clear:hover {{
      background: rgba(255,255,255,0.18);
      color: #f4f7fb;
    }}
    .search-kbd {{
      flex-shrink: 0;
      padding: 3px 7px;
      border-radius: 6px;
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.10);
      color: rgba(244,247,251,0.32);
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.04em;
      font-family: inherit;
      white-space: nowrap;
    }}
    .search-dropdown {{
      position: absolute;
      top: calc(100% + 8px);
      left: 0;
      right: 0;
      border-radius: 16px;
      overflow: hidden;
      background: rgba(10,16,26,0.97);
      border: 1px solid rgba(255,255,255,0.10);
      box-shadow: 0 24px 56px rgba(0,0,0,0.54);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      z-index: 50;
      display: none;
      max-height: 320px;
      overflow-y: auto;
      scrollbar-width: thin;
      scrollbar-color: rgba(255,255,255,0.14) transparent;
    }}
    .search-dropdown.open {{
      display: block;
    }}
    .search-dd-section {{
      padding: 8px 0 4px;
    }}
    .search-dd-header {{
      padding: 4px 14px 6px;
      font-size: 9px;
      font-weight: 800;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: rgba(244,247,251,0.26);
    }}
    .search-dd-row {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 9px 14px;
      cursor: pointer;
      transition: background 80ms;
    }}
    .search-dd-row:hover, .search-dd-row.focused {{
      background: rgba(0,229,255,0.07);
    }}
    .search-dd-dot {{
      width: 8px;
      height: 8px;
      border-radius: 999px;
      flex-shrink: 0;
      background: #00E5FF;
    }}
    .search-dd-dot.sel {{ background: #FF4B2B; }}
    .search-dd-dot.apt {{
      width: 8px;
      height: 8px;
      border-radius: 2px;
      background: #bcff5c;
    }}
    .search-dd-flight {{
      font-size: 13px;
      font-weight: 800;
      color: #eef4ff;
      min-width: 72px;
    }}
    .search-dd-apt-code {{
      font-size: 13px;
      font-weight: 900;
      color: #d4ff80;
      min-width: 46px;
      letter-spacing: 0.02em;
    }}
    .search-dd-route {{
      font-size: 12px;
      font-weight: 600;
      color: rgba(244,247,251,0.44);
      flex: 1;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .search-dd-alt {{
      font-size: 11px;
      font-weight: 700;
      color: rgba(244,247,251,0.30);
      flex-shrink: 0;
    }}
    .search-dd-hub {{
      font-size: 9px;
      font-weight: 800;
      letter-spacing: 0.10em;
      text-transform: uppercase;
      padding: 2px 6px;
      border-radius: 999px;
      background: rgba(188,255,92,0.14);
      border: 1px solid rgba(188,255,92,0.22);
      color: #bcff5c;
      flex-shrink: 0;
    }}
    .search-dd-divider {{
      height: 1px;
      background: rgba(255,255,255,0.06);
      margin: 2px 0;
    }}
    .search-dd-empty {{
      padding: 18px 14px;
      font-size: 13px;
      color: rgba(244,247,251,0.32);
      text-align: center;
    }}
    .left-rail {{
      position: absolute;
      top: 126px;
      left: 14px;
      width: 260px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      pointer-events: auto;
      max-height: 440px;
      overflow-x: hidden;
      overflow-y: scroll;
      scrollbar-width: thin;
      scrollbar-color: rgba(255,210,74,0.45) rgba(255,255,255,0.05);
    }}
    .left-rail::-webkit-scrollbar {{ width: 6px; }}
    .left-rail::-webkit-scrollbar-track {{
      background: rgba(255,255,255,0.04);
      border-radius: 99px;
      margin: 4px 0;
    }}
    .left-rail::-webkit-scrollbar-thumb {{
      background: rgba(255,210,74,0.45);
      border-radius: 99px;
    }}
    .left-rail::-webkit-scrollbar-thumb:hover {{ background: rgba(255,210,74,0.75); }}
    .rail-card {{
      border-radius: 12px;
      overflow: hidden;
      flex-shrink: 0;
    }}
    .rail-card-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 8px 12px 7px;
      border-bottom: 1px solid rgba(255,255,255,0.06);
    }}
    .rail-heading {{
      color: rgba(244,247,251,0.46);
      font-size: 10px;
      font-weight: 800;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      margin: 0;
    }}
    .rail-count-pill {{
      padding: 1px 6px;
      border-radius: 999px;
      background: rgba(255,255,255,0.07);
      color: rgba(244,247,251,0.50);
      font-size: 9px;
      font-weight: 700;
      letter-spacing: 0.04em;
    }}
    .rail-rows {{ padding: 3px 0 5px; }}
    .rail-row {{
      display: flex;
      align-items: center;
      gap: 0;
      padding: 0;
      cursor: pointer;
      transition: background 0.14s ease;
      position: relative;
      border-left: 2.5px solid transparent;
      border-bottom: 1px solid rgba(255,255,255,0.045);
    }}
    .rail-row:last-child {{ border-bottom: none; }}
    .rail-row:hover {{
      background: rgba(255,255,255,0.055);
      border-left-color: rgba(255,210,74,0.70);
    }}
    .rail-row:active {{ background: rgba(255,255,255,0.09); }}
    .rail-row-inner {{
      display: flex;
      align-items: center;
      width: 100%;
      padding: 8px 12px 8px 10px;
      gap: 0;
    }}
    .rr-icon {{
      width: 26px;
      height: 26px;
      border-radius: 7px;
      display: grid;
      place-items: center;
      flex-shrink: 0;
      font-size: 11px;
      margin-right: 8px;
    }}
    .rr-icon.airport {{ background: rgba(188,255,92,0.10); color: #bcff5c; }}
    .rr-icon.route   {{ background: rgba(0,210,255,0.10);  color: #00d2ff; }}
    .rr-icon.flight  {{ background: rgba(255,166,60,0.12);  color: #ffa63c; }}
    .rr-body {{
      flex: 1;
      min-width: 0;
    }}
    .rr-primary {{
      color: #f0f5ff;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: -0.01em;
      line-height: 1.2;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .rr-secondary {{
      color: rgba(244,247,251,0.42);
      font-size: 10px;
      font-weight: 500;
      line-height: 1.3;
      margin-top: 1px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .rr-right {{
      display: flex;
      flex-direction: column;
      align-items: flex-end;
      gap: 3px;
      flex-shrink: 0;
      margin-left: 6px;
    }}
    .rr-badge {{
      padding: 1px 6px;
      border-radius: 999px;
      font-size: 9px;
      font-weight: 800;
      letter-spacing: 0.04em;
      white-space: nowrap;
    }}
    .rr-badge.high   {{ background: rgba(210,48,55,0.28); color: #ff9a9c; border: 1px solid rgba(210,48,55,0.30); }}
    .rr-badge.medium {{ background: rgba(200,130,15,0.22); color: #ffd060; border: 1px solid rgba(200,130,15,0.28); }}
    .rr-badge.low    {{ background: rgba(40,160,80,0.22); color: #7ef0a0; border: 1px solid rgba(40,160,80,0.28); }}
    .rr-badge.neutral{{ background: rgba(255,255,255,0.08); color: rgba(244,247,251,0.60); border: 1px solid rgba(255,255,255,0.10); }}
    .rr-meta {{
      color: rgba(244,247,251,0.38);
      font-size: 9px;
      font-weight: 700;
    }}
    .rr-track-btn {{
      opacity: 0;
      transition: opacity 0.15s ease;
      padding: 2px 6px;
      border-radius: 5px;
      background: rgba(0,229,255,0.10);
      border: 1px solid rgba(0,229,255,0.20);
      color: #00e5ff;
      font-size: 8px;
      font-weight: 800;
      letter-spacing: 0.08em;
      cursor: pointer;
      white-space: nowrap;
    }}
    .rail-row:hover .rr-track-btn {{ opacity: 1; }}
    /* legacy tag class kept for alerts panel */
    .tag {{
      padding: 3px 9px;
      border-radius: 999px;
      font-size: 10px;
      font-weight: 800;
      letter-spacing: 0.02em;
      color: #f4f7fb;
      background: rgba(255,255,255,0.08);
    }}
    .tag.high   {{ background: rgba(175,48,55,0.42); color: #ffb2b4; }}
    .tag.medium {{ background: rgba(155,108,21,0.34); color: #ffd678; }}
    .tag.low    {{ background: rgba(49,121,63,0.32); color: #9ff0ae; }}
    .list-row {{
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 10px;
      align-items: center;
      padding: 6px 14px;
      color: #edf3f9;
      font-size: 13px;
      font-weight: 700;
    }}
    .list-row .meta {{
      color: rgba(244,247,251,0.54);
      font-size: 12px;
      font-weight: 600;
    }}
    @keyframes rpSlideIn {{
      from {{ opacity: 0; transform: translateX(18px) scale(0.97); }}
      to   {{ opacity: 1; transform: translateX(0) scale(1); }}
    }}
    @keyframes rpSlideUp {{
      from {{ opacity: 0; transform: translateY(100%); }}
      to   {{ opacity: 1; transform: translateY(0); }}
    }}
    #map-shield {{
      display: none;
      position: absolute;
      inset: 0;
      z-index: 49; /* below right-panel (auto stacking) but above map canvas */
      background: transparent;
      touch-action: none;
      -webkit-tap-highlight-color: transparent;
    }}
    @media (max-width: 600px) {{
      .right-panel.rp-visible {{
        animation: rpSlideUp 0.28s cubic-bezier(0.16, 1, 0.3, 1) both;
      }}
    }}
    .right-panel {{
      position: absolute;
      top: 132px;
      right: 14px;
      width: 300px;
      border-radius: 16px;
      overflow-x: hidden;
      overflow-y: scroll;
      pointer-events: auto;
      display: none;
      max-height: 560px;
      scrollbar-width: thin;
      scrollbar-color: rgba(0,229,255,0.40) rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.11);
      background: rgba(8,13,21,0.97);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      box-shadow: 0 28px 64px rgba(0,0,0,0.52);
      scroll-behavior: smooth;
    }}
    .right-panel.rp-visible {{
      animation: rpSlideIn 0.26s cubic-bezier(0.16, 1, 0.3, 1) both;
    }}
    .right-panel::-webkit-scrollbar {{ width: 5px; }}
    .right-panel::-webkit-scrollbar-track {{
      background: rgba(255,255,255,0.04);
      border-radius: 99px;
      margin: 8px 0;
    }}
    .right-panel::-webkit-scrollbar-thumb {{
      background: rgba(0,229,255,0.40);
      border-radius: 99px;
    }}
    .right-panel::-webkit-scrollbar-thumb:hover {{ background: rgba(0,229,255,0.70); }}

    /* ── Flight Board (FIDS) overlay ─────────────────────────────────────── */
    #flight-board-overlay {{
      display: none;
      position: absolute;
      inset: 0;
      background: rgba(8,11,17,0.98);
      backdrop-filter: blur(8px);
      -webkit-backdrop-filter: blur(8px);
      z-index: 200;
      flex-direction: column;
      overflow: hidden;
    }}
    #flight-board-overlay.visible {{ display: flex; }}
    .fids-toolbar {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 12px 18px 10px;
      border-bottom: 1px solid rgba(255,255,255,0.07);
      flex-shrink: 0;
    }}
    .fids-title {{
      font-size: 12px;
      font-weight: 700;
      letter-spacing: .12em;
      text-transform: uppercase;
      color: #00e5ff;
      flex: 1;
    }}
    .fids-filter-group {{
      display: flex;
      gap: 4px;
    }}
    .fids-filter-btn {{
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.10);
      border-radius: 99px;
      color: #aaa;
      font-size: 11px;
      font-weight: 600;
      padding: 5px 14px;
      cursor: pointer;
      transition: all .15s;
    }}
    .fids-filter-btn:hover {{ background: rgba(255,255,255,0.10); color: #fff; }}
    .fids-filter-btn.active {{
      background: rgba(0,229,255,0.18);
      color: #00e5ff;
      border-color: rgba(0,229,255,0.40);
    }}
    .fids-toolbar-right {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .fids-live-badge {{
      display: flex;
      align-items: center;
      gap: 5px;
      font-size: 10px;
      font-weight: 700;
      color: #50dc78;
      letter-spacing: .08em;
    }}
    .fids-live-dot {{
      width: 7px; height: 7px;
      border-radius: 50%;
      background: #50dc78;
      animation: pulse 1.8s ease-in-out infinite;
    }}
    .fids-close-btn {{
      background: rgba(255,255,255,0.07);
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 8px;
      color: #aaa;
      font-size: 20px;
      line-height: 1;
      width: 32px; height: 32px;
      cursor: pointer;
      display: flex; align-items: center; justify-content: center;
      transition: all .15s;
    }}
    .fids-close-btn:hover {{ background: rgba(255,60,60,0.18); color: #ff6b6b; border-color: rgba(255,60,60,0.35); }}
    .fids-col-hint {{
      font-size: 10px;
      color: #444;
      padding: 4px 18px 2px;
      flex-shrink: 0;
    }}
    .fids-table-wrap {{
      flex: 1;
      overflow-y: auto;
      scrollbar-width: thin;
      scrollbar-color: rgba(0,229,255,0.30) transparent;
    }}
    .fids-table-wrap::-webkit-scrollbar {{ width: 5px; }}
    .fids-table-wrap::-webkit-scrollbar-thumb {{ background: rgba(0,229,255,0.30); border-radius: 99px; }}
    .fids-table {{
      width: 100%;
      border-collapse: collapse;
    }}
    .fids-table thead th {{
      position: sticky;
      top: 0;
      background: rgba(15,17,24,0.98);
      font-size: 10px;
      font-weight: 700;
      letter-spacing: .1em;
      text-transform: uppercase;
      color: #555;
      padding: 6px 12px;
      text-align: left;
      border-bottom: 1px solid rgba(255,255,255,0.06);
      white-space: nowrap;
    }}
    .fids-table thead th.sortable {{ cursor: pointer; user-select: none; }}
    .fids-table thead th.sortable:hover {{ color: #aaa; }}
    .fids-row {{
      border-bottom: 1px solid rgba(255,255,255,0.04);
      transition: background .12s;
      cursor: pointer;
    }}
    .fids-row:hover {{ background: rgba(255,255,255,0.04); }}
    .fids-row td {{
      padding: 7px 12px;
      font-size: 12px;
      color: #ccc;
      white-space: nowrap;
    }}
    .fids-flight {{
      font-weight: 700;
      color: #fff;
      font-family: 'JetBrains Mono', monospace;
      font-size: 13px;
    }}
    .fids-route {{
      font-weight: 600;
      color: #aaa;
      font-size: 12px;
    }}
    .fids-airline {{
      color: #888;
      font-size: 11px;
    }}
    .fids-chip {{
      display: inline-block;
      border-radius: 5px;
      padding: 2px 7px;
      font-size: 10px;
      font-weight: 700;
      letter-spacing: .05em;
      text-transform: uppercase;
    }}
    .fids-chip.enroute  {{ background: rgba(0,229,255,0.12);  color: #00e5ff; }}
    .fids-chip.landed   {{ background: rgba(80,220,120,0.12); color: #50dc78; }}
    .fids-chip.ground   {{ background: rgba(255,255,255,0.07); color: #888; }}
    .fids-chip.delayed  {{ background: rgba(255,140,60,0.15);  color: #ff8c3c; }}
    .fids-chip.ontime   {{ background: rgba(80,220,120,0.12);  color: #50dc78; }}
    .fids-chip.early    {{ background: rgba(0,200,255,0.12);   color: #00c8ff; }}
    .fids-delay-num {{ font-weight: 700; }}
    .fids-delay-num.pos {{ color: #ff8c3c; }}
    .fids-delay-num.neg {{ color: #50dc78; }}
    .fids-delay-num.zero {{ color: #888; }}
    .fids-empty {{
      text-align: center;
      padding: 48px 0;
      color: #444;
      font-size: 13px;
    }}
    .fids-footer {{
      flex-shrink: 0;
      padding: 6px 16px;
      border-top: 1px solid rgba(255,255,255,0.05);
      font-size: 10px;
      color: #444;
      display: flex;
      justify-content: space-between;
    }}
    .panel-top-accent {{
      height: 3px;
      background: linear-gradient(90deg, #00e5ff 0%, #0097c4 100%);
      flex-shrink: 0;
      border-radius: 16px 16px 0 0;
    }}
    .panel-inner {{
      padding: 11px 14px 14px;
    }}
    .panel-head-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 8px;
    }}
    .panel-live-badge {{
      display: flex;
      align-items: center;
      gap: 5px;
      padding: 3px 8px;
      border-radius: 999px;
      background: rgba(0,229,255,0.09);
      border: 1px solid rgba(0,229,255,0.20);
      font-size: 8px;
      font-weight: 800;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: #00e5ff;
    }}
    .panel-live-dot {{
      width: 6px;
      height: 6px;
      border-radius: 999px;
      background: #00e5ff;
      animation: pulseDot 1.8s ease-in-out infinite;
      flex-shrink: 0;
    }}
    @keyframes pulseDot {{
      0%, 100% {{ opacity: 1; transform: scale(1); }}
      50% {{ opacity: 0.35; transform: scale(0.65); }}
    }}
    .panel-close {{
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.10);
      border-radius: 10px;
      color: rgba(244,247,251,0.52);
      font-size: 18px;
      line-height: 1;
      font-weight: 500;
      cursor: pointer;
      width: 32px;
      height: 32px;
      display: grid;
      place-items: center;
      transition: background 120ms, color 120ms;
    }}
    .panel-close:hover {{
      background: rgba(255,255,255,0.12);
      color: rgba(244,247,251,0.90);
    }}
    .panel-flight-hero {{
      font-size: 24px;
      font-weight: 900;
      letter-spacing: -0.04em;
      line-height: 1;
      color: #fff;
      margin-bottom: 4px;
    }}
    .panel-airline-sub {{
      font-size: 11px;
      font-weight: 600;
      color: rgba(244,247,251,0.52);
      margin-bottom: 9px;
    }}
    .panel-chip-row {{
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
    }}
    .pchip {{
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 10px;
      font-weight: 700;
      background: rgba(255,255,255,0.07);
      border: 1px solid rgba(255,255,255,0.10);
      color: rgba(244,247,251,0.80);
    }}
    .pchip.status-enroute {{
      background: rgba(20,170,80,0.16);
      border-color: rgba(40,200,100,0.24);
      color: #5de888;
    }}
    .pchip.status-ground {{
      background: rgba(230,170,30,0.14);
      border-color: rgba(255,200,60,0.22);
      color: #ffd24a;
    }}
    /* ── Phase of flight chip ──────────────────────────────────────────── */
    .pchip.phase-climb  {{ background:rgba(60,180,255,0.14); border-color:rgba(60,180,255,0.28); color:#7dd8ff; }}
    .pchip.phase-cruise {{ background:rgba(0,229,255,0.10); border-color:rgba(0,229,255,0.22); color:#00e5ff; }}
    .pchip.phase-descent {{ background:rgba(255,160,30,0.14); border-color:rgba(255,160,30,0.28); color:#ffad50; }}
    .pchip.phase-approach {{ background:rgba(255,90,50,0.14); border-color:rgba(255,90,50,0.28); color:#ff8060; }}
    .pchip.phase-ground {{ background:rgba(180,180,180,0.10); border-color:rgba(180,180,180,0.18); color:rgba(244,247,251,0.45); }}
    /* ── Aircraft type filter strip ───────────────────────────────────── */
    .ac-filter-strip {{
      display: flex; gap: 4px; padding: 0 2px; flex-wrap: nowrap;
    }}
    .ac-filter-btn {{
      padding: 3px 9px; border-radius: 999px; font-size: 10px; font-weight: 700;
      background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.10);
      color: rgba(244,247,251,0.55); cursor: pointer; white-space: nowrap;
      transition: background 0.15s, color 0.15s, border-color 0.15s;
    }}
    .ac-filter-btn.active {{
      background: rgba(0,229,255,0.14); border-color: rgba(0,229,255,0.30); color: #00e5ff;
    }}
    /* ── Nations card bar ─────────────────────────────────────────────── */
    .nations-row {{ display:flex; align-items:center; gap:7px; padding:4px 0; font-size:11px; }}
    .nations-flag {{ font-size:15px; flex-shrink:0; }}
    .nations-bar-wrap {{ flex:1; height:4px; border-radius:2px; background:rgba(255,255,255,0.07); overflow:hidden; }}
    .nations-bar-fill {{ height:100%; border-radius:2px; background:rgba(0,229,255,0.55); }}
    .nations-count {{ font-size:10px; font-weight:700; color:rgba(244,247,251,0.65); flex-shrink:0; min-width:22px; text-align:right; }}
    .panel-section-divider {{
      height: 1px;
      background: rgba(255,255,255,0.07);
      margin: 10px 0;
    }}
    .panel-section-label {{
      font-size: 8px;
      font-weight: 800;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: rgba(244,247,251,0.28);
      margin-bottom: 8px;
    }}
    .route-display {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .route-apt {{
      flex: 1;
      min-width: 0;
    }}
    .route-apt-code {{
      font-size: 26px;
      font-weight: 900;
      letter-spacing: -0.04em;
      line-height: 1;
      color: #f6fbff;
    }}
    .route-apt-city {{
      margin-top: 3px;
      font-size: 11px;
      font-weight: 600;
      color: rgba(244,247,251,0.48);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .route-center {{
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 5px;
      flex-shrink: 0;
      min-width: 36px;
    }}
    .route-plane-icon {{
      color: #ffd24a;
      font-size: 15px;
    }}
    .route-dash-line {{
      width: 28px;
      height: 0;
      border-top: 2px dashed rgba(255,210,74,0.36);
    }}
    .metrics-grid-4 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
      margin-bottom: 6px;
    }}
    .mcard {{
      padding: 8px 10px;
      border-radius: 10px;
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.07);
      transition: background 0.15s ease, border-color 0.15s ease, transform 0.15s ease;
    }}
    .mcard:hover {{
      background: rgba(255,255,255,0.08);
      border-color: rgba(255,255,255,0.14);
      transform: translateY(-2px);
    }}
    .mcard-label {{
      font-size: 8px;
      font-weight: 800;
      letter-spacing: 0.13em;
      text-transform: uppercase;
      color: rgba(244,247,251,0.36);
      margin-bottom: 5px;
    }}
    .mcard-value {{
      font-size: 15px;
      font-weight: 800;
      color: #eef4ff;
      line-height: 1;
    }}
    .mcard-unit {{
      font-size: 10px;
      font-weight: 600;
      color: rgba(244,247,251,0.46);
      margin-left: 2px;
    }}
    .pos-card {{
      padding: 8px 10px;
      border-radius: 10px;
      background: rgba(17,71,45,0.26);
      border: 1px solid rgba(60,170,100,0.22);
      margin-bottom: 6px;
    }}
    .pos-card-label {{
      font-size: 8px;
      font-weight: 800;
      letter-spacing: 0.13em;
      text-transform: uppercase;
      color: rgba(60,200,120,0.68);
      margin-bottom: 5px;
    }}
    .pos-card-value {{
      font-size: 14px;
      font-weight: 800;
      color: #7ef0a8;
      line-height: 1.2;
    }}
    .pos-card-sub {{
      margin-top: 4px;
      font-size: 11px;
      font-weight: 600;
      color: rgba(100,220,140,0.55);
    }}
    .sparkline-card {{
      padding: 8px 10px;
      border-radius: 10px;
      background: rgba(14,20,30,0.55);
      border: 1px solid rgba(0,229,255,0.14);
      margin-bottom: 6px;
    }}
    .sparkline-label {{
      font-size: 8px;
      font-weight: 800;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: rgba(0,229,255,0.55);
      margin-bottom: 5px;
      display: flex;
      justify-content: space-between;
    }}
    .sparkline-svg {{
      display: block;
      width: 100%;
      height: 44px;
      overflow: visible;
    }}
    .delay-card {{
      padding: 8px 10px;
      border-radius: 10px;
      margin-bottom: 6px;
    }}
    .delay-card.low {{
      background: rgba(17,71,45,0.20);
      border: 1px solid rgba(60,170,100,0.20);
    }}
    .delay-card.medium {{
      background: rgba(90,60,10,0.26);
      border: 1px solid rgba(200,150,30,0.22);
    }}
    .delay-card.high {{
      background: rgba(80,18,22,0.32);
      border: 1px solid rgba(200,55,55,0.22);
    }}
    .delay-card-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 8px;
    }}
    .delay-badge {{
      padding: 3px 7px;
      border-radius: 999px;
      font-size: 9px;
      font-weight: 800;
      letter-spacing: 0.04em;
    }}
    .delay-badge.low {{ background: rgba(25,150,75,0.26); color: #72eca0; }}
    .delay-badge.medium {{ background: rgba(200,130,15,0.24); color: #ffcf50; }}
    .delay-badge.high {{ background: rgba(200,45,45,0.26); color: #ff8282; }}
    .delay-headline {{
      font-size: 13px;
      font-weight: 800;
      color: #f0f6ff;
      line-height: 1.3;
    }}
    .delay-eta {{
      margin-top: 3px;
      font-size: 11px;
      font-weight: 600;
      color: rgba(244,247,251,0.46);
    }}
    .delay-reasons {{
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      margin-top: 10px;
      max-height: 80px;
      overflow-y: auto;
      scrollbar-width: thin;
      scrollbar-color: rgba(255,255,255,0.18) transparent;
    }}
    .delay-reasons::-webkit-scrollbar {{ width: 3px; }}
    .delay-reasons::-webkit-scrollbar-thumb {{
      background: rgba(255,255,255,0.22);
      border-radius: 99px;
    }}
    /* ── Delay score bar ───────────────────────────────────────────────── */
    .delay-score-bar-wrap {{
      height: 4px;
      border-radius: 99px;
      background: rgba(255,255,255,0.08);
      margin: 8px 0 10px;
      overflow: hidden;
    }}
    .delay-score-bar-fill {{
      height: 100%;
      border-radius: 99px;
      transition: width 0.4s ease;
    }}
    .delay-score-bar-fill.low    {{ background: linear-gradient(90deg, #3ddc84, #72eca0); }}
    .delay-score-bar-fill.medium {{ background: linear-gradient(90deg, #e8a020, #ffd060); }}
    .delay-score-bar-fill.high   {{ background: linear-gradient(90deg, #d03040, #ff8282); }}
    /* ── Signal breakdown rows ─────────────────────────────────────────── */
    .delay-signals {{
      display: flex;
      flex-direction: column;
      gap: 4px;
      margin-top: 2px;
      max-height: 160px;
      overflow-y: auto;
      scrollbar-width: thin;
      scrollbar-color: rgba(255,255,255,0.15) transparent;
    }}
    .delay-signals::-webkit-scrollbar {{ width: 3px; }}
    .delay-signals::-webkit-scrollbar-thumb {{
      background: rgba(255,255,255,0.20);
      border-radius: 99px;
    }}
    .delay-signal-row {{
      display: flex;
      align-items: center;
      gap: 7px;
      padding: 4px 6px;
      border-radius: 7px;
      background: rgba(255,255,255,0.04);
    }}
    .delay-signal-icon {{
      font-size: 11px;
      flex-shrink: 0;
      width: 16px;
      text-align: center;
    }}
    .delay-signal-label {{
      flex: 1;
      font-size: 10px;
      font-weight: 600;
      color: rgba(244,247,251,0.72);
      line-height: 1.3;
    }}
    .delay-signal-min {{
      font-size: 9px;
      font-weight: 800;
      color: rgba(244,247,251,0.42);
      flex-shrink: 0;
      letter-spacing: 0.02em;
    }}
    .delay-reason-tag {{
      padding: 4px 9px;
      border-radius: 999px;
      font-size: 10px;
      font-weight: 700;
      background: rgba(255,255,255,0.07);
      border: 1px solid rgba(255,255,255,0.10);
      color: rgba(244,247,251,0.68);
    }}
    /* ── Delay Threshold Slider ────────────────────────────────────── */
    .dp-slider-wrap {{
      display: flex;
      align-items: center;
      gap: 7px;
      padding: 8px 12px 4px;
    }}
    .dp-slider-lbl {{
      color: rgba(244,247,251,0.36);
      font-size: 9px;
      font-weight: 700;
      flex-shrink: 0;
      letter-spacing: 0.04em;
    }}
    .dp-range {{
      flex: 1;
      -webkit-appearance: none;
      appearance: none;
      height: 3px;
      border-radius: 99px;
      background: linear-gradient(90deg, #ffd24a var(--fill, 0%), rgba(255,255,255,0.12) var(--fill, 0%));
      outline: none;
      cursor: pointer;
    }}
    .dp-range::-webkit-slider-thumb {{
      -webkit-appearance: none;
      width: 14px;
      height: 14px;
      border-radius: 50%;
      background: #ffd24a;
      border: 2px solid rgba(0,0,0,0.35);
      box-shadow: 0 2px 8px rgba(0,0,0,0.50);
      cursor: pointer;
      transition: transform 0.12s ease;
    }}
    .dp-range::-webkit-slider-thumb:hover {{ transform: scale(1.22); }}
    .dp-range::-moz-range-thumb {{
      width: 14px;
      height: 14px;
      border-radius: 50%;
      background: #ffd24a;
      border: 2px solid rgba(0,0,0,0.35);
      cursor: pointer;
    }}
    .dp-ticks {{
      display: flex;
      justify-content: space-between;
      padding: 0 12px 8px;
    }}
    .dp-tick {{
      color: rgba(244,247,251,0.28);
      font-size: 8px;
      font-weight: 700;
      letter-spacing: 0.02em;
    }}
    .dp-val-badge {{
      padding: 1px 7px;
      border-radius: 999px;
      background: rgba(255,210,74,0.14);
      border: 1px solid rgba(255,210,74,0.28);
      color: #ffd24a;
      font-size: 9px;
      font-weight: 800;
      letter-spacing: 0.04em;
      white-space: nowrap;
      flex-shrink: 0;
    }}
    .detail-table {{
      display: grid;
      grid-template-columns: 94px 1fr;
      row-gap: 7px;
      column-gap: 8px;
      font-size: 11px;
      margin-top: 8px;
    }}
    .tech-details-toggle {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      width: 100%;
      background: none;
      border: none;
      cursor: pointer;
      padding: 4px 0;
      margin: 0;
    }}
    .tech-toggle-chevron {{
      font-size: 11px;
      color: rgba(244,247,251,0.35);
      transition: transform 0.15s;
    }}
    .detail-table-key {{
      color: rgba(244,247,251,0.36);
      font-weight: 600;
      line-height: 1.4;
    }}
    .detail-table-val {{
      color: #eaf2ff;
      font-weight: 700;
      line-height: 1.4;
      word-break: break-all;
    }}
    /* legacy compat */
    .detail-key {{
      color: rgba(244,247,251,0.44);
      font-weight: 600;
    }}
    .detail-val {{
      color: #f4f8fc;
      font-weight: 700;
    }}
    .aircraft-marker-wrap {{
      position: relative;
      width: 42px;
      height: 20px;
      pointer-events: auto;
    }}
    .aircraft-icon-shell {{
      position: absolute;
      left: 0;
      top: 2px;
      width: 18px;
      height: 18px;
      transform-origin: center center;
      filter: drop-shadow(0 1px 6px rgba(0,0,0,0.42));
    }}
    .aircraft-label {{
      position: absolute;
      left: 20px;
      top: -1px;
      color: #24cfe2;
      font-size: 9px;
      font-weight: 800;
      letter-spacing: 0.01em;
      text-shadow: 0 2px 4px rgba(0,0,0,0.42);
      white-space: nowrap;
      opacity: 0;
      transform: translateY(2px);
      transition: opacity 160ms ease, transform 160ms ease;
      pointer-events: none;
    }}
    .aircraft-marker-wrap.show-label .aircraft-label {{
      opacity: 1;
      transform: translateY(0);
    }}
    .aircraft-marker-wrap.selected .aircraft-label {{
      color: #ffe35d;
      opacity: 1;
      transform: translateY(0);
    }}
    .selected-ring {{
      position: absolute;
      left: 1px;
      top: 2px;
      width: 18px;
      height: 18px;
      border-radius: 999px;
      border: 1.5px solid rgba(255, 202, 74, 0.95);
      box-shadow: 0 0 0 3px rgba(255, 202, 74, 0.14);
    }}
    .mapboxgl-ctrl-bottom-left {{
      bottom: 18px !important;
      left: 18px !important;
    }}
    /* ── Airport Traffic module cards ────────────────────────────────────── */
    .at-airport-card {{
      padding: 10px 12px;
      border-radius: 10px;
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.07);
      margin-bottom: 6px;
      cursor: pointer;
      transition: background .15s, border-color .15s;
    }}
    .at-airport-card:hover {{
      background: rgba(255,255,255,0.08);
      border-color: rgba(0,229,255,0.22);
    }}
    .at-airport-card-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 6px;
    }}
    .at-airport-code {{
      font-size: 16px;
      font-weight: 900;
      color: #f6fbff;
      letter-spacing: -0.02em;
    }}
    .at-airport-city {{
      font-size: 10px;
      color: rgba(244,247,251,0.45);
      font-weight: 600;
      margin-top: 1px;
    }}
    .at-level-badge {{
      padding: 2px 8px;
      border-radius: 99px;
      font-size: 9px;
      font-weight: 800;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }}
    .at-level-badge.critical {{ background: rgba(200,30,30,0.22); color: #ff6868; border: 1px solid rgba(200,30,30,0.30); }}
    .at-level-badge.high     {{ background: rgba(200,100,15,0.22); color: #ffad50; border: 1px solid rgba(200,100,15,0.28); }}
    .at-level-badge.moderate {{ background: rgba(180,160,20,0.18); color: #f0d050; border: 1px solid rgba(180,160,20,0.25); }}
    .at-level-badge.low      {{ background: rgba(30,160,80,0.18);  color: #50dc88; border: 1px solid rgba(30,160,80,0.24); }}
    .at-score-bar-wrap {{
      height: 3px;
      border-radius: 99px;
      background: rgba(255,255,255,0.07);
      margin: 6px 0;
      overflow: hidden;
    }}
    .at-score-bar-fill {{
      height: 100%;
      border-radius: 99px;
      transition: width 0.4s ease;
    }}
    .at-score-bar-fill.critical {{ background: linear-gradient(90deg,#d03040,#ff6868); }}
    .at-score-bar-fill.high     {{ background: linear-gradient(90deg,#c07010,#ffad50); }}
    .at-score-bar-fill.moderate {{ background: linear-gradient(90deg,#a0900a,#f0d050); }}
    .at-score-bar-fill.low      {{ background: linear-gradient(90deg,#1ea050,#50dc88); }}
    .at-metrics-mini {{
      display: grid;
      grid-template-columns: 1fr 1fr 1fr 1fr;
      gap: 4px;
      margin-bottom: 6px;
    }}
    .at-metric-cell {{
      background: rgba(255,255,255,0.04);
      border-radius: 6px;
      padding: 4px 5px;
      text-align: center;
    }}
    .at-metric-label {{
      font-size: 7px;
      font-weight: 800;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: rgba(244,247,251,0.30);
      margin-bottom: 2px;
    }}
    .at-metric-value {{
      font-size: 13px;
      font-weight: 800;
      color: #eef4ff;
      line-height: 1;
    }}
    .at-sched-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 4px;
      padding-top: 5px;
      border-top: 1px solid rgba(255,255,255,0.05);
    }}
    .at-sched-item {{
      display: flex;
      align-items: center;
      gap: 4px;
      font-size: 10px;
      font-weight: 700;
      color: rgba(244,247,251,0.55);
    }}
    .at-sched-num {{
      font-size: 12px;
      font-weight: 800;
      color: #00e5ff;
    }}
    .at-sched-num.dep {{ color: #ffd24a; }}
    /* ── 3-hour outlook chart ──────────────────────────────────────────── */
    .at-sched-chart {{
      margin-top: 6px;
      padding-top: 5px;
      border-top: 1px solid rgba(255,255,255,0.05);
    }}
    .at-sched-chart-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 5px;
    }}
    .at-sched-chart-title {{
      font-size: 7px;
      font-weight: 800;
      letter-spacing: 0.09em;
      text-transform: uppercase;
      color: rgba(244,247,251,0.30);
    }}
    .at-sched-bars {{
      display: flex;
      align-items: flex-end;
      gap: 3px;
      height: 32px;
    }}
    .at-sched-bar-group {{
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 2px;
      flex: 1;
    }}
    .at-sched-bar-pair {{
      display: flex;
      align-items: flex-end;
      gap: 2px;
      height: 26px;
      width: 100%;
    }}
    .at-sched-bar {{
      border-radius: 2px 2px 0 0;
      min-height: 2px;
      flex: 1;
      transition: height 0.35s ease;
    }}
    .at-sched-bar.arr {{ background: rgba(0,229,255,0.55); }}
    .at-sched-bar.dep {{ background: rgba(255,210,74,0.55); }}
    .at-sched-bar-label {{
      font-size: 7px;
      font-weight: 700;
      color: rgba(244,247,251,0.28);
      text-align: center;
      width: 100%;
    }}
    .at-sched-total {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 4px;
      font-size: 9px;
      font-weight: 700;
      color: rgba(244,247,251,0.38);
    }}
    .at-pressure-badge {{
      font-size: 8px;
      font-weight: 800;
      padding: 2px 6px;
      border-radius: 99px;
      letter-spacing: 0.06em;
    }}
    .at-pressure-badge.Extreme {{ background: rgba(200,30,30,0.20); color: #ff6868; }}
    .at-pressure-badge.High    {{ background: rgba(200,100,15,0.18); color: #ffad50; }}
    .at-pressure-badge.Moderate{{ background: rgba(180,160,20,0.15); color: #f0d050; }}
    .at-pressure-badge.Low     {{ background: rgba(255,255,255,0.07); color: #aaa; }}
    /* ── Stale warning banner ─────────────────────────────────────────────── */
    .stale-warn {{
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 8px;
      background: rgba(200,120,10,0.16);
      border: 1px solid rgba(200,120,10,0.28);
      margin-bottom: 6px;
      font-size: 10px;
      font-weight: 700;
      color: #ffc060;
    }}
    /* ── Weather badge row ────────────────────────────────────────────────── */
    .wx-badge-row {{
      display: flex;
      align-items: center;
      gap: 6px;
      margin-bottom: 2px;
    }}
    .wx-badge {{
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 2px 8px;
      border-radius: 99px;
      font-size: 9px;
      font-weight: 800;
      letter-spacing: 0.06em;
    }}
    .wx-badge.Low      {{ background: rgba(255,255,255,0.07); color: #888; }}
    .wx-badge.Moderate {{ background: rgba(200,150,10,0.18); color: #f0c840; }}
    .wx-badge.Severe   {{ background: rgba(200,30,30,0.20); color: #ff6868; }}
    /* ── AFRI section ─────────────────────────────────────────────────────── */
    .afri-card {{
      padding: 8px 10px;
      border-radius: 10px;
      margin-bottom: 6px;
    }}
    .afri-card.Normal   {{ background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.07); }}
    .afri-card.Elevated {{ background: rgba(180,160,20,0.12); border: 1px solid rgba(180,160,20,0.20); }}
    .afri-card.High     {{ background: rgba(190,90,10,0.14); border: 1px solid rgba(190,90,10,0.22); }}
    .afri-card.Critical {{ background: rgba(180,25,25,0.18); border: 1px solid rgba(180,25,25,0.28); }}
    .afri-card-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 5px;
    }}
    .afri-level-badge {{
      padding: 2px 7px;
      border-radius: 99px;
      font-size: 8px;
      font-weight: 800;
      letter-spacing: 0.06em;
    }}
    .afri-level-badge.Normal   {{ background: rgba(255,255,255,0.09); color: #888; }}
    .afri-level-badge.Elevated {{ background: rgba(180,160,20,0.20); color: #f0d050; }}
    .afri-level-badge.High     {{ background: rgba(190,90,10,0.22); color: #ffad50; }}
    .afri-level-badge.Critical {{ background: rgba(180,25,25,0.26); color: #ff6868; }}
    .afri-score-row {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 5px;
    }}
    .afri-score-num {{
      font-size: 20px;
      font-weight: 900;
      color: #eef4ff;
      line-height: 1;
      flex-shrink: 0;
    }}
    .afri-bar-wrap {{
      flex: 1;
      height: 4px;
      border-radius: 99px;
      background: rgba(255,255,255,0.08);
      overflow: hidden;
    }}
    .afri-bar-fill {{ height: 100%; border-radius: 99px; transition: width 0.5s ease; }}
    .afri-bar-fill.Normal   {{ background: rgba(255,255,255,0.30); }}
    .afri-bar-fill.Elevated {{ background: linear-gradient(90deg,#a09010,#f0d050); }}
    .afri-bar-fill.High     {{ background: linear-gradient(90deg,#be6010,#ffad50); }}
    .afri-bar-fill.Critical {{ background: linear-gradient(90deg,#c01818,#ff6868); }}
    .afri-drivers {{
      font-size: 10px;
      color: rgba(244,247,251,0.50);
      line-height: 1.5;
    }}
    /* ── Fuel / CO₂ card ──────────────────────────────────────────────────── */
    .fuel-card {{
      padding: 8px 10px;
      border-radius: 10px;
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.07);
      margin-bottom: 6px;
    }}
    .fuel-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 5px;
      margin-top: 6px;
    }}
    .fuel-cell {{ text-align: center; }}
    .fuel-cell-label {{
      font-size: 7px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: rgba(244,247,251,0.30);
      margin-bottom: 2px;
    }}
    .fuel-cell-value {{
      font-size: 12px;
      font-weight: 800;
      color: #eef4ff;
    }}
    .fuel-eff-badge {{
      display: inline-block;
      padding: 1px 7px;
      border-radius: 99px;
      font-size: 9px;
      font-weight: 800;
    }}
    .fuel-eff-badge.Excellent {{ background: rgba(30,160,80,0.18); color: #50dc88; }}
    .fuel-eff-badge.Good      {{ background: rgba(30,180,100,0.12); color: #70dca0; }}
    .fuel-eff-badge.Average   {{ background: rgba(180,160,20,0.15); color: #f0d050; }}
    .fuel-eff-badge.High      {{ background: rgba(190,40,40,0.18); color: #ff8080; }}
    .fuel-eff-badge.Unknown   {{ background: rgba(255,255,255,0.07); color: #888; }}
    @media (max-width: 1480px) {{
      .top-center-stack {{
        left: 308px;
        right: 362px;
      }}
      .stat-cell {{
        flex: 1 1 65px;
        padding: 5px 8px;
      }}
      .stat-value {{
        font-size: 12.5px;
      }}
      .search-wrap {{
        width: 100%;
        min-width: 0;
        max-width: 400px;
      }}
      .right-panel {{
        width: 288px;
      }}
    }}
    @media (max-width: 1320px) {{
      .brand-panel {{
        min-width: 250px;
        padding: 11px 14px;
      }}
      .brand-title {{
        font-size: 15px;
      }}
      .top-center-stack {{
        left: 272px;
        right: 342px;
      }}
      .module-strip {{
        gap: 8px;
      }}
      .tool-btn, .module-btn {{
        padding: 5px 8px;
        font-size: 10px;
      }}
      .tools-strip {{
        max-width: 292px;
      }}
      .search-wrap {{
        width: 100%;
        min-width: 0;
        max-width: 340px;
      }}
      .left-rail {{
        width: 238px;
      }}
      .right-panel {{
        width: 270px;
      }}
    }}
    @media (max-height: 900px) {{
      .brand-panel {{
        top: 12px;
        left: 12px;
        padding: 10px 12px;
        min-width: 232px;
      }}
      .brand-mark {{
        width: 38px;
        height: 38px;
      }}
      .brand-title {{
        font-size: 14px;
      }}
      .brand-subtitle {{
        font-size: 10px;
      }}
      .top-center-stack {{
        top: 12px;
        gap: 8px;
      }}
      .stats-pill {{
        padding: 5px 7px;
      }}
      .stat-cell {{
        flex: 1 1 65px;
        padding: 5px 8px;
      }}
      .stat-value {{
        font-size: 13px;
      }}
      .top-actions {{
        top: 12px;
        right: 12px;
        gap: 6px;
      }}
      .tool-btn, .module-btn {{
        padding: 5px 8px;
        border-radius: 10px;
        font-size: 10px;
      }}
      .search-shell {{
        min-width: 220px;\n        height: 40px;
        padding: 0 8px 0 10px;
      }}
      .left-rail {{
        top: 108px;
        left: 12px;
        width: 252px;
        gap: 6px;
        max-height: 500px;
      }}
      .rail-card {{
        padding: 10px 12px 12px;
      }}
      .list-row {{
        font-size: 13px;
        padding: 4px 0;
      }}
      .right-panel {{
        top: 118px;
        right: 12px;
        width: 310px;
        max-height: 520px;
      }}
      .panel-flight {{
        font-size: 24px;
      }}
      .route-strip {{
        padding: 10px 0 12px;
        margin-bottom: 12px;
      }}
      .route-code {{
        font-size: 30px;
      }}
      .panel-metrics {{
        gap: 8px;
      }}
      .panel-card {{
        padding: 10px 12px;
      }}
      .detail-grid {{
        grid-template-columns: 92px 1fr;
        gap: 8px 10px;
        font-size: 13px;
      }}
    }}

    /* ── Alert feed ─────────────────────────────────────────────── */
    #alert-feed {{
      position: absolute;
      top: 132px;
      right: 14px;
      width: 296px;
      max-height: 70vh;
      overflow-y: auto;
      scrollbar-width: thin;
      display: flex;
      flex-direction: column;
      gap: 7px;
      pointer-events: none;
      z-index: 420;
    }}
    .alert-card {{
      pointer-events: auto;
      background: rgba(10,14,22,0.92);
      border: 1px solid rgba(255,255,255,0.10);
      border-left: 3px solid #fbbf24;
      border-radius: 10px;
      padding: 9px 12px;
      font-size: 11px;
      color: rgba(244,247,251,0.90);
      animation: alertSlide 0.3s ease;
      backdrop-filter: blur(12px);
      display: flex;
      align-items: flex-start;
      gap: 9px;
    }}
    .alert-card.critical {{ border-left-color: #f87171; }}
    .alert-card.info     {{ border-left-color: #38bdf8; }}
    .alert-card-icon {{ font-size: 15px; flex-shrink: 0; margin-top: 1px; }}
    .alert-card-body {{ flex: 1; }}
    .alert-card-title {{ font-weight: 700; margin-bottom: 2px; }}
    .alert-card-time  {{ font-size: 10px; color: rgba(244,247,251,0.45); }}
    .alert-card-dismiss {{
      background: none; border: none; cursor: pointer;
      color: rgba(244,247,251,0.35); font-size: 13px; padding: 0 2px;
      flex-shrink: 0; line-height: 1;
    }}
    @keyframes alertSlide {{
      from {{ opacity: 0; transform: translateX(20px); }}
      to   {{ opacity: 1; transform: translateX(0); }}
    }}

    /* ── Schedule popup ──────────────────────────────────────────── */
    #schedule-popup {{
      display: none;
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      z-index: 900;
      width: min(480px, 96vw);
      height: min(520px, 78vh);
      max-height: 78vh;
      overflow: hidden;
      background: rgba(10,14,22,0.97);
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 14px;
      padding: 20px;
      backdrop-filter: blur(20px);
      color: #f4f7fb;
      flex-direction: column;
    }}
    #schedule-popup.visible {{ display: flex; }}
    #schedule-popup-backdrop {{
      display: none;
      position: absolute;
      inset: 0;
      background: rgba(0,0,0,0.55);
      z-index: 899;
      cursor: pointer;
    }}
    #schedule-popup-backdrop.visible {{ display: block; }}
    .sched-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 14px;
      flex-shrink: 0;
    }}
    .sched-title {{ font-size: 15px; font-weight: 800; letter-spacing: 0.02em; }}
    .sched-close {{
      background: none; border: none; color: rgba(244,247,251,0.5);
      font-size: 18px; cursor: pointer; padding: 0 4px; line-height: 1;
    }}
    .sched-tab-strip {{
      display: flex; gap: 8px; margin-bottom: 12px; flex-shrink: 0;
    }}
    #sched-content {{
      overflow-y: auto;
      flex: 1;
      min-height: 0;
      scrollbar-width: thin;
      scrollbar-color: rgba(0,229,255,0.25) transparent;
      padding-right: 4px;
    }}
    #sched-content::-webkit-scrollbar {{
      width: 5px;
    }}
    #sched-content::-webkit-scrollbar-track {{
      background: transparent;
    }}
    #sched-content::-webkit-scrollbar-thumb {{
      background: rgba(0,229,255,0.22);
      border-radius: 4px;
    }}
    #sched-content::-webkit-scrollbar-thumb:hover {{
      background: rgba(0,229,255,0.45);
    }}
    .sched-tab {{
      padding: 5px 14px; border-radius: 20px; font-size: 11px; font-weight: 700;
      border: 1px solid rgba(255,255,255,0.10); background: none;
      color: rgba(244,247,251,0.55); cursor: pointer; transition: all 0.2s;
    }}
    .sched-tab.active {{
      background: rgba(0,229,255,0.12); color: #00e5ff;
      border-color: rgba(0,229,255,0.30);
    }}
    .sched-row {{
      display: grid; grid-template-columns: 52px 1fr 70px 68px;
      gap: 6px; align-items: center;
      padding: 7px 6px; border-bottom: 1px solid rgba(255,255,255,0.05);
      font-size: 12px;
    }}
    .sched-row:last-child {{ border-bottom: none; }}
    .sched-row .fnum {{ font-weight: 700; color: #00e5ff; }}
    .sched-row .dest {{ color: rgba(244,247,251,0.75); }}
    .sched-row .time {{ font-weight: 600; font-variant-numeric: tabular-nums; }}
    .sched-row .status-on-time {{ color: #4ade80; font-weight: 700; font-size: 10px; }}
    .sched-row .status-delayed  {{ color: #f87171; font-weight: 700; font-size: 10px; }}
    .sched-row .status-departed {{ color: rgba(244,247,251,0.45); font-size: 10px; }}
    /* ── Airport overview tab ──────────────────────────────────────── */
    .ap-overview {{
      padding: 4px 2px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }}
    .ap-stat-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 8px;
    }}
    .ap-stat {{
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 10px;
      padding: 10px 6px;
      text-align: center;
    }}
    .ap-stat-val {{
      font-size: 20px;
      font-weight: 800;
      color: #00e5ff;
      line-height: 1.1;
    }}
    .ap-stat-lbl {{
      font-size: 9px;
      color: rgba(244,247,251,0.45);
      margin-top: 3px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .ap-section-title {{
      font-size: 10px;
      font-weight: 700;
      color: rgba(244,247,251,0.4);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 6px;
    }}
    .ap-cong-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 6px;
    }}
    .ap-cong-label {{
      font-size: 11px;
      color: rgba(244,247,251,0.6);
    }}
    .ap-cong-badge {{
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.04em;
    }}
    .ap-cong-bar-wrap {{
      height: 6px;
      background: rgba(255,255,255,0.07);
      border-radius: 4px;
      overflow: hidden;
    }}
    .ap-cong-bar-fill {{
      height: 100%;
      border-radius: 4px;
      transition: width 0.6s ease;
    }}
    .ap-routes-list {{
      display: flex;
      flex-direction: column;
      gap: 5px;
    }}
    .ap-route-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 6px 10px;
      background: rgba(255,255,255,0.04);
      border-radius: 8px;
      font-size: 11px;
    }}
    .ap-route-iata {{
      font-weight: 700;
      color: #00e5ff;
      min-width: 34px;
    }}
    .ap-route-city {{
      flex: 1;
      color: rgba(244,247,251,0.7);
      padding: 0 8px;
    }}
    .ap-route-count {{
      font-weight: 700;
      color: rgba(244,247,251,0.55);
      font-size: 10px;
    }}
    .ap-pressure-row {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .ap-pressure-badge {{
      padding: 3px 10px;
      border-radius: 20px;
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.05em;
    }}
    .ap-pressure-high {{
      background: rgba(248,113,113,0.15);
      color: #f87171;
      border: 1px solid rgba(248,113,113,0.3);
    }}
    .ap-pressure-medium {{
      background: rgba(251,191,36,0.12);
      color: #fbbf24;
      border: 1px solid rgba(251,191,36,0.25);
    }}
    .ap-pressure-low {{
      background: rgba(74,222,128,0.1);
      color: #4ade80;
      border: 1px solid rgba(74,222,128,0.25);
    }}
    .ap-pressure-normal {{
      background: rgba(0,229,255,0.08);
      color: #00e5ff;
      border: 1px solid rgba(0,229,255,0.2);
    }}
    .ap-pressure-lbl {{
      font-size: 11px;
      color: rgba(244,247,251,0.45);
    }}

    /* ── Alerts side panel ───────────────────────────────────────── */
    .alerts-side-panel {{
      display: none;
      position: absolute;
      top: 0; right: 0; bottom: 0;
      width: 340px;
      z-index: 800;
      background: rgba(10,14,22,0.97);
      border-left: 1px solid rgba(255,255,255,0.10);
      backdrop-filter: blur(20px);
      flex-direction: column;
      overflow: hidden;
    }}
    .alerts-side-panel.visible {{ display: flex; }}
    .asp-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 16px 16px 10px;
      border-bottom: 1px solid rgba(255,255,255,0.07);
      flex-shrink: 0;
    }}
    .asp-title {{
      font-size: 13px;
      font-weight: 800;
      color: #f4f7fb;
      letter-spacing: 0.02em;
    }}
    .asp-clear-btn {{
      background: rgba(248,113,113,0.12);
      border: 1px solid rgba(248,113,113,0.25);
      color: #f87171;
      font-size: 10px;
      font-weight: 700;
      padding: 3px 8px;
      border-radius: 6px;
      cursor: pointer;
    }}
    .asp-close-btn {{
      background: none;
      border: none;
      color: rgba(244,247,251,0.5);
      font-size: 16px;
      cursor: pointer;
      padding: 0 2px;
    }}
    .asp-body {{
      flex: 1;
      overflow-y: auto;
      padding: 10px 12px;
      display: flex;
      flex-direction: column;
      gap: 7px;
      scrollbar-width: thin;
      scrollbar-color: rgba(0,229,255,0.2) transparent;
    }}
    .asp-empty {{
      text-align: center;
      font-size: 12px;
      color: rgba(244,247,251,0.3);
      padding: 40px 0;
    }}
    .asp-group-label {{
      font-size: 9px;
      font-weight: 700;
      color: rgba(244,247,251,0.35);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      padding: 4px 0 2px;
    }}
    .asp-item {{
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.08);
      border-left: 3px solid #fbbf24;
      border-radius: 8px;
      padding: 9px 12px;
      font-size: 11px;
      color: rgba(244,247,251,0.88);
      display: flex;
      align-items: flex-start;
      gap: 8px;
    }}
    .asp-item.critical {{ border-left-color: #f87171; }}
    .asp-item.info     {{ border-left-color: #38bdf8; }}
    .asp-item-icon {{ font-size: 13px; flex-shrink: 0; margin-top: 1px; }}
    .asp-item-body {{ flex: 1; }}
    .asp-item-title {{ font-weight: 700; margin-bottom: 2px; }}
    .asp-item-sub {{ color: rgba(244,247,251,0.55); font-size: 10px; }}
    .asp-item-time {{ font-size: 9px; color: rgba(244,247,251,0.35); margin-top: 3px; }}
    .asp-item-dismiss {{
      background: none; border: none;
      color: rgba(244,247,251,0.3);
      cursor: pointer; font-size: 12px; padding: 0;
    }}
    .asp-item-dismiss:hover {{ color: rgba(244,247,251,0.7); }}

    /* ── Filter panel ────────────────────────────────────────────── */
    .filter-panel {{
      display: none;
      position: absolute;
      top: 132px;
      right: 14px;
      width: 240px;
      z-index: 780;
      background: rgba(10,14,22,0.97);
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 12px;
      padding: 14px 16px;
      backdrop-filter: blur(20px);
      color: #f4f7fb;
    }}
    .filter-panel.visible {{ display: block; }}
    .fp-title {{
      font-size: 12px;
      font-weight: 800;
      margin-bottom: 12px;
      color: rgba(244,247,251,0.9);
      letter-spacing: 0.03em;
    }}
    .fp-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 4px;
    }}
    .fp-lbl {{
      font-size: 10px;
      font-weight: 700;
      color: rgba(244,247,251,0.55);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .fp-val {{
      font-size: 10px;
      font-weight: 700;
      color: #00e5ff;
      font-variant-numeric: tabular-nums;
    }}
    .fp-dual-wrap {{
      position: relative;
      height: 20px;
      display: flex;
      flex-direction: column;
      gap: 3px;
    }}
    .fp-range {{
      width: 100%;
      accent-color: #00e5ff;
      cursor: pointer;
      height: 4px;
    }}
    .fp-reset-btn {{
      margin-top: 12px;
      width: 100%;
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.12);
      color: rgba(244,247,251,0.6);
      font-size: 10px;
      font-weight: 700;
      padding: 5px;
      border-radius: 6px;
      cursor: pointer;
    }}
    .fp-active-badge {{
      font-size: 9px;
      font-weight: 700;
      background: rgba(0,229,255,0.15);
      color: #00e5ff;
      border-radius: 4px;
      padding: 1px 5px;
      margin-left: 4px;
    }}

    /* ── Airline scorecard expand ────────────────────────────────── */
    .airline-scorecard {{
      margin-top: 8px;
      padding: 8px 10px;
      background: rgba(0,229,255,0.05);
      border-radius: 8px;
      border: 1px solid rgba(0,229,255,0.12);
      font-size: 11px;
      display: none;
    }}
    .airline-scorecard.open {{ display: block; }}
    .al-sc-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 3px 0;
      border-bottom: 1px solid rgba(255,255,255,0.05);
    }}
    .al-sc-row:last-child {{ border-bottom: none; }}
    .al-sc-lbl {{ color: rgba(244,247,251,0.5); font-size: 10px; }}
    .al-sc-val {{ font-weight: 700; color: #f4f7fb; }}

    /* ── Route demand chart ──────────────────────────────────────── */
    .route-demand-card {{
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 10px;
      padding: 10px 12px;
      margin-top: 8px;
    }}
    .route-demand-title {{
      font-size: 10px;
      font-weight: 700;
      color: rgba(244,247,251,0.45);
      text-transform: uppercase;
      letter-spacing: 0.07em;
      margin-bottom: 8px;
    }}
    .route-demand-bars {{
      display: flex;
      align-items: flex-end;
      gap: 3px;
      height: 40px;
    }}
    .rdb-slot {{
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 2px;
    }}
    .rdb-bar {{
      width: 100%;
      border-radius: 2px 2px 0 0;
      background: rgba(0,229,255,0.45);
      min-height: 2px;
      transition: height 0.3s ease;
    }}
    .rdb-bar.active {{ background: #00e5ff; }}
    .rdb-lbl {{
      font-size: 8px;
      color: rgba(244,247,251,0.3);
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }}

    /* ── Follow mode button ──────────────────────────────────────── */
    .follow-btn {{
      font-size: 10px;
      font-weight: 700;
      padding: 3px 9px;
      border-radius: 12px;
      border: 1px solid rgba(0,229,255,0.3);
      background: none;
      color: rgba(0,229,255,0.7);
      cursor: pointer;
      transition: all 0.2s;
    }}
    .follow-btn.active {{
      background: rgba(0,229,255,0.15);
      color: #00e5ff;
      border-color: #00e5ff;
    }}

    /* ── Keyboard shortcuts modal ────────────────────────────────── */
    #shortcuts-modal {{
      display: none;
      position: absolute;
      top: 50%; left: 50%;
      transform: translate(-50%, -50%);
      z-index: 950;
      width: min(400px, 94vw);
      background: rgba(10,14,22,0.97);
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 14px;
      padding: 20px 24px;
      backdrop-filter: blur(20px);
      color: #f4f7fb;
    }}
    #shortcuts-modal.visible {{ display: block; }}
    #shortcuts-backdrop {{
      display: none;
      position: absolute; inset: 0;
      background: rgba(0,0,0,0.50);
      z-index: 949; cursor: pointer;
    }}
    #shortcuts-backdrop.visible {{ display: block; }}
    .shortcuts-title {{
      font-size: 14px; font-weight: 800; margin-bottom: 14px;
      display: flex; justify-content: space-between; align-items: center;
    }}
    .shortcuts-close {{
      background: none; border: none; color: rgba(244,247,251,0.45);
      font-size: 18px; cursor: pointer; padding: 0;
    }}
    .shortcut-row {{
      display: flex; align-items: center; justify-content: space-between;
      padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.05);
      font-size: 12px; color: rgba(244,247,251,0.75);
    }}
    .shortcut-row:last-child {{ border-bottom: none; }}
    .shortcut-key {{
      display: inline-flex; align-items: center; gap: 4px;
    }}
    .kbd {{
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(255,255,255,0.14);
      border-radius: 4px; padding: 2px 7px;
      font-family: monospace; font-size: 11px;
      color: #f4f7fb;
    }}
    @media print {{
      body * {{ visibility: hidden !important; }}
      #flight-board-overlay, #flight-board-overlay * {{ visibility: visible !important; }}
      #flight-board-overlay {{
        position: fixed !important; top: 0 !important; left: 0 !important;
        width: 100% !important; height: auto !important;
        background: #fff !important; color: #000 !important;
      }}
      .fids-table {{ color: #000 !important; }}
      .fids-close-btn, #fids-print-btn {{ display: none !important; }}
    }}
    /* ── Flight board mobile (≤600px) ──────────────────────────────── */
    @media (max-width: 600px) {{
      /* Toolbar: stack title on top, filters below */
      .fids-toolbar {{
        flex-wrap: wrap;
        gap: 6px;
        padding: 10px 12px 8px;
      }}
      .fids-title {{
        width: 100%;
        font-size: 11px;
      }}
      .fids-filter-group {{
        flex: 1;
        gap: 4px;
      }}
      .fids-filter-btn {{
        font-size: 10px;
        padding: 5px 10px;
      }}
      /* Hide print button on mobile */
      #fids-print-btn {{ display: none; }}
      /* Make close button bigger */
      .fids-close-btn {{
        width: 38px;
        height: 38px;
        font-size: 22px;
      }}
      /* Hide hint — wastes vertical space */
      .fids-col-hint {{ display: none; }}
      /* Hide non-essential columns: Airline, STA, ETA, AFRI, Phase, Wx, Speed */
      .fids-table thead th:nth-child(2),
      .fids-table tbody td:nth-child(2),
      .fids-table thead th:nth-child(5),
      .fids-table tbody td:nth-child(5),
      .fids-table thead th:nth-child(6),
      .fids-table tbody td:nth-child(6),
      .fids-table thead th:nth-child(9),
      .fids-table tbody td:nth-child(9),
      .fids-table thead th:nth-child(10),
      .fids-table tbody td:nth-child(10),
      .fids-table thead th:nth-child(11),
      .fids-table tbody td:nth-child(11),
      .fids-table thead th:nth-child(12),
      .fids-table tbody td:nth-child(12) {{
        display: none;
      }}
      /* Tighter cells for remaining: Flight, Origin, Dest, Delay, Status */
      .fids-table thead th,
      .fids-row td {{
        padding: 6px 8px;
        font-size: 11px;
      }}
      .fids-flight {{ font-size: 12px; }}
      .fids-footer {{ font-size: 10px; padding: 5px 12px; }}
      /* Ensure table-wrap fills remaining height */
      .fids-table-wrap {{
        -webkit-overflow-scrolling: touch;
        touch-action: pan-y;
        overscroll-behavior-y: contain;
      }}
    }}

    /* ── Phone layout (≤600px) ──────────────────────────────────────── */
    @media (max-width: 600px) {{
      html, body {{ height: 100vh !important; overflow: hidden; }}
      #stage {{ height: 100vh !important; }}
      /* Brand: icon-only pill */
      .brand-panel {{
        min-width: 0;
        width: 52px;
        padding: 8px;
        border-radius: 16px;
        top: 8px;
        left: 8px;
        justify-content: center;
        gap: 0;
      }}
      .brand-copy {{ display: none; }}
      .brand-mark {{ width: 36px; height: 36px; border-radius: 12px; flex-shrink: 0; overflow: hidden; }}
      /* Module + search strip */
      .top-center-stack {{
        top: 8px;
        left: 68px;
        right: 78px;
        align-items: stretch;
        gap: 5px;
      }}
      #top-stats {{ display: none; }}
      .module-strip {{
        overflow-x: auto;
        overflow-y: hidden;
        scrollbar-width: none;
        -webkit-overflow-scrolling: touch;
        touch-action: pan-x;
        overscroll-behavior-x: contain;
        flex-wrap: nowrap;
        gap: 4px;
        justify-content: flex-start;
      }}
      .module-strip::-webkit-scrollbar {{ display: none; }}
      .module-btn {{
        white-space: nowrap;
        flex-shrink: 0;
        padding: 6px 11px;
        font-size: 11px;
        border-radius: 10px;
        min-width: 0;
      }}
      .top-center-stack .search-wrap {{
        max-width: 100%;
        width: 100%;
      }}
      .search-shell {{
        height: 38px;
        min-width: 0;
        border-radius: 12px;
      }}
      /* Top-actions: freshness badge only */
      .top-actions {{
        top: 8px;
        right: 8px;
        gap: 0;
        align-items: flex-end;
      }}
      .ac-filter-strip,
      .tools-strip {{ display: none; }}
      #freshness-badge {{
        font-size: 10px;
        padding: 5px 9px;
        border-radius: 12px;
      }}
      /* Left rail: horizontal bottom drawer */
      .left-rail {{
        top: auto;
        bottom: 0;
        left: 0;
        right: 0;
        width: 100%;
        max-height: 280px;
        flex-direction: row;
        overflow-x: auto;
        overflow-y: hidden;
        -webkit-overflow-scrolling: touch;
        touch-action: pan-x;
        overscroll-behavior-x: contain;
        border-radius: 16px 16px 0 0;
        padding: 10px 8px 12px;
        gap: 8px;
        max-width: 100%;
        background: linear-gradient(180deg, rgba(10,14,22,0.98), rgba(8,12,18,0.98));
        border-top: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 -12px 32px rgba(0,0,0,0.4);
      }}
      .left-rail .rail-card {{
        flex-shrink: 0;
        width: 220px;
        min-width: 180px;
      }}
      /* Right panel: full-width bottom sheet */
      .right-panel {{
        /* Bottom sheet: anchored to the bottom of the iframe.
           Works correctly because DASHBOARD_HEIGHT = 680px on mobile,
           so bottom:0 is within the visible phone screen. */
        top: auto !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        width: 100% !important;
        max-width: 100% !important;
        height: 72% !important;
        max-height: 72% !important;
        min-height: 0;
        border-radius: 20px 20px 0 0 !important;
        border-left: none !important;
        border-right: none !important;
        border-bottom: none !important;
        overflow-y: scroll !important;
        overflow-x: hidden !important;
        touch-action: pan-y !important;
        overscroll-behavior-y: contain !important;
        -webkit-overflow-scrolling: touch !important;
        padding-bottom: calc(env(safe-area-inset-bottom, 0px) + 32px) !important;
      }}
      /* Drag handle pill */
      .right-panel::before {{
        content: '';
        display: block;
        width: 40px;
        height: 4px;
        border-radius: 999px;
        background: rgba(255,255,255,0.25);
        margin: 10px auto 6px;
        flex-shrink: 0;
        position: sticky;
        top: 0;
        z-index: 1;
      }}
      /* Make inner content fit full width */
      .panel-inner {{
        padding: 12px 14px 24px !important;
      }}
      .panel-flight-hero {{
        font-size: 28px !important;
      }}
      /* 2-col metrics grid on phone */
      .metrics-grid-4 {{
        grid-template-columns: 1fr 1fr !important;
      }}
      /* Route display full width */
      .route-display {{
        gap: 6px !important;
      }}
      .route-apt-code {{
        font-size: 26px !important;
      }}
      /* Delay section full width */
      .delay-row {{
        flex-direction: column !important;
        gap: 8px !important;
      }}
      /* Make close button bigger and easier to tap */
      .panel-close {{
        width: 40px !important;
        height: 40px !important;
        font-size: 22px !important;
      }}
      /* Sparkline full width */
      .sparkline-card {{
        margin: 0 !important;
      }}
      /* Alert feed: full width */
      #alert-feed {{
        right: 8px;
        left: 8px;
        top: 108px;
        width: auto;
        max-width: calc(100% - 16px);
      }}
      /* Filter and alerts panels: full width */
      .filter-panel, .alerts-side-panel {{
        left: 8px;
        right: 8px;
        width: auto;
        max-width: calc(100% - 16px);
        touch-action: pan-y;
        overscroll-behavior-y: contain;
      }}
    }}
  </style>
</head>
<body>
  <div id="stage">
  <div id="map"></div>
  <!-- Alert feed (top-right) -->
  <div id="alert-feed"></div>
  <!-- Schedule popup backdrop + modal -->
  <div id="schedule-popup-backdrop"></div>
  <div id="schedule-popup"></div>
  <!-- Keyboard shortcuts modal -->
  <div id="shortcuts-backdrop"></div>
  <div id="shortcuts-modal">
    <div class="shortcuts-title">
      Keyboard Shortcuts
      <button class="shortcuts-close" id="shortcuts-close-btn" type="button">&#x2715;</button>
    </div>
    <div class="shortcut-row"><span>Open this help</span><span class="shortcut-key"><span class="kbd">?</span></span></div>
    <div class="shortcut-row"><span>Search flights</span><span class="shortcut-key"><span class="kbd">/</span> or <span class="kbd">F</span></span></div>
    <div class="shortcut-row"><span>Dismiss / close panel</span><span class="shortcut-key"><span class="kbd">Esc</span></span></div>
    <div class="shortcut-row"><span>Switch to Live Ops</span><span class="shortcut-key"><span class="kbd">1</span></span></div>
    <div class="shortcut-row"><span>Switch to Flight Board</span><span class="shortcut-key"><span class="kbd">2</span></span></div>
    <div class="shortcut-row"><span>Switch to Routes</span><span class="shortcut-key"><span class="kbd">3</span></span></div>
    <div class="shortcut-row"><span>Switch to Airport Traffic</span><span class="shortcut-key"><span class="kbd">4</span></span></div>
    <div class="shortcut-row"><span>Switch to Alerts</span><span class="shortcut-key"><span class="kbd">5</span></span></div>
    <div class="shortcut-row"><span>Toggle flight trails</span><span class="shortcut-key"><span class="kbd">T</span></span></div>
    <div class="shortcut-row"><span>Toggle weather radar</span><span class="shortcut-key"><span class="kbd">W</span></span></div>
    <div class="shortcut-row"><span>Toggle heatmap</span><span class="shortcut-key"><span class="kbd">M</span></span></div>
    <div class="shortcut-row"><span>Fit map to all flights</span><span class="shortcut-key"><span class="kbd">Home</span></span></div>
  </div>
  <div id="overlay-root">
    <div class="brand-panel glass-panel">
      <div class="brand-mark"><svg width="42" height="42" viewBox="24 30 300 300" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="mbx-ic-a" x1="32" y1="40" x2="260" y2="300" gradientUnits="userSpaceOnUse"><stop stop-color="#FFC14D"/><stop offset="1" stop-color="#FF9F1A"/></linearGradient><linearGradient id="mbx-ic-d" x1="0" y1="0" x2="320" y2="320" gradientUnits="userSpaceOnUse"><stop stop-color="#121A26"/><stop offset="1" stop-color="#0A1018"/></linearGradient></defs><rect x="24" y="30" width="300" height="300" rx="72" fill="url(#mbx-ic-d)"/><path d="M112 246L176 88H228L292 246H239L224 207H180L166 246H112ZM194 163H210L202 140L194 163Z" fill="url(#mbx-ic-a)"/><path d="M94 212C120 167 163 136 219 120C243 113 267 110 291 110" stroke="#5CD2FF" stroke-width="10" stroke-linecap="round" stroke-dasharray="2 18"/><path d="M274 98L309 111L283 139" stroke="#5CD2FF" stroke-width="10" stroke-linecap="round" stroke-linejoin="round"/></svg></div>
      <div class="brand-copy">
        <div class="brand-title">Aviation Intelligence</div>
        <div class="brand-subtitle">Live India Airspace Analytics</div>
      </div>
    </div>
    <div class="top-center-stack">
      <div class="stats-pill glass-panel" id="top-stats"></div>
      <div class="module-strip">
        {module_nav_html}
      </div>
      <div class="search-wrap" id="search-wrap">
        <div class="search-shell">
          <span class="search-icon">&#9906;</span>
          <input id="flight-search-input" class="search-input" type="text"
            value="{selected_flight or ''}"
            placeholder="Search flights, routes, airlines..."
            autocomplete="off" spellcheck="false">
          <button id="search-clear-btn" class="search-clear" type="button" aria-label="Clear">&#x2715;</button>
          <span class="search-kbd">&#x23CE;</span>
        </div>
        <div class="search-dropdown" id="search-dropdown"></div>
      </div>
    </div>
    <div class="top-actions">
      <div class="tools-row-status">
        <span id="freshness-badge" class="fresh"><span id="freshness-dot"></span><span id="freshness-text">Live</span></span>
        <div class="ac-filter-strip">
          <button class="ac-filter-btn active" data-ac-filter="all">All</button>
          <button class="ac-filter-btn" data-ac-filter="wide">Wide</button>
          <button class="ac-filter-btn" data-ac-filter="narrow">Narrow</button>
          <button class="ac-filter-btn" data-ac-filter="regional">Regional</button>
        </div>
      </div>
      <div class="tools-strip">
        <button class="tool-btn" type="button" id="btn-weather" data-action="toggle-weather" title="Rain radar overlay">&#x26C8; Weather</button>
        <button class="tool-btn" type="button" id="btn-heatmap" data-action="toggle-heatmap" title="Traffic density heatmap">&#x1F525; Heatmap</button>
        <button class="tool-btn" type="button" id="btn-delay-map" data-action="toggle-delay-map" title="Airport delay rate heatmap">&#x1F534; Delay Map</button>
        <button class="tool-btn" type="button" id="btn-filters" data-action="toggle-filters" title="Speed / altitude filters">&#x2699; Filters</button>
        <button class="tool-btn" type="button" id="btn-alerts-panel" data-action="toggle-alerts-panel" title="Alert log">&#x1F6A8; Alerts</button>
        <button class="tool-btn" type="button" data-action="fit-map" title="Fit map to all flights (Home)">&#x2922; Fit</button>
        <button class="tool-btn" type="button" data-action="locate">&#x25CE; Locate</button>
        <button class="tool-btn" type="button" data-action="shortcuts" title="Keyboard shortcuts (?)">?</button>
      </div>
    </div>
    <div class="left-rail" id="left-rail"></div>
    <div class="right-panel glass-panel" id="right-panel"></div>
    <!-- Transparent shield: sits above the map, below the panel.
         Blocks ALL touch/click on the map while a mobile bottom sheet is open.
         Tapping the shield closes the panel (feels like a backdrop dismiss). -->
    <div id="map-shield"></div>
  </div>
  <!-- Alerts side panel -->
  <div id="alerts-side-panel" class="alerts-side-panel glass-panel">
    <div class="asp-header">
      <span class="asp-title">&#x1F6A8; Alert Log</span>
      <div style="display:flex;gap:6px;align-items:center;">
        <button class="asp-clear-btn" id="asp-clear-btn" type="button">Clear all</button>
        <button class="asp-close-btn" id="asp-close-btn" type="button">&#x2715;</button>
      </div>
    </div>
    <div id="asp-body" class="asp-body"></div>
  </div>
  <!-- Alt/Speed filter panel -->
  <div id="filter-panel" class="filter-panel glass-panel">
    <div class="fp-title">&#x2699; Filters</div>
    <div class="fp-row">
      <label class="fp-lbl">Altitude</label>
      <span class="fp-val" id="fp-alt-val">0 – 45,000 ft</span>
    </div>
    <div class="fp-dual-wrap">
      <input type="range" class="fp-range" id="fp-alt-min" min="0" max="45000" step="500" value="0">
      <input type="range" class="fp-range" id="fp-alt-max" min="0" max="45000" step="500" value="45000">
    </div>
    <div class="fp-row" style="margin-top:10px;">
      <label class="fp-lbl">Speed</label>
      <span class="fp-val" id="fp-spd-val">0 – 600 kts</span>
    </div>
    <div class="fp-dual-wrap">
      <input type="range" class="fp-range" id="fp-spd-min" min="0" max="600" step="10" value="0">
      <input type="range" class="fp-range" id="fp-spd-max" min="0" max="600" step="10" value="600">
    </div>
    <button class="fp-reset-btn" id="fp-reset-btn" type="button">Reset</button>
  </div>

  <!-- Flight Board (FIDS) — direct child of #stage so pointer-events work -->
  <div id="flight-board-overlay">
    <div class="fids-toolbar">
      <span class="fids-title">&#9992;&#xFE0F;&nbsp; Flight Board &mdash; Live Indian Airspace</span>
      <div class="fids-filter-group">
        <button class="fids-filter-btn active" data-fids-filter="all" type="button">All Flights</button>
        <button class="fids-filter-btn" data-fids-filter="arr" type="button">Arrivals</button>
        <button class="fids-filter-btn" data-fids-filter="dep" type="button">Departures</button>
      </div>
      <div class="fids-toolbar-right">
        <div class="fids-live-badge"><span class="fids-live-dot"></span>Live</div>
        <button class="fids-filter-btn" id="fids-print-btn" type="button" title="Print flight board">&#x2399; Print</button>
        <button class="fids-close-btn" id="fids-close-btn" type="button" title="Back to map">&times;</button>
      </div>
    </div>
    <div class="fids-col-hint">Click column headers to sort</div>
    <div class="fids-table-wrap">
      <table class="fids-table">
        <thead>
          <tr>
            <th class="sortable" data-fids-sort="flight">Flight</th>
            <th class="sortable" data-fids-sort="airline">Airline</th>
            <th class="sortable" data-fids-sort="dep">Origin</th>
            <th class="sortable" data-fids-sort="arr">Destination</th>
            <th class="sortable" data-fids-sort="sta">STA&nbsp;(IST)</th>
            <th>ETA&nbsp;(IST)</th>
            <th class="sortable" data-fids-sort="delayed">Delay</th>
            <th>Status</th>
            <th class="sortable" data-fids-sort="afri">AFRI</th>
            <th>Phase</th>
            <th>Wx</th>
            <th class="sortable" data-fids-sort="speed">Speed</th>
          </tr>
        </thead>
        <tbody id="fids-tbody"></tbody>
      </table>
    </div>
    <div class="fids-footer">
      <span id="fids-count">0 flights</span>
      <span>Click any row to fly to aircraft on map</span>
    </div>
  </div>
  <div id="splash">
    <div class="splash-card">
      <div class="splash-title">Aviation Intelligence</div>
      <div class="skeleton small"></div>
      <div class="skeleton medium"></div>
      <div class="skeleton large"></div>
      <div class="skeleton large"></div>
    </div>
  </div>
  <div id="error-box"></div>
  <div class="credit-bar">
    <a href="https://github.com/rinkuujangraa/indian-avitation-intelligence" target="_blank" rel="noopener">GitHub</a>
    <span class="credit-sep"></span>
    <a href="https://linkedin.com/in/rinkuu-jangra" target="_blank" rel="noopener">LinkedIn</a>
    <span class="credit-sep"></span>
    <a href="https://web-production-39c38.up.railway.app" target="_blank" rel="noopener">Built by Rinku</a>
  </div>
  </div>

  <script>
    const MAPBOX_TOKEN = {json.dumps(mapbox_token)};
    const FLIGHTS = {flights_json};
    const AIRPORTS = {airports_json};
    const ICON_TEMPLATES = {icon_templates_json};
    const SELECTED_QUERY = {selected_query};
    let ACTIVE_MODULE = {active_module_json};
    const MAX_BOUNDS = {max_bounds};
    const ANOMALIES = {anomalies_json};
    const AIRPORT_METRICS = {airport_metrics_json};
    const SCHEDULE_PRESSURE = {schedule_pressure_json};
    const TOP_AIRLINES = {top_airlines_json};
    const PAGE_FETCHED_TS = {int(_time.time())};
    let _dataFetchTs = PAGE_FETCHED_TS;  // updated on live data pushes
    const TRAILS = {trails_json};
    const AIRPORT_LOOKUP = Object.fromEntries(AIRPORTS.map((airport) => [airport.iata, airport]));
    const INDIAN_IATA_SET = new Set(AIRPORTS.map((a) => a.iata));
    let ACTIVE_SELECTION = null;
    let _mapRef = null;
    let _dpThreshold = 0;   // delay prediction slider state — survives rail re-renders
    let ROUTE_SOURCE_READY = false;
    let USER_LOCATION_MARKER = null;

    // ── Speed / altitude filter state ─────────────────────────────────────────
    let _altMin = 0, _altMax = 45000;
    let _spdMin = 0, _spdMax = 600;

    // ── Camera follow mode ───────────────────────────────────────────────────
    let _cameraFollow = false;

    // ── Live update bridge ──────────────────────────────────────────────────────
    // External code (e.g. a sibling Streamlit component) can call
    // window.updateFlightData(newFlights) to push fresh flight data into the
    // already-loaded map without a full page reload.
    let _pendingFlightUpdate = null;
    window.AVIATION_MAP_FRAME = true;

    // ── Position interpolation ────────────────────────────────────────────────
    // When new positions arrive we smoothly animate each aircraft from its
    // previous rendered position to the new one over INTERP_MS milliseconds.
    // Between updates the aircraft continues to dead-reckon forward using its
    // last known heading and speed so movement never stalls.
    const INTERP_MS = 9500;        // slightly under 10 s update interval
    const KTS_TO_DEG_PER_MS = 1.852 / 111.32 / 3600000; // knots → deg-lat per ms
    const _interpFrom = {{}};       // icao24 → {{lat, lng, heading}}
    const _interpTo   = {{}};       // icao24 → {{lat, lng, heading}}
    let   _interpStartTs = 0;      // timestamp when current tween began
    let   _rafId = null;           // requestAnimationFrame handle

    function _shortAngleDiff(a, b) {{
      // Shortest signed angular distance a→b in [-180, 180]
      let d = ((b - a) % 360 + 540) % 360 - 180;
      return d;
    }}

    function _interpCoords(icao24) {{
      // Returns {{lat, lng, heading}} — interpolated or dead-reckoned position.
      const from = _interpFrom[icao24];
      const to   = _interpTo[icao24];
      if (!from || !to) return null;
      const elapsed = Date.now() - _interpStartTs;
      const t = Math.min(elapsed / INTERP_MS, 1.0);   // 0..1 eased
      const ease = t < 0.5 ? 2*t*t : -1+(4-2*t)*t;   // ease-in-out quad
      const lat = from.lat + (to.lat - from.lat) * ease;
      const lng = from.lng + (to.lng - from.lng) * ease;
      const hdg = from.heading + _shortAngleDiff(from.heading, to.heading) * ease;
      return {{ lat, lng, heading: (hdg + 360) % 360 }};
    }}

    function _applyInterpolation() {{
      // Write interpolated lat/lng/heading back onto each FLIGHTS entry
      // so all other code (panel, popups, nearest-airport) sees live positions.
      FLIGHTS.forEach(function(f) {{
        const pos = _interpCoords(f.icao24);
        if (!pos) return;
        f._iLat     = pos.lat;
        f._iLng     = pos.lng;
        f._iHeading = pos.heading;
      }});
    }}

    function _startInterpLoop() {{
      if (_rafId) cancelAnimationFrame(_rafId);
      _interpStartTs = Date.now();
      function _tick() {{
        _applyInterpolation();
        // Redraw the map source with interpolated positions
        if (_mapRef && _mapRef.isStyleLoaded()) {{
          const src = _mapRef.getSource('flights-src');
          if (src) src.setData(buildFlightsGeoJson(ACTIVE_SELECTION ? ACTIVE_SELECTION.icao24 : null));
          // Camera follow mode
          if (_cameraFollow && ACTIVE_SELECTION) {{
            const ipos = _interpCoords(ACTIVE_SELECTION.icao24);
            if (ipos) {{
              _mapRef.easeTo({{
                center: [ipos.lng, ipos.lat],
                duration: 200,
                essential: false,
              }});
            }}
          }}
        }}
        if (Date.now() - _interpStartTs < INTERP_MS + 2000) {{
          _rafId = requestAnimationFrame(_tick);
        }} else {{
          // Tween finished — switch to slow dead-reckoning at ~2 fps to keep
          // aircraft moving forward until the next data push arrives.
          _rafId = null;
          _deadReckonLoop();
        }}
      }}
      _rafId = requestAnimationFrame(_tick);
    }}

    function _deadReckonLoop() {{
      // After the tween completes, extrapolate positions at 2 fps so aircraft
      // don't freeze on screen while waiting for the next API call.
      let _drLast = Date.now();
      function _drTick() {{
        const now = Date.now();
        const dtMs = now - _drLast;
        _drLast = now;
        FLIGHTS.forEach(function(f) {{
          const spd = Number(f.speed_kts || 0);
          if (spd < 30) return;   // on ground / parked — don't move
          const hdgRad = ((f._iHeading !== undefined ? f._iHeading : (f.heading || 0)) * Math.PI) / 180;
          const distDeg = spd * KTS_TO_DEG_PER_MS * dtMs;
          f._iLat = (f._iLat !== undefined ? f._iLat : f.lat) + Math.cos(hdgRad) * distDeg;
          f._iLng = (f._iLng !== undefined ? f._iLng : f.lng) + Math.sin(hdgRad) * distDeg / Math.cos(((f._iLat || f.lat) * Math.PI) / 180);
        }});
        if (_mapRef && _mapRef.isStyleLoaded()) {{
          const src = _mapRef.getSource('flights-src');
          if (src) src.setData(buildFlightsGeoJson(ACTIVE_SELECTION ? ACTIVE_SELECTION.icao24 : null));
        }}
        _rafId = setTimeout(_drTick, 500);   // ~2 fps is enough when dead-reckoning
      }}
      _rafId = setTimeout(_drTick, 500);
    }}

    window.updateFlightData = function(newFlights) {{
      // Snapshot current interpolated positions as the "from" keyframe
      FLIGHTS.forEach(function(f) {{
        _interpFrom[f.icao24] = {{
          lat:     f._iLat !== undefined ? f._iLat : f.lat,
          lng:     f._iLng !== undefined ? f._iLng : f.lng,
          heading: f._iHeading !== undefined ? f._iHeading : (f.heading || 0),
        }};
      }});

      // Splice FLIGHTS in-place so all existing closures still reference the array
      FLIGHTS.length = 0;
      newFlights.forEach(function(f) {{ FLIGHTS.push(f); }});
      _dataFetchTs = Math.floor(Date.now() / 1000);

      // Record "to" keyframe — new API positions.
      // Guard against backwards animation: if the new API position is behind the
      // current dead-reckoned position along the flight's heading, the aircraft
      // would visually reverse.  We detect this with a dot-product check and, when
      // it fires, keep the current interpolated position as the tween target so the
      // aircraft holds its position and only re-syncs heading and speed.
      FLIGHTS.forEach(function(f) {{
        const from = _interpFrom[f.icao24];
        // Seed _iLat/_iLng so first frame doesn't jump if this is a new aircraft
        if (!from) {{
          _interpFrom[f.icao24] = {{ lat: f.lat, lng: f.lng, heading: f.heading || 0 }};
          _interpTo[f.icao24]   = {{ lat: f.lat, lng: f.lng, heading: f.heading || 0 }};
          f._iLat     = f.lat;
          f._iLng     = f.lng;
          f._iHeading = f.heading || 0;
          return;
        }}

        const newLat = f.lat;
        const newLng = f.lng;
        const newHdg = f.heading || 0;

        // Vector from current interp position to new API position
        const dLat = newLat - from.lat;
        const dLng = newLng - from.lng;

        // Current heading as unit vector (lat/lng plane, lng corrected for latitude)
        const hdgRad = (from.heading * Math.PI) / 180;
        const cosLat = Math.cos((from.lat * Math.PI) / 180) || 1;
        const hLat = Math.cos(hdgRad);
        const hLng = Math.sin(hdgRad) * cosLat;

        // Dot product of displacement with heading vector
        const dot = dLat * hLat + dLng * hLng;

        // Great-circle distance² (rough, in degrees²) to detect large teleports
        const dist2 = dLat * dLat + dLng * dLng;

        if (dot < 0 && dist2 < 0.25) {{
          // New position is behind current position by less than ~55 km:
          // this is dead-reckoning overshoot — DO NOT animate backwards.
          // Keep the current interpolated position as the target, just update heading.
          _interpTo[f.icao24] = {{ lat: from.lat, lng: from.lng, heading: newHdg }};
        }} else {{
          // Normal case: new position is ahead (or a real large position change) — tween to it.
          _interpTo[f.icao24] = {{ lat: newLat, lng: newLng, heading: newHdg }};
        }}
      }});

      if (_mapRef && _mapRef.isStyleLoaded()) {{
        _startInterpLoop();
        // Refresh stats panel and left rail with new data
        const ac = computeAirportCounts();
        const rc = computeRouteCounts();
        const md = getMostDelayedFlights(ac, rc);
        renderTopStats();
        renderLeftRail(ac, rc, md);
        if (ACTIVE_MODULE === 'flight_board') renderFlightBoard();
        _pendingFlightUpdate = null;
      }} else {{
        // Map not yet ready — queue and apply once initFlights completes
        _pendingFlightUpdate = newFlights;
      }}
    }};
    const SPEED_VALUES = FLIGHTS.map((flight) => Number(flight.speed_kts || 0)).filter((value) => Number.isFinite(value) && value > 0);
    const ALTITUDE_VALUES = FLIGHTS.map((flight) => Number(flight.altitude_ft || 0)).filter((value) => Number.isFinite(value) && value > 0);
    const AIRLINE_COUNT = new Set(FLIGHTS.map((flight) => flight.airline_code).filter((value) => value && value !== 'N/A')).size;

    function hideSplash() {{
      const el = document.getElementById('splash');
      if (el) el.classList.add('hidden');
    }}

    function showError(message) {{
      // Always dismiss splash — even if error-box is missing.
      hideSplash();
      const box = document.getElementById('error-box');
      if (!box) return;
      box.textContent = message;
      box.style.display = 'block';
    }}

    function escapeHtml(value) {{
      return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }}

    // ── Registration prefix → country identity ─────────────────────────────────
    // ICAO registration prefixes are internationally standardized.
    // Sorted longest-prefix-first so "AP-" is checked before "A" fallbacks.
    const _REG_PREFIXES = [
      // ── South Asia ──────────────────────────────────────────────────────────
      ['VT',  '🇮🇳', 'India'],
      ['AP',  '🇵🇰', 'Pakistan'],
      ['S2',  '🇧🇩', 'Bangladesh'],
      ['4R',  '🇱🇰', 'Sri Lanka'],
      ['9N',  '🇳🇵', 'Nepal'],
      ['A2',  '🇧🇼', 'Botswana'],  // not SA but short prefix — order matters
      // ── Gulf / Middle East ───────────────────────────────────────────────────
      ['A6',  '🇦🇪', 'UAE'],
      ['A9C', '🇧🇭', 'Bahrain'],
      ['HZ',  '🇸🇦', 'Saudi Arabia'],
      ['A4O', '🇴🇲', 'Oman'],
      ['P4',  '🇦🇼', 'Aruba'],
      ['9K',  '🇰🇼', 'Kuwait'],
      ['A7',  '🇶🇦', 'Qatar'],
      ['EP',  '🇮🇷', 'Iran'],
      ['YK',  '🇸🇾', 'Syria'],
      ['OD',  '🇱🇧', 'Lebanon'],
      ['JY',  '🇯🇴', 'Jordan'],
      ['4X',  '🇮🇱', 'Israel'],
      ['TC',  '🇹🇷', 'Turkey'],
      // ── Europe ───────────────────────────────────────────────────────────────
      ['G',   '🇬🇧', 'UK'],
      ['F',   '🇫🇷', 'France'],
      ['D',   '🇩🇪', 'Germany'],
      ['EC',  '🇪🇸', 'Spain'],
      ['EI',  '🇮🇪', 'Ireland'],
      ['CS',  '🇵🇹', 'Portugal'],
      ['I',   '🇮🇹', 'Italy'],
      ['OO',  '🇧🇪', 'Belgium'],
      ['PH',  '🇳🇱', 'Netherlands'],
      ['HB',  '🇨🇭', 'Switzerland'],
      ['OE',  '🇦🇹', 'Austria'],
      ['OK',  '🇨🇿', 'Czech Republic'],
      ['OM',  '🇸🇰', 'Slovakia'],
      ['SP',  '🇵🇱', 'Poland'],
      ['HA',  '🇭🇺', 'Hungary'],
      ['YR',  '🇷🇴', 'Romania'],
      ['LZ',  '🇧🇬', 'Bulgaria'],
      ['SX',  '🇬🇷', 'Greece'],
      ['TC',  '🇹🇷', 'Turkey'],
      ['SE',  '🇸🇪', 'Sweden'],
      ['LN',  '🇳🇴', 'Norway'],
      ['OH',  '🇫🇮', 'Finland'],
      ['OY',  '🇩🇰', 'Denmark'],
      ['ES',  '🇪🇪', 'Estonia'],
      ['YL',  '🇱🇻', 'Latvia'],
      ['LY',  '🇱🇹', 'Lithuania'],
      ['EW',  '🇧🇾', 'Belarus'],
      ['RA',  '🇷🇺', 'Russia'],
      ['UR',  '🇺🇦', 'Ukraine'],
      ['EK',  '🇦🇲', 'Armenia'],
      ['4K',  '🇦🇿', 'Azerbaijan'],
      ['4L',  '🇬🇪', 'Georgia'],
      // ── Americas ─────────────────────────────────────────────────────────────
      ['N',   '🇺🇸', 'USA'],
      ['C',   '🇨🇦', 'Canada'],
      ['XA',  '🇲🇽', 'Mexico'],
      ['XB',  '🇲🇽', 'Mexico'],
      ['PP',  '🇧🇷', 'Brazil'],
      ['PR',  '🇧🇷', 'Brazil'],
      ['PT',  '🇧🇷', 'Brazil'],
      ['LV',  '🇦🇷', 'Argentina'],
      ['CC',  '🇨🇱', 'Chile'],
      ['HC',  '🇪🇨', 'Ecuador'],
      ['OB',  '🇵🇪', 'Peru'],
      // ── East / SE Asia ────────────────────────────────────────────────────────
      ['B',   '🇨🇳', 'China'],
      ['JA',  '🇯🇵', 'Japan'],
      ['HL',  '🇰🇷', 'South Korea'],
      ['VN',  '🇻🇳', 'Vietnam'],
      ['HS',  '🇹🇭', 'Thailand'],
      ['PK',  '🇮🇩', 'Indonesia'],
      ['9M',  '🇲🇾', 'Malaysia'],
      ['9V',  '🇸🇬', 'Singapore'],
      ['RP',  '🇵🇭', 'Philippines'],
      ['VH',  '🇦🇺', 'Australia'],
      ['ZK',  '🇳🇿', 'New Zealand'],
      // ── Africa ───────────────────────────────────────────────────────────────
      ['ET',  '🇪🇹', 'Ethiopia'],
      ['5A',  '🇱🇾', 'Libya'],
      ['SU',  '🇪🇬', 'Egypt'],
      ['5H',  '🇹🇿', 'Tanzania'],
      ['5N',  '🇳🇬', 'Nigeria'],
      ['ZS',  '🇿🇦', 'South Africa'],
    ];
    // Build a sorted lookup (longest prefix first prevents short-prefix false matches)
    const _REG_SORTED = _REG_PREFIXES.slice().sort((a, b) => b[0].length - a[0].length);

    function regToCountry(reg) {{
      if (!reg || reg === 'N/A') return null;
      const r = reg.toUpperCase().replace(/[^A-Z0-9]/g, '');
      for (const [prefix, flag, country] of _REG_SORTED) {{
        if (r.startsWith(prefix.replace('-', ''))) return {{ flag, country }};
      }}
      return null;
    }}

    // ── Phase of flight ────────────────────────────────────────────────────────
    // Derived purely from altitude_ft, v_speed (m/s), and speed_kts.
    // Returns {{ label, css, icon }}
    function flightPhase(f) {{
      const alt  = Number(f.altitude_ft || 0);
      const vs   = f.v_speed != null ? Number(f.v_speed) : null;   // m/s (positive = climb)
      const spd  = Number(f.speed_kts || 0);
      const vsFpm = vs != null ? vs * 196.85 : null;               // convert to fpm

      if (alt < 500 || (alt < 1500 && spd < 80)) {{
        return {{ label: 'Ground',   css: 'phase-ground',   icon: '🛞' }};
      }}
      // Approach: low altitude + decelerating toward airport
      if (alt < 10000 && spd <= 250 && (vsFpm == null || vsFpm < 200)) {{
        if (alt < 3500) return {{ label: 'Final',    css: 'phase-approach', icon: '🛬' }};
        return {{ label: 'Approach', css: 'phase-approach', icon: '🛬' }};
      }}
      // Climb: significant positive vertical speed or below cruise band
      if (vsFpm != null && vsFpm > 300) {{
        return {{ label: 'Climb',    css: 'phase-climb',    icon: '📈' }};
      }}
      // Cruise: high altitude, level
      if (alt >= 25000 && (vsFpm == null || Math.abs(vsFpm) <= 400)) {{
        return {{ label: 'Cruise',   css: 'phase-cruise',   icon: '✈️' }};
      }}
      // Descent
      if (vsFpm != null && vsFpm < -300) {{
        return {{ label: 'Descent',  css: 'phase-descent',  icon: '📉' }};
      }}
      // Undetermined — still climbing to cruise or shallow descent
      if (alt >= 15000) return {{ label: 'Cruise',  css: 'phase-cruise',  icon: '✈️' }};
      return {{ label: 'Climb', css: 'phase-climb', icon: '📈' }};
    }}

    // ── Aircraft family → category ─────────────────────────────────────────────
    const _FAMILY_CAT = {{
      b747:'wide', a380:'wide', a340:'wide', b777:'wide', a350:'wide',
      a330:'wide', b787:'wide', b767:'wide', b757:'wide',
      a320:'narrow', b737:'narrow',
      e190:'regional', crj:'regional', atr:'regional',
      concorde:'wide', military:'other', default:'narrow',
    }};
    function acCategory(family) {{ return _FAMILY_CAT[family] || 'narrow'; }}
    let _acFilter = 'all';   // 'all' | 'wide' | 'narrow' | 'regional'

    function withTopUrl(mutator, navigate = true) {{
      const apply = (targetWindow) => {{
        const url = new URL(targetWindow.location.href);
        mutator(url);
        if (navigate) {{
          targetWindow.location.href = url.toString();
        }} else if (targetWindow.history && targetWindow.history.replaceState) {{
          targetWindow.history.replaceState({{}}, '', url.toString());
        }}
      }};

      try {{
        apply(window.top);
      }} catch (error) {{
        apply(window);
      }}
    }}

    function formatNumber(value) {{
      if (value === null || value === undefined || Number.isNaN(Number(value))) return 'N/A';
      return Number(value).toLocaleString('en-US');
    }}

    function makeAircraftSvg(family, color) {{
      const body = ICON_TEMPLATES[family] || ICON_TEMPLATES.default;
      return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100%" height="100%">${{body.replaceAll('COLORTOK', color)}}</svg>`;
    }}

    function metricCell(value, label, color) {{
      const clr = color ? (' style="color:' + color + '"') : '';
      return `
        <div class="stat-cell">
          <div class="stat-value"${{clr}}>${{escapeHtml(value)}}</div>
          <div class="stat-label">${{escapeHtml(label)}}</div>
        </div>
      `;
    }}

    function haversineKm(lat1, lng1, lat2, lng2) {{
      const toRad = (deg) => (deg * Math.PI) / 180;
      const dLat = toRad(lat2 - lat1);
      const dLng = toRad(lng2 - lng1);
      const a =
        Math.sin(dLat / 2) ** 2 +
        Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLng / 2) ** 2;
      return 6371 * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    }}

    function computeAirportCounts() {{
      // Prefer backend AIRPORT_METRICS (nearby_count = flights in airport envelope)
      if (AIRPORT_METRICS && AIRPORT_METRICS.length) {{
        return Object.fromEntries(AIRPORT_METRICS.map((a) => [a.airport_iata, a.nearby_count || 0]));
      }}
      // Fallback: manually derive from FLIGHTS dep/arr
      const counts = {{}};
      const indiaOnly = ACTIVE_MODULE === 'delay_prediction';
      FLIGHTS.forEach((flight) => {{
        [flight.dep, flight.arr].forEach((code) => {{
          if (!code || code === 'N/A') return;
          if (indiaOnly && !INDIAN_IATA_SET.has(code)) return;
          counts[code] = (counts[code] || 0) + 1;
        }});
      }});
      return counts;
    }}

    function computeRouteCounts() {{
      const counts = {{}};
      const indiaOnly = ACTIVE_MODULE === 'delay_prediction';
      FLIGHTS.forEach((flight) => {{
        if (!flight.dep || !flight.arr || flight.dep === 'N/A' || flight.arr === 'N/A') return;
        if (indiaOnly && (!INDIAN_IATA_SET.has(flight.dep) || !INDIAN_IATA_SET.has(flight.arr))) return;
        const key = `${{flight.dep}}-${{flight.arr}}`;
        counts[key] = (counts[key] || 0) + 1;
      }});
      return counts;
    }}

    function getMostDelayedFlights(airportCounts, routeCounts) {{
      const indiaOnly = ACTIVE_MODULE === 'delay_prediction';
      // Backend pred_delay_min is the ONLY ranking signal.
      // AirLabs delayed_min is kept separate (schedule-verified) and shown as
      // an annotation, never blended into the prediction score.
      return [...FLIGHTS]
        .filter((f) => !indiaOnly || INDIAN_IATA_SET.has(f.dep) || INDIAN_IATA_SET.has(f.arr))
        .map((flight) => {{
          const minutes  = Number(flight.pred_delay_min  || 0);
          const risk     = flight.pred_delay_risk || (minutes >= 30 ? 'High' : minutes >= 14 ? 'Medium' : 'Low');
          const reasons  = flight.pred_delay_reason
            ? flight.pred_delay_reason.split(',').map(function(s) {{ return s.trim(); }}).filter(Boolean)
            : (minutes > 0 ? ['delay predicted'] : ['stable profile']);
          return {{
            ...flight,
            delayInfo: {{
              minutes: minutes,
              level: risk,
              reasons: reasons,
              signals: [],
              onFinal: false,
            }},
          }};
        }})
        .filter((f) => f.delayInfo.minutes > 0)
        .sort((a, b) => b.delayInfo.minutes - a.delayInfo.minutes)
        .slice(0, 5);
    }}

    function getNearestAirport(flight) {{
      let best = null;
      let bestDistance = Infinity;
      AIRPORTS.forEach((airport) => {{
        const distance = haversineKm(flight.lat, flight.lng, airport.lat, airport.lng);
        if (distance < bestDistance) {{
          best = airport;
          bestDistance = distance;
        }}
      }});
      return {{
        airport: best,
        distanceKm: Number.isFinite(bestDistance) ? Math.round(bestDistance) : null,
      }};
    }}

    function renderTopStats() {{
      const topStats = document.getElementById('top-stats');
      if (!topStats) return;
      const avgSpeed = SPEED_VALUES.length
        ? `${{Math.round(SPEED_VALUES.reduce((sum, value) => sum + value, 0) / SPEED_VALUES.length)}} kts`
        : 'N/A';
      const avgAltitude = ALTITUDE_VALUES.length
        ? `${{Math.round(ALTITUDE_VALUES.reduce((sum, value) => sum + value, 0) / ALTITUDE_VALUES.length).toLocaleString('en-US')}} ft`
        : 'N/A';
      // Airspace CO₂: sum backend values where available, skip nulls
      const co2flights = FLIGHTS.filter((f) => f.fuel_co2_kg != null);
      const co2Total = co2flights.length
        ? co2flights.reduce((s, f) => s + (f.fuel_co2_kg || 0), 0)
        : null;
      const co2Label = co2Total != null ? (co2Total / 1000).toFixed(0) + ' t CO₂' : 'N/A';
      // Avg delay from live flights with delay data
      const delayedFlights = FLIGHTS.filter((f) => f.delayed_min != null && f.delayed_min > 0);
      const _avgDlyRaw = delayedFlights.length
        ? Math.round(delayedFlights.reduce((s, f) => s + Number(f.delayed_min), 0) / delayedFlights.length)
        : 0;
      const avgDelayLabel = delayedFlights.length ? ('+' + _avgDlyRaw + ' min') : '—';
      // Most congested airport from AIRPORT_METRICS
      const topAp = (AIRPORT_METRICS && AIRPORT_METRICS.length)
        ? AIRPORT_METRICS.reduce((best, m) => (!best || (m.congestion_score || 0) > (best.congestion_score || 0)) ? m : best, null)
        : null;
      const congLabel = topAp ? (topAp.airport_iata + ' (' + (topAp.congestion_level || '?') + ')') : '—';
      topStats.innerHTML = [
        metricCell(String(FLIGHTS.length), 'Flights'),
        metricCell(avgSpeed, 'Avg Speed'),
        metricCell(avgAltitude, 'Avg Altitude'),
        metricCell(String(AIRLINE_COUNT), 'Airlines'),
        metricCell(co2Label, 'Airspace CO₂'),
        metricCell(avgDelayLabel, 'Avg Delay', delayedFlights.length > 0 ? '#fbbf24' : null),
        metricCell(congLabel, 'Top Congested', topAp ? '#ff9f43' : null),
      ].join('');
    }}

    function renderLeftRail(airportCounts, routeCounts, mostDelayed) {{
      const leftRail = document.getElementById('left-rail');
      if (!leftRail) return;
      if (ACTIVE_MODULE === 'live_ops') {{
        leftRail.innerHTML = '';
        leftRail.style.display = 'none';
        return;
      }}
      if (ACTIVE_MODULE === 'alerts') {{
        leftRail.style.display = 'flex';
        const sevIcon = {{ alert: '🔴', warning: '🟡', info: '🔵' }};
        const alertCount  = ANOMALIES.filter((a) => a.severity === 'alert').length;
        const warnCount   = ANOMALIES.filter((a) => a.severity === 'warning').length;
        const anomalyRows = ANOMALIES.slice(0, 15).map((a) => `
          <div class="list-row" style="cursor:pointer;" data-action="track-flight" data-flight="${{escapeHtml(a.flight_iata || '')}}">
            <span>${{sevIcon[a.severity] || '⚪'}} ${{escapeHtml(a.flight_iata || 'N/A')}}</span>
            <span class="meta" style="max-width:110px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${{escapeHtml(a.description || '')}}">${{escapeHtml((a.anomaly_type||'').replace(/_/g,' '))}}</span>
            <span class="tag ${{a.severity === 'alert' ? 'high' : a.severity === 'warning' ? 'medium' : 'low'}}">${{escapeHtml(a.severity || '')}}</span>
          </div>
        `).join('');
        leftRail.innerHTML = `
          <div class="rail-card glass-panel" style="border-left:3px solid ${{alertCount > 0 ? '#f87171' : warnCount > 0 ? '#fbbf24' : '#4ade80'}}">
            <div class="rail-card-header">
              <span class="rail-heading">🚨 Anomaly Detection</span>
              <span class="rail-count-pill">${{ANOMALIES.length}}</span>
            </div>
            <div style="display:flex;gap:10px;padding:6px 0 10px;font-size:11px;">
              <span style="color:#f87171;font-weight:700;">${{alertCount}} alerts</span>
              <span style="color:#fbbf24;font-weight:700;">${{warnCount}} warnings</span>
              <span style="color:rgba(244,247,251,0.4);">${{ANOMALIES.length - alertCount - warnCount}} info</span>
            </div>
            <div class="rail-rows">
              ${{anomalyRows || '<div class="list-row"><span style="color:rgba(244,247,251,0.38)">No anomalies detected</span></div>'}}
            </div>
          </div>
        `;
        leftRail.querySelectorAll('[data-action="track-flight"]').forEach((row) => {{
          row.addEventListener('click', () => {{
            const fn = row.getAttribute('data-flight');
            const f = FLIGHTS.find((x) => x.flight === fn);
            if (f && _mapRef) setActiveFlight(_mapRef, f);
          }});
        }});
        return;
      }}

      // ── Airport Traffic module — full detail cards from backend metrics ────
      if (ACTIVE_MODULE === 'airport_traffic') {{
        leftRail.style.display = 'flex';

        // Fuel rate lookup by aircraft family (kg/km cruise)
        const _FUEL_RATE = {{
          b747:8.8, a380:9.5, a340:7.0, b777:6.5, a350:5.0, a330:6.2,
          b787:5.2, b767:5.8, b757:4.8, a320:2.8, b737:2.9,
          e190:2.4, crj:2.1, atr:1.2, default:3.0,
        }};
        function _fuelForFlight(f) {{
          const rate = _FUEL_RATE[f.family] || _FUEL_RATE.default;
          const arrAp = AIRPORT_LOOKUP[f.arr];
          if (!arrAp) return null;
          const depAp = AIRPORT_LOOKUP[f.dep];
          const totalKm = depAp
            ? haversineKm(depAp.lat, depAp.lng, arrAp.lat, arrAp.lng)
            : haversineKm(f.lat, f.lng, arrAp.lat, arrAp.lng) * 1.25;
          if (!totalKm) return null;
          const fuel = Math.round(rate * totalKm * 1.08);
          const co2  = Math.round(fuel * 3.16);
          const seats = {{b747:416,a380:555,b777:396,a350:369,a330:277,b787:296,b737:162,a320:150,e190:110,crj:70,atr:68}};
          const s = seats[f.family];
          return {{ fuel, co2, co2pp: s ? Math.round(co2/s) : null }};
        }}

        // Build airport cards from AIRPORT_METRICS (backend-computed)
        const maxScore = AIRPORT_METRICS.reduce((m, a) => Math.max(m, a.congestion_score || 0), 1);
        const atCards = AIRPORT_METRICS.slice(0, 8).map((a) => {{
          const ap = AIRPORT_LOOKUP[a.airport_iata] || {{}};
          const lvlRaw = (a.congestion_level || 'Low').toLowerCase();
          const lvl = lvlRaw === 'critical' ? 'critical' : lvlRaw === 'high' ? 'high' : lvlRaw === 'moderate' ? 'moderate' : 'low';
          const score = a.congestion_score || 0;
          const pct = Math.round(score / maxScore * 100);
          const sp = SCHEDULE_PRESSURE[a.airport_iata] || null;
          const hasSchedule = sp && sp.pressure_level && sp.pressure_level !== 'Unknown';
          const pressLvl = hasSchedule ? sp.pressure_level : null;
          return `
            <div class="at-airport-card" data-action="zoom-airport"
                 data-code="${{escapeHtml(a.airport_iata)}}"
                 data-lat="${{ap.lat || ''}}" data-lng="${{ap.lng || ''}}">
              <div class="at-airport-card-head">
                <div>
                  <div class="at-airport-code">${{escapeHtml(a.airport_iata)}}</div>
                  <div class="at-airport-city">${{escapeHtml(a.city || ap.city || '')}}</div>
                </div>
                <div style="display:flex;flex-direction:column;align-items:flex-end;gap:4px;">
                  <span class="at-level-badge ${{lvl}}">${{escapeHtml(a.congestion_level || 'Low')}}</span>
                  ${{hasSchedule
                    ? `<span class="at-pressure-badge ${{pressLvl}}">${{escapeHtml(pressLvl)}} pressure</span>`
                    : `<span class="at-pressure-badge" style="color:rgba(244,247,251,0.28);font-size:9px;">no sched data</span>`
                  }}
                </div>
              </div>
              <div class="at-score-bar-wrap">
                <div class="at-score-bar-fill ${{lvl}}" style="width:${{pct}}%"></div>
              </div>
              <div class="at-metrics-mini">
                <div class="at-metric-cell">
                  <div class="at-metric-label">Nearby</div>
                  <div class="at-metric-value">${{a.nearby_count || 0}}</div>
                </div>
                <div class="at-metric-cell">
                  <div class="at-metric-label">Inbound</div>
                  <div class="at-metric-value">${{a.inbound_count || 0}}</div>
                </div>
                <div class="at-metric-cell">
                  <div class="at-metric-label">Approach</div>
                  <div class="at-metric-value">${{a.approach_count || 0}}</div>
                </div>
                <div class="at-metric-cell">
                  <div class="at-metric-label">Low Alt</div>
                  <div class="at-metric-value">${{a.low_altitude_count || 0}}</div>
                </div>
              </div>
              <div class="at-sched-chart">
                <div class="at-sched-chart-header">
                  <span class="at-sched-chart-title">3-Hour Outlook</span>
                  <span class="at-sched-chart-title"><span style="color:rgba(0,229,255,0.6);">━</span> arr &nbsp;<span style="color:rgba(255,210,74,0.6);">━</span> dep</span>
                </div>
                ${{hasSchedule ? (() => {{
                  const arrH = [sp.arrivals_h0 || 0, sp.arrivals_h1 || 0, sp.arrivals_h2 || 0];
                  const depH = [sp.departures_h0 || 0, sp.departures_h1 || 0, sp.departures_h2 || 0];
                  const maxV = Math.max(...arrH, ...depH, 1);
                  const lbls = ['+0h', '+1h', '+2h'];
                  return `<div class="at-sched-bars">` + arrH.map((a, i) => {{
                    const d = depH[i];
                    const aH = Math.max(Math.round(a / maxV * 26), 2);
                    const dH = Math.max(Math.round(d / maxV * 26), 2);
                    return `<div class="at-sched-bar-group">
                      <div class="at-sched-bar-pair">
                        <div class="at-sched-bar arr" style="height:${{aH}}px" title="${{a}} arrivals"></div>
                        <div class="at-sched-bar dep" style="height:${{dH}}px" title="${{d}} departures"></div>
                      </div>
                      <div class="at-sched-bar-label">${{lbls[i]}}</div>
                    </div>`;
                  }}).join('') + `</div>`;
                }})() : `<div style="font-size:9px;color:rgba(244,247,251,0.25);padding:4px 0;">no schedule data</div>`}}
                <div class="at-sched-total">
                  <span>
                    <span style="color:rgba(0,229,255,0.7);">${{hasSchedule ? (sp.arrivals_next_3h || 0) : '—'}}</span> arr
                    &nbsp;/&nbsp;
                    <span style="color:rgba(255,210,74,0.7);">${{hasSchedule ? (sp.departures_next_3h || 0) : '—'}}</span> dep
                    <span style="color:rgba(244,247,251,0.22);"> in 3h</span>
                  </span>
                  <span style="color:rgba(244,247,251,0.22);">Score ${{score.toFixed(1)}}</span>
                </div>
              </div>
            </div>
          `;
        }}).join('');

        leftRail.innerHTML = `
          <div class="rail-card glass-panel">
            <div class="rail-card-header">
              <span class="rail-heading">Hub Pressure</span>
              <span class="rail-count-pill">${{AIRPORT_METRICS.length}}</span>
            </div>
            <div style="padding:8px 10px 4px;">
              ${{atCards || '<div style="color:rgba(244,247,251,0.38);font-size:12px;padding:8px 0;">No airport data</div>'}}
            </div>
          </div>
          <div style="height:40px;flex-shrink:0;"></div>
        `;

        leftRail.querySelectorAll('[data-action="zoom-airport"]').forEach((row) => {{
          row.addEventListener('click', () => {{
            const lat = parseFloat(row.getAttribute('data-lat'));
            const lng = parseFloat(row.getAttribute('data-lng'));
            if (_mapRef && Number.isFinite(lat) && Number.isFinite(lng)) {{
              _mapRef.flyTo({{ center: [lng, lat], zoom: 10.5, duration: 1100, essential: true }});
            }}
          }});
        }});
        return;
      }}

      // ── Route Intelligence module — corridor congestion from live data ──
      if (ACTIVE_MODULE === 'route_intelligence') {{
        leftRail.style.display = 'flex';

        // Aggregate route traffic from live flights
        const routeAgg = Object.entries(routeCounts)
          .sort((a, b) => b[1] - a[1])
          .slice(0, 10);

        // Risk distribution across all visible flights
        let riHigh = 0, riMed = 0, riLow = 0;
        FLIGHTS.forEach((f) => {{
          const r = (f.pred_delay_risk || '').toLowerCase();
          if (r === 'high') riHigh++;
          else if (r === 'medium') riMed++;
          else riLow++;
        }});
        const riTotal = riHigh + riMed + riLow || 1;
        const riHighPct  = Math.round(riHigh  / riTotal * 100);
        const riMedPct   = Math.round(riMed   / riTotal * 100);

        const corridorRows = routeAgg.map(([route, count]) => {{
          const [dep, arr] = route.split('-');
          const depAp = AIRPORT_LOOKUP[dep];
          const arrAp = AIRPORT_LOOKUP[arr];
          const label = (depAp && arrAp) ? `${{depAp.city || dep}} → ${{arrAp.city || arr}}` : route;
          // Avg delay on this corridor
          const routeFlights = FLIGHTS.filter((f) => f.dep === dep && f.arr === arr);
          const avgDelay = routeFlights.length
            ? Math.round(routeFlights.reduce((s, f) => s + Number(f.pred_delay_min || 0), 0) / routeFlights.length)
            : 0;
          const maxDelay = routeFlights.length
            ? Math.max(...routeFlights.map((f) => Number(f.pred_delay_min || 0)))
            : 0;
          const anyHigh = routeFlights.some((f) => (f.pred_delay_risk || '').toLowerCase() === 'high');
          const anyMed  = routeFlights.some((f) => (f.pred_delay_risk || '').toLowerCase() === 'medium');
          const lvl = anyHigh ? 'high' : anyMed ? 'medium' : 'low';
          const lvlLabel = anyHigh ? 'High' : anyMed ? 'Med' : 'Stable';
          return `
            <div class="rail-row ri-corridor-row" data-action="track-route"
                 data-dep="${{escapeHtml(dep)}}" data-arr="${{escapeHtml(arr)}}">
              <div class="rail-row-inner">
                <div class="rr-icon route">⇢</div>
                <div class="rr-body">
                  <div class="rr-primary">${{escapeHtml(route)}}</div>
                  <div class="rr-secondary">${{escapeHtml(label)}}</div>
                </div>
                <div class="rr-right">
                  <span class="rr-badge ${{lvl}}">${{lvlLabel}}</span>
                  <span class="rr-meta">${{count}} flt · avg +${{avgDelay}}m</span>
                </div>
              </div>
              ${{maxDelay > 0 ? `<div style="margin:2px 0 4px 32px;height:2px;border-radius:2px;background:rgba(255,255,255,0.06);overflow:hidden;">
                <div style="height:100%;width:${{Math.min(100, Math.round(maxDelay/90*100))}}%;background:${{anyHigh ? '#ff6868' : anyMed ? '#f0c840' : 'rgba(0,229,255,0.45)'}};border-radius:2px;"></div>
              </div>` : ''}}
            </div>
          `;
        }}).join('');

        // Selected flight corridor detail
        let corridorDetail = '';
        if (ACTIVE_SELECTION) {{
          const sf = ACTIVE_SELECTION;
          const sfRoute = sf.dep && sf.arr ? `${{sf.dep}}-${{sf.arr}}` : '';
          const sfCount = sfRoute ? (routeCounts[sfRoute] || 0) : 0;
          const sfAvgDelay = sfCount ? Math.round(
            FLIGHTS.filter((f) => f.dep === sf.dep && f.arr === sf.arr)
              .reduce((s, f) => s + Number(f.pred_delay_min || 0), 0) / sfCount
          ) : 0;
          corridorDetail = `
            <div class="rail-card glass-panel">
              <div class="rail-card-header">
                <span class="rail-heading">Selected Corridor</span>
              </div>
              <div style="padding:8px 12px 10px;font-size:11px;color:rgba(244,247,251,0.75);">
                <div style="font-size:13px;font-weight:600;margin-bottom:4px;">
                  ${{escapeHtml(sf.dep || '?')}} → ${{escapeHtml(sf.arr || '?')}}
                </div>
                <div style="display:flex;gap:14px;">
                  <div><span style="color:rgba(244,247,251,0.45);">Flights:</span> ${{sfCount}}</div>
                  <div><span style="color:rgba(244,247,251,0.45);">Avg delay:</span> ${{sfAvgDelay > 0 ? '+' + sfAvgDelay + ' min' : 'None'}}</div>
                  <div><span style="color:rgba(244,247,251,0.45);">Risk:</span> ${{escapeHtml(sf.pred_delay_risk || 'Low')}}</div>
                </div>
              </div>
            </div>
          `;
        }}

        leftRail.innerHTML = `
          ${{corridorDetail}}
          <div class="rail-card glass-panel">
            <div class="rail-card-header">
              <span class="rail-heading">Network Risk Distribution</span>
            </div>
            <div style="padding:8px 12px 10px;">
              <div style="display:flex;align-items:center;gap:6px;margin-bottom:8px;">
                <div style="display:flex;height:6px;border-radius:3px;overflow:hidden;flex:1;gap:1px;">
                  ${{riHigh  ? `<div style="flex:${{riHigh}};background:#ff6868;"></div>`  : ''}}
                  ${{riMed   ? `<div style="flex:${{riMed}};background:#f0c840;"></div>`   : ''}}
                  ${{riLow   ? `<div style="flex:${{riLow}};background:rgba(0,229,255,0.5);"></div>` : ''}}
                </div>
              </div>
              <div style="display:flex;gap:10px;font-size:10px;color:rgba(244,247,251,0.6);">
                <span style="color:#ff6868;">■</span> High ${{riHighPct}}%
                <span style="color:#f0c840;">■</span> Med ${{riMedPct}}%
                <span style="color:rgba(0,229,255,0.7);">■</span> Stable ${{100 - riHighPct - riMedPct}}%
                <span style="margin-left:auto;">${{riTotal}} flt</span>
              </div>
            </div>
          </div>
          <div class="rail-card glass-panel">
            <div class="rail-card-header">
              <span class="rail-heading">Route Corridors</span>
              <span class="rail-count-pill">${{routeAgg.length}}</span>
            </div>
            <div class="rail-rows">
              ${{corridorRows || '<div class="list-row"><span style="color:rgba(244,247,251,0.38)">No route data</span></div>'}}
            </div>
          </div>
          <div style="height:40px;flex-shrink:0;"></div>
        `;

        leftRail.querySelectorAll('[data-action="track-route"]').forEach((row) => {{
          row.addEventListener('click', () => {{
            const dep = row.getAttribute('data-dep');
            const arr = row.getAttribute('data-arr');
            const f = FLIGHTS.find((x) => x.dep === dep && x.arr === arr);
            if (f && _mapRef) setActiveFlight(_mapRef, f);
          }});
        }});
        return;
      }}

      leftRail.style.display = 'flex';

      // ── Top Airports ─────────────────────────────────────────────────────
      const airportEntries = Object.entries(airportCounts)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 4);

      const airportRows = airportEntries.map(([code, count]) => {{
        const ap = AIRPORT_LOOKUP[code];
        const city = ap ? ap.city : '';
        const badgeCls = count >= 16 ? 'high' : count >= 10 ? 'medium' : 'low';
        const badgeLabel = count >= 16 ? 'Severe' : count >= 10 ? 'High' : 'Moderate';
        return `
          <div class="rail-row" data-action="zoom-airport" data-code="${{escapeHtml(code)}}"
               data-lat="${{ap ? ap.lat : ''}}" data-lng="${{ap ? ap.lng : ''}}">
            <div class="rail-row-inner">
              <div class="rr-icon airport">✦</div>
              <div class="rr-body">
                <div class="rr-primary">${{escapeHtml(code)}}</div>
                <div class="rr-secondary">${{city ? escapeHtml(city) : escapeHtml(String(count)) + ' flt in envelope'}}</div>
              </div>
              <div class="rr-right">
                <span class="rr-badge ${{badgeCls}}">${{badgeLabel}}</span>
                <span class="rr-meta">${{escapeHtml(String(count))}} flt</span>
              </div>
            </div>
          </div>
        `;
      }}).join('');

      // ── Busiest Routes ────────────────────────────────────────────────────
      const routeRows = Object.entries(routeCounts)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 4)
        .map(([route, count]) => {{
          const [dep, arr] = route.split('-');
          const depAp = AIRPORT_LOOKUP[dep];
          const arrAp = AIRPORT_LOOKUP[arr];
          const subLabel = (depAp && arrAp) ? `${{depAp.city || dep}} → ${{arrAp.city || arr}}` : route;
          const score = (count * 3.2).toFixed(1);
          const badgeCls = count >= 6 ? 'high' : count >= 3 ? 'medium' : 'neutral';
          const firstFlight = FLIGHTS.find((f) => f.dep === dep && f.arr === arr);
          return `
            <div class="rail-row" data-action="track-route" data-dep="${{escapeHtml(dep)}}" data-arr="${{escapeHtml(arr)}}"
                 data-icao24="${{firstFlight ? escapeHtml(firstFlight.icao24) : ''}}">
              <div class="rail-row-inner">
                <div class="rr-icon route">⇢</div>
                <div class="rr-body">
                  <div class="rr-primary">${{escapeHtml(route)}}</div>
                  <div class="rr-secondary">${{escapeHtml(subLabel)}}</div>
                </div>
                <div class="rr-right">
                  <span class="rr-badge ${{badgeCls}}">${{escapeHtml(String(count))}} flt</span>
                  <span class="rr-meta">${{score}} idx</span>
                </div>
              </div>
            </div>
          `;
        }}).join('');

      // ── Most Delayed ──────────────────────────────────────────────────────
      const _delayFiltered = mostDelayed.filter((f) => f.delayInfo.minutes >= _dpThreshold);
      const delayRows = _delayFiltered.map((flight) => {{
        const level = flight.delayInfo.level.toLowerCase();
        const reasons = flight.delayInfo.reasons.slice(0, 1).join('');
        const airline = flight.airline_code || flight.airline || '';
        return `
          <div class="rail-row" data-action="track-flight" data-icao24="${{escapeHtml(flight.icao24 || '')}}">
            <div class="rail-row-inner">
              <div class="rr-icon flight">✈</div>
              <div class="rr-body">
                <div class="rr-primary">${{escapeHtml(flight.flight || 'Unknown')}}</div>
                <div class="rr-secondary">${{airline ? escapeHtml(airline) + ' · ' : ''}}${{escapeHtml(reasons)}}</div>
              </div>
              <div class="rr-right">
                <span class="rr-badge ${{level}}">+${{flight.delayInfo.minutes}} min</span>
                <span class="rr-track-btn">TRACK</span>
              </div>
            </div>
          </div>
        `;
      }}).join('');

      const isDelayModule = ACTIVE_MODULE === 'delay_prediction';

      // ── Risk Distribution (delay_prediction only) ────────────────────────
      let dpRiskDistHtml = '';
      if (isDelayModule) {{
        let dpHigh = 0, dpMed = 0, dpLow = 0;
        FLIGHTS.forEach((f) => {{
          const r = (f.pred_delay_risk || '').toLowerCase();
          if (r === 'high') dpHigh++;
          else if (r === 'medium') dpMed++;
          else dpLow++;
        }});
        const dpTotal = dpHigh + dpMed + dpLow || 1;
        const dpHighPct = Math.round(dpHigh / dpTotal * 100);
        const dpMedPct  = Math.round(dpMed  / dpTotal * 100);
        const dpLowPct  = 100 - dpHighPct - dpMedPct;
        const dpAvgDelay = FLIGHTS.length
          ? Math.round(FLIGHTS.reduce((s, f) => s + Number(f.pred_delay_min || 0), 0) / FLIGHTS.length)
          : 0;
        dpRiskDistHtml = `
          <div class="rail-card glass-panel">
            <div class="rail-card-header">
              <span class="rail-heading">Risk Distribution</span>
              <span class="dp-val-badge">${{dpTotal}} flights</span>
            </div>
            <div style="padding:8px 12px 10px;">
              <div style="display:flex;height:8px;border-radius:4px;overflow:hidden;gap:1px;margin-bottom:8px;">
                ${{dpHigh ? `<div style="flex:${{dpHigh}};background:#ff6868;border-radius:4px 0 0 4px;"></div>` : ''}}
                ${{dpMed  ? `<div style="flex:${{dpMed}};background:#f0c840;"></div>` : ''}}
                ${{dpLow  ? `<div style="flex:${{dpLow}};background:rgba(0,229,255,0.45);${{!dpHigh && !dpMed ? 'border-radius:4px;' : 'border-radius:0 4px 4px 0;'}}"></div>` : ''}}
              </div>
              <div style="display:flex;gap:10px;font-size:10px;color:rgba(244,247,251,0.6);">
                <span><span style="color:#ff6868;">■</span> High ${{dpHighPct}}%</span>
                <span><span style="color:#f0c840;">■</span> Med ${{dpMedPct}}%</span>
                <span><span style="color:rgba(0,229,255,0.7);">■</span> Low ${{dpLowPct}}%</span>
              </div>
              ${{dpAvgDelay > 0 ? `<div style="margin-top:7px;font-size:10px;color:rgba(244,247,251,0.45);">Network avg delay: <span style="color:rgba(244,247,251,0.75);">+${{dpAvgDelay}} min</span></div>` : ''}}
            </div>
          </div>
        `;
      }}

      // ── Top Airlines ─────────────────────────────────────────────────────
      const airlineRows = (TOP_AIRLINES && TOP_AIRLINES.length) ? TOP_AIRLINES.map((al, idx) => {{
        const maxCount = TOP_AIRLINES[0].count || 1;
        const pct = Math.round(al.count / maxCount * 100);
        // Compute scorecard from FLIGHTS in browser
        const alFlights = FLIGHTS.filter(f => f.airline_code === al.code);
        const alDelayed = alFlights.filter(f => (f.delayed_min || 0) > 0).length;
        const alOnTime = alFlights.length ? Math.round((1 - alDelayed / alFlights.length) * 100) : null;
        const alAvgDly = alDelayed ? Math.round(alFlights.filter(f => (f.delayed_min || 0) > 0).reduce((s, f) => s + Number(f.delayed_min), 0) / alDelayed) : 0;
        // Busiest route for this airline
        const routeMap = {{}};
        alFlights.forEach(f => {{ if (f.dep && f.arr) {{ const k = `${{f.dep}}-${{f.arr}}`; routeMap[k] = (routeMap[k]||0)+1; }} }});
        const topRoute = Object.entries(routeMap).sort((a,b)=>b[1]-a[1])[0];
        const scorecardHtml = alOnTime !== null ? `<div class="airline-scorecard" id="al-sc-${{idx}}">
          <div class="al-sc-row"><span class="al-sc-lbl">On-Time Rate</span><span class="al-sc-val" style="color:${{alOnTime>=80?'#4ade80':alOnTime>=60?'#fbbf24':'#f87171'}}">${{alOnTime}}%</span></div>
          <div class="al-sc-row"><span class="al-sc-lbl">Avg Delay</span><span class="al-sc-val">${{alAvgDly > 0 ? '+'+alAvgDly+' min' : '—'}}</span></div>
          ${{topRoute ? `<div class="al-sc-row"><span class="al-sc-lbl">Busiest Route</span><span class="al-sc-val" style="color:#00e5ff">${{topRoute[0]}}</span></div>` : ''}}
        </div>` : '';
        return `
          <div class="rail-row" data-al-idx="${{idx}}" style="cursor:pointer;">
            <div class="rail-row-inner">
              <div class="rr-icon route" style="font-size:11px;font-weight:800;">${{escapeHtml(al.code)}}</div>
              <div class="rr-body">
                <div class="rr-primary">${{escapeHtml(al.name || al.code)}}</div>
                <div class="rr-secondary" style="margin-top:3px;">
                  <div style="height:3px;border-radius:2px;background:rgba(255,255,255,0.08);overflow:hidden;width:80px;">
                    <div style="height:100%;width:${{pct}}%;background:rgba(0,229,255,0.55);border-radius:2px;"></div>
                  </div>
                </div>
              </div>
              <div class="rr-right">
                <span class="rr-badge neutral">${{al.count}} flt</span>
                <span style="font-size:9px;color:rgba(244,247,251,0.3);">▾</span>
              </div>
            </div>
            ${{scorecardHtml}}
          </div>
        `;
      }}).join('') : '';

      leftRail.innerHTML = `
        ${{dpRiskDistHtml}}
        ${{isDelayModule ? `
        <div class="rail-card glass-panel">
          <div class="rail-card-header">
            <span class="rail-heading">Delay Threshold</span>
            <span class="dp-val-badge" id="dp-slider-val">≥ ${{_dpThreshold}} min</span>
          </div>
          <div class="dp-slider-wrap">
            <span class="dp-slider-lbl">0</span>
            <input type="range" id="dp-threshold-slider" class="dp-range"
              min="0" max="60" step="5" value="${{_dpThreshold}}" style="--fill:${{(_dpThreshold/60*100).toFixed(1)}}%">
            <span class="dp-slider-lbl">60</span>
          </div>
          <div class="dp-ticks">
            <span class="dp-tick">Low &lt;10</span>
            <span class="dp-tick">Med 10-30</span>
            <span class="dp-tick">High &gt;30</span>
          </div>
        </div>` : ''}}
        <div class="rail-card glass-panel">
          <div class="rail-card-header">
            <span class="rail-heading">Most Delayed</span>
          </div>
          <div class="rail-rows" id="dp-delayed-rows">${{delayRows || `<div class="list-row"><span style="color:rgba(244,247,251,0.38)">${{_dpThreshold > 0 ? 'No flights above ' + _dpThreshold + ' min' : 'No delay data'}}</span></div>`}}</div>
        </div>
        ${{!isDelayModule ? `
        <div class="rail-card glass-panel">
          <div class="rail-card-header">
            <span class="rail-heading">Top Airports</span>
            <span class="rail-count-pill">${{airportEntries.length}}</span>
          </div>
          <div class="rail-rows">${{airportRows || '<div class="list-row"><span style="color:rgba(244,247,251,0.38)">No data</span></div>'}}</div>
        </div>
        <div class="rail-card glass-panel">
          <div class="rail-card-header">
            <span class="rail-heading">Busiest Routes</span>
          </div>
          <div class="rail-rows">${{routeRows || '<div class="list-row"><span style="color:rgba(244,247,251,0.38)">No data</span></div>'}}</div>
        </div>` : ''}}
        ${{airlineRows ? `
        <div class="rail-card glass-panel">
          <div class="rail-card-header">
            <span class="rail-heading">Top Airlines</span>
            <span class="rail-count-pill">${{TOP_AIRLINES.length}}</span>
          </div>
          <div class="rail-rows">${{airlineRows}}</div>
        </div>` : ''}}
        ${{!isDelayModule ? (() => {{
          // Fleet nationality card — group by registration prefix
          const natMap = {{}};
          FLIGHTS.forEach((f) => {{
            const c = regToCountry(f.registration);
            if (!c) {{ natMap['Unknown'] = (natMap['Unknown'] || {{flag:'🌐', count:0}}); natMap['Unknown'].count++; return; }}
            if (!natMap[c.country]) natMap[c.country] = {{ flag: c.flag, count: 0 }};
            natMap[c.country].count++;
          }});
          const natList = Object.entries(natMap)
            .sort((a, b) => b[1].count - a[1].count)
            .filter(([, v]) => v.count > 0)
            .slice(0, 8);
          if (!natList.length) return '';
          const maxNat = natList[0][1].count || 1;
          const natRows = natList.map(([country, {{flag, count}}]) => {{
            const pct = Math.round(count / maxNat * 100);
            return `<div class="nations-row">
              <span class="nations-flag">${{flag}}</span>
              <span style="font-size:11px;color:rgba(244,247,251,0.70);min-width:64px;">${{escapeHtml(country)}}</span>
              <div class="nations-bar-wrap"><div class="nations-bar-fill" style="width:${{pct}}%"></div></div>
              <span class="nations-count">${{count}}</span>
            </div>`;
          }}).join('');
          return `<div class="rail-card glass-panel">
            <div class="rail-card-header">
              <span class="rail-heading">Fleet Nations</span>
              <span class="rail-count-pill">${{natList.length}}</span>
            </div>
            <div style="padding:6px 12px 10px;">${{natRows}}</div>
          </div>`;
        }})() : ''}}
        <div style="height:40px;flex-shrink:0;"></div>
      `;

      // ── Delay threshold slider ─────────────────────────────────────────────
      const dpSlider = document.getElementById('dp-threshold-slider');
      const dpValBadge = document.getElementById('dp-slider-val');
      const dpDelayedRows = document.getElementById('dp-delayed-rows');
      if (dpSlider && dpValBadge && dpDelayedRows) {{
        function buildDelayRows(threshold) {{
          const filtered = mostDelayed.filter((f) => f.delayInfo.minutes >= threshold);
          if (!filtered.length) return '<div class="list-row"><span style="color:rgba(244,247,251,0.38)">No flights above threshold</span></div>';
          return filtered.map((flight) => {{
            const lvl = flight.delayInfo.level.toLowerCase();
            const reasons = flight.delayInfo.reasons.slice(0, 1).join('');
            const airline = flight.airline_code || flight.airline || '';
            return `
              <div class="rail-row" data-action="track-flight" data-icao24="${{escapeHtml(flight.icao24 || '')}}">
                <div class="rail-row-inner">
                  <div class="rr-icon flight">✈</div>
                  <div class="rr-body">
                    <div class="rr-primary">${{escapeHtml(flight.flight || 'Unknown')}}</div>
                    <div class="rr-secondary">${{airline ? escapeHtml(airline) + ' · ' : ''}}${{escapeHtml(reasons)}}</div>
                  </div>
                  <div class="rr-right">
                    <span class="rr-badge ${{lvl}}">+${{flight.delayInfo.minutes}} min</span>
                    <span class="rr-track-btn">TRACK</span>
                  </div>
                </div>
              </div>
            `;
          }}).join('');
        }}
        dpSlider.addEventListener('input', () => {{
          const t = parseInt(dpSlider.value, 10);
          _dpThreshold = t;
          dpValBadge.textContent = `≥ ${{t}} min`;
          dpSlider.style.setProperty('--fill', `${{(t / 60 * 100).toFixed(1)}}%`);
          dpDelayedRows.innerHTML = buildDelayRows(t);
          dpDelayedRows.querySelectorAll('[data-action="track-flight"]').forEach((row) => {{
            row.addEventListener('click', () => {{
              const icao24 = row.getAttribute('data-icao24');
              const f = FLIGHTS.find((x) => x.icao24 === icao24);
              if (f && _mapRef) setActiveFlight(_mapRef, f);
            }});
          }});
        }});
      }}

      // ── Click handlers ────────────────────────────────────────────────────
      leftRail.querySelectorAll('[data-action="track-flight"]').forEach((row) => {{
        row.addEventListener('click', () => {{
          const icao24 = row.getAttribute('data-icao24');
          const flight = FLIGHTS.find((f) => f.icao24 === icao24);
          if (flight && _mapRef) setActiveFlight(_mapRef, flight);
        }});
      }});

      leftRail.querySelectorAll('[data-action="zoom-airport"]').forEach((row) => {{
        row.addEventListener('click', () => {{
          const lat = parseFloat(row.getAttribute('data-lat'));
          const lng = parseFloat(row.getAttribute('data-lng'));
          if (_mapRef && Number.isFinite(lat) && Number.isFinite(lng)) {{
            _mapRef.flyTo({{ center: [lng, lat], zoom: 10.5, duration: 1100, essential: true }});
          }}
        }});
      }});

      leftRail.querySelectorAll('[data-action="track-route"]').forEach((row) => {{
        row.addEventListener('click', () => {{
          const icao24 = row.getAttribute('data-icao24');
          const flight = icao24 ? FLIGHTS.find((f) => f.icao24 === icao24) : null;
          if (flight && _mapRef) setActiveFlight(_mapRef, flight);
        }});
      }});

      // Airline scorecard expand toggle
      leftRail.querySelectorAll('[data-al-idx]').forEach((row) => {{
        row.addEventListener('click', () => {{
          const idx = row.getAttribute('data-al-idx');
          const sc = document.getElementById(`al-sc-${{idx}}`);
          if (sc) sc.classList.toggle('open');
        }});
      }});
    }}

    function renderRightPanel(flight, airportCounts, routeCounts) {{
      const panel = document.getElementById('right-panel');
      if (!panel || !flight) return;
      const arrivalAirport = AIRPORT_LOOKUP[flight.arr] || null;
      const nearest = getNearestAirport(flight);
      const now = new Date();
      // Use embedded coords if destination is outside the Indian lookup
      const arrLat = arrivalAirport ? arrivalAirport.lat : flight.arr_lat;
      const arrLng = arrivalAirport ? arrivalAirport.lng : flight.arr_lng;
      const hasArrCoords = arrLat != null && arrLng != null;

      // ── ETA: prefer live schedule eta_ts, then sta_ts+delayed, then backend heuristic ──
      // If schedule-based ETA is already in the past (stale AirLabs data), fall back to heuristic.
      const _heuristicEta = () => {{
        const etaMinutes = Math.max(3, Math.round(
          (nearest.distanceKm || 140) / Math.max(Number(flight.speed_kts || 300), 180) * 60
        ));
        const predMin = Number(flight.pred_delay_min || 0);
        return new Date(now.getTime() + (etaMinutes + predMin) * 60000);
      }};
      let eta;
      let etaSource = 'backend';
      if (flight.eta_ts) {{
        const _schedEta = new Date(flight.eta_ts * 1000);
        if (_schedEta > now) {{
          eta = _schedEta;
          etaSource = 'schedule';
        }} else {{
          eta = _heuristicEta();
        }}
      }} else if (flight.sta_ts && flight.delayed_min != null) {{
        const _schedEta = new Date((flight.sta_ts + flight.delayed_min * 60) * 1000);
        if (_schedEta > now) {{
          eta = _schedEta;
          etaSource = 'schedule';
        }} else {{
          eta = _heuristicEta();
        }}
      }} else {{
        eta = _heuristicEta();
      }}

      // ── Delay: compute from live timestamps first (most accurate), fall back to AirLabs field, then backend prediction ──
      // eta_ts = arr_estimated_ts (carrier's live expected arrival)
      // sta_ts = arr_time_ts     (original scheduled arrival)
      // Delay = eta_ts - sta_ts  → exact minutes late vs scheduled arrival time
      // Only use eta_ts for delay if it's not stale (i.e. in the future)
      const tsDelayMin = (flight.eta_ts && flight.sta_ts && new Date(flight.eta_ts * 1000) > now)
        ? Math.round((flight.eta_ts - flight.sta_ts) / 60)
        : null;
      const airlabsDelayMin   = (flight.delayed_min != null && flight.delayed_min > 0) ? Number(flight.delayed_min) : null;
      const backendDelayMin   = Number(flight.pred_delay_min || 0);
      const backendRisk       = (flight.pred_delay_risk || 'Low');
      // Priority: live timestamp diff > AirLabs field > backend heuristic
      const displayDelay = tsDelayMin != null
        ? Math.max(0, tsDelayMin)
        : (airlabsDelayMin != null ? airlabsDelayMin : backendDelayMin);
      const displayLevel      = backendRisk.toLowerCase();
      const displayLevelLabel = backendRisk;
      const dataSourceLabel   = tsDelayMin != null
        ? '🕐 Live · ETA vs scheduled arrival'
        : (airlabsDelayMin != null ? '📋 Schedule-verified · AirLabs' : '✅ Backend analytics · 11 live signals');

      const istEta = new Date(eta.getTime() + 330 * 60000);
      const istH = String(istEta.getUTCHours()).padStart(2, '0');
      const istM = String(istEta.getUTCMinutes()).padStart(2, '0');
      const etaLabel = istH + ':' + istM + ' IST';
      const remaining = hasArrCoords ? Math.round(haversineKm(flight.lat, flight.lng, arrLat, arrLng)) : nearest.distanceKm;

      // ── On-final detection — suppress delay score for aircraft already landing ──
      const onFinal = (remaining != null && remaining < 50)
        && (flight.altitude_ft > 0)
        && (flight.altitude_ft < 10000);

      // ── Backend delay reason signals → chip rows ───────────────────────────
      const predReasonStr  = flight.pred_delay_reason || '';
      const predReasonTags = predReasonStr
        ? predReasonStr.split(',').map(function(s) {{ return s.trim(); }}).filter(Boolean)
        : [];
      const backendSignalRows = (predReasonTags.length
        ? predReasonTags.map(function(r) {{ return (
            '<div class="delay-signal-row">'
            + '<span class="delay-signal-icon">⚡</span>'
            + '<span class="delay-signal-label">' + escapeHtml(r) + '</span>'
            + '</div>');
          }}).join('')
        : '<div class="delay-signal-row"><span class="delay-signal-icon">✅</span><span class="delay-signal-label">Stable operating profile</span></div>')
        

      const delayLevel = displayLevel;
      const statusText = (flight.status || 'Unknown');
      const isEnRoute = statusText.toLowerCase().includes('en');
      const statusChipClass = isEnRoute ? 'status-enroute' : 'status-ground';
      const depCity = (AIRPORT_LOOKUP[flight.dep] && AIRPORT_LOOKUP[flight.dep].city) || 'Departure';
      const arrCity = (arrivalAirport && arrivalAirport.city) || 'Destination';
      const nearestName = (nearest.airport && nearest.airport.city) || 'Nearest';

      // ── Stale data warning ─────────────────────────────────────────────────
      const staleMin = Number(flight.stale_min || 0);
      const staleWarn = staleMin >= 5 ? `
        <div class="stale-warn">⚠ Position data ${{Math.round(staleMin)}} min stale — may be unreliable</div>` : '';

      // ── Anomaly banners (holding / go-around / diversion) ──────────────────
      const holdingWarn   = flight.pred_holding   ? `<div class="stale-warn" style="background:rgba(255,170,50,0.15);border-color:rgba(255,170,50,0.35);">🔁 Holding pattern detected — additional delay expected</div>` : '';
      const goAroundWarn  = flight.pred_go_around ? `<div class="stale-warn" style="background:rgba(255,110,40,0.15);border-color:rgba(255,110,40,0.35);">↩ Go-around detected — missed approach, retry in progress</div>` : '';
      const diversionWarn = flight.pred_diversion ? `<div class="stale-warn" style="background:rgba(220,30,30,0.18);border-color:rgba(220,30,30,0.5);color:#ff8a8a;">⚠ Diversion suspected — flight may be heading to alternate</div>` : '';

      // ── Weather badge ──────────────────────────────────────────────────────
      const wxSev  = flight.weather_sev || 'Unknown';
      const wxIcon = wxSev === 'Severe' ? '⛈' : wxSev === 'Moderate' ? '🌦' : wxSev === 'Unknown' ? '❓' : '🌤';
      const wxLabel = wxSev === 'Unknown' ? 'N/A' : wxSev;

      // ── AFRI (Arrival Flow Risk Index, 0–100) ──────────────────────────────
      const afriScore   = Number(flight.afri_score || 0);
      const afriLevel   = flight.afri_level || 'Normal';
      const afriDrivers = flight.afri_drivers || 'stable arrival';

      // ── Fuel / CO₂ estimate — prefer backend (accurate), fall back to JS ──
      const _FUEL_RATE = {{
        b747:8.8,a380:9.5,a340:7.0,b777:6.5,a350:5.0,a330:6.2,
        b787:5.2,b767:5.8,b757:4.8,a320:2.8,b737:2.9,
        e190:2.4,crj:2.1,atr:1.2,default:3.0,
      }};
      const _SEATS = {{b747:416,a380:555,b777:396,a350:369,a330:277,b787:296,b737:162,a320:150,e190:110,crj:70,atr:68}};
      const fuelRate   = _FUEL_RATE[flight.family] || _FUEL_RATE.default;
      const depAp2     = AIRPORT_LOOKUP[flight.dep];
      const arrLat2    = arrLat;
      const arrLng2    = arrLng;
      const depHasCoords = depAp2 != null;
      const totalRouteKm = (depHasCoords && hasArrCoords)
        ? haversineKm(depAp2.lat, depAp2.lng, arrLat2, arrLng2) * 1.08
        : (remaining ? remaining * 1.3 : null);
      // ── Backend values override JS heuristic when available ───────────────
      const fuelTotal  = flight.fuel_burn_kg  != null ? flight.fuel_burn_kg  : (totalRouteKm ? Math.round(fuelRate * totalRouteKm) : null);
      const co2Total   = flight.fuel_co2_kg   != null ? flight.fuel_co2_kg   : (fuelTotal ? Math.round(fuelTotal * 3.16) : null);
      const seats2     = _SEATS[flight.family];
      const co2pp      = flight.fuel_co2_pax  != null ? flight.fuel_co2_pax  : ((co2Total && seats2) ? Math.round(co2Total / seats2) : null);
      const fuelRem    = (remaining && fuelRate) ? Math.round(fuelRate * remaining * 1.08) : null;
      const effLabel   = flight.fuel_eff || (co2pp == null ? 'Unknown' : co2pp < 80 ? 'Excellent' : co2pp < 140 ? 'Good' : co2pp < 220 ? 'Average' : 'High');
      const fuelSource = flight.fuel_burn_kg != null ? 'backend (phased corrections)' : 'JS estimate';

      panel.style.display = 'block';
      panel.classList.remove('rp-visible');
      void panel.offsetWidth;
      panel.classList.add('rp-visible');
      // On mobile: show the map shield so touches on the map area
      // can't reach the Mapbox canvas and interfere with panel scroll.
      // Tapping the shield closes the panel (backdrop-dismiss UX).
      if (window.innerWidth <= 600) {{
        panel.scrollTop = 0;
        const shield = document.getElementById('map-shield');
        if (shield) {{
          shield.style.display = 'block';
          shield.onclick = function() {{ hideRightPanel(); }};
        }}
      }}

      panel.innerHTML = `
        <div class="panel-top-accent"></div>
        <div class="panel-inner">
          <div class="panel-head-row">
            <div class="panel-live-badge">
              <div class="panel-live-dot"></div>
              Live Tracking
            </div>
            <div style="display:flex;gap:6px;align-items:center;">
              <button class="follow-btn${{_cameraFollow ? ' active' : ''}}" id="follow-btn" type="button" title="Lock camera to this aircraft">&#x1F4CC; Follow</button>
              <button class="panel-close" type="button" data-close-panel aria-label="Close">×</button>
            </div>
          </div>
          ${{staleWarn}}
          ${{diversionWarn}}
          ${{goAroundWarn}}
          ${{holdingWarn}}
          <div class="panel-flight-hero" id="copy-flight-hero" title="Click to copy flight number" style="cursor:pointer">${{escapeHtml(flight.flight)}}</div>
          <div class="panel-airline-sub">${{escapeHtml(flight.airline)}} &nbsp;·&nbsp; ${{escapeHtml(flight.aircraft)}}</div>
          <div class="panel-chip-row">
            <div class="pchip">${{escapeHtml(flight.airline_code)}}</div>
            <div class="pchip">${{escapeHtml(flight.aircraft)}}</div>
            <div class="pchip ${{statusChipClass}}">${{escapeHtml(statusText)}}</div>
            ${{(() => {{ const ph = flightPhase(flight); return `<div class="pchip ${{ph.css}}" title="Phase of flight">${{ph.icon}} ${{ph.label}}</div>`; }})()}}
            <div class="wx-badge ${{wxSev}}">${{wxIcon}} ${{wxLabel}}</div>
            ${{(() => {{
              const mlP = Number(flight.pred_ml_prob || 0);
              if (mlP <= 0) return '';
              const mlPct = Math.round(mlP * 100);
              const mlCls = mlPct >= 65 ? 'High' : mlPct >= 35 ? 'Medium' : 'Low';
              const mlClr = mlPct >= 65 ? '#ff6b6b' : mlPct >= 35 ? '#ffb347' : '#4ecdc4';
              return `<div class="pchip" title="XGBoost model probability of delay on this route/time (historical pattern)" style="border-color:${{mlClr}};color:${{mlClr}};background:rgba(0,0,0,0.25);">🤖 ${{mlPct}}% ML</div>`;
            }})()}}
          </div>

          <div class="panel-section-divider"></div>
          <div class="panel-section-label">Route</div>
          <div class="route-display">
            <div class="route-apt">
              <div class="route-apt-code">${{escapeHtml(flight.dep)}}</div>
              <div class="route-apt-city">${{escapeHtml(depCity)}}</div>
            </div>
            <div class="route-center">
              <div class="route-plane-icon">✈</div>
              <div class="route-dash-line"></div>
            </div>
            <div class="route-apt" style="text-align:right;">
              <div class="route-apt-code">${{escapeHtml(flight.arr)}}</div>
              <div class="route-apt-city" style="text-align:right;">${{escapeHtml(arrCity)}}</div>
            </div>
          </div>

          <div class="panel-section-divider"></div>
          <div class="panel-section-label">Live Telemetry</div>
          <div class="metrics-grid-4">
            <div class="mcard">
              <div class="mcard-label">Altitude</div>
              <div class="mcard-value">${{formatNumber(flight.altitude_ft)}}<span class="mcard-unit">ft</span></div>
            </div>
            <div class="mcard">
              <div class="mcard-label">Airspeed</div>
              <div class="mcard-value">${{formatNumber(flight.speed_kts)}}<span class="mcard-unit">kts</span></div>
            </div>
            <div class="mcard">
              <div class="mcard-label">Heading</div>
              <div class="mcard-value">${{formatNumber(flight.heading)}}<span class="mcard-unit">°</span></div>
            </div>
            <div class="mcard">
              <div class="mcard-label">V/S</div>
              <div class="mcard-value">${{flight.v_speed == null ? '—' : Math.abs(flight.v_speed) < 0.5 ? 'LVL' : (flight.v_speed > 0 ? '▲ ' : '▼ ') + Math.abs(Math.round(flight.v_speed * 196.85)).toLocaleString()}}<span class="mcard-unit">${{flight.v_speed == null || Math.abs(flight.v_speed) < 0.5 ? '' : 'fpm'}}</span></div>
            </div>
            <div class="mcard" style="grid-column: 1 / -1;">
              <div class="mcard-label">Registration</div>
              ${{(() => {{
                const reg = (flight.registration && flight.registration !== 'N/A') ? flight.registration : null;
                const hex = (flight.icao24 && flight.icao24 !== 'N/A') ? flight.icao24.toUpperCase() + ' (hex)' : null;
                const regStr = reg || hex || '—';
                const c = reg ? regToCountry(reg) : null;
                return `<div class="mcard-value" style="font-size:12px;letter-spacing:0;display:flex;align-items:center;gap:6px;">`
                  + `<span>${{escapeHtml(regStr)}}</span>`
                  + (c ? `<span style="font-size:16px;" title="${{escapeHtml(c.country)}}">${{c.flag}}</span><span style="font-size:10px;color:rgba(244,247,251,0.45);">${{escapeHtml(c.country)}}</span>` : '')
                  + `</div>`;
              }})()}}
            </div>
          </div>

          <div class="pos-card">
            <div class="pos-card-label">◉ Live Position</div>
            <div class="pos-card-value">${{escapeHtml(String(remaining || 'N/A'))}} km to destination</div>
            <div class="pos-card-sub">${{escapeHtml(nearestName)}} nearest &nbsp;·&nbsp; ${{escapeHtml(String(nearest.distanceKm || 'N/A'))}} km away</div>
          </div>

          ${{(() => {{
            // Altitude sparkline built from TRAILS history for this aircraft
            const pts = TRAILS[flight.icao24];
            if (!pts || pts.length < 3) return '';
            const alts = pts.map(p => Number(p.altitude_ft || 0)).filter(a => a > 0);
            if (alts.length < 3) return '';
            const minA = Math.min(...alts), maxA = Math.max(...alts);
            const range = maxA - minA || 1;
            const W = 220, H = 40, pad = 2;
            const xs = alts.map((_, i) => pad + (i / (alts.length - 1)) * (W - pad * 2));
            const ys = alts.map(a => H - pad - ((a - minA) / range) * (H - pad * 2));
            const polyline = xs.map((x, i) => `${{x.toFixed(1)}},${{ys[i].toFixed(1)}}`).join(' ');
            // Current altitude dot (last point)
            const cx = xs[xs.length - 1], cy = ys[ys.length - 1];
            const curAlt = alts[alts.length - 1];
            const trend = alts.length >= 2
              ? (alts[alts.length-1] > alts[alts.length-2] ? '▲' : alts[alts.length-1] < alts[alts.length-2] ? '▼' : '—')
              : '—';
            return `
            <div class="sparkline-card">
              <div class="sparkline-label">
                <span>Altitude History</span>
                <span style="color:rgba(244,247,251,0.6);font-size:9px;font-weight:700;letter-spacing:0;">${{trend}} ${{(curAlt/100).toFixed(0)}} FL &nbsp;·&nbsp; ${{alts.length}} pts</span>
              </div>
              <svg class="sparkline-svg" viewBox="0 0 ${{W}} ${{H}}" preserveAspectRatio="none">
                <polyline points="${{polyline}}" fill="none" stroke="rgba(0,229,255,0.25)" stroke-width="1.5" stroke-linejoin="round"/>
                <polyline points="${{polyline}}" fill="none" stroke="#00e5ff" stroke-width="1" stroke-linejoin="round" stroke-dasharray="none"/>
                <circle cx="${{cx.toFixed(1)}}" cy="${{cy.toFixed(1)}}" r="3" fill="#00e5ff" stroke="rgba(14,20,30,0.8)" stroke-width="1.5"/>
              </svg>
            </div>`;
          }})()}}

          ${{(() => {{
            // Route demand chart: flights/hr on this dep→arr pair
            if (!flight.dep || !flight.arr) return '';
            const now = Date.now() / 1000;
            const slots = 6;  // 6 x 1-hour buckets covering last 6 hrs
            const buckets = new Array(slots).fill(0);
            const allRouteFlights = FLIGHTS.filter(f => f.dep === flight.dep && f.arr === flight.arr);
            if (allRouteFlights.length < 2) return '';
            // Use sta_ts/std_ts to bucket; fall back to counting all as 1 slot
            allRouteFlights.forEach(f => {{
              const ts = f.std_ts || f.sta_ts;
              if (!ts) {{ buckets[slots-1]++; return; }}
              const hoursAgo = (now - ts) / 3600;
              const idx = Math.min(Math.max(0, Math.floor(hoursAgo)), slots-1);
              buckets[slots-1-idx]++;
            }});
            const maxB = Math.max(...buckets, 1);
            const barHtml = buckets.map((cnt, i) => {{
              const h = Math.max(2, Math.round(cnt / maxB * 100));
              const lbl = (i - slots + 1) + 'h';
              const isNow = i === slots - 1;
              return `<div class="rdb-slot"><div class="rdb-bar${{isNow ? ' active' : ''}}" style="height:${{h}}%"></div><span class="rdb-lbl">${{isNow ? 'now' : (i-slots+1)+'h'}}</span></div>`;
            }}).join('');
            return `<div class="route-demand-card">
              <div class="route-demand-title">Flights on ${{escapeHtml(flight.dep)}} → ${{escapeHtml(flight.arr)}} · ${{allRouteFlights.length}} live</div>
              <div class="route-demand-bars">${{barHtml}}</div>
            </div>`;
          }})()}}

          ${{ACTIVE_MODULE !== 'live_ops' ? `
          <div class="delay-card ${{onFinal ? 'low' : delayLevel}}">
            <div class="delay-card-header">
              <div class="panel-section-label" style="margin:0;">${{onFinal ? 'Arrival' : 'Delay Forecast'}}</div>
              <div class="delay-badge ${{onFinal ? 'low' : displayLevel}}">${{onFinal ? 'On Final' : escapeHtml(displayLevelLabel)}}</div>
            </div>
            <div class="delay-headline">${{onFinal ? '🛬 Arriving' : (displayDelay > 0 ? '+' + displayDelay + ' min' : 'On time')}} &nbsp;·&nbsp; ETA ${{etaLabel}}</div>
            <div class="delay-eta">${{onFinal ? 'Aircraft on final approach — no further delays expected' : dataSourceLabel}}</div>
            <div class="delay-score-bar-wrap">
              <div class="delay-score-bar-fill ${{displayLevel}}" style="width:${{Math.min(100, Math.round(displayDelay / 45 * 100))}}%"></div>
            </div>
            <div class="delay-signals">
              ${{onFinal ? '<div class="delay-signal-row"><span class="delay-signal-icon">🛬</span><span class="delay-signal-label">On final approach — arrival imminent</span></div>' : backendSignalRows}}
            </div>
          </div>

          <div class="afri-card ${{afriLevel}}">
            <div class="afri-card-head">
              <div class="panel-section-label" style="margin:0;">Arrival Flow Risk (AFRI)</div>
              <span class="afri-level-badge ${{afriLevel}}">${{afriLevel}}</span>
            </div>
            <div class="afri-score-row">
              <span class="afri-score-num">${{afriScore}}</span>
              <div class="afri-bar-wrap">
                <div class="afri-bar-fill ${{afriLevel}}" style="width:${{afriScore}}%"></div>
              </div>
              <span style="font-size:9px;color:rgba(244,247,251,0.35);">/100</span>
            </div>
            <div class="afri-drivers">${{escapeHtml(afriDrivers)}}</div>
          </div>` : ''}}

          ${{fuelTotal ? `
          <div class="panel-section-divider"></div>
          <div class="panel-section-label">Fuel &amp; CO₂ Estimate <span style="font-size:8px;color:rgba(244,247,251,0.35);font-weight:400;">${{fuelSource}}</span></div>
          <div class="fuel-card">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:2px;">
              <span style="font-size:10px;color:rgba(244,247,251,0.45);font-weight:600;">${{escapeHtml(flight.aircraft)}} · ${{fuelRate}} kg/km</span>
              <span class="fuel-eff-badge ${{effLabel}}">${{effLabel}}</span>
            </div>
            <div class="fuel-grid">
              <div class="fuel-cell">
                <div class="fuel-cell-label">Total Fuel</div>
                <div class="fuel-cell-value">${{(fuelTotal/1000).toFixed(1)}} t</div>
              </div>
              <div class="fuel-cell">
                <div class="fuel-cell-label">CO₂</div>
                <div class="fuel-cell-value">${{(co2Total/1000).toFixed(1)}} t</div>
              </div>
              <div class="fuel-cell">
                <div class="fuel-cell-label">CO₂/pax</div>
                <div class="fuel-cell-value">${{co2pp != null ? co2pp + ' kg' : '—'}}</div>
              </div>
            </div>
            ${{fuelRem ? `<div style="font-size:9px;color:rgba(244,247,251,0.30);margin-top:5px;">~${{Math.round(fuelRem/1000*10)/10}} t fuel remaining to destination</div>` : ''}}
          </div>` : ''}}

          <div class="panel-section-divider"></div>
          <button class="tech-details-toggle" id="tech-toggle" aria-expanded="false" type="button">
            <span class="panel-section-label" style="margin:0;pointer-events:none;">Technical Details</span>
            <span class="tech-toggle-chevron" id="tech-chevron">▸</span>
          </button>
          <div class="detail-table" id="tech-detail-table" style="display:none;">
            <div class="detail-table-key">Status</div>
            <div class="detail-table-val">${{escapeHtml(statusText)}}</div>
            <div class="detail-table-key">Position</div>
            <div class="detail-table-val">${{escapeHtml(Number(flight.lat).toFixed(4) + '°, ' + Number(flight.lng).toFixed(4) + '°')}}</div>
            ${{(flight.icao24 && flight.icao24 !== 'N/A') ? `
            <div class="detail-table-key">ICAO24</div>
            <div class="detail-table-val">${{escapeHtml(flight.icao24.toUpperCase())}}</div>` : ''}}
            <div class="detail-table-key">Aircraft Type</div>
            <div class="detail-table-val">${{escapeHtml(flight.aircraft)}}</div>
            <div class="detail-table-key">Airline</div>
            <div class="detail-table-val">${{escapeHtml(flight.airline)}}</div>
            ${{flight.pred_delay_risk && flight.pred_delay_risk !== 'Low' ? `
            <div class="detail-table-key">Backend Risk</div>
            <div class="detail-table-val">${{escapeHtml(flight.pred_delay_risk)}} (${{flight.pred_delay_min}}m · ${{escapeHtml(flight.pred_delay_reason)}})</div>` : ''}}
          </div>
        </div>
      `;

      // Tech details toggle
      const techToggle = panel.querySelector('#tech-toggle');
      const techTable  = panel.querySelector('#tech-detail-table');
      const techChevron = panel.querySelector('#tech-chevron');
      if (techToggle && techTable) {{
        techToggle.addEventListener('click', () => {{
          const open = techTable.style.display !== 'none';
          techTable.style.display = open ? 'none' : 'grid';
          if (techChevron) techChevron.textContent = open ? '▸' : '▾';
          techToggle.setAttribute('aria-expanded', String(!open));
        }});
      }}

      // Copy flight number on click
      const copyHero = panel.querySelector('#copy-flight-hero');
      if (copyHero) {{
        copyHero.addEventListener('click', () => {{
          const txt = flight.flight || '';
          if (!txt) return;
          navigator.clipboard.writeText(txt).then(() => {{
            const prev = copyHero.textContent;
            copyHero.textContent = 'Copied!';
            copyHero.style.opacity = '0.7';
            setTimeout(() => {{
              copyHero.textContent = prev;
              copyHero.style.opacity = '';
            }}, 1400);
          }}).catch(() => {{}});
        }});
      }}
    }}

    function hideRightPanel() {{
      const panel = document.getElementById('right-panel');
      if (!panel) return;
      panel.classList.remove('rp-visible');
      panel.style.display = 'none';
      panel.innerHTML = '';
      _cameraFollow = false;
      const shield = document.getElementById('map-shield');
      if (shield) shield.style.display = 'none';
    }}

    // ── SDF icon registration ──────────────────────────────────────────────────
    // We render each SVG family as a Signed Distance Field raster image so that
    // ── Icon registration ──────────────────────────────────────────────────────
    // Rasterize SVG via canvas (data: URI is same-origin → no taint → getImageData OK).
    // Two variants registered per family: ac-<family>-std and ac-<family>-sel.
    // Non-SDF so the baked color shows as-is — no black-background SDF artifacts.
    function _rasterizeToImageData(svgMarkup, size) {{
      return new Promise((resolve) => {{
        const encoded = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgMarkup)));
        const img = new Image(size, size);
        img.onload = () => {{
          try {{
            const canvas = document.createElement('canvas');
            canvas.width = canvas.height = size;
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, size, size);
            ctx.drawImage(img, 0, 0, size, size);
            resolve(ctx.getImageData(0, 0, size, size));
          }} catch (_) {{
            resolve(null);
          }}
        }};
        img.onerror = () => resolve(null);
        img.src = encoded;
      }});
    }}

    async function registerAircraftIcons(map) {{
      const SIZE = 72;
      const VARIANTS = {{
        'std': '#00E5FF',   // bright cyan — normal en-route
        'dim': '#4DB6C7',   // muted cyan  — unfocused
        'sel': '#FF4B2B',   // red-orange  — selected
      }};
      let loaded = 0;
      for (const [family, body] of Object.entries(ICON_TEMPLATES)) {{
        for (const [variant, color] of Object.entries(VARIANTS)) {{
          const name = `ac-${{family}}-${{variant}}`;
          if (map.hasImage(name)) {{ loaded++; continue; }}
          const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="${{SIZE}}" height="${{SIZE}}">${{body.replaceAll('COLORTOK', color)}}</svg>`;
          const imgData = await _rasterizeToImageData(svg, SIZE);
          if (imgData) {{
            try {{
              map.addImage(name, {{ width: SIZE, height: SIZE, data: new Uint8Array(imgData.data) }});
              loaded++;
            }} catch (_) {{}}
          }}
        }}
      }}
      return loaded > 0;
    }}

    // Build GeoJSON FeatureCollection for the flights layer
    function buildFlightsGeoJson(selectedHex) {{
      const hasQuery = Boolean(SELECTED_QUERY);
      let _flightsForGeo = _acFilter === 'all' ? FLIGHTS : FLIGHTS.filter(f => acCategory(f.family) === _acFilter);
      // Apply alt/speed filters (skip ground-stopped aircraft for speed filter)
      if (_altMin > 0 || _altMax < 45000) {{
        _flightsForGeo = _flightsForGeo.filter(f => {{
          const alt = Number(f.altitude_ft || 0);
          return alt >= _altMin && alt <= _altMax;
        }});
      }}
      if (_spdMin > 0 || _spdMax < 600) {{
        _flightsForGeo = _flightsForGeo.filter(f => {{
          const spd = Number(f.speed_kts || 0);
          return spd >= _spdMin && spd <= _spdMax;
        }});
      }}
      return {{
        type: 'FeatureCollection',
        features: _flightsForGeo.map((f) => {{
          const isSel = Boolean(f.icao24 && f.icao24 === selectedHex);
          const variant = isSel ? 'sel' : (hasQuery && !isSel ? 'dim' : 'std');
          // Use interpolated position when available, fall back to raw API coords
          const lat = f._iLat !== undefined ? f._iLat : f.lat;
          const lng = f._iLng !== undefined ? f._iLng : f.lng;
          const hdg = f._iHeading !== undefined ? f._iHeading : (f.heading || 0);
          return {{
            type: 'Feature',
            geometry: {{ type: 'Point', coordinates: [lng, lat] }},
            properties: {{
              icao24:   f.icao24 || '',
              flight:   f.flight || '',
              family:   f.family || 'default',
              heading:  hdg,
              altitude_ft: f.altitude_ft,
              speed_kts:   f.speed_kts,
              dep:      f.dep || '',
              arr:      f.arr || '',
              airline:  f.airline || '',
              airline_code: f.airline_code || '',
              aircraft: f.aircraft || '',
              status:   f.status || '',
              registration: f.registration || '',
              lat:      lat,
              lng:      lng,
              variant:  variant,
              selected: isSel,
            }},
          }};
        }}),
      }};
    }}

    function addFlightsLayer(map) {{
      map.addSource('flights-src', {{
        type: 'geojson',
        data: buildFlightsGeoJson(null),
        tolerance: 0,
      }});

      // ── Primary: circle layer ─────────────────────────────────────────────
      // Circles ALWAYS render — no image dependency. This is the guaranteed
      // fallback even if SDF icon registration fails.
      map.addLayer({{
        id: 'flights-circles',
        type: 'circle',
        source: 'flights-src',
        paint: {{
          'circle-radius': [
            'interpolate', ['linear'], ['zoom'],
            2, 2.5, 5, 3.8, 8, 5.5, 12, 8.0,
          ],
          'circle-color': [
            'case',
            ['==', ['get', 'variant'], 'sel'], '#FF4B2B',
            ['==', ['get', 'variant'], 'dim'], '#26C6DA',
            '#00E5FF',
          ],
          'circle-opacity': ['case', ['==', ['get', 'variant'], 'dim'], 0.72, 0.95],
          'circle-stroke-color': [
            'case',
            ['==', ['get', 'variant'], 'sel'], 'rgba(255,75,43,0.52)',
            'rgba(0,8,18,0.90)',
          ],
          'circle-stroke-width': [
            'interpolate', ['linear'], ['zoom'],
            3, ['case', ['==', ['get', 'variant'], 'sel'], 3.0, 1.0],
            10, ['case', ['==', ['get', 'variant'], 'sel'], 4.0, 1.5],
          ],
        }},
      }});

      // ── Label layer ───────────────────────────────────────────────────────
      map.addLayer({{
        id: 'flights-labels',
        type: 'symbol',
        source: 'flights-src',
        layout: {{
          'text-field': ['get', 'flight'],
          'text-size': ['interpolate', ['linear'], ['zoom'], 8, 10, 13, 13],
          'text-font': ['DIN Offc Pro Medium', 'Arial Unicode MS Regular'],
          'text-anchor': 'top',
          'text-offset': [0, 0.8],
          'text-allow-overlap': false,
          'text-optional': true,
        }},
        filter: [
          'any',
          ['==', ['get', 'variant'], 'sel'],
          ['>=', ['zoom'], 8],
        ],
        paint: {{
          'text-color': [
            'case',
            ['==', ['get', 'variant'], 'sel'], '#FF6B50',
            '#7FDDEA',
          ],
          'text-halo-color': 'rgba(3,7,13,0.96)',
          'text-halo-width': 2,
        }},
      }});

      // ── Interactions on circle layer ──────────────────────────────────────
      map.on('click', 'flights-circles', (e) => {{
        if (!e.features || !e.features.length) return;
        const icao24 = e.features[0].properties.icao24;
        const flight = FLIGHTS.find((f) => f.icao24 === icao24);
        if (flight) setActiveFlight(map, flight);
      }});

      let _hoverPopup = null;
      map.on('mouseenter', 'flights-circles', (e) => {{
        map.getCanvas().style.cursor = 'pointer';
        if (!e.features || !e.features.length) return;
        const props = e.features[0].properties;
        const flight = FLIGHTS.find((f) => f.icao24 === props.icao24);
        if (!flight) return;
        if (_hoverPopup) {{ _hoverPopup.remove(); _hoverPopup = null; }}
        _hoverPopup = new mapboxgl.Popup({{
          closeButton: false,
          closeOnClick: false,
          offset: 14,
          maxWidth: '370px',
          className: 'flight-hover-popup',
        }})
          .setLngLat([flight.lng, flight.lat])
          .setHTML(buildPopupHtml(flight))
          .addTo(map);
      }});
      map.on('mouseleave', 'flights-circles', () => {{
        map.getCanvas().style.cursor = '';
        if (_hoverPopup) {{ _hoverPopup.remove(); _hoverPopup = null; }}
      }});
    }}

    function addFlightIconsLayer(map) {{
      // Optional silhouette overlay — non-SDF, pre-colored per variant.
      // Uses 'ac-<family>-std/dim/sel' images registered by registerAircraftIcons.
      if (map.getLayer('flights-icons')) return;
      map.addLayer({{
        id: 'flights-icons',
        type: 'symbol',
        source: 'flights-src',
        layout: {{
          'icon-image': ['concat', 'ac-', ['get', 'family'], '-', ['get', 'variant']],
          'icon-size': [
            'interpolate', ['linear'], ['zoom'],
            3, 0.28, 6, 0.44, 9, 0.60, 12, 0.76,
          ],
          'icon-rotate': ['get', 'heading'],
          'icon-rotation-alignment': 'map',
          'icon-allow-overlap': true,
          'icon-ignore-placement': true,
          'icon-pitch-alignment': 'map',
        }},
        paint: {{
          'icon-opacity': ['case', ['==', ['get', 'variant'], 'dim'], 0.78, 1.0],
        }},
      }});
      map.on('click', 'flights-icons', (e) => {{
        if (!e.features || !e.features.length) return;
        const icao24 = e.features[0].properties.icao24;
        const flight = FLIGHTS.find((f) => f.icao24 === icao24);
        if (flight) setActiveFlight(map, flight);
      }});

      // ── Background click: deselect / close right panel ───────────────────
      map.on('click', (e) => {{
        const hits = map.queryRenderedFeatures(e.point, {{
          layers: ['flights-circles', 'flights-icons', 'airport-points', 'airport-hub-points'],
        }});
        if (!hits.length && ACTIVE_SELECTION) {{
          hideRightPanel();
          ACTIVE_SELECTION = null;
          refreshFlightsLayer(map, null);
          const empty = {{ type: 'FeatureCollection', features: [] }};
          const r = map.getSource('selected-route'); if (r) r.setData(empty);
          const p = map.getSource('route-progress-src'); if (p) p.setData(empty);
        }}
      }});
    }}

    function refreshFlightsLayer(map, selectedFlight) {{
      const src = map.getSource('flights-src');
      if (!src) return;
      const selectedHex = selectedFlight ? (selectedFlight.icao24 || null) : null;
      // All layers (circles, labels, icons) share the same source — one setData updates all.
      src.setData(buildFlightsGeoJson(selectedHex));
    }}

    function ensureSelectedRouteLayer(map) {{
      if (ROUTE_SOURCE_READY) return;
      // Arc line (remaining route — dep→current→arr)
      map.addSource('selected-route', {{
        type: 'geojson',
        data: {{ type: 'FeatureCollection', features: [] }},
      }});
      map.addLayer({{
        id: 'selected-route-flown',
        type: 'line',
        source: 'selected-route',
        filter: ['==', ['get', 'part'], 'flown'],
        paint: {{
          'line-color': 'rgba(255,60,111,0.25)',
          'line-width': 2,
          'line-dasharray': [2, 4],
        }},
      }});
      map.addLayer({{
        id: 'selected-route-line',
        type: 'line',
        source: 'selected-route',
        filter: ['==', ['get', 'part'], 'remaining'],
        paint: {{
          'line-color': '#ff3c6f',
          'line-width': 2.5,
          'line-opacity': 0.90,
          'line-dasharray': [2, 3],
        }},
      }});
      // Progress dot on the arc
      map.addSource('route-progress-src', {{
        type: 'geojson',
        data: {{ type: 'FeatureCollection', features: [] }},
      }});
      map.addLayer({{
        id: 'route-progress-dot',
        type: 'circle',
        source: 'route-progress-src',
        paint: {{
          'circle-radius': 6,
          'circle-color': '#ff3c6f',
          'circle-stroke-width': 2,
          'circle-stroke-color': '#fff',
          'circle-opacity': 0.95,
        }},
      }});
      ROUTE_SOURCE_READY = true;
    }}

    // Great-circle interpolation (N intermediate points)
    function _gcInterp(lng1, lat1, lng2, lat2, n) {{
      const toRad = (d) => d * Math.PI / 180;
      const toDeg = (r) => r * 180 / Math.PI;
      const φ1 = toRad(lat1), λ1 = toRad(lng1);
      const φ2 = toRad(lat2), λ2 = toRad(lng2);
      const d = 2 * Math.asin(Math.sqrt(
        Math.sin((φ2-φ1)/2)**2 + Math.cos(φ1)*Math.cos(φ2)*Math.sin((λ2-λ1)/2)**2
      ));
      if (d < 1e-9) return [[lng1, lat1], [lng2, lat2]];
      const pts = [];
      for (let i = 0; i <= n; i++) {{
        const f = i / n;
        const A = Math.sin((1-f)*d) / Math.sin(d);
        const B = Math.sin(f*d) / Math.sin(d);
        const x = A*Math.cos(φ1)*Math.cos(λ1) + B*Math.cos(φ2)*Math.cos(λ2);
        const y = A*Math.cos(φ1)*Math.sin(λ1) + B*Math.cos(φ2)*Math.sin(λ2);
        const z = A*Math.sin(φ1) + B*Math.sin(φ2);
        pts.push([toDeg(Math.atan2(y,x)), toDeg(Math.atan2(z, Math.sqrt(x*x+y*y)))]);
      }}
      return pts;
    }}

    function updateSelectedRoute(map, flight) {{
      ensureSelectedRouteLayer(map);
      const routeSrc = map.getSource('selected-route');
      const progSrc  = map.getSource('route-progress-src');
      const empty = {{ type: 'FeatureCollection', features: [] }};
      const arrAp  = AIRPORT_LOOKUP[flight.arr];
      const depAp  = AIRPORT_LOOKUP[flight.dep];
      if (!routeSrc || !arrAp) {{
        routeSrc && routeSrc.setData(empty);
        progSrc  && progSrc.setData(empty);
        return;
      }}
      // Remaining arc: flight position → destination
      const remCoords = _gcInterp(flight.lng, flight.lat, arrAp.lng, arrAp.lat, 60);
      const features = [{{
        type: 'Feature',
        geometry: {{ type: 'LineString', coordinates: remCoords }},
        properties: {{ part: 'remaining' }},
      }}];
      // Flown arc: departure → flight position (if dep coords known)
      if (depAp) {{
        const flownCoords = _gcInterp(depAp.lng, depAp.lat, flight.lng, flight.lat, 40);
        features.push({{
          type: 'Feature',
          geometry: {{ type: 'LineString', coordinates: flownCoords }},
          properties: {{ part: 'flown' }},
        }});
      }}
      routeSrc.setData({{ type: 'FeatureCollection', features }});
      // Progress dot at flight position
      progSrc && progSrc.setData({{
        type: 'FeatureCollection',
        features: [{{
          type: 'Feature',
          geometry: {{ type: 'Point', coordinates: [flight.lng, flight.lat] }},
          properties: {{}},
        }}],
      }});
    }}

    function setActiveFlight(map, flight) {{
      ACTIVE_SELECTION = flight;
      const airportCounts = computeAirportCounts();
      const routeCounts = computeRouteCounts();
      const mostDelayed = getMostDelayedFlights(airportCounts, routeCounts);
      refreshFlightsLayer(map, flight);
      renderTopStats();
      renderLeftRail(airportCounts, routeCounts, mostDelayed);
      renderRightPanel(flight, airportCounts, routeCounts);
      // Wire up follow button
      const followBtn = document.getElementById('follow-btn');
      if (followBtn) {{
        followBtn.addEventListener('click', () => {{
          _cameraFollow = !_cameraFollow;
          followBtn.classList.toggle('active', _cameraFollow);
          if (_cameraFollow && ACTIVE_SELECTION && _mapRef) {{
            _mapRef.easeTo({{ center: [ACTIVE_SELECTION.lng, ACTIVE_SELECTION.lat], zoom: Math.max(_mapRef.getZoom(), 8), duration: 600, essential: true }});
          }}
        }});
      }}
      updateSelectedRoute(map, flight);
      map.easeTo({{
        center: [flight.lng, flight.lat],
        zoom: Math.max(map.getZoom(), 6.9),
        duration: 900,
        offset: ACTIVE_MODULE === 'live_ops' ? [0, 0] : [-150, 0],
        essential: true,
      }});
      const searchInput = document.getElementById('flight-search-input');
      if (searchInput) searchInput.value = flight.flight || '';
      withTopUrl((url) => {{
        if (flight.flight) {{
          url.searchParams.set('flight', flight.flight);
        }} else {{
          url.searchParams.delete('flight');
        }}
      }}, false);
    }}

    function switchModule(map, moduleKey) {{
      if (!moduleKey) return;
      ACTIVE_MODULE = moduleKey;

      // Show/hide flight board overlay
      const boardOverlay = document.getElementById('flight-board-overlay');
      if (boardOverlay) {{
        boardOverlay.classList.toggle('visible', moduleKey === 'flight_board');
        if (moduleKey === 'flight_board') {{
          renderFlightBoard();
        }}
      }}

      // Update active tab button
      document.querySelectorAll('[data-module-btn]').forEach((btn) => {{
        btn.classList.toggle('active', btn.getAttribute('data-module') === moduleKey);
      }});

      // Re-render left rail with new module context
      const airportCounts = computeAirportCounts();
      const routeCounts = computeRouteCounts();
      const mostDelayed = getMostDelayedFlights(airportCounts, routeCounts);
      renderLeftRail(airportCounts, routeCounts, mostDelayed);
      if (ACTIVE_MODULE === 'flight_board') renderFlightBoard();

      // Re-render right panel if a flight is selected (delay card visibility depends on module)
      if (ACTIVE_SELECTION) {{
        renderRightPanel(ACTIVE_SELECTION, airportCounts, routeCounts);
        // Update route line (live_ops hides it, others show it)
        updateSelectedRoute(map, ACTIVE_SELECTION);
      }} else {{
        hideRightPanel();
      }}

      // Silently try to update URL (may fail in sandboxed iframe — that's fine)
      try {{
        withTopUrl((url) => {{ url.searchParams.set('module', moduleKey); }}, false);
      }} catch (_) {{}}
    }}

    // ── Flight Board (FIDS) ───────────────────────────────────────────────────
    let _fidsSortKey   = 'flight';
    let _fidsSortAsc   = true;
    let _fidsFilter    = 'all';   // 'all' | 'arr' | 'dep'

    function _fidsFormatTime(ts) {{
      if (!ts) return '—';
      const d = new Date(ts * 1000);
      const ist = new Date(d.getTime() + 330 * 60000);
      return String(ist.getUTCHours()).padStart(2,'0') + ':' + String(ist.getUTCMinutes()).padStart(2,'0');
    }}

    function _fidsDelayChip(delayMin, onFinal) {{
      if (onFinal) return '<span class="fids-chip ontime">On Final</span>';
      if (delayMin == null) return '<span class="fids-delay-num zero">—</span>';
      if (delayMin > 0)  return `<span class="fids-delay-num pos">+${{delayMin}}m</span>`;
      if (delayMin < 0)  return `<span class="fids-delay-num neg">${{delayMin}}m</span>`;
      return '<span class="fids-chip ontime">On Time</span>';
    }}

    function _fidsStatusChip(status, delayMin) {{
      const s = (status || '').toLowerCase();
      if (s.includes('land')) return '<span class="fids-chip landed">Landed</span>';
      if (s.includes('en') || s.includes('route')) {{
        if (delayMin != null && delayMin >= 10) return '<span class="fids-chip delayed">Delayed</span>';
        return '<span class="fids-chip enroute">En Route</span>';
      }}
      return '<span class="fids-chip ground">Ground</span>';
    }}

    function renderFlightBoard() {{
      const wrap = document.getElementById('fids-tbody');
      if (!wrap) return;

      const airportCounts = computeAirportCounts();
      const routeCounts   = computeRouteCounts();

      // Filter
      let rows = FLIGHTS.filter((f) => {{
        if (_fidsFilter === 'arr') return !!f.arr && f.arr !== 'N/A';
        if (_fidsFilter === 'dep') return !!f.dep && f.dep !== 'N/A';
        return true;
      }});
      if (_acFilter !== 'all') rows = rows.filter(f => acCategory(f.family) === _acFilter);

      // Sort
      rows = rows.slice().sort((a, b) => {{
        let av, bv;
        switch (_fidsSortKey) {{
          case 'flight':    av = a.flight || ''; bv = b.flight || ''; break;
          case 'dep':       av = a.dep || ''; bv = b.dep || ''; break;
          case 'arr':       av = a.arr || ''; bv = b.arr || ''; break;
          case 'airline':   av = a.airline_code || ''; bv = b.airline_code || ''; break;
          case 'sta':       av = a.sta_ts || 0; bv = b.sta_ts || 0; break;
          case 'delayed':   av = (a.delayed_min == null ? -999 : a.delayed_min);
                            bv = (b.delayed_min == null ? -999 : b.delayed_min); break;
          case 'speed':     av = a.speed_kts || 0; bv = b.speed_kts || 0; break;
          case 'afri':      av = a.afri_score || 0; bv = b.afri_score || 0; break;
          default:          av = ''; bv = '';
        }}
        if (av < bv) return _fidsSortAsc ? -1 : 1;
        if (av > bv) return _fidsSortAsc ?  1 : -1;
        return 0;
      }});

      if (!rows.length) {{
        wrap.innerHTML = '<tr><td colspan="11" class="fids-empty">No flights matching current filter</td></tr>';
      }} else {{
        wrap.innerHTML = rows.map((f) => {{
          const arrAp    = AIRPORT_LOOKUP[f.arr] || null;
        const distKm   = (f.arr_lat != null && f.arr_lng != null)
            ? Math.round(haversineKm(f.lat, f.lng, f.arr_lat, f.arr_lng))
            : (arrAp ? Math.round(haversineKm(f.lat, f.lng, arrAp.lat, arrAp.lng)) : null);
          const onFinal  = distKm != null && distKm < 50 && f.altitude_ft > 0 && f.altitude_ft < 10000;
          const airlabsDel = (f.delayed_min != null && f.delayed_min > 0) ? Number(f.delayed_min) : null;
          const predDel    = Number(f.pred_delay_min || 0);
          const tsDel      = (f.eta_ts && f.sta_ts) ? Math.max(0, Math.round((f.eta_ts - f.sta_ts) / 60)) : null;
          const delMin     = tsDel != null ? tsDel : (airlabsDel != null ? airlabsDel : (predDel > 0 ? predDel : null));
          const staLabel = _fidsFormatTime(f.sta_ts);
          const etaLabel = f.eta_ts ? _fidsFormatTime(f.eta_ts) : (f.sta_ts && delMin ? _fidsFormatTime(f.sta_ts + delMin * 60) : '—');
          const afriLvl  = f.afri_level || 'Normal';
          const afriClr  = afriLvl === 'Critical' ? '#ff6868' : afriLvl === 'High' ? '#ffad50' : afriLvl === 'Elevated' ? '#f0d050' : '#555';
          const wxIcon   = f.weather_sev === 'Severe' ? '⛈' : f.weather_sev === 'Moderate' ? '🌦' : f.weather_sev === 'Unknown' ? '❓' : '🌤';
          const wxClr    = f.weather_sev === 'Severe' ? '#ff6868' : f.weather_sev === 'Moderate' ? '#f0c840' : f.weather_sev === 'Unknown' ? 'rgba(244,247,251,0.25)' : '#4fa';
          return `<tr class="fids-row" data-icao24="${{escapeHtml(f.icao24)}}">
            <td><span class="fids-flight">${{escapeHtml(f.flight)}}</span></td>
            <td><span class="fids-airline">${{escapeHtml(f.airline_code)}}</span></td>
            <td><span class="fids-route">${{escapeHtml(f.dep || '—')}}</span></td>
            <td><span class="fids-route">${{escapeHtml(f.arr || '—')}}</span></td>
            <td style="color:#888;font-size:11px;">${{escapeHtml(staLabel)}}</td>
            <td style="color:#aaa;font-size:11px;">${{escapeHtml(etaLabel)}}</td>
            <td>${{_fidsDelayChip(delMin, onFinal)}}</td>
            <td>${{_fidsStatusChip(f.status, delMin)}}</td>
            <td style="font-size:11px;font-weight:700;color:${{afriClr}};" title="${{escapeHtml(f.afri_drivers || '')}}"><span style="font-size:9px;color:rgba(244,247,251,0.35);">&nbsp;</span>${{f.afri_score || 0}} <span style="font-size:9px;color:${{afriClr}};font-weight:800;">${{escapeHtml(afriLvl)}}</span></td>
            ${{(() => {{ const ph = flightPhase(f); return `<td><span class="pchip ${{ph.css}}" style="padding:2px 6px;">${{ph.icon}} ${{ph.label}}</span></td>`; }})()}}
            <td style="font-size:13px;color:${{wxClr}};">${{wxIcon}}</td>
            <td style="color:#666;font-size:11px;">${{f.speed_kts ? f.speed_kts + ' kts' : '—'}}</td>
          </tr>`;
        }}).join('');
      }}

      // Footer count
      const footer = document.getElementById('fids-count');
      if (footer) footer.textContent = rows.length + ' flights';
    }}

    function bindFlightBoard(map) {{
      const tbody = document.getElementById('fids-tbody');
      if (!tbody) return;   // guard — prevents crash if HTML missing

      // Close button
      const closeBtn = document.getElementById('fids-close-btn');
      if (closeBtn) {{
        closeBtn.addEventListener('click', () => switchModule(map, 'live_ops'));
      }}

      // Print button
      const printBtn = document.getElementById('fids-print-btn');
      if (printBtn) {{
        printBtn.addEventListener('click', () => window.print());
      }}

      // Filter buttons
      document.querySelectorAll('[data-fids-filter]').forEach((btn) => {{
        btn.addEventListener('click', () => {{
          _fidsFilter = btn.getAttribute('data-fids-filter');
          document.querySelectorAll('[data-fids-filter]').forEach((b) => b.classList.remove('active'));
          btn.classList.add('active');
          renderFlightBoard();
        }});
      }});

      // Column header sort click
      document.querySelectorAll('[data-fids-sort]').forEach((th) => {{
        th.addEventListener('click', () => {{
          const key = th.getAttribute('data-fids-sort');
          if (_fidsSortKey === key) {{
            _fidsSortAsc = !_fidsSortAsc;
          }} else {{
            _fidsSortKey = key;
            _fidsSortAsc = true;
          }}
          document.querySelectorAll('[data-fids-sort]').forEach((h) => h.removeAttribute('data-sort-active'));
          th.setAttribute('data-sort-active', _fidsSortAsc ? 'asc' : 'desc');
          renderFlightBoard();
        }});
      }});

      // Row click → fly to flight on map
      tbody.addEventListener('click', (e) => {{
        const row = e.target.closest('.fids-row');
        if (!row) return;
        const icao24 = row.getAttribute('data-icao24');
        const flight = FLIGHTS.find((f) => f.icao24 === icao24);
        if (flight && map) {{
          switchModule(map, 'live_ops');
          setActiveFlight(map, flight);
        }}
      }});
    }}

    function bindChromeInteractions(map) {{
      document.querySelectorAll('[data-module-btn]').forEach((button) => {{
        button.addEventListener('click', () => {{
          const moduleKey = button.getAttribute('data-module') || 'live_ops';
          switchModule(map, moduleKey);
        }});
      }});

      const searchInput = document.getElementById('flight-search-input');
      const searchDropdown = document.getElementById('search-dropdown');
      const searchClear = document.getElementById('search-clear-btn');
      let _ddFocusIdx = -1;

      function _ddRows() {{
        return searchDropdown ? Array.from(searchDropdown.querySelectorAll('.search-dd-row')) : [];
      }}

      function _updateClear() {{
        if (!searchClear || !searchInput) return;
        if (searchInput.value.trim()) {{
          searchClear.classList.add('visible');
        }} else {{
          searchClear.classList.remove('visible');
        }}
      }}

      function _closeDropdown() {{
        if (searchDropdown) searchDropdown.classList.remove('open');
        _ddFocusIdx = -1;
      }}

      function _selectResult(flight) {{
        if (!flight) return;
        _closeDropdown();
        setActiveFlight(map, flight);
      }}

      function _selectAirport(airport) {{
        _closeDropdown();
        if (searchInput) {{ searchInput.value = airport.iata; _updateClear(); }}
        map.flyTo({{ center: [airport.lng, airport.lat], zoom: 9.5, duration: 1100, essential: true }});
      }}

      function _buildDropdown(query) {{
        if (!searchDropdown) return;
        const q = query.trim().toUpperCase().replace(/[ \t]/g, '');
        if (!q) {{ _closeDropdown(); return; }}

        // ── Route search: "BOM DEL" / "BOM→DEL" / "BOM-DEL" ─────────────────
        const routeMatch = query.trim().toUpperCase().match(/^([A-Z]{{3}})\\s*[-→/]\\s*([A-Z]{{3}})$/) ||
                           query.trim().toUpperCase().match(/^([A-Z]{{3}})\\s+([A-Z]{{3}})$/);
        if (routeMatch) {{
          const [, depQ, arrQ] = routeMatch;
          const routeFlights = FLIGHTS.filter((f) => f.dep === depQ && f.arr === arrQ);
          if (routeFlights.length) {{
            const flRows = routeFlights.slice(0, 8).map((f) => {{
              const isSel = ACTIVE_SELECTION && ACTIVE_SELECTION.icao24 === f.icao24;
              const alt = f.altitude_ft ? (Math.round(f.altitude_ft / 100) * 100).toLocaleString() + ' ft' : '';
              const dly = (f.delayed_min && f.delayed_min > 2) ? `+${{f.delayed_min}}m` : '';
              const dlyColor = (f.delayed_min || 0) > 20 ? '#f87171' : '#fbbf24';
              return `
                <div class="search-dd-row" data-type="flight" data-icao24="${{escapeHtml(f.icao24 || '')}}">
                  <div class="search-dd-dot${{isSel ? ' sel' : ''}}"></div>
                  <div class="search-dd-flight">${{escapeHtml(f.flight || 'N/A')}}</div>
                  <div class="search-dd-route">${{escapeHtml(f.dep || '?')}} → ${{escapeHtml(f.arr || '?')}} · ${{escapeHtml(f.airline_code || '')}}</div>
                  <div class="search-dd-alt">${{dly ? `<span style="color:${{dlyColor}};font-weight:700;">${{escapeHtml(dly)}}</span>` : escapeHtml(alt)}}</div>
                </div>
              `;
            }}).join('');
            searchDropdown.innerHTML = `<div class="search-dd-section"><div class="search-dd-header">Route ${{depQ}} → ${{arrQ}} (${{routeFlights.length}})</div>${{flRows}}</div>`;
            searchDropdown.classList.add('open');
            _ddFocusIdx = -1;
            searchDropdown.querySelectorAll('.search-dd-row').forEach((row) => {{
              row.addEventListener('mousedown', (e) => {{
                e.preventDefault();
                const icao24 = row.getAttribute('data-icao24');
                const flight = FLIGHTS.find((f) => f.icao24 === icao24);
                if (flight) _selectResult(flight);
              }});
            }});
            return;
          }}
          searchDropdown.innerHTML = `<div class="search-dd-empty">No flights on route ${{depQ}} → ${{arrQ}}</div>`;
          searchDropdown.classList.add('open');
          return;
        }}

        // ── Flight matches ──────────────────────────────────────────
        const flightMatches = FLIGHTS.filter((f) => {{
          const fl = (f.flight || '').toUpperCase().replace(/[ \t]/g, '');
          const dep = (f.dep || '').toUpperCase();
          const arr = (f.arr || '').toUpperCase();
          const al = (f.airline || '').toUpperCase();
          const ac = (f.airline_code || '').toUpperCase();
          return fl.includes(q) || dep.includes(q) || arr.includes(q) || al.includes(q) || ac.includes(q);
        }}).slice(0, 6);

        // ── Airport matches ─────────────────────────────────────────
        const aptMatches = AIRPORTS.filter((a) => {{
          const iata = (a.iata || '').toUpperCase();
          const city = (a.city || '').toUpperCase().replace(/[ \t]/g, '');
          const name = (a.name || '').toUpperCase().replace(/[ \t]/g, '');
          return iata.includes(q) || city.includes(q) || name.includes(q);
        }}).slice(0, 4);

        if (!flightMatches.length && !aptMatches.length) {{
          searchDropdown.innerHTML = `<div class="search-dd-empty">No flights or airports matching "${{escapeHtml(query.trim())}}"</div>`;
          searchDropdown.classList.add('open');
          return;
        }}

        let html = '';

        if (flightMatches.length) {{
          const flRows = flightMatches.map((f) => {{
            const isSel = ACTIVE_SELECTION && ACTIVE_SELECTION.icao24 === f.icao24;
            const alt = f.altitude_ft ? (Math.round(f.altitude_ft / 100) * 100).toLocaleString() + ' ft' : '';
            const dly = (f.delayed_min && f.delayed_min > 2) ? `+${{f.delayed_min}}m` : '';
            const dlyColor = (f.delayed_min || 0) > 20 ? '#f87171' : '#fbbf24';
            return `
              <div class="search-dd-row" data-type="flight" data-icao24="${{escapeHtml(f.icao24 || '')}}">
                <div class="search-dd-dot${{isSel ? ' sel' : ''}}"></div>
                <div class="search-dd-flight">${{escapeHtml(f.flight || 'N/A')}}</div>
                <div class="search-dd-route">${{escapeHtml(f.dep || '?')}} → ${{escapeHtml(f.arr || '?')}} · ${{escapeHtml(f.airline_code || '')}}</div>
                <div class="search-dd-alt">${{dly ? `<span style="color:${{dlyColor}};font-weight:700;">${{escapeHtml(dly)}}</span>` : escapeHtml(alt)}}</div>
              </div>
            `;
          }}).join('');
          html += `<div class="search-dd-section"><div class="search-dd-header">Flights (${{flightMatches.length}})</div>${{flRows}}</div>`;
        }}

        if (aptMatches.length) {{
          if (flightMatches.length) html += '<div class="search-dd-divider"></div>';
          const aptRows = aptMatches.map((a) => {{
            const isHub = Boolean(a.is_primary_hub);
            return `
              <div class="search-dd-row" data-type="airport" data-iata="${{escapeHtml(a.iata || '')}}">
                <div class="search-dd-dot apt"></div>
                <div class="search-dd-apt-code">${{escapeHtml(a.iata || '')}}</div>
                <div class="search-dd-route">${{escapeHtml(a.city || '')}} · ${{escapeHtml(a.name || '')}}</div>
                ${{isHub ? '<div class="search-dd-hub">Hub</div>' : ''}}
              </div>
            `;
          }}).join('');
          html += `<div class="search-dd-section"><div class="search-dd-header">Airports (${{aptMatches.length}})</div>${{aptRows}}</div>`;
        }}

        searchDropdown.innerHTML = html;
        searchDropdown.classList.add('open');
        _ddFocusIdx = -1;

        searchDropdown.querySelectorAll('.search-dd-row').forEach((row) => {{
          row.addEventListener('mousedown', (e) => {{
            e.preventDefault();
            if (row.getAttribute('data-type') === 'airport') {{
              const iata = row.getAttribute('data-iata');
              const airport = AIRPORTS.find((a) => a.iata === iata);
              if (airport) _selectAirport(airport);
            }} else {{
              const icao24 = row.getAttribute('data-icao24');
              const flight = FLIGHTS.find((f) => f.icao24 === icao24);
              if (flight) _selectResult(flight);
            }}
          }});
        }});
      }}

      if (searchInput) {{
        searchInput.addEventListener('input', () => {{
          _updateClear();
          _buildDropdown(searchInput.value);
        }});
        searchInput.addEventListener('focus', () => {{
          if (searchInput.value.trim()) _buildDropdown(searchInput.value);
        }});
        searchInput.addEventListener('keydown', (event) => {{
          const rows = _ddRows();
          if (event.key === 'ArrowDown') {{
            event.preventDefault();
            _ddFocusIdx = Math.min(_ddFocusIdx + 1, rows.length - 1);
            rows.forEach((r, i) => r.classList.toggle('focused', i === _ddFocusIdx));
          }} else if (event.key === 'ArrowUp') {{
            event.preventDefault();
            _ddFocusIdx = Math.max(_ddFocusIdx - 1, -1);
            rows.forEach((r, i) => r.classList.toggle('focused', i === _ddFocusIdx));
          }} else if (event.key === 'Escape') {{
            _closeDropdown();
          }} else if (event.key === 'Enter') {{
            event.preventDefault();
            if (_ddFocusIdx >= 0 && rows[_ddFocusIdx]) {{
              const icao24 = rows[_ddFocusIdx].getAttribute('data-icao24');
              const flight = FLIGHTS.find((f) => f.icao24 === icao24);
              if (flight) {{ _selectResult(flight); return; }}
            }}
            const q = searchInput.value.trim();
            if (q) withTopUrl((url) => url.searchParams.set('flight', q));
          }}
        }});
      }}

      if (searchClear) {{
        searchClear.addEventListener('click', () => {{
          if (searchInput) searchInput.value = '';
          _updateClear();
          _closeDropdown();
          searchInput && searchInput.focus();
        }});
      }}

      document.addEventListener('click', (e) => {{
        const wrap = document.getElementById('search-wrap');
        if (wrap && !wrap.contains(e.target)) _closeDropdown();
      }});

      document.querySelectorAll('[data-ac-filter]').forEach(btn => {{
        btn.addEventListener('click', () => {{
          _acFilter = btn.getAttribute('data-ac-filter');
          document.querySelectorAll('[data-ac-filter]').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          if (_mapRef) refreshFlightsLayer(_mapRef, ACTIVE_SELECTION);
          renderFlightBoard();
        }});
      }});

      document.querySelectorAll('[data-action="fit-map"]').forEach((btn) => {{
        btn.addEventListener('click', () => {{
          if (!_mapRef) return;
          const bounds = new mapboxgl.LngLatBounds();
          FLIGHTS.forEach((f) => {{ if (f.lng && f.lat) bounds.extend([f.lng, f.lat]); }});
          if (!bounds.isEmpty()) _mapRef.fitBounds(bounds, {{ padding: 80, maxZoom: 6, duration: 800 }});
        }});
      }});

      document.querySelectorAll('[data-action="locate"]').forEach((button) => {{
        button.addEventListener('click', () => {{
          if (!navigator.geolocation) return;
          navigator.geolocation.getCurrentPosition((position) => {{
            const lng = position.coords.longitude;
            const lat = position.coords.latitude;
            if (USER_LOCATION_MARKER) {{
              USER_LOCATION_MARKER.remove();
            }}
            const markerEl = document.createElement('div');
            markerEl.style.width = '14px';
            markerEl.style.height = '14px';
            markerEl.style.borderRadius = '999px';
            markerEl.style.background = '#d4ff69';
            markerEl.style.boxShadow = '0 0 0 6px rgba(212,255,105,0.14)';
            markerEl.style.border = '2px solid rgba(6,10,16,0.92)';
            USER_LOCATION_MARKER = new mapboxgl.Marker({{ element: markerEl, anchor: 'center' }})
              .setLngLat([lng, lat])
              .addTo(map);
            map.flyTo({{ center: [lng, lat], zoom: Math.max(map.getZoom(), 8), duration: 1200 }});
          }});
        }});
      }});

      document.addEventListener('click', (event) => {{
        const closeTarget = event.target.closest('[data-close-panel]');
        if (!closeTarget) return;
        hideRightPanel();
        ACTIVE_SELECTION = null;
        refreshFlightsLayer(map, null);
        const empty = {{ type: 'FeatureCollection', features: [] }};
        const rSrc = map.getSource('selected-route');
        if (rSrc) rSrc.setData(empty);
        const pSrc = map.getSource('route-progress-src');
        if (pSrc) pSrc.setData(empty);
        withTopUrl((url) => {{
          url.searchParams.delete('flight');
        }}, false);
      }});
    }}

    function buildPopupHtml(flight) {{
      const statusText = (flight.status || 'Unknown');
      const isEnRoute = statusText.toLowerCase().includes('en');
      const statusClass = isEnRoute ? 'enroute' : 'ground';
      const depCity = (AIRPORT_LOOKUP[flight.dep] && AIRPORT_LOOKUP[flight.dep].city) || '';
      const arrCity = (AIRPORT_LOOKUP[flight.arr] && AIRPORT_LOOKUP[flight.arr].city) || '';
      return `
        <div class="popup-card">
          <div class="popup-accent-bar${{flight.selected ? ' selected' : ''}}"></div>
          <div class="popup-head">
            <div>
              <div class="popup-flight-num">${{escapeHtml(flight.flight)}}</div>
              <div class="popup-airline-row">${{escapeHtml(flight.airline)}} &nbsp;·&nbsp; ${{escapeHtml(flight.aircraft)}}</div>
            </div>
            <div class="popup-status-pill ${{statusClass}}">${{escapeHtml(statusText)}}</div>
          </div>
          <div class="popup-divider"></div>
          <div class="popup-route-row">
            <div>
              <div class="popup-iata">${{escapeHtml(flight.dep)}}</div>
              <div class="popup-city">${{escapeHtml(depCity)}}</div>
            </div>
            <div class="popup-route-mid">
              <div class="popup-route-line"></div>
              <div class="popup-plane-icon">✈</div>
              <div class="popup-route-line"></div>
            </div>
            <div style="text-align:right;">
              <div class="popup-iata">${{escapeHtml(flight.arr)}}</div>
              <div class="popup-city" style="text-align:right;">${{escapeHtml(arrCity)}}</div>
            </div>
          </div>
          <div class="popup-divider"></div>
          <div class="popup-metrics-row">
            <div class="popup-metric">
              <div class="popup-metric-value">${{formatNumber(flight.altitude_ft)}}</div>
              <div class="popup-metric-label">Altitude ft</div>
            </div>
            <div class="popup-metric">
              <div class="popup-metric-value">${{formatNumber(flight.speed_kts)}}</div>
              <div class="popup-metric-label">Speed kts</div>
            </div>
            <div class="popup-metric">
              <div class="popup-metric-value">${{formatNumber(flight.heading)}}°</div>
              <div class="popup-metric-label">Heading</div>
            </div>
          </div>
          <div class="popup-footer">
            <span>Reg: <span class="popup-footer-val">${{escapeHtml(flight.registration)}}${{(() => {{ const c = regToCountry(flight.registration); return c ? ' ' + c.flag : ''; }})()}}</span></span>
            <span>ICAO24: <span class="popup-footer-val">${{escapeHtml(flight.icao24)}}</span></span>
          </div>
        </div>
      `;
    }}

    function addBasemapTone(map) {{
      map.addSource('night-scrim', {{
        type: 'geojson',
        data: {{
          type: 'FeatureCollection',
          features: [{{
            type: 'Feature',
            properties: {{}},
            geometry: {{
              type: 'Polygon',
              coordinates: [[
                [-180, -85],
                [180, -85],
                [180, 85],
                [-180, 85],
                [-180, -85]
              ]]
            }}
          }}]
        }},
      }});

      map.addLayer({{
        id: 'night-scrim-base',
        type: 'fill',
        source: 'night-scrim',
        paint: {{
          'fill-color': 'rgba(2, 5, 10, 0.46)',
        }},
      }});

      map.addLayer({{
        id: 'night-scrim-fill',
        type: 'fill',
        source: 'night-scrim',
        paint: {{
          'fill-color': 'rgba(7, 16, 28, 0.28)',
        }},
      }});
    }}

    function addAirportLayers(map) {{
      const hubFeatures = [];
      const airportFeatures = [];

      AIRPORTS.forEach((airport) => {{
        const feature = {{
          type: 'Feature',
          properties: {{
            iata: airport.iata,
            city: airport.city,
            name: airport.name,
          }},
          geometry: {{
            type: 'Point',
            coordinates: [airport.lng, airport.lat],
          }},
        }};

        if (airport.is_primary_hub) {{
          hubFeatures.push(feature);
        }} else {{
          airportFeatures.push(feature);
        }}
      }});

      map.addSource('airports', {{
        type: 'geojson',
        data: {{
          type: 'FeatureCollection',
          features: airportFeatures,
        }},
      }});

      map.addSource('airport-hubs', {{
        type: 'geojson',
        data: {{
          type: 'FeatureCollection',
          features: hubFeatures,
        }},
      }});

      map.addLayer({{
        id: 'airport-hub-glow',
        type: 'circle',
        source: 'airport-hubs',
        paint: {{
          'circle-radius': 14,
          'circle-color': 'rgba(163,214,109,0.16)',
          'circle-opacity': 0.92,
        }},
      }});

      map.addLayer({{
        id: 'airport-points',
        type: 'circle',
        source: 'airports',
        paint: {{
          'circle-radius': 4.5,
          'circle-color': '#9fd86a',
          'circle-stroke-width': 1.5,
          'circle-stroke-color': '#04111a',
          'circle-opacity': 0.84,
        }},
      }});

      map.addLayer({{
        id: 'airport-hub-points',
        type: 'circle',
        source: 'airport-hubs',
        paint: {{
          'circle-radius': 7.5,
          'circle-color': '#cbe977',
          'circle-stroke-width': 2.2,
          'circle-stroke-color': '#04111a',
          'circle-opacity': 0.98,
        }},
      }});

      map.addLayer({{
        id: 'airport-labels-hubs',
        type: 'symbol',
        source: 'airport-hubs',
        minzoom: 4.5,
        layout: {{
          'text-field': ['get', 'iata'],
          'text-font': ['Open Sans Bold', 'Arial Unicode MS Bold'],
          'text-size': 15,
          'text-offset': [0, 1.15],
          'text-anchor': 'top',
          'text-allow-overlap': true,
          'text-ignore-placement': true,
        }},
        paint: {{
          'text-color': '#f7fbff',
          'text-halo-color': 'rgba(8,16,24,0.98)',
          'text-halo-width': 1.8,
        }},
      }});

      map.addLayer({{
        id: 'airport-labels',
        type: 'symbol',
        source: 'airports',
        minzoom: 5.8,
        layout: {{
          'text-field': ['get', 'iata'],
          'text-font': ['Open Sans Bold', 'Arial Unicode MS Bold'],
          'text-size': 12,
          'text-offset': [0, 1.1],
          'text-anchor': 'top',
          'text-allow-overlap': false,
        }},
        paint: {{
          'text-color': 'rgba(235,242,249,0.88)',
          'text-halo-color': 'rgba(8,16,24,0.95)',
          'text-halo-width': 1.5,
        }},
      }});
    }}

    async function initFlights(map) {{
      // Seed interpolation state so aircraft start at their initial API positions
      FLIGHTS.forEach(function(f) {{
        f._iLat     = f.lat;
        f._iLng     = f.lng;
        f._iHeading = f.heading || 0;
        _interpFrom[f.icao24] = {{ lat: f.lat, lng: f.lng, heading: f.heading || 0 }};
        _interpTo[f.icao24]   = {{ lat: f.lat, lng: f.lng, heading: f.heading || 0 }};
      }});
      const bounds = new mapboxgl.LngLatBounds();
      FLIGHTS.forEach((f) => bounds.extend([f.lng, f.lat]));

      let selectedFlight = SELECTED_QUERY
        ? FLIGHTS.find((f) => f.flight && f.flight.toUpperCase().replace(/[ \t]/g, '') === SELECTED_QUERY.toUpperCase().replace(/[ \t]/g, ''))
        : null;

      // Step 1: Always add circles + labels first — guaranteed visible.
      addFlightsLayer(map);

      // Step 2: Async — try to overlay aircraft silhouette icons.
      registerAircraftIcons(map).then((iconsLoaded) => {{
        if (iconsLoaded) {{
          addFlightIconsLayer(map);
          // Hide circles when silhouettes are on — they'd be double-rendered
          if (map.getLayer('flights-circles')) {{
            map.setLayoutProperty('flights-circles', 'visibility', 'none');
          }}
        }}
      }}).catch(() => {{}});

      const airportCounts = computeAirportCounts();
      const routeCounts = computeRouteCounts();
      const mostDelayed = getMostDelayedFlights(airportCounts, routeCounts);

      if (selectedFlight) {{
        setActiveFlight(map, selectedFlight);
        return;
      }}

      if (!bounds.isEmpty()) {{
        hideRightPanel();
        renderTopStats();
        renderLeftRail(airportCounts, routeCounts, mostDelayed);
        // Keep flight board up-to-date if it's the active module
        if (ACTIVE_MODULE === 'flight_board') renderFlightBoard();
        const padding = ACTIVE_MODULE === 'flight_board'
          ? {{ top: 88, right: 32, bottom: 32, left: 32 }}
          : {{ top: 88, right: 420, bottom: 56, left: 340 }};
        map.fitBounds(bounds, {{
          padding,
          maxZoom: ACTIVE_MODULE === 'live_ops' ? 5.5 : 6.2,
          duration: 600,
        }});
      }}
    }}

    try {{
      if (typeof mapboxgl === 'undefined') {{
        throw new Error('Mapbox GL JS failed to load from CDN. Check network connectivity.');
      }}
      if (!mapboxgl.supported()) {{
        throw new Error('WebGL is not supported in this browser or context. Try opening the app in a modern browser.');
      }}
      if (!MAPBOX_TOKEN || !String(MAPBOX_TOKEN).startsWith('pk.')) {{
        throw new Error('Mapbox public token missing. Set MAPBOX_TOKEN=pk.... in your environment.');
      }}

      mapboxgl.accessToken = MAPBOX_TOKEN;
      const PRIMARY_STYLE = 'mapbox://styles/mapbox/dark-v11';
      const _isMobile = window.innerWidth <= 768;
      const map = new mapboxgl.Map({{
        container: 'map',
        style: PRIMARY_STYLE,
        center: [{lng_center}, {lat_center}],
        zoom: {zoom},
        pitch: 0,
        bearing: 0,
        minZoom: 1.5,
        maxPitch: 60,
        projection: 'globe',
        renderWorldCopies: true,
      }});
      // On mobile, globe projection strains WebGL — downgrade to mercator so the map actually loads
      if (_isMobile) {{
        try {{ map.setProjection('mercator'); }} catch(e) {{}}
      }}
      _mapRef = map;

      map.addControl(new mapboxgl.NavigationControl({{ visualizePitch: true, showCompass: true }}), 'bottom-left');
      map.addControl(new mapboxgl.ScaleControl({{ unit: 'metric' }}), 'bottom-right');
      map.addControl(new mapboxgl.FullscreenControl(), 'bottom-left');

      // Safety net: if nothing fires within 3s force-hide the splash.
      window.setTimeout(hideSplash, 3000);

      let ready = false;
      const readyTimer = window.setTimeout(() => {{
        if (!ready) {{
          showError('Map is taking too long. Check your Mapbox token or network.');
        }}
      }}, 12000);

      // Helper: addLayer before 'flights-circles' only if that layer exists.
      // Avoids "Layer does not exist" when addLayer is called before initFlights runs.
      function _addLayerBeforeFlights(map, layerDef) {{
        const before = map.getLayer('flights-circles') ? 'flights-circles' : undefined;
        map.addLayer(layerDef, before);
      }}

      function _runMapInit() {{
        try {{
          if (typeof map.setFog === 'function') {{
            map.setFog({{
              color: 'rgb(10, 16, 24)',
              'high-color': 'rgb(28, 48, 72)',
              'horizon-blend': 0.08,
              'space-color': 'rgb(3, 8, 14)',
              'star-intensity': 0.0,
            }});
          }}
          bindChromeInteractions(map);
          bindFlightBoard(map);
          addAirportLayers(map);
          addTrailLayers(map);
        }} catch (initErr) {{
          showError('Map setup error: ' + (initErr && initErr.message ? initErr.message : String(initErr)));
          return;
        }}
        initFlights(map).then(() => {{
          ready = true;
          window.clearTimeout(readyTimer);
          // Apply any flight data that arrived before the map was ready
          if (_pendingFlightUpdate) {{
            window.updateFlightData(_pendingFlightUpdate);
          }}
          // Run initial anomaly scan
          detectAnomalies();
          // Wire airport marker click → schedule popup (all airport layers)
          ['airport-points','airport-hub-points','airport-labels','airport-labels-hubs'].forEach((layerId) => {{
            if (!map.getLayer(layerId)) return;
            map.on('click', layerId, (e) => {{
              const iata = e.features && e.features[0] && e.features[0].properties.iata;
              if (iata) openAirportSchedule(iata);
            }});
            map.on('mouseenter', layerId, () => {{ map.getCanvas().style.cursor = 'pointer'; }});
            map.on('mouseleave', layerId, () => {{ map.getCanvas().style.cursor = ''; }});
          }});
        }}).catch((err) => {{
          showError('Failed to load flights: ' + (err && err.message ? err.message : String(err)));
        }});
      }}

      map.on('load', () => {{
        // Hide splash immediately when map fires load — don't wait for initFlights.
        hideSplash();
        // Force canvas to fill the iframe — Streamlit iframes sometimes have sizing delays.
        map.resize();
        _runMapInit();
        // Keep canvas sized correctly if the iframe is later resized.
        if (typeof ResizeObserver !== 'undefined') {{
          new ResizeObserver(() => map.resize()).observe(document.getElementById('stage') || document.body);
        }}
      }});

      // ── Flight trail polylines ────────────────────────────────────────────────
      let _trailsVisible = true;

      function addTrailLayers(map) {{
        const features = [];
        FLIGHTS.forEach((f) => {{
          const pts = TRAILS[f.icao24];
          if (!pts || pts.length < 2) return;
          const coords = pts.map((p) => [p.lng, p.lat]);
          // append current position
          coords.push([f.lng, f.lat]);
          features.push({{
            type: 'Feature',
            geometry: {{ type: 'LineString', coordinates: coords }},
            properties: {{ icao24: f.icao24, alt: f.altitude_ft || 0 }},
          }});
        }});
        if (map.getSource('trails-src')) {{
          map.getSource('trails-src').setData({{ type: 'FeatureCollection', features }});
          return;
        }}
        map.addSource('trails-src', {{
          type: 'geojson',
          data: {{ type: 'FeatureCollection', features }},
          lineMetrics: true,
        }});
        // Glow (wide, faint)
        _addLayerBeforeFlights(map, {{
          id: 'trails-glow',
          type: 'line',
          source: 'trails-src',
          paint: {{
            'line-color': '#00e5ff',
            'line-width': 5,
            'line-opacity': 0.08,
            'line-blur': 4,
          }},
        }});
        // Main trail
        _addLayerBeforeFlights(map, {{
          id: 'trails-line',
          type: 'line',
          source: 'trails-src',
          paint: {{
            'line-color': '#00e5ff',
            'line-width': 1.4,
            'line-opacity': ['interpolate', ['linear'], ['line-progress'], 0, 0.0, 1, 0.55],
            'line-gradient': ['interpolate', ['linear'], ['line-progress'],
              0, 'rgba(0,229,255,0)',
              0.6, 'rgba(0,229,255,0.4)',
              1, 'rgba(0,229,255,0.85)'],
          }},
          layout: {{ 'line-cap': 'round', 'line-join': 'round' }},
        }});
      }}

      function setTrailsVisible(map, visible) {{
        _trailsVisible = visible;
        ['trails-line', 'trails-glow'].forEach((id) => {{
          if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', visible ? 'visible' : 'none');
        }});
        const btn = document.querySelector('[data-action="toggle-trails"]');
        if (btn) btn.classList.toggle('active', visible);
      }}

      // ── Go-around / holding detection ────────────────────────────────────────
      const _alertLog = [];
      let _alertSeq = 0;

      function pushAlert(type, icon, title, body) {{
        const id = ++_alertSeq;
        _alertLog.unshift({{ id, type, icon, title, body, ts: new Date() }});
        if (_alertLog.length > 30) _alertLog.pop();
        renderAlertFeed();
        // Audio chime via Web Audio API (no external file needed)
        try {{
          const ctx = new (window.AudioContext || window.webkitAudioContext)();
          const osc = ctx.createOscillator();
          const gain = ctx.createGain();
          osc.connect(gain);
          gain.connect(ctx.destination);
          osc.frequency.value = type === 'critical' ? 880 : 660;
          gain.gain.setValueAtTime(0.12, ctx.currentTime);
          gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.45);
          osc.start(ctx.currentTime);
          osc.stop(ctx.currentTime + 0.45);
        }} catch (_) {{}}
        // Auto-dismiss non-critical after 20s
        if (type !== 'critical') setTimeout(() => dismissAlert(id), 20000);
      }}

      function dismissAlert(id) {{
        const idx = _alertLog.findIndex((a) => a.id === id);
        if (idx !== -1) _alertLog.splice(idx, 1);
        renderAlertFeed();
      }}

      function renderAlertFeed() {{
        const feed = document.getElementById('alert-feed');
        if (!feed) return;
        feed.innerHTML = _alertLog.slice(0, 8).map((a) => `
          <div class="alert-card ${{a.type}}" data-alert-id="${{a.id}}">
            <span class="alert-card-icon">${{a.icon}}</span>
            <div class="alert-card-body">
              <div class="alert-card-title">${{a.title}}</div>
              <div>${{a.body}}</div>
              <div class="alert-card-time">${{a.ts.toLocaleTimeString('en-IN', {{hour:'2-digit',minute:'2-digit',second:'2-digit'}})}}</div>
            </div>
            <button class="alert-card-dismiss" data-alert-id="${{a.id}}" type="button" title="Dismiss">&#x2715;</button>
          </div>`).join('');
        feed.querySelectorAll('.alert-card-dismiss').forEach((btn) => {{
          btn.addEventListener('click', (e) => {{
            e.stopPropagation();
            dismissAlert(Number(btn.getAttribute('data-alert-id')));
          }});
        }});
        // Also refresh side panel if open
        if (typeof _alertsPanelOpen !== 'undefined' && _alertsPanelOpen) {{
          renderAlertsSidePanel();
        }}
      }}

      // Detection state
      const _prevPos = {{}};   // icao24 → {{lat,lng,ts,headings[]}}
      const _alertedSet = new Set();

      function detectAnomalies() {{
        const nowTs = Date.now() / 1000;
        FLIGHTS.forEach((f) => {{
          if (!f.icao24 || f.altitude_ft > 8000) return;
          const key = f.icao24;
          const prev = _prevPos[key];
          if (!prev) {{
            _prevPos[key] = {{ lat: f.lat, lng: f.lng, ts: nowTs, headings: [f.heading] }};
            return;
          }}
          // Update heading history (keep last 6)
          prev.headings.push(f.heading);
          if (prev.headings.length > 6) prev.headings.shift();
          prev.lat = f.lat; prev.lng = f.lng; prev.ts = nowTs;

          // Holding pattern: heading variance > 150° across last 6 readings
          if (prev.headings.length >= 5) {{
            const hMax = Math.max(...prev.headings);
            const hMin = Math.min(...prev.headings);
            const variance = hMax - hMin;
            if (variance > 150 && !_alertedSet.has(key + '_hold')) {{
              _alertedSet.add(key + '_hold');
              const airp = f.arr || '?';
              pushAlert('warning', '🔄', `Holding pattern · ${{f.flight || key}}`,
                `Near ${{airp}} at ${{Math.round(f.altitude_ft).toLocaleString()}} ft`);
              setTimeout(() => _alertedSet.delete(key + '_hold'), 300000);
            }}
          }}

          // Low & slow near destination: possible go-around
          if (f.altitude_ft < 4000 && f.speed_kts > 120 && f.speed_kts < 200) {{
            const arrAp = AIRPORT_LOOKUP[f.arr];
            if (arrAp) {{
              const dLat = f.lat - arrAp.lat, dLng = f.lng - arrAp.lng;
              const distDeg = Math.sqrt(dLat * dLat + dLng * dLng);
              if (distDeg < 0.3 && !_alertedSet.has(key + '_ga')) {{
                _alertedSet.add(key + '_ga');
                pushAlert('critical', '⚠️', `Possible go-around · ${{f.flight || key}}`,
                  `Near ${{f.arr}} at ${{Math.round(f.altitude_ft).toLocaleString()}} ft · ${{Math.round(f.speed_kts)}} kts`);
                setTimeout(() => _alertedSet.delete(key + '_ga'), 300000);
              }}
            }}
          }}
        }});

        // ── Minimum separation scan (50 nm / 1000 ft) ──────────────────────
        // ICAO horizontal separation minima: 5 nm radar, 50 nm = high-density simplified check.
        // Only scan en-route aircraft (altitude > 5000 ft) to avoid false positives on ground.
        const enRoute = FLIGHTS.filter((f) => f.altitude_ft > 5000 && f.lat && f.lng);
        for (let i = 0; i < enRoute.length; i++) {{
          for (let j = i + 1; j < enRoute.length; j++) {{
            const a = enRoute[i], b = enRoute[j];
            const altDiff = Math.abs((a.altitude_ft || 0) - (b.altitude_ft || 0));
            if (altDiff > 1000) continue;  // vertical separation OK — skip
            const dLat = a.lat - b.lat, dLng = a.lng - b.lng;
            // Quick rectangular pre-filter (1° ≈ 111 km ≈ 60 nm)
            if (Math.abs(dLat) > 0.85 || Math.abs(dLng) > 0.85) continue;
            const distKm = haversineKm(a.lat, a.lng, b.lat, b.lng);
            const distNm = distKm * 0.54;
            if (distNm <= 8) {{
              const pairKey = [a.icao24, b.icao24].sort().join('_') + '_sep';
              if (!_alertedSet.has(pairKey)) {{
                _alertedSet.add(pairKey);
                pushAlert('critical', '⚡',
                  `Separation alert · ${{a.flight || a.icao24}} / ${{b.flight || b.icao24}}`,
                  `${{Math.round(distNm * 10) / 10}} nm apart · alt diff ${{Math.round(altDiff)}} ft`);
                setTimeout(() => _alertedSet.delete(pairKey), 120000);
              }}
            }}
          }}
        }}
      }}

      // ── Airport schedule popup ────────────────────────────────────────────────
      let _schedTabActive = 'overview';

      function openAirportSchedule(iata) {{
        const flights = FLIGHTS.filter((f) => f.arr === iata || f.dep === iata);
        const arrivals   = flights.filter((f) => f.arr === iata)
          .sort((a, b) => (a.sta_ts || 0) - (b.sta_ts || 0)).slice(0, 15);
        const departures = flights.filter((f) => f.dep === iata)
          .sort((a, b) => (a.std_ts || 0) - (b.std_ts || 0)).slice(0, 15);
        const ap = AIRPORT_LOOKUP[iata] || {{}};
        const apMetrics = (AIRPORT_METRICS || []).find((m) => m.airport_iata === iata) || {{}};
        const schedPressure = (SCHEDULE_PRESSURE || {{}})[iata] || {{}};

        function fmtTs(ts) {{
          if (!ts) return '--:--';
          const d = new Date(ts * 1000);
          const ist = new Date(d.getTime() + 330 * 60000);
          return String(ist.getUTCHours()).padStart(2,'0') + ':' + String(ist.getUTCMinutes()).padStart(2,'0');
        }}
        function rowHtml(f, isArr) {{
          const time = isArr ? fmtTs(f.sta_ts) : fmtTs(f.std_ts);
          const peer = isArr ? (f.dep || '?') : (f.arr || '?');
          const peerCity = (AIRPORT_LOOKUP[peer] && AIRPORT_LOOKUP[peer].city) || peer;
          const delay = (f.delayed_min && f.delayed_min > 2) ? f.delayed_min : 0;
          const statusCls = delay > 0 ? 'status-delayed' : 'status-on-time';
          const statusTxt = delay > 0 ? `+${{delay}}m` : 'On Time';
          return `<div class="sched-row">
            <span class="fnum">${{f.flight || '—'}}</span>
            <span class="dest">${{peerCity}}</span>
            <span class="time">${{time}}</span>
            <span class="${{statusCls}}">${{statusTxt}}</span>
          </div>`;
        }}
        function overviewHtml() {{
          // Congestion
          const congScore = apMetrics.congestion_score || 0;
          const congLevel = apMetrics.congestion_level || 'Normal';
          const congClr = congLevel === 'Critical' ? '#ff4b4b'
            : congLevel === 'High' ? '#ffb347'
            : congLevel === 'Medium' ? '#ffd700'
            : '#4ecdc4';

          // Top routes from/to this airport (from live FLIGHTS)
          const routeCounts = {{}};
          FLIGHTS.forEach((f) => {{
            const peer = (f.dep === iata) ? f.arr : (f.arr === iata ? f.dep : null);
            if (peer) routeCounts[peer] = (routeCounts[peer] || 0) + 1;
          }});
          const topRoutes = Object.entries(routeCounts)
            .sort((a, b) => b[1] - a[1]).slice(0, 4);
          const routesHtml = topRoutes.length
            ? topRoutes.map(([code, cnt]) => {{
                const city = (AIRPORT_LOOKUP[code] && AIRPORT_LOOKUP[code].city) || code;
                return `<div class="ap-route-row">
                  <span class="ap-route-iata">${{code}}</span>
                  <span class="ap-route-city">${{city}}</span>
                  <span class="ap-route-count">${{cnt}} flight${{cnt !== 1 ? 's' : ''}}</span>
                </div>`;
              }}).join('')
            : '<div style="font-size:11px;color:rgba(244,247,251,0.35);padding:4px 0">No live route data</div>';

          // Schedule pressure badge
          const pressLvl = schedPressure.pressure_level || null;
          let pressHtml = '';
          if (pressLvl && pressLvl !== 'Unknown') {{
            const pressClass = pressLvl === 'High' ? 'ap-pressure-high'
              : pressLvl === 'Medium' ? 'ap-pressure-medium'
              : pressLvl === 'Low' ? 'ap-pressure-low'
              : 'ap-pressure-normal';
            pressHtml = `<div class="ap-pressure-row">
              <span class="ap-pressure-badge ${{pressClass}}">${{pressLvl}}</span>
              <span class="ap-pressure-lbl">Schedule pressure</span>
            </div>`;
          }}

          return `<div class="ap-overview">
            <div class="ap-stat-grid">
              <div class="ap-stat">
                <div class="ap-stat-val">${{apMetrics.nearby_count ?? '—'}}</div>
                <div class="ap-stat-lbl">Nearby</div>
              </div>
              <div class="ap-stat">
                <div class="ap-stat-val">${{apMetrics.inbound_count ?? '—'}}</div>
                <div class="ap-stat-lbl">Inbound</div>
              </div>
              <div class="ap-stat">
                <div class="ap-stat-val">${{apMetrics.approach_count ?? '—'}}</div>
                <div class="ap-stat-lbl">Approach</div>
              </div>
              <div class="ap-stat">
                <div class="ap-stat-val">${{apMetrics.low_altitude_count ?? '—'}}</div>
                <div class="ap-stat-lbl">Low Alt</div>
              </div>
            </div>
            <div>
              <div class="ap-section-title">Congestion</div>
              <div class="ap-cong-row">
                <span class="ap-cong-label">Score: ${{Math.round(congScore)}}</span>
                <span class="ap-cong-badge" style="color:${{congClr}}">${{congLevel}}</span>
              </div>
              <div class="ap-cong-bar-wrap">
                <div class="ap-cong-bar-fill" style="width:${{Math.min(Math.max(congScore,0),100)}}%;background:${{congClr}}"></div>
              </div>
            </div>
            <div>
              <div class="ap-section-title">Top Routes</div>
              <div class="ap-routes-list">${{routesHtml}}</div>
            </div>
            ${{pressHtml ? `<div>${{pressHtml}}</div>` : ''}}
          </div>`;
        }}
        function tabContent(tab) {{
          if (tab === 'overview') return overviewHtml();
          const list = tab === 'arr' ? arrivals : departures;
          if (!list.length) return '<div style="padding:16px;text-align:center;color:rgba(244,247,251,0.4);font-size:12px">No flights in dataset</div>';
          return list.map((f) => rowHtml(f, tab === 'arr')).join('');
        }}

        const popup = document.getElementById('schedule-popup');
        const backdrop = document.getElementById('schedule-popup-backdrop');
        if (!popup) return;
        _schedTabActive = 'overview';
        popup.innerHTML = `
          <div class="sched-header">
            <div class="sched-title">&#9992;&#xFE0F;&nbsp; ${{iata}} · ${{ap.city || ap.name || iata}}</div>
            <button class="sched-close" id="sched-close-btn" type="button">&#x2715;</button>
          </div>
          <div class="sched-tab-strip">
            <button class="sched-tab${{_schedTabActive==='overview'?' active':''}}" data-sched-tab="overview">&#x2139; Overview</button>
            <button class="sched-tab${{_schedTabActive==='arr'?' active':''}}" data-sched-tab="arr">&#x2193; Arr (${{arrivals.length}})</button>
            <button class="sched-tab${{_schedTabActive==='dep'?' active':''}}" data-sched-tab="dep">&#x2191; Dep (${{departures.length}})</button>
          </div>
          <div id="sched-content">${{tabContent(_schedTabActive)}}</div>`;
        popup.classList.add('visible');
        backdrop.classList.add('visible');
        popup.querySelectorAll('[data-sched-tab]').forEach((btn) => {{
          btn.addEventListener('click', () => {{
            _schedTabActive = btn.getAttribute('data-sched-tab');
            popup.querySelectorAll('[data-sched-tab]').forEach((b) => b.classList.toggle('active', b === btn));
            const content = popup.querySelector('#sched-content');
            if (content) content.innerHTML = tabContent(_schedTabActive);
          }});
        }});
        document.getElementById('sched-close-btn').addEventListener('click', closeAirportSchedule);
        backdrop.addEventListener('click', closeAirportSchedule, {{ once: true }});
      }}

      function closeAirportSchedule() {{
        const popup = document.getElementById('schedule-popup');
        const backdrop = document.getElementById('schedule-popup-backdrop');
        popup && popup.classList.remove('visible');
        backdrop && backdrop.classList.remove('visible');
      }}

      // ── Keyboard shortcuts modal ──────────────────────────────────────────────
      function openShortcutsModal() {{
        document.getElementById('shortcuts-modal')?.classList.add('visible');
        document.getElementById('shortcuts-backdrop')?.classList.add('visible');
      }}
      function closeShortcutsModal() {{
        document.getElementById('shortcuts-modal')?.classList.remove('visible');
        document.getElementById('shortcuts-backdrop')?.classList.remove('visible');
      }}

      document.getElementById('shortcuts-close-btn')?.addEventListener('click', closeShortcutsModal);
      document.getElementById('shortcuts-backdrop')?.addEventListener('click', closeShortcutsModal);

      // ── Global keyboard handler ───────────────────────────────────────────────
      document.addEventListener('keydown', (e) => {{
        if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;
        if (e.key === '?' || e.key === 'h' || e.key === 'H') {{ openShortcutsModal(); return; }}
        if (e.key === 'Escape') {{
          closeShortcutsModal();
          closeAirportSchedule();
          return;
        }}
        if (e.key === '/' || e.key === 'f' || e.key === 'F') {{
          e.preventDefault();
          document.getElementById('flight-search-input')?.focus();
          return;
        }}
        if (e.key === 't' || e.key === 'T') {{ if (_mapRef) setTrailsVisible(_mapRef, !_trailsVisible); return; }}
        if (e.key === 'w' || e.key === 'W') {{ if (_mapRef) setWeatherVisible(_mapRef, !_weatherVisible); return; }}
        if (e.key === 'm' || e.key === 'M') {{ if (_mapRef) setHeatmapVisible(_mapRef, !_heatmapVisible); return; }}
        const moduleKeys = ['live_ops','flight_board','routes','airport_traffic','alerts'];
        const idx = parseInt(e.key, 10) - 1;
        if (idx >= 0 && idx < moduleKeys.length && _mapRef) {{
          switchModule(_mapRef, moduleKeys[idx]);
        }}
        if (e.key === 'Home' && _mapRef) {{
          const bounds = new mapboxgl.LngLatBounds();
          FLIGHTS.forEach((f) => bounds.extend([f.lng, f.lat]));
          if (!bounds.isEmpty()) _mapRef.fitBounds(bounds, {{ padding: 80, maxZoom: 6, duration: 800 }});
        }}
      }});

      // ── Wire up tool-strip buttons ────────────────────────────────────────────
      document.querySelectorAll('[data-action="toggle-trails"]').forEach((btn) => {{
        btn.addEventListener('click', () => {{ if (_mapRef) setTrailsVisible(_mapRef, !_trailsVisible); }});
      }});
      document.querySelectorAll('[data-action="shortcuts"]').forEach((btn) => {{
        btn.addEventListener('click', openShortcutsModal);
      }});

      // ── Alerts side panel ─────────────────────────────────────────────────────
      function renderAlertsSidePanel() {{
        const body = document.getElementById('asp-body');
        if (!body) return;
        if (!_alertLog.length) {{
          body.innerHTML = '<div class="asp-empty">No alerts yet</div>';
          return;
        }}
        const grouped = {{ critical: [], warning: [], info: [] }};
        _alertLog.forEach(a => {{ (grouped[a.type] || grouped.info).push(a); }});
        let html = '';
        [['critical','Critical'], ['warning','Warning'], ['info','Info']].forEach(([type, label]) => {{
          const items = grouped[type];
          if (!items.length) return;
          html += `<div class="asp-group-label">${{label}} (${{items.length}})</div>`;
          items.forEach(a => {{
            html += `<div class="asp-item ${{a.type}}">
              <span class="asp-item-icon">${{a.icon}}</span>
              <div class="asp-item-body">
                <div class="asp-item-title">${{escapeHtml(a.title)}}</div>
                <div class="asp-item-sub">${{escapeHtml(a.body)}}</div>
                <div class="asp-item-time">${{a.ts.toLocaleTimeString('en-IN',{{hour:'2-digit',minute:'2-digit',second:'2-digit'}})}}</div>
              </div>
              <button class="asp-item-dismiss" data-dismiss-id="${{a.id}}" type="button">&#x2715;</button>
            </div>`;
          }});
        }});
        body.innerHTML = html;
        body.querySelectorAll('[data-dismiss-id]').forEach(btn => {{
          btn.addEventListener('click', () => {{
            dismissAlert(Number(btn.getAttribute('data-dismiss-id')));
            renderAlertsSidePanel();
          }});
        }});
      }}

      let _alertsPanelOpen = false;
      function toggleAlertsPanel() {{
        _alertsPanelOpen = !_alertsPanelOpen;
        const panel = document.getElementById('alerts-side-panel');
        const btn = document.getElementById('btn-alerts-panel');
        if (panel) panel.classList.toggle('visible', _alertsPanelOpen);
        if (btn) btn.classList.toggle('active', _alertsPanelOpen);
        if (_alertsPanelOpen) renderAlertsSidePanel();
      }}

      document.getElementById('asp-close-btn')?.addEventListener('click', () => {{
        _alertsPanelOpen = false;
        document.getElementById('alerts-side-panel')?.classList.remove('visible');
        document.getElementById('btn-alerts-panel')?.classList.remove('active');
      }});
      document.getElementById('asp-clear-btn')?.addEventListener('click', () => {{
        _alertLog.length = 0;
        renderAlertFeed();
        renderAlertsSidePanel();
      }});
      document.querySelectorAll('[data-action="toggle-alerts-panel"]').forEach(btn => {{
        btn.addEventListener('click', toggleAlertsPanel);
      }});

      // ── Speed / altitude filter panel ────────────────────────────────────────
      let _filterPanelOpen = false;
      function updateFilterLabels() {{
        const altVal = document.getElementById('fp-alt-val');
        const spdVal = document.getElementById('fp-spd-val');
        if (altVal) altVal.textContent = `${{_altMin.toLocaleString()}} – ${{_altMax.toLocaleString()}} ft`;
        if (spdVal) spdVal.textContent = `${{_spdMin}} – ${{_spdMax}} kts`;
        const btn = document.getElementById('btn-filters');
        const isActive = _altMin > 0 || _altMax < 45000 || _spdMin > 0 || _spdMax < 600;
        if (btn) btn.classList.toggle('active', isActive);
        if (_mapRef && _mapRef.isStyleLoaded()) {{
          const src = _mapRef.getSource('flights-src');
          if (src) src.setData(buildFlightsGeoJson(ACTIVE_SELECTION ? ACTIVE_SELECTION.icao24 : null));
        }}
      }}

      document.getElementById('fp-alt-min')?.addEventListener('input', (e) => {{
        _altMin = Math.min(Number(e.target.value), _altMax - 500);
        e.target.value = _altMin;
        updateFilterLabels();
      }});
      document.getElementById('fp-alt-max')?.addEventListener('input', (e) => {{
        _altMax = Math.max(Number(e.target.value), _altMin + 500);
        e.target.value = _altMax;
        updateFilterLabels();
      }});
      document.getElementById('fp-spd-min')?.addEventListener('input', (e) => {{
        _spdMin = Math.min(Number(e.target.value), _spdMax - 10);
        e.target.value = _spdMin;
        updateFilterLabels();
      }});
      document.getElementById('fp-spd-max')?.addEventListener('input', (e) => {{
        _spdMax = Math.max(Number(e.target.value), _spdMin + 10);
        e.target.value = _spdMax;
        updateFilterLabels();
      }});
      document.getElementById('fp-reset-btn')?.addEventListener('click', () => {{
        _altMin = 0; _altMax = 45000; _spdMin = 0; _spdMax = 600;
        document.getElementById('fp-alt-min').value = 0;
        document.getElementById('fp-alt-max').value = 45000;
        document.getElementById('fp-spd-min').value = 0;
        document.getElementById('fp-spd-max').value = 600;
        updateFilterLabels();
      }});
      document.querySelectorAll('[data-action="toggle-filters"]').forEach(btn => {{
        btn.addEventListener('click', () => {{
          _filterPanelOpen = !_filterPanelOpen;
          document.getElementById('filter-panel')?.classList.toggle('visible', _filterPanelOpen);
          btn.classList.toggle('active', _filterPanelOpen);
        }});
      }});

      // ── Weather (RainViewer radar tile) ──────────────────────────────────────
      let _weatherVisible = false;
      const _RV_API = 'https://api.rainviewer.com/public/weather-maps.json';

      function addWeatherLayer(map, tileUrl) {{
        if (map.getSource('rain-src')) {{
          map.getSource('rain-src').tiles = [tileUrl];
          map.style.sourceCaches['rain-src'].clearTiles();
          map.style.sourceCaches['rain-src'].update(map.transform);
          return;
        }}
        map.addSource('rain-src', {{
          type: 'raster',
          tiles: [tileUrl],
          tileSize: 256,
          attribution: 'RainViewer',
        }});
        _addLayerBeforeFlights(map, {{
          id: 'rain-layer',
          type: 'raster',
          source: 'rain-src',
          paint: {{ 'raster-opacity': 0.55 }},
        }});
      }}

      function setWeatherVisible(map, visible) {{
        _weatherVisible = visible;
        const btn = document.getElementById('btn-weather');
        if (!visible) {{
          if (map.getLayer('rain-layer')) map.setLayoutProperty('rain-layer', 'visibility', 'none');
          if (btn) btn.classList.remove('active');
          return;
        }}
        if (btn) btn.classList.add('active');
        fetch(_RV_API)
          .then((r) => r.json())
          .then((data) => {{
            const nowcast = data.radar && data.radar.nowcast;
            if (!nowcast || !nowcast.length) return;
            const latest = nowcast[nowcast.length - 1];
            const tile = latest.path + '/256/{{z}}/{{x}}/{{y}}/2/1_1.png';
            addWeatherLayer(map, tile);
            if (map.getLayer('rain-layer')) map.setLayoutProperty('rain-layer', 'visibility', 'visible');
          }})
          .catch(() => {{ if (btn) btn.classList.remove('active'); }});
      }}

      document.querySelectorAll('[data-action="toggle-weather"]').forEach((btn) => {{
        btn.addEventListener('click', () => {{ if (_mapRef) setWeatherVisible(_mapRef, !_weatherVisible); }});
      }});

      // ── Traffic density heatmap ───────────────────────────────────────────────
      let _heatmapVisible = false;

      function setHeatmapVisible(map, visible) {{
        _heatmapVisible = visible;
        const btn = document.getElementById('btn-heatmap');
        if (visible) {{
          if (btn) btn.classList.add('active');
          const features = FLIGHTS.map((f) => ({{
            type: 'Feature',
            geometry: {{ type: 'Point', coordinates: [f.lng, f.lat] }},
            properties: {{ weight: 1 }},
          }}));
          if (map.getSource('heat-src')) {{
            map.getSource('heat-src').setData({{ type: 'FeatureCollection', features }});
            map.setLayoutProperty('heat-layer', 'visibility', 'visible');
            return;
          }}
          map.addSource('heat-src', {{
            type: 'geojson',
            data: {{ type: 'FeatureCollection', features }},
          }});
          _addLayerBeforeFlights(map, {{
            id: 'heat-layer',
            type: 'heatmap',
            source: 'heat-src',
            paint: {{
              'heatmap-weight': 1,
              'heatmap-intensity': ['interpolate', ['linear'], ['zoom'], 3, 0.6, 9, 2],
              'heatmap-radius': ['interpolate', ['linear'], ['zoom'], 3, 18, 9, 40],
              'heatmap-color': [
                'interpolate', ['linear'], ['heatmap-density'],
                0,   'rgba(0,0,0,0)',
                0.2, 'rgba(0,229,255,0.3)',
                0.4, 'rgba(74,222,128,0.6)',
                0.6, 'rgba(251,191,36,0.8)',
                0.8, 'rgba(248,113,113,0.9)',
                1,   'rgba(220,38,38,1)',
              ],
              'heatmap-opacity': 0.75,
            }},
          }});
        }} else {{
          if (btn) btn.classList.remove('active');
          if (map.getLayer('heat-layer')) map.setLayoutProperty('heat-layer', 'visibility', 'none');
        }}
      }}

      document.querySelectorAll('[data-action="toggle-heatmap"]').forEach((btn) => {{
        btn.addEventListener('click', () => {{ if (_mapRef) setHeatmapVisible(_mapRef, !_heatmapVisible); }});
      }});

      // ── Airport delay rate heatmap ────────────────────────────────────────────
      let _delayMapVisible = false;
      function setDelayMapVisible(map, visible) {{
        _delayMapVisible = visible;
        const btn = document.getElementById('btn-delay-map');
        if (btn) btn.classList.toggle('active', visible);
        if (visible) {{
          // Compute per-airport avg delay from live FLIGHTS
          const apDelay = {{}};
          const apCount = {{}};
          FLIGHTS.forEach(f => {{
            if (!f.dep || f.delayed_min == null) return;
            apDelay[f.dep] = (apDelay[f.dep] || 0) + Number(f.delayed_min);
            apCount[f.dep] = (apCount[f.dep] || 0) + 1;
          }});
          const features = AIRPORTS.filter(a => a.lat && a.lng).map(a => {{
            const avg = apCount[a.iata] ? apDelay[a.iata] / apCount[a.iata] : 0;
            return {{
              type: 'Feature',
              geometry: {{ type: 'Point', coordinates: [a.lng, a.lat] }},
              properties: {{ iata: a.iata, avgDelay: avg, count: apCount[a.iata] || 0 }},
            }};
          }});
          if (map.getSource('delay-map-src')) {{
            map.getSource('delay-map-src').setData({{ type: 'FeatureCollection', features }});
            map.setLayoutProperty('delay-map-layer', 'visibility', 'visible');
            map.setLayoutProperty('delay-map-labels', 'visibility', 'visible');
            return;
          }}
          map.addSource('delay-map-src', {{ type: 'geojson', data: {{ type: 'FeatureCollection', features }} }});
          _addLayerBeforeFlights(map, {{
            id: 'delay-map-layer',
            type: 'circle',
            source: 'delay-map-src',
            paint: {{
              'circle-radius': ['interpolate', ['linear'], ['get', 'count'], 0, 4, 5, 9, 20, 16],
              'circle-color': [
                'interpolate', ['linear'], ['get', 'avgDelay'],
                0,  '#4ade80',
                10, '#fbbf24',
                25, '#f97316',
                45, '#f87171',
              ],
              'circle-opacity': 0.80,
              'circle-stroke-width': 1,
              'circle-stroke-color': 'rgba(0,0,0,0.4)',
            }},
          }});
          map.addLayer({{
            id: 'delay-map-labels',
            type: 'symbol',
            source: 'delay-map-src',
            filter: ['>', ['get', 'count'], 1],
            layout: {{
              'text-field': ['concat', ['get', 'iata'], '\\n+', ['to-string', ['round', ['get', 'avgDelay']]], 'm'],
              'text-font': ['DIN Pro Bold', 'Arial Unicode MS Bold'],
              'text-size': 9,
              'text-offset': [0, 1.4],
              'text-anchor': 'top',
            }},
            paint: {{
              'text-color': '#f4f7fb',
              'text-halo-color': 'rgba(0,0,0,0.6)',
              'text-halo-width': 1,
            }},
          }});
        }} else {{
          if (map.getLayer('delay-map-layer')) map.setLayoutProperty('delay-map-layer', 'visibility', 'none');
          if (map.getLayer('delay-map-labels')) map.setLayoutProperty('delay-map-labels', 'visibility', 'none');
        }}
      }}

      document.querySelectorAll('[data-action="toggle-delay-map"]').forEach((btn) => {{
        btn.addEventListener('click', () => {{ if (_mapRef) setDelayMapVisible(_mapRef, !_delayMapVisible); }});
      }});

      // ── Surface backend ANOMALIES into alert feed on load ────────────────────
      (function () {{
        if (!ANOMALIES || !ANOMALIES.length) return;
        const sevMap = {{ alert: 'critical', warning: 'warning', info: 'info' }};
        const iconMap = {{ alert: '🚨', warning: '⚠️', info: 'ℹ️' }};
        // Only push high-severity ones to avoid noise
        ANOMALIES.filter((a) => a.severity === 'alert' || a.severity === 'warning')
          .slice(0, 6)
          .forEach((a) => {{
            pushAlert(
              sevMap[a.severity] || 'info',
              iconMap[a.severity] || 'ℹ️',
              `${{a.anomaly_type.replace(/_/g,' ')}} · ${{a.flight_iata || 'Unknown'}}`,
              a.description || a.all_anomalies || ''
            );
          }});
      }})();

      // ── Data freshness badge ──────────────────────────────────────────────────
      (function () {{
        const badge  = document.getElementById('freshness-badge');
        const text   = document.getElementById('freshness-text');
        if (!badge || !text) return;
        const REFRESH_INTERVAL_SEC = 60;
        function _tick() {{
          const now    = Math.floor(Date.now() / 1000);
          const ageSec = now - _dataFetchTs;
          const nextIn = Math.max(0, REFRESH_INTERVAL_SEC - ageSec);
          const nextLabel = nextIn > 0 ? ' \u00b7 next in ' + nextIn + 's' : ' \u00b7 updating\u2026';
          let cls, label;
          if (ageSec < 90) {{
            cls = 'fresh';
            label = 'Live \u00b7 ' + ageSec + 's ago' + nextLabel;
          }} else if (ageSec < 300) {{
            const m = Math.floor(ageSec / 60);
            cls = 'stale';
            label = 'Stale \u00b7 ' + m + 'm ago' + nextLabel;
          }} else {{
            const m = Math.floor(ageSec / 60);
            cls = 'offline';
            label = 'Offline \u00b7 ' + m + 'm ago';
          }}
          badge.className = cls;
          text.textContent = label;
        }}
        _tick();
        setInterval(_tick, 1000);
      }})();

      map.on('error', (event) => {{
        // Individual tile/image fetch failures are non-fatal — silence them.
        const detail = event && event.error && event.error.message ? event.error.message : '';
        const isTileNoise = !detail ||
          detail.toLowerCase().includes('tile') ||
          detail.toLowerCase().includes('image') ||
          detail.toLowerCase().includes('fetch') ||
          detail.toLowerCase().includes('network');
        if (!isTileNoise) {{
          // Fatal map error (bad token, missing style, WebGL loss) — surface to user.
          showError('Map error: ' + detail);
        }}
      }});
    }} catch (error) {{
      showError(error && error.message ? error.message : 'Unable to initialize the Mapbox base map.');
    }}
  </script>
  <script>
    /* Height is fixed at {height}px — Streamlit sets the iframe height to the same value.
       No dynamic resize needed; it causes iframe height conflicts. */

    // ── Mobile touch fix: disable Mapbox handlers while finger is on a panel ──
    // stopPropagation() does NOT work — Mapbox listens on the canvas element
    // directly. The only reliable fix is disabling dragPan / touchZoomRotate
    // on touchstart inside a panel and re-enabling on touchend/touchcancel.
    (function() {{
      var PANEL_SELECTOR =
        '.left-rail, .right-panel, .filter-panel, ' +
        '.alerts-side-panel, #asp-body, #sched-content, ' +
        '.search-dropdown, .module-strip, ' +
        '#flight-board-overlay .fids-table-wrap';

      var _touching = 0; // reference-count in case of nested panels

      function mapHandlers(action) {{
        var m = window._mapRef;
        if (!m) return;
        try {{
          m.dragPan[action]();
          m.touchZoomRotate[action]();
          m.dragRotate[action]();
        }} catch(e) {{ /* map not yet ready — ignore */ }}
      }}

      document.addEventListener('touchstart', function(e) {{
        if (e.target.closest(PANEL_SELECTOR)) {{
          _touching++;
          if (_touching === 1) mapHandlers('disable');
        }}
      }}, {{ passive: true, capture: true }});

      function onTouchEnd(e) {{
        if (_touching > 0) {{
          _touching--;
          if (_touching === 0) mapHandlers('enable');
        }}
      }}
      document.addEventListener('touchend',    onTouchEnd, {{ passive: true, capture: true }});
      document.addEventListener('touchcancel', onTouchEnd, {{ passive: true, capture: true }});
    }})();
  </script>
</body>
</html>
"""
