"""
cesium_map.py
-------------
Generates a MapLibre GL JS satellite map in a 2D operations view with real
AirLabs flight data — aircraft markers with heading rotation, click popups,
flight trails, great-circle route arcs, altitude sparklines,
heatmap layer, weather overlay, time-slider playback,
arrival/departure board, flight comparison, keyboard shortcuts,
sound alerts, export, and responsive layout.

Usage:
    from cesium_map import generate_cesium_html
    html = generate_cesium_html(df, selected_flight="AI101")
    components.html(html, height=680)

No map token is required. The map uses MapLibre GL JS with Esri satellite tiles.
"""

import json
import pandas as pd
import logging

from aircraft_icons import get_family, _get_svg
from data_fetcher import get_airline_name, get_airports_for_region
from analytics import compute_fuel_estimate, detect_anomalies
from utils import safe_text as _safe_text, normalize_flight_query as _normalize_flight_query, is_selected_flight as _is_selected_flight

logger = logging.getLogger(__name__)


# ── High-visibility icon colours (satellite map optimised) ───────────────────────
_FR24_YELLOW = "#00E5FF"
_FR24_SELECTED = "#FF1744"
_FR24_GROUND = "#00B0CC"   # slightly darker cyan for ground aircraft

def _alt_hex(altitude_ft, highlight: bool = False) -> str:
    if highlight:
        return _FR24_SELECTED
    if pd.isna(altitude_ft):
        return _FR24_YELLOW
    alt = float(altitude_ft)
    if alt < 1_000:    return _FR24_GROUND
    else:              return _FR24_YELLOW


def _dim_hex(altitude_ft) -> str:
    """Muted teal when a single tracked flight should stand out."""
    return "#4DD0E1"


# ── Airline colour palette ─────────────────────────────────────────────────────
_AIRLINE_COLORS: dict[str, str] = {
    "AI": "#FF6D00",   # Air India — saffron orange
    "6E": "#BCFF5C",   # IndiGo — lime green
    "UK": "#FF4081",   # Vistara — magenta pink
    "SG": "#FFCA28",   # SpiceJet — spicy yellow
    "G8": "#40C4FF",   # GoFirst — sky blue
    "QP": "#E040FB",   # Akasa Air — purple
    "IX": "#FF8A65",   # Air India Express — coral
    "I5": "#26C6DA",   # AirAsia India — teal
    "S5": "#7C4DFF",   # StarAir — violet
    "2T": "#69F0AE",   # TruJet — mint
}

def _airline_color(airline_code: str, altitude_ft) -> str:
    """Return airline-specific colour, falling back to altitude-based default."""
    c = _AIRLINE_COLORS.get(airline_code)
    if c:
        return c
    return _alt_hex(altitude_ft)


# ── Prepare flight JSON for embedding ─────────────────────────────────────────
def _build_flights_json(
    df: pd.DataFrame,
    selected_flight: str,
    trails: dict,
    airports: list[dict] | None = None,
) -> str:
    """Convert DataFrame + trail history into embedded JS JSON with route arcs, fuel, anomalies."""
    flights = []
    has_selected_query = bool(_normalize_flight_query(selected_flight))
    airport_by_code = {a["iata"]: a for a in (airports or [])}

    # Cap flight count to prevent oversized HTML (Streamlit iframe ~500KB limit)
    max_flights = 600
    processing_df = df.head(max_flights) if len(df) > max_flights else df

    for _, row in processing_df.iterrows():
        try:
            lat  = row.get("latitude")
            lng  = row.get("longitude")
            if pd.isna(lat) or pd.isna(lng):
                continue

            alt_ft   = row.get("altitude_ft",  0)
            alt_m    = row.get("altitude_m",    0)
            heading  = row.get("heading",       0)
            speed    = row.get("speed_kts",     0)
            v_speed  = row.get("v_speed",       0)

            flight   = str(row.get("flight_iata",  "N/A"))
            airline_code = str(row.get("airline_iata", "N/A"))
            airline  = get_airline_name(airline_code) if airline_code != "N/A" else "Unknown airline"
            dep      = str(row.get("dep_iata",     "N/A"))
            arr      = str(row.get("arr_iata",     "N/A"))
            aircraft = str(row.get("aircraft_icao","N/A"))
            reg      = str(row.get("reg_number",   "N/A"))
            status   = str(row.get("status",       "unknown"))
            hex_code = str(row.get("hex",          ""))
            stale_min = float(row.get("stale_minutes", 0) or 0)

            # Safe numeric
            alt_ft_f  = float(alt_ft)  if pd.notna(alt_ft)  else 0.0
            alt_m_f   = float(alt_m)   if pd.notna(alt_m)   else 0.0
            hdg_f     = float(heading) if pd.notna(heading)  else 0.0
            spd_f     = float(speed)   if pd.notna(speed)    else 0.0
            vs_f      = float(v_speed) if pd.notna(v_speed)  else 0.0

            is_selected = _is_selected_flight(flight, selected_flight)

            color = (
                _alt_hex(alt_ft_f, highlight=True)
                if is_selected else
                _dim_hex(alt_ft_f) if has_selected_query else
                _airline_color(airline_code, alt_ft_f)
            )
            family = get_family(aircraft)
            route = f"{dep} → {arr}" if dep != "N/A" and arr != "N/A" else "N/A"

            vstatus = ("⬆ Climbing"   if vs_f > 1
                  else "⬇ Descending" if vs_f < -1
                  else "➡ Level")

            # Trail positions + altitude history for sparkline
            trail = []
            alt_history = []
            trail_positions_raw = []
            if hex_code and hex_code in trails:
                for p in trails[hex_code].get("positions", []):
                    trail_positions_raw.append(p)
                    if p.get("lat") is not None and p.get("lng") is not None:
                        trail.append({
                            "lat": float(p["lat"]),
                            "lng": float(p["lng"]),
                        })
                    if p.get("altitude_ft") is not None:
                        alt_history.append(round(float(p["altitude_ft"])))

            # Segment trail: keep only positions after the last large geographic jump
            # (handles aircraft reuse across different flight legs)
            if len(trail) >= 2:
                last_seg_start = 0
                for ti in range(1, len(trail)):
                    dlat = trail[ti]["lat"] - trail[ti-1]["lat"]
                    dlng = trail[ti]["lng"] - trail[ti-1]["lng"]
                    # ~150km jump threshold (rough: 1 deg ≈ 111km)
                    if (dlat*dlat + dlng*dlng) > 1.8:
                        last_seg_start = ti
                if last_seg_start > 0:
                    trail = trail[last_seg_start:]
                    alt_history = alt_history[-len(trail):] if alt_history else []

            # Destination airport coords for great-circle arc
            dest_lat = None
            dest_lng = None
            dep_lat = None
            dep_lng = None
            dest_airport = airport_by_code.get(arr)
            dep_airport = airport_by_code.get(dep)
            if dest_airport:
                dest_lat = dest_airport["lat"]
                dest_lng = dest_airport["lng"]
            if dep_airport:
                dep_lat = dep_airport["lat"]
                dep_lng = dep_airport["lng"]

            # Fuel / CO₂ estimate
            fuel_data = {}
            if dest_airport and pd.notna(lat) and pd.notna(lng):
                from analytics import haversine_km
                dist = haversine_km(
                    dest_airport["lat"], dest_airport["lng"],
                    pd.Series([float(lat)]), pd.Series([float(lng)])
                ).iloc[0]
                if pd.notna(dist):
                    est = compute_fuel_estimate(aircraft, float(dist))
                    fuel_data = {
                        "fuel_kg": est.fuel_burn_kg,
                        "co2_kg": est.co2_kg,
                        "co2_per_pax": est.co2_per_pax_kg,
                        "efficiency": est.efficiency_label,
                        "distance_km": est.distance_km,
                    }

            # Anomaly detection
            anomaly_list = detect_anomalies(row, trail_positions_raw, airports or [])
            anomaly_data = []
            for a in anomaly_list[:3]:
                anomaly_data.append({
                    "type": a.anomaly_type,
                    "severity": a.severity,
                    "desc": a.description,
                })

            flights.append({
                "lat":      float(lat),
                "lng":      float(lng),
                "alt_m":    alt_m_f,
                "alt_ft":   round(alt_ft_f),
                "heading":  hdg_f,
                "flight":   flight,
                "airline":  airline,
                "airline_code": airline_code,
                "dep":      dep,
                "arr":      arr,
                "route":    route,
                "aircraft": aircraft,
                "reg":      reg,
                "speed":    round(spd_f, 1),
                "vstatus":  vstatus,
                "status":   status.capitalize(),
                "color":    color,
                "selected": is_selected,
                "dimmed":   bool(has_selected_query and not is_selected),
                "family":   family,
                "trail":    trail,
                "alt_history": alt_history,
                "dest_lat": dest_lat,
                "dest_lng": dest_lng,
                "dep_lat":  dep_lat,
                "dep_lng":  dep_lng,
                "fuel":     fuel_data,
                "anomalies": anomaly_data,
                "delay_risk":   str(row.get("predicted_delay_risk", "Low")),
                "delay_min":    int(row.get("predicted_delay_min", 0) or 0),
                "delay_eta":    str(row.get("predicted_eta_utc", "")),
                "delay_reason": str(row.get("predicted_delay_reason", "")),
                "weather_sev":  str(row.get("weather_severity", "Low")),
                "stale_min":    round(stale_min, 1),
                "afri":         int(row.get("afri_score", 0) or 0),
                "afri_level":   str(row.get("afri_level", "Normal")),
                "afri_drivers": str(row.get("afri_drivers", "")),
            })

        except Exception as e:
            logger.debug(f"Skipped flight: {e}")

    return json.dumps(flights)


# ── Main HTML generator ────────────────────────────────────────────────────────
def generate_cesium_html(
    df: pd.DataFrame,
    selected_flight: str = None,
    trails: dict = None,
    center_lat: float = 25.0,
    center_lng: float = 45.0,
    zoom_level: int = 5,
    region: str = "india",
    snapshot_history: list[dict] | None = None,
    weather_overlays: list[dict] | None = None,
    anomalies_df=None,
    schedule_arrivals: list[dict] | None = None,
    schedule_departures: list[dict] | None = None,
    schedule_airport: str = "",
    airport_metrics: list[dict] | None = None,
    route_metrics: list[dict] | None = None,
) -> str:
    """Generate a complete MapLibre GL JS satellite map HTML page."""
    if trails is None:
        trails = {}

    airports_list = get_airports_for_region(region)
    flights_json = _build_flights_json(df, selected_flight or "", trails, airports=airports_list)
    airports_json = json.dumps(airports_list)

    _all_families = [
        "b747", "a380", "a340", "b777", "a350", "a330", "b787", "b767",
        "b757", "a320", "b737", "e190", "crj", "atr", "concorde", "military", "default",
    ]
    _icon_bodies = {}
    for fam in _all_families:
        body = _get_svg(fam, "COLORTOK")
        _icon_bodies[fam] = body
    icon_templates_json = json.dumps(_icon_bodies)

    flight_count = len(df)
    history_json = json.dumps(snapshot_history or [])
    weather_json = json.dumps(weather_overlays or [])
    arrivals_json = json.dumps(schedule_arrivals or [])
    departures_json = json.dumps(schedule_departures or [])
    schedule_airport_safe = json.dumps(str(schedule_airport or ""))[1:-1]  # JSON-escaped, no quotes
    airport_metrics_json = json.dumps(airport_metrics or [])
    route_metrics_json = json.dumps(route_metrics or [])

    # Selected flight fly-to JS
    selected_js = ""
    if selected_flight and not df.empty:
        normalized_selected = _normalize_flight_query(selected_flight)
        normalized_flights = df["flight_iata"].fillna("").map(_normalize_flight_query)
        matches = df[normalized_flights == normalized_selected]
        if matches.empty:
            matches = df[normalized_flights.str.contains(normalized_selected, na=False)]
        if not matches.empty:
            r = matches.iloc[0]
            s_lat = float(r["latitude"])
            s_lng = float(r["longitude"])
            selected_js = f"""
            showPopupByFlight('{str(r.get("flight_iata", "")).replace("'", "\\'")}');
            map.flyTo({{ center: [{s_lng}, {s_lat}], zoom: 7, duration: 2000 }});"""

    return _build_full_html(
        flights_json=flights_json,
        airports_json=airports_json,
        flight_count=flight_count,
        selected_flight=selected_flight or "",
        selected_js=selected_js,
        center_lat=center_lat,
        center_lng=center_lng,
        zoom_level=zoom_level,
        history_json=history_json,
        weather_json=weather_json,
        arrivals_json=arrivals_json,
        departures_json=departures_json,
        schedule_airport=schedule_airport_safe,
        airport_metrics_json=airport_metrics_json,
        route_metrics_json=route_metrics_json,
        icon_templates_json=icon_templates_json,
    )


# ── Full HTML builder — MapLibre GL JS ────────────────────────────────────────

def _build_full_html(
    flights_json: str,
    airports_json: str,
    flight_count: int,
    selected_flight: str,
    selected_js: str,
    center_lat: float,
    center_lng: float,
    zoom_level: int,
    history_json: str,
    weather_json: str,
    arrivals_json: str,
    departures_json: str,
    schedule_airport: str,
    airport_metrics_json: str = "[]",
    route_metrics_json: str = "[]",
    icon_templates_json: str = "{}",
) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Aviation Intelligence — Satellite Map</title>
  <link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet">
  <script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
  <script src="https://html2canvas.hertzen.com/dist/html2canvas.min.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html, body {{ width: 100%; height: 100%; overflow: hidden; background: #0a1018; font-family: 'SF Pro Display', 'Segoe UI', Arial, sans-serif; }}
    #map {{ width: 100%; height: 100%; }}
    .maplibregl-ctrl-logo, .maplibregl-ctrl-attrib {{ display: none !important; }}

    /* ── Loading skeleton ── */
    #skeleton {{
      position: fixed; inset: 0; z-index: 9999; background: #0a1018;
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      gap: 18px; transition: opacity 0.6s ease;
    }}
    #skeleton.hidden {{ opacity: 0; pointer-events: none; }}
    .skel-bar {{
      width: 280px; height: 14px; border-radius: 7px;
      background: linear-gradient(90deg, rgba(255,255,255,0.04) 25%, rgba(255,255,255,0.10) 50%, rgba(255,255,255,0.04) 75%);
      background-size: 400% 100%; animation: shimmer 1.8s ease infinite;
    }}
    .skel-bar:nth-child(2) {{ width: 200px; animation-delay: 0.2s; }}
    .skel-bar:nth-child(3) {{ width: 240px; animation-delay: 0.4s; }}
    .skel-title {{ color: rgba(255,255,255,0.25); font-size: 22px; font-weight: 800; letter-spacing: -0.03em; margin-bottom: 8px; }}
    @keyframes shimmer {{ 0% {{ background-position: 200% 0; }} 100% {{ background-position: -200% 0; }} }}

    /* ── Glass panel base ── */
    .glass-panel {{
      background: rgba(14, 18, 26, 0.82); border: 1px solid rgba(255,255,255,0.14);
      border-radius: 16px; backdrop-filter: blur(12px); color: #fff;
      box-shadow: 0 12px 32px rgba(0,0,0,0.28); font-size: 12px;
    }}

    /* ── Top bar ── */
    #top-bar {{
      position: absolute; top: 10px; left: 50%; transform: translateX(-50%);
      z-index: 200; padding: 7px 20px; display: flex; gap: 16px; align-items: center;
    }}
    #top-bar .item {{ display: flex; flex-direction: column; align-items: center; }}
    #top-bar .val {{ font-size: 15px; font-weight: 700; color: #f5f7fa; }}
    #top-bar .lbl {{ font-size: 9px; color: rgba(255,255,255,0.62); text-transform: uppercase; letter-spacing: 0.06em; }}
    #top-bar .sep {{ width: 1px; height: 24px; background: rgba(255,255,255,0.14); }}

    /* ── Toggle buttons ── */
    #toggles {{
      position: absolute; top: 10px; right: 12px; z-index: 210;
      display: flex; gap: 6px; flex-wrap: wrap;
    }}
    .toggle-btn {{
      padding: 6px 12px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.12);
      background: rgba(14, 18, 26, 0.78); color: rgba(255,255,255,0.72);
      font-size: 11px; font-weight: 700; cursor: pointer; backdrop-filter: blur(8px);
      transition: all 0.15s ease;
    }}
    .toggle-btn:hover {{ color: #fff; border-color: rgba(255,255,255,0.28); }}
    .toggle-btn.active {{ background: rgba(188,255,92,0.18); color: #bcff5c; border-color: rgba(188,255,92,0.32); }}

    /* ── Legend ── */
    #legend {{
      position: absolute; bottom: 16px; left: 10px; z-index: 200; padding: 10px 14px; min-width: 160px;
    }}
    #legend b {{ display: block; margin-bottom: 6px; font-size: 12px; }}
    #legend .row {{ display: flex; align-items: center; gap: 7px; margin-bottom: 4px; font-size: 11px; }}
    #legend .dot {{ width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }}

    /* ── My Location pulse ── */
    @keyframes locPulse {{
      0%   {{ transform: scale(1);   opacity: 0.9; }}
      100% {{ transform: scale(3.5); opacity: 0; }}
    }}
    #my-loc-panel {{
      position: absolute; bottom: 16px; right: 12px; z-index: 210;
      width: 280px; padding: 12px 14px; display: none;
    }}
    #my-loc-panel .oh-title {{ font-size: 13px; font-weight: 800; color: #bcff5c; margin-bottom: 8px; letter-spacing: -0.02em; }}
    #my-loc-panel .oh-item {{
      display: flex; justify-content: space-between; padding: 5px 0;
      border-bottom: 1px solid rgba(255,255,255,0.06); font-size: 11px; cursor: pointer;
    }}
    #my-loc-panel .oh-item:hover {{ color: #00E5FF; }}
    #my-loc-panel .oh-flight {{ font-weight: 700; color: #fff; }}
    #my-loc-panel .oh-detail {{ color: rgba(255,255,255,0.55); }}
    #my-loc-panel .oh-alt {{ color: #FFCA28; font-weight: 600; }}
    #my-loc-panel .oh-empty {{ color: rgba(255,255,255,0.35); font-size: 11px; padding: 8px 0; }}

    /* ── Flight popup ── */
    #popup {{
      position: absolute; top: 60px; right: 12px; z-index: 220;
      width: 340px; padding: 16px; display: none; max-height: calc(100vh - 80px); overflow-y: auto;
      scrollbar-width: thin; scrollbar-color: rgba(255,255,255,0.1) transparent;
    }}
    #popup h4 {{ font-size: 22px; margin: 0; font-weight: 800; }}
    #popup-header {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 10px; margin-bottom: 8px; }}
    #popup-meta {{ display: flex; flex-direction: column; gap: 3px; }}
    #popup-chip-row {{ display: flex; gap: 6px; flex-wrap: wrap; margin-top: 6px; }}
    .popup-chip {{
      display: inline-flex; align-items: center; border-radius: 999px;
      padding: 4px 9px; font-size: 10px; font-weight: 700;
      background: rgba(255,255,255,0.08); color: rgba(255,255,255,0.82);
      border: 1px solid rgba(255,255,255,0.06);
    }}
    .popup-chip.accent {{ background: rgba(255,59,48,0.16); color: #ff9d97; border-color: rgba(255,59,48,0.24); }}
    .popup-chip.warn {{ background: rgba(255,165,0,0.16); color: #ffc97a; border-color: rgba(255,165,0,0.24); }}
    #popup-close {{ position: absolute; top: 8px; right: 12px; background: none; border: none; color: #666; font-size: 16px; cursor: pointer; }}
    #popup-close:hover {{ color: #fff; }}
    #popup-route {{
      display: grid; grid-template-columns: 1fr auto 1fr; gap: 8px;
      align-items: center; margin: 8px 0 12px;
      padding: 12px 0; border-top: 1px solid rgba(255,255,255,0.08);
      border-bottom: 1px solid rgba(255,255,255,0.08);
    }}
    #popup-route .code {{ font-size: 30px; font-weight: 800; line-height: 1; }}
    #popup-route .city {{ font-size: 11px; color: rgba(255,255,255,0.62); margin-top: 4px; }}
    #popup-route .plane {{ font-size: 18px; color: #FFCF5C; }}
    #popup-stats {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 12px; }}
    .popup-stat {{ background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 8px 10px; }}
    .popup-stat .label {{ font-size: 9px; text-transform: uppercase; letter-spacing: 0.06em; color: rgba(255,255,255,0.52); margin-bottom: 4px; }}
    .popup-stat .value {{ font-size: 16px; font-weight: 800; }}
    #popup table {{ width: 100%; border-collapse: collapse; }}
    #popup td {{ padding: 5px 0; vertical-align: top; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 11px; }}
    #popup td:first-child {{ color: rgba(255,255,255,0.6); width: 38%; }}
    #popup td:last-child {{ font-weight: 600; }}
    #popup-actions {{ display: flex; gap: 6px; margin-top: 10px; }}
    #popup-actions button {{ flex: 1; border: none; border-radius: 8px; padding: 8px; background: rgba(255,255,255,0.08); color: #fff; cursor: pointer; font-weight: 600; font-size: 11px; }}
    #popup-actions button:hover {{ background: rgba(255,255,255,0.14); }}

    /* ── Sparkline ── */
    #sparkline-wrap {{ margin: 10px 0 6px; padding: 8px; background: rgba(255,255,255,0.04); border-radius: 10px; border: 1px solid rgba(255,255,255,0.06); }}
    #sparkline-wrap .label {{ font-size: 9px; text-transform: uppercase; letter-spacing: 0.06em; color: rgba(255,255,255,0.5); margin-bottom: 4px; }}
    #sparkline-canvas {{ width: 100%; height: 50px; }}

    /* ── Fuel card ── */
    #fuel-card {{ margin: 8px 0; padding: 10px; background: rgba(46,204,113,0.08); border: 1px solid rgba(46,204,113,0.18); border-radius: 10px; display: none; }}
    #fuel-card .fuel-title {{ font-size: 9px; text-transform: uppercase; letter-spacing: 0.06em; color: rgba(46,204,113,0.8); margin-bottom: 5px; }}
    #fuel-card .fuel-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }}
    #fuel-card .fuel-item {{ font-size: 11px; }}
    #fuel-card .fuel-item strong {{ display: block; font-size: 14px; font-weight: 800; color: #fff; }}
    #fuel-card .fuel-item span {{ color: rgba(255,255,255,0.52); }}

    /* ── Anomaly badge ── */
    #anomaly-bar {{ margin: 8px 0 4px; }}
    .anomaly-chip {{
      display: inline-flex; align-items: center; gap: 4px; padding: 4px 9px; border-radius: 999px;
      font-size: 10px; font-weight: 700; margin-right: 4px; margin-bottom: 4px;
    }}
    .anomaly-chip.alert {{ background: rgba(255,59,48,0.18); color: #ff7b73; border: 1px solid rgba(255,59,48,0.28); }}
    .anomaly-chip.warning {{ background: rgba(255,165,0,0.16); color: #ffc97a; border: 1px solid rgba(255,165,0,0.24); }}
    .anomaly-chip.info {{ background: rgba(89,183,255,0.14); color: #9dd4ff; border: 1px solid rgba(89,183,255,0.22); }}
    .stale-badge {{ display: inline-block; padding: 2px 7px; border-radius: 999px; font-size: 10px; font-weight: 700; letter-spacing: 0.04em; margin-left: 6px; }}
    .stale-badge.lost {{ background: rgba(255,59,48,0.22); color: #ff7b73; border: 1px solid rgba(255,59,48,0.3); }}
    .stale-badge.warn {{ background: rgba(255,165,0,0.18); color: #ffc97a; border: 1px solid rgba(255,165,0,0.25); }}
    .stale-badge.lag  {{ background: rgba(255,255,255,0.08); color: rgba(255,255,255,0.55); border: 1px solid rgba(255,255,255,0.12); }}

    /* ── Flight board ── */
    #flight-board {{
      position: absolute; bottom: 16px; right: 12px; z-index: 200;
      width: 360px; max-height: 340px; display: none; overflow: hidden;
    }}
    #flight-board .board-title {{
      font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.10em;
      padding: 10px 14px 8px; color: rgba(255,255,255,0.6); border-bottom: 1px solid rgba(255,255,255,0.08);
      display: flex; justify-content: space-between; align-items: center;
    }}
    #flight-board .board-tabs {{ display: flex; gap: 0; }}
    #flight-board .board-tabs button {{
      padding: 5px 10px; border: none; background: none; color: rgba(255,255,255,0.5);
      font-size: 10px; font-weight: 700; cursor: pointer; border-bottom: 2px solid transparent;
    }}
    #flight-board .board-tabs button.active {{ color: #bcff5c; border-bottom-color: #bcff5c; }}
    #board-body {{
      overflow-y: auto; max-height: 280px; padding: 0 14px 10px;
      scrollbar-width: thin; scrollbar-color: rgba(255,255,255,0.1) transparent;
    }}
    .board-row {{
      display: grid; grid-template-columns: 62px 1fr 60px 68px; gap: 8px;
      padding: 7px 0; border-bottom: 1px solid rgba(255,255,255,0.05);
      font-size: 11px; align-items: center;
    }}
    .board-row .fl {{ font-weight: 800; color: #fff; }}
    .board-row .route {{ color: rgba(255,255,255,0.6); }}
    .board-row .time {{ color: rgba(255,255,255,0.72); font-weight: 600; font-variant-numeric: tabular-nums; }}
    .board-row .st {{ font-size: 9px; font-weight: 700; padding: 3px 7px; border-radius: 999px; text-align: center; }}
    .st-active {{ background: rgba(46,204,113,0.16); color: #5dd39e; }}
    .st-scheduled {{ background: rgba(89,183,255,0.14); color: #8fc8ff; }}
    .st-delayed {{ background: rgba(255,165,0,0.16); color: #ffb74d; }}
    .st-landed {{ background: rgba(255,255,255,0.08); color: rgba(255,255,255,0.52); }}

    /* ── Time slider ── */
    #time-slider-wrap {{
      position: absolute; bottom: 16px; left: 50%; transform: translateX(-50%);
      z-index: 200; width: 380px; padding: 10px 16px; display: none;
    }}
    #time-slider-wrap .ts-label {{ font-size: 10px; color: rgba(255,255,255,0.58); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; }}
    #time-slider-wrap .ts-value {{ font-size: 16px; font-weight: 800; color: #bcff5c; margin-bottom: 8px; }}
    #time-slider {{
      width: 100%; appearance: none; height: 4px; border-radius: 2px;
      background: rgba(255,255,255,0.14); outline: none; cursor: pointer;
    }}
    #time-slider::-webkit-slider-thumb {{
      appearance: none; width: 16px; height: 16px; border-radius: 50%;
      background: #bcff5c; cursor: pointer; box-shadow: 0 0 10px rgba(188,255,92,0.4);
    }}
    #time-slider-info {{ font-size: 10px; color: rgba(255,255,255,0.5); margin-top: 6px; text-align: center; }}

    /* ── Comparison panel ── */
    #compare-panel {{
      position: absolute; top: 60px; left: 12px; z-index: 200;
      width: 320px; padding: 14px; display: none;
    }}
    #compare-panel h5 {{ font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em; color: rgba(255,255,255,0.55); margin-bottom: 10px; }}
    .compare-grid {{ display: grid; grid-template-columns: 1fr auto 1fr; gap: 8px; }}
    .compare-col {{ text-align: center; }}
    .compare-col .cf {{ font-size: 20px; font-weight: 800; color: #fff; margin-bottom: 4px; }}
    .compare-col .ca {{ font-size: 11px; color: rgba(255,255,255,0.55); margin-bottom: 8px; }}
    .compare-vs {{ font-size: 12px; font-weight: 800; color: #ffcf5c; align-self: center; }}
    .compare-row {{ display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 11px; }}
    .compare-row .cr-label {{ color: rgba(255,255,255,0.5); }}
    .compare-row .cr-a, .compare-row .cr-b {{ font-weight: 700; color: #fff; min-width: 60px; }}
    .compare-row .cr-a {{ text-align: left; }}
    .compare-row .cr-b {{ text-align: right; }}

    /* ── Keyboard hint ── */
    #kbd-hint {{
      position: absolute; bottom: 16px; left: 50%; transform: translateX(-50%);
      z-index: 190; padding: 8px 16px; font-size: 10px; color: rgba(255,255,255,0.4);
      display: flex; gap: 14px;
    }}
    #kbd-hint kbd {{
      display: inline-block; padding: 2px 6px; border-radius: 4px;
      background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.12);
      font-size: 10px; font-weight: 700; color: rgba(255,255,255,0.6); margin-right: 3px;
    }}

    /* ── Export toast ── */
    #export-toast {{
      position: fixed; top: 20px; left: 50%; transform: translateX(-50%); z-index: 9999;
      padding: 10px 22px; border-radius: 12px; background: rgba(46,204,113,0.92);
      color: #fff; font-size: 13px; font-weight: 700; display: none;
      box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }}

    /* ── Delay card ── */
    #delay-card {{ margin: 8px 0; padding: 10px 12px; border-radius: 10px; display: none; }}
    #delay-card.risk-low {{ background: rgba(46,204,113,0.10); border: 1px solid rgba(46,204,113,0.22); }}
    #delay-card.risk-medium {{ background: rgba(255,165,0,0.10); border: 1px solid rgba(255,165,0,0.22); }}
    #delay-card.risk-high {{ background: rgba(255,59,48,0.10); border: 1px solid rgba(255,59,48,0.22); }}
    #delay-card .dc-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }}
    #delay-card .dc-title {{ font-size: 9px; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 700; }}
    .risk-low .dc-title {{ color: rgba(46,204,113,0.85); }}
    .risk-medium .dc-title {{ color: rgba(255,165,0,0.85); }}
    .risk-high .dc-title {{ color: rgba(255,59,48,0.85); }}
    #delay-card .dc-badge {{ padding: 3px 10px; border-radius: 999px; font-size: 10px; font-weight: 800; }}
    .risk-low .dc-badge {{ background: rgba(46,204,113,0.22); color: #5dd39e; }}
    .risk-medium .dc-badge {{ background: rgba(255,165,0,0.22); color: #ffb74d; }}
    .risk-high .dc-badge {{ background: rgba(255,59,48,0.22); color: #ff7b73; }}
    #delay-card .dc-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }}
    #delay-card .dc-item span {{ font-size: 10px; color: rgba(255,255,255,0.52); display: block; }}
    #delay-card .dc-item strong {{ font-size: 15px; font-weight: 800; color: #fff; }}
    #delay-card .dc-reasons {{ margin-top: 6px; display: flex; gap: 4px; flex-wrap: wrap; }}
    #delay-card .dc-reason-tag {{
      padding: 3px 8px; border-radius: 999px; font-size: 9px; font-weight: 700;
      background: rgba(255,255,255,0.07); color: rgba(255,255,255,0.65);
      border: 1px solid rgba(255,255,255,0.08);
    }}

    /* ── AFRI card ── */
    #afri-card {{ margin: 8px 0; padding: 10px 12px; border-radius: 10px; display: none; }}
    #afri-card.afri-normal    {{ background: rgba(46,204,113,0.08); border: 1px solid rgba(46,204,113,0.18); }}
    #afri-card.afri-elevated  {{ background: rgba(89,183,255,0.10); border: 1px solid rgba(89,183,255,0.22); }}
    #afri-card.afri-high      {{ background: rgba(255,165,0,0.10); border: 1px solid rgba(255,165,0,0.22); }}
    #afri-card.afri-critical  {{ background: rgba(255,59,48,0.10); border: 1px solid rgba(255,59,48,0.22); }}
    #afri-card .afri-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }}
    #afri-card .afri-title {{ font-size: 9px; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 700; }}
    .afri-normal .afri-title {{ color: rgba(46,204,113,0.85); }}
    .afri-elevated .afri-title {{ color: rgba(89,183,255,0.85); }}
    .afri-high .afri-title {{ color: rgba(255,165,0,0.85); }}
    .afri-critical .afri-title {{ color: rgba(255,59,48,0.85); }}
    #afri-card .afri-badge {{ padding: 3px 10px; border-radius: 999px; font-size: 10px; font-weight: 800; }}
    .afri-normal .afri-badge {{ background: rgba(46,204,113,0.22); color: #5dd39e; }}
    .afri-elevated .afri-badge {{ background: rgba(89,183,255,0.22); color: #8fc8ff; }}
    .afri-high .afri-badge {{ background: rgba(255,165,0,0.22); color: #ffb74d; }}
    .afri-critical .afri-badge {{ background: rgba(255,59,48,0.22); color: #ff7b73; }}
    #afri-card .afri-score-ring {{ display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }}
    #afri-card .afri-number {{ font-size: 28px; font-weight: 900; letter-spacing: -0.04em; line-height: 1; }}
    .afri-normal .afri-number {{ color: #5dd39e; }}
    .afri-elevated .afri-number {{ color: #8fc8ff; }}
    .afri-high .afri-number {{ color: #ffb74d; }}
    .afri-critical .afri-number {{ color: #ff7b73; }}
    #afri-card .afri-bar-track {{ flex: 1; height: 6px; background: rgba(255,255,255,0.08); border-radius: 3px; overflow: hidden; }}
    #afri-card .afri-bar-fill {{ height: 100%; border-radius: 3px; transition: width 0.3s ease; }}
    .afri-normal .afri-bar-fill {{ background: linear-gradient(90deg, #2ecc71, #5dd39e); }}
    .afri-elevated .afri-bar-fill {{ background: linear-gradient(90deg, #3498db, #8fc8ff); }}
    .afri-high .afri-bar-fill {{ background: linear-gradient(90deg, #e67e22, #ffb74d); }}
    .afri-critical .afri-bar-fill {{ background: linear-gradient(90deg, #e74c3c, #ff7b73); }}
    #afri-card .afri-drivers {{ margin-top: 4px; display: flex; gap: 4px; flex-wrap: wrap; }}
    #afri-card .afri-driver-tag {{
      padding: 3px 8px; border-radius: 999px; font-size: 9px; font-weight: 700;
      background: rgba(255,255,255,0.07); color: rgba(255,255,255,0.65);
      border: 1px solid rgba(255,255,255,0.08);
    }}

    /* ── Intel panel ── */
    #intel-panel {{
      position: absolute; top: 56px; left: 10px; z-index: 200;
      width: 240px; padding: 0; display: flex; flex-direction: column; gap: 0;
      max-height: calc(100vh - 80px); overflow-y: auto;
      scrollbar-width: thin; scrollbar-color: rgba(255,255,255,0.08) transparent;
    }}
    .intel-section {{ padding: 10px 12px; border-bottom: 1px solid rgba(255,255,255,0.06); }}
    .intel-section:last-child {{ border-bottom: none; }}
    .intel-section .is-title {{
      font-size: 9px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em;
      color: rgba(255,255,255,0.45); margin-bottom: 8px;
    }}
    .intel-item {{
      display: flex; justify-content: space-between; align-items: center;
      padding: 4px 0; font-size: 11px; cursor: pointer; border-radius: 6px; transition: background 0.12s;
    }}
    .intel-item:hover {{ background: rgba(255,255,255,0.05); }}
    .intel-item .ii-code {{ font-weight: 700; color: #e0e6ed; min-width: 46px; }}
    .intel-item .ii-detail {{ color: rgba(255,255,255,0.48); font-size: 10px; flex: 1; text-align: right; }}
    .intel-badge {{
      padding: 2px 7px; border-radius: 999px; font-size: 9px; font-weight: 700; margin-left: 6px;
    }}
    .intel-badge.sev-low {{ background: rgba(46,204,113,0.16); color: #5dd39e; }}
    .intel-badge.sev-moderate {{ background: rgba(89,183,255,0.14); color: #8fc8ff; }}
    .intel-badge.sev-high {{ background: rgba(255,165,0,0.16); color: #ffb74d; }}
    .intel-badge.sev-severe {{ background: rgba(255,59,48,0.16); color: #ff7b73; }}

    /* ── Airport filter bar ── */
    #airport-filter-bar {{
      position: absolute; top: 44px; left: 50%; transform: translateX(-50%);
      z-index: 210; padding: 7px 18px; display: none; align-items: center; gap: 12px;
    }}
    #airport-filter-bar .af-label {{ font-size: 12px; font-weight: 700; color: #c8d6e5; }}
    #airport-filter-bar .af-count {{ font-size: 11px; color: rgba(255,255,255,0.55); }}
    #airport-filter-bar .af-close {{
      background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.12);
      color: #fff; padding: 4px 12px; border-radius: 8px; font-size: 10px;
      font-weight: 700; cursor: pointer;
    }}
    #airport-filter-bar .af-close:hover {{ background: rgba(255,255,255,0.16); }}

    /* ── Responsive ── */
    @media (max-width: 768px) {{
      #top-bar {{ flex-wrap: wrap; padding: 6px 12px; gap: 8px; width: 90%; }}
      #toggles {{ top: auto; bottom: 60px; right: 8px; flex-direction: column; }}
      #popup {{ width: 280px; top: 48px; right: 8px; }}
      #flight-board {{ width: calc(100% - 16px); right: 8px; left: 8px; bottom: 8px; }}
      #compare-panel {{ width: calc(100% - 16px); left: 8px; right: 8px; top: auto; bottom: 8px; }}
      #time-slider-wrap {{ width: calc(100% - 24px); bottom: 8px; }}
      #legend {{ bottom: 8px; left: 6px; max-width: 150px; font-size: 10px; }}
      #kbd-hint {{ display: none; }}
      #intel-panel {{ width: 180px; top: 48px; left: 6px; font-size: 10px; }}
    }}
    @media (max-width: 480px) {{
      #top-bar .sep {{ display: none; }}
      #popup {{ width: calc(100% - 16px); right: 8px; }}
    }}
  </style>
</head>
<body>

<!-- Loading skeleton -->
<div id="skeleton">
  <div class="skel-title">Aviation Intelligence</div>
  <div class="skel-bar"></div><div class="skel-bar"></div>
  <div class="skel-bar"></div><div class="skel-bar"></div>
</div>

<div id="map"></div>

<div id="export-toast">Screenshot saved!</div>

<!-- Top stats bar -->
<div id="top-bar" class="glass-panel">
  <div class="item"><span class="val" id="stat-flights">{flight_count}</span><span class="lbl">Flights</span></div>
  <div class="sep"></div>

  <div class="sep"></div>
  <div class="item"><span class="val" id="stat-anomalies">0</span><span class="lbl">Alerts</span></div>
  <div class="sep"></div>
  <div class="item"><span class="val" id="stat-selected">{selected_flight or "—"}</span><span class="lbl">Tracked</span></div>
</div>

<!-- Toggles -->
<div id="toggles">
  <button class="toggle-btn" onclick="toggleLayer('heatmap')" id="btn-heatmap">Heatmap</button>
  <button class="toggle-btn" onclick="toggleLayer('weather')" id="btn-weather">Weather</button>
  <button class="toggle-btn" onclick="toggleLayer('board')" id="btn-board">Board</button>
  <button class="toggle-btn" onclick="toggleLayer('timeline')" id="btn-timeline">Timeline</button>
  <button class="toggle-btn active" onclick="toggleLayer('intel')" id="btn-intel">Intel</button>
  <button class="toggle-btn" onclick="locateMe()" id="btn-locate">📍 My Location</button>
  <button class="toggle-btn" onclick="exportScreenshot()" id="btn-export">Export</button>
</div>

<!-- Overhead flights panel -->
<div id="my-loc-panel" class="glass-panel">
  <div class="oh-title">✈️ Overhead — <span id="oh-count">0</span> flights nearby</div>
  <div id="oh-list"></div>
</div>

<!-- Legend -->
<div id="legend" class="glass-panel">
  <b>Airlines</b>
  <div class="row"><div class="dot" style="background:#FF6D00"></div>Air India</div>
  <div class="row"><div class="dot" style="background:#BCFF5C"></div>IndiGo</div>
  <div class="row"><div class="dot" style="background:#FF4081"></div>Vistara</div>
  <div class="row"><div class="dot" style="background:#FFCA28"></div>SpiceJet</div>
  <div class="row"><div class="dot" style="background:#40C4FF"></div>GoFirst</div>
  <div class="row"><div class="dot" style="background:#E040FB"></div>Akasa Air</div>
  <div class="row"><div class="dot" style="background:#00E5FF"></div>Other</div>
  <hr style="border:none;border-top:1px solid rgba(255,255,255,0.12);margin:6px 0;">
  <div class="row"><div class="dot" style="background:#FF1744"></div>Tracked</div>
  <div class="row"><div class="dot" style="background:#aab4c6;border:1px solid #1a1a2e"></div>Airport</div>
</div>

<!-- Flight popup -->
<div id="popup" class="glass-panel">
  <button id="popup-close" onclick="closePopup()">✕</button>
  <div id="popup-header">
    <div id="popup-meta">
      <h4 id="popup-title">Flight</h4>
      <div id="popup-subtitle" style="font-size:11px;color:rgba(255,255,255,0.65);"></div>
      <div id="popup-chip-row">
        <span id="popup-airline" class="popup-chip">Airline</span>
        <span id="popup-aircraft-chip" class="popup-chip">Aircraft</span>
        <span id="popup-status-chip" class="popup-chip accent">Status</span>
      </div>
    </div>
  </div>
  <div id="popup-route">
    <div><div id="popup-dep-code" class="code">DEP</div><div id="popup-dep-city" class="city">Departure</div></div>
    <div class="plane">✈</div>
    <div style="text-align:right;"><div id="popup-arr-code" class="code">ARR</div><div id="popup-arr-city" class="city">Arrival</div></div>
  </div>
  <div id="popup-stats">
    <div class="popup-stat"><div class="label">Altitude</div><div class="value" id="popup-altitude-stat">0 ft</div></div>
    <div class="popup-stat"><div class="label">Speed</div><div class="value" id="popup-speed-stat">0 kts</div></div>
  </div>
  <div id="sparkline-wrap"><div class="label">Altitude History</div><canvas id="sparkline-canvas"></canvas></div>
  <div id="fuel-card">
    <div class="fuel-title">Environmental Impact</div>
    <div class="fuel-grid">
      <div class="fuel-item"><span>Fuel burn</span><strong id="fuel-burn">-</strong></div>
      <div class="fuel-item"><span>CO2</span><strong id="fuel-co2">-</strong></div>
      <div class="fuel-item"><span>Per passenger</span><strong id="fuel-pax">-</strong></div>
      <div class="fuel-item"><span>Efficiency</span><strong id="fuel-eff">-</strong></div>
    </div>
  </div>
  <div id="anomaly-bar"></div>
  <div id="delay-card" class="risk-low">
    <div class="dc-header">
      <span class="dc-title">Delay Prediction</span>
      <span class="dc-badge" id="dc-badge">Low</span>
    </div>
    <div class="dc-grid">
      <div class="dc-item"><span>Expected Delay</span><strong id="dc-delay">+0 min</strong></div>
      <div class="dc-item"><span>Est. Arrival</span><strong id="dc-eta">--:-- UTC</strong></div>
    </div>
    <div class="dc-reasons" id="dc-reasons"></div>
  </div>
  <div id="afri-card" class="afri-normal">
    <div class="afri-header">
      <span class="afri-title">Arrival Flow Risk Index</span>
      <span class="afri-badge" id="afri-badge">Normal</span>
    </div>
    <div class="afri-score-ring">
      <span class="afri-number" id="afri-number">0</span>
      <div class="afri-bar-track"><div class="afri-bar-fill" id="afri-bar" style="width:0%"></div></div>
    </div>
    <div class="afri-drivers" id="afri-drivers"></div>
  </div>
  <table id="popup-body"></table>
  <div id="popup-actions">
    <button onclick="refocusSelectedFlight()">Center</button>
    <button onclick="startCompare()">Compare</button>
    <button onclick="closePopup()">Close</button>
  </div>
</div>

<!-- Board -->
<div id="flight-board" class="glass-panel">
  <div class="board-title">
    <span id="board-airport">{schedule_airport or 'Airport'} Board</span>
    <div class="board-tabs">
      <button class="active" onclick="showBoard('arr')">Arrivals</button>
      <button onclick="showBoard('dep')">Departures</button>
    </div>
  </div>
  <div id="board-body"></div>
</div>

<!-- Time Slider -->
<div id="time-slider-wrap" class="glass-panel">
  <div class="ts-label">Historical Playback</div>
  <div class="ts-value" id="ts-display">Live</div>
  <input type="range" id="time-slider" min="0" max="100" value="100">
  <div id="time-slider-info">Drag to replay past snapshots</div>
</div>

<!-- Comparison -->
<div id="compare-panel" class="glass-panel">
  <h5>Flight Comparison</h5>
  <div class="compare-grid">
    <div class="compare-col"><div class="cf" id="cmp-a-flight">-</div><div class="ca" id="cmp-a-airline">-</div></div>
    <div class="compare-vs">VS</div>
    <div class="compare-col"><div class="cf" id="cmp-b-flight">-</div><div class="ca" id="cmp-b-airline">-</div></div>
  </div>
  <div id="compare-rows"></div>
  <div style="margin-top:10px;text-align:center;">
    <button class="toggle-btn" onclick="cancelCompare()">Close comparison</button>
  </div>
</div>

<!-- Keyboard hints -->
<div id="kbd-hint" class="glass-panel">
  <span><kbd>F</kbd> Search</span>
  <span><kbd>Esc</kbd> Close</span>
  <span><kbd>←</kbd><kbd>→</kbd> Cycle</span>
  <span><kbd>R</kbd> Routes</span>
  <span><kbd>H</kbd> Heatmap</span>
  <span><kbd>E</kbd> Export</span>
  <span><kbd>I</kbd> Intel</span>
</div>

<!-- Airport filter -->
<div id="airport-filter-bar" class="glass-panel">
  <span class="af-label" id="af-label">DEL</span>
  <span class="af-count" id="af-count">0 flights</span>
  <button class="af-close" onclick="clearAirportFilter()">Show All</button>
</div>

<!-- Intel sidebar -->
<div id="intel-panel" class="glass-panel">
  <div class="intel-section">
    <div class="is-title">Top Airports</div>
    <div id="intel-airports"></div>
  </div>
  <div class="intel-section">
    <div class="is-title">Busiest Routes</div>
    <div id="intel-routes"></div>
  </div>
  <div class="intel-section">
    <div class="is-title">Most Delayed</div>
    <div id="intel-delays"></div>
  </div>
  <div class="intel-section">
    <div class="is-title">Weather Impact</div>
    <div id="intel-weather"></div>
  </div>
  <div class="intel-section">
    <div class="is-title">Arrival Risk (AFRI)</div>
    <div id="intel-afri"></div>
  </div>
</div>

<script>
(function() {{
try {{
// =============================================================================
//  DATA
// =============================================================================
const FLIGHTS = {flights_json};
const AIRPORTS = {airports_json};
const HISTORY = {history_json};
const WEATHER = {weather_json};
const ARRIVALS = {arrivals_json};
const DEPARTURES = {departures_json};
const AIRPORT_METRICS = {airport_metrics_json};
const ROUTE_METRICS = {route_metrics_json};
const ICON_TEMPLATES = {icon_templates_json};

// Icon factory
const _iconCache = {{}};
function makeIconUri(family, color, size) {{
  const key = family + '|' + color + '|' + size;
  if (_iconCache[key]) return _iconCache[key];
  const body = (ICON_TEMPLATES[family] || ICON_TEMPLATES['default'] || '').replace(/COLORTOK/g, color);
  const svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="' + size + '" height="' + size + '">' + body + '</svg>';
  const uri = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg);
  _iconCache[key] = uri;
  return uri;
}}

// =============================================================================
//  MAPLIBRE GL JS INIT
// =============================================================================
const SATELLITE_STYLE = {{
  version: 8,
  sources: {{
    esri_satellite: {{
      type: 'raster',
      tiles: [
        'https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}'
      ],
      tileSize: 256,
      attribution: 'Esri World Imagery'
    }}
  }},
  layers: [
    {{
      id: 'esri_satellite',
      type: 'raster',
      source: 'esri_satellite'
    }}
  ]
}};

console.log('[AVI] MapLibre init starting');

// Failsafe: remove skeleton after 10s no matter what
setTimeout(function() {{
  var sk = document.getElementById('skeleton');
  if (sk && !sk.classList.contains('hidden')) {{
    console.warn('[AVI] Failsafe: removing skeleton after 10s timeout');
    sk.classList.add('hidden');
    setTimeout(function() {{ if(sk) sk.remove(); }}, 700);
  }}
}}, 10000);

// Restore map state from previous session (survives Streamlit refresh)
const _savedState = (function() {{
  try {{
    const s = sessionStorage.getItem('avi_map_state');
    return s ? JSON.parse(s) : null;
  }} catch(e) {{ return null; }}
}})();
const _isReload = !!_savedState;

const map = new maplibregl.Map({{
  container: 'map',
  style: SATELLITE_STYLE,
  center: _savedState ? _savedState.center : [{center_lng}, {center_lat}],
  zoom: _savedState ? _savedState.zoom : {zoom_level},
  bearing: _savedState ? (_savedState.bearing || 0) : 0,
  pitch: _savedState ? (_savedState.pitch || 0) : 0,
  attributionControl: false,
}});

// Save map state before page unloads (Streamlit refresh)
window.addEventListener('beforeunload', function() {{
  try {{
    const c = map.getCenter();
    sessionStorage.setItem('avi_map_state', JSON.stringify({{
      center: [c.lng, c.lat],
      zoom: map.getZoom(),
      bearing: map.getBearing(),
      pitch: map.getPitch(),
      selectedFlight: selectedFlight ? selectedFlight.flight : null,
    }}));
  }} catch(e) {{}}
}});

// Skip skeleton on reloads — instant map show
if (_isReload) {{
  var sk = document.getElementById('skeleton');
  if (sk) sk.remove();
}}

map.on('error', function(e) {{
  console.error('[AVI] MapLibre error:', e.error ? e.error.message : e);
  var sk = document.getElementById('skeleton');
  if (sk) sk.innerHTML = '<div style="color:#ff5555;font-size:14px;padding:20px;">Map error: ' + (e.error ? e.error.message : 'Unknown') + '</div>';
}});

// =============================================================================
//  HELPERS
// =============================================================================
function nearestAirport(lat, lng) {{
  let n = null, d = Infinity;
  AIRPORTS.forEach(a => {{ const dd = Math.hypot(a.lat-lat, a.lng-lng); if(dd<d){{ d=dd; n=a; }} }});
  return n;
}}
function normalizedFlightCode(v) {{ return String(v||'').toUpperCase().replace(/\\s/g,''); }}
function fmtNum(n) {{ return n != null ? n.toLocaleString() : '-'; }}

// Sound system
const AudioCtx = window.AudioContext || window.webkitAudioContext;
let audioCtx = null;
function playTone(freq, dur, type) {{
  try {{
    if (!audioCtx) audioCtx = new AudioCtx();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = type || 'sine';
    osc.frequency.value = freq;
    gain.gain.setValueAtTime(0.08, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + dur);
    osc.connect(gain); gain.connect(audioCtx.destination);
    osc.start(); osc.stop(audioCtx.currentTime + dur);
  }} catch(e) {{}}
}}

// Great-circle arc interpolation
function greatCircleArc(lat1, lng1, lat2, lng2, nPoints) {{
  const pts = [];
  const toRad = d => d * Math.PI / 180;
  const toDeg = r => r * 180 / Math.PI;
  const f1=toRad(lat1),l1=toRad(lng1),f2=toRad(lat2),l2=toRad(lng2);
  const d = 2*Math.asin(Math.sqrt(Math.pow(Math.sin((f2-f1)/2),2)+Math.cos(f1)*Math.cos(f2)*Math.pow(Math.sin((l2-l1)/2),2)));
  if(d < 0.0001) return [[lng1,lat1],[lng2,lat2]];
  for(let i=0;i<=nPoints;i++) {{
    const frac = i/nPoints;
    const A = Math.sin((1-frac)*d)/Math.sin(d);
    const B = Math.sin(frac*d)/Math.sin(d);
    const x = A*Math.cos(f1)*Math.cos(l1)+B*Math.cos(f2)*Math.cos(l2);
    const y = A*Math.cos(f1)*Math.sin(l1)+B*Math.cos(f2)*Math.sin(l2);
    const z = A*Math.sin(f1)+B*Math.sin(f2);
    pts.push([toDeg(Math.atan2(y,x)), toDeg(Math.atan2(z,Math.sqrt(x*x+y*y)))]);
  }}
  return pts;
}}

// =============================================================================
//  MAP LOAD — add all layers/sources after style ready
// =============================================================================
map.on('load', function() {{

  // Build set of busiest hub IATA codes from AIRPORT_METRICS (top 6)
  const hubSet = new Set();
  AIRPORT_METRICS.slice(0, 6).forEach(m => hubSet.add(m.airport_iata));

  // ── AIRPORTS source ──
  const airportFeatures = AIRPORTS.map(a => ({{
    type: 'Feature',
    geometry: {{ type: 'Point', coordinates: [a.lng, a.lat] }},
    properties: {{ iata: a.iata, name: a.name, city: a.city, lat: a.lat, lng: a.lng, isHub: hubSet.has(a.iata) }}
  }}));
  map.addSource('airports', {{
    type: 'geojson',
    data: {{ type: 'FeatureCollection', features: airportFeatures }}
  }});

  // ── Airport circle layers (no image loading — renders instantly) ──
  // Hub outer radar ring 1
  map.addLayer({{
    id: 'airport-hub-ring-outer',
    type: 'circle',
    source: 'airports',
    filter: ['==', ['get', 'isHub'], true],
    paint: {{
      'circle-radius': ['interpolate', ['linear'], ['zoom'], 4, 12, 8, 22],
      'circle-color': 'transparent',
      'circle-stroke-width': 0.8,
      'circle-stroke-color': '#FFB300',
      'circle-stroke-opacity': 0.25,
    }}
  }});
  // Hub outer radar ring 2
  map.addLayer({{
    id: 'airport-hub-ring-inner',
    type: 'circle',
    source: 'airports',
    filter: ['==', ['get', 'isHub'], true],
    paint: {{
      'circle-radius': ['interpolate', ['linear'], ['zoom'], 4, 8, 8, 16],
      'circle-color': 'transparent',
      'circle-stroke-width': 1,
      'circle-stroke-color': '#FFB300',
      'circle-stroke-opacity': 0.35,
    }}
  }});
  // All airports: core dot
  map.addLayer({{
    id: 'airport-dots',
    type: 'circle',
    source: 'airports',
    paint: {{
      'circle-radius': ['interpolate', ['linear'], ['zoom'],
        4, ['case', ['get', 'isHub'], 6, 4],
        8, ['case', ['get', 'isHub'], 10, 7]
      ],
      'circle-color': '#1a1a2e',
      'circle-stroke-width': ['case', ['get', 'isHub'], 2, 1.5],
      'circle-stroke-color': ['case', ['get', 'isHub'], '#FFB300', '#aab4c6'],
      'circle-opacity': 0.92,
    }}
  }});
  // Hub center highlight dot
  map.addLayer({{
    id: 'airport-hub-center',
    type: 'circle',
    source: 'airports',
    filter: ['==', ['get', 'isHub'], true],
    paint: {{
      'circle-radius': ['interpolate', ['linear'], ['zoom'], 4, 2, 8, 3.5],
      'circle-color': '#FFB300',
      'circle-opacity': 0.9,
    }}
  }});
  // Airport text labels
  map.addLayer({{
    id: 'airport-icons',
    type: 'symbol',
    source: 'airports',
    layout: {{
      'text-field': ['get', 'iata'],
      'text-font': ['DIN Pro Bold', 'Arial Unicode MS Bold'],
      'text-size': ['interpolate', ['linear'], ['zoom'],
        4, ['case', ['get', 'isHub'], 12, 10],
        8, ['case', ['get', 'isHub'], 15, 13]
      ],
      'text-offset': [0, -1.8],
      'text-allow-overlap': true,
      'text-ignore-placement': true,
    }},
    paint: {{
      'text-color': ['case', ['get', 'isHub'], '#FFB300', '#c8d6e5'],
      'text-halo-color': '#000',
      'text-halo-width': 2,
    }}
  }});
  const airportLayerReady = Promise.resolve();

  // ── Load only the aircraft icons actually needed for the current frame ──
  // Preloading every family/color/size combination can stall the browser.
  const requiredIconKeys = new Set();
  FLIGHTS.forEach(f => {{
    const sz = f.selected ? 56 : (f.dimmed ? 30 : 36);
    requiredIconKeys.add(f.family + '|' + f.color + '|' + sz);
  }});
  let pendingImages = 0;
  const allImagesLoaded = new Promise(resolve => {{
    let resolved = false;
    function finish() {{
      if (!resolved) {{
        resolved = true;
        resolve();
      }}
    }}
    function checkDone() {{ if (pendingImages <= 0) finish(); }}
    requiredIconKeys.forEach(key => {{
      if (map.hasImage(key)) return;
      const parts = key.split('|');
      const fam = parts[0] || 'default';
      const col = parts[1] || '#00E5FF';
      const sz = Number(parts[2] || 36);
      pendingImages++;
      const img = new Image(sz, sz);
      img.onload = function() {{
        if (!map.hasImage(key)) map.addImage(key, img);
        pendingImages--;
        checkDone();
      }};
      img.onerror = function() {{
        pendingImages--;
        checkDone();
      }};
      img.src = makeIconUri(fam, col, sz);
    }});
    if (pendingImages === 0) finish();
    setTimeout(finish, 1800);
  }});

  // ── Build FLIGHTS GeoJSON ──
  let totalAnomalies = 0;
  const flightFeatures = [];
  const heatmapFeatures = [];

  FLIGHTS.forEach((f, idx) => {{
    const sz = f.selected ? 56 : (f.dimmed ? 30 : 36);
    const iconKey = f.family + '|' + f.color + '|' + sz;

    flightFeatures.push({{
      type: 'Feature',
      geometry: {{ type: 'Point', coordinates: [f.lng, f.lat] }},
      properties: {{
        idx: idx,
        flight: f.flight,
        iconKey: iconKey,
        heading: f.heading,
        selected: f.selected,
        dimmed: f.dimmed,
        opacity: f.dimmed ? 0.55 : 1.0,
        iconSize: f.selected ? 0.9 : (f.dimmed ? 0.5 : 0.6),
      }}
    }});





    // Heatmap point
    heatmapFeatures.push({{
      type: 'Feature',
      geometry: {{ type: 'Point', coordinates: [f.lng, f.lat] }},
      properties: {{ weight: 1 }}
    }});

    if (f.anomalies && f.anomalies.length > 0) totalAnomalies += f.anomalies.length;
  }});

  document.getElementById('stat-anomalies').textContent = totalAnomalies;

  // ── Add sources ──
  map.addSource('flights', {{
    type: 'geojson',
    data: {{ type: 'FeatureCollection', features: flightFeatures }}
  }});

  // Fallback flight dots render immediately so the map never looks empty
  // while custom SVG aircraft icons are decoding.
  map.addLayer({{
    id: 'flight-dots-glow',
    type: 'circle',
    source: 'flights',
    paint: {{
      'circle-radius': ['case', ['get', 'selected'], 12, 7],
      'circle-color': '#00E5FF',
      'circle-opacity': ['case', ['get', 'dimmed'], 0.10, 0.16],
      'circle-blur': 0.9,
    }}
  }});
  map.addLayer({{
    id: 'flight-dots',
    type: 'circle',
    source: 'flights',
    paint: {{
      'circle-radius': ['case', ['get', 'selected'], 5.5, 3.8],
      'circle-color': ['case', ['get', 'selected'], '#FF1744', '#00E5FF'],
      'circle-stroke-width': 1,
      'circle-stroke-color': '#051018',
      'circle-opacity': ['get', 'opacity'],
    }}
  }});


  map.addSource('heatmap-data', {{
    type: 'geojson',
    data: {{ type: 'FeatureCollection', features: heatmapFeatures }}
  }});

  // ── Weather source ──
  const catColors = {{ VFR: '#2ecc71', MVFR: '#8fc8ff', IFR: '#ff9d35', LIFR: '#ff5a4f' }};
  const weatherFeatures = WEATHER.map(w => ({{
    type: 'Feature',
    geometry: {{ type: 'Point', coordinates: [w.lng, w.lat] }},
    properties: {{ station: w.station, category: w.category, color: catColors[w.category] || '#888' }}
  }}));
  map.addSource('weather', {{
    type: 'geojson',
    data: {{ type: 'FeatureCollection', features: weatherFeatures }}
  }});





  // ── Route arc source (for selected flight path) ──
  map.addSource('route-arc', {{
    type: 'geojson',
    data: {{ type: 'FeatureCollection', features: [] }}
  }});
  map.addLayer({{
    id: 'route-arc-line',
    type: 'line',
    source: 'route-arc',
    paint: {{
      'line-color': '#FF1744',
      'line-width': 2.5,
      'line-dasharray': [2, 4],
      'line-opacity': 0.85,
    }},
    layout: {{ 'line-cap': 'round', 'line-join': 'round' }}
  }});

  // ── Flight trail source (actual position history) ──
  map.addSource('flight-trail', {{
    type: 'geojson',
    data: {{ type: 'FeatureCollection', features: [] }}
  }});
  map.addLayer({{
    id: 'flight-trail-glow',
    type: 'line',
    source: 'flight-trail',
    paint: {{
      'line-color': '#00E5FF',
      'line-width': 6,
      'line-opacity': 0.15,
      'line-blur': 4,
    }},
    layout: {{ 'line-cap': 'round', 'line-join': 'round' }}
  }});
  map.addLayer({{
    id: 'flight-trail-line',
    type: 'line',
    source: 'flight-trail',
    paint: {{
      'line-color': '#00E5FF',
      'line-width': 2.5,
      'line-opacity': 0.85,
      'line-dasharray': [1, 2],
    }},
    layout: {{ 'line-cap': 'round', 'line-join': 'round' }}
  }});
  map.addLayer({{
    id: 'flight-trail-dots',
    type: 'circle',
    source: 'flight-trail',
    filter: ['==', '$type', 'Point'],
    paint: {{
      'circle-radius': 3,
      'circle-color': '#00E5FF',
      'circle-opacity': ['get', 'opacity'],
      'circle-stroke-width': 1,
      'circle-stroke-color': '#00E5FF',
      'circle-stroke-opacity': 0.3,
    }}
  }});

  // ── Heatmap layer (hidden by default) ──
  map.addLayer({{
    id: 'heatmap-layer',
    type: 'heatmap',
    source: 'heatmap-data',
    paint: {{
      'heatmap-weight': 1,
      'heatmap-intensity': ['interpolate', ['linear'], ['zoom'], 0, 1, 9, 3],
      'heatmap-color': [
        'interpolate', ['linear'], ['heatmap-density'],
        0, 'rgba(0,0,0,0)',
        0.2, 'rgba(255,100,60,0.15)',
        0.4, 'rgba(255,100,60,0.3)',
        0.6, 'rgba(255,80,40,0.5)',
        0.8, 'rgba(255,60,30,0.7)',
        1,   'rgba(255,40,20,0.85)'
      ],
      'heatmap-radius': ['interpolate', ['linear'], ['zoom'], 0, 15, 9, 40],
      'heatmap-opacity': 0.7,
    }},
    layout: {{ 'visibility': 'none' }}
  }});

  // ── Weather layer (hidden by default) ──
  map.addLayer({{
    id: 'weather-circles',
    type: 'circle',
    source: 'weather',
    paint: {{
      'circle-radius': ['interpolate', ['linear'], ['zoom'], 4, 8, 8, 30],
      'circle-color': ['get', 'color'],
      'circle-opacity': 0.18,
      'circle-stroke-color': ['get', 'color'],
      'circle-stroke-width': 1,
      'circle-stroke-opacity': 0.45,
    }},
    layout: {{ 'visibility': 'none' }}
  }});
  map.addLayer({{
    id: 'weather-labels',
    type: 'symbol',
    source: 'weather',
    layout: {{
      'text-field': ['concat', ['get', 'station'], ' ', ['get', 'category']],
      'text-font': ['DIN Pro Medium', 'Arial Unicode MS Regular'],
      'text-size': 10,
      'text-offset': [0, -1.5],
      'visibility': 'none',
    }},
    paint: {{
      'text-color': ['get', 'color'],
      'text-halo-color': '#000',
      'text-halo-width': 1,
    }}
  }});

  function hideSkeleton() {{
    var sk = document.getElementById('skeleton');
    if (sk) {{
      sk.classList.add('hidden');
      setTimeout(function() {{ if(sk.parentNode) sk.remove(); }}, 700);
    }}
  }}

  // Remove skeleton as soon as the base map and fallback flight dots are ready.
  setTimeout(hideSkeleton, 250);

  // ── Flight icons layer ──
  allImagesLoaded.then(function() {{
    try {{
      map.addLayer({{
        id: 'flight-icons',
        type: 'symbol',
        source: 'flights',
        layout: {{
          'icon-image': ['get', 'iconKey'],
          'icon-size': ['get', 'iconSize'],
          'icon-rotate': ['get', 'heading'],
          'icon-rotation-alignment': 'map',
          'icon-allow-overlap': true,
          'icon-ignore-placement': true,
          'text-field': ['get', 'flight'],
          'text-font': ['DIN Pro Bold', 'Arial Unicode MS Bold'],
          'text-size': 10,
          'text-offset': [0, -2.0],
          'text-allow-overlap': false,
        }},
        paint: {{
          'icon-opacity': ['get', 'opacity'],
          'text-color': '#00E5FF',
          'text-halo-color': '#000',
          'text-halo-width': 2,
          'text-opacity': ['case', ['get', 'dimmed'], 0.3, 1.0],
        }}
      }});
      map.setLayoutProperty('flight-dots-glow', 'visibility', 'none');
      map.setLayoutProperty('flight-dots', 'visibility', 'none');
    }} catch (e) {{
      console.warn('[AVI] Icon layer failed, keeping fallback dots:', e);
    }}

    hideSkeleton();

    // Restore previously selected flight on reload, or use server selection
    if (_savedState && _savedState.selectedFlight) {{
      showPopupByFlight(_savedState.selectedFlight);
    }} else {{
      {selected_js}
    }}

    // =========================================================================
    //  DEAD RECKONING ANIMATION — smooth aircraft movement between API refreshes
    // =========================================================================
    let _animRunning = true;
    let _lastAnimTime = performance.now();

    // Snapshot base positions for dead reckoning
    FLIGHTS.forEach(f => {{
      f._baseLat = f.lat;
      f._baseLng = f.lng;
    }});

    function animatePlanes(now) {{
      if (!_animRunning) return;
      const dtSec = (now - _lastAnimTime) / 1000;
      _lastAnimTime = now;

      // Only update if enough time passed (~60ms → ~16fps is fine for smooth)
      if (dtSec <= 0 || dtSec > 2) {{
        requestAnimationFrame(animatePlanes);
        return;
      }}

      const src = map.getSource('flights');
      if (!src) {{ requestAnimationFrame(animatePlanes); return; }}

      let needsUpdate = false;

      FLIGHTS.forEach(f => {{
        // Skip stationary / no-speed / on-ground flights
        if (!f.speed || f.speed < 50) return;
        // speed is in knots → km/s = kts * 1.852 / 3600
        const kmPerSec = f.speed * 1.852 / 3600;
        const distKm = kmPerSec * dtSec;
        const hdgRad = f.heading * Math.PI / 180;

        // lat/lng displacement
        const dLat = (distKm / 111.32) * Math.cos(hdgRad);
        const dLng = (distKm / (111.32 * Math.cos(f.lat * Math.PI / 180))) * Math.sin(hdgRad);

        f.lat += dLat;
        f.lng += dLng;
        needsUpdate = true;
      }});

      if (needsUpdate) {{
        // Rebuild GeoJSON with animated positions
        const tc = selectedFlight ? normalizedFlightCode(selectedFlight.flight) : null;
        const features = FLIGHTS.map((f, idx) => {{
          const isSel = tc && normalizedFlightCode(f.flight) === tc;
          const isDim = tc && !isSel;
          const sz = isSel ? 56 : (isDim ? 30 : 36);
          const col = isSel ? '#FF1744' : (isDim ? '#4DD0E1' : f.color);
          const iconKey = f.family + '|' + col + '|' + sz;
          return {{
            type: 'Feature',
            geometry: {{ type: 'Point', coordinates: [f.lng, f.lat] }},
            properties: {{
              idx: idx, flight: f.flight, iconKey: iconKey,
              heading: f.heading, selected: isSel, dimmed: isDim,
              opacity: isDim ? 0.55 : 1.0,
              iconSize: isSel ? 0.9 : (isDim ? 0.5 : 0.6),
              show: true,
            }}
          }};
        }});
        src.setData({{ type: 'FeatureCollection', features: features }});
      }}

      requestAnimationFrame(animatePlanes);
    }}

    requestAnimationFrame(animatePlanes);

    // Stop animation before page unloads to avoid errors
    window.addEventListener('beforeunload', function() {{ _animRunning = false; }});

    // =========================================================================
    //  GEOLOCATION — My Location + Overhead flights
    // =========================================================================
    let _myLocMarker = null;
    let _myLocCoords = null;
    let _watchId = null;
    let _overheadInterval = null;

    function _createLocMarker() {{
      const el = document.createElement('div');
      el.style.cssText = 'position:relative;width:20px;height:20px;';
      const dot = document.createElement('div');
      dot.style.cssText = 'width:14px;height:14px;border-radius:50%;background:#4285F4;border:2.5px solid #fff;position:absolute;top:3px;left:3px;z-index:2;box-shadow:0 0 8px rgba(66,133,244,0.6);';
      el.appendChild(dot);
      const ring = document.createElement('div');
      ring.style.cssText = 'width:14px;height:14px;border-radius:50%;background:rgba(66,133,244,0.35);position:absolute;top:3px;left:3px;z-index:1;animation:locPulse 2s ease-out infinite;';
      el.appendChild(ring);
      return el;
    }}

    function _updateOverhead() {{
      if (!_myLocCoords) return;
      const list = document.getElementById('oh-list');
      const countEl = document.getElementById('oh-count');
      const R = 0.72; // ~80km in degrees
      const nearby = [];
      FLIGHTS.forEach(f => {{
        const dLat = f.lat - _myLocCoords[1];
        const dLng = f.lng - _myLocCoords[0];
        const dist = Math.sqrt(dLat*dLat + dLng*dLng);
        if (dist <= R && f.speed > 50) {{
          nearby.push({{ f: f, dist: Math.round(dist * 111.32) }});
        }}
      }});
      nearby.sort((a,b) => a.dist - b.dist);
      countEl.textContent = nearby.length;
      if (nearby.length === 0) {{
        list.innerHTML = '<div class="oh-empty">No aircraft within 80 km</div>';
      }} else {{
        list.innerHTML = nearby.slice(0, 8).map(n => {{
          const f = n.f;
          const alt = f.alt_ft ? f.alt_ft.toLocaleString() + ' ft' : '-';
          return '<div class="oh-item" onclick="showPopupByFlight(\\''+f.flight+'\\')">'+
            '<span class="oh-flight">'+f.flight+'</span>'+
            '<span class="oh-detail">'+f.route+'</span>'+
            '<span class="oh-alt">'+alt+' · '+n.dist+' km</span>'+
          '</div>';
        }}).join('');
      }}
    }}

    window.locateMe = function() {{
      if (!navigator.geolocation) {{
        alert('Geolocation not supported by your browser');
        return;
      }}
      const btn = document.getElementById('btn-locate');
      btn.classList.toggle('active');
      if (_watchId !== null) {{
        navigator.geolocation.clearWatch(_watchId);
        _watchId = null;
        if (_myLocMarker) {{ _myLocMarker.remove(); _myLocMarker = null; }}
        _myLocCoords = null;
        document.getElementById('my-loc-panel').style.display = 'none';
        if (_overheadInterval) {{ clearInterval(_overheadInterval); _overheadInterval = null; }}
        return;
      }}
      btn.textContent = 'Locating...';
      _watchId = navigator.geolocation.watchPosition(
        function(pos) {{
          _myLocCoords = [pos.coords.longitude, pos.coords.latitude];
          if (!_myLocMarker) {{
            _myLocMarker = new maplibregl.Marker({{ element: _createLocMarker() }})
              .setLngLat(_myLocCoords).addTo(map);
            map.flyTo({{ center: _myLocCoords, zoom: 9, duration: 1500 }});
            document.getElementById('my-loc-panel').style.display = 'block';
            _updateOverhead();
            _overheadInterval = setInterval(_updateOverhead, 5000);
          }} else {{
            _myLocMarker.setLngLat(_myLocCoords);
          }}
          btn.textContent = '📍 My Location';
        }},
        function(err) {{
          btn.textContent = '📍 My Location';
          btn.classList.remove('active');
          _watchId = null;
          if (err.code === 1) alert('Location access denied. Enable it in browser settings.');
          else alert('Could not get location: ' + err.message);
        }},
        {{ enableHighAccuracy: true, maximumAge: 30000 }}
      );
    }};
  }});

  // =============================================================================
  //  LAYER TOGGLES
  // =============================================================================
  const layerState = {{ routes: true, heatmap: false, weather: false, board: false, timeline: false, intel: true }};

  window.toggleLayer = function(layer) {{
    layerState[layer] = !layerState[layer];
    const btn = document.getElementById('btn-' + layer);
    if (btn) btn.classList.toggle('active', layerState[layer]);


    if (layer === 'heatmap') {{
      map.setLayoutProperty('heatmap-layer', 'visibility', layerState.heatmap ? 'visible' : 'none');
    }}
    if (layer === 'weather') {{
      map.setLayoutProperty('weather-circles', 'visibility', layerState.weather ? 'visible' : 'none');
      map.setLayoutProperty('weather-labels', 'visibility', layerState.weather ? 'visible' : 'none');
    }}
    if (layer === 'board') {{
      document.getElementById('flight-board').style.display = layerState.board ? 'block' : 'none';
      if (layerState.board) showBoard('arr');
    }}
    if (layer === 'timeline') {{
      document.getElementById('time-slider-wrap').style.display = layerState.timeline ? 'block' : 'none';
      if (layerState.timeline) initTimeline();
    }}
    if (layer === 'intel') {{
      document.getElementById('intel-panel').style.display = layerState.intel ? 'flex' : 'none';
    }}
  }};

  // =============================================================================
  //  SELECTION & POPUP
  // =============================================================================
  let selectedFlight = null;
  let compareFlightA = null;
  let compareFlightB = null;
  let currentFlightIndex = -1;
  let compareMode = false;

  function drawSparkline(altHistory) {{
    const canvas = document.getElementById('sparkline-canvas');
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.offsetWidth * 2;
    const h = canvas.height = 100;
    ctx.clearRect(0,0,w,h);
    if (!altHistory || altHistory.length < 2) {{
      ctx.fillStyle = 'rgba(255,255,255,0.2)'; ctx.font = '18px Arial';
      ctx.fillText('No history', 10, h/2+6); return;
    }}
    const mn = Math.min(...altHistory), mx = Math.max(...altHistory);
    const range = mx - mn || 1; const pad = 4;
    ctx.beginPath(); ctx.strokeStyle = '#bcff5c'; ctx.lineWidth = 2;
    altHistory.forEach((v,i) => {{
      const x = pad + (i/(altHistory.length-1)) * (w - 2*pad);
      const y = h - pad - ((v-mn)/range) * (h - 2*pad);
      i === 0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    }});
    ctx.stroke();
    ctx.lineTo(pad + (w - 2*pad), h - pad);
    ctx.lineTo(pad, h - pad); ctx.closePath();
    ctx.fillStyle = 'rgba(188,255,92,0.08)'; ctx.fill();
    ctx.fillStyle = 'rgba(255,255,255,0.4)'; ctx.font = '16px Arial';
    ctx.fillText(fmtNum(mx)+' ft', pad, 16);
    ctx.fillText(fmtNum(mn)+' ft', pad, h - 2);
  }}

  function showPopup(f) {{
    const near = nearestAirport(f.lat, f.lng);
    document.getElementById('popup-title').textContent = (f.flight||'Unknown');
    document.getElementById('popup-subtitle').textContent = (f.airline||'Unknown')+' | '+(f.aircraft||'Unknown');
    document.getElementById('popup-airline').textContent = f.airline||'Unknown';
    document.getElementById('popup-aircraft-chip').textContent = f.aircraft||'Unknown';
    document.getElementById('popup-status-chip').textContent = f.status||'Unknown';
    document.getElementById('popup-dep-code').textContent = f.dep||'N/A';
    document.getElementById('popup-arr-code').textContent = f.arr||'N/A';
    document.getElementById('popup-dep-city').textContent = near ? near.city+' nearest' : 'Departure';
    document.getElementById('popup-arr-city').textContent = 'Destination';
    document.getElementById('popup-altitude-stat').textContent = fmtNum(f.alt_ft)+' ft';
    document.getElementById('popup-speed-stat').textContent = f.speed+' kts';
    drawSparkline(f.alt_history || []);
    // Fuel
    const fc = document.getElementById('fuel-card');
    if (f.fuel && f.fuel.co2_kg) {{
      fc.style.display = 'block';
      document.getElementById('fuel-burn').textContent = fmtNum(Math.round(f.fuel.fuel_kg))+' kg';
      document.getElementById('fuel-co2').textContent = fmtNum(Math.round(f.fuel.co2_kg))+' kg';
      document.getElementById('fuel-pax').textContent = f.fuel.co2_per_pax ? f.fuel.co2_per_pax+' kg' : '-';
      document.getElementById('fuel-eff').textContent = f.fuel.efficiency || '-';
    }} else {{ fc.style.display = 'none'; }}
    // Stale badge
    const panel = document.getElementById('popup');
    const existingStale = panel.querySelector('.stale-badge');
    if (existingStale) existingStale.remove();
    if (f.stale_min >= 3) {{
      const sb = document.createElement('span');
      sb.className = 'stale-badge ' + (f.stale_min >= 10 ? 'lost' : f.stale_min >= 5 ? 'warn' : 'lag');
      sb.textContent = f.stale_min >= 10 ? 'LOST CONTACT ' + Math.round(f.stale_min) + 'm' : 'STALE ' + Math.round(f.stale_min) + 'm';
      const statusEl = document.getElementById('p-vstatus');
      if (statusEl) statusEl.parentNode.insertBefore(sb, statusEl.nextSibling);
    }}
    // Anomalies
    const ab = document.getElementById('anomaly-bar'); ab.innerHTML = '';
    if (f.anomalies && f.anomalies.length) {{
      f.anomalies.forEach(a => {{
        const chip = document.createElement('span');
        chip.className = 'anomaly-chip ' + a.severity;
        chip.textContent = (a.severity==='alert'?'!! ':a.severity==='warning'?'! ':'') + a.desc;
        ab.appendChild(chip);
      }});
      playTone(880, 0.15, 'triangle');
    }}
    // Delay
    const dc = document.getElementById('delay-card');
    const risk = (f.delay_risk || 'Low').toLowerCase();
    if (f.delay_min > 0 || risk !== 'low') {{
      dc.style.display = 'block'; dc.className = 'risk-' + risk;
      document.getElementById('dc-badge').textContent = f.delay_risk || 'Low';
      document.getElementById('dc-delay').textContent = '+' + (f.delay_min||0) + ' min';
      document.getElementById('dc-eta').textContent = f.delay_eta || '--:-- UTC';
      const reasonsEl = document.getElementById('dc-reasons'); reasonsEl.innerHTML = '';
      if (f.delay_reason) {{
        f.delay_reason.split(',').map(s => s.trim()).filter(Boolean).forEach(r => {{
          const tag = document.createElement('span'); tag.className = 'dc-reason-tag'; tag.textContent = r;
          reasonsEl.appendChild(tag);
        }});
      }}
      if (f.weather_sev && f.weather_sev !== 'Low') {{
        const wt = document.createElement('span'); wt.className = 'dc-reason-tag';
        wt.style.borderColor = f.weather_sev==='Severe' ? 'rgba(255,59,48,0.3)' : 'rgba(255,165,0,0.3)';
        wt.style.color = f.weather_sev==='Severe' ? '#ff7b73' : '#ffc97a';
        wt.textContent = 'weather: ' + f.weather_sev.toLowerCase(); reasonsEl.appendChild(wt);
      }}
    }} else {{ dc.style.display = 'none'; }}
    // AFRI card
    const ac = document.getElementById('afri-card');
    const afriScore = f.afri || 0;
    const afriLevel = (f.afri_level || 'Normal').toLowerCase();
    if (afriScore > 0) {{
      ac.style.display = 'block';
      ac.className = 'afri-' + afriLevel;
      document.getElementById('afri-badge').textContent = f.afri_level || 'Normal';
      document.getElementById('afri-number').textContent = afriScore;
      document.getElementById('afri-bar').style.width = afriScore + '%';
      const driversEl = document.getElementById('afri-drivers'); driversEl.innerHTML = '';
      if (f.afri_drivers) {{
        f.afri_drivers.split(',').map(s => s.trim()).filter(Boolean).forEach(d => {{
          const tag = document.createElement('span'); tag.className = 'afri-driver-tag'; tag.textContent = d;
          driversEl.appendChild(tag);
        }});
      }}
    }} else {{ ac.style.display = 'none'; }}
    const rows = [
      ['Route', f.route], ['Registration', f.reg],
      ['Position', f.lat.toFixed(2)+', '+f.lng.toFixed(2)],
      ['Vertical', f.vstatus], ['Status', f.status],
      ['Distance', f.fuel && f.fuel.distance_km ? f.fuel.distance_km+' km remaining' : '-'],
    ].filter(r => r[1] && String(r[1]).trim());
    document.getElementById('popup-body').innerHTML = rows.map(r => '<tr><td>'+r[0]+'</td><td>'+r[1]+'</td></tr>').join('');
    document.getElementById('popup').style.display = 'block';
  }}

  function selectFlight(flight, flyTo) {{
    selectedFlight = flight;
    showPopup(flight);
    currentFlightIndex = FLIGHTS.indexOf(flight);
    document.getElementById('stat-selected').textContent = flight ? (flight.flight || '-') : '-';
    if (flyTo) {{
      map.flyTo({{ center: [flight.lng, flight.lat], zoom: 7, duration: 1200 }});
      playTone(660, 0.1, 'sine');
    }}
    // Draw great-circle arc from departure to arrival
    drawRouteArc(flight);
    // Draw actual position trail history
    drawFlightTrail(flight);
    // Update flight icon highlights — rebuild features with new selection state
    updateFlightSelection(flight);
  }}

  function updateFlightSelection(selFlight) {{
    const tc = selFlight ? normalizedFlightCode(selFlight.flight) : null;
    const updatedFeatures = FLIGHTS.map((f, idx) => {{
      const isSel = tc && normalizedFlightCode(f.flight) === tc;
      const isDim = tc && !isSel;
      const sz = isSel ? 56 : (isDim ? 30 : 36);
      const col = isSel ? '#FF1744' : (isDim ? '#4DD0E1' : f.color);
      const iconKey = f.family + '|' + col + '|' + sz;
      return {{
        type: 'Feature',
        geometry: {{ type: 'Point', coordinates: [f.lng, f.lat] }},
        properties: {{
          idx: idx, flight: f.flight, iconKey: iconKey,
          heading: f.heading, selected: isSel, dimmed: isDim,
          opacity: isDim ? 0.55 : 1.0,
          iconSize: isSel ? 0.9 : (isDim ? 0.5 : 0.6),
          show: true,
        }}
      }};
    }});
    const src = map.getSource('flights');
    if (src) src.setData({{ type: 'FeatureCollection', features: updatedFeatures }});
  }}

  function drawRouteArc(f) {{
    const src = map.getSource('route-arc');
    if (!src) return;
    if (f && f.dep_lat != null && f.dep_lng != null && f.dest_lat != null && f.dest_lng != null) {{
      const pts = greatCircleArc(f.dep_lat, f.dep_lng, f.dest_lat, f.dest_lng, 64);
      src.setData({{
        type: 'FeatureCollection',
        features: [{{
          type: 'Feature',
          geometry: {{ type: 'LineString', coordinates: pts }},
          properties: {{}}
        }}]
      }});
    }} else {{
      src.setData({{ type: 'FeatureCollection', features: [] }});
    }}
  }}

  function drawFlightTrail(f) {{
    const src = map.getSource('flight-trail');
    if (!src) return;
    if (!f || !f.trail || f.trail.length < 2) {{
      src.setData({{ type: 'FeatureCollection', features: [] }});
      return;
    }}
    const coords = f.trail.map(p => [p.lng, p.lat]);
    // Add current position as final point
    coords.push([f.lng, f.lat]);
    const features = [{{
      type: 'Feature',
      geometry: {{ type: 'LineString', coordinates: coords }},
      properties: {{}}
    }}];
    // Position dots with fading opacity (older = more transparent)
    const total = coords.length;
    coords.forEach((c, i) => {{
      features.push({{
        type: 'Feature',
        geometry: {{ type: 'Point', coordinates: c }},
        properties: {{ opacity: 0.3 + 0.7 * (i / (total - 1)) }}
      }});
    }});
    src.setData({{ type: 'FeatureCollection', features: features }});
  }}

  function clearFlightTrail() {{
    const src = map.getSource('flight-trail');
    if (src) src.setData({{ type: 'FeatureCollection', features: [] }});
  }}

  function clearRouteArc() {{
    const src = map.getSource('route-arc');
    if (src) src.setData({{ type: 'FeatureCollection', features: [] }});
  }}

  window.refocusSelectedFlight = function() {{
    if (!selectedFlight) return;
    map.flyTo({{ center: [selectedFlight.lng, selectedFlight.lat], zoom: 7, duration: 1000 }});
  }};

  window.showPopupByFlight = function(code) {{
    const t = normalizedFlightCode(code);
    for (const f of FLIGHTS) {{
      const n = normalizedFlightCode(f.flight);
      if (n === t || n.includes(t)) {{ selectFlight(f, true); return; }}
    }}
  }};

  window.closePopup = function() {{
    document.getElementById('popup').style.display = 'none';
    selectedFlight = null;
    updateFlightSelection(null);
    clearRouteArc();
    clearFlightTrail();
    document.getElementById('stat-selected').textContent = '-';
  }};

  // ── Click handler (registered after both flight + airport layers are ready) ──
  Promise.all([allImagesLoaded, airportLayerReady]).then(function() {{
    map.on('click', 'flight-icons', function(e) {{
      const idx = e.features[0].properties.idx;
      const f = FLIGHTS[idx];
      if (f) {{
        if (compareMode) {{
          compareFlightB = f;
          renderComparison();
          compareMode = false;
        }} else {{
          selectFlight(f, false);
        }}
      }}
    }});

    map.on('click', 'airport-icons', function(e) {{
      const props = e.features[0].properties;
      filterByAirport(props.iata);
    }});

    map.on('click', function(e) {{
      const layers = [];
      if (map.getLayer('flight-icons')) layers.push('flight-icons');
      if (map.getLayer('airport-icons')) layers.push('airport-icons');
      const features = layers.length ? map.queryRenderedFeatures(e.point, {{ layers: layers }}) : [];
      if (features.length === 0) {{
        closePopup();
        clearAirportFilter();
      }}
    }});

    // Cursor changes
    map.on('mouseenter', 'flight-icons', function() {{ map.getCanvas().style.cursor = 'pointer'; }});
    map.on('mouseleave', 'flight-icons', function() {{ map.getCanvas().style.cursor = ''; }});
    map.on('mouseenter', 'airport-icons', function() {{ map.getCanvas().style.cursor = 'pointer'; }});
    map.on('mouseleave', 'airport-icons', function() {{ map.getCanvas().style.cursor = ''; }});
  }});

  // =============================================================================
  //  COMPARISON
  // =============================================================================
  window.startCompare = function() {{
    if (!selectedFlight) return;
    compareFlightA = selectedFlight;
    compareFlightB = null;
    compareMode = true;
    document.getElementById('cmp-a-flight').textContent = compareFlightA.flight;
    document.getElementById('cmp-a-airline').textContent = compareFlightA.airline;
    document.getElementById('cmp-b-flight').textContent = 'Click another...';
    document.getElementById('cmp-b-airline').textContent = '';
    document.getElementById('compare-rows').innerHTML = '';
    document.getElementById('compare-panel').style.display = 'block';
    closePopup();
  }};

  function renderComparison() {{
    if (!compareFlightA || !compareFlightB) return;
    const a = compareFlightA, b = compareFlightB;
    document.getElementById('cmp-a-flight').textContent = a.flight;
    document.getElementById('cmp-a-airline').textContent = a.airline;
    document.getElementById('cmp-b-flight').textContent = b.flight;
    document.getElementById('cmp-b-airline').textContent = b.airline;
    const metrics = [
      ['Altitude', fmtNum(a.alt_ft)+' ft', fmtNum(b.alt_ft)+' ft'],
      ['Speed', a.speed+' kts', b.speed+' kts'],
      ['Route', a.route, b.route],
      ['Aircraft', a.aircraft, b.aircraft],
      ['Status', a.status, b.status],
      ['CO2', a.fuel&&a.fuel.co2_kg?fmtNum(Math.round(a.fuel.co2_kg))+' kg':'-', b.fuel&&b.fuel.co2_kg?fmtNum(Math.round(b.fuel.co2_kg))+' kg':'-'],
      ['Efficiency', a.fuel?a.fuel.efficiency||'-':'-', b.fuel?b.fuel.efficiency||'-':'-'],
    ];
    document.getElementById('compare-rows').innerHTML = metrics.map(m =>
      '<div class="compare-row"><span class="cr-a">'+m[1]+'</span><span class="cr-label">'+m[0]+'</span><span class="cr-b">'+m[2]+'</span></div>'
    ).join('');
    playTone(520, 0.1, 'sine');
  }}

  window.cancelCompare = function() {{
    document.getElementById('compare-panel').style.display = 'none';
    compareFlightA = null; compareFlightB = null; compareMode = false;
  }};

  // =============================================================================
  //  BOARD
  // =============================================================================
  let boardMode = 'arr';
  window.showBoard = function(mode) {{
    boardMode = mode;
    document.querySelectorAll('#flight-board .board-tabs button').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('#flight-board .board-tabs button').forEach(b => {{
      if ((mode==='arr' && b.textContent==='Arrivals') || (mode==='dep' && b.textContent==='Departures')) b.classList.add('active');
    }});
    const data = mode === 'arr' ? ARRIVALS : DEPARTURES;
    const body = document.getElementById('board-body');
    if (!data.length) {{ body.innerHTML = '<div style="padding:14px;color:rgba(255,255,255,0.4);text-align:center;">No schedule data</div>'; return; }}
    body.innerHTML = data.slice(0, 30).map(r => {{
      const st = String(r.status||'scheduled').toLowerCase();
      const stClass = st.includes('active')||st.includes('en-route')?'st-active':st.includes('delay')?'st-delayed':st.includes('land')?'st-landed':'st-scheduled';
      const timeVal = mode==='arr' ? (r.arr_time||'-') : (r.dep_time||'-');
      const routeVal = mode==='arr' ? (r.dep_iata||'-') : (r.arr_iata||'-');
      return '<div class="board-row">' +
        '<div class="fl">'+(r.flight_iata||'-')+'</div>' +
        '<div class="route">'+(mode==='arr'?'from ':'to ')+routeVal+'</div>' +
        '<div class="time">'+timeVal+'</div>' +
        '<div class="st '+stClass+'">'+(r.status||'Sched')+'</div></div>';
    }}).join('');
  }};

  // =============================================================================
  //  TIMELINE
  // =============================================================================
  function initTimeline() {{
    const slider = document.getElementById('time-slider');
    if (!HISTORY.length) {{
      document.getElementById('ts-display').textContent = 'No history';
      slider.disabled = true; return;
    }}
    slider.min = 0; slider.max = HISTORY.length - 1; slider.value = HISTORY.length - 1;
    updateTimelineDisplay(HISTORY.length - 1);
  }}
  function updateTimelineDisplay(idx) {{
    const h = HISTORY[idx]; if (!h) return;
    const d = new Date(h.ts * 1000);
    const label = d.toUTCString().slice(17, 25) + ' UTC - ' + h.flight_count + ' flights';
    document.getElementById('ts-display').textContent = idx >= HISTORY.length - 1 ? 'Live' : label;
    document.getElementById('time-slider-info').textContent =
      idx >= HISTORY.length - 1 ? 'Showing live data' :
      'Avg alt: ' + fmtNum(Math.round(h.avg_alt||0)) + ' ft | Avg spd: ' + (h.avg_spd||0).toFixed(1) + ' kts';
  }}
  document.getElementById('time-slider').addEventListener('input', function() {{
    updateTimelineDisplay(parseInt(this.value));
  }});

  // =============================================================================
  //  INTEL SIDEBAR
  // =============================================================================
  function renderIntelPanel() {{
    const apEl = document.getElementById('intel-airports');
    if (AIRPORT_METRICS.length) {{
      apEl.innerHTML = AIRPORT_METRICS.slice(0, 5).map(a => {{
        const lv = (a.congestion_level||'Low').toLowerCase();
        return '<div class="intel-item" onclick="filterByAirport(\\'' + a.airport_iata + '\\')">' +
          '<span class="ii-code">' + a.airport_iata + '</span>' +
          '<span class="ii-detail">' + a.nearby_count + ' flt &middot; ' + a.congestion_score + '</span>' +
          '<span class="intel-badge sev-' + lv + '">' + a.congestion_level + '</span></div>';
      }}).join('');
    }} else {{ apEl.innerHTML = '<div style="color:rgba(255,255,255,0.35);font-size:10px;">No data</div>'; }}

    const rtEl = document.getElementById('intel-routes');
    if (ROUTE_METRICS.length) {{
      rtEl.innerHTML = ROUTE_METRICS.slice(0, 5).map(r => {{
        return '<div class="intel-item" onclick="highlightRoute(\\'' + r.dep_iata + '\\',\\'' + r.arr_iata + '\\')">' +
          '<span class="ii-code">' + r.route_key + '</span>' +
          '<span class="ii-detail">' + r.flight_count + ' flt &middot; ' + r.congestion_score + '</span></div>';
      }}).join('');
    }} else {{ rtEl.innerHTML = '<div style="color:rgba(255,255,255,0.35);font-size:10px;">No data</div>'; }}

    const dlEl = document.getElementById('intel-delays');
    const delayed = [...FLIGHTS].filter(f => f.delay_min > 0).sort((a,b) => b.delay_min - a.delay_min).slice(0,5);
    if (delayed.length) {{
      dlEl.innerHTML = delayed.map(f => {{
        const lv = (f.delay_risk||'Low').toLowerCase();
        return '<div class="intel-item" onclick="showPopupByFlight(\\'' + f.flight + '\\')">' +
          '<span class="ii-code">' + f.flight + '</span>' +
          '<span class="ii-detail">+' + f.delay_min + ' min</span>' +
          '<span class="intel-badge sev-' + (lv==='high'?'severe':lv==='medium'?'high':'moderate') + '">' + f.delay_risk + '</span></div>';
      }}).join('');
    }} else {{ dlEl.innerHTML = '<div style="color:rgba(255,255,255,0.35);font-size:10px;">All flights on time</div>'; }}

    const wxEl = document.getElementById('intel-weather');
    if (WEATHER.length) {{
      const cats = {{ VFR: 'low', MVFR: 'moderate', IFR: 'high', LIFR: 'severe' }};
      wxEl.innerHTML = WEATHER.filter(w => w.category !== 'VFR').slice(0, 5).map(w => {{
        const lv = cats[w.category] || 'low';
        return '<div class="intel-item"><span class="ii-code">' + w.station + '</span>' +
          '<span class="ii-detail">' + w.category + '</span>' +
          '<span class="intel-badge sev-' + lv + '">' + w.category + '</span></div>';
      }}).join('') || '<div style="color:rgba(255,255,255,0.35);font-size:10px;">All VFR</div>';
    }} else {{ wxEl.innerHTML = '<div style="color:rgba(255,255,255,0.35);font-size:10px;">No data</div>'; }}

    // AFRI top-risk flights
    const afriEl = document.getElementById('intel-afri');
    const afriFlights = [...FLIGHTS].filter(f => f.afri > 0).sort((a,b) => b.afri - a.afri).slice(0, 5);
    if (afriFlights.length) {{
      afriEl.innerHTML = afriFlights.map(f => {{
        const lv = (f.afri_level||'Normal').toLowerCase();
        const sevClass = lv === 'critical' ? 'severe' : lv === 'high' ? 'high' : lv === 'elevated' ? 'moderate' : 'low';
        return '<div class="intel-item" onclick="showPopupByFlight(\'' + f.flight + '\')">'
          + '<span class="ii-code">' + f.flight + '</span>'
          + '<span class="ii-detail">AFRI ' + f.afri + '</span>'
          + '<span class="intel-badge sev-' + sevClass + '">' + f.afri_level + '</span></div>';
      }}).join('');
    }} else {{ afriEl.innerHTML = '<div style="color:rgba(255,255,255,0.35);font-size:10px;">All arrivals normal</div>'; }}
  }}
  renderIntelPanel();

  // =============================================================================
  //  AIRPORT FILTER
  // =============================================================================
  let activeAirportFilter = null;

  window.filterByAirport = function(code) {{
    activeAirportFilter = code;
    // Filter flights by showing only matching dep/arr
    const updatedFeatures = FLIGHTS.map((f, idx) => {{
      const match = (f.dep === code || f.arr === code);
      const sz = f.selected ? 56 : (f.dimmed ? 30 : 36);
      const iconKey = f.family + '|' + f.color + '|' + sz;
      return {{
        type: 'Feature',
        geometry: {{ type: 'Point', coordinates: [f.lng, f.lat] }},
        properties: {{
          idx: idx, flight: f.flight, iconKey: iconKey,
          heading: f.heading, selected: f.selected, dimmed: !match,
          opacity: match ? 1.0 : 0.25, iconSize: match ? 0.6 : 0.35, show: match,
        }}
      }};
    }});
    map.getSource('flights').setData({{type:'FeatureCollection',features:updatedFeatures}});

    let count = FLIGHTS.filter(f => f.dep === code || f.arr === code).length;
    document.getElementById('af-label').textContent = '✈ ' + code;
    document.getElementById('af-count').textContent = count + ' flights';
    document.getElementById('airport-filter-bar').style.display = 'flex';

    const metric = AIRPORT_METRICS.find(a => a.airport_iata === code);
    if (metric) {{
      document.getElementById('af-count').textContent = count + ' flights · Score ' + metric.congestion_score + ' · ' + metric.congestion_level;
    }}

    const apt = AIRPORTS.find(a => a.iata === code);
    if (apt) map.flyTo({{ center: [apt.lng, apt.lat], zoom: 7, duration: 1000 }});
    closePopup();
  }};

  window.clearAirportFilter = function() {{
    if (!activeAirportFilter) return;
    activeAirportFilter = null;
    updateFlightSelection(selectedFlight);


    document.getElementById('airport-filter-bar').style.display = 'none';
  }};

  window.highlightRoute = function(dep, arr) {{
    activeAirportFilter = null;
    const updatedFeatures = FLIGHTS.map((f, idx) => {{
      const match = (f.dep===dep&&f.arr===arr)||(f.dep===arr&&f.arr===dep);
      const sz = f.selected ? 56 : (f.dimmed ? 30 : 36);
      const iconKey = f.family + '|' + f.color + '|' + sz;
      return {{
        type:'Feature', geometry:{{type:'Point',coordinates:[f.lng,f.lat]}},
        properties:{{
          idx:idx, flight:f.flight, iconKey:iconKey, heading:f.heading,
          selected:f.selected, dimmed:!match, opacity:match?1.0:0.25,
          iconSize:match?0.6:0.35, show:match,
        }}
      }};
    }});
    map.getSource('flights').setData({{type:'FeatureCollection',features:updatedFeatures}});

    let count = FLIGHTS.filter(f => (f.dep===dep&&f.arr===arr)||(f.dep===arr&&f.arr===dep)).length;
    document.getElementById('af-label').textContent = '✈ ' + dep + ' ↔ ' + arr;
    document.getElementById('af-count').textContent = count + ' flights';
    document.getElementById('airport-filter-bar').style.display = 'flex';
  }};

  // =============================================================================
  //  KEYBOARD SHORTCUTS
  // =============================================================================
  document.addEventListener('keydown', function(e) {{
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    const key = e.key.toLowerCase();
    if (key === 'escape') {{ closePopup(); cancelCompare(); }}
    if (key === 'arrowright' || key === 'arrowdown') {{
      e.preventDefault();
      currentFlightIndex = (currentFlightIndex + 1) % FLIGHTS.length;
      selectFlight(FLIGHTS[currentFlightIndex], true);
    }}
    if (key === 'arrowleft' || key === 'arrowup') {{
      e.preventDefault();
      currentFlightIndex = (currentFlightIndex - 1 + FLIGHTS.length) % FLIGHTS.length;
      selectFlight(FLIGHTS[currentFlightIndex], true);
    }}

    if (key === 'h') toggleLayer('heatmap');
    if (key === 'w') toggleLayer('weather');
    if (key === 'b') toggleLayer('board');
    if (key === 't') toggleLayer('timeline');
    if (key === 'e') exportScreenshot();
    if (key === 'i') toggleLayer('intel');
    if (key === 'c' && selectedFlight) startCompare();
  }});

  // =============================================================================
  //  EXPORT
  // =============================================================================
  window.exportScreenshot = function() {{
    const toast = document.getElementById('export-toast');
    try {{
      const canvas = map.getCanvas();
      canvas.toBlob(function(blob) {{
        if (!blob) return;
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = 'aviation-intel-' + Date.now() + '.png';
        document.body.appendChild(a); a.click(); document.body.removeChild(a);
        URL.revokeObjectURL(url);
        toast.style.display = 'block';
        setTimeout(function() {{ toast.style.display = 'none'; }}, 2000);
        playTone(1200, 0.08, 'sine');
      }});
    }} catch(ex) {{
      if (typeof html2canvas !== 'undefined') {{
        html2canvas(document.body).then(function(c) {{
          const a = document.createElement('a');
          a.href = c.toDataURL('image/png');
          a.download = 'aviation-intel-' + Date.now() + '.png'; a.click();
          toast.style.display = 'block';
          setTimeout(function() {{ toast.style.display = 'none'; }}, 2000);
        }});
      }}
    }}
  }};

}}); // end map.on('load')

}} catch(err) {{
  console.error('Map init error:', err);
  document.getElementById('skeleton').innerHTML = '<div style="color:#ff5555;font-size:16px;padding:20px;">Map error: ' + err.message + '</div>';
}}
}})();
</script>
</body>
</html>"""
