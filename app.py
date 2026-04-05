"""
app.py
------
Aviation Intelligence Platform — India-first live operations dashboard.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import concurrent.futures
import html
import json
import logging
import os
import threading
import time

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from analytics import (
    compute_arrival_estimate,
    compute_airport_schedule_pressure,
    compute_airport_traffic_metrics,
    compute_delay_prediction,
    enrich_flights_with_predictions,
    compute_route_congestion,
    compute_fuel_estimate,
    detect_all_anomalies,
)
from data_fetcher import (
    REGION_MAP_SETTINGS,
    get_airline_name,
    get_airports_for_region,
    get_airport_schedules,
    get_flight_data,
)
from snapshot_store import snapshot_store
from tracker import tracker as flight_tracker
from utils import create_map, safe_text as _safe_text, normalize_flight_query as _normalize_flight_query
from mapbox_base import build_flights_json, generate_mapbox_base_html

from weather_fetcher import get_nearest_airport_weather, WeatherImpact


logging.basicConfig(level=logging.WARNING)

st.set_page_config(
    page_title="Aviation Intelligence Platform",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
  [data-testid="stSidebar"],
  [data-testid="stToolbar"],
  [data-testid="stHeader"],
  [data-testid="stStatusWidget"],
  #MainMenu,
  footer {
    display: none !important;
  }
  html, body, .stApp {
    height: 100vh !important;
    max-height: 100vh !important;
    overflow: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
  }
  .stApp {
    background:
      radial-gradient(circle at top left, rgba(143,209,255,0.09), transparent 18%),
      radial-gradient(circle at bottom right, rgba(188,255,92,0.05), transparent 14%),
      linear-gradient(145deg, #04080d, #0a1017 44%, #0d1420);
  }
  .block-container {
    padding: 0 !important;
    max-width: 100% !important;
    height: 100vh !important;
    overflow: hidden !important;
  }
  /* hide Streamlit iframe border/gap */
  iframe {
    display: block !important;
    border: none !important;
    vertical-align: top !important;
  }
  /* remove Streamlit component wrapper offsets */
  [data-testid="stCustomComponentV1"],
  [data-testid="stIframe"] {
    margin: 0 !important;
    padding: 0 !important;
    line-height: 0 !important;
    font-size: 0 !important;
  }
  /* hide ALL Streamlit spinner/status/toast overlays */
  [data-testid="stSpinner"],
  [data-testid="stSpinnerContainer"],
  [data-testid="stNotification"],
  [data-testid="stAlert"],
  .stSpinner,
  [data-baseweb="notification"],
  [class*="StatusWidget"],
  [class*="AppRunningSpinner"] {
    display: none !important;
  }
</style>
""",
    unsafe_allow_html=True,
)


DASHBOARD_HEIGHT = 1080


@st.cache_data(ttl=60, show_spinner=False)
def fetch_flights(region: str) -> pd.DataFrame:
    return get_flight_data(region=region).copy()


@st.cache_data(ttl=10, show_spinner=False)
def _fetch_live_positions(region: str, _tick: int) -> pd.DataFrame:
    """Short-TTL live position fetch used by the autonomous fragment."""
    return get_flight_data(region=region).copy()


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_airport_schedules_cached(airport_code: str) -> tuple[object, object]:
    """Fetch arrival+departure schedules for one airport; cached for 15 min."""
    arr = get_airport_schedules(airport_code, direction="arrival")
    dep = get_airport_schedules(airport_code, direction="departure")
    return arr, dep



def _find_selected_flight(df: pd.DataFrame, query: str) -> pd.DataFrame:
    normalized_query = _normalize_flight_query(query)
    if not normalized_query or df.empty or "flight_iata" not in df.columns:
        return df.iloc[0:0]

    normalized_flights = df["flight_iata"].fillna("").map(_normalize_flight_query)
    exact_matches = df[normalized_flights == normalized_query]
    if not exact_matches.empty:
        return exact_matches
    return df[normalized_flights.str.contains(normalized_query, na=False)]


def _nearest_airport(lat: float | None, lng: float | None, airports: list[dict]) -> dict | None:
    if pd.isna(lat) or pd.isna(lng) or not airports:
        return None
    return min(
        airports,
        key=lambda airport: ((airport["lat"] - float(lat)) ** 2 + (airport["lng"] - float(lng)) ** 2),
    )


def _fmt_altitude(value) -> str:
    return f"{int(value):,} ft" if pd.notna(value) else "N/A"


def _fmt_speed(value) -> str:
    return f"{float(value):.0f} kts" if pd.notna(value) else "N/A"


def _fmt_float(value) -> str:
    return f"{float(value):.2f}" if pd.notna(value) else "N/A"


def _js(s) -> str:
    """Escape a value for safe embedding inside a JS double-quoted string literal."""
    return json.dumps(str(s))[1:-1]


def _build_reason_chips(reason_text: str) -> str:
    parts = [p.strip() for p in reason_text.split(",") if p.strip()]
    if not parts:
        parts = ["stable profile"]
    chips = []
    for part in parts[:3]:
        chips.append(f"<span class='chip'>{html.escape(part)}</span>")
    return "".join(chips)


from dataclasses import dataclass, field


@dataclass
class DashboardContext:
    map_embed_html: str
    selected_row: pd.Series
    selected_flight: str
    search_has_match: bool
    airport_count: int
    flight_count: int
    avg_speed: str
    avg_altitude: str
    total_airlines: int
    top_airport: str
    top_airport_volume: int
    top_airport_pressure: str
    top_route: str
    top_route_volume: int
    top_route_score: str
    nearest_airport: "dict | None"
    delay_risk: str
    expected_delay: str
    delay_reasons: str
    destination_pressure: str
    expected_arrival: str
    passenger_summary: str
    arrival_distance: str
    weather_severity: str
    weather_summary: str
    weather_station: str
    top5_airports_html: str
    top5_routes_html: str
    delay_list_html: str
    route_list_html: str
    delay_reason_chips: str
    airport_focus_code: str
    airport_focus_count: int
    airport_index_text: str
    autorefresh_paused: bool = False
    flights_csv: str = ""
    anomaly_count: int = 0


def _build_dashboard_html(ctx: DashboardContext) -> str:
    map_embed_html = ctx.map_embed_html
    selected_row = ctx.selected_row
    selected_flight = ctx.selected_flight
    search_has_match = ctx.search_has_match
    airport_count = ctx.airport_count
    flight_count = ctx.flight_count
    avg_speed = ctx.avg_speed
    avg_altitude = ctx.avg_altitude
    total_airlines = ctx.total_airlines
    top_airport = ctx.top_airport
    top_airport_volume = ctx.top_airport_volume
    top_airport_pressure = ctx.top_airport_pressure
    top_route = ctx.top_route
    top_route_volume = ctx.top_route_volume
    top_route_score = ctx.top_route_score
    nearest_airport = ctx.nearest_airport
    delay_risk = ctx.delay_risk
    expected_delay = ctx.expected_delay
    delay_reasons = ctx.delay_reasons
    destination_pressure = ctx.destination_pressure
    expected_arrival = ctx.expected_arrival
    passenger_summary = ctx.passenger_summary
    arrival_distance = ctx.arrival_distance
    weather_severity = ctx.weather_severity
    weather_summary = ctx.weather_summary
    weather_station = ctx.weather_station
    top5_airports_html = ctx.top5_airports_html
    top5_routes_html = ctx.top5_routes_html
    delay_list_html = ctx.delay_list_html
    route_list_html = ctx.route_list_html
    delay_reason_chips = ctx.delay_reason_chips
    airport_focus_code = ctx.airport_focus_code
    airport_focus_count = ctx.airport_focus_count
    airport_index_text = ctx.airport_index_text
    autorefresh_paused = ctx.autorefresh_paused
    flights_csv = ctx.flights_csv
    anomaly_count = ctx.anomaly_count

    airline_code = _safe_text(selected_row.get("airline_iata"))
    airline_name = get_airline_name(airline_code) if airline_code != "N/A" else "Unknown airline"
    flight_code = _safe_text(selected_row.get("flight_iata"))
    dep = _safe_text(selected_row.get("dep_iata"))
    arr = _safe_text(selected_row.get("arr_iata"))
    aircraft = _safe_text(selected_row.get("aircraft_icao"), "Unknown aircraft")
    status = _safe_text(selected_row.get("status"), "en route").replace("-", " ").title()
    altitude = _fmt_altitude(selected_row.get("altitude_ft"))
    speed = _fmt_speed(selected_row.get("speed_kts"))
    lat = _fmt_float(selected_row.get("latitude"))
    lng = _fmt_float(selected_row.get("longitude"))
    nearest_code = nearest_airport["iata"] if nearest_airport else "N/A"
    nearest_city = nearest_airport["city"] if nearest_airport else "India"
    selected_label = selected_flight.strip() if selected_flight else flight_code
    search_state = "Tracking Flight" if search_has_match and selected_flight else ("No Match Found" if selected_flight else "Live Search")
    search_state_color = "#bcff5c" if search_has_match and selected_flight else ("#ffb74d" if selected_flight else "rgba(243,247,251,0.62)")

    risk_label = delay_risk
    variance = expected_delay

    # Pre-encode HTML blobs for safe embedding in JS string literals
    # (onclick attributes contain double-quotes that break bare JS string embedding)
    _j_airports = json.dumps(top5_airports_html)
    _j_routes   = json.dumps(top5_routes_html)
    _j_delay    = json.dumps(delay_list_html)
    _j_routelist = json.dumps(route_list_html)

    # JS-safe versions of dynamic text strings (prevents newline/quote crashes)
    _j_flight_code   = _js(flight_code)
    _j_airline_name  = _js(airline_name)
    _j_risk_label    = _js(risk_label)
    _j_variance      = _js(variance)
    _j_delay_reasons = _js(delay_reasons)
    _j_dest_pressure = _js(destination_pressure)
    _j_wx_sev        = _js(weather_severity)
    _j_wx_summary    = _js(weather_summary)
    _j_wx_station    = _js(weather_station)
    _j_top_airport   = _js(top_airport)
    _j_top_route     = _js(top_route)
    _j_top_route_score = _js(top_route_score)
    _j_top_ap_pressure = _js(top_airport_pressure)
    _j_top_ap_pressure_lower = _js(top_airport_pressure.lower())
    _j_ap_focus_code = _js(airport_focus_code)
    _j_ap_index_text = _js(airport_index_text)
    _j_dep           = _js(dep)
    _j_arr           = _js(arr)
    _j_expected_arr  = _js(expected_arrival)
    _j_pax_summary   = _js(passenger_summary)
    _j_arr_distance  = _js(arrival_distance)
    _j_aircraft      = _js(aircraft)
    _j_status        = _js(status)

    shell_style = f"""
  <style>
    *, *::before, *::after {{
      box-sizing: border-box;
    }}
    :root {{
      --text: #f3f7fb;
      --muted: rgba(243,247,251,0.68);
      --dim: rgba(243,247,251,0.44);
      --line: rgba(255,255,255,0.10);
      --yellow: #ffcf34;
      --lime: #bcff5c;
      --shadow: 0 22px 60px rgba(0,0,0,0.42);
      --stage-height: {DASHBOARD_HEIGHT}px;
      --glass: linear-gradient(180deg, rgba(14, 21, 31, 0.84), rgba(10, 16, 24, 0.78));
      --glass-strong: linear-gradient(180deg, rgba(12, 18, 28, 0.92), rgba(9, 14, 22, 0.88));
    }}
    html, body {{
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      font-family: "SF Pro Display", "Segoe UI", Arial, sans-serif;
      color-scheme: light dark;
    }}
    body {{ padding: 0; background: #081018; overflow: hidden; }}
    .frame {{
      position: relative;
      width: 100%;
      height: var(--stage-height);
      border-radius: 24px;
      overflow: hidden;
      border: 1px solid rgba(255,255,255,0.07);
      box-shadow: var(--shadow);
      background: linear-gradient(140deg, #131c22 0%, #172632 38%, #101722 100%);
    }}
    .frame::after {{
      content: "";
      position: absolute;
      inset: 0;
      background:
        linear-gradient(180deg, rgba(6, 10, 16, 0.34), rgba(6, 10, 16, 0.06) 24%, rgba(6, 10, 16, 0.22)),
        radial-gradient(circle at top left, rgba(255,255,255,0.05), transparent 18%);
      pointer-events: none;
      z-index: 18;
    }}
    .frame::before {{
      content: "";
      position: absolute;
      inset: 0;
      background:
        repeating-linear-gradient(90deg, rgba(255,255,255,0.018) 0 1px, transparent 1px 86px),
        repeating-linear-gradient(0deg, rgba(255,255,255,0.018) 0 1px, transparent 1px 86px);
      opacity: 0.42;
      pointer-events: none;
      z-index: 20;
    }}
    .map-shell {{
      position: absolute;
      inset: 0;
      z-index: 1;
      background: #071019;
    }}
    .map-shell > div,
    .map-shell > div > div {{
      width: 100% !important;
      height: 100% !important;
    }}
    .map-shell iframe {{
      position: absolute !important;
      inset: 0 !important;
      width: 100% !important;
      height: 100% !important;
      border: 0 !important;
      background: #071019 !important;
      display: block !important;
    }}
    .overlay {{
      position: absolute;
      inset: 0;
      z-index: 50;
      pointer-events: none;
      color: var(--text);
    }}
    .panel, .search, .time-pill, .map-pill, .menu-pill, .detail-bar {{
      border: 1px solid var(--line);
      backdrop-filter: blur(18px);
      box-shadow: 0 14px 32px rgba(0,0,0,0.18);
    }}
    .topbar {{
      position: absolute;
      top: 12px;
      left: 12px;
      right: 12px;
      display: grid;
      grid-template-columns: 250px minmax(420px, 1fr) 84px minmax(280px, 380px) 82px 46px;
      gap: 12px;
      align-items: center;
      pointer-events: auto;
    }}
    .brand {{
      height: 62px;
      display: flex;
      align-items: center;
      gap: 11px;
      padding: 0 15px;
      border-radius: 20px;
      background: var(--glass-strong);
      border: 1px solid var(--line);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
    }}
    .brand-mark {{
      width: 32px;
      height: 32px;
      border-radius: 11px;
      display: grid;
      place-items: center;
      background: linear-gradient(145deg, var(--yellow), #ff9d35);
      color: #091018;
      font-weight: 900;
      font-size: 20px;
      flex: 0 0 auto;
    }}
    .brand-text {{
      font-size: 16px;
      line-height: 0.96;
      font-weight: 800;
      letter-spacing: -0.03em;
    }}
    .nav {{
      height: 62px;
      padding: 6px;
      border-radius: 20px;
      background: var(--glass-strong);
      border: 1px solid var(--line);
      display: flex;
      align-items: center;
      gap: 6px;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
    }}
    .nav button {{
      min-width: 86px;
      padding: 10px 12px;
      border-radius: 14px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      line-height: 1.1;
      text-align: left;
      display: flex;
      flex-direction: column;
      justify-content: center;
      gap: 2px;
      border: 0;
      background: transparent;
      cursor: pointer;
    }}
    .nav button strong {{
      font-size: 11px;
      font-weight: 800;
      letter-spacing: -0.02em;
    }}
    .nav button small {{
      font-size: 9px;
      font-weight: 600;
      color: rgba(243,247,251,0.44);
      line-height: 1.05;
    }}
    .nav button.active {{
      background: rgba(255,255,255,0.95);
      color: #0d1621;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.55);
    }}
    .nav button.active small {{
      color: rgba(13,22,33,0.54);
    }}
    .nav button:not(.active) {{
      opacity: 0.92;
      transition: opacity 140ms ease, color 140ms ease;
    }}
    .nav button:hover {{
      color: rgba(255,255,255,0.92);
      opacity: 1;
    }}
    .time-pill, .map-pill, .menu-pill {{
      height: 62px;
      border-radius: 20px;
      background: var(--glass-strong);
      color: var(--muted);
      display: grid;
      place-items: center;
      font-weight: 700;
      font-size: 14px;
      line-height: 1.05;
      text-align: center;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
    }}
    .ar-toggle {{
      cursor: pointer;
      border: 1px solid var(--line);
      backdrop-filter: blur(18px);
      box-shadow: 0 14px 32px rgba(0,0,0,0.18);
      transition: background 200ms ease, color 200ms ease;
    }}
    .ar-toggle:hover {{
      background: rgba(255,207,52,0.15);
      color: var(--yellow);
    }}
    .export-btn {{
      position: absolute;
      bottom: 14px;
      left: 16px;
      pointer-events: auto;
      height: 38px;
      padding: 0 16px;
      border-radius: 12px;
      background: var(--glass-strong);
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
      backdrop-filter: blur(18px);
      box-shadow: 0 8px 20px rgba(0,0,0,0.22);
      display: flex;
      align-items: center;
      gap: 6px;
      transition: background 200ms ease, color 200ms ease;
    }}
    .export-btn:hover {{
      background: rgba(188,255,92,0.12);
      color: var(--lime);
    }}
    .search {{
      height: 62px;
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 0 14px 0 18px;
      border-radius: 20px;
      background: rgba(251,253,255,0.97) !important;
      color: #091018 !important;
      box-shadow: 0 12px 28px rgba(0,0,0,0.12);
      color-scheme: light;
    }}
    .search input {{
      flex: 1;
      border: 0;
      outline: none;
      background: transparent !important;
      font-size: 14px;
      color: #0d1621 !important;
      font-weight: 600;
    }}
    .search button {{
      border: 0;
      border-radius: 12px;
      background: linear-gradient(180deg, rgba(17,24,34,0.12), rgba(17,24,34,0.08));
      color: rgba(9,16,24,0.72);
      padding: 9px 14px;
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
    }}
    .scope-card {{
      position: absolute;
      top: 154px;
      left: 16px;
      width: 260px;
      padding: 14px 14px 15px;
      border-radius: 18px;
      background: var(--glass);
      pointer-events: auto;
      box-shadow: 0 12px 28px rgba(0,0,0,0.18), inset 0 1px 0 rgba(255,255,255,0.04);
    }}
    .scope-card .k, .mini-card .k, .metric .k {{
      color: var(--dim);
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 10px;
      margin-bottom: 6px;
    }}
    .scope-card .v {{
      font-size: 18px;
      font-weight: 800;
      margin-bottom: 6px;
      letter-spacing: -0.03em;
    }}
    .scope-card .sub {{
      color: var(--muted);
      font-size: 11px;
      line-height: 1.42;
      margin-bottom: 12px;
      max-width: 30ch;
    }}
    .scope-kpis {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-bottom: 12px;
    }}
    .scope-kpis div {{
      padding-top: 10px;
      border-top: 1px solid rgba(255,255,255,0.08);
    }}
    .scope-kpis strong {{
      display: block;
      font-size: 11px;
      color: var(--muted);
      margin-bottom: 6px;
      font-weight: 600;
    }}
    .scope-kpis span {{
      font-size: 20px;
      font-weight: 800;
      letter-spacing: -0.03em;
    }}
    .scope-button {{
      display: block;
      width: 100%;
      text-align: center;
      padding: 12px 14px;
      border-radius: 999px;
      background: linear-gradient(180deg, rgba(245,248,252,0.96), rgba(222,233,246,0.94));
      color: #0c1520;
      font-size: 13px;
      font-weight: 800;
      text-decoration: none;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.8);
      transition: transform 140ms ease, box-shadow 140ms ease;
    }}
    .scope-button:hover {{
      transform: translateY(-1px);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.8), 0 10px 24px rgba(10,16,24,0.22);
    }}
    .right-stack {{
      position: absolute;
      top: 154px;
      right: 16px;
      width: 200px;
      display: grid;
      gap: 10px;
      pointer-events: auto;
    }}
    #right-stack-secondary {{
      top: 480px;
    }}
    .mini-card {{
      padding: 12px 13px;
      border-radius: 16px;
      background: var(--glass);
      position: relative;
      overflow: hidden;
      box-shadow: 0 12px 28px rgba(0,0,0,0.18), inset 0 1px 0 rgba(255,255,255,0.04);
    }}
    .mini-card::before {{
      content: "";
      position: absolute;
      left: 0;
      top: 0;
      width: 100%;
      height: 1px;
      background: linear-gradient(90deg, rgba(255,255,255,0.16), rgba(255,255,255,0.02));
    }}
    .mini-card .v {{
      font-size: 14px;
      font-weight: 800;
      line-height: 1.2;
      margin-bottom: 4px;
    }}
    .mini-card .sub {{
      color: var(--muted);
      font-size: 11px;
      line-height: 1.42;
      max-width: 24ch;
    }}
    .list-card {{
      padding: 12px 13px;
      border-radius: 16px;
      background: var(--glass);
      border: 1px solid var(--line);
      box-shadow: 0 12px 28px rgba(0,0,0,0.18), inset 0 1px 0 rgba(255,255,255,0.04);
    }}
    .list-card h4 {{
      margin: 0 0 8px;
      font-size: 12px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--dim);
    }}
    .list-card ul {{
      list-style: none;
      margin: 0;
      padding: 0;
      display: grid;
      gap: 6px;
    }}
    .list-card li {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      font-size: 12px;
      color: var(--text);
    }}
    .list-card li span {{
      color: var(--muted);
      font-size: 11px;
    }}
    .list-card a {{
      color: var(--text);
      text-decoration: none;
      font-weight: 700;
    }}
    .list-card a:hover {{
      color: #bcff5c;
    }}
    .chip-row {{
      margin-top: 6px;
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .chip {{
      padding: 3px 8px;
      border-radius: 999px;
      background: rgba(89,183,255,0.14);
      color: #cde6ff;
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.04em;
    }}
    .detail-bar {{
      position: absolute;
      left: 50%;
      bottom: 14px;
      transform: translateX(-50%);
      width: min(980px, calc(100% - 140px));
      padding: 10px 12px;
      border-radius: 18px;
      background: linear-gradient(180deg, rgba(12, 18, 28, 0.92), rgba(10, 15, 24, 0.88));
      pointer-events: auto;
      box-shadow: 0 18px 42px rgba(0,0,0,0.24), inset 0 1px 0 rgba(255,255,255,0.04);
      transition: opacity 160ms ease, transform 160ms ease;
    }}
    .detail-bar.hidden {{
      opacity: 0;
      transform: translateX(-50%) translateY(12px);
      pointer-events: none;
    }}
    .detail-grid {{
      display: grid;
      grid-template-columns: 1fr 0.95fr 0.55fr 0.55fr 0.72fr 0.62fr 0.8fr;
      gap: 10px;
      align-items: center;
    }}
    .detail-main h2 {{
      margin: 0;
      color: var(--lime);
      font-size: 26px;
      letter-spacing: -0.05em;
      line-height: 0.95;
      text-shadow: 0 0 18px rgba(188,255,92,0.12);
    }}
    .detail-main p {{
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.12;
      max-width: 24ch;
    }}
    .route-box {{
      display: grid;
      grid-template-columns: 1fr auto 1fr;
      align-items: center;
      gap: 10px;
      padding: 2px 8px;
      border-left: 1px solid rgba(255,255,255,0.08);
      border-right: 1px solid rgba(255,255,255,0.08);
    }}
    .route-box .code {{
      display: block;
      font-size: 26px;
      font-weight: 800;
      line-height: 0.95;
      letter-spacing: -0.04em;
    }}
    .route-box .city {{
      color: var(--muted);
      font-size: 11px;
      margin-top: 4px;
    }}
    .route-box .plane {{
      color: var(--yellow);
      font-size: 14px;
    }}
    .metric .v {{
      font-size: 18px;
      font-weight: 800;
      line-height: 1.05;
      letter-spacing: -0.03em;
    }}
    .metric {{
      min-width: 0;
      padding-left: 4px;
    }}
    .metric .sub {{
      margin-top: 3px;
      color: var(--muted);
      font-size: 10px;
    }}
    .kpi-strip {{
      position: absolute;
      top: 86px;
      left: 50%;
      transform: translateX(-50%);
      display: grid;
      grid-template-columns: repeat(4, minmax(118px, 138px));
      gap: 8px;
      pointer-events: auto;
    }}
    .kpi-card {{
      padding: 9px 11px;
      border-radius: 14px;
      background: linear-gradient(180deg, rgba(12, 18, 28, 0.82), rgba(10, 15, 24, 0.76));
      border: 1px solid var(--line);
      backdrop-filter: blur(14px);
      box-shadow: 0 10px 24px rgba(0,0,0,0.14);
      transition: border-color 160ms ease, transform 160ms ease;
    }}
    .kpi-card:hover {{
      border-color: rgba(255,255,255,0.22);
      transform: translateY(-1px);
    }}
    .kpi-card .k {{
      color: var(--dim);
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 9px;
      margin-bottom: 6px;
    }}
    .kpi-card .v {{
      color: #f7fbff;
      font-size: 17px;
      font-weight: 800;
      letter-spacing: -0.03em;
      line-height: 1;
    }}
    .search-status {{
      position: absolute;
      top: 84px;
      right: 148px;
      padding: 7px 10px;
      border-radius: 999px;
      background: rgba(10, 15, 24, 0.72);
      border: 1px solid rgba(255,255,255,0.08);
      color: {search_state_color};
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.10em;
      text-transform: uppercase;
      backdrop-filter: blur(12px);
      pointer-events: none;
    }}
    @media (max-width: 1440px) {{
      :root {{
        --stage-height: 860px;
      }}
      .topbar {{
        grid-template-columns: 220px minmax(300px, 1fr) 78px minmax(220px, 320px) 76px 44px;
      }}
      .kpi-strip {{
        grid-template-columns: repeat(4, minmax(102px, 122px));
      }}
    }}
    @media (max-width: 1280px) {{
      :root {{
        --stage-height: 820px;
      }}
      .scope-card,
      .right-stack {{
        top: 146px;
      }}
      #right-stack-secondary {{
        top: 460px;
      }}
      .kpi-strip {{
        top: 82px;
      }}
    }}
    @media (max-width: 1024px) {{
      :root {{
        --stage-height: 760px;
      }}
      .topbar {{
        grid-template-columns: 48px 1fr 64px minmax(0,1fr) 64px 44px;
        gap: 8px;
        top: 8px;
        left: 8px;
        right: 8px;
      }}
      .brand-text {{
        display: none;
      }}
      .brand {{
        justify-content: center;
        width: 48px;
        padding: 0;
      }}
      .nav button small {{
        display: none;
      }}
      .nav button {{
        padding: 0 8px;
        font-size: 11px;
      }}
      .kpi-strip {{
        top: 80px;
        grid-template-columns: repeat(2, minmax(90px, 1fr));
        gap: 6px;
      }}
      .scope-card {{
        top: 180px;
        left: 8px;
        width: 220px;
      }}
      .right-stack {{
        top: 180px;
        right: 8px;
      }}
    }}
    @media (max-width: 768px) {{
      :root {{
        --stage-height: 680px;
      }}
      .topbar {{
        grid-template-columns: 48px 1fr 0 0 64px 44px;
        gap: 6px;
      }}
      .search {{
        display: none;
      }}
      .map-pill {{
        display: none;
      }}
      .search-status {{
        right: 56px;
      }}
      .kpi-strip {{
        top: 76px;
        grid-template-columns: repeat(2, 1fr);
        left: 8px;
        right: 8px;
      }}
      .right-stack {{
        display: none;
      }}
      .scope-card {{
        top: 176px;
        left: 8px;
        right: 8px;
        width: auto;
      }}
      .detail-bar {{
        left: 8px;
        right: 8px;
      }}
      .export-btn {{
        bottom: 8px;
        left: 8px;
      }}
    }}
  </style>
"""

    overlay_html = f"""
  <div class="map-shell">{map_embed_html}</div>
  <div class="overlay">
    <div class="topbar">
      <div class="brand">
        <div class="brand-mark">A</div>
        <div class="brand-text">Aviation Intelligence<br>Platform</div>
      </div>
      <div class="nav panel">
        <button type="button" class="active module-tab" data-module="live_ops"><strong>Live Ops</strong><small>Tracking</small></button>
        <button type="button" class="module-tab" data-module="delay_prediction"><strong>Delay Prediction</strong><small>Risk engine</small></button>
        <button type="button" class="module-tab" data-module="airport_traffic"><strong>Airport Traffic</strong><small>Hub pressure</small></button>
        <button type="button" class="module-tab" data-module="route_intelligence"><strong>Route Intelligence</strong><small>Corridor load</small></button>
        <button type="button" class="module-tab" data-module="passenger_view"><strong>Passenger View</strong><small>Inbound status</small></button>
        <button type="button" class="module-tab" data-module="alerts"><strong>Alerts</strong><small>{anomaly_count} active</small></button>
      </div>
      <div class="time-pill">{time.strftime("%H:%M", time.gmtime())}<br>UTC</div>
      <form class="search" method="get" action="javascript:void(0)" onsubmit="return submitDashboardSearch(event)">
        <input type="text" name="flight" value="{html.escape(selected_flight)}" placeholder="Find flights, airports and more">
        <button type="submit">Go</button>
      </form>
      <div class="map-pill">India<br>Map</div>
      <button type="button" class="menu-pill ar-toggle" id="ar-toggle-btn"
        onclick="(function(){{
          try{{
            var url=new URL(window.top.location.href);
            var cur=url.searchParams.get('autorefresh');
            url.searchParams.set('autorefresh', cur==='0'?'1':'0');
            window.top.location.href=url.toString();
          }}catch(e){{
            var url=new URL(window.parent.location.href);
            var cur=url.searchParams.get('autorefresh');
            url.searchParams.set('autorefresh', cur==='0'?'1':'0');
            window.parent.location.href=url.toString();
          }}
        }})();"
        title="{{'Resume live refresh' if autorefresh_paused else 'Pause live refresh'}}">
        {'▶' if autorefresh_paused else '⏸'}<br>{'Live' if not autorefresh_paused else 'Paused'}
      </button>
    </div>
    <div class="search-status">{search_state}</div>
    <button type="button" class="export-btn" id="export-csv-btn"
      onclick="(function(){{
        var csv=window._FLIGHTS_CSV||'';
        if(!csv)return;
        var blob=new Blob([csv],{{type:'text/csv'}});
        var url=URL.createObjectURL(blob);
        var a=document.createElement('a');
        a.href=url;
        a.download='flights_' + new Date().toISOString().slice(0,16).replace(':','-') + '.csv';
        document.body.appendChild(a);a.click();
        setTimeout(function(){{URL.revokeObjectURL(url);a.remove();}},1000);
      }})();">&#x2193; Export CSV</button>
    <script>window._FLIGHTS_CSV={json.dumps(flights_csv)};</script>
    <div class="kpi-strip">
      <div class="kpi-card">
        <div class="k">Active Flights</div>
        <div class="v">{flight_count:,}</div>
      </div>
      <div class="kpi-card">
        <div class="k">Avg Speed</div>
        <div class="v">{avg_speed}</div>
      </div>
      <div class="kpi-card">
        <div class="k">Avg Altitude</div>
        <div class="v">{avg_altitude}</div>
      </div>
      <div class="kpi-card">
        <div class="k">Airlines</div>
        <div class="v">{total_airlines}</div>
      </div>
    </div>
    <div class="scope-card panel">
      <div id="scope-k" class="k">Live Ops</div>
      <div id="scope-v" class="v">India Airspace</div>
      <div id="scope-sub" class="sub">Map-first operations layer with live aircraft, airport pinpoints, selected flight tracking, and India-wide situational awareness.</div>
      <div class="scope-kpis">
        <div>
          <strong id="scope-kpi-1-label">Active Flights</strong>
          <span id="scope-kpi-1-value">{flight_count:,}</span>
        </div>
        <div>
          <strong id="scope-kpi-2-label">Pinned Airports</strong>
          <span id="scope-kpi-2-value">{airport_count}</span>
        </div>
      </div>
      <a class="scope-button" href="?flight={html.escape(selected_label)}"
         id="scope-button"
         onclick="try {{ const url = new URL(window.top.location.href); url.searchParams.set('flight', '{html.escape(selected_label)}'); window.top.location.href = url.toString(); }} catch (e) {{ const url = new URL(window.parent.location.href); url.searchParams.set('flight', '{html.escape(selected_label)}'); window.parent.location.href = url.toString(); }} return false;">India Live Operations</a>
    </div>
    <div class="right-stack" id="right-stack-primary">
      <div class="mini-card panel">
        <div id="card-1-k" class="k">Delay Prediction</div>
        <div id="card-1-v" class="v">{variance}</div>
        <div id="card-1-sub" class="sub">{risk_label} based on {html.escape(delay_reasons)}, destination pressure: {html.escape(destination_pressure)}, and weather impact: {html.escape(weather_severity)}.</div>
      </div>
      <div class="mini-card panel">
        <div id="card-2-k" class="k">Airport Traffic</div>
        <div id="card-2-v" class="v">{html.escape(top_airport)} highest load</div>
        <div id="card-2-sub" class="sub">{top_airport_volume} flights in the live airport envelope. Schedule pressure is {html.escape(top_airport_pressure)}.</div>
      </div>
      <div class="mini-card panel">
        <div id="card-3-k" class="k">Route Intelligence</div>
        <div id="card-3-v" class="v">{html.escape(top_route)}</div>
        <div id="card-3-sub" class="sub">{top_route_volume} flights active on the corridor. Congestion score: {html.escape(top_route_score)}.</div>
      </div>
    </div>
    <div class="right-stack" id="right-stack-secondary">
      <div class="list-card panel">
        <h4 id="list-1-title">Top Airports</h4>
        <ul id="list-1-body">{top5_airports_html}</ul>
      </div>
      <div class="list-card panel">
        <h4 id="list-2-title">Top Routes</h4>
        <ul id="list-2-body">{top5_routes_html}</ul>
      </div>
    </div>
    <div class="detail-bar hidden">
      <div class="detail-grid">
        <div class="detail-main">
          <h2>{html.escape(flight_code)}</h2>
          <p>{html.escape(airline_name)} • {html.escape(aircraft)} • {html.escape(status)} • {html.escape(risk_label)} / {html.escape(variance)} • {html.escape(weather_severity)} weather</p>
          <div class="chip-row">{delay_reason_chips}</div>
        </div>
        <div class="route-box">
          <div>
            <span class="code">{html.escape(dep)}</span>
            <div class="city">Departure</div>
          </div>
          <div class="plane">✈</div>
          <div style="text-align:right;">
            <span class="code">{html.escape(arr)}</span>
            <div class="city">Arrival</div>
          </div>
        </div>
        <div class="metric">
          <div class="k">Altitude</div>
          <div class="v">{html.escape(altitude)}</div>
          <div class="sub">Live vertical state</div>
        </div>
        <div class="metric">
          <div class="k">Speed</div>
          <div class="v">{html.escape(speed)}</div>
          <div class="sub">Ground speed</div>
        </div>
        <div class="metric">
          <div class="k">Delay Risk</div>
          <div class="v">{html.escape(risk_label)}</div>
          <div class="sub">{html.escape(delay_reasons)}</div>
        </div>
        <div class="metric">
          <div class="k">Expected Arrival</div>
          <div class="v">{html.escape(expected_arrival)}</div>
          <div class="sub">{html.escape(variance)} • {html.escape(arrival_distance)}</div>
        </div>
        <div class="metric">
          <div class="k">Nearest Airport</div>
          <div class="v">{html.escape(nearest_code)}</div>
          <div class="sub">{html.escape(nearest_city)} • {html.escape(lat)}, {html.escape(lng)}</div>
        </div>
      </div>
    </div>
  </div>
"""
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {shell_style}
</head>
<body>
  <div class="frame">
    {overlay_html}
  </div>
</body>
<script>
  const activeModule = "{html.escape(active_module)}";
  const hasUserSelection = {str(search_has_match and bool(selected_flight)).lower()} && activeModule !== 'live_ops';

  function showDelayPopup() {{
    const bar = document.querySelector('.detail-bar');
    if (!bar) return;
    bar.classList.remove('hidden');
  }}

  if (hasUserSelection) {{
    showDelayPopup();
  }}

  const moduleContent = {{
    live_ops: {{
      showPanels: false,
      scopeK: "Live Ops",
      scopeV: "India Airspace",
      scopeSub: "Map-first operations layer with live aircraft, airport pinpoints, and India-wide situational awareness.",
      kpi1Label: "Active Flights",
      kpi1Value: "{flight_count:,}",
      kpi2Label: "Pinned Airports",
      kpi2Value: "{airport_count}",
      buttonLabel: "India Live Operations",
      card1K: "Live Flights",
      card1V: "{flight_count:,}",
      card1Sub: "Aircraft currently active in Indian airspace. Click any aircraft for details.",
      card2K: "Airport Traffic",
      card2V: "{_j_top_airport} busiest",
      card2Sub: "{top_airport_volume} flights in the live airport envelope right now.",
      card3K: "Route Activity",
      card3V: "{_j_top_route}",
      card3Sub: "{top_route_volume} flights active on the corridor.",
      list1Title: "Top Airports",
      list1Html: {_j_airports},
      list2Title: "Top Routes",
      list2Html: {_j_routes}
    }},
    delay_prediction: {{
      showPanels: true,
      scopeK: "Delay Prediction",
      scopeV: "{_j_flight_code} Risk Model",
      scopeSub: "Rule-based delay engine using live speed, altitude, destination congestion, airport schedule pressure, and nearest-airport aviation weather.",
      kpi1Label: "Risk Level",
      kpi1Value: "{_j_risk_label}",
      kpi2Label: "Expected Delay",
      kpi2Value: "{_j_variance}",
      buttonLabel: "Selected Flight Prediction",
      card1K: "Primary Driver",
      card1V: "{_j_delay_reasons}",
      card1Sub: "Prediction reasons are explainable and derived from live operational signals rather than a black-box model.",
      card2K: "Destination Pressure",
      card2V: "{_j_dest_pressure}",
      card2Sub: "Arrival airport load combines current live congestion with near-term scheduled flow.",
      card3K: "Nearest-Airport Weather",
      card3V: "{_j_wx_sev}",
      card3Sub: "{_j_wx_summary} \u2022 {_j_wx_station}.",
      list1Title: "Most Delayed Flights",
      list1Html: {_j_delay},
      list2Title: "Top Airports",
      list2Html: {_j_airports}
    }},
    airport_traffic: {{
      showPanels: true,
      scopeK: "Airport Traffic",
      scopeV: "{_j_ap_focus_code} Focus Hub",
      scopeSub: "Real-time airport intelligence based on aircraft in the terminal area, inbound traffic, and low-altitude arrival activity.",
      kpi1Label: "Selected Airport",
      kpi1Value: "{_j_ap_focus_code}",
      kpi2Label: "Congestion Index",
      kpi2Value: "{_j_ap_index_text}",
      buttonLabel: "Airport Traffic Intelligence",
      card1K: "Schedule Pressure",
      card1V: "{_j_top_ap_pressure}",
      card1Sub: "Upcoming arrival and departure banks are used to estimate pressure in the next hour.",
      card2K: "Airport Envelope",
      card2V: "{airport_focus_count} flights",
      card2Sub: "Inbound + outbound aircraft within the selected airport envelope.",
      card3K: "Busiest Airport Now",
      card3V: "{_j_top_airport}",
      card3Sub: "Current leader in India based on congestion score and inbound traffic density.",
      list1Title: "Top Airports",
      list1Html: {_j_airports},
      list2Title: "Top Routes",
      list2Html: {_j_routes}
    }},
    route_intelligence: {{
      showPanels: true,
      scopeK: "Route Intelligence",
      scopeV: "{_j_top_route}",
      scopeSub: "Corridor analysis surfaces the busiest live routes in Indian airspace using active aircraft count and low-altitude flow.",
      kpi1Label: "Top Corridor",
      kpi1Value: "{_j_top_route}",
      kpi2Label: "Congestion",
      kpi2Value: "{_j_top_route_score}",
      buttonLabel: "Route Congestion Intelligence",
      card1K: "Active Corridor",
      card1V: "{_j_top_route}",
      card1Sub: "{top_route_volume} flights are currently active on this route corridor.",
      card2K: "Congestion Score",
      card2V: "{_j_top_route_score}",
      card2Sub: "Score blends flight density with low-altitude pressure on the route.",
      card3K: "Selected Flight Route",
      card3V: "{_j_dep} \u2192 {_j_arr}",
      card3Sub: "Use this view to compare the selected flight against the busiest Indian corridors.",
      list1Title: "Top Routes",
      list1Html: {_j_routelist},
      list2Title: "Top Airports",
      list2Html: {_j_airports}
    }},
    passenger_view: {{
      showPanels: true,
      scopeK: "Passenger View",
      scopeV: "{_j_flight_code} Inbound Aircraft",
      scopeSub: "Passenger-facing flight insight focused on where the aircraft is now, when it should arrive, and what may be causing delay.",
      kpi1Label: "Expected Arrival",
      kpi1Value: "{_j_expected_arr}",
      kpi2Label: "Expected Delay",
      kpi2Value: "{_j_variance}",
      buttonLabel: "Passenger Flight Insight",
      card1K: "Inbound Aircraft",
      card1V: "{_j_flight_code}",
      card1Sub: "{_j_pax_summary}",
      card2K: "Likely Delay Driver",
      card2V: "{_j_delay_reasons}",
      card2Sub: "Predicted from live speed, destination congestion, airport schedule pressure, and aviation weather.",
      card3K: "Arrival Expectation",
      card3V: "{_j_expected_arr}",
      card3Sub: "{_j_arr_distance} \u2022 {_j_wx_sev} weather impact.",
      list1Title: "Most Delayed Flights",
      list1Html: {_j_delay},
      list2Title: "Top Airports",
      list2Html: {_j_airports}
    }},
    alerts: {{
      showPanels: true,
      scopeK: "Alerts",
      scopeV: "{_j_risk_label} Watch",
      scopeSub: "Operational watch items surface flights and hubs that may require closer attention in the next hour.",
      kpi1Label: "Delay Watch",
      kpi1Value: "{_j_variance}",
      kpi2Label: "Hub Pressure",
      kpi2Value: "{_j_top_ap_pressure}",
      buttonLabel: "Operational Watchlist",
      card1K: "Selected Flight",
      card1V: "{_j_flight_code}",
      card1Sub: "Current watch state: {_j_risk_label} driven by {_j_delay_reasons}.",
      card2K: "Airport Watch",
      card2V: "{_j_top_airport}",
      card2Sub: "Primary airport watch item due to {_j_top_ap_pressure_lower} schedule pressure.",
      card3K: "Corridor Watch",
      card3V: "{_j_top_route}",
      card3Sub: "Most active route corridor currently in the live India frame.",
      list1Title: "Most Delayed Flights",
      list1Html: {_j_delay},
      list2Title: "Top Airports",
      list2Html: {_j_airports}
    }}
  }};

  function setModule(moduleKey, shouldReload = false) {{
    const content = moduleContent[moduleKey];
    if (!content) return;
    document.querySelectorAll('.module-tab').forEach((button) => {{
      button.classList.toggle('active', button.dataset.module === moduleKey);
    }});
    const primary = document.getElementById('right-stack-primary');
    const secondary = document.getElementById('right-stack-secondary');
    if (primary && secondary) {{
      const show = content.showPanels !== false;
      primary.style.display = show ? 'grid' : 'none';
      secondary.style.display = show ? 'grid' : 'none';
    }}
    document.getElementById('scope-k').textContent = content.scopeK;
    document.getElementById('scope-v').textContent = content.scopeV;
    document.getElementById('scope-sub').textContent = content.scopeSub;
    document.getElementById('scope-kpi-1-label').textContent = content.kpi1Label;
    document.getElementById('scope-kpi-1-value').textContent = content.kpi1Value;
    document.getElementById('scope-kpi-2-label').textContent = content.kpi2Label;
    document.getElementById('scope-kpi-2-value').textContent = content.kpi2Value;
    document.getElementById('scope-button').textContent = content.buttonLabel;
    document.getElementById('card-1-k').textContent = content.card1K;
    document.getElementById('card-1-v').textContent = content.card1V;
    document.getElementById('card-1-sub').textContent = content.card1Sub;
    document.getElementById('card-2-k').textContent = content.card2K;
    document.getElementById('card-2-v').textContent = content.card2V;
    document.getElementById('card-2-sub').textContent = content.card2Sub;
    document.getElementById('card-3-k').textContent = content.card3K;
    document.getElementById('card-3-v').textContent = content.card3V;
    document.getElementById('card-3-sub').textContent = content.card3Sub;
    document.getElementById('list-1-title').textContent = content.list1Title;
    document.getElementById('list-1-body').innerHTML = content.list1Html;
    document.getElementById('list-2-title').textContent = content.list2Title;
    document.getElementById('list-2-body').innerHTML = content.list2Html;

    if (shouldReload) {{
      try {{
        const url = new URL(window.top.location.href);
        url.searchParams.set('module', moduleKey);
        window.top.location.href = url.toString();
      }} catch (error) {{
        const url = new URL(window.parent.location.href);
        url.searchParams.set('module', moduleKey);
        window.parent.location.href = url.toString();
      }}
    }}
  }}

  document.querySelectorAll('.module-tab').forEach((button) => {{
    button.addEventListener('click', () => setModule(button.dataset.module, true));
  }});

  function submitDashboardSearch(event) {{
    event.preventDefault();
    const input = event.target.querySelector('input[name="flight"]');
    const value = input ? input.value.trim() : '';
    try {{
      const url = new URL(window.top.location.href);
      if (activeModule) {{
        url.searchParams.set('module', activeModule);
      }}
      if (value) {{
        url.searchParams.set('flight', value);
      }} else {{
        url.searchParams.delete('flight');
      }}
      url.searchParams.delete('airport');
      window.top.location.href = url.toString();
    }} catch (error) {{
      const url = new URL(window.parent.location.href);
      if (activeModule) {{
        url.searchParams.set('module', activeModule);
      }}
      if (value) {{
        url.searchParams.set('flight', value);
      }} else {{
        url.searchParams.delete('flight');
      }}
      url.searchParams.delete('airport');
      window.parent.location.href = url.toString();
    }}
    return false;
  }}

  setModule(activeModule);
</script>
</html>
"""


query_flight = st.query_params.get("flight", "")
active_module = st.query_params.get("module", "live_ops")
selected_airport = st.query_params.get("airport", "")
_autorefresh_param = st.query_params.get("autorefresh", "1")
st.session_state["_autorefresh_paused"] = str(_autorefresh_param) == "0"
if isinstance(query_flight, list):
    query_flight = query_flight[0] if query_flight else ""
if isinstance(active_module, list):
    active_module = active_module[0] if active_module else "live_ops"
if isinstance(selected_airport, list):
    selected_airport = selected_airport[0] if selected_airport else ""
_VALID_MODULES = frozenset({
    "live_ops", "delay_prediction", "airport_traffic",
    "route_intelligence", "passenger_view", "flight_board", "alerts",
})
active_module = (active_module or "live_ops").strip() or "live_ops"
if active_module not in _VALID_MODULES:
    active_module = "live_ops"
selected_airport = (selected_airport or "").strip().upper()

selected_region = "india"
settings = REGION_MAP_SETTINGS[selected_region]
airports = get_airports_for_region(selected_region)

# ── Startup health check ──────────────────────────────────────────────────────
_missing_keys = []
if not os.getenv("AIRLABS_API_KEY"):
    _missing_keys.append("AIRLABS_API_KEY")
if _missing_keys:
    components.html(
        f"""
        <div style="font-family:'SF Pro Display','Segoe UI',Arial,sans-serif;
                    background:linear-gradient(135deg,#0d1520,#131c2a);
                    border:1px solid rgba(255,90,90,0.35);border-radius:18px;
                    padding:32px 36px;max-width:640px;margin:60px auto;
                    box-shadow:0 20px 60px rgba(0,0,0,0.55);">
          <div style="font-size:22px;font-weight:800;color:#ff6557;margin-bottom:12px;">
            ⚠ Missing Environment Variables
          </div>
          <p style="color:rgba(243,247,251,0.78);font-size:15px;line-height:1.6;margin:0 0 18px;">
            The following required API keys are not set:
            <strong style="color:#ffcf34;">{', '.join(_missing_keys)}</strong>
          </p>
          <p style="color:rgba(243,247,251,0.58);font-size:13px;margin:0;">
            Create a <code style="background:rgba(255,255,255,0.08);padding:2px 6px;border-radius:5px;">.env</code>
            file in the project root with the missing keys.
            See <code>.env.example</code> for the required format.
          </p>
        </div>
        """,
        height=260,
    )
    st.stop()

all_flights_df = fetch_flights(selected_region)

if all_flights_df.empty:
    st.error("No flight data returned. The AirLabs API monthly quota may be exhausted, or the API key may be invalid. Check your AIRLABS_API_KEY and try again.")
    st.info("💡 The free AirLabs tier resets on the 1st of each month.")
    st.stop()

flight_tracker.record(all_flights_df)

# Filter to airborne flights only for meaningful KPI averages
_airborne_mask = all_flights_df["altitude_ft"].fillna(0) > 1000
avg_speed_value = all_flights_df.loc[_airborne_mask, "speed_kts"].dropna()
avg_altitude_value = all_flights_df.loc[_airborne_mask, "altitude_ft"].dropna()
avg_speed = f"{avg_speed_value.mean():.0f} kts" if not avg_speed_value.empty else "N/A"
avg_altitude = f"{avg_altitude_value.mean():,.0f} ft" if not avg_altitude_value.empty else "N/A"
total_airlines = int(all_flights_df["airline_iata"].fillna("N/A").replace("N/A", pd.NA).dropna().nunique())

airport_metrics = compute_airport_traffic_metrics(all_flights_df, airports)
route_metrics = compute_route_congestion(all_flights_df, top_n=5)

top_airport_row = airport_metrics.iloc[0] if not airport_metrics.empty else None
top_airport = str(top_airport_row["airport_iata"]) if top_airport_row is not None else "DEL"
top_airport_volume = int(top_airport_row["nearby_count"]) if top_airport_row is not None else 0

pre_selected_matches = _find_selected_flight(all_flights_df, query_flight)
pre_selected_row = pre_selected_matches.iloc[0] if not pre_selected_matches.empty else all_flights_df.iloc[0]


_need_predictions = active_module in {
    "delay_prediction", "passenger_view", "alerts", "live_ops",
}
_need_schedule    = True  
_need_weather     = active_module in {"delay_prediction", "passenger_view", "alerts"}
_need_anomaly     = active_module in {"alerts"}


_TOP_SCHEDULE_AIRPORTS = [
    "DEL", "BOM", "BLR", "HYD", "MAA", "CCU", "AMD", "PNQ", "GOI", "COK",
    "GAU", "IXB", "LKO", "BBI", "VTZ", "IXC", "ATQ", "JAI", "BHO", "NAG",
]
schedule_airports: list[str] = list(_TOP_SCHEDULE_AIRPORTS)

# Promote the selected flight's origin/destination and the busiest live airport
# to the front so their schedule pressure is always resolved.
for code in [
    _safe_text(pre_selected_row.get("arr_iata"), ""),
    _safe_text(pre_selected_row.get("dep_iata"), ""),
    top_airport,
]:
    if code and code != "N/A" and code not in schedule_airports:
        schedule_airports.insert(0, code)

airport_schedule_pressure: dict[str, dict] = {}
now_ts = int(time.time())


flight_schedule_lookup: dict[str, dict] = {}
if _need_schedule:
    _SCHED_BUDGET = 10
    _fetch_airports = schedule_airports[:_SCHED_BUDGET]

    def _fetch_one_airport(airport_code: str) -> tuple[str, pd.DataFrame, pd.DataFrame]:
        arr, dep = _fetch_airport_schedules_cached(airport_code)
        return airport_code, arr, dep

    _sched_results: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as _pool:
        _sched_results = list(_pool.map(_fetch_one_airport, _fetch_airports))

    for airport_code, arrivals_df, departures_df in _sched_results:
        if arrivals_df.empty and departures_df.empty:
            airport_schedule_pressure[airport_code] = {
                "arrivals_next_hour": 0,
                "departures_next_hour": 0,
                "arrivals_next_3h": 0,
                "departures_next_3h": 0,
                "pressure_level": "Unknown",
                "pressure_score": 0.0,
                "arrivals_h0": 0, "arrivals_h1": 0, "arrivals_h2": 0,
                "departures_h0": 0, "departures_h1": 0, "departures_h2": 0,
            }
            continue
        airport_schedule_pressure[airport_code] = compute_airport_schedule_pressure(arrivals_df, departures_df, now_ts)

        for sched_df in (arrivals_df, departures_df):
            if sched_df.empty or "flight_iata" not in sched_df.columns:
                continue
            for _, srow in sched_df.iterrows():
                fid = _safe_text(srow.get("flight_iata"), "")
                if not fid or fid == "N/A":
                    continue
                if fid in flight_schedule_lookup:
                    continue

                _arr_raw = srow.get("arr_delayed")
                _dep_raw = srow.get("delayed")
                try:
                    _delayed_int = int(_arr_raw) if pd.notna(_arr_raw) else (
                        int(_dep_raw) if pd.notna(_dep_raw) else None
                    )
                except (TypeError, ValueError):
                    _delayed_int = None

                _est = srow.get("arr_estimated_ts")
                _sch = srow.get("arr_time_ts")
                try:
                    _ts_delay = int((float(_est) - float(_sch)) / 60) if pd.notna(_est) and pd.notna(_sch) else None
                except (TypeError, ValueError):
                    _ts_delay = None

                _eff_delay = _delayed_int if _delayed_int else (_ts_delay if _ts_delay and _ts_delay > 0 else None)
                flight_schedule_lookup[fid] = {
                    "std_ts":     int(srow["dep_time_ts"])      if pd.notna(srow.get("dep_time_ts"))      else None,
                    "sta_ts":     int(srow["arr_time_ts"])      if pd.notna(srow.get("arr_time_ts"))      else None,
                    "eta_ts":     int(srow["arr_estimated_ts"]) if pd.notna(srow.get("arr_estimated_ts")) else None,
                    "delayed_min": _eff_delay,
                }

top_airport_pressure = airport_schedule_pressure.get(top_airport, {}).get("pressure_level", "Unknown")

top_route_row = route_metrics.iloc[0] if not route_metrics.empty else None
top_route = str(top_route_row["route_key"]) if top_route_row is not None else "DEL-BOM"
top_route_volume = int(top_route_row["flight_count"]) if top_route_row is not None else 0
top_route_score = f"{float(top_route_row['congestion_score']):.1f}" if top_route_row is not None else "0.0"

top5_airports_html = ""
if not airport_metrics.empty:
    top5_airports_html += (
        "<li>"
        "<a href='?module=airport_traffic' "
        "onclick=\"try { const url = new URL(window.top.location.href); url.searchParams.set('module','airport_traffic'); url.searchParams.delete('airport'); window.top.location.href = url.toString(); } catch (e) { const url = new URL(window.parent.location.href); url.searchParams.set('module','airport_traffic'); url.searchParams.delete('airport'); window.parent.location.href = url.toString(); } return false;\">All Airports</a>"
        "<span>reset filter</span></li>"
    )
    for _, row in airport_metrics.head(5).iterrows():
        code = str(row["airport_iata"])
        label = html.escape(code)
        score = f"{float(row['congestion_score']):.1f}"
        flights = int(row["nearby_count"])
        top5_airports_html += (
            "<li>"
            f"<a href='?module=airport_traffic&airport={label}' "
            f"onclick=\"try {{ const url = new URL(window.top.location.href); url.searchParams.set('module','airport_traffic'); url.searchParams.set('airport','{label}'); window.top.location.href = url.toString(); }} catch (e) {{ const url = new URL(window.parent.location.href); url.searchParams.set('module','airport_traffic'); url.searchParams.set('airport','{label}'); window.parent.location.href = url.toString(); }} return false;\">{label}</a>"
            f"<span>{flights} flights • {score}</span></li>"
        )
else:
    top5_airports_html = "<li>N/A<span>No data</span></li>"

top5_routes_html = ""
if not route_metrics.empty:
    for _, row in route_metrics.head(5).iterrows():
        label = html.escape(str(row["route_key"]))
        score = f"{float(row['congestion_score']):.1f}"
        flights = int(row["flight_count"])
        top5_routes_html += f"<li>{label}<span>{flights} flights • {score}</span></li>"
else:
    top5_routes_html = "<li>N/A<span>No data</span></li>"

most_delayed_html = ""


_airport_by_iata = {a["iata"]: a for a in airports}
_WX_MAX = 8
_top_weather_iatas: list[str] = []
if _need_weather:
    _busy = airport_metrics.head(8)["airport_iata"].tolist() if not airport_metrics.empty else []
    _live_dests: list[str] = []
    if not all_flights_df.empty and "arr_iata" in all_flights_df.columns:
        _live_dests = (
            all_flights_df["arr_iata"]
            .dropna().astype(str).str.strip().str.upper()
            .replace("N/A", pd.NA).dropna().unique().tolist()
        )
    # Always include the selected flight's destination so the per-flight
    # fallback call below is guaranteed to hit the cache instead of the API.
    _sel_dest_wx = _safe_text(pre_selected_row.get("arr_iata"), "").upper()
    _priority = [_sel_dest_wx] if _sel_dest_wx and _sel_dest_wx != "N/A" else []
    _top_weather_iatas = list(dict.fromkeys(_priority + _busy + _live_dests))[:_WX_MAX + 1]

_weather_map: dict[str, "WeatherImpact | None"] = {}
if _top_weather_iatas:
    def _fetch_wx(iata: str) -> tuple[str, "WeatherImpact | None"]:
        _ap = _airport_by_iata.get(iata)
        if not _ap:
            return iata, None
        try:
            return iata, get_nearest_airport_weather(_ap["lat"], _ap["lng"], iata_hint=iata)
        except Exception:
            return iata, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as _wx_pool:
        for _iata, _wi in _wx_pool.map(_fetch_wx, _top_weather_iatas):
            _weather_map[_iata] = _wi

def _sev_from_cache(iata: str) -> str:
    if not _need_weather:
        return "Low"
    if iata not in _weather_map:
        return "Unknown"   # not fetched — mark clearly, no false Low
    wi = _weather_map.get(iata)
    return wi.severity if wi else "Unknown"


if "updated" in all_flights_df.columns:
    _updated_ts = pd.to_numeric(all_flights_df["updated"], errors="coerce")
    all_flights_df["stale_minutes"] = (_updated_ts.apply(
        lambda ts: round((now_ts - ts) / 60, 1) if pd.notna(ts) and ts > 0 else float("nan")
    ))
else:
    all_flights_df["stale_minutes"] = float("nan")


all_flights_df["weather_severity"] = (
    all_flights_df["arr_iata"].fillna("").map(_sev_from_cache)
    if _need_weather else "Low"
)


all_flights_df["dep_delayed"] = (
    all_flights_df["flight_iata"].map(
        lambda f: (flight_schedule_lookup.get(f) or {}).get("delayed_min") or float("nan")
    )
    if _need_schedule else float("nan")
)

if _need_predictions:
    all_flights_df = enrich_flights_with_predictions(
        all_flights_df,
        airports=airports,
        airport_metrics=airport_metrics,
        airport_schedule_pressure=airport_schedule_pressure,
        now_ts=now_ts,
    )


anomaly_df: "pd.DataFrame" = pd.DataFrame()
if _need_anomaly:
    trails_for_anomaly = flight_tracker.get_trails()
    anomaly_df = detect_all_anomalies(all_flights_df, trails_for_anomaly, airports)

most_delayed_df = (
    all_flights_df.sort_values(by="predicted_delay_min", ascending=False).head(5)
    if _need_predictions and "predicted_delay_min" in all_flights_df.columns
    else pd.DataFrame()
)
if not most_delayed_df.empty:
    for _, row in most_delayed_df.iterrows():
        flight_id = html.escape(str(row.get("flight_iata", "N/A")))
        delay_min = int(pd.to_numeric(row.get("predicted_delay_min"), errors="coerce") or 0)
        most_delayed_html += f"<li>{flight_id}<span>+{delay_min} min</span></li>"
else:
    most_delayed_html = "<li>N/A<span>No data</span></li>"


top_airlines_list: list[dict] = []
if not all_flights_df.empty and "airline_iata" in all_flights_df.columns:
    _al_counts = all_flights_df["airline_iata"].value_counts().head(8)
    for _al_code, _al_count in _al_counts.items():
        _code_str = str(_al_code)
        if _code_str in ("N/A", "", "nan"):
            continue
        top_airlines_list.append({
            "code": _code_str,
            "name": get_airline_name(_code_str),
            "count": int(_al_count),
        })

highlight_flights = set(
  most_delayed_df["flight_iata"].fillna("N/A").astype(str).tolist()
  if "flight_iata" in most_delayed_df.columns
  else []
)

display_df = all_flights_df
if active_module == "airport_traffic" and selected_airport:
    display_df = all_flights_df[
        (all_flights_df["dep_iata"].fillna("") == selected_airport)
        | (all_flights_df["arr_iata"].fillna("") == selected_airport)
    ]

airport_index_value = None
airport_selected_count = None
if not airport_metrics.empty:
    max_score = float(airport_metrics["congestion_score"].max() or 1.0)
    if selected_airport:
        selected_row_airport = airport_metrics[airport_metrics["airport_iata"] == selected_airport]
        if not selected_row_airport.empty:
            score = float(selected_row_airport.iloc[0]["congestion_score"])
            airport_index_value = round((score / max_score) * 100)
            airport_selected_count = len(display_df)
    if airport_index_value is None:
        score = float(top_airport_row["congestion_score"]) if top_airport_row is not None else 0.0
        airport_index_value = round((score / max_score) * 100) if max_score else 0
        airport_selected_count = top_airport_volume
airport_focus_code = selected_airport if selected_airport else top_airport
airport_index_text = f"{airport_index_value}" if airport_index_value is not None else "0"
airport_focus_count = airport_selected_count if airport_selected_count is not None else top_airport_volume

selection_df = display_df if not display_df.empty else all_flights_df
selected_matches = _find_selected_flight(selection_df, query_flight)
search_has_match = not selected_matches.empty
selected_row = selected_matches.iloc[0] if not selected_matches.empty else selection_df.iloc[0]
selected_flight = _safe_text(query_flight, "") if not query_flight else _safe_text(query_flight, _safe_text(selected_row.get("flight_iata")))
has_user_selection = bool(query_flight) and search_has_match

selected_route_key = f"{_safe_text(selected_row.get('dep_iata'))}-{_safe_text(selected_row.get('arr_iata'))}"
selected_route_row = route_metrics[route_metrics["route_key"] == selected_route_key]
route_focus_html = ""
if not selected_route_row.empty:
    row = selected_route_row.iloc[0]
    route_focus_html = (
        f"<li>{html.escape(selected_route_key)}"
        f"<span>{int(row['flight_count'])} flights • {float(row['congestion_score']):.1f}</span></li>"
    )
elif has_user_selection:
    route_focus_html = f"<li>{html.escape(selected_route_key)}<span>1 flight</span></li>"
else:
    route_focus_html = top5_routes_html

selected_delay_html = ""
if has_user_selection:
    delay_min = int(pd.to_numeric(selected_row.get("predicted_delay_min"), errors="coerce") or 0)
    selected_delay_html = f"<li>{html.escape(selected_flight)}<span>+{delay_min} min</span></li>"
delay_list_html = selected_delay_html if has_user_selection else most_delayed_html
route_list_html = route_focus_html if has_user_selection else top5_routes_html

if active_module in {"delay_prediction", "passenger_view"} and has_user_selection:
    display_df = selected_matches.head(1)
elif active_module == "route_intelligence" and has_user_selection:
    display_df = all_flights_df[
        (all_flights_df["dep_iata"].fillna("") == _safe_text(selected_row.get("dep_iata")))
        & (all_flights_df["arr_iata"].fillna("") == _safe_text(selected_row.get("arr_iata")))
    ]

nearest_airport = _nearest_airport(selected_row.get("latitude"), selected_row.get("longitude"), airports)


_dest_iata = _safe_text(selected_row.get("arr_iata"), "")
_dest_airport = _airport_by_iata.get(_dest_iata) if _dest_iata and _dest_iata != "N/A" else None
weather_impact = None
_weather_lookup_airport = _dest_airport or nearest_airport  # dest preferred; fallback to nearest
if _weather_lookup_airport:
    
    _cached_wi = _weather_map.get(_dest_iata) if _dest_iata else None
    if _cached_wi is not None:
        weather_impact = _cached_wi
    else:
        try:
            weather_impact = get_nearest_airport_weather(
                _weather_lookup_airport["lat"], _weather_lookup_airport["lng"],
                iata_hint=_dest_iata if _dest_iata and _dest_iata != "N/A" else None,
            )
        except Exception:
            weather_impact = None

_flight_iata_for_history = str(selected_row.get("flight_iata") or "").strip()
_flight_history = []
if _flight_iata_for_history:
    try:
        _flight_history = snapshot_store.get_single_flight_history(_flight_iata_for_history, hours=1)
    except Exception:
        _flight_history = []

delay_prediction = compute_delay_prediction(
    selected_row,
    airports=airports,
    airport_metrics=airport_metrics,
    airport_schedule_pressure=airport_schedule_pressure,
    now_ts=now_ts,
    weather_impact=weather_impact,
    flight_history=_flight_history,
)

delay_risk = delay_prediction.risk_level
# Weather is already scored inside compute_delay_prediction via weather_severity
# column — do NOT add weather_penalty again (was causing double-counting)
delay_minutes_total = delay_prediction.expected_delay_min
expected_delay = f"+{delay_minutes_total} min"
delay_reasons = ", ".join(delay_prediction.reason_tags[:2])
if weather_impact and weather_impact.severity != "Low":
    _weather_tag = weather_impact.severity.lower() + " weather"
    if _weather_tag not in delay_reasons:
        delay_reasons = ", ".join(filter(None, [delay_reasons, _weather_tag]))
delay_reason_chips = _build_reason_chips(delay_reasons)
destination_pressure = (
    f"{delay_prediction.congestion_level} live load • {delay_prediction.schedule_pressure_level} schedule pressure"
)
arrival_estimate = compute_arrival_estimate(
    selected_row,
    airports=airports,
    delay_minutes=delay_minutes_total,
    now_ts=now_ts,
)
expected_arrival = arrival_estimate.eta_label_utc
passenger_summary = arrival_estimate.arrival_summary
arrival_distance = (
    f"{arrival_estimate.distance_km:.0f} km remaining"
    if arrival_estimate.distance_km is not None
    else "Distance unavailable"
)
weather_severity = weather_impact.severity if weather_impact else "Low"
weather_summary = weather_impact.summary if weather_impact else "stable airport weather"
weather_station = weather_impact.station_icao if weather_impact else "No CheckWX station"

if snapshot_store.should_record():
    _snap_flights = all_flights_df.copy()
    _snap_airports = airport_metrics.copy()
    _snap_routes = route_metrics.copy()
    _snap_pressure = dict(airport_schedule_pressure)
    _snap_ts = now_ts

    def _do_snapshot() -> None:
        try:
            snapshot_store.record_flights(_snap_flights, ts=_snap_ts)
            snapshot_store.record_airports(_snap_airports, _snap_pressure, ts=_snap_ts)
            snapshot_store.record_routes(_snap_routes, ts=_snap_ts)
            snapshot_store.purge_old_snapshots(retain_days=7)
        except Exception as _e:
            logging.getLogger(__name__).warning("Snapshot write failed: %s", _e)

    threading.Thread(target=_do_snapshot, daemon=True).start()

mapbox_token = os.getenv("MAPBOX_TOKEN") or os.getenv("CESIUM_TOKEN") or ""
use_mapbox = mapbox_token.startswith("pk.")
mapbox_df = (
    display_df
    if active_module in {"airport_traffic", "route_intelligence"} and not display_df.empty
    else all_flights_df
)

# ── Store live-push config so the fragment can access it on autonomous re-runs ─
st.session_state["_live_region"] = selected_region
st.session_state["_live_sched"] = flight_schedule_lookup
st.session_state["_live_sel_flight"] = selected_flight if search_has_match else ""
st.session_state["_live_module"] = active_module
st.session_state["_live_airports"] = airports
st.session_state["_live_airport_metrics"] = airport_metrics
st.session_state["_live_schedule_pressure"] = airport_schedule_pressure

if use_mapbox:
    # ── Render the map once and cache it — the map iframe persists across
    # ── subsequent Streamlit reruns (fragment or user interaction), avoiding
    # ── full iframe destruction/recreation.
    _map_cache_key = f"{active_module}|{selected_region}|{settings['zoom']}"
    if st.session_state.get("_map_base_html_key") != _map_cache_key:
        # Module / region changed (or first load) → bake fresh HTML
        mapbox_html = generate_mapbox_base_html(
            df=mapbox_df,
            mapbox_token=mapbox_token,
            selected_flight=selected_flight if search_has_match else None,
            active_module=active_module,
            region=selected_region,
            center_lat=settings["center"][0],
            center_lng=settings["center"][1],
            zoom_level=settings["zoom"],
            anomaly_df=anomaly_df,
            schedule_lookup=flight_schedule_lookup,
            airport_metrics=airport_metrics,
            schedule_pressure=airport_schedule_pressure,
            top_airlines=top_airlines_list,
            height=DASHBOARD_HEIGHT,
        )
        st.session_state["_map_base_html"] = mapbox_html
        st.session_state["_map_base_html_key"] = _map_cache_key
    else:
        mapbox_html = st.session_state["_map_base_html"]

    components.html(mapbox_html, height=DASHBOARD_HEIGHT, scrolling=False)

    # ── Live position update fragment ─────────────────────────────────────────
    # Runs autonomously every 10 s without touching the map iframe.
    # Fetches fresh flight positions and pushes them via the JS bridge.
    _autorefresh_paused = st.session_state.get("_autorefresh_paused", False)
    _fragment_run_every = None if _autorefresh_paused else 10

    @st.fragment(run_every=_fragment_run_every)
    def _live_push() -> None:
        _region   = st.session_state.get("_live_region",            "india")
        _sched    = st.session_state.get("_live_sched",             {})
        _sel      = st.session_state.get("_live_sel_flight",        "")
        _airports = st.session_state.get("_live_airports",          [])
        _metrics  = st.session_state.get("_live_airport_metrics",   pd.DataFrame())
        _pressure = st.session_state.get("_live_schedule_pressure", {})

        _tick    = int(time.time()) // 10   # 10-s bucket → cache bust key
        fresh_df = _fetch_live_positions(_region, _tick)
        if fresh_df.empty:
            return

        # Run the same enrichment pipeline as the main page so pred_delay_min
        # is populated — without it the delayed-flights list would show zeros.
        try:
            fresh_df = enrich_flights_with_predictions(
                fresh_df,
                airports=_airports,
                airport_metrics=_metrics,
                airport_schedule_pressure=_pressure,
                now_ts=int(time.time()),
            )
        except Exception:
            pass  # fall back to unenriched data rather than crashing

        fresh_json = build_flights_json(fresh_df, _sel, _sched, region=_region)
        _push = f"""<script>(function(){{
  var _roots=[window.top,window.parent];
  var _found=false;
  for(var r=0;r<_roots.length&&!_found;r++){{
    try{{
      var f=_roots[r].frames;
      for(var i=0;i<f.length;i++){{
        try{{if(f[i].AVIATION_MAP_FRAME){{f[i].updateFlightData({fresh_json});_found=true;break;}}}}catch(e){{}}
      }}
    }}catch(e){{}}
  }}
}})();</script>"""
        components.html(_push, height=0)

    _live_push()

else:
    flight_map = create_map(
        df=display_df,
        region_center=settings["center"],
        zoom=settings["zoom"],
        show_trails=False,
        selected_flight=selected_flight,
        region=selected_region,
        clean_ui=True,
        cluster_markers=False,
        enable_user_location=True,
        show_delay_fields=active_module in {"delay_prediction", "passenger_view"},
        highlight_flights=highlight_flights if active_module == "delay_prediction" else None,
        module_param=active_module,
    )

    map_embed_html = flight_map._repr_html_()

    _dashboard_html = _build_dashboard_html(DashboardContext(
        map_embed_html=map_embed_html,
        selected_row=selected_row,
        selected_flight=selected_flight,
        search_has_match=search_has_match,
        airport_count=len(airports),
        flight_count=len(all_flights_df),
        avg_speed=avg_speed,
        avg_altitude=avg_altitude,
        total_airlines=total_airlines,
        top_airport=top_airport,
        top_airport_volume=top_airport_volume,
        top_airport_pressure=top_airport_pressure,
        top_route=top_route,
        top_route_volume=top_route_volume,
        top_route_score=top_route_score,
        nearest_airport=nearest_airport,
        delay_risk=delay_risk,
        expected_delay=expected_delay,
        delay_reasons=delay_reasons,
        destination_pressure=destination_pressure,
        expected_arrival=expected_arrival,
        passenger_summary=passenger_summary,
        arrival_distance=arrival_distance,
        weather_severity=weather_severity,
        weather_summary=weather_summary,
        weather_station=weather_station,
        top5_airports_html=top5_airports_html,
        top5_routes_html=top5_routes_html,
        delay_list_html=delay_list_html,
        route_list_html=route_list_html,
        delay_reason_chips=delay_reason_chips,
        airport_focus_code=airport_focus_code,
        airport_focus_count=airport_focus_count,
        airport_index_text=airport_index_text,
        autorefresh_paused=st.session_state.get("_autorefresh_paused", False),
        flights_csv=all_flights_df.to_csv(index=False),
        anomaly_count=len(anomaly_df),
    ))
    components.html(_dashboard_html, height=DASHBOARD_HEIGHT, scrolling=False)
