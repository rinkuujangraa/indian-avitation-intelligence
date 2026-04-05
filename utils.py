"""
utils.py
--------
Builds a professional interactive Folium map.

What's in here:
  - Type-accurate aircraft silhouettes via aircraft_icons.py (B747, A380, A320, etc.)
  - Icons rotate to match aircraft heading
  - Indian airport pinpoints with IATA labels
  - Rich popup: flight, airline, route, aircraft type, altitude, speed, heading, status
  - Selected flight highlight (red, larger icon)
  - Altitude colour coding with proper NaN safety
  - Satellite basemap, minimap, fullscreen, legend
"""

from urllib.parse import quote

import folium
from folium.plugins import MarkerCluster, MiniMap, Fullscreen, AntPath
import pandas as pd
import logging

from data_fetcher import get_airline_name, get_airports_for_region
from tracker import tracker as flight_tracker
from aircraft_icons import get_aircraft_icon_html, get_family

logger = logging.getLogger(__name__)


# ── Shared text/search helpers (canonical — all modules import from here) ──────

def safe_text(value: object, fallback: str = "N/A") -> str:
    """Convert any value to a non-empty string, returning *fallback* on None/NaN/blank."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return fallback
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return fallback
    return text


def normalize_flight_query(value: object) -> str:
    """Upper-case, whitespace-stripped flight/IATA code for fuzzy search."""
    return "".join(str(value or "").upper().split())


def is_selected_flight(flight_value: object, selected_flight: str) -> bool:
    """True if *flight_value* matches or contains the *selected_flight* query."""
    norm_sel = normalize_flight_query(selected_flight)
    norm_val = normalize_flight_query(flight_value)
    if not norm_sel or not norm_val:
        return False
    return norm_val == norm_sel or norm_sel in norm_val


# ── Aircraft family → icon size (px) ──────────────────────────────────────────
# Heavier aircraft get bigger icons so they stand out visually,
# just like FlightRadar24 scales heavies vs regionals.
FAMILY_SIZE = {
    "b747": 38, "a380": 40, "a340": 36,
    "b777": 36, "a350": 34, "a330": 34,
    "b787": 34, "b767": 32, "b757": 30,
    "a320": 28, "b737": 28, "e190": 24,
    "crj":  22, "atr":  22, "concorde": 32,
    "military": 36, "default": 26,
}

FAMILY_LABEL = {
    "b747": "Boeing 747 (4-engine)",
    "a380": "Airbus A380 (4-engine, double deck)",
    "a340": "Airbus A340 (4-engine)",
    "b777": "Boeing 777 (widebody)",
    "a350": "Airbus A350 (widebody)",
    "a330": "Airbus A330 (widebody)",
    "b787": "Boeing 787 Dreamliner",
    "b767": "Boeing 767 (widebody)",
    "b757": "Boeing 757",
    "a320": "Airbus A320 family",
    "b737": "Boeing 737 family",
    "e190": "Embraer E-jet",
    "crj":  "Bombardier CRJ",
    "atr":  "ATR / Turboprop",
    "concorde": "Concorde",
    "military": "Military / Cargo",
    "default":  "Unknown type",
}


# ── Altitude colour ────────────────────────────────────────────────────────────
def _get_altitude_color(altitude_ft, highlight: bool = False) -> str:
    """
    Return hex colour based on altitude band.
    highlight=True → always red (selected/tracked flight).
    pd.isna() check MUST come before any numeric comparison to avoid
    silent NaN comparison bugs.
    """
    if highlight:
        return "#FF3B30"

    if pd.isna(altitude_ft):
        return "#09D2E0"

    alt = float(altitude_ft)
    if alt < 1_000:
        return "#09D2E0"   # grey   — ground / very low
    elif alt < 10_000:
        return "#2ecc71"   # green  — climbing / descending
    elif alt < 25_000:
        return "#f39c12"   # orange — mid level
    else:
        return "#e74c3c"   # red    — cruise altitude


# ── Rich popup ─────────────────────────────────────────────────────────────────
def _build_popup(
    row: pd.Series,
    highlight: bool = False,
    show_delay_fields: bool = True,
    module_param: str | None = None,
) -> str:
    """
    Build a professional HTML popup card with all AirLabs fields.
    """

    def safe_text(value, fallback="N/A"):
        if value is None or pd.isna(value):
            return fallback
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return fallback
        return text

    def fmt_alt(v):
        return f"{int(v):,} ft" if pd.notna(v) else "N/A"

    def fmt_spd(v):
        return f"{float(v):.1f} kts" if pd.notna(v) else "N/A"

    def fmt_hdg(v):
        return f"{float(v):.0f}°" if pd.notna(v) else "N/A"

    altitude = fmt_alt(row.get("altitude_ft"))
    speed    = fmt_spd(row.get("speed_kts"))
    heading  = fmt_hdg(row.get("heading"))
    predicted_delay_min = pd.to_numeric(pd.Series([row.get("predicted_delay_min")]), errors="coerce").iloc[0]
    predicted_delay_risk = safe_text(row.get("predicted_delay_risk"), "Low")
    predicted_eta_utc = safe_text(row.get("predicted_eta_utc"), "ETA unavailable")
    predicted_reason = safe_text(row.get("predicted_delay_reason"), "stable profile")

    v = row.get("v_speed")
    if pd.notna(v):
        v = float(v)
        vstatus = "⬆ Climbing" if v > 1 else "⬇ Descending" if v < -1 else "➡ Level"
    else:
        vstatus = "—"

    airline_code = safe_text(row.get("airline_iata"))
    airline_name = get_airline_name(airline_code) if airline_code != "N/A" else "N/A"
    flight_num   = safe_text(row.get("flight_iata"))
    dep          = safe_text(row.get("dep_iata"))
    arr          = safe_text(row.get("arr_iata"))
    route        = f"{dep} → {arr}" if dep != "N/A" and arr != "N/A" else "N/A"
    aircraft     = safe_text(row.get("aircraft_icao"))
    reg          = safe_text(row.get("reg_number"))
    hex_code     = safe_text(row.get("hex"))
    flt_status   = safe_text(row.get("status"), "unknown").replace("-", " ").title()
    family       = get_family(aircraft)
    type_label   = FAMILY_LABEL.get(family, aircraft)
    track_target = quote(flight_num) if flight_num != "N/A" else ""
    module_param = (module_param or "").strip()
    module_query = f"&module={quote(module_param)}" if module_param else ""
    module_js = f"url.searchParams.set('module', '{module_param}');" if module_param else ""

    accent = "#ff6156" if highlight else "#59b7ff"
    badge  = ('<span style="background:#ff6156;color:white;font-size:10px;'
              'padding:4px 8px;border-radius:999px;margin-left:8px;letter-spacing:0.08em;">TRACKED</span>'
              if highlight else "")

    return f"""
    <div style="
      min-width:250px;
      max-width:272px;
      font-family:Arial,sans-serif;
      color:#eef4fb;
      background:linear-gradient(180deg, rgba(12,18,28,0.96), rgba(10,15,24,0.94));
      border:1px solid rgba(255,255,255,0.08);
      border-radius:16px;
      padding:12px;
      box-shadow:0 16px 34px rgba(0,0,0,0.24);
    ">
      <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:8px;margin-bottom:8px;">
        <div>
          <div style="display:flex;align-items:center;flex-wrap:wrap;gap:0;">
            <span style="font-size:21px;font-weight:800;letter-spacing:-0.03em;color:#f7fbff;">{flight_num}</span>
            {badge}
          </div>
          <div style="margin-top:3px;font-size:12px;color:rgba(238,244,251,0.70);font-weight:600;">
            {airline_name}
          </div>
        </div>
        <div style="
          padding:5px 8px;
          border-radius:999px;
          background:rgba(89,183,255,0.14);
          color:{accent};
          font-size:9px;
          font-weight:800;
          letter-spacing:0.10em;
          text-transform:uppercase;
          white-space:nowrap;
        ">{flt_status}</div>
      </div>

      <div style="
        margin-bottom:10px;
        padding:10px 10px 9px;
        border-radius:12px;
        background:rgba(255,255,255,0.04);
        border:1px solid rgba(255,255,255,0.06);
      ">
        <div style="font-size:10px;letter-spacing:0.10em;text-transform:uppercase;color:rgba(238,244,251,0.48);margin-bottom:6px;">
          Route
        </div>
        <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;">
          <div>
            <div style="font-size:24px;font-weight:800;line-height:1;color:#f7fbff;">{dep}</div>
            <div style="font-size:10px;color:rgba(238,244,251,0.56);margin-top:3px;">Departure</div>
          </div>
          <div style="font-size:16px;color:{accent};">→</div>
          <div style="text-align:right;">
            <div style="font-size:24px;font-weight:800;line-height:1;color:#f7fbff;">{arr}</div>
            <div style="font-size:10px;color:rgba(238,244,251,0.56);margin-top:3px;">Arrival</div>
          </div>
        </div>
      </div>

      <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:9px;">
        <div style="padding:8px 10px;border-radius:11px;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.05);">
          <div style="font-size:9px;letter-spacing:0.10em;text-transform:uppercase;color:rgba(238,244,251,0.46);margin-bottom:5px;">Altitude</div>
          <div style="font-size:16px;font-weight:800;color:#f7fbff;">{altitude}</div>
        </div>
        <div style="padding:8px 10px;border-radius:11px;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.05);">
          <div style="font-size:9px;letter-spacing:0.10em;text-transform:uppercase;color:rgba(238,244,251,0.46);margin-bottom:5px;">Speed</div>
          <div style="font-size:16px;font-weight:800;color:#f7fbff;">{speed}</div>
        </div>
      </div>

      {(
        f'''
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:9px;">
        <div style="padding:8px 10px;border-radius:11px;background:rgba(255,97,86,0.08);border:1px solid rgba(255,97,86,0.16);">
          <div style="font-size:9px;letter-spacing:0.10em;text-transform:uppercase;color:rgba(238,244,251,0.46);margin-bottom:5px;">Predicted Delay</div>
          <div style="font-size:16px;font-weight:800;color:#f7fbff;">+{int(predicted_delay_min) if pd.notna(predicted_delay_min) else 0} min</div>
          <div style="font-size:10px;color:rgba(238,244,251,0.56);margin-top:4px;">{predicted_delay_risk} risk</div>
        </div>
        <div style="padding:8px 10px;border-radius:11px;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.05);">
          <div style="font-size:9px;letter-spacing:0.10em;text-transform:uppercase;color:rgba(238,244,251,0.46);margin-bottom:5px;">Estimated Arrival</div>
          <div style="font-size:16px;font-weight:800;color:#f7fbff;">{predicted_eta_utc}</div>
          <div style="font-size:10px;color:rgba(238,244,251,0.56);margin-top:4px;">based on live position</div>
        </div>
      </div>
        ''' if show_delay_fields else ''
      )}

      <div style="display:grid;grid-template-columns:88px 1fr;row-gap:6px;column-gap:8px;font-size:11px;line-height:1.3;margin-bottom:12px;">
        <div style="color:rgba(238,244,251,0.50);">Aircraft</div>
        <div style="color:#f0f6fc;font-weight:700;">{aircraft} — {type_label}</div>
        <div style="color:rgba(238,244,251,0.50);">Registration</div>
        <div style="color:#f0f6fc;font-weight:700;">{reg}</div>
        <div style="color:rgba(238,244,251,0.50);">Heading</div>
        <div style="color:#f0f6fc;font-weight:700;">{heading}</div>
        <div style="color:rgba(238,244,251,0.50);">Vertical</div>
        <div style="color:#f0f6fc;font-weight:700;">{vstatus}</div>
        <div style="color:rgba(238,244,251,0.50);">ICAO24</div>
        <div style="color:#f0f6fc;font-weight:700;font-family:monospace;">{hex_code}</div>
        {(
          f'''
        <div style="color:rgba(238,244,251,0.50);">Delay Driver</div>
        <div style="color:#f0f6fc;font-weight:700;">{predicted_reason}</div>
          ''' if show_delay_fields else ''
        )}
      </div>

      <div style="display:flex;gap:8px;">
        <a href="?flight={track_target}{module_query}"
           onclick="try {{ const url = new URL(window.top.location.href); url.searchParams.set('flight', '{track_target}'); {module_js} window.top.location.href = url.toString(); }} catch (e) {{ const url = new URL(window.parent.location.href); url.searchParams.set('flight', '{track_target}'); {module_js} window.parent.location.href = url.toString(); }} return false;"
           style="
             flex:1;
             text-align:center;
             background:linear-gradient(180deg, {accent}, #2f7fc6);
             color:white;
             text-decoration:none;
             border-radius:11px;
             padding:9px 10px;
             font-size:12px;
             font-weight:800;
             letter-spacing:0.01em;
             box-shadow:inset 0 1px 0 rgba(255,255,255,0.18);
           ">
          Track Flight
        </a>
      </div>
    </div>"""


# ── Trail drawing ──────────────────────────────────────────────────────────────
def _draw_trails(flight_map: folium.Map, df: pd.DataFrame) -> None:
    """
    Draw animated AntPath trails underneath aircraft icons.
    Trails are added to the map FIRST so they render below icons.
    """
    trails      = flight_tracker.get_trails()
    trail_layer = folium.FeatureGroup(name="Flight trails", show=True)
    count       = 0

    for _, row in df.iterrows():
        hex_code = str(row.get("hex", ""))
        if not hex_code or hex_code not in trails:
            continue

        positions = trails[hex_code]["positions"]
        if len(positions) < 2:
            continue

        coords = [
            (p["lat"], p["lng"])
            for p in positions
            if p.get("lat") is not None and p.get("lng") is not None
        ]
        if len(coords) < 2:
            continue

        color = _get_altitude_color(row.get("altitude_ft"))

        try:
            AntPath(
                locations=coords,
                color=color,
                weight=2,
                opacity=0.75,
                delay=800,
                dash_array=[10, 20],
                pulse_color="#FFFFFF",
            ).add_to(trail_layer)
        except Exception:
            folium.PolyLine(
                locations=coords,
                color=color,
                weight=1.5,
                opacity=0.6,
                dash_array="6 10",
            ).add_to(trail_layer)
        count += 1

    trail_layer.add_to(flight_map)
    logger.info(f"Drew {count} flight trails")


# ── Main map builder ───────────────────────────────────────────────────────────
def create_map(
    df: pd.DataFrame,
    region_center: tuple = (20, 0),
    zoom: int = 2,
    show_trails: bool = True,
    selected_flight: str = None,
    region: str = "india",
    clean_ui: bool = False,
    cluster_markers: bool = True,
    enable_user_location: bool = True,
    show_delay_fields: bool = True,
    highlight_flights: set[str] | None = None,
    module_param: str | None = None,
) -> folium.Map:
    """
    Build a clean satellite Folium map with type-accurate
    aircraft icons, Indian airport pinpoints, and rich popups.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned AirLabs flight DataFrame.
    region_center : tuple
        (lat, lon) map centre.
    zoom : int
        Initial zoom level.
    show_trails : bool
        Retained for compatibility; trails are disabled in the app.
    selected_flight : str or None
        Flight IATA to highlight e.g. "EK203". Red icon, larger size.
    """
    if df.empty:
        logger.warning("Empty DataFrame — rendering empty map.")

    # ── Base map ───────────────────────────────────────────────────────────────
    flight_map = folium.Map(
        location=region_center,
        zoom_start=zoom,
        tiles=None,
        prefer_canvas=True,
        width="100%",
        height="100%",
        zoom_control=True,
    )

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
        overlay=False,
        control=False,
    ).add_to(flight_map)

    if not clean_ui:
        # ── Tile layers ────────────────────────────────────────────────────────
        folium.TileLayer("OpenStreetMap",       name="Street map").add_to(flight_map)
        folium.TileLayer("CartoDB dark_matter", name="Dark mode").add_to(flight_map)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri", name="Satellite",
        ).add_to(flight_map)

        # ── Plugins ────────────────────────────────────────────────────────────
        MiniMap(toggle_display=True, tile_layer="OpenStreetMap").add_to(flight_map)
        Fullscreen(position="topright").add_to(flight_map)
    else:
        flight_map.get_root().header.add_child(
            folium.Element(
                """
                <style>
                  html, body {
                    height: 100%;
                    margin: 0;
                  }
                  .folium-map {
                    height: 100% !important;
                  }
                  .leaflet-top.leaflet-left {
                    top: auto !important;
                    bottom: 22px !important;
                  }
                  .leaflet-control-zoom {
                    box-shadow: 0 10px 26px rgba(0,0,0,0.28) !important;
                    border: 1px solid rgba(255,255,255,0.10) !important;
                  }
                  .leaflet-control-zoom a {
                    width: 34px !important;
                    height: 34px !important;
                    line-height: 34px !important;
                    background: rgba(13, 19, 28, 0.88) !important;
                    color: #f3f7fb !important;
                    border-bottom-color: rgba(255,255,255,0.08) !important;
                  }
                </style>
                <script>
                  document.addEventListener('DOMContentLoaded', function() {
                    try {
                      const mapEl = document.querySelector('.folium-map');
                      if (mapEl) {
                        mapEl.style.height = '100%';
                      }
                      if (window._leaflet_map) {
                        window._leaflet_map.invalidateSize(true);
                      }
                    } catch (e) {}
                  });
                </script>
                """
            )
        )

    if enable_user_location:
        map_name = flight_map.get_name()
        flight_map.get_root().html.add_child(
            folium.Element(
                f"""
                <style>
                  .locate-control {{
                    background: rgba(13, 19, 28, 0.9);
                    border: 1px solid rgba(255,255,255,0.12);
                    color: #f3f7fb;
                    width: 34px;
                    height: 34px;
                    line-height: 34px;
                    text-align: center;
                    border-radius: 10px;
                    box-shadow: 0 10px 26px rgba(0,0,0,0.28);
                    cursor: pointer;
                    font-size: 16px;
                  }}
                  .locate-control:hover {{
                    background: rgba(21, 30, 42, 0.96);
                  }}
                </style>
                <script>
                  (function() {{
                    function addLocateControl(map) {{
                      if (!map || map._locateControlAdded) return;
                      map._locateControlAdded = true;
                      var LocateControl = L.Control.extend({{
                        options: {{ position: 'bottomleft' }},
                        onAdd: function() {{
                          var container = L.DomUtil.create('div', 'locate-control');
                          container.title = 'Show my location';
                          container.innerHTML = '◎';
                          L.DomEvent.on(container, 'click', function(e) {{
                            L.DomEvent.stopPropagation(e);
                            L.DomEvent.preventDefault(e);
                            if (!navigator.geolocation) {{
                              alert('Geolocation not supported in this browser.');
                              return;
                            }}
                            navigator.geolocation.getCurrentPosition(function(pos) {{
                              var lat = pos.coords.latitude;
                              var lng = pos.coords.longitude;
                              var accuracy = pos.coords.accuracy || 0;
                              if (map._userLocationMarker) {{
                                map.removeLayer(map._userLocationMarker);
                              }}
                              if (map._userLocationCircle) {{
                                map.removeLayer(map._userLocationCircle);
                              }}
                              map._userLocationMarker = L.circleMarker([lat, lng], {{
                                radius: 6,
                                color: '#bcff5c',
                                weight: 2,
                                fillColor: '#bcff5c',
                                fillOpacity: 0.9
                              }}).addTo(map);
                              if (accuracy > 0) {{
                                map._userLocationCircle = L.circle([lat, lng], {{
                                  radius: accuracy,
                                  color: 'rgba(188,255,92,0.35)',
                                  fillColor: 'rgba(188,255,92,0.18)',
                                  fillOpacity: 0.2,
                                  weight: 1
                                }}).addTo(map);
                              }}
                              map.flyTo([lat, lng], Math.max(map.getZoom(), 8), {{ animate: true, duration: 0.8 }});
                            }}, function(err) {{
                              alert('Unable to fetch your location.');
                            }}, {{ enableHighAccuracy: true, timeout: 8000 }});
                          }});
                          return container;
                        }}
                      }});
                      map.addControl(new LocateControl());
                    }}
                    if (window.{map_name}) {{
                      addLocateControl(window.{map_name});
                    }} else {{
                      document.addEventListener('DOMContentLoaded', function() {{
                        if (window.{map_name}) addLocateControl(window.{map_name});
                      }});
                    }}
                  }})();
                </script>
                """
            )
        )

    # Trails first — renders underneath icons
    if show_trails:
        _draw_trails(flight_map, df)

    # ── Airport markers ───────────────────────────────────────────────────────
    airport_layer = folium.FeatureGroup(name="Indian airports", show=True)
    for airport in get_airports_for_region(region):
        folium.CircleMarker(
            location=[airport["lat"], airport["lng"]],
            radius=5,
            color="#111827",
            weight=1.5,
            fill=True,
            fill_color="#ffcf5c",
            fill_opacity=0.95,
            tooltip=f"{airport['iata']} • {airport['city']}",
            popup=folium.Popup(
                f"<b>{airport['iata']}</b><br>{airport['name']}<br>{airport['city']}",
                max_width=260,
            ),
        ).add_to(airport_layer)
        folium.map.Marker(
            [airport["lat"], airport["lng"]],
            icon=folium.DivIcon(
                html=(
                    "<div style='color:#ffcf5c;font-weight:700;font-size:11px;"
                    "text-shadow:0 1px 3px rgba(0,0,0,0.9);'>"
                    f"{airport['iata']}</div>"
                )
            ),
        ).add_to(airport_layer)

    airport_layer.add_to(flight_map)

    if cluster_markers:
        marker_layer = MarkerCluster(
            name="Aircraft",
            options={
                "maxClusterRadius":        40,
                "disableClusteringAtZoom": 8,
                "spiderfyOnMaxZoom":       True,
            },
        ).add_to(flight_map)
    else:
        marker_layer = folium.FeatureGroup(name="Aircraft", show=True).add_to(flight_map)

    added = 0

    for _, row in df.iterrows():
        try:
            aircraft_icao = str(row.get("aircraft_icao", ""))
            heading       = row.get("heading", 0)
            altitude_ft   = row.get("altitude_ft")
            flight_iata   = str(row.get("flight_iata", ""))
            dep           = str(row.get("dep_iata", "?"))
            arr           = str(row.get("arr_iata", "?"))

            is_selected = bool(
                selected_flight
                and selected_flight.strip()
                and selected_flight.strip().upper() in flight_iata.upper()
            )

            is_highlight = bool(highlight_flights and flight_iata in highlight_flights)
            color  = _get_altitude_color(altitude_ft, highlight=is_selected or is_highlight)
            family = get_family(aircraft_icao)
            size   = FAMILY_SIZE.get(family, 26)
            if is_selected:
                size = min(size + 12, 52)   # selected flight gets bigger icon
            elif is_highlight:
                size = min(size + 6, 44)

            icon_html = get_aircraft_icon_html(
                aircraft_icao=aircraft_icao,
                heading=heading,
                color=color,
                size=size,
            )

            marker = folium.Marker(
                location=[row["latitude"], row["longitude"]],
                icon=folium.DivIcon(
                    html=icon_html,
                    icon_size=(size, size),
                    icon_anchor=(size // 2, size // 2),
                ),
                tooltip=f"✈ {flight_iata}  {dep} → {arr}  [{aircraft_icao}]",
                popup=folium.Popup(
                    _build_popup(
                        row,
                        highlight=is_selected,
                        show_delay_fields=show_delay_fields,
                        module_param=module_param,
                    ),
                    max_width=300,
                ),
            )
            marker.add_to(marker_layer)

            added += 1

        except Exception as e:
            logger.debug(f"Skipped marker: {e}")

    logger.info(f"Icons added: {added}")

    if not clean_ui:
        # ── Legend ─────────────────────────────────────────────────────────────
        flight_map.get_root().html.add_child(folium.Element("""
        <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                    background:white;padding:14px 18px;border-radius:10px;
                    border:1px solid #ddd;font-family:Arial,sans-serif;
                    font-size:12px;box-shadow:2px 2px 10px rgba(0,0,0,0.12);
                    min-width:195px;">
            <b style="font-size:13px;display:block;margin-bottom:8px;">✈ Live Flight Map</b>
            <b style="color:#555;display:block;margin-bottom:5px;">Altitude</b>
            <span style="color:#e74c3c;">▲</span> &gt;25,000 ft — Cruise<br>
            <span style="color:#f39c12;">▲</span> 10–25,000 ft — Mid<br>
            <span style="color:#2ecc71;">▲</span> 1–10,000 ft — Low<br>
            <span style="color:#888;">▲</span> &lt;1,000 ft — Ground<br>
            <span style="color:#FF3B30;">▲</span> Tracked flight<br>
            <hr style="margin:8px 0;border:none;border-top:1px solid #eee;">
            <span style="font-size:11px;color:#666;line-height:1.7;">
                Icon size = aircraft size<br>
                Icon shape = aircraft type<br>
                Rotates with heading<br>
                Yellow dots = Indian airports<br>
                Click any icon for details
            </span>
        </div>"""))

        folium.LayerControl(position="topright", collapsed=False).add_to(flight_map)
    return flight_map


# ── Terminal summary ───────────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("  No flight data.")
        return
    stats = flight_tracker.get_stats()
    print("\n" + "=" * 54)
    print("   LIVE FLIGHT MAP — AirLabs + Type-Accurate Icons")
    print("=" * 54)
    print(f"  Flights on map            : {len(df):,}")
    print(f"  Aircraft in trail history : {stats['aircraft_tracked']:,}")
    print(f"  Unique airlines           : {df['airline_iata'].nunique()}")
    print(f"  Avg altitude              : {df['altitude_ft'].mean():,.0f} ft")
    print(f"  Avg speed                 : {df['speed_kts'].mean():.1f} kts")
    print(f"  Fastest                   : {df['speed_kts'].max():.1f} kts")
    print("\n  Top airlines:")
    for code, count in df["airline_iata"].value_counts().head(5).items():
        print(f"    {get_airline_name(code):<28} ({code})  {count:>4}")
    print("=" * 54 + "\n")
