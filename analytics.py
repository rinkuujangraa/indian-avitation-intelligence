"""
analytics.py
------------
Phase 2 intelligence layer for the Aviation Intelligence Platform.

This module computes:
  - airport traffic and congestion
  - schedule pressure
  - route congestion
  - rule-based delay risk and expected delay minutes
"""

from __future__ import annotations

import logging as _logging
import os as _os
import threading as _threading
from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt
from typing import Iterable

import numpy as _np

import pandas as pd

# ── ML delay model (optional — only loaded when models/delay_lgbm.pkl exists) ──
_log = _logging.getLogger(__name__)
_ML_BUNDLE: dict | None = None
_ML_LOADED = False

_ML_LOCK = _threading.Lock()

def _get_ml_bundle() -> dict | None:
    """Lazy-load the ML model bundle once (thread-safe double-checked locking)."""
    global _ML_BUNDLE, _ML_LOADED  # noqa: PLW0603
    if _ML_LOADED:           # fast-path — no lock needed after first load
        return _ML_BUNDLE
    with _ML_LOCK:
        if _ML_LOADED:       # re-check under lock
            return _ML_BUNDLE
        _ML_LOADED = True
        _model_path = _os.path.join(_os.path.dirname(__file__), "models", "delay_lgbm.pkl")
        if not _os.path.exists(_model_path):
            return None
        try:
            from delay_model import load_model as _load_model  # noqa: PLC0415
            _ML_BUNDLE = _load_model(_model_path)
            if _ML_BUNDLE:
                _log.info("ML delay model loaded from %s", _model_path)
        except Exception as _exc:  # noqa: BLE001
            _log.warning("ML model load failed: %s", _exc)
        return _ML_BUNDLE

# IATA airline code → OpenSky callsign prefix (ICAO prefix used in training data)
_IATA_TO_ICAO_AIRLINE: dict[str, str] = {
    "AI": "AIC",  # Air India
    "6E": "IGO",  # IndiGo
    "SG": "SEJ",  # SpiceJet
    "G8": "GOW",  # Go First
    "IX": "AXB",  # Air India Express
    "UK": "VTI",  # Vistara
    "I5": "IST",  # AirAsia India
}

# IATA airport code → ICAO code for the 20 Indian airports in training data
_IATA_TO_ICAO_AIRPORT: dict[str, str] = {
    "DEL": "VIDP", "BOM": "VABB", "MAA": "VOMM", "BLR": "VOBL",
    "COK": "VOCI", "CCU": "VECC", "HYD": "VOHY", "GAU": "VEGT",
    "JAI": "VIJP", "IXC": "VIPT", "BHO": "VABV", "JLR": "VAJB",
    "AMD": "VAAH", "IXZ": "VOPB", "AGR": "VIAG", "ATQ": "VIAR",
    "LKO": "VILK", "BBI": "VEBS", "PAT": "VEPY",
}


EARTH_RADIUS_KM = 6371.0

# ── Airline On-Time Performance profiles ───────────────────────────────────────
# Estimated OTP % from public DGCA / industry data. Lower = more delay-prone.
AIRLINE_OTP: dict[str, float] = {
    "6E": 0.87,  # IndiGo — consistently high OTP
    "AI": 0.64,  # Air India — historically lower
    "UK": 0.78,  # Vistara
    "SG": 0.72,  # SpiceJet
    "G8": 0.68,  # GoFirst
    "QP": 0.80,  # Akasa Air
    "I5": 0.74,  # AirAsia India
    "S5": 0.70,  # StarAir
    "9I": 0.66,  # Alliance Air
    "EK": 0.85,  # Emirates
    "EY": 0.82,  # Etihad
    "QR": 0.84,  # Qatar Airways
    "SQ": 0.88,  # Singapore Airlines
    "BA": 0.76,  # British Airways
    "LH": 0.78,  # Lufthansa
    "TK": 0.77,  # Turkish Airlines
    "FZ": 0.80,  # flydubai
    "WY": 0.79,  # Oman Air
    "TG": 0.75,  # Thai Airways
    "CX": 0.82,  # Cathay Pacific
}

# Metro airports with known peak-hour congestion (India)
_PEAK_AIRPORTS = {"DEL", "BOM", "BLR", "HYD", "MAA", "CCU", "AMD", "PNQ", "GOI", "COK"}

# Chronically congested route pairs (both directions). Fixed delay penalty.
_HOTSPOT_ROUTES: dict[frozenset, tuple[float, int]] = {
    frozenset({"DEL", "BOM"}): (6.0, 4),
    frozenset({"DEL", "BLR"}): (5.0, 3),
    frozenset({"BOM", "BLR"}): (4.0, 3),
    frozenset({"DEL", "HYD"}): (4.0, 3),
    frozenset({"DEL", "MAA"}): (4.0, 3),
    frozenset({"DEL", "CCU"}): (4.0, 3),
    frozenset({"BOM", "GOI"}): (3.0, 2),
    frozenset({"BLR", "MAA"}): (3.0, 2),
    frozenset({"DEL", "AMD"}): (3.0, 2),
}

# Monsoon-affected airports (Jun-Sep) — heavier disruptions
# Keep in sync with mapbox_base.py _MONSOON_AIRPORTS
_MONSOON_AIRPORTS = {"BOM", "GOI", "COK", "CCU", "MAA", "GAU", "IXB", "PNQ", "IXZ", "CNN"}

# Day-of-week congestion weights (0=Mon, 6=Sun)
_DOW_WEIGHTS: dict[int, tuple[float, int]] = {
    0: (4.0, 3),  # Monday morning rush
    4: (5.0, 3),  # Friday evening rush
    6: (4.0, 3),  # Sunday evening returns
}

# ── Slot-controlled airport maximum runway capacity (movements/hour) ───────────
# Source: AAI slot committee declarations + DGCA ARC reports
SLOT_CAPACITY: dict[str, int] = {
    "DEL": 78, "BOM": 46, "BLR": 40, "MAA": 36,
    "HYD": 32, "CCU": 28, "AMD": 24, "PNQ": 20,
    "COK": 22, "GOI": 18, "GAU": 16, "IXZ": 12,
}

# ── Primary runway headings per airport (degrees) — used for crosswind calc ─────
# True heading (°) of the primary ILS runway in use
_RUNWAY_HEADINGS: dict[str, int] = {
    "DEL": 280, "BOM": 270, "BLR": 90, "MAA": 70,
    "HYD": 90, "CCU": 19, "AMD": 230, "PNQ": 270,
    "COK": 230, "GOI": 310, "GAU": 25, "IXB": 10,
}

# ── METAR severe weather code penalties (minutes) ────────────────────────────────
_METAR_CODE_PENALTIES: dict[str, tuple[int, str]] = {
    "TS":   (22, "thunderstorm at destination"),
    "TSRA": (25, "thunderstorm with rain"),
    "FZRA": (32, "freezing rain / de-ice queue"),
    "FZDZ": (18, "freezing drizzle"),
    "BLSN": (28, "blowing snow / poor visibility"),
    "SN":   (18, "snow at destination"),
    "GR":   (20, "hail reported"),
    "FG":   (16, "fog at destination"),
    "FZFG": (22, "freezing fog"),
    "BCFG": (12, "patchy fog"),
    "VCTS": (14, "vicinity thunderstorm"),
    "SQ":   (16, "squall line"),
    "FC":   (30, "funnel cloud / tornado warning"),
    "DS":   (18, "duststorm"),
    "SS":   (18, "sandstorm"),
    "VA":   (25, "volcanic ash"),
}


def _safe_float(value) -> float:
    """Convert a value to float, returning NaN on failure."""
    if value is None:
        return float('nan')
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


# ── Haversine scalar (single point → single point) ────────────────────────────
def _haversine_scalar(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return distance in km between two lat/lon points."""
    try:
        p = 3.141592653589793 / 180.0
        a = (
            (sin((lat2 - lat1) * p / 2)) ** 2
            + cos(lat1 * p) * cos(lat2 * p) * (sin((lon2 - lon1) * p / 2)) ** 2
        )
        return 2 * 6371.0 * asin(sqrt(max(0.0, min(1.0, a))))
    except Exception:
        return float('nan')


# ── Holding pattern detection ─────────────────────────────────────────────────
def detect_holding_pattern(flight_history: list[dict]) -> tuple[bool, int]:
    """
    Detect holding patterns from recent position snapshots.
    Each dict must have: lat, lng, heading, speed_kts, altitude_ft, ts.
    Returns (is_holding, estimated_extra_minutes).

    Holding signatures:
      - heading variance > 160° within a 6-minute window
      - mean speed < 240 kts
      - altitude variance < 500 ft (level hold)
      - position spread < 22 km radius (not going anywhere)
    """
    if not flight_history or len(flight_history) < 3:
        return False, 0
    required = ("lat", "lng", "heading", "speed_kts", "altitude_ft", "ts")
    recent = [p for p in flight_history if all(p.get(k) is not None for k in required)]
    if len(recent) < 3:
        return False, 0
    now_ts = recent[-1]["ts"]
    window = [p for p in recent if now_ts - p["ts"] <= 370]
    if len(window) < 3:
        return False, 0
    headings = [float(p["heading"]) for p in window]
    speeds   = [float(p["speed_kts"]) for p in window]
    alts     = [float(p["altitude_ft"]) for p in window]
    lats     = [float(p["lat"]) for p in window]
    lngs     = [float(p["lng"]) for p in window]
    heading_range = max(headings) - min(headings)
    if heading_range > 180:
        adjusted = [h - 360 if h > 180 else h for h in headings]
        heading_range = max(adjusted) - min(adjusted)
    avg_speed = sum(speeds) / len(speeds)
    alt_variance = max(alts) - min(alts)
    lat_c = sum(lats) / len(lats)
    lng_c = sum(lngs) / len(lngs)
    max_spread_km = max(_haversine_scalar(lat_c, lng_c, p["lat"], p["lng"]) for p in window)
    is_holding = (
        heading_range >= 160
        and avg_speed <= 240
        and alt_variance <= 500
        and max_spread_km <= 22
    )
    if not is_holding:
        return False, 0
    window_minutes = (window[-1]["ts"] - window[0]["ts"]) / 60.0
    stacks = max(1, round(window_minutes / 4.5))
    return True, min(stacks * 10, 40)


# ── Go-around detection ───────────────────────────────────────────────────────
def detect_go_around(flight_history: list[dict], dest_lat: float, dest_lng: float) -> tuple[bool, int]:
    """
    Detect a go-around: aircraft was on short final then suddenly climbed.
    Returns (detected, extra_minutes).
    """
    if not flight_history or len(flight_history) < 4:
        return False, 0
    required = ("lat", "lng", "altitude_ft", "speed_kts", "ts")
    recent = [p for p in flight_history if all(p.get(k) is not None for k in required)]
    if len(recent) < 4:
        return False, 0
    now_ts = recent[-1]["ts"]
    window = [p for p in recent if now_ts - p["ts"] <= 300]
    if len(window) < 4:
        return False, 0
    short_final = None
    for i, p in enumerate(window):
        dist = _haversine_scalar(float(p["lat"]), float(p["lng"]), dest_lat, dest_lng)
        if float(p["altitude_ft"]) < 3000 and dist < 15:
            short_final = (i, float(p["altitude_ft"]))
            break  # capture first short-final point; subsequent climb is the go-around
    if short_final is None:
        return False, 0
    idx, final_alt = short_final
    subsequent_alts = [float(p["altitude_ft"]) for p in window[idx + 1:]]
    if subsequent_alts and max(subsequent_alts) - final_alt > 800:
        return True, 14
    return False, 0


# ── Slow / unstabilised approach detection ────────────────────────────────────
def detect_slow_approach(altitude_ft: float, speed_kts: float, distance_km: float) -> tuple[bool, int, str]:
    """
    Detect an unstabilised or abnormal approach profile.
    Normal 3° glidepath: ~328 ft per km from threshold.
    Returns (detected, extra_minutes, reason).
    """
    import math
    if any(math.isnan(v) for v in (altitude_ft, speed_kts, distance_km)):
        return False, 0, ""
    expected_alt = distance_km * 328.0
    if distance_km <= 20 and expected_alt > 0 and altitude_ft > expected_alt * 1.35:
        overshoot = (altitude_ft / expected_alt - 1.0) * 100.0
        extra = min(int(overshoot / 10) * 2 + 3, 12)
        return True, extra, "extended / high approach"
    if altitude_ft < 2000 and distance_km < 12 and speed_kts < 128:
        return True, 5, "slow final — wind shear possible"
    if altitude_ft < 3000 and distance_km < 15 and speed_kts > 190:
        return True, 6, "fast approach — go-around risk"
    return False, 0, ""


# ── Diversion detection ───────────────────────────────────────────────────────
def detect_diversion(
    lat: float, lng: float,
    dest_lat: float, dest_lng: float,
    heading: float, altitude_ft: float,
    flight_history: list[dict],
) -> bool:
    """
    Detect if aircraft is heading persistently away from destination
    at descent altitude — strong diversion indicator.
    """
    import math
    if any(math.isnan(v) for v in (lat, lng, dest_lat, dest_lng, heading, altitude_ft)):
        return False
    if altitude_ft > 25000:
        return False
    d_lng = math.radians(dest_lng - lng)
    a = math.atan2(
        math.sin(d_lng) * math.cos(math.radians(dest_lat)),
        math.cos(math.radians(lat)) * math.sin(math.radians(dest_lat))
        - math.sin(math.radians(lat)) * math.cos(math.radians(dest_lat)) * math.cos(d_lng),
    )
    bearing_to_dest = (math.degrees(a) + 360) % 360
    heading_diff = abs(heading - bearing_to_dest) % 360
    if heading_diff > 180:
        heading_diff = 360 - heading_diff
    if heading_diff < 120 or len(flight_history) < 3:
        return False
    now_ts = flight_history[-1].get("ts", 0)
    window = [p for p in flight_history if now_ts - p.get("ts", 0) <= 360]
    return (
        len(window) >= 3
        and all(float(p.get("altitude_ft", 99999)) < 20000 for p in window if p.get("altitude_ft"))
    )


# ── METAR detailed penalty scorer ─────────────────────────────────────────────
def score_metar_penalties(
    raw_text: str,
    wind_kts: int | None,
    gust_kts: int | None,
    visibility_miles: float | None,
    arr_iata: str,
) -> tuple[int, list[str]]:
    """
    Score additional delay minutes from METAR raw text beyond what
    WeatherImpact.penalty_minutes already captures.
    Returns (extra_minutes, reasons).
    """
    import re, math
    extra = 0
    reasons: list[str] = []
    raw = (raw_text or "").upper()

    # Split into tokens; drop first (station ICAO like VABB, VAAH) so
    # "VA" volcanic-ash code doesn't match every Indian airport identifier.
    raw_parts = raw.split()
    wx_tokens = set(raw_parts[1:]) if len(raw_parts) > 1 else set()

    # Severe weather phenomenon codes — whole-token match, take worst.
    # Also match intensity-prefixed forms: -TSRA, +TSRA, VCTS etc.
    best_penalty, best_reason = 0, ""
    for code, (penalty, reason) in _METAR_CODE_PENALTIES.items():
        if any(tok == code or tok.lstrip("+-") == code for tok in wx_tokens):
            if penalty > best_penalty:
                best_penalty, best_reason = penalty, reason
    if best_reason:
        extra += best_penalty
        reasons.append(best_reason)

    # Visibility tiers
    if visibility_miles is not None:
        vis_m = visibility_miles * 1609.34
        if vis_m < 550:
            extra += 28
            reasons.append("CAT-III / near-zero visibility")
        elif vis_m < 1500:
            extra += 18
            reasons.append("CAT-II RVR — reduced throughput")
        elif vis_m < 3000:
            extra += 10
            reasons.append("CAT-I minimums")
        elif vis_m < 5000:
            extra += 5
            reasons.append("reduced visibility")

    # Crosswind component vs primary runway heading
    rwy_hdg = _RUNWAY_HEADINGS.get(arr_iata)
    if rwy_hdg and wind_kts:
        m = re.search(r"(\d{3})(\d{2,3})(?:G(\d{2,3}))?KT", raw)
        if m:
            wind_dir = int(m.group(1))
            w_speed = int(m.group(2))
            angle_diff = abs(wind_dir - rwy_hdg) % 360
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            crosswind = w_speed * math.sin(math.radians(angle_diff))
            if crosswind > 25:
                extra += 14
                reasons.append(f"severe crosswind {crosswind:.0f} kts")
            elif crosswind > 15:
                extra += 8
                reasons.append(f"crosswind {crosswind:.0f} kts")
    if gust_kts and wind_kts:
        spread = gust_kts - wind_kts
        if spread >= 20:
            extra += 10
            reasons.append("severe gust spread")
        elif spread >= 12:
            extra += 5
            reasons.append("gusty conditions")

    return min(extra, 45), reasons


# ── Departure delay recovery curve ───────────────────────────────────────────
def dep_delay_recovery_minutes(dep_delay_min: float, distance_km: float | None) -> int:
    """
    Net arrival delay after en-route recovery.
    Short-haul < 500 km: 20% recovery; medium 500-1500: 40%; long > 1500: 55%.
    """
    import math as _math
    if dep_delay_min <= 0 or _math.isnan(float(dep_delay_min)):
        return 0
    if distance_km is None or _math.isnan(float(distance_km)):
        recovery = 0.30
    elif distance_km < 500:
        recovery = 0.20
    elif distance_km < 1500:
        recovery = 0.40
    else:
        recovery = 0.55
    return min(int(dep_delay_min * (1.0 - recovery)), 45)


# ── Slot capacity burst penalty ───────────────────────────────────────────────
def slot_capacity_burst_minutes(arr_iata: str, arrivals_next_hour: int) -> int:
    """
    When arrivals approach slot-declared capacity, ground-delay metering begins.
    """
    capacity = SLOT_CAPACITY.get(arr_iata)
    if not capacity or arrivals_next_hour <= 0:
        return 0
    util = arrivals_next_hour / capacity
    if util >= 1.0:
        return min(int((arrivals_next_hour - capacity) * 2.2 + 18), 40)
    if util >= 0.88:
        return min(int((util - 0.88) * 100 * 0.8 + 8), 20)
    return 0


@dataclass(frozen=True)
class DelayPrediction:
    risk_level: str
    expected_delay_min: int
    reason_tags: tuple[str, ...]
    score: float
    congestion_level: str
    schedule_pressure_level: str
    holding_detected: bool = False
    go_around_detected: bool = False
    diversion_suspected: bool = False
    ml_prob: float = 0.0


@dataclass(frozen=True)
class ArrivalEstimate:
    distance_km: float | None
    eta_minutes: int | None
    eta_label_utc: str
    arrival_summary: str


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def haversine_km(lat1: float, lon1: float, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """
    Vectorized haversine distance for one origin point vs many destination points.
    Uses NumPy broadcasting — ~100x faster than Series.map(math.sin).
    """
    lat1_r = _np.radians(float(lat1))
    lon1_r = _np.radians(float(lon1))
    lat2_r = _np.radians(lat2.to_numpy(dtype=float))
    lon2_r = _np.radians(lon2.to_numpy(dtype=float))
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = _np.sin(dlat / 2) ** 2 + _np.cos(lat1_r) * _np.cos(lat2_r) * _np.sin(dlon / 2) ** 2
    a = _np.clip(a, 0.0, 1.0)
    return pd.Series(2 * EARTH_RADIUS_KM * _np.arcsin(_np.sqrt(a)), index=lat2.index)


def classify_congestion(score: float) -> str:
    if score >= 34:
        return "Severe"
    if score >= 22:
        return "High"
    if score >= 12:
        return "Moderate"
    return "Low"


def classify_schedule_pressure(arrivals_next_hour: int, departures_next_hour: int) -> tuple[str, float]:
    score = arrivals_next_hour * 1.3 + departures_next_hour * 1.0
    if score >= 16:
        return "High", score
    if score >= 9:
        return "Moderate", score
    return "Low", score


def compute_airport_traffic_metrics(
    flights_df: pd.DataFrame,
    airports: Iterable[dict],
    radius_km: float = 90.0,
) -> pd.DataFrame:
    """
    Score live airport pressure using nearby aircraft, inbound flow, and low-altitude activity.
    """
    if flights_df.empty:
        return pd.DataFrame(
            columns=[
                "airport_iata",
                "airport_name",
                "city",
                "nearby_count",
                "inbound_count",
                "low_altitude_count",
                "approach_count",
                "congestion_score",
                "congestion_level",
            ]
        )

    flights = flights_df.copy()
    flights["latitude"] = _coerce_numeric(flights["latitude"])
    flights["longitude"] = _coerce_numeric(flights["longitude"])
    # Drop rows with NaN coordinates — can't compute distance
    flights = flights.dropna(subset=["latitude", "longitude"])
    if flights.empty:
        return pd.DataFrame(
            columns=[
                "airport_iata",
                "airport_name",
                "city",
                "nearby_count",
                "inbound_count",
                "low_altitude_count",
                "approach_count",
                "congestion_score",
                "congestion_level",
            ]
        )
    flights["altitude_ft"] = _coerce_numeric(flights["altitude_ft"])
    flights["speed_kts"] = _coerce_numeric(flights["speed_kts"])

    rows: list[dict] = []
    for airport in airports:
        distances = haversine_km(airport["lat"], airport["lng"], flights["latitude"], flights["longitude"])
        nearby_mask = distances <= radius_km
        nearby = flights.loc[nearby_mask]
        inbound = nearby["arr_iata"].fillna("").eq(airport["iata"])
        low_alt = nearby["altitude_ft"].fillna(0) <= 12000
        approach = low_alt & (nearby["speed_kts"].fillna(9999) <= 260)

        nearby_count = int(nearby_mask.sum())
        inbound_count = int(inbound.sum())
        low_altitude_count = int(low_alt.sum())
        approach_count = int(approach.sum())

        congestion_score = round(
            nearby_count * 1.0
            + inbound_count * 2.2
            + low_altitude_count * 1.7
            + approach_count * 1.6,
            1,
        )
        rows.append(
            {
                "airport_iata": airport["iata"],
                "airport_name": airport["name"],
                "city": airport["city"],
                "nearby_count": nearby_count,
                "inbound_count": inbound_count,
                "low_altitude_count": low_altitude_count,
                "approach_count": approach_count,
                "congestion_score": congestion_score,
                "congestion_level": classify_congestion(congestion_score),
            }
        )

    metrics = pd.DataFrame(rows).sort_values(
        by=["congestion_score", "inbound_count", "nearby_count"],
        ascending=False,
    )
    metrics.reset_index(drop=True, inplace=True)
    return metrics


def compute_airport_schedule_pressure(
    arrivals_df: pd.DataFrame,
    departures_df: pd.DataFrame,
    now_ts: int,
) -> dict:
    """
    Compute next-hour and next-3-hour airport schedule pressure.
    """

    def _count_window(df: pd.DataFrame, estimated_col: str, scheduled_col: str, horizon_seconds: int) -> int:
        if df.empty:
            return 0
        timestamps = _coerce_numeric(df.get(estimated_col, pd.Series(dtype=float))).fillna(
            _coerce_numeric(df.get(scheduled_col, pd.Series(dtype=float)))
        )
        valid = timestamps[(timestamps >= now_ts) & (timestamps < now_ts + horizon_seconds)]
        return int(valid.count())

    def _bucket(df: pd.DataFrame, estimated_col: str, scheduled_col: str, h_start: int, h_end: int) -> int:
        if df.empty:
            return 0
        ts = _coerce_numeric(df.get(estimated_col, pd.Series(dtype=float))).fillna(
            _coerce_numeric(df.get(scheduled_col, pd.Series(dtype=float)))
        )
        start = now_ts + h_start * 3600
        end = now_ts + h_end * 3600
        return int(((ts >= start) & (ts < end)).sum())

    arrivals_next_hour = _count_window(arrivals_df, "arr_estimated_ts", "arr_time_ts", 3600)
    departures_next_hour = _count_window(departures_df, "dep_estimated_ts", "dep_time_ts", 3600)
    arrivals_next_3h = _count_window(arrivals_df, "arr_estimated_ts", "arr_time_ts", 3 * 3600)
    departures_next_3h = _count_window(departures_df, "dep_estimated_ts", "dep_time_ts", 3 * 3600)
    pressure_level, pressure_score = classify_schedule_pressure(arrivals_next_hour, departures_next_hour)

    arr_hb  = [_bucket(arrivals_df,   "arr_estimated_ts", "arr_time_ts",  i, i + 1) for i in range(3)]
    dep_hb  = [_bucket(departures_df, "dep_estimated_ts", "dep_time_ts",  i, i + 1) for i in range(3)]

    return {
        "arrivals_next_hour": arrivals_next_hour,
        "departures_next_hour": departures_next_hour,
        "arrivals_next_3h": arrivals_next_3h,
        "departures_next_3h": departures_next_3h,
        "pressure_level": pressure_level,
        "pressure_score": round(pressure_score, 1),
        "arrivals_h0": arr_hb[0],
        "arrivals_h1": arr_hb[1],
        "arrivals_h2": arr_hb[2],
        "departures_h0": dep_hb[0],
        "departures_h1": dep_hb[1],
        "departures_h2": dep_hb[2],
    }


def compute_route_congestion(flights_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Rank the busiest active routes in the live frame.
    """
    if flights_df.empty:
        return pd.DataFrame(
            columns=["route_key", "dep_iata", "arr_iata", "flight_count", "low_altitude_count", "congestion_score"]
        )

    routes = flights_df.copy()
    routes["dep_iata"] = routes["dep_iata"].fillna("N/A")
    routes["arr_iata"] = routes["arr_iata"].fillna("N/A")
    routes["altitude_ft"] = _coerce_numeric(routes["altitude_ft"])
    routes = routes[(routes["dep_iata"] != "N/A") & (routes["arr_iata"] != "N/A")]
    if routes.empty:
        return pd.DataFrame(
            columns=["route_key", "dep_iata", "arr_iata", "flight_count", "low_altitude_count", "congestion_score"]
        )

    grouped = (
        routes.assign(low_altitude=routes["altitude_ft"].fillna(99999) <= 12000)
        .groupby(["dep_iata", "arr_iata"], as_index=False)
        .agg(
            flight_count=("hex", "count"),
            low_altitude_count=("low_altitude", "sum"),
        )
    )
    grouped["congestion_score"] = (
        grouped["flight_count"] * 3.0 + grouped["low_altitude_count"] * 1.8
    ).round(1)
    grouped["route_key"] = grouped["dep_iata"] + "-" + grouped["arr_iata"]
    grouped = grouped.sort_values(by=["congestion_score", "flight_count"], ascending=False).head(top_n)
    grouped.reset_index(drop=True, inplace=True)
    return grouped


def compute_delay_prediction(
    flight_row: pd.Series,
    airports: Iterable[dict],
    airport_metrics: pd.DataFrame,
    airport_schedule_pressure: dict[str, dict],
    now_ts: int = 0,
    weather_impact=None,
    flight_history: list[dict] | None = None,
    _airport_lookup: "dict | None" = None,
    _metric_map: "dict | None" = None,
) -> DelayPrediction:
    """
    Rule-based delay-risk scoring for a single flight.
    Uses live congestion, schedule pressure, weather severity,
    airline OTP profile, time-of-day patterns, holding/go-around detection,
    METAR penalties, dep-delay recovery curve, and slot capacity bursts.
    """
    reasons: list[str] = []
    score = 0.0
    minutes = 0
    holding_detected = False
    go_around_detected = False
    diversion_suspected = False

    altitude_ft = _safe_float(flight_row.get("altitude_ft"))
    speed_kts = _safe_float(flight_row.get("speed_kts"))
    arr_iata = str(flight_row.get("arr_iata") or "").strip().upper()
    latitude = _safe_float(flight_row.get("latitude"))
    longitude = _safe_float(flight_row.get("longitude"))
    weather_sev = str(flight_row.get("weather_severity") or "Low").strip()
    airline_code = str(flight_row.get("airline_iata") or "").strip().upper()
    stale_minutes = _safe_float(flight_row.get("stale_minutes"))

    if not arr_iata or arr_iata == "N/A":
        return DelayPrediction(
            risk_level="Low",
            expected_delay_min=0,
            reason_tags=("no destination data",),
            score=0.0,
            congestion_level="Low",
            schedule_pressure_level="Low",
        )

    airport_by_code = _airport_lookup if _airport_lookup is not None else {airport["iata"]: airport for airport in airports}
    airport_metric_map = _metric_map if _metric_map is not None else (
        airport_metrics.set_index("airport_iata").to_dict("index") if not airport_metrics.empty else {}
    )
    metric = airport_metric_map.get(arr_iata, {})
    schedule = airport_schedule_pressure.get(arr_iata, {})

    if pd.notna(speed_kts) and pd.notna(altitude_ft):
        if altitude_ft >= 22000 and speed_kts < 360:
            score += 18
            minutes += 8
            reasons.append("slow cruise profile")
        elif altitude_ft >= 12000 and speed_kts < 280:
            score += 12
            minutes += 6
            reasons.append("slow en-route speed")

    destination_airport = airport_by_code.get(arr_iata)
    distance_km = None
    is_on_approach = False
    dest_lat = dest_lng = None
    if destination_airport and pd.notna(latitude) and pd.notna(longitude):
        dest_lat = float(destination_airport["lat"])
        dest_lng = float(destination_airport["lng"])
        distance_km = haversine_km(
            dest_lat,
            dest_lng,
            pd.Series([latitude]),
            pd.Series([longitude]),
        ).iloc[0]
        if pd.notna(distance_km):
            # Detect approach: descending within terminal area, normal behavior
            if distance_km <= 250 and pd.notna(altitude_ft) and altitude_ft <= 15000:
                is_on_approach = True
                reasons.append("on approach")
            elif distance_km <= 140 and pd.notna(speed_kts) and speed_kts <= 240:
                is_on_approach = True
                reasons.append("on final")

    # ── Holding pattern detection ──────────────────────────────────────────
    fh = flight_history or []
    if fh:
        holding_detected, hold_min = detect_holding_pattern(fh)
        if holding_detected:
            score += 28
            minutes += hold_min
            reasons.append(f"holding pattern +{hold_min} min")
        # Go-around detection
        if dest_lat is not None and dest_lng is not None:
            go_around_detected, ga_min = detect_go_around(fh, dest_lat, dest_lng)
            if go_around_detected:
                score += 24
                minutes += ga_min
                reasons.append("go-around detected")
        # Diversion detection
        heading = _safe_float(flight_row.get("heading") or flight_row.get("direction"))
        if dest_lat is not None and pd.notna(heading):
            diversion_suspected = detect_diversion(
                latitude, longitude, dest_lat, dest_lng, heading, altitude_ft, fh
            )
            if diversion_suspected:
                score += 45
                minutes += 35
                reasons.append("possible diversion")

    # ── Slow / unstabilised approach ───────────────────────────────────────
    if pd.notna(altitude_ft) and pd.notna(speed_kts) and distance_km is not None and pd.notna(distance_km):
        approach_anomaly, approach_min, approach_reason = detect_slow_approach(
            float(altitude_ft), float(speed_kts), float(distance_km)
        )
        if approach_anomaly:
            score += 15
            minutes += approach_min
            reasons.append(approach_reason)

    congestion_level = str(metric.get("congestion_level", "Low"))
    congestion_score = float(metric.get("congestion_score", 0.0) or 0.0)
    if congestion_level == "Severe":
        score += 22
        minutes += 14
        reasons.append("destination congestion")
    elif congestion_level == "High":
        score += 15
        minutes += 10
        reasons.append("destination congestion")
    elif congestion_level == "Moderate":
        score += 8
        minutes += 5
        reasons.append("moderate airport load")

    schedule_pressure_level = str(schedule.get("pressure_level", "Low"))
    arrivals_next_hour = int(schedule.get("arrivals_next_hour", 0) or 0)
    departures_next_hour = int(schedule.get("departures_next_hour", 0) or 0)
    if schedule_pressure_level == "High":
        score += 16
        minutes += 9
        reasons.append("arrival bank pressure")
    elif schedule_pressure_level == "Moderate":
        score += 9
        minutes += 5
        reasons.append("upcoming airport bank")

    if arrivals_next_hour >= 8:
        score += 6
        minutes += 4
        reasons.append("heavy arrivals next hour")
    if departures_next_hour >= 8:
        score += 4
        minutes += 3
        reasons.append("busy departure bank")

    # ── Slot capacity burst (GDP / metering) ──────────────────────────────
    slot_extra = slot_capacity_burst_minutes(arr_iata, arrivals_next_hour)
    if slot_extra:
        score += slot_extra * 0.8
        minutes += slot_extra
        reasons.append("slot capacity burst / GDP")

    # ── Detailed METAR penalties ───────────────────────────────────────────
    if weather_impact is not None:
        raw_text = getattr(weather_impact, "raw_text", None)
        w_wind = getattr(weather_impact, "wind_kts", None)
        w_gust = getattr(weather_impact, "gust_kts", None)
        w_vis  = getattr(weather_impact, "visibility_miles", None)
        metar_extra, metar_reasons = score_metar_penalties(raw_text, w_wind, w_gust, w_vis, arr_iata)
        if metar_extra:
            score += metar_extra * 0.9
            minutes += metar_extra
            reasons.extend(metar_reasons)

    # ── Weather at destination ─────────────────────────────────────────────
    if weather_sev == "Severe":
        score += 24
        minutes += 18
        reasons.append("severe weather at destination")
    elif weather_sev == "Moderate":
        score += 12
        minutes += 8
        reasons.append("weather impact at destination")

    # ── Airline on-time performance ────────────────────────────────────────
    otp = AIRLINE_OTP.get(airline_code)
    if otp is not None:
        if otp < 0.70:
            score += 10
            minutes += 6
            reasons.append("airline delay-prone")
        elif otp < 0.78:
            score += 5
            minutes += 3
            reasons.append("airline moderate OTP")
    # Unknown airlines get no penalty (benefit of doubt)

    # ── Time-of-day peak hour penalty ──────────────────────────────────────
    dep_iata = str(flight_row.get("dep_iata") or "").strip().upper()
    ist_hour = -1
    ist_dow = -1
    _ist_dt = None  # initialise so the monsoon block below is never unbound
    if now_ts > 0:
        import datetime as _dt
        utc_dt = _dt.datetime.fromtimestamp(now_ts, tz=_dt.timezone.utc)
        _ist_dt = utc_dt + _dt.timedelta(hours=5, minutes=30)  # IST = UTC+5:30
        ist_hour = _ist_dt.hour
        ist_dow = _ist_dt.weekday()  # 0=Mon, 6=Sun
        if arr_iata in _PEAK_AIRPORTS and (7 <= ist_hour <= 10 or 17 <= ist_hour <= 21):
            score += 8
            minutes += 5
            reasons.append("peak hour at metro airport")

    # ── Route-specific hotspots ────────────────────────────────────────────
    if dep_iata and arr_iata:
        route_key = frozenset({dep_iata, arr_iata})
        hotspot = _HOTSPOT_ROUTES.get(route_key)
        if hotspot:
            score += hotspot[0]
            minutes += hotspot[1]
            reasons.append("congested route corridor")

    # ── Departure delay recovery curve ────────────────────────────────────
    dep_delay = _safe_float(flight_row.get("dep_delayed"))
    if dep_delay > 0:
        net_delay = dep_delay_recovery_minutes(dep_delay, distance_km)
        score += min(net_delay * 0.8, 18)
        minutes += net_delay
        reasons.append("late departure (cascading)")

    # ── Day-of-week pattern ────────────────────────────────────────────────
    if ist_dow >= 0:
        dow_penalty = _DOW_WEIGHTS.get(ist_dow)
        if dow_penalty:
            score += dow_penalty[0]
            minutes += dow_penalty[1]
            reasons.append("high-traffic day")

    # ── Monsoon season penalty (Jun-Sep) ───────────────────────────────────
    if _ist_dt is not None:
        month = _ist_dt.month  # use IST month (consistent with feature engineering)
        if 6 <= month <= 9:
            if arr_iata in _MONSOON_AIRPORTS:
                score += 10
                minutes += 7
                reasons.append("monsoon season at destination")
            if dep_iata in _MONSOON_AIRPORTS:
                score += 5
                minutes += 3
                reasons.append("monsoon season at origin")

    # ── Approach discount ────────────────────────────────────────────────────
    # Aircraft on approach/descent into destination is committed to landing.
    # ATC has already sequenced it. External penalties (congestion, weather,
    # peak) should be heavily discounted — any real delay at this point is
    # limited to a go-around or short hold, not 30+ minutes.
    if is_on_approach and score > 0:
        if distance_km is not None and pd.notna(altitude_ft):
            if distance_km <= 40 and altitude_ft <= 4000:
                # Virtually on the runway — cap hard
                minutes = min(minutes, 2)
                score = min(score, 5.0)
            elif distance_km <= 80 and altitude_ft <= 8000:
                minutes = min(minutes, 4)
                score = min(score, 10.0)
            elif distance_km <= 200 and altitude_ft <= 12000:
                # Standard approach / descent into terminal area
                discount = 0.15
                minutes = int(minutes * discount)
                score *= discount
            else:
                # Early descent / extended terminal area
                discount = 0.25
                minutes = int(minutes * discount)
                score *= discount
        reasons.append("landing imminent")

    # ── Healthy-cruise discount ────────────────────────────────────────────
    # A flight cruising normally at high altitude with good speed and no
    # departure delay is very likely on-time. External factors (congestion,
    # peak hours) hit these flights less — better planning, priority ATC.
    is_normal_cruise = (
        pd.notna(altitude_ft) and altitude_ft >= 28000
        and pd.notna(speed_kts) and speed_kts >= 380
    )
    dep_delay_val = _safe_float(flight_row.get("dep_delayed"))
    is_on_time_dep = pd.isna(dep_delay_val) or dep_delay_val <= 0

    # ── Stale position data ─────────────────────────────────────────────
    if pd.notna(stale_minutes) and stale_minutes >= 3:
        if stale_minutes >= 10:
            score += 12
            minutes += 8
            reasons.append("lost contact")
        elif stale_minutes >= 5:
            score += 6
            minutes += 4
            reasons.append("stale position")
        else:
            score += 3
            minutes += 2
            reasons.append("position lag")

    if is_normal_cruise and is_on_time_dep and score > 0:
        # Reliable airlines get a bigger discount — if severe delays were
        # imminent, ATC would already have the aircraft holding or slowing.
        if otp is not None and otp >= 0.82:
            discount = 0.25   # 75% off — high-OTP + healthy cruise
        elif otp is not None and otp >= 0.70:
            discount = 0.40   # 60% off — decent OTP + healthy cruise
        else:
            discount = 0.50   # 50% off — healthy cruise, unknown/low-OTP airline
        score *= discount
        minutes = int(minutes * discount)
        reasons.append("on-schedule cruise")

    minutes = min(minutes, 45)
    score = min(score, 100)

    # ── ML model blending (when model is available) ──────────────────────
    # Blend the historical-pattern ML probability with the rule-based score.
    # Weight: 35% ML + 65% rules. If model unavailable falls back to 100% rules.
    ml_prob = 0.0
    ml_bundle = _get_ml_bundle()
    if ml_bundle is not None and not (holding_detected or go_around_detected or diversion_suspected):
        try:
            from delay_model import predict_delay_prob  # noqa: PLC0415
            icao_airline = _IATA_TO_ICAO_AIRLINE.get(airline_code, airline_code)
            dep_icao_ml = _IATA_TO_ICAO_AIRPORT.get(dep_iata or "", dep_iata or "?")
            arr_icao_ml = _IATA_TO_ICAO_AIRPORT.get(arr_iata, arr_iata)
            dep_month_ml = _ist_dt.month if _ist_dt is not None else __import__("datetime").datetime.now().month
            ml_prob = predict_delay_prob(
                ml_bundle,
                airline_prefix=icao_airline,
                dep_icao=dep_icao_ml,
                arr_icao=arr_icao_ml,
                dep_hour_ist=ist_hour if now_ts > 0 else 12,
                dep_dow=ist_dow if ist_dow >= 0 else 0,
                dep_month=dep_month_ml,
            )
            ml_score = ml_prob * 100.0
            _ens_w = float(ml_bundle.get("ensemble_weight", 0.55))
            score = score * (1.0 - _ens_w) + ml_score * _ens_w
            if ml_prob >= 0.65 and score < 38:
                reasons.append(f"ML: high historical delay rate ({ml_prob:.0%})")
            elif ml_prob <= 0.25 and score > 0:
                score *= 0.85  # ML says this route/time historically reliable
        except Exception:  # noqa: BLE001
            pass  # ML unavailable — pure rules

    score = min(score, 100)
    if score >= 38:
        risk_level = "High"
    elif score >= 18:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    if not reasons:
        reasons.append("stable operating profile")

    return DelayPrediction(
        risk_level=risk_level,
        expected_delay_min=int(minutes),
        reason_tags=tuple(dict.fromkeys(reasons)),
        score=min(round(score + congestion_score * 0.12, 1), 100.0),
        congestion_level=congestion_level,
        schedule_pressure_level=schedule_pressure_level,
        holding_detected=holding_detected,
        go_around_detected=go_around_detected,
        diversion_suspected=diversion_suspected,
        ml_prob=round(ml_prob, 3),
    )


def compute_arrival_estimate(
    flight_row: pd.Series,
    airports: Iterable[dict],
    delay_minutes: int,
    now_ts: int,
    _airport_lookup: "dict | None" = None,
) -> ArrivalEstimate:
    """
    Estimate arrival time for the selected aircraft.

    Priority order:
      1. arr_estimated_ts  (airline-reported estimate, most accurate)
      2. arr_time_ts       (scheduled arrival)
      3. Speed-based       (groundspeed × remaining distance)
    Delay minutes are added on top of whichever baseline is used.
    """
    import datetime as _dt

    arr_iata = str(flight_row.get("arr_iata") or "").strip().upper()
    latitude = _safe_float(flight_row.get("latitude"))
    longitude = _safe_float(flight_row.get("longitude"))
    speed_kts = _safe_float(flight_row.get("speed_kts"))

    airport_by_code = _airport_lookup if _airport_lookup is not None else {airport["iata"]: airport for airport in airports}
    airport = airport_by_code.get(arr_iata)
    if airport is None or pd.isna(latitude) or pd.isna(longitude):
        return ArrivalEstimate(
            distance_km=None,
            eta_minutes=None,
            eta_label_utc="ETA unavailable",
            arrival_summary="Selected route does not have a supported destination estimate.",
        )

    distance_km = haversine_km(
        airport["lat"],
        airport["lng"],
        pd.Series([latitude]),
        pd.Series([longitude]),
    ).iloc[0]
    if pd.isna(distance_km):
        return ArrivalEstimate(
            distance_km=None,
            eta_minutes=None,
            eta_label_utc="ETA unavailable",
            arrival_summary="Distance to destination could not be estimated.",
        )

    # ── Baseline ETA: scheduled or estimated timestamp ───────────────────
    arr_estimated = pd.to_numeric(flight_row.get("arr_estimated_ts"), errors="coerce")
    arr_scheduled = pd.to_numeric(flight_row.get("arr_time_ts"), errors="coerce")

    if pd.notna(arr_estimated) and arr_estimated > now_ts:
        base_ts = int(arr_estimated)
        source = "estimated"
    elif pd.notna(arr_scheduled) and arr_scheduled > now_ts:
        base_ts = int(arr_scheduled)
        source = "scheduled"
    else:
        # Fall back to speed-based calculation
        if pd.notna(speed_kts) and speed_kts > 120:
            km_per_hour = float(speed_kts) * 1.852
            travel_minutes = int(round((float(distance_km) / km_per_hour) * 60))
        else:
            travel_minutes = 45
        base_ts = int(now_ts + travel_minutes * 60)
        source = "speed"

    # Apply predicted delay on top of baseline
    eta_ts = int(base_ts + max(delay_minutes, 0) * 60)
    eta_minutes = max(int((eta_ts - now_ts) / 60), 1)

    # Format in IST (UTC+5:30) — flights are Indian domestic
    utc_dt = _dt.datetime.fromtimestamp(eta_ts, tz=_dt.timezone.utc)
    ist_dt = utc_dt + _dt.timedelta(hours=5, minutes=30)
    eta_label_ist = ist_dt.strftime("%H:%M IST")

    summary = f"Expected into {arr_iata} in ~{eta_minutes} min at {eta_label_ist} ({source})."
    return ArrivalEstimate(
        distance_km=round(float(distance_km), 1),
        eta_minutes=eta_minutes,
        eta_label_utc=eta_label_ist,   # field stores IST; name retained for API compat
        arrival_summary=summary,
    )


def enrich_flights_with_predictions(
    flights_df: pd.DataFrame,
    airports: Iterable[dict],
    airport_metrics: pd.DataFrame,
    airport_schedule_pressure: dict[str, dict],
    now_ts: int,
) -> pd.DataFrame:
    """
    Add lightweight per-aircraft prediction columns for map popups and search.
    """
    if flights_df.empty:
        return flights_df.copy()

    enriched = flights_df.copy()
    airports = list(airports)  # materialise once — sub-functions iterate it per call
    # Pre-build shared lookup structures once instead of rebuilding per flight
    _lookup: dict = {a["iata"]: a for a in airports}
    _mmap: dict = (
        airport_metrics.set_index("airport_iata").to_dict("index")
        if not airport_metrics.empty else {}
    )
    risk_levels: list[str] = []
    delay_minutes: list[int] = []
    eta_labels: list[str] = []
    delay_reasons: list[str] = []
    pred_scores: list[float] = []
    pred_ml_probs: list[float] = []
    pred_holdings: list[bool] = []
    pred_go_arounds: list[bool] = []
    pred_diversions: list[bool] = []

    for _row_t in enriched.itertuples(index=False):
        row = pd.Series(_row_t._asdict())
        prediction = compute_delay_prediction(
            row,
            airports=airports,
            airport_metrics=airport_metrics,
            airport_schedule_pressure=airport_schedule_pressure,
            now_ts=now_ts,
            _airport_lookup=_lookup,
            _metric_map=_mmap,
        )
        arrival = compute_arrival_estimate(
            row,
            airports=airports,
            delay_minutes=prediction.expected_delay_min,
            now_ts=now_ts,
            _airport_lookup=_lookup,
        )
        risk_levels.append(prediction.risk_level)
        delay_minutes.append(prediction.expected_delay_min)
        eta_labels.append(arrival.eta_label_utc)
        delay_reasons.append(", ".join(prediction.reason_tags[:5]))
        pred_scores.append(prediction.score)
        pred_ml_probs.append(prediction.ml_prob)
        pred_holdings.append(prediction.holding_detected)
        pred_go_arounds.append(prediction.go_around_detected)
        pred_diversions.append(prediction.diversion_suspected)

    enriched["predicted_delay_risk"] = risk_levels
    enriched["predicted_delay_min"] = delay_minutes
    enriched["predicted_eta_utc"] = eta_labels
    enriched["predicted_delay_reason"] = delay_reasons
    enriched["pred_score"] = pred_scores
    enriched["pred_ml_prob"] = pred_ml_probs
    enriched["pred_holding"] = pred_holdings
    enriched["pred_go_around"] = pred_go_arounds
    enriched["pred_diversion"] = pred_diversions

    # ── Arrival Flow Risk Index (AFRI) ─────────────────────────────────────
    enriched = compute_afri(enriched, airport_metrics, airport_schedule_pressure)

    return enriched


# ── Arrival Flow Risk Index (AFRI) ─────────────────────────────────────────────
# A per-flight composite score (0–100) that captures how risky the arrival
# environment is *right now* at each flight's destination.  Five live signals
# are blended with configurable weights so airline dispatchers can triage.

_AFRI_WEIGHTS = {
    "density":   0.30,   # how many aircraft are converging on the same airport
    "bunching":  0.20,   # flights arriving in the same 30-min window
    "weather":   0.25,   # METAR severity at destination
    "schedule":  0.15,   # upcoming schedule pressure
    "stale":     0.10,   # lost-contact inbound flights add uncertainty
}


def _afri_level(score: float) -> str:
    if score >= 75:
        return "Critical"
    if score >= 50:
        return "High"
    if score >= 25:
        return "Elevated"
    return "Normal"


def compute_afri(
    flights_df: pd.DataFrame,
    airport_metrics: pd.DataFrame,
    airport_schedule_pressure: dict[str, dict],
) -> pd.DataFrame:
    """Compute per-flight Arrival Flow Risk Index and append columns."""
    if flights_df.empty:
        out = flights_df.copy()
        out["afri_score"] = 0
        out["afri_level"] = "Normal"
        out["afri_drivers"] = ""
        return out

    df = flights_df.copy()

    # Pre-build lookup dicts for airport-level signals
    # Density: normalize inbound_count against the busiest airport
    density_map: dict[str, float] = {}
    bunching_map: dict[str, float] = {}
    if not airport_metrics.empty:
        max_inbound = max(float(airport_metrics["inbound_count"].max() or 1), 1.0)
        for _, row in airport_metrics.iterrows():
            iata = str(row["airport_iata"])
            density_map[iata] = min(float(row["inbound_count"]) / max_inbound, 1.0)
            # Bunching: approach_count (low-alt + slow = imminent) relative to inbound
            inb = max(float(row["inbound_count"]), 1.0)
            bunching_map[iata] = min(float(row.get("approach_count", 0)) / inb, 1.0)

    # Schedule pressure lookup
    sched_map: dict[str, float] = {}
    for iata, sp in airport_schedule_pressure.items():
        level = str(sp.get("pressure_level", "Low"))
        sched_map[iata] = {"Low": 0.1, "Moderate": 0.45, "High": 0.75, "Extreme": 1.0}.get(level, 0.1)

    # Weather severity lookup (already on the df)
    wx_score_map = {"Low": 0.0, "Moderate": 0.45, "Severe": 1.0}

    # Count stale inbound flights per destination
    stale_inbound: dict[str, int] = {}
    total_inbound: dict[str, int] = {}
    if "stale_minutes" in df.columns:
        for _, row in df.iterrows():
            arr = str(row.get("arr_iata", ""))
            if not arr:
                continue
            total_inbound[arr] = total_inbound.get(arr, 0) + 1
            sm = row.get("stale_minutes")
            if pd.notna(sm) and float(sm) >= 5:
                stale_inbound[arr] = stale_inbound.get(arr, 0) + 1

    scores: list[int] = []
    levels: list[str] = []
    drivers: list[str] = []

    for _, row in df.iterrows():
        arr = str(row.get("arr_iata", ""))
        tags: list[str] = []

        # 1. Density
        d = density_map.get(arr, 0.0)
        if d >= 0.6:
            tags.append("high inbound density")

        # 2. Bunching
        b = bunching_map.get(arr, 0.0)
        if b >= 0.5:
            tags.append("arrival bunching")

        # 3. Weather at destination
        wx_sev = str(row.get("weather_severity", "Low"))
        w = wx_score_map.get(wx_sev, 0.0)
        if w >= 0.45:
            tags.append("dest weather")

        # 4. Schedule pressure
        s = sched_map.get(arr, 0.1)
        if s >= 0.45:
            tags.append("schedule pressure")

        # 5. Stale inbound uncertainty
        tot = max(total_inbound.get(arr, 1), 1)
        st_count = stale_inbound.get(arr, 0)
        st = min(st_count / tot, 1.0)
        if st >= 0.2:
            tags.append("stale data uncertainty")

        # Weighted composite → 0-100
        raw = (
            _AFRI_WEIGHTS["density"]  * d
            + _AFRI_WEIGHTS["bunching"] * b
            + _AFRI_WEIGHTS["weather"]  * w
            + _AFRI_WEIGHTS["schedule"] * s
            + _AFRI_WEIGHTS["stale"]    * st
        )
        score = int(min(round(raw * 100), 100))
        scores.append(score)
        levels.append(_afri_level(score))
        drivers.append(", ".join(tags[:3]) if tags else "stable arrival")

    df["afri_score"] = scores
    df["afri_level"] = levels
    df["afri_drivers"] = drivers
    return df


# ── CO₂ / Fuel estimation ─────────────────────────────────────────────────────

# Average fuel burn in kg per km by aircraft family (cruise phase approximation).
# Sources: ICAO Carbon Emissions Calculator methodology, EASA type certificates.
FUEL_BURN_KG_PER_KM: dict[str, float] = {
    "B744": 11.2, "B748": 10.8, "B742": 11.5, "B743": 11.3, "BLCF": 11.0,
    "A388": 12.0, "A389": 12.0,
    "A342": 7.8, "A343": 8.0, "A345": 8.5, "A346": 9.0,
    "B772": 7.6, "B773": 8.0, "B77L": 7.8, "B77W": 8.2, "B778": 7.4, "B779": 7.6,
    "A35K": 5.8, "A358": 5.6, "A359": 5.7,
    "A332": 6.2, "A333": 6.4, "A338": 5.9, "A339": 5.8,
    "B787": 5.2, "B788": 5.0, "B789": 5.3, "B78X": 5.1,
    "B762": 5.8, "B763": 6.0, "B764": 6.2,
    "B752": 4.2, "B753": 4.4,
    "A318": 2.6, "A319": 2.7, "A320": 2.9, "A321": 3.2,
    "A20N": 2.5, "A21N": 2.8, "A19N": 2.4,
    "B731": 3.0, "B732": 3.0, "B733": 2.9, "B734": 2.9,
    "B735": 2.8, "B736": 2.8, "B737": 2.9, "B738": 2.8,
    "B739": 2.9, "B37M": 2.5, "B38M": 2.5, "B39M": 2.6, "B3XM": 2.5,
    "E170": 2.2, "E175": 2.3, "E190": 2.4, "E195": 2.5, "E290": 2.3, "E295": 2.4,
    "CRJ1": 1.8, "CRJ2": 1.8, "CRJ7": 2.0, "CRJ9": 2.1, "CRJX": 2.2,
    "AT72": 1.2, "AT76": 1.3, "DH8D": 1.4,
}

# Passenger capacity (typical 1-class/high-density) for per-pax calculations
TYPICAL_SEATS: dict[str, int] = {
    "B744": 416, "B748": 410, "A388": 555, "A346": 380,
    "B772": 314, "B773": 368, "B77W": 350, "B778": 325, "B779": 350,
    "A359": 325, "A35K": 369, "A332": 277, "A333": 292, "A339": 287,
    "B788": 242, "B789": 290, "B78X": 318, "B763": 218,
    "B752": 200, "B753": 243,
    "A318": 132, "A319": 156, "A320": 180, "A321": 220, "A20N": 180, "A21N": 220,
    "B737": 162, "B738": 189, "B739": 189, "B37M": 172, "B38M": 178,
    "E190": 100, "E195": 120, "CRJ9": 90,
    "AT72": 72, "AT76": 72, "DH8D": 78,
}

CO2_PER_KG_FUEL = 3.16  # kg CO₂ per kg jet fuel (ICAO)

# ── Phase-of-flight correction factors ─────────────────────────────────────────
# Climb/descent burns significantly more than cruise. The shorter the sector,
# the larger the fraction spent climbing/descending.
# These multipliers are applied on top of the cruise-only fuel estimate.
_CLIMB_DESCENT_FACTOR: list[tuple[float, float]] = [
    # (max_distance_km, multiplier)
    (300, 1.35),    # ultra-short: climb/descent ~35% of flight
    (600, 1.25),    # short domestic
    (1200, 1.18),   # medium domestic
    (2500, 1.12),   # long domestic / short international
    (5000, 1.08),   # medium international
    (float('inf'), 1.05),  # ultra-long-haul
]

# Taxi + APU flat fuel adder (kg) by size class
# Covers average taxi time (~15-20 min India) + APU running + engine start
_WIDEBODY_CODES = frozenset({
    "B744", "B748", "B742", "B743", "BLCF", "A388", "A389",
    "A342", "A343", "A345", "A346",
    "B772", "B773", "B77L", "B77W", "B778", "B779",
    "A35K", "A358", "A359", "A332", "A333", "A338", "A339",
    "B787", "B788", "B789", "B78X", "B762", "B763", "B764",
})
_TURBOPROP_CODES = frozenset({"AT72", "AT76", "DH8D"})

def _taxi_fuel_kg(icao: str) -> float:
    """Flat taxi + APU fuel adder based on aircraft size class."""
    if icao in _WIDEBODY_CODES:
        return 300.0
    if icao in _TURBOPROP_CODES:
        return 60.0
    return 150.0  # narrow-body default

# Routing factor: airways, SIDs/STARs, ATC vectoring add distance over great-circle.
# Domestic India routes are more indirect (congested airspace, military zones).
_ROUTING_FACTOR_DOMESTIC = 1.10   # +10% for <2500 km
_ROUTING_FACTOR_INTL = 1.05       # +5% for >=2500 km


@dataclass(frozen=True)
class FuelEstimate:
    aircraft_icao: str
    distance_km: float
    actual_distance_km: float        # after routing factor
    fuel_burn_kg: float
    fuel_cruise_only_kg: float       # without corrections (for comparison)
    co2_kg: float
    co2_per_pax_kg: float | None
    fuel_rate_label: str
    efficiency_label: str
    correction_notes: tuple[str, ...]


def compute_fuel_estimate(
    aircraft_icao: str,
    distance_km: float,
) -> FuelEstimate:
    """Estimate fuel burn and CO₂ with phase-of-flight corrections."""
    icao = str(aircraft_icao or "").strip().upper()
    gc_dist = max(float(distance_km or 0), 0.0)
    notes: list[str] = []

    rate = FUEL_BURN_KG_PER_KM.get(icao, 3.0)  # default ~narrow-body

    # 1) Routing factor — actual flight path > great-circle
    routing_factor = _ROUTING_FACTOR_DOMESTIC if gc_dist < 2500 else _ROUTING_FACTOR_INTL
    actual_dist = gc_dist * routing_factor
    notes.append(f"routing +{int((routing_factor - 1) * 100)}%")

    # 2) Cruise-only baseline (for comparison)
    cruise_only_kg = round(rate * actual_dist, 1)

    # 3) Climb/descent phase multiplier
    climb_factor = 1.0
    for max_km, factor in _CLIMB_DESCENT_FACTOR:
        if actual_dist <= max_km:
            climb_factor = factor
            break
    fuel_kg = rate * actual_dist * climb_factor
    notes.append(f"climb/descent +{int((climb_factor - 1) * 100)}%")

    # 4) Taxi + APU adder
    taxi = _taxi_fuel_kg(icao)
    fuel_kg += taxi
    notes.append(f"taxi/APU +{int(taxi)}kg")

    fuel_kg = round(fuel_kg, 1)
    co2_kg = round(fuel_kg * CO2_PER_KG_FUEL, 1)

    seats = TYPICAL_SEATS.get(icao)
    co2_per_pax = round(co2_kg / seats, 1) if seats and co2_kg > 0 else None

    if co2_per_pax is not None:
        if co2_per_pax < 80:
            eff = "Excellent"
        elif co2_per_pax < 140:
            eff = "Good"
        elif co2_per_pax < 220:
            eff = "Average"
        else:
            eff = "High"
    else:
        eff = "Unknown"

    return FuelEstimate(
        aircraft_icao=icao,
        distance_km=round(gc_dist, 1),
        actual_distance_km=round(actual_dist, 1),
        fuel_burn_kg=fuel_kg,
        fuel_cruise_only_kg=cruise_only_kg,
        co2_kg=co2_kg,
        co2_per_pax_kg=co2_per_pax,
        fuel_rate_label=f"{rate} kg/km",
        efficiency_label=eff,
        correction_notes=tuple(notes),
    )


# ── Anomaly detection ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FlightAnomaly:
    anomaly_type: str          # "circling", "diverting", "squawk_emergency", "speed_anomaly", "altitude_anomaly"
    severity: str              # "info", "warning", "alert"
    description: str
    score: float


def detect_anomalies(
    flight_row: pd.Series,
    trail_positions: list[dict],
    airports: Iterable[dict],
) -> list[FlightAnomaly]:
    """Detect circling, diverting, speed/altitude anomalies, and emergency squawks."""
    anomalies: list[FlightAnomaly] = []

    altitude_ft = _safe_float(flight_row.get("altitude_ft"))
    speed_kts = _safe_float(flight_row.get("speed_kts"))
    squawk = str(flight_row.get("squawk") or "").strip()
    arr_iata = str(flight_row.get("arr_iata") or "").strip().upper()
    latitude = _safe_float(flight_row.get("latitude"))
    longitude = _safe_float(flight_row.get("longitude"))

    # ── Stale position (lost contact) ──────────────────────────────────
    stale_min = _safe_float(flight_row.get("stale_minutes"))
    if pd.notna(stale_min) and stale_min >= 3:
        if stale_min >= 10:
            anomalies.append(FlightAnomaly(
                anomaly_type="lost_contact",
                severity="alert",
                description=f"Position not updated for {stale_min:.0f} min — possible lost contact",
                score=round(min(stale_min * 5, 100), 1),
            ))
        elif stale_min >= 5:
            anomalies.append(FlightAnomaly(
                anomaly_type="stale_position",
                severity="warning",
                description=f"Position data {stale_min:.0f} min stale",
                score=round(min(stale_min * 4, 80), 1),
            ))
        else:
            anomalies.append(FlightAnomaly(
                anomaly_type="stale_position",
                severity="info",
                description=f"Position data {stale_min:.0f} min old",
                score=15.0,
            ))

    # ── Emergency squawk codes ─────────────────────────────────────────────
    if squawk in ("7500", "7600", "7700"):
        labels = {"7500": "Hijack", "7600": "Radio failure", "7700": "Emergency"}
        anomalies.append(FlightAnomaly(
            anomaly_type="squawk_emergency",
            severity="alert",
            description=f"Squawk {squawk} — {labels[squawk]}",
            score=100.0,
        ))

    # ── Circling detection (heading variance in trail) ─────────────────────
    if len(trail_positions) >= 5:
        recent = trail_positions[-8:]
        headings = [p.get("heading") for p in recent if p.get("heading") is not None]
        if len(headings) >= 4:
            deltas = []
            for i in range(1, len(headings)):
                d = abs(headings[i] - headings[i - 1])
                if d > 180:
                    d = 360 - d
                deltas.append(d)
            total_turn = sum(deltas)
            if total_turn >= 270:
                anomalies.append(FlightAnomaly(
                    anomaly_type="circling",
                    severity="warning",
                    description=f"Aircraft circling — {total_turn:.0f}° heading change in recent trail",
                    score=round(min(total_turn / 3.6, 100), 1),
                ))

    # ── Diverting detection (moving away from destination) ─────────────────
    if arr_iata and arr_iata != "N/A" and len(trail_positions) >= 3 and pd.notna(latitude) and pd.notna(longitude):
        airport_by_code = {a["iata"]: a for a in airports}
        dest = airport_by_code.get(arr_iata)
        if dest:
            current_dist = haversine_km(
                dest["lat"], dest["lng"],
                pd.Series([latitude]), pd.Series([longitude]),
            ).iloc[0]

            older = trail_positions[-3]
            if older.get("lat") is not None and older.get("lng") is not None:
                older_dist = haversine_km(
                    dest["lat"], dest["lng"],
                    pd.Series([older["lat"]]), pd.Series([older["lng"]]),
                ).iloc[0]

                if pd.notna(current_dist) and pd.notna(older_dist):
                    if current_dist > older_dist + 40 and current_dist > 100:
                        anomalies.append(FlightAnomaly(
                            anomaly_type="diverting",
                            severity="warning",
                            description=f"Moving away from {arr_iata} — now {current_dist:.0f} km (+{current_dist - older_dist:.0f} km)",
                            score=round(min((current_dist - older_dist) * 1.5, 100), 1),
                        ))

    # ── Speed anomaly ──────────────────────────────────────────────────────
    if pd.notna(speed_kts) and pd.notna(altitude_ft):
        if altitude_ft > 25000 and speed_kts < 280:
            anomalies.append(FlightAnomaly(
                anomaly_type="speed_anomaly",
                severity="info",
                description=f"Unusually slow at cruise altitude — {speed_kts:.0f} kts at {altitude_ft:,.0f} ft",
                score=40.0,
            ))
        if altitude_ft < 5000 and speed_kts > 350:
            anomalies.append(FlightAnomaly(
                anomaly_type="speed_anomaly",
                severity="warning",
                description=f"High speed at low altitude — {speed_kts:.0f} kts at {altitude_ft:,.0f} ft",
                score=65.0,
            ))

    # ── Altitude anomaly ────────────────────────────────────────────────
    if pd.notna(altitude_ft) and len(trail_positions) >= 3:
        recent_alts = [p.get("altitude_ft") for p in trail_positions[-4:] if p.get("altitude_ft") is not None]
        if len(recent_alts) >= 2:
            alt_change = abs(recent_alts[-1] - recent_alts[0])
            if alt_change > 15000:
                anomalies.append(FlightAnomaly(
                    anomaly_type="altitude_anomaly",
                    severity="warning",
                    description=f"Rapid altitude change — {alt_change:,.0f} ft in recent trail",
                    score=round(min(alt_change / 200, 100), 1),
                ))

    return anomalies


def detect_all_anomalies(
    flights_df: pd.DataFrame,
    trails: dict,
    airports: Iterable[dict],
) -> pd.DataFrame:
    """Run anomaly detection across all flights. Returns DataFrame of flagged flights."""
    rows: list[dict] = []
    for _, row in flights_df.iterrows():
        hex_code = str(row.get("hex", ""))
        trail_positions = trails.get(hex_code, {}).get("positions", []) if hex_code else []
        anomalies = detect_anomalies(row, trail_positions, airports)
        if anomalies:
            top = max(anomalies, key=lambda a: a.score)
            rows.append({
                "flight_iata": row.get("flight_iata", "N/A"),
                "hex": hex_code,
                "anomaly_type": top.anomaly_type,
                "severity": top.severity,
                "description": top.description,
                "score": top.score,
                "all_anomalies": "; ".join(a.description for a in anomalies),
                "anomaly_count": len(anomalies),
            })
    if not rows:
        return pd.DataFrame(columns=["flight_iata", "hex", "anomaly_type", "severity", "description", "score", "all_anomalies", "anomaly_count"])
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
