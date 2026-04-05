"""
opensky_trino.py
----------------
OpenSky Network – Trino historical database client (via pyopensky).

Authenticates with OpenSky's KeyCloak OAuth2 using the credentials in .env:
  OPENSKY_USERNAME=your_opensky_username
  OPENSKY_PASSWORD=your_opensky_password

Provides:
  test_connection()          → bool
  query_indian_flights()     → DataFrame  (raw flight records)
  query_route_delay_stats()  → DataFrame  (per-route aggregates for ML features)
  query_airline_stats()      → DataFrame  (per-airline aggregates)
  build_ml_dataset()         → DataFrame  (full feature matrix for model training)

OpenSky table notes (MUST follow to avoid getting banned):
  - flights_data4  → partition column is `day`  (unix day start)
  - state_vectors_data4 → partition column is `hour` (unix hour start)
  - Always include WHERE day=... or WHERE hour=... in every query
"""

import logging
import os
from datetime import datetime, timezone, timedelta

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


INDIAN_PREFIXES = [
    "AIC",  # Air India
    "IGO",  # IndiGo
    "SEJ",  # SpiceJet
    "GOW",  # Go First
    "AXB",  # Air India Express
    "VTI",  # Vistara
    "IST",  # AirAsia India
]


INDIAN_AIRPORTS = [
    "VIDP", "VABB", "VOMM", "VOBL", "VOCI", "VECC", "VOHY",
    "VEGT", "VIJP", "VIPT", "VABV", "VAJB", "VAAH", "VOPB",
    "VIAG", "VIAR", "VILK", "VIDL", "VEBS", "VEPY",
]


def _trino():
    """Return a pyopensky Trino instance (credentials from env/config)."""
    from pyopensky.trino import Trino
    return Trino()


def _date_to_day_ts(date_str: str) -> int:
    """Convert 'YYYY-MM-DD' to unix timestamp of midnight UTC (day partition)."""
    return int(datetime.strptime(date_str, "%Y-%m-%d")
               .replace(tzinfo=timezone.utc).timestamp())


def _day_range_list(start_date: str, end_date: str):
    """Yield (day_ts, date_str) for each day in [start, end] inclusive."""
    d = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    while d <= end:
        yield int(d.timestamp()), d.strftime("%Y-%m-%d")
        d += timedelta(days=1)




def test_connection() -> bool:
    """
    Quick ping. Returns True if authenticated and Trino responds.
    On first run this will trigger an OAuth2 token fetch (username+password
    from .env — no browser needed when credentials are present).
    """
    try:
        t = _trino()
        df = t.query("SELECT 1 AS ok", cached=False)
        ok = not df.empty
        if ok:
            logger.info("OpenSky Trino: connected OK")
        return ok
    except Exception as e:
        logger.error(f"OpenSky Trino connection failed: {e}")
        return False




def query_indian_flights(
    start_date: str = "2024-01-01",
    end_date:   str = "2024-01-31",
    limit:      int = 200_000,
) -> pd.DataFrame:
    """
    Pull historical flight records from flights_data4 for Indian airlines.

    Returns one row per flight:
      icao24, callsign, dep_icao, arr_icao,
      firstseen (unix), lastseen (unix),
      fl_duration_min, dep_hour_ist, dep_dow

    Queries one day at a time (required for proper `day` partitioning).
    """
    t = _trino()
    prefix_filter = " OR ".join(f"callsign LIKE '{p}%'" for p in INDIAN_PREFIXES)
    airport_csv   = ", ".join(f"'{a}'" for a in INDIAN_AIRPORTS)

    chunks = []
    total = 0
    for day_ts, date_str in _day_range_list(start_date, end_date):
        if total >= limit:
            break
        sql = f"""
        SELECT
            icao24,
            TRIM(callsign)                                    AS callsign,
            estdepartureairport                               AS dep_icao,
            estarrivalairport                                 AS arr_icao,
            firstseen,
            lastseen,
            (lastseen - firstseen) / 60.0                     AS fl_duration_min,
            (CAST(firstseen AS BIGINT) + 19800) / 3600 % 24  AS dep_hour_ist,
            (CAST(firstseen AS BIGINT) + 19800) / 86400 % 7  AS dep_dow
        FROM flights_data4
        WHERE day = {day_ts}
          AND ({prefix_filter})
          AND estdepartureairport IN ({airport_csv})
          AND estarrivalairport   IN ({airport_csv})
          AND callsign IS NOT NULL
        LIMIT {min(10000, limit - total)}
        """
        try:
            df = t.query(sql)
            if not df.empty:
                chunks.append(df)
                total += len(df)
                logger.info(f"  {date_str}: {len(df):,} flights (total {total:,})")
        except Exception as e:
            logger.warning(f"  {date_str}: query failed — {e}")

    if not chunks:
        logger.warning("No Indian flight data returned")
        return pd.DataFrame()

    result = pd.concat(chunks, ignore_index=True)
    logger.info(f"query_indian_flights: {len(result):,} total rows")
    return result


def query_route_delay_stats(
    start_date: str = "2024-01-01",
    end_date:   str = "2024-03-31",
) -> pd.DataFrame:
    """
    Aggregate per-route stats across the date range:
      dep_icao, arr_icao, flight_count,
      avg_duration_min, min_duration_min, max_duration_min, median_duration_min

    These are strong ML features (route baseline duration).
    Queries one day at a time and aggregates in Python.
    """
    t = _trino()
    prefix_filter = " OR ".join(f"callsign LIKE '{p}%'" for p in INDIAN_PREFIXES)
    airport_csv   = ", ".join(f"'{a}'" for a in INDIAN_AIRPORTS)

    chunks = []
    for day_ts, date_str in _day_range_list(start_date, end_date):
        sql = f"""
        SELECT
            estdepartureairport                              AS dep_icao,
            estarrivalairport                                AS arr_icao,
            COUNT(*)                                         AS cnt,
            AVG((lastseen - firstseen) / 60.0)              AS avg_dur
        FROM flights_data4
        WHERE day = {day_ts}
          AND ({prefix_filter})
          AND estdepartureairport IN ({airport_csv})
          AND estarrivalairport   IN ({airport_csv})
          AND (lastseen - firstseen) BETWEEN 600 AND 18000
        GROUP BY 1, 2
        """
        try:
            df = t.query(sql)
            if not df.empty:
                chunks.append(df)
        except Exception as e:
            logger.warning(f"  {date_str}: {e}")

    if not chunks:
        return pd.DataFrame()

    raw = pd.concat(chunks, ignore_index=True)
    # Aggregate across days: weighted average duration + total count
    agg = (
        raw.groupby(["dep_icao", "arr_icao"])
        .apply(lambda g: pd.Series({
            "flight_count":      g["cnt"].sum(),
            "avg_duration_min":  (g["avg_dur"] * g["cnt"]).sum() / g["cnt"].sum(),
        }), include_groups=False)
        .reset_index()
    )
    # Keep only well-sampled routes
    agg = agg[agg["flight_count"] >= 20].sort_values("flight_count", ascending=False)
    logger.info(f"query_route_delay_stats: {len(agg):,} routes")
    return agg


def query_airline_stats(
    start_date: str = "2024-01-01",
    end_date:   str = "2024-03-31",
) -> pd.DataFrame:
    """Per-airline aggregate: flight_count, avg_duration_min."""
    t = _trino()
    airport_csv = ", ".join(f"'{a}'" for a in INDIAN_AIRPORTS)

    prefix_cases = " ".join(
        f"WHEN callsign LIKE '{p}%' THEN '{p}'" for p in INDIAN_PREFIXES
    )

    chunks = []
    for day_ts, date_str in _day_range_list(start_date, end_date):
        sql = f"""
        SELECT
            CASE {prefix_cases} ELSE 'OTHER' END AS airline_prefix,
            COUNT(*)                              AS cnt,
            AVG((lastseen - firstseen) / 60.0)   AS avg_dur
        FROM flights_data4
        WHERE day = {day_ts}
          AND estdepartureairport IN ({airport_csv})
          AND estarrivalairport   IN ({airport_csv})
          AND (lastseen - firstseen) BETWEEN 600 AND 18000
        GROUP BY 1
        """
        try:
            df = t.query(sql)
            if not df.empty:
                chunks.append(df)
        except Exception as e:
            logger.warning(f"  {date_str}: {e}")

    if not chunks:
        return pd.DataFrame()

    raw = pd.concat(chunks, ignore_index=True)
    agg = (
        raw.groupby("airline_prefix")
        .apply(lambda g: pd.Series({
            "flight_count":     g["cnt"].sum(),
            "avg_duration_min": (g["avg_dur"] * g["cnt"]).sum() / g["cnt"].sum(),
        }), include_groups=False)
        .reset_index()
        .sort_values("flight_count", ascending=False)
    )
    return agg




def _route_delay_stats_from_flights(flights: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-route aggregates locally from an already-fetched flight frame.

    This avoids a second full historical Trino sweep inside build_ml_dataset(),
    which is the main reason long training windows feel slow.
    """
    if flights.empty:
        return pd.DataFrame(columns=["dep_icao", "arr_icao", "flight_count", "avg_duration_min"])

    valid = flights.copy()
    valid["fl_duration_min"] = pd.to_numeric(valid["fl_duration_min"], errors="coerce")
    valid = valid.dropna(subset=["dep_icao", "arr_icao", "fl_duration_min"])
    valid = valid[valid["fl_duration_min"].between(10, 300)]
    if valid.empty:
        return pd.DataFrame(columns=["dep_icao", "arr_icao", "flight_count", "avg_duration_min"])

    agg = (
        valid.groupby(["dep_icao", "arr_icao"], as_index=False)
        .agg(
            flight_count=("fl_duration_min", "size"),
            avg_duration_min=("fl_duration_min", "mean"),
            std_duration_min=("fl_duration_min", "std"),
        )
        .sort_values("flight_count", ascending=False)
    )
    agg["std_duration_min"] = agg["std_duration_min"].fillna(0.0)
    agg = agg[agg["flight_count"] >= 5]
    logger.info(
        "Local route stats ready: %s routes from %s fetched flights",
        f"{len(agg):,}",
        f"{len(valid):,}",
    )
    return agg

def build_ml_dataset(
    start_date: str = "2024-01-01",
    end_date:   str = "2024-03-31",
    limit:      int = 200_000,
) -> pd.DataFrame:
    """
    Build a labelled feature DataFrame for ML training.

    Features:
      - route_id      (dep_icao + arr_icao)
      - airline_prefix
      - dep_hour_ist  (0-23 in IST timezone)
      - dep_dow       (day of week, adjusted)
      - dep_period    (night/morning/afternoon/evening)
      - is_weekend    (0/1)
      - fl_duration_min       (actual flight time)
      - route_avg_duration    (historical route average from same window)
      - delay_proxy_min       (actual − route_avg  →  proxy for delay)

    Target:
      is_delayed  = 1 if delay_proxy_min > 15 minutes

    NOTE: This is a proxy target. True delay requires scheduled time data
    (available from AirLabs API for live flights). Combine both for best results.
    """
    logger.info(f"Building ML dataset {start_date} → {end_date} …")

    flights    = query_indian_flights(start_date, end_date, limit)
    if flights.empty:
        logger.warning("No flight data — check credentials or date range")
        return pd.DataFrame()

    route_stats = _route_delay_stats_from_flights(flights)

    
    df = flights.merge(
        route_stats[["dep_icao", "arr_icao", "avg_duration_min", "flight_count", "std_duration_min"]].rename(
            columns={
                "avg_duration_min": "route_avg_duration",
                "flight_count": "route_flight_count",
                "std_duration_min": "route_std_duration",
            }
        ),
        on=["dep_icao", "arr_icao"],
        how="left",
    )

   
    df["fl_duration_min"]    = pd.to_numeric(df["fl_duration_min"],    errors="coerce")
    df["route_avg_duration"] = pd.to_numeric(df["route_avg_duration"], errors="coerce")


    df["delay_proxy_min"] = df["fl_duration_min"] - df["route_avg_duration"]
    df["is_delayed"]      = (df["delay_proxy_min"] > 15).astype(int)


    df["airline_prefix"] = "OTHER"
    for p in INDIAN_PREFIXES:
        mask = df["callsign"].str.startswith(p, na=False)
        df.loc[mask, "airline_prefix"] = p


    df["route_id"] = df["dep_icao"] + "-" + df["arr_icao"]


    bins   = [0, 6, 12, 18, 24]
    labels = ["night", "morning", "afternoon", "evening"]
    df["dep_period"] = pd.cut(
        pd.to_numeric(df["dep_hour_ist"], errors="coerce"),
        bins=bins, labels=labels, right=False,
    ).astype(str)  # <-- avoid Categorical dtype leaking into training pipeline


    df["is_weekend"] = (
        pd.to_datetime(pd.to_numeric(df["firstseen"], errors="coerce") + 19800, unit="s", utc=True)
        .dt.weekday
        .isin([5, 6])  # 5=Saturday, 6=Sunday (pandas weekday: 0=Mon)
        .astype(int)
    )

    # dep_month: month of year (1-12) — captures monsoon/fog/festive seasonality
    if "firstseen" in df.columns:
        df["dep_month"] = (
            pd.to_datetime(pd.to_numeric(df["firstseen"], errors="coerce"), unit="s", utc=True)
            .dt.tz_convert("Asia/Kolkata")
            .dt.month
            .fillna(1)
            .astype(int)
        )
    else:
        df["dep_month"] = 1

    # duration_ratio: relative deviation from route average (1.0 = on-time, >1 = slower)
    df["duration_ratio"] = (
        df["fl_duration_min"] / df["route_avg_duration"].replace(0, float("nan"))
    ).fillna(1.0).clip(0.5, 2.0)

    # route_delay_rate: historical fraction of delayed flights on this route
    _rdm = df.groupby("route_id")["is_delayed"].mean()
    df["route_delay_rate"] = df["route_id"].map(_rdm).fillna(df["is_delayed"].mean())

    # Drop rows missing key fields
    df = df.dropna(subset=["dep_icao", "arr_icao", "fl_duration_min", "route_avg_duration"])

    delay_rate = df["is_delayed"].mean() * 100
    logger.info(
        f"ML dataset ready: {len(df):,} flights | "
        f"{delay_rate:.1f}% delayed | "
        f"{df['route_id'].nunique()} routes | "
        f"{df['airline_prefix'].nunique()} airlines"
    )
    return df



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    print("Testing OpenSky Trino connection …")
    if test_connection():
        print("✓ Connected!\n")

        print("Fetching sample route stats (Jan 2024) …")
        stats = query_route_delay_stats("2024-01-01", "2024-01-07")
        if not stats.empty:
            print(stats.head(10).to_string(index=False))
        else:
            print("No data returned (check OPENSKY_USERNAME / OPENSKY_PASSWORD in .env)")
    else:
        print("✗ Connection failed")
        print("  Make sure OPENSKY_USERNAME and OPENSKY_PASSWORD are set in .env")
