"""
snapshot_store.py
-----------------
Lightweight SQLite snapshot foundation for Phase 2 analytics.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pandas as pd


DB_PATH = Path(__file__).with_name("aviation_intelligence.db")

SCHEMA_VERSION = 3
_SNAPSHOT_TABLES = frozenset({"flight_snapshots", "airport_snapshots", "route_snapshots", "route_otp"})


class SnapshotStore:
    def __init__(self, db_path: Path = DB_PATH, min_interval_seconds: int = 240) -> None:
        self.db_path = Path(db_path)
        self.min_interval_seconds = min_interval_seconds
        self._write_lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _get_schema_version(self, conn: sqlite3.Connection) -> int:
        try:
            row = conn.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1").fetchone()
            return int(row[0]) if row else 0
        except sqlite3.OperationalError:
            return 0

    def _init_db(self) -> None:
        with self._write_lock, self._connect() as conn:
            current = self._get_schema_version(conn)
            if current < 1:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS schema_version (
                        version INTEGER NOT NULL,
                        applied_at INTEGER NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS flight_snapshots (
                        ts INTEGER NOT NULL,
                        flight_iata TEXT,
                        hex TEXT,
                        airline_iata TEXT,
                        dep_iata TEXT,
                        arr_iata TEXT,
                        latitude REAL,
                        longitude REAL,
                        altitude_ft REAL,
                        speed_kts REAL,
                        heading REAL,
                        status TEXT
                    );

                    CREATE TABLE IF NOT EXISTS airport_snapshots (
                        ts INTEGER NOT NULL,
                        airport_iata TEXT,
                        nearby_count INTEGER,
                        inbound_count INTEGER,
                        low_altitude_count INTEGER,
                        approach_count INTEGER,
                        congestion_score REAL,
                        congestion_level TEXT,
                        arrivals_next_hour INTEGER,
                        departures_next_hour INTEGER,
                        pressure_level TEXT
                    );

                    CREATE TABLE IF NOT EXISTS route_snapshots (
                        ts INTEGER NOT NULL,
                        route_key TEXT,
                        dep_iata TEXT,
                        arr_iata TEXT,
                        flight_count INTEGER,
                        low_altitude_count INTEGER,
                        congestion_score REAL
                    );
                    """
                )
                conn.execute("INSERT INTO schema_version (version, applied_at) VALUES (1, ?)", (int(time.time()),))

            if current < 2:
                # Add indexes for common queries
                conn.executescript(
                    """
                    CREATE INDEX IF NOT EXISTS idx_flight_snap_ts ON flight_snapshots(ts);
                    CREATE INDEX IF NOT EXISTS idx_airport_snap_ts ON airport_snapshots(ts);
                    CREATE INDEX IF NOT EXISTS idx_route_snap_ts ON route_snapshots(ts);
                    """
                )
                conn.execute("INSERT INTO schema_version (version, applied_at) VALUES (2, ?)", (int(time.time()),))

            if current < 3:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS route_otp (
                        ts INTEGER NOT NULL,
                        route_key TEXT NOT NULL,
                        airline_iata TEXT,
                        actual_delay_min INTEGER NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_route_otp_key ON route_otp(route_key, airline_iata);
                    """
                )
                conn.execute("INSERT INTO schema_version (version, applied_at) VALUES (3, ?)", (int(time.time()),))

    def _latest_ts(self, table_name: str) -> int | None:
        if table_name not in _SNAPSHOT_TABLES:
            raise ValueError(f"Invalid snapshot table: {table_name!r}")
        with self._connect() as conn:
            row = conn.execute(f"SELECT MAX(ts) FROM {table_name}").fetchone()
        return int(row[0]) if row and row[0] is not None else None

    def should_record(self) -> bool:
        latest = self._latest_ts("flight_snapshots")
        if latest is None:
            return True
        return (int(time.time()) - latest) >= self.min_interval_seconds

    def record_flights(self, flights_df: pd.DataFrame, ts: int | None = None) -> None:
        if flights_df.empty:
            return
        ts = ts or int(time.time())
        rows = []
        for _, row in flights_df.iterrows():
            rows.append(
                (
                    ts,
                    row.get("flight_iata"),
                    row.get("hex"),
                    row.get("airline_iata"),
                    row.get("dep_iata"),
                    row.get("arr_iata"),
                    row.get("latitude"),
                    row.get("longitude"),
                    row.get("altitude_ft"),
                    row.get("speed_kts"),
                    row.get("heading"),
                    row.get("status"),
                )
            )
        with self._write_lock:
            with self._connect() as conn:
                conn.executemany(
                    """
                    INSERT INTO flight_snapshots (
                        ts, flight_iata, hex, airline_iata, dep_iata, arr_iata,
                        latitude, longitude, altitude_ft, speed_kts, heading, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )

    def record_airports(
        self,
        airport_metrics_df: pd.DataFrame,
        airport_schedule_pressure: dict[str, dict],
        ts: int | None = None,
    ) -> None:
        if airport_metrics_df.empty:
            return
        ts = ts or int(time.time())
        rows = []
        for _, row in airport_metrics_df.iterrows():
            schedule = airport_schedule_pressure.get(row["airport_iata"], {})
            rows.append(
                (
                    ts,
                    row["airport_iata"],
                    int(row["nearby_count"]),
                    int(row["inbound_count"]),
                    int(row["low_altitude_count"]),
                    int(row["approach_count"]),
                    float(row["congestion_score"]),
                    row["congestion_level"],
                    int(schedule.get("arrivals_next_hour", 0)),
                    int(schedule.get("departures_next_hour", 0)),
                    schedule.get("pressure_level", "Low"),
                )
            )
        with self._write_lock:
            with self._connect() as conn:
                conn.executemany(
                    """
                    INSERT INTO airport_snapshots (
                        ts, airport_iata, nearby_count, inbound_count, low_altitude_count,
                        approach_count, congestion_score, congestion_level,
                        arrivals_next_hour, departures_next_hour, pressure_level
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )

    def record_routes(self, route_metrics_df: pd.DataFrame, ts: int | None = None) -> None:
        if route_metrics_df.empty:
            return
        ts = ts or int(time.time())
        rows = []
        for _, row in route_metrics_df.iterrows():
            rows.append(
                (
                    ts,
                    row["route_key"],
                    row["dep_iata"],
                    row["arr_iata"],
                    int(row["flight_count"]),
                    int(row["low_altitude_count"]),
                    float(row["congestion_score"]),
                )
            )
        with self._write_lock:
            with self._connect() as conn:
                conn.executemany(
                    """
                    INSERT INTO route_snapshots (
                        ts, route_key, dep_iata, arr_iata, flight_count, low_altitude_count, congestion_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
            )

    # ── History queries for time-slider playback ───────────────────────────────

    def get_flight_history(self, hours: int = 4) -> list[dict]:
        """Return distinct snapshot timestamps and flight counts for the last N hours."""
        cutoff = int(time.time()) - hours * 3600
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT ts, COUNT(*) AS flight_count,
                       ROUND(AVG(altitude_ft), 0) AS avg_alt,
                       ROUND(AVG(speed_kts), 1) AS avg_spd
                FROM flight_snapshots
                WHERE ts >= ?
                GROUP BY ts
                ORDER BY ts ASC
                """,
                (cutoff,),
            ).fetchall()
        return [
            {"ts": r[0], "flight_count": r[1], "avg_alt": r[2], "avg_spd": r[3]}
            for r in rows
        ]

    def get_snapshot_at(self, ts: int) -> pd.DataFrame:
        """Return the flight snapshot closest to the given timestamp."""
        with self._connect() as conn:
            closest = conn.execute(
                "SELECT ts FROM flight_snapshots ORDER BY ABS(ts - ?) LIMIT 1",
                (ts,),
            ).fetchone()
            if not closest:
                return pd.DataFrame()
            rows = conn.execute(
                """
                SELECT flight_iata, hex, airline_iata, dep_iata, arr_iata,
                       latitude, longitude, altitude_ft, speed_kts, heading, status
                FROM flight_snapshots
                WHERE ts = ?
                """,
                (closest[0],),
            ).fetchall()
        cols = [
            "flight_iata", "hex", "airline_iata", "dep_iata", "arr_iata",
            "latitude", "longitude", "altitude_ft", "speed_kts", "heading", "status",
        ]
        return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

    def purge_old_snapshots(self, retain_days: int = 7) -> int:
        """Delete snapshot rows older than *retain_days* days. Returns total rows deleted."""
        cutoff = int(time.time()) - retain_days * 86400
        total_deleted = 0
        tables = ("flight_snapshots", "airport_snapshots", "route_snapshots", "route_otp")
        with self._write_lock, self._connect() as conn:
            for table in tables:
                if table not in _SNAPSHOT_TABLES:
                    continue  # safety guard — skip unknown tables
                cursor = conn.execute(f"DELETE FROM {table} WHERE ts < ?", (cutoff,))
                total_deleted += cursor.rowcount
        # WAL checkpoint runs in a separate connection *after* the write transaction
        # commits, and is best-effort — a locked DB just means another writer is
        # active and the checkpoint will happen automatically later.
        if total_deleted > 0:
            try:
                with self._connect() as ckpt_conn:
                    ckpt_conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except Exception:
                pass
        return total_deleted

    def get_single_flight_history(self, flight_iata: str, hours: int = 1) -> list[dict]:
        """Return per-point position history for a specific flight (for holding/go-around detection)."""
        cutoff = int(time.time()) - hours * 3600
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT ts, latitude, longitude, altitude_ft, speed_kts, heading
                FROM flight_snapshots
                WHERE flight_iata = ? AND ts >= ?
                  AND latitude IS NOT NULL AND longitude IS NOT NULL
                ORDER BY ts ASC
                """,
                (flight_iata, cutoff),
            ).fetchall()
        return [
            {"ts": r[0], "lat": r[1], "lng": r[2], "altitude_ft": r[3], "speed_kts": r[4], "heading": r[5]}
            for r in rows
        ]

    def record_route_otp(self, route_key: str, airline_iata: str, actual_delay_min: int) -> None:
        """Record an observed arrival delay for a route to improve future predictions."""
        ts = int(time.time())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO route_otp (ts, route_key, airline_iata, actual_delay_min) VALUES (?, ?, ?, ?)",
                (ts, route_key, airline_iata, int(actual_delay_min)),
            )

    def get_route_otp_p50(self, route_key: str, airline_iata: str | None = None, lookback_days: int = 60) -> float | None:
        """Return the median observed delay (p50) for a route over the past lookback_days, or None."""
        cutoff = int(time.time()) - lookback_days * 86400
        with self._connect() as conn:
            if airline_iata:
                rows = conn.execute(
                    "SELECT actual_delay_min FROM route_otp WHERE route_key=? AND airline_iata=? AND ts>=? ORDER BY actual_delay_min",
                    (route_key, airline_iata, cutoff),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT actual_delay_min FROM route_otp WHERE route_key=? AND ts>=? ORDER BY actual_delay_min",
                    (route_key, cutoff),
                ).fetchall()
        if not rows:
            return None
        vals = [r[0] for r in rows]
        mid = len(vals) // 2
        return float(vals[mid]) if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0

    def db_size_bytes(self) -> int:
        try:
            return self.db_path.stat().st_size
        except FileNotFoundError:
            return 0


snapshot_store = SnapshotStore()
