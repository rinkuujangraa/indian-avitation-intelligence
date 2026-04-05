"""
tests/test_snapshot_store.py
-----------------------------
Unit tests for snapshot_store.SnapshotStore.

Run with: python -m pytest tests/test_snapshot_store.py -v
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from snapshot_store import SnapshotStore


# ── Helpers ────────────────────────────────────────────────────────────────────

def _store(min_interval: int = 1) -> SnapshotStore:
    """Return a fresh in-memory SnapshotStore backed by a temp SQLite file."""
    tmp = tempfile.mktemp(suffix=".db")
    return SnapshotStore(db_path=tmp, min_interval_seconds=min_interval)


def _flights_df(n: int = 3) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "flight_iata": f"AI{100+i}", "hex": f"abc{i:03d}",
            "airline_iata": "AI", "dep_iata": "DEL", "arr_iata": "BOM",
            "latitude": 28.5 + i * 0.1, "longitude": 77.0 + i * 0.1,
            "altitude_ft": 35000.0, "speed_kts": 450.0,
            "heading": 220.0, "status": "en-route",
        })
    return pd.DataFrame(rows)


def _airport_metrics_df() -> pd.DataFrame:
    return pd.DataFrame([{
        "airport_iata": "DEL", "nearby_count": 12, "inbound_count": 5,
        "low_altitude_count": 2, "approach_count": 3,
        "congestion_score": 0.62, "congestion_level": "Moderate",
    }])


def _route_metrics_df() -> pd.DataFrame:
    return pd.DataFrame([{
        "route_key": "DEL-BOM", "dep_iata": "DEL", "arr_iata": "BOM",
        "flight_count": 7, "low_altitude_count": 1, "congestion_score": 0.45,
    }])


# ── should_record() ────────────────────────────────────────────────────────────

class TestShouldRecord:
    def test_true_when_no_snapshots(self):
        store = _store()
        assert store.should_record() is True

    def test_false_immediately_after_record(self):
        store = _store(min_interval=300)
        store.record_flights(_flights_df())
        assert store.should_record() is False

    def test_true_after_interval_passes(self):
        store = _store(min_interval=1)
        store.record_flights(_flights_df(), ts=int(time.time()) - 5)
        assert store.should_record() is True


# ── record_flights() ───────────────────────────────────────────────────────────

class TestRecordFlights:
    def test_skips_empty_df(self):
        store = _store()
        store.record_flights(pd.DataFrame())
        assert store.should_record() is True  # nothing was written

    def test_saves_rows(self):
        store = _store()
        ts = int(time.time())
        store.record_flights(_flights_df(3), ts=ts)
        history = store.get_flight_history(hours=1)
        assert any(r["ts"] == ts for r in history)
        matching = [r for r in history if r["ts"] == ts]
        assert matching[0]["flight_count"] == 3


# ── record_airports() ─────────────────────────────────────────────────────────

class TestRecordAirports:
    def test_saves_airport_row(self):
        store = _store()
        pressure = {"DEL": {"arrivals_next_hour": 8, "departures_next_hour": 6, "pressure_level": "High"}}
        store.record_airports(_airport_metrics_df(), pressure, ts=int(time.time()))
        # no exception = pass; we verify via direct SQL
        with store._connect() as conn:
            rows = conn.execute("SELECT airport_iata FROM airport_snapshots").fetchall()
        assert any(r[0] == "DEL" for r in rows)

    def test_skips_empty_df(self):
        store = _store()
        store.record_airports(pd.DataFrame(), {})
        with store._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM airport_snapshots").fetchone()[0]
        assert count == 0


# ── record_routes() ───────────────────────────────────────────────────────────

class TestRecordRoutes:
    def test_saves_route_row(self):
        store = _store()
        store.record_routes(_route_metrics_df(), ts=int(time.time()))
        with store._connect() as conn:
            rows = conn.execute("SELECT route_key FROM route_snapshots").fetchall()
        assert any(r[0] == "DEL-BOM" for r in rows)


# ── get_snapshot_at() ─────────────────────────────────────────────────────────

class TestGetSnapshotAt:
    def test_returns_closest_snapshot(self):
        store = _store()
        ts = int(time.time())
        store.record_flights(_flights_df(2), ts=ts)
        df = store.get_snapshot_at(ts)
        assert not df.empty
        assert len(df) == 2

    def test_returns_empty_when_no_data(self):
        store = _store()
        df = store.get_snapshot_at(int(time.time()))
        assert df.empty


# ── purge_old_snapshots() ─────────────────────────────────────────────────────

class TestPurgeOldSnapshots:
    def test_deletes_old_rows(self):
        store = _store()
        old_ts = int(time.time()) - 10 * 86400  # 10 days ago
        store.record_flights(_flights_df(2), ts=old_ts)
        deleted = store.purge_old_snapshots(retain_days=7)
        assert deleted >= 2

    def test_preserves_recent_rows(self):
        store = _store()
        ts = int(time.time())
        store.record_flights(_flights_df(2), ts=ts)
        deleted = store.purge_old_snapshots(retain_days=7)
        assert deleted == 0
        df = store.get_snapshot_at(ts)
        assert not df.empty
