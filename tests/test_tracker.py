"""
tests/test_tracker.py
---------------------
Unit tests for tracker.FlightTracker.

Run with: python -m pytest tests/test_tracker.py -v
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from tracker import FlightTracker, MAX_TRAIL_AGE_H, MAX_POINTS


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_df(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "hex": None, "flight_iata": "AI101", "airline_iata": "AI",
        "dep_iata": "DEL", "arr_iata": "BOM", "aircraft_icao": "B738",
        "latitude": 28.56, "longitude": 77.10, "altitude_ft": 35000.0,
        "speed_kts": 450.0, "heading": 220.0, "updated": time.time(),
        "status": "en-route",
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _tracker_in_tmpdir() -> FlightTracker:
    """Return a fresh FlightTracker backed by a temp file that is cleaned up."""
    tmp = tempfile.mktemp(suffix=".json")
    return FlightTracker(trails_file=tmp)


# ── record() ──────────────────────────────────────────────────────────────────

class TestRecord:
    def test_records_new_aircraft(self):
        t = _tracker_in_tmpdir()
        df = _make_df([{"hex": "abc123"}])
        t.record(df)
        trails = t.get_trails()
        assert "abc123" in trails
        assert len(trails["abc123"]["positions"]) == 1

    def test_skips_rows_without_hex(self):
        t = _tracker_in_tmpdir()
        df = _make_df([{"hex": None}, {"hex": "abc123"}])
        t.record(df)
        assert len(t.get_trails()) == 1

    def test_skips_rows_without_latlon(self):
        t = _tracker_in_tmpdir()
        df = _make_df([{"hex": "abc123", "latitude": None}])
        t.record(df)
        assert len(t.get_trails()) == 0

    def test_does_not_record_duplicate_position(self):
        t = _tracker_in_tmpdir()
        now = time.time()
        row = {"hex": "abc123", "latitude": 28.56, "longitude": 77.10,
               "altitude_ft": 35000.0, "speed_kts": 450.0, "heading": 220.0, "updated": now}
        t.record(_make_df([row]))
        t.record(_make_df([row]))
        positions = t.get_trail("abc123")
        assert len(positions) == 1

    def test_appends_new_position(self):
        t = _tracker_in_tmpdir()
        now = time.time()
        t.record(_make_df([{"hex": "abc123", "latitude": 28.56, "longitude": 77.10, "updated": now}]))
        t.record(_make_df([{"hex": "abc123", "latitude": 28.60, "longitude": 77.14, "updated": now + 10}]))
        assert len(t.get_trail("abc123")) == 2

    def test_caps_positions_at_max_points(self):
        t = _tracker_in_tmpdir()
        for i in range(MAX_POINTS + 5):
            t.record(_make_df([{
                "hex": "abc123",
                "latitude": 28.0 + i * 0.01,
                "longitude": 77.0,
                "updated": time.time() + i,
            }]))
        assert len(t.get_trail("abc123")) <= MAX_POINTS

    def test_empty_dataframe_is_safe(self):
        t = _tracker_in_tmpdir()
        t.record(pd.DataFrame())
        assert t.get_trails() == {}


# ── _is_duplicate_position() ──────────────────────────────────────────────────

class TestIsDuplicatePosition:
    def test_identical_positions_are_duplicate(self):
        pos = {"lat": 1.0, "lng": 2.0, "altitude_ft": 1000.0,
               "speed_kts": 400.0, "heading": 90.0, "timestamp": 12345}
        assert FlightTracker._is_duplicate_position(pos, pos)

    def test_different_lat_not_duplicate(self):
        pos = {"lat": 1.0, "lng": 2.0, "altitude_ft": 1000.0,
               "speed_kts": 400.0, "heading": 90.0, "timestamp": 12345}
        new = {**pos, "lat": 1.1}
        assert not FlightTracker._is_duplicate_position(pos, new)

    def test_different_timestamp_not_duplicate(self):
        pos = {"lat": 1.0, "lng": 2.0, "altitude_ft": 1000.0,
               "speed_kts": 400.0, "heading": 90.0, "timestamp": 12345}
        new = {**pos, "timestamp": 99999}
        assert not FlightTracker._is_duplicate_position(pos, new)


# ── _prune_old_data() ─────────────────────────────────────────────────────────

class TestPruneOldData:
    def test_prunes_stale_positions(self):
        t = _tracker_in_tmpdir()
        old_ts = int(time.time()) - (MAX_TRAIL_AGE_H * 3600 + 600)
        t.trails = {
            "abc123": {
                "flight_iata": "AI101", "airline_iata": "AI",
                "dep_iata": "DEL", "arr_iata": "BOM", "aircraft_icao": "B738",
                "positions": [
                    {"lat": 28.0, "lng": 77.0, "timestamp": old_ts},
                    {"lat": 28.5, "lng": 77.5, "timestamp": int(time.time())},
                ]
            }
        }
        t._prune_old_data(int(time.time()))
        trail = t.get_trail("abc123")
        assert len(trail) == 1
        assert trail[0]["lat"] == 28.5

    def test_removes_aircraft_with_no_positions(self):
        t = _tracker_in_tmpdir()
        old_ts = int(time.time()) - (MAX_TRAIL_AGE_H * 3600 + 600)
        t.trails = {
            "abc123": {
                "flight_iata": "AI101", "airline_iata": "AI",
                "dep_iata": "DEL", "arr_iata": "BOM", "aircraft_icao": "B738",
                "positions": [{"lat": 28.0, "lng": 77.0, "timestamp": old_ts}]
            }
        }
        t._prune_old_data(int(time.time()))
        assert "abc123" not in t.get_trails()


# ── get_trails() / get_trail() ────────────────────────────────────────────────

class TestGetTrails:
    def test_get_trails_returns_copy(self):
        t = _tracker_in_tmpdir()
        t.record(_make_df([{"hex": "abc123"}]))
        result = t.get_trails()
        result["injected"] = {}
        assert "injected" not in t.get_trails()

    def test_get_trail_returns_copy(self):
        t = _tracker_in_tmpdir()
        t.record(_make_df([{"hex": "abc123"}]))
        trail = t.get_trail("abc123")
        trail.append({"fake": True})
        assert len(t.get_trail("abc123")) == 1

    def test_get_trail_missing_hex_returns_empty(self):
        t = _tracker_in_tmpdir()
        assert t.get_trail("nonexistent") == []
