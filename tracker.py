"""
tracker.py
----------
Tracks real flight movements by saving position snapshots over time.
Each time you fetch flight data, this module stores the positions.
Over multiple fetches, it builds a trail (history of positions) for
each aircraft — which we can then draw as a line on the map.

How it works:
    1. Every time get_flight_data() runs, call tracker.record(df)
    2. Tracker saves each aircraft's position with a timestamp
    3. When building the map, call tracker.get_trails() to get
       the historical path for every aircraft
    4. Draw those paths as polylines on the map

Storage:
    Trails are saved to a local JSON file (flight_trails.json)
    so history persists between script runs.
"""

import json
import os
import time
import logging
import threading
import pandas as pd

# fcntl is Unix-only; provide a no-op shim on Windows
try:
    import fcntl as _fcntl  # type: ignore[import]

    def _lock_shared(f):   _fcntl.flock(f, _fcntl.LOCK_SH)
    def _lock_exclusive(f): _fcntl.flock(f, _fcntl.LOCK_EX)
    def _unlock(f):         _fcntl.flock(f, _fcntl.LOCK_UN)
except ImportError:  # Windows — file locking not available; use no-ops
    def _lock_shared(f):   pass
    def _lock_exclusive(f): pass
    def _unlock(f):         pass

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
TRAILS_FILE     = os.path.join(os.path.dirname(__file__), "flight_trails.json")
MAX_TRAIL_AGE_H = 2                      # discard positions older than 2 hours
MAX_POINTS      = 30                     # max trail points per aircraft


class FlightTracker:
    """
    Stores and retrieves position history for each tracked aircraft.

    Each aircraft is identified by its ICAO24 hex code (e.g. "a1b2c3").
    We store a list of timestamped positions for each hex code.

    Data structure stored in flight_trails.json:
    {
        "a1b2c3": {
            "flight_iata": "EK203",
            "airline_iata": "EK",
            "positions": [
                {
                    "lat": 25.1,
                    "lng": 55.3,
                    "altitude_ft": 37000,
                    "speed_kts": 480,
                    "heading": 315,
                    "timestamp": 1710000000
                },
                ...
            ]
        },
        ...
    }
    """

    def __init__(self, trails_file: str = TRAILS_FILE):
        self.trails_file = trails_file
        self._lock = threading.Lock()  # guards self.trails across concurrent Streamlit reruns
        # trails is a dict: hex_code → {flight info + positions list}
        self.trails = self._load()
        logger.info(f"Tracker loaded: {len(self.trails)} aircraft in history")


    # ── Load & Save ────────────────────────────────────────────────────────────
    def _load(self) -> dict:
        """Load trail data from the JSON file on disk."""
        if os.path.exists(self.trails_file):
            try:
                with open(self.trails_file, "r") as f:
                    _lock_shared(f)
                    try:
                        data = json.load(f)
                    finally:
                        _unlock(f)
                logger.info(f"Loaded trail history from '{self.trails_file}'")
                return data
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load trails file: {e}. Starting fresh.")
        return {}

    def _save(self) -> None:
        """Save current trail data to the JSON file on disk with atomic write."""
        tmp_file = self.trails_file + ".tmp"
        try:
            with open(tmp_file, "w") as f:
                _lock_exclusive(f)
                try:
                    json.dump(self.trails, f)
                finally:
                    _unlock(f)
            os.replace(tmp_file, self.trails_file)
        except IOError as e:
            logger.error(f"Could not save trails: {e}")
            if os.path.exists(tmp_file):
                os.remove(tmp_file)


    # ── Record new positions ───────────────────────────────────────────────────
    def record(self, df: pd.DataFrame) -> None:
        """
        Record the current position of every flight in the DataFrame.

        Call this every time you fetch new flight data. Over time,
        each aircraft builds up a list of positions — its trail.

        Parameters
        ----------
        df : pd.DataFrame
            Current flight DataFrame from data_fetcher.get_flight_data()
        """
        if df.empty:
            logger.warning("Empty DataFrame passed to tracker — nothing recorded.")
            return

        with self._lock:
            now = int(time.time())   # current Unix timestamp
            new_points = 0

            for _, row in df.iterrows():
                hex_code = row.get("hex")
                if not hex_code or pd.isna(hex_code):
                    continue

                hex_code = str(hex_code)

                # Build the position snapshot for this moment
                source_timestamp = row.get("updated")
                if pd.notna(source_timestamp):
                    try:
                        point_timestamp = int(float(source_timestamp))
                    except (TypeError, ValueError):
                        point_timestamp = now
                else:
                    point_timestamp = now

                position = {
                    "lat":         row.get("latitude"),
                    "lng":         row.get("longitude"),
                    "altitude_ft": row.get("altitude_ft"),
                    "speed_kts":   row.get("speed_kts"),
                    "heading":     row.get("heading"),
                    "timestamp":   point_timestamp,
                    "ts":          point_timestamp,  # alias used by analytics detection functions
                }

                # Skip if lat/lng is missing
                if pd.isna(position["lat"]) or pd.isna(position["lng"]):
                    continue

                # Convert numpy floats to native Python floats for JSON serialization
                for k, v in position.items():
                    if hasattr(v, "item"):      # numpy scalar → Python scalar
                        try:
                            position[k] = v.item()
                        except (TypeError, ValueError):
                            position[k] = None
                    elif (not isinstance(v, str)) and pd.isna(v):
                        position[k] = None

                # Create entry for new aircraft
                if hex_code not in self.trails:
                    self.trails[hex_code] = {
                        "flight_iata":  str(row.get("flight_iata",  "N/A")),
                        "airline_iata": str(row.get("airline_iata", "N/A")),
                        "dep_iata":     str(row.get("dep_iata",     "N/A")),
                        "arr_iata":     str(row.get("arr_iata",     "N/A")),
                        "aircraft_icao":str(row.get("aircraft_icao","N/A")),
                        "positions":    []
                    }

                # Update flight info (may change e.g. status updates)
                self.trails[hex_code]["flight_iata"]   = str(row.get("flight_iata",  "N/A"))
                self.trails[hex_code]["airline_iata"]  = str(row.get("airline_iata", "N/A"))
                self.trails[hex_code]["dep_iata"]      = str(row.get("dep_iata",     "N/A"))
                self.trails[hex_code]["arr_iata"]      = str(row.get("arr_iata",     "N/A"))
                self.trails[hex_code]["aircraft_icao"] = str(row.get("aircraft_icao","N/A"))

                last_position = (
                    self.trails[hex_code]["positions"][-1]
                    if self.trails[hex_code]["positions"]
                    else None
                )
                if last_position and self._is_duplicate_position(last_position, position):
                    continue

                # Append new position
                self.trails[hex_code]["positions"].append(position)
                new_points += 1

                # Keep only the last MAX_POINTS positions per aircraft
                # This prevents the file growing too large over time
                if len(self.trails[hex_code]["positions"]) > MAX_POINTS:
                    self.trails[hex_code]["positions"] = \
                        self.trails[hex_code]["positions"][-MAX_POINTS:]

            # ── Prune old data ─────────────────────────────────────────────────
            self._prune_old_data(now)

            # ── Save to disk ───────────────────────────────────────────────────
            self._save()
            logger.info(f"Recorded {new_points} new positions | "
                        f"Tracking {len(self.trails)} aircraft total")

    @staticmethod
    def _is_duplicate_position(existing: dict, new_position: dict) -> bool:
        """Skip cached/rerun duplicates so trails reflect actual movement."""
        keys = ("lat", "lng", "altitude_ft", "speed_kts", "heading", "timestamp")
        return all(existing.get(key) == new_position.get(key) for key in keys)

    def _prune_old_data(self, now: int) -> None:
        """
        Remove positions older than MAX_TRAIL_AGE_H hours.
        Also remove aircraft entries that have no positions left.

        This keeps the trails file small and the map clean —
        we don't want trails from flights that landed hours ago.
        """
        cutoff = now - (MAX_TRAIL_AGE_H * 3600)   # cutoff in Unix seconds
        to_delete = []

        for hex_code, data in self.trails.items():
            # Keep only positions newer than the cutoff
            data["positions"] = [
                p for p in data["positions"]
                if p.get("timestamp", 0) > cutoff
            ]
            # Mark for deletion if no positions remain
            if not data["positions"]:
                to_delete.append(hex_code)

        for hex_code in to_delete:
            del self.trails[hex_code]

        if to_delete:
            logger.info(f"Pruned {len(to_delete)} stale aircraft from trail history")


    # ── Get trail data ─────────────────────────────────────────────────────────
    def get_trails(self) -> dict:
        """
        Return a snapshot of all current trail data (thread-safe copy).
        """
        with self._lock:
            return dict(self.trails)

    def get_trail(self, hex_code: str) -> list:
        with self._lock:
            entry = self.trails.get(hex_code, {})
            return list(entry.get("positions", []))

    def get_stats(self) -> dict:
        """Return summary statistics about the current tracking state."""
        total_positions = sum(
            len(v["positions"]) for v in self.trails.values()
        )
        airlines = set(
            v["airline_iata"] for v in self.trails.values()
            if v["airline_iata"] != "N/A"
        )
        return {
            "aircraft_tracked": len(self.trails),
            "total_positions":  total_positions,
            "airlines_seen":    len(airlines),
        }

    def clear(self) -> None:
        """Wipe all trail history (useful for testing)."""
        self.trails = {}
        self._save()
        logger.info("Trail history cleared.")


# ── Module-level singleton ─────────────────────────────────────────────────────
# We create one shared tracker instance so all parts of the app
# use the same history. Import and use like:
#   from tracker import tracker
#   tracker.record(df)
#   trails = tracker.get_trails()
tracker = FlightTracker()
