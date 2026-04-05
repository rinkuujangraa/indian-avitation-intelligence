"""
tests/test_analytics.py
-----------------------
Basic validation tests for the analytics backend.

Run with: python -m pytest tests/test_analytics.py -v
"""

import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from analytics import (
    compute_afri,
    compute_delay_prediction,
    compute_fuel_estimate,
    compute_route_congestion,
    compute_airport_schedule_pressure,
    dep_delay_recovery_minutes,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _base_airports() -> list[dict]:
    """Minimal airport stubs for DEL and BOM."""
    return [
        {"iata": "DEL", "lat": 28.5562, "lng": 77.1000, "city": "Delhi"},
        {"iata": "BOM", "lat": 19.0896, "lng": 72.8656, "city": "Mumbai"},
        {"iata": "BLR", "lat": 13.1979, "lng": 77.7063, "city": "Bengaluru"},
    ]


def _empty_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["airport_iata", "congestion_score", "congestion_level",
                 "nearby_count", "inbound_count", "approach_count",
                 "low_altitude_count", "city"]
    )


# ── compute_delay_prediction ───────────────────────────────────────────────────

class TestComputeDelayPrediction:
    def test_high_risk_congested_slow_flight(self):
        """A slow en-route flight arriving at a congested airport → High/Medium risk.

        Aircraft is ~350 km from DEL (still en-route, not on approach) so the
        landing-imminent score cap does NOT apply.  Combined congestion + weather
        + schedule-pressure + low-OTP should push risk to High or Medium.

        The ML bundle is patched out so the assertion is against pure rule-based
        scoring and is not affected by whatever model may be on disk.
        """
        flight = pd.Series({
            "arr_iata": "DEL",
            "dep_iata": "BOM",
            "airline_iata": "AI",       # historically lower OTP
            "altitude_ft": 18000,       # descent begun but not on approach
            "speed_kts": 260,           # below normal cruise — slow en-route
            "latitude": 25.5,           # ~350 km south of DEL
            "longitude": 77.5,
            "weather_severity": "High",
            "stale_minutes": 0,
        })
        metrics = pd.DataFrame([{
            "airport_iata": "DEL",
            "congestion_score": 95,
            "congestion_level": "High",
            "nearby_count": 20,
            "inbound_count": 12,
            "approach_count": 6,
            "low_altitude_count": 5,
            "city": "Delhi",
        }])
        pressure = {
            "DEL": {
                "pressure_level": "High",
                "pressure_score": 0.85,
                "arrivals_next_hour": 18,
                "departures_next_hour": 16,
            }
        }
        with patch("analytics._get_ml_bundle", return_value=None):
            result = compute_delay_prediction(
                flight,
                airports=_base_airports(),
                airport_metrics=metrics,
                airport_schedule_pressure=pressure,
                now_ts=0,
            )
        assert result.risk_level in ("High", "Medium"), (
            f"Expected High or Medium risk for congested high-weather flight, got {result.risk_level}"
        )
        assert result.expected_delay_min > 0, "Expected non-zero delay minutes"

    def test_low_risk_healthy_cruise(self):
        """A cruising aircraft far from destination, normal speed context → Low/Medium."""
        flight = pd.Series({
            "arr_iata": "BOM",
            "dep_iata": "DEL",
            "airline_iata": "6E",       # IndiGo — high OTP
            "altitude_ft": 35000,       # cruising
            "speed_kts": 460,           # normal cruise
            "latitude": 22.0,           # ~900 km from BOM
            "longitude": 75.0,
            "weather_severity": "Low",
            "stale_minutes": 0,
        })
        result = compute_delay_prediction(
            flight,
            airports=_base_airports(),
            airport_metrics=_empty_metrics(),
            airport_schedule_pressure={},
            now_ts=0,
        )
        assert result.risk_level in ("Low", "Medium"), (
            f"Expected Low or Medium for healthy cruise, got {result.risk_level}"
        )

    def test_no_destination_returns_low(self):
        """Flight with no arr_iata should return Low risk gracefully."""
        flight = pd.Series({
            "arr_iata": "",
            "dep_iata": "DEL",
            "airline_iata": "6E",
            "altitude_ft": 35000,
            "speed_kts": 460,
            "latitude": 28.5,
            "longitude": 77.1,
            "weather_severity": "Low",
            "stale_minutes": 0,
        })
        result = compute_delay_prediction(
            flight,
            airports=_base_airports(),
            airport_metrics=_empty_metrics(),
            airport_schedule_pressure={},
            now_ts=0,
        )
        assert result.risk_level == "Low"
        assert result.expected_delay_min == 0

    def test_on_approach_caps_delay(self):
        """A flight on short final (< 40 km, < 4000 ft) should have a small expected delay."""
        flight = pd.Series({
            "arr_iata": "DEL",
            "dep_iata": "BOM",
            "airline_iata": "6E",
            "altitude_ft": 3000,
            "speed_kts": 160,
            "latitude": 28.58,          # ~4 km from DEL
            "longitude": 77.12,
            "weather_severity": "Low",
            "stale_minutes": 0,
        })
        result = compute_delay_prediction(
            flight,
            airports=_base_airports(),
            airport_metrics=_empty_metrics(),
            airport_schedule_pressure={},
            now_ts=0,
        )
        # Landing in minutes — delay should be capped small
        assert result.expected_delay_min <= 20, (
            f"On-final delay should be capped low, got {result.expected_delay_min}"
        )

    def test_prediction_returns_dataclass(self):
        """Return value must have required fields."""
        flight = pd.Series({
            "arr_iata": "BLR",
            "dep_iata": "DEL",
            "airline_iata": "UK",
            "altitude_ft": 28000,
            "speed_kts": 440,
            "latitude": 15.0,
            "longitude": 77.0,
            "weather_severity": "Low",
            "stale_minutes": 0,
        })
        result = compute_delay_prediction(
            flight,
            airports=_base_airports(),
            airport_metrics=_empty_metrics(),
            airport_schedule_pressure={},
            now_ts=0,
        )
        assert hasattr(result, "risk_level")
        assert hasattr(result, "expected_delay_min")
        assert hasattr(result, "reason_tags")
        assert hasattr(result, "score")
        assert result.risk_level in ("Low", "Medium", "High")
        assert isinstance(result.expected_delay_min, (int, float))
        assert result.expected_delay_min >= 0

    def test_unknown_weather_does_not_create_false_penalty(self):
        """Unknown weather should be handled safely and not act like Severe/Moderate."""
        flight = pd.Series({
            "arr_iata": "BOM",
            "dep_iata": "DEL",
            "airline_iata": "6E",
            "altitude_ft": 34000,
            "speed_kts": 450,
            "latitude": 22.0,
            "longitude": 75.0,
            "weather_severity": "Unknown",
            "stale_minutes": 0,
        })
        result = compute_delay_prediction(
            flight,
            airports=_base_airports(),
            airport_metrics=_empty_metrics(),
            airport_schedule_pressure={},
            now_ts=0,
        )
        assert result.risk_level in ("Low", "Medium")
        assert result.expected_delay_min >= 0


# ── compute_route_congestion ───────────────────────────────────────────────────

class TestComputeRouteCongestion:
    def _sample_flights(self) -> pd.DataFrame:
        """12 DEL→BOM, 6 BOM→BLR, 3 DEL→MAA flights."""
        rows = []
        for _ in range(12):
            rows.append({"hex": "AAA", "dep_iata": "DEL", "arr_iata": "BOM", "altitude_ft": 35000})
        for _ in range(6):
            rows.append({"hex": "BBB", "dep_iata": "BOM", "arr_iata": "BLR", "altitude_ft": 28000})
        for _ in range(3):
            rows.append({"hex": "CCC", "dep_iata": "DEL", "arr_iata": "MAA", "altitude_ft": 32000})
        return pd.DataFrame(rows)

    def test_busiest_route_ranked_first(self):
        df = self._sample_flights()
        result = compute_route_congestion(df, top_n=5)
        assert not result.empty
        assert result.iloc[0]["route_key"] == "DEL-BOM", (
            f"Expected DEL-BOM as busiest, got {result.iloc[0]['route_key']}"
        )

    def test_correct_flight_counts(self):
        df = self._sample_flights()
        result = compute_route_congestion(df)
        del_bom = result[result["route_key"] == "DEL-BOM"]
        assert len(del_bom) == 1
        assert del_bom.iloc[0]["flight_count"] == 12

    def test_empty_dataframe(self):
        result = compute_route_congestion(pd.DataFrame())
        assert result.empty
        assert "route_key" in result.columns

    def test_top_n_respected(self):
        df = self._sample_flights()
        result = compute_route_congestion(df, top_n=2)
        assert len(result) <= 2

    def test_missing_iata_excluded(self):
        """Rows with N/A dep or arr should be excluded from route calc."""
        df = pd.DataFrame([
            {"hex": "X1", "dep_iata": "DEL", "arr_iata": "N/A", "altitude_ft": 30000},
            {"hex": "X2", "dep_iata": "N/A", "arr_iata": "BOM", "altitude_ft": 30000},
            {"hex": "X3", "dep_iata": "DEL", "arr_iata": "BOM", "altitude_ft": 30000},
        ])
        result = compute_route_congestion(df)
        assert len(result) == 1
        assert result.iloc[0]["route_key"] == "DEL-BOM"

    def test_low_altitude_penalty_applied(self):
        """Flights at or below 12000 ft increase congestion score."""
        df_high = pd.DataFrame([
            {"hex": "H%d" % i, "dep_iata": "DEL", "arr_iata": "BOM", "altitude_ft": 35000}
            for i in range(5)
        ])
        df_low = pd.DataFrame([
            {"hex": "L%d" % i, "dep_iata": "DEL", "arr_iata": "BOM", "altitude_ft": 8000}
            for i in range(5)
        ])
        score_high = compute_route_congestion(df_high).iloc[0]["congestion_score"]
        score_low  = compute_route_congestion(df_low).iloc[0]["congestion_score"]
        assert score_low > score_high, (
            f"Low-altitude flights should produce higher congestion score ({score_low} > {score_high})"
        )


# ── compute_airport_schedule_pressure ─────────────────────────────────────────

class TestComputeAirportSchedulePressure:
    def _sched_df(self, count: int, direction: str, now_ts: int) -> pd.DataFrame:
        """Synthetic schedule rows all landing/departing 10 min from now.

        Uses the *estimated* timestamp columns that compute_airport_schedule_pressure
        reads (arr_estimated_ts for arrivals, dep_estimated_ts for departures).
        Without these columns the _count_window helper sees an empty series and
        returns 0, producing spuriously Low pressure.
        """
        estimated_col = "arr_estimated_ts" if direction == "arrival" else "dep_estimated_ts"
        scheduled_col = "arr_time_ts" if direction == "arrival" else "dep_time_ts"
        return pd.DataFrame([
            {"flight_iata": f"AI{100+i}", estimated_col: now_ts + 600, scheduled_col: now_ts + 600, "delayed": 0}
            for i in range(count)
        ])

    def test_high_pressure_many_movements(self):
        from analytics import compute_airport_schedule_pressure
        now_ts = 1_700_000_000
        arrivals  = self._sched_df(12, "arrival", now_ts)
        departures = self._sched_df(12, "departure", now_ts)
        result = compute_airport_schedule_pressure(arrivals, departures, now_ts)
        assert result["pressure_level"] in ("High", "Critical", "Medium"), (
            f"12+12 movements should be at least Medium pressure, got {result['pressure_level']}"
        )
        assert result["arrivals_next_hour"] >= 12

    def test_empty_schedule_returns_low(self):
        from analytics import compute_airport_schedule_pressure
        result = compute_airport_schedule_pressure(
            pd.DataFrame(), pd.DataFrame(), now_ts=1_700_000_000
        )
        assert result["pressure_level"] in ("Low", "Unknown")
        assert result["arrivals_next_hour"] == 0


class TestDepartureDelayRecovery:
    def test_short_haul_recovers_less(self):
        short = dep_delay_recovery_minutes(30, 300)
        medium = dep_delay_recovery_minutes(30, 900)
        long = dep_delay_recovery_minutes(30, 2000)
        assert short > medium > long

    def test_non_positive_delay_returns_zero(self):
        assert dep_delay_recovery_minutes(0, 500) == 0
        assert dep_delay_recovery_minutes(-5, 500) == 0


class TestFuelEstimate:
    def test_returns_positive_values(self):
        result = compute_fuel_estimate("A320", 1000)
        assert result.fuel_burn_kg > 0
        assert result.co2_kg > result.fuel_burn_kg
        assert result.actual_distance_km >= result.distance_km

    def test_unknown_aircraft_uses_default_rate(self):
        result = compute_fuel_estimate("ZZZZ", 800)
        assert result.fuel_burn_kg > 0
        assert result.fuel_rate_label.endswith("kg/km")


class TestAFRI:
    def test_afri_adds_expected_columns(self):
        flights = pd.DataFrame([
            {
                "arr_iata": "DEL",
                "weather_severity": "Moderate",
                "stale_minutes": 6,
            },
            {
                "arr_iata": "DEL",
                "weather_severity": "Low",
                "stale_minutes": 0,
            },
        ])
        metrics = pd.DataFrame([
            {
                "airport_iata": "DEL",
                "inbound_count": 10,
                "approach_count": 5,
            }
        ])
        pressure = {
            "DEL": {"pressure_level": "High"}
        }
        result = compute_afri(flights, metrics, pressure)
        assert "afri_score" in result.columns
        assert "afri_level" in result.columns
        assert "afri_drivers" in result.columns
        assert result["afri_score"].max() > 0
