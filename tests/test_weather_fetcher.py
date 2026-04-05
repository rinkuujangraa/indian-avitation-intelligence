"""
tests/test_weather_fetcher.py
------------------------------
Unit tests for weather_fetcher._classify_weather and the circuit breaker.

Run with: python -m pytest tests/test_weather_fetcher.py -v
"""

import os
import sys
import time
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import weather_fetcher
from weather_fetcher import _classify_weather, get_nearest_airport_weather, WeatherImpact


# ── Helpers ────────────────────────────────────────────────────────────────────

def _decoded(
    flight_category: str = "VFR",
    visibility_miles: float | None = 10.0,
    wind_speed: int | None = None,
    gust_speed: int | None = None,
    conditions: list[dict] | None = None,
    icao: str = "VIDP",
) -> dict:
    """Build a minimal decoded METAR dict for _classify_weather."""
    return {
        "icao": icao,
        "raw_text": f"METAR {icao} AUTO",
        "flight_category": flight_category,
        "visibility": {"miles": visibility_miles} if visibility_miles is not None else {},
        "wind": {
            "speed": {"kts": wind_speed} if wind_speed is not None else {},
            "gust": {"kts": gust_speed} if gust_speed is not None else {},
        },
        "conditions": conditions or [],
    }


# ── _classify_weather() ────────────────────────────────────────────────────────

class TestClassifyWeather:
    def test_vfr_clear_is_low_severity(self):
        impact = _classify_weather(_decoded())
        assert impact.severity == "Low"
        assert impact.penalty_minutes == 0

    def test_lifr_is_severe(self):
        impact = _classify_weather(_decoded(flight_category="LIFR"))
        assert impact.severity == "Severe"
        assert impact.penalty_minutes >= 14

    def test_ifr_is_severe(self):
        impact = _classify_weather(_decoded(flight_category="IFR"))
        assert impact.severity == "Severe"
        assert impact.penalty_minutes >= 10

    def test_mvfr_is_moderate(self):
        impact = _classify_weather(_decoded(flight_category="MVFR"))
        assert impact.severity == "Moderate"
        assert impact.penalty_minutes >= 5

    def test_poor_visibility_is_severe(self):
        impact = _classify_weather(_decoded(visibility_miles=1.0))
        assert impact.severity == "Severe"
        assert impact.penalty_minutes >= 8

    def test_reduced_visibility_is_moderate(self):
        impact = _classify_weather(_decoded(visibility_miles=4.0))
        assert impact.severity == "Moderate"

    def test_strong_gusts_are_severe(self):
        impact = _classify_weather(_decoded(gust_speed=35))
        assert impact.severity == "Severe"
        assert impact.penalty_minutes >= 7

    def test_moderate_wind_is_moderate(self):
        impact = _classify_weather(_decoded(wind_speed=22))
        assert impact.severity == "Moderate"

    def test_thunderstorm_is_severe(self):
        impact = _classify_weather(_decoded(conditions=[{"text": "Thunderstorm"}]))
        assert impact.severity == "Severe"
        assert impact.penalty_minutes >= 10

    def test_rain_is_moderate(self):
        impact = _classify_weather(_decoded(conditions=[{"text": "Light Rain"}]))
        assert impact.severity == "Moderate"

    def test_penalty_capped_at_20(self):
        # LIFR + poor visibility + thunderstorm could exceed 20 without cap
        impact = _classify_weather(_decoded(
            flight_category="LIFR",
            visibility_miles=0.5,
            conditions=[{"text": "Thunderstorm"}],
        ))
        assert impact.penalty_minutes <= 20

    def test_icao_propagated(self):
        impact = _classify_weather(_decoded(icao="VABB"))
        assert impact.station_icao == "VABB"

    def test_summary_not_empty_when_conditions_exist(self):
        impact = _classify_weather(_decoded(flight_category="IFR"))
        assert impact.summary != "stable airport weather"
        assert len(impact.summary) > 0


# ── Circuit breaker ────────────────────────────────────────────────────────────

class TestCircuitBreaker:
    def _reset_circuit(self):
        """Force circuit closed and reset error counter."""
        with weather_fetcher._CIRCUIT_LOCK:
            weather_fetcher._CIRCUIT_OPEN_UNTIL = 0.0
            weather_fetcher._CONSECUTIVE_ERRORS = 0
        weather_fetcher._WEATHER_CACHE.clear()

    def test_returns_none_when_no_api_key(self):
        self._reset_circuit()
        with patch.dict(os.environ, {}, clear=True):
            # Ensure CHECKWX_API_KEY is absent
            os.environ.pop("CHECKWX_API_KEY", None)
            result = get_nearest_airport_weather(28.56, 77.10, "DEL")
        assert result is None

    def test_returns_cached_result_without_http_call(self):
        self._reset_circuit()
        cached_impact = _classify_weather(_decoded())
        weather_fetcher._WEATHER_CACHE["DEL"] = (time.time(), cached_impact)
        with patch("weather_fetcher.requests.get") as mock_get:
            with patch.dict(os.environ, {"CHECKWX_API_KEY": "test-key"}):
                result = get_nearest_airport_weather(28.56, 77.10, "DEL")
        mock_get.assert_not_called()
        assert result is cached_impact

    def test_circuit_opens_after_threshold_failures(self):
        self._reset_circuit()
        with patch.dict(os.environ, {"CHECKWX_API_KEY": "test-key"}):
            with patch("weather_fetcher.requests.get", side_effect=Exception("timeout")):
                for _ in range(weather_fetcher._CIRCUIT_ERROR_THRESHOLD):
                    get_nearest_airport_weather(28.56, 77.10)
        with weather_fetcher._CIRCUIT_LOCK:
            assert weather_fetcher._CIRCUIT_OPEN_UNTIL > time.time()

    def test_circuit_blocks_calls_while_open(self):
        self._reset_circuit()
        with weather_fetcher._CIRCUIT_LOCK:
            weather_fetcher._CIRCUIT_OPEN_UNTIL = time.time() + 3600  # far future
        with patch("weather_fetcher.requests.get") as mock_get:
            with patch.dict(os.environ, {"CHECKWX_API_KEY": "test-key"}):
                result = get_nearest_airport_weather(28.56, 77.10)
        mock_get.assert_not_called()
        assert result is None

    def test_successful_call_resets_error_counter(self):
        self._reset_circuit()
        with weather_fetcher._CIRCUIT_LOCK:
            weather_fetcher._CONSECUTIVE_ERRORS = 2
        good_response = MagicMock()
        good_response.json.return_value = {"data": [_decoded()]}
        good_response.raise_for_status.return_value = None
        with patch("weather_fetcher.requests.get", return_value=good_response):
            with patch.dict(os.environ, {"CHECKWX_API_KEY": "test-key"}):
                get_nearest_airport_weather(28.56, 77.10, "TEST")
        with weather_fetcher._CIRCUIT_LOCK:
            assert weather_fetcher._CONSECUTIVE_ERRORS == 0
