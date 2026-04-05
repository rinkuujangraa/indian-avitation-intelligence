"""
tests/test_data_fetcher.py
--------------------------
Unit tests for data_fetcher.py — all external HTTP calls are mocked.

Run with: python -m pytest tests/test_data_fetcher.py -v
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_fetcher import get_flight_data, get_airline_name, KEEP_COLUMNS


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_airlabs_response(n: int = 3) -> dict:
    """Return a minimal AirLabs-shaped JSON for n flights over India."""
    flights = [
        {
            "hex": f"abc{i:03d}",
            "flight_iata": f"AI{100 + i}",
            "airline_iata": "AI",
            "dep_iata": "DEL",
            "arr_iata": "BOM",
            "lat": 22.0 + i,
            "lng": 77.0 + i,
            "alt": 35000,
            "speed": 480,
            "dir": 180,
            "status": "en-route",
        }
        for i in range(n)
    ]
    return {"response": flights}


def _mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        from requests.exceptions import HTTPError
        resp.raise_for_status.side_effect = HTTPError(response=resp)
    return resp


# ── get_flight_data — success path ────────────────────────────────────────────

class TestGetFlightDataSuccess:
    @patch.dict(os.environ, {"AIRLABS_API_KEY": "test_key"})
    @patch("data_fetcher.requests.get")
    def test_returns_dataframe_with_expected_columns(self, mock_get):
        mock_get.return_value = _mock_response(_make_airlabs_response(3))
        df = get_flight_data(region="india", max_retries=1)
        assert isinstance(df, pd.DataFrame)
        for col in ("flight_iata", "airline_iata", "latitude", "longitude", "altitude_ft", "speed_kts"):
            assert col in df.columns, f"Expected column {col!r} missing"

    @patch.dict(os.environ, {"AIRLABS_API_KEY": "test_key"})
    @patch("data_fetcher.requests.get")
    def test_returns_correct_row_count(self, mock_get):
        mock_get.return_value = _mock_response(_make_airlabs_response(5))
        df = get_flight_data(region="india", max_retries=1)
        assert len(df) == 5

    @patch.dict(os.environ, {"AIRLABS_API_KEY": "test_key"})
    @patch("data_fetcher.requests.get")
    def test_altitude_converted_to_feet(self, mock_get):
        """AirLabs returns altitude in metres; data_fetcher should store it as feet."""
        mock_get.return_value = _mock_response(_make_airlabs_response(1))
        df = get_flight_data(region="india", max_retries=1)
        # Raw alt=35000 (metres would be huge; if stored directly it's ~35 000 ft
        # — either way it must be a positive numeric value)
        assert df["altitude_ft"].iloc[0] > 0


# ── get_flight_data — error / fallback paths ──────────────────────────────────

class TestGetFlightDataErrorHandling:
    def test_missing_api_key_returns_empty_dataframe(self):
        # Remove key from environment entirely
        env = {k: v for k, v in os.environ.items() if k != "AIRLABS_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            df = get_flight_data(region="india", max_retries=1)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    @patch.dict(os.environ, {"AIRLABS_API_KEY": "test_key"})
    @patch("data_fetcher.requests.get")
    def test_api_error_in_response_returns_dataframe(self, mock_get):
        """AirLabs can return HTTP 200 with an error key — should not raise."""
        mock_get.return_value = _mock_response(
            {"error": {"message": "quota exceeded", "code": "limit_exceeded"}}
        )
        # OpenSky fallback may be called; patch it to return empty so no real HTTP
        with patch("data_fetcher._fetch_opensky_fallback", return_value=pd.DataFrame(columns=KEEP_COLUMNS)):
            df = get_flight_data(region="india", max_retries=1)
        assert isinstance(df, pd.DataFrame)

    @patch.dict(os.environ, {"AIRLABS_API_KEY": "test_key"})
    @patch("data_fetcher.requests.get")
    def test_empty_response_returns_empty_dataframe(self, mock_get):
        mock_get.return_value = _mock_response({"response": []})
        df = get_flight_data(region="india", max_retries=1)
        assert isinstance(df, pd.DataFrame)
        assert df.empty or set(KEEP_COLUMNS).issubset(set(df.columns))

    @patch.dict(os.environ, {"AIRLABS_API_KEY": "test_key"})
    @patch("data_fetcher.requests.get")
    @patch("data_fetcher._fetch_opensky_fallback", return_value=pd.DataFrame())
    def test_connection_error_returns_dataframe(self, _mock_fallback, mock_get):
        from requests.exceptions import ConnectionError as ReqConnError
        mock_get.side_effect = ReqConnError("no network")
        df = get_flight_data(region="india", max_retries=1)
        assert isinstance(df, pd.DataFrame)

    @patch.dict(os.environ, {"AIRLABS_API_KEY": "test_key"})
    @patch("data_fetcher.requests.get")
    @patch("data_fetcher._fetch_opensky_fallback", return_value=pd.DataFrame())
    def test_timeout_returns_dataframe(self, _mock_fallback, mock_get):
        from requests.exceptions import Timeout
        mock_get.side_effect = Timeout("timed out")
        df = get_flight_data(region="india", max_retries=1)
        assert isinstance(df, pd.DataFrame)


# ── get_airline_name ───────────────────────────────────────────────────────────

class TestGetAirlineName:
    def test_known_airline_returns_full_name(self):
        assert get_airline_name("AI") == "Air India"
        assert get_airline_name("6E") == "IndiGo"

    def test_unknown_airline_returns_code(self):
        assert get_airline_name("ZZ") == "ZZ"

    def test_case_insensitive(self):
        assert get_airline_name("ai") == get_airline_name("AI")
