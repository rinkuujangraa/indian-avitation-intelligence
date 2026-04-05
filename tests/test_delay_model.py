"""
tests/test_delay_model.py
-------------------------
Unit tests for delay_model.py — covers load_model, predict_delay_prob,
build_features, and LabelEnc. No trained model file is required for the
majority of tests; model-dependent tests are skipped when the file is absent.

Run with: python -m pytest tests/test_delay_model.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from delay_model import (
    FEATURE_NAMES,
    LabelEnc,
    build_features,
    load_model,
    predict_delay_prob,
)

# Path checked by load_model
from delay_model import MODEL_PATH


# ── LabelEnc ──────────────────────────────────────────────────────────────────

class TestLabelEnc:
    def test_fit_transform_assigns_sequential_ints(self):
        enc = LabelEnc()
        result = enc.fit_transform(pd.Series(["b", "a", "c", "a"]))
        assert set(result.unique()) == {0, 1, 2}

    def test_unseen_gets_minus_one(self):
        enc = LabelEnc()
        enc.fit(pd.Series(["x", "y"]))
        transformed = enc.transform(pd.Series(["x", "z"]))
        assert transformed.iloc[1] == -1

    def test_roundtrip_serialisation(self):
        enc = LabelEnc()
        enc.fit(pd.Series(["AI", "6E", "UK"]))
        enc2 = LabelEnc.from_dict(enc.to_dict())
        assert enc2.transform(pd.Series(["AI"])).iloc[0] == enc.transform(pd.Series(["AI"])).iloc[0]


# ── build_features ────────────────────────────────────────────────────────────

def _minimal_df(n: int = 2) -> pd.DataFrame:
    """Return a minimal DataFrame that build_features can process."""
    return pd.DataFrame({
        "airline_prefix": ["AI"] * n,
        "dep_icao": ["VIDP"] * n,
        "arr_icao": ["VABB"] * n,
        "dep_hour_ist": [9] * n,
        "dep_dow": [1] * n,
        "dep_month": [7] * n,
        "is_weekend": [0] * n,
        "dep_period": ["morning"] * n,
        "route_avg_duration": [90.0] * n,
        "route_flight_count": [50.0] * n,
        "is_delayed": [0] * n,
    })


class TestBuildFeatures:
    def test_returns_correct_columns(self):
        X, _ = build_features(_minimal_df(), fit=True)
        assert list(X.columns) == FEATURE_NAMES

    def test_output_is_float(self):
        X, _ = build_features(_minimal_df(), fit=True)
        assert X.dtypes.apply(lambda d: np.issubdtype(d, np.floating)).all()

    def test_no_nans_in_output(self):
        X, _ = build_features(_minimal_df(), fit=True)
        assert not X.isnull().any().any()

    def test_fit_false_reuses_encoders(self):
        df = _minimal_df(4)
        X_train, encoders = build_features(df.iloc[:2], fit=True)
        X_infer, _ = build_features(df.iloc[2:], encoders=encoders, fit=False)
        # Same shape
        assert X_infer.shape[1] == X_train.shape[1]

    def test_monsoon_flag_june_to_sep(self):
        df = _minimal_df(1)
        df["dep_month"] = 7
        X, _ = build_features(df, fit=True)
        assert X["is_monsoon"].iloc[0] == 1

    def test_monsoon_flag_off_season(self):
        df = _minimal_df(1)
        df["dep_month"] = 3
        X, _ = build_features(df, fit=True)
        assert X["is_monsoon"].iloc[0] == 0

    def test_peak_hour_flag_morning_rush(self):
        df = _minimal_df(1)
        df["dep_hour_ist"] = 8
        X, _ = build_features(df, fit=True)
        assert X["is_peak_hour"].iloc[0] == 1

    def test_peak_hour_flag_off_peak(self):
        df = _minimal_df(1)
        df["dep_hour_ist"] = 14
        X, _ = build_features(df, fit=True)
        assert X["is_peak_hour"].iloc[0] == 0

    def test_cyclical_encodings_bounded(self):
        X, _ = build_features(_minimal_df(4), fit=True)
        assert X["hour_sin"].between(-1, 1).all()
        assert X["hour_cos"].between(-1, 1).all()


# ── load_model ────────────────────────────────────────────────────────────────

class TestLoadModel:
    def test_returns_none_when_file_absent(self, tmp_path):
        result = load_model(str(tmp_path / "nonexistent.pkl"))
        assert result is None

    def test_returns_none_for_corrupt_file(self, tmp_path):
        bad = tmp_path / "bad.pkl"
        bad.write_bytes(b"not a pickle")
        result = load_model(str(bad))
        assert result is None

    @pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Trained model not present")
    def test_loaded_bundle_has_required_keys(self):
        bundle = load_model()
        assert bundle is not None
        for key in ("model", "encoders", "feature_names", "threshold"):
            assert key in bundle, f"Missing key {key!r} in model bundle"

    @pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Trained model not present")
    def test_feature_names_match_constant(self):
        bundle = load_model()
        assert bundle["feature_names"] == FEATURE_NAMES


# ── predict_delay_prob ────────────────────────────────────────────────────────

class TestPredictDelayProb:
    @pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Trained model not present")
    def test_returns_float_in_unit_interval(self):
        bundle = load_model()
        prob = predict_delay_prob(
            bundle,
            airline_prefix="AI",
            dep_icao="VIDP",
            arr_icao="VABB",
            dep_hour_ist=9,
            dep_dow=1,
            dep_month=7,
        )
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    @pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Trained model not present")
    def test_unknown_route_returns_fallback(self):
        bundle = load_model()
        prob = predict_delay_prob(
            bundle,
            airline_prefix="ZZ",
            dep_icao="ZZZZ",
            arr_icao="ZZZZ",
            dep_hour_ist=12,
            dep_dow=3,
        )
        assert 0.0 <= prob <= 1.0

    def test_graceful_fallback_on_bad_bundle(self):
        """predict_delay_prob must not raise with a malformed bundle."""
        bad_bundle = {
            "model": None,
            "lgb_model": None,
            "encoders": {},
            "feature_names": FEATURE_NAMES,
            "threshold": 0.5,
        }
        prob = predict_delay_prob(
            bad_bundle,
            airline_prefix="AI",
            dep_icao="VIDP",
            arr_icao="VABB",
            dep_hour_ist=9,
            dep_dow=1,
        )
        # Should return fallback 0.3
        assert prob == 0.3
