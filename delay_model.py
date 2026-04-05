"""
delay_model.py
--------------
ML training pipeline for Indian airline delay prediction.

Uses historical flight data from OpenSky Trino (flights_data4) to train an
XGBoost binary classifier that predicts if a flight will be delayed (>15 min).

Features:
    dep_hour_ist       - departure hour (IST, 0-23)
    dep_dow            - day of week (0=Mon, 6=Sun)
    is_weekend         - 1 if Sat/Sun
    dep_period_enc     - encoded: Night=0, Morning=1, Afternoon=2, Evening=3
    route_avg_duration - historical median duration of the route (minutes)
    route_flight_count - monthly frequency of this route (proxy for congestion)
    airline_enc        - label-encoded airline prefix
    route_id_enc       - label-encoded dep_icao+arr_icao pair

Target:
    is_delayed - 1 if delay_proxy_min > 15

Run:
    python3 delay_model.py
    python3 delay_model.py --start 2024-01-01 --end 2024-03-31 --limit 300000

Model saved to:
    models/delay_lgbm.pkl   (dict: model, encoders, feature_names, threshold)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

# xgboost and lightgbm are only imported inside train() — they are heavy (~1.3s)
# and not needed for inference (load_model / predict_delay_prob use joblib + numpy).

# ── Logging ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("delay_model")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "delay_lgbm.pkl")

FEATURE_NAMES = [
    "dep_hour_ist",
    "dep_dow",
    "dep_month",              # month 1-12: monsoon / fog / festive
    "is_weekend",
    "is_peak_hour",            # rush hour flag (7-10am, 5-9pm IST)
    "is_monsoon",              # 1 if June-Sep (heavy rain delays)
    "is_fog_season",           # 1 if Dec-Feb at northern India airports
    "is_high_season",          # 1 if Oct-Jan (Diwali/Navratri/Christmas/NY)
    "dep_period_enc",
    "route_avg_duration",
    "route_flight_count",
    "route_delay_rate",        # per-route historical delay rate (train-set only)
    "airline_route_delay_rate",# per (airline × route) delay rate (train-set only)
    "airline_enc",
    "route_id_enc",
    "dep_airport_enc",         # departure airport
    "arr_airport_enc",         # arrival airport
    "route_avg_delay_min",     # mean delay minutes for this dep→arr pair
    "dep_airport_delay_rate",  # fraction of departures from this airport that are delayed
    "arr_airport_delay_rate",  # fraction of arrivals at this airport that are delayed
    "hour_sin",                # cyclical: sin(2π · hour / 24)
    "hour_cos",                # cyclical: cos(2π · hour / 24)
    "month_sin",               # cyclical: sin(2π · month / 12)
    "month_cos",               # cyclical: cos(2π · month / 12)
    "route_std_duration",      # std deviation of flight duration on route
]

PERIOD_MAP = {"night": 0, "morning": 1, "afternoon": 2, "evening": 3}


# ── Simple label encoder (no sklearn) ───────────────────────────────────────────
class LabelEnc:
    """Minimal label encoder backed by a plain dict."""

    def __init__(self) -> None:
        self._map: dict[str, int] = {}
        self._unseen = -1

    def fit(self, series: pd.Series) -> "LabelEnc":
        unique = sorted(series.dropna().unique())
        self._map = {v: i for i, v in enumerate(unique)}
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        return series.map(self._map).fillna(self._unseen).astype(int)

    def fit_transform(self, series: pd.Series) -> pd.Series:
        return self.fit(series).transform(series)

    def to_dict(self) -> dict:
        return {"map": self._map, "unseen": self._unseen}

    @classmethod
    def from_dict(cls, d: dict) -> "LabelEnc":
        enc = cls()
        enc._map = d["map"]
        enc._unseen = d["unseen"]
        return enc


# ── Data loading ─────────────────────────────────────────────────────────────────
def load_training_data(
    start: str,
    end: str,
    limit: int = 300_000,
) -> pd.DataFrame:
    """Pull ML dataset from OpenSky Trino."""
    log.info("Loading training data %s → %s (limit=%d)…", start, end, limit)
    try:
        from opensky_trino import build_ml_dataset  # noqa: PLC0415
    except ImportError as exc:
        log.error("Cannot import opensky_trino: %s", exc)
        sys.exit(1)

    df = build_ml_dataset(start, end, limit=limit)
    log.info("Loaded %d flight records", len(df))
    return df


# ── Feature engineering ──────────────────────────────────────────────────────────
def build_features(
    df: pd.DataFrame,
    encoders: dict[str, LabelEnc] | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict[str, LabelEnc]]:
    """
    Build the feature matrix from a raw ML dataset DataFrame.

    If `fit=True`  → create new LabelEnc objects and fit them.
    If `fit=False` → use the provided `encoders` dict (inference mode).
    """
    _FOG_AIRPORTS = {"VIDP", "VILK", "VIAG", "VIJP", "VIAR", "VIPT", "VIDL"}

    df = df.copy()

    # ── Derived columns ──────────────────────────────────────────────────────
    df["route_id"] = df["dep_icao"].fillna("?") + "-" + df["arr_icao"].fillna("?")
    # Drop column first so pandas never tries to validate against old Categorical dtype
    raw_period = df["dep_period"].astype(str).str.lower() if "dep_period" in df.columns else pd.Series(["morning"] * len(df), index=df.index)
    df = df.drop(columns=["dep_period"], errors="ignore")
    df["dep_period"] = raw_period.fillna("morning")
    df["dep_period_enc"] = df["dep_period"].map(PERIOD_MAP).fillna(1).astype(int)
    df["is_weekend"] = df.get("is_weekend", 0).fillna(0).astype(int)
    df["dep_hour_ist"] = df.get("dep_hour_ist", 12).fillna(12).astype(int)
    df["dep_dow"] = df.get("dep_dow", 0).fillna(0).astype(int)
    df["route_avg_duration"] = df.get("route_avg_duration", 90.0).fillna(90.0).astype(float)
    df["route_flight_count"] = df.get("route_flight_count", 1.0).fillna(1.0).astype(float)

    # New features
    df["dep_month"] = pd.to_numeric(df.get("dep_month", datetime.now().month), errors="coerce").fillna(datetime.now().month).astype(int).clip(1, 12)
    df["is_peak_hour"] = df["dep_hour_ist"].apply(lambda h: 1 if (7 <= h <= 10 or 17 <= h <= 21) else 0)
    df["route_delay_rate"] = pd.to_numeric(df.get("route_delay_rate", pd.Series([0.15]*len(df), index=df.index)), errors="coerce").fillna(0.15).clip(0.0, 1.0)
    df["airline_route_delay_rate"] = pd.to_numeric(df.get("airline_route_delay_rate", pd.Series([0.15]*len(df), index=df.index)), errors="coerce").fillna(0.15).clip(0.0, 1.0)
    df["route_avg_delay_min"] = pd.to_numeric(df.get("route_avg_delay_min", pd.Series([12.0]*len(df), index=df.index)), errors="coerce").fillna(12.0).clip(0.0, 300.0)
    df["dep_airport_delay_rate"] = pd.to_numeric(df.get("dep_airport_delay_rate", pd.Series([0.15]*len(df), index=df.index)), errors="coerce").fillna(0.15).clip(0.0, 1.0)
    df["arr_airport_delay_rate"] = pd.to_numeric(df.get("arr_airport_delay_rate", pd.Series([0.15]*len(df), index=df.index)), errors="coerce").fillna(0.15).clip(0.0, 1.0)

    # Season flags — derived from dep_month and dep_icao (work at both train and inference)
    dep_month_s = df["dep_month"]
    df["is_monsoon"] = dep_month_s.isin([6, 7, 8, 9]).astype(int)
    df["is_high_season"] = dep_month_s.isin([10, 11, 12, 1]).astype(int)
    df["is_fog_season"] = (
        dep_month_s.isin([12, 1, 2]) & df["dep_icao"].isin(_FOG_AIRPORTS)
    ).astype(int)

    # Cyclical time encodings (let model understand circular nature of time)
    df["hour_sin"] = np.sin(2 * np.pi * df["dep_hour_ist"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["dep_hour_ist"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["dep_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["dep_month"] / 12)

    # Route volatility (high std = unreliable route = more delays)
    df["route_std_duration"] = pd.to_numeric(
        df.get("route_std_duration", pd.Series([0.0] * len(df), index=df.index)),
        errors="coerce"
    ).fillna(0.0).clip(0, 120)

    # Clip plausible ranges
    df["dep_hour_ist"] = df["dep_hour_ist"].clip(0, 23)
    df["dep_dow"] = df["dep_dow"].clip(0, 6)
    df["route_avg_duration"] = df["route_avg_duration"].clip(20, 500)
    df["route_flight_count"] = df["route_flight_count"].clip(1, 10_000)

    # ── Label encoding ───────────────────────────────────────────────────────
    if fit:
        encoders = {}
        airline_enc = LabelEnc()
        route_enc = LabelEnc()
        dep_airport_enc = LabelEnc()
        arr_airport_enc = LabelEnc()
        df["airline_enc"] = airline_enc.fit_transform(df["airline_prefix"].fillna("UNK"))
        df["route_id_enc"] = route_enc.fit_transform(df["route_id"])
        df["dep_airport_enc"] = dep_airport_enc.fit_transform(df["dep_icao"].fillna("UNK"))
        df["arr_airport_enc"] = arr_airport_enc.fit_transform(df["arr_icao"].fillna("UNK"))
        encoders["airline"] = airline_enc
        encoders["route_id"] = route_enc
        encoders["dep_airport"] = dep_airport_enc
        encoders["arr_airport"] = arr_airport_enc
    else:
        assert encoders is not None, "Must provide encoders in inference mode"
        df["airline_enc"] = encoders["airline"].transform(df["airline_prefix"].fillna("UNK"))
        df["route_id_enc"] = encoders["route_id"].transform(df["route_id"])
        df["dep_airport_enc"] = encoders["dep_airport"].transform(df["dep_icao"].fillna("UNK"))
        df["arr_airport_enc"] = encoders["arr_airport"].transform(df["arr_icao"].fillna("UNK"))

    X = df[FEATURE_NAMES].astype(float)
    return X, encoders


# ── Training ──────────────────────────────────────────────────────────────────────
def train(df: pd.DataFrame) -> dict:
    """Train XGBoost classifier and return a model bundle."""
    import xgboost as xgb   # noqa: PLC0415 — lazy import, only needed for training
    import lightgbm as lgb  # noqa: PLC0415
    if "is_delayed" not in df.columns:
        log.error("Dataset missing 'is_delayed' column")
        sys.exit(1)

    y = df["is_delayed"].fillna(0).astype(int)

    # Drop rows with missing target
    mask = y.notna()
    df = df[mask]
    y = y[mask]

    log.info("Class distribution: delayed=%d  on-time=%d", y.sum(), (y == 0).sum())

    X, encoders = build_features(df, fit=True)

    # Stratified shuffle split for better-distributed validation
    from sklearn.model_selection import StratifiedShuffleSplit  # noqa: PLC0415
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, val_idx = next(sss.split(X, y))
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Compute leak-free aggregates from training portion only
    train_df = df.iloc[train_idx].copy()
    val_df   = df.iloc[val_idx].copy()
    global_rate = float(train_df["is_delayed"].mean())

    # Per-route delay rate
    rdm = train_df.groupby("route_id")["is_delayed"].mean().to_dict()

    # Per-route mean delay minutes (stronger signal than binary rate)
    delay_col = "delay_proxy_min" if "delay_proxy_min" in train_df.columns else None
    if delay_col:
        radm = train_df.groupby("route_id")[delay_col].mean().to_dict()
        global_avg_delay = float(train_df[delay_col].mean())
    else:
        radm = {}
        global_avg_delay = 12.0

    # Per-airport departure delay rate
    dep_ap_dm = train_df.groupby("dep_icao")["is_delayed"].mean().to_dict()
    # Per-airport arrival delay rate
    arr_ap_dm = train_df.groupby("arr_icao")["is_delayed"].mean().to_dict()

    # Per (airline × route) delay rate
    _ar_groups = train_df.groupby(["airline_prefix", "route_id"])["is_delayed"].mean()
    ardm = {f"{a}:{r}": v for (a, r), v in _ar_groups.items()}

    X_train = X_train.copy()
    X_val   = X_val.copy()

    if "route_delay_rate" in FEATURE_NAMES:
        col_idx = FEATURE_NAMES.index("route_delay_rate")
        X_train.iloc[:, col_idx] = train_df["route_id"].map(rdm).fillna(global_rate).values
        X_val.iloc[:, col_idx]   = val_df["route_id"].map(rdm).fillna(global_rate).values

    if "route_avg_delay_min" in FEATURE_NAMES:
        rad_idx = FEATURE_NAMES.index("route_avg_delay_min")
        X_train.iloc[:, rad_idx] = train_df["route_id"].map(radm).fillna(global_avg_delay).values
        X_val.iloc[:, rad_idx]   = val_df["route_id"].map(radm).fillna(global_avg_delay).values

    if "dep_airport_delay_rate" in FEATURE_NAMES:
        dep_idx = FEATURE_NAMES.index("dep_airport_delay_rate")
        X_train.iloc[:, dep_idx] = train_df["dep_icao"].map(dep_ap_dm).fillna(global_rate).values
        X_val.iloc[:, dep_idx]   = val_df["dep_icao"].map(dep_ap_dm).fillna(global_rate).values

    if "arr_airport_delay_rate" in FEATURE_NAMES:
        arr_idx = FEATURE_NAMES.index("arr_airport_delay_rate")
        X_train.iloc[:, arr_idx] = train_df["arr_icao"].map(arr_ap_dm).fillna(global_rate).values
        X_val.iloc[:, arr_idx]   = val_df["arr_icao"].map(arr_ap_dm).fillna(global_rate).values

    if "airline_route_delay_rate" in FEATURE_NAMES:
        def _ar_rate(airline: str, route: str) -> float:
            return ardm.get(f"{airline}:{route}", rdm.get(route, global_rate))
        ar_idx = FEATURE_NAMES.index("airline_route_delay_rate")
        X_train.iloc[:, ar_idx] = [
            _ar_rate(a, r)
            for a, r in zip(train_df["airline_prefix"].fillna("UNK"), train_df["route_id"])
        ]
        X_val.iloc[:, ar_idx] = [
            _ar_rate(a, r)
            for a, r in zip(val_df["airline_prefix"].fillna("UNK"), val_df["route_id"])
        ]

    # Route std duration (leak-free: computed from train split only)
    _rstd = {}
    if "route_std_duration" in FEATURE_NAMES:
        if "fl_duration_min" in train_df.columns:
            _rstd = train_df.groupby("route_id")["fl_duration_min"].std().fillna(0).to_dict()
        std_idx = FEATURE_NAMES.index("route_std_duration")
        X_train.iloc[:, std_idx] = train_df["route_id"].map(_rstd).fillna(0).values
        X_val.iloc[:, std_idx] = val_df["route_id"].map(_rstd).fillna(0).values

    # Route stats maps for inference lookup
    route_avg_dur_map = train_df.groupby("route_id")["route_avg_duration"].first().to_dict() if "route_avg_duration" in train_df.columns else {}
    route_count_map = train_df.groupby("route_id").size().to_dict()
    route_std_map = _rstd

    # Class imbalance ratio
    neg = (y_train == 0).sum()
    pos = y_train.sum()
    scale_pos = float(neg) / float(pos) if pos > 0 else 1.0
    log.info("scale_pos_weight=%.2f", scale_pos)

    model = xgb.XGBClassifier(
        n_estimators=2500,
        learning_rate=0.02,
        max_depth=7,
        min_child_weight=3,
        gamma=0.15,
        max_delta_step=1,
        subsample=0.85,
        colsample_bytree=0.8,
        colsample_bylevel=0.8,
        reg_alpha=0.05,
        reg_lambda=1.5,
        scale_pos_weight=scale_pos,
        eval_metric="aucpr",
        early_stopping_rounds=100,
        random_state=42,
        verbosity=0,
        tree_method="hist",
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # LightGBM ensemble partner
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=-1,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=1.5,
        is_unbalance=True,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
    )

    # Find optimal ensemble weight (XGB vs LGB) and threshold jointly
    xgb_val_probs = model.predict_proba(X_val)[:, 1]
    lgb_val_probs = lgb_model.predict_proba(X_val)[:, 1]

    best_w, best_f1_ens, best_t_ens = 1.0, 0.0, 0.5
    for _w in [i / 10 for i in range(0, 11)]:
        _wp = _w * xgb_val_probs + (1 - _w) * lgb_val_probs
        for _t in [i / 100 for i in range(20, 70)]:
            _preds = (_wp >= _t).astype(int)
            _tp = int(((_preds == 1) & (y_val == 1)).sum())
            _fp = int(((_preds == 1) & (y_val == 0)).sum())
            _fn = int(((_preds == 0) & (y_val == 1)).sum())
            _p = _tp / (_tp + _fp) if (_tp + _fp) else 0.0
            _r = _tp / (_tp + _fn) if (_tp + _fn) else 0.0
            _f = 2 * _p * _r / (_p + _r) if (_p + _r) else 0.0
            if _f > best_f1_ens:
                best_f1_ens, best_w, best_t_ens = _f, _w, _t

    ensemble_weight = best_w
    log.info("Optimal ensemble: XGB_weight=%.1f  threshold=%.2f  F1=%.3f",
             best_w, best_t_ens, best_f1_ens)
    val_probs = ensemble_weight * xgb_val_probs + (1 - ensemble_weight) * lgb_val_probs

    best_t, best_f1_val = 0.5, 0.0
    for _t in [i / 100 for i in range(15, 75)]:
        _preds = (val_probs >= _t).astype(int)
        _tp = int(((_preds == 1) & (y_val == 1)).sum())
        _fp = int(((_preds == 1) & (y_val == 0)).sum())
        _fn = int(((_preds == 0) & (y_val == 1)).sum())
        _p = _tp / (_tp + _fp) if (_tp + _fp) else 0.0
        _r = _tp / (_tp + _fn) if (_tp + _fn) else 0.0
        _f = 2 * _p * _r / (_p + _r) if (_p + _r) else 0.0
        if _f > best_f1_val:
            best_f1_val, best_t = _f, _t
    log.info("Optimal threshold: %.2f (F1=%.3f)", best_t, best_f1_val)

    val_preds = (val_probs >= best_t).astype(int)
    tp = int(((val_preds == 1) & (y_val == 1)).sum())
    fp = int(((val_preds == 1) & (y_val == 0)).sum())
    fn = int(((val_preds == 0) & (y_val == 1)).sum())
    tn = int(((val_preds == 0) & (y_val == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_val) if len(y_val) else 0.0

    log.info("Val — Accuracy: %.3f  Precision: %.3f  Recall: %.3f  F1: %.3f",
             accuracy, precision, recall, f1)
    log.info("Val — TP=%d  FP=%d  FN=%d  TN=%d", tp, fp, fn, tn)

    # Store rate maps for inference lookup
    bundle_route_delay_map = rdm
    bundle_route_avg_delay_map = radm
    bundle_airline_route_map = ardm
    bundle_dep_airport_map = dep_ap_dm
    bundle_arr_airport_map = arr_ap_dm
    global_delay_rate = global_rate

    # Feature importance
    importance = dict(zip(FEATURE_NAMES, model.feature_importances_))
    log.info("Feature importance: %s",
             json.dumps({k: round(float(v), 4) for k, v in
                         sorted(importance.items(), key=lambda x: -x[1])}))

    return {
        "model": model,
        "lgb_model": lgb_model,
        "encoders": {k: v.to_dict() for k, v in encoders.items()},
        "feature_names": FEATURE_NAMES,
        "threshold": best_t,
        "route_delay_map": bundle_route_delay_map,
        "route_avg_delay_map": bundle_route_avg_delay_map,
        "airline_route_delay_map": bundle_airline_route_map,
        "dep_airport_delay_map": bundle_dep_airport_map,
        "arr_airport_delay_map": bundle_arr_airport_map,
        "route_avg_duration_map": route_avg_dur_map,
        "route_flight_count_map": route_count_map,
        "route_std_duration_map": route_std_map,
        "ensemble_weight": ensemble_weight,
        "global_delay_rate": global_delay_rate,
        "global_avg_delay_min": global_avg_delay,
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        },
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "n_train": len(X_train),
        "n_val": len(X_val),
    }


# ── Inference helper (used by analytics.py) ──────────────────────────────────────
def load_model(path: str = MODEL_PATH) -> dict | None:
    """Load the saved model bundle. Returns None if not found."""
    if not os.path.exists(path):
        return None
    try:
        bundle = joblib.load(path)
        # Re-hydrate LabelEnc objects from dicts
        bundle["encoders"] = {
            k: LabelEnc.from_dict(v) for k, v in bundle["encoders"].items()
        }
        return bundle
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("delay_model").warning("Failed to load model: %s", exc)
        return None


def predict_delay_prob(
    bundle: dict,
    airline_prefix: str,
    dep_icao: str,
    arr_icao: str,
    dep_hour_ist: int,
    dep_dow: int,
    dep_month: int | None = None,
) -> float:
    """
    Return predicted delay probability (0-1) for a single flight.

    Parameters match the kind of data available from live AirLabs feed.
    Falls back to 0.3 if the route is unknown.
    """
    if dep_month is None:
        dep_month = datetime.now().month
    route_id = f"{dep_icao}-{arr_icao}"  # must match build_features separator
    global_rate = bundle.get("global_delay_rate", 0.15)
    route_delay_rate = bundle.get("route_delay_map", {}).get(route_id, global_rate)
    ar_key = f"{airline_prefix}:{route_id}"
    airline_route_delay_rate = bundle.get("airline_route_delay_map", {}).get(ar_key, route_delay_rate)
    route_avg_delay_min = bundle.get("route_avg_delay_map", {}).get(route_id, bundle.get("global_avg_delay_min", 12.0))
    dep_airport_delay_rate = bundle.get("dep_airport_delay_map", {}).get(dep_icao, global_rate)
    arr_airport_delay_rate = bundle.get("arr_airport_delay_map", {}).get(arr_icao, global_rate)
    route_avg_duration = bundle.get("route_avg_duration_map", {}).get(route_id, 90.0)
    route_flight_count = bundle.get("route_flight_count_map", {}).get(route_id, 50.0)
    route_std_duration = bundle.get("route_std_duration_map", {}).get(route_id, 0.0)
    row = pd.DataFrame([{
        "airline_prefix": airline_prefix,
        "dep_icao": dep_icao,
        "arr_icao": arr_icao,
        "dep_hour_ist": dep_hour_ist,
        "dep_dow": dep_dow,
        "dep_month": dep_month,
        "is_weekend": 1 if dep_dow >= 5 else 0,
        "dep_period": (
            "night" if dep_hour_ist < 6 else
            "morning" if dep_hour_ist < 12 else
            "afternoon" if dep_hour_ist < 18 else
            "evening"
        ),
        "route_avg_duration": route_avg_duration,
        "route_flight_count": route_flight_count,
        "route_delay_rate": route_delay_rate,
        "airline_route_delay_rate": airline_route_delay_rate,
        "route_avg_delay_min": route_avg_delay_min,
        "dep_airport_delay_rate": dep_airport_delay_rate,
        "arr_airport_delay_rate": arr_airport_delay_rate,
        "route_std_duration": route_std_duration,
        "route_id": route_id,
    }])

    try:
        X, _ = build_features(row, encoders=bundle["encoders"], fit=False)
        ew = bundle.get("ensemble_weight", 0.5)
        xgb_prob = float(bundle["model"].predict_proba(X)[0, 1])
        if "lgb_model" in bundle and bundle["lgb_model"] is not None:
            lgb_prob = float(bundle["lgb_model"].predict_proba(X)[0, 1])
            prob = ew * xgb_prob + (1 - ew) * lgb_prob
        else:
            prob = xgb_prob
    except Exception:  # noqa: BLE001
        return 0.3

    return round(prob, 4)


# ── CLI ───────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Train Indian airline delay prediction model")
    parser.add_argument("--start", default="2024-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2024-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--limit", type=int, default=300_000, help="Max row fetch limit")
    parser.add_argument("--output", default=MODEL_PATH, help="Output path for model bundle")
    args = parser.parse_args()

    df = load_training_data(args.start, args.end, args.limit)

    if df.empty:
        log.error("No data returned — check OpenSky credentials and date range")
        sys.exit(1)

    bundle = train(df)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    joblib.dump(bundle, args.output)
    log.info("Model saved → %s  (accuracy=%.3f  F1=%.3f)",
             args.output,
             bundle["metrics"]["accuracy"],
             bundle["metrics"]["f1"])
    log.info("Training summary: %d train / %d val records", bundle["n_train"], bundle["n_val"])


if __name__ == "__main__":
    main()
