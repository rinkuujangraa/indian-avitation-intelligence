# Aviation Intelligence Platform

A real-time Indian aviation dashboard built with Streamlit, Mapbox GL JS, and an XGBoost + LightGBM delay prediction model.

## Features

- **Live flight map** — Mapbox GL JS with per-aircraft icons, route arcs, and fuel/CO₂ estimates
- **Delay prediction** — rule-based scoring blended with an ML ensemble (XGBoost + LightGBM) trained on OpenSky historical data
- **Airport congestion** — AFRI (Arrival Flow Risk Index), schedule pressure, and traffic heatmap
- **Weather overlay** — live METAR data via CheckWX with penalty scoring
- **Anomaly detection** — holding patterns, go-arounds, diversions, and slow approaches
- **Time-slider playback** — SQLite-backed 24 h flight history snapshots
- **Route intelligence** — per-route congestion scores and hotspot delay minutes

## Quick start

```bash
# 1. Clone
git clone https://github.com/your-org/aviation-intelligence-platform
cd aviation-intelligence-platform

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure credentials
cp .env.example .env
# Edit .env and fill in your API keys (see below)

# 5. Run
streamlit run app.py
```

## Environment variables

Copy `.env.example` to `.env` and set the following:

| Variable | Required | Description |
|---|---|---|
| `AIRLABS_API_KEY` | ✅ | [AirLabs](https://airlabs.co/) API key for live flight data |
| `MAPBOX_TOKEN` | ✅ | [Mapbox](https://account.mapbox.com/) public token (`pk.…`) for the map |
| `CHECKWX_API_KEY` | Recommended | [CheckWX](https://www.checkwx.com/) key for live METAR weather |
| `CESIUM_TOKEN` | Optional | [Cesium Ion](https://ion.cesium.com/) token for the 3-D satellite map |
| `OPENSKY_USERNAME` | Training only | [OpenSky Network](https://opensky-network.org/) credentials for ML training |
| `OPENSKY_PASSWORD` | Training only | See above |

The app runs without `CHECKWX_API_KEY` (weather panels show "Unavailable") and without `CESIUM_TOKEN` (falls back to Mapbox 2-D map).

## ML model training (optional)

A pre-trained model is not included in the repository. To train your own:

```bash
python delay_model.py --start 2024-01-01 --end 2024-12-31
```

This requires valid `OPENSKY_USERNAME` / `OPENSKY_PASSWORD` in `.env` and may take ~10 minutes.  
The trained bundle is saved to `models/delay_lgbm.pkl`. The app lazily loads it on first run if present.

## Project structure

```
app.py                  Streamlit entry point
analytics.py            Delay prediction, congestion, anomaly detection
data_fetcher.py         AirLabs API client + flight cache
mapbox_base.py          Mapbox GL JS map HTML generator
tracker.py              In-memory + on-disk flight trail tracker
snapshot_store.py       SQLite snapshot store (time-slider playback)
weather_fetcher.py      CheckWX METAR client with circuit breaker
delay_model.py          XGBoost + LightGBM training pipeline
utils.py                Shared helpers (haversine, map builder)
aircraft_icons.py       SVG aircraft icon registry
tests/                  Pytest test suite
models/                 Trained model bundle (gitignored — train locally)
```

## Running tests

```bash
python -m pytest tests/ -v
```

## License

MIT
