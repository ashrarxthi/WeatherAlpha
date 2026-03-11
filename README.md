# WeatherAlpha — Setup Guide

A weather-driven energy price intelligence engine powered by
WeatherNext 2 (Google DeepMind) + ERCOT market data + Claude.

---

## What This Does

```
WeatherNext 2 (BigQuery)          ERCOT Historical Prices
       │                                    │
       ▼                                    ▼
  15-day weather forecast     2 years of hourly settlement prices
  (temp, wind, solar, cloud)  (DAM + RTM, $/MWh, spike flags)
       │                                    │
       └─────────────┬──────────────────────┘
                     ▼
           Calibration Model
           (LinearRegression + LogisticRegression)
           Learns: tightness → price, P(spike)
                     │
                     ▼
           15-Day Price Forecast
           (predicted $/MWh + spike probability per day)
                     │
                     ▼
           Claude Trading Signal
           (BUY/SELL, thesis, specific trades, risks)
```

---

## Setup: Step by Step

### 1. Python environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Google Cloud (for WeatherNext 2)

1. Go to https://console.cloud.google.com
2. Create a new project (e.g. `weatheralpha`)
3. Enable the **BigQuery API**:
   - Search "BigQuery API" in the search bar → Enable
4. Create a **Service Account**:
   - IAM & Admin → Service Accounts → Create Service Account
   - Name: `weatheralpha-reader`
   - Role: `BigQuery Data Viewer` + `BigQuery Job User`
5. Download the JSON key:
   - Click the service account → Keys → Add Key → JSON
   - Save to `./credentials/google_service_account.json`
6. Update `config.py`:
   ```python
   GCP_PROJECT_ID = "your-actual-project-id"
   ```

> **Cost note:** WeatherNext 2 is a public dataset. BigQuery charges ~$5/TB
> queried. A typical 15-day ERCOT bounding box query is ~50MB = ~$0.25.

### 3. ERCOT API

1. Go to https://developer.ercot.com
2. Click **Sign Up** (free account)
3. Subscribe to:
   - `NP6-905-CD` — DAM Settlement Point Prices
   - `NP6-970-CD` — RTM Settlement Point Prices
4. Copy your **Subscription Key** from the API portal
5. Update `config.py` (or set environment variables):
   ```python
   ERCOT_USERNAME     = "you@email.com"
   ERCOT_SUBSCRIPTION = "abc123your-key-here"
   ```

### 4. Anthropic API

1. Go to https://console.anthropic.com
2. API Keys → Create Key
3. Update `config.py`:
   ```python
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```

---

## Running It

```bash
# First run — fetches all historical data and trains the model (~5-10 min)
python main.py --train

# Subsequent runs — uses cached model, just fetches today's forecast (~30 sec)
python main.py

# Different ERCOT hub
python main.py --zone HB_HOUSTON

# Skip Claude signal (faster, just price forecast)
python main.py --no-signal
```

### Output files (in `./outputs/`)

| File | Contents |
|------|----------|
| `forecast_YYYYMMDD_HHMM.csv` | 15-day price + spike probability per day |
| `signal_YYYYMMDD_HHMM.json`  | Full Claude trading signal (JSON) |
| `model_metrics.json`          | Model accuracy stats (MAE, AUC) |

---

## Extending This

### Add PJM support
- PJM API: https://api.pjm.com (free registration)
- Replace ERCOT bounding box in `weathernext.py` with PJM footprint
- Swap ERCOT API calls in `ercot.py` with PJM equivalents

### Improve the model
- Add **natural gas price** as a feature (gas = marginal fuel for ERCOT)
  - Henry Hub futures via EIA API: https://api.eia.gov
- Add **ERCOT load forecast** from their API (NP3-566-CD)
- Try **XGBoost** instead of LinearRegression for the price model

### Automate it
- Run `main.py` on a cron job every 6 hours (WeatherNext updates 4x/day)
- Push signal to Slack via webhook
- Store results in Postgres for backtesting

---

## Architecture

```
weatheralpha/
├── config.py               ← All credentials and settings
├── main.py                 ← Entry point / pipeline orchestrator
├── requirements.txt
├── data/
│   ├── weathernext.py      ← BigQuery fetcher + unit conversions
│   └── ercot.py            ← ERCOT API client + price history builder
├── models/
│   ├── calibration.py      ← ML model (train + predict)
│   └── weather_price_model.pkl  ← Saved model (auto-generated)
├── signals/
│   └── generator.py        ← Claude signal synthesis
├── outputs/                ← Generated forecasts + signals
└── .cache/                 ← Cached API responses (auto-generated)
```
