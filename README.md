# WeatherAlpha ⚡

**Weather-driven energy price alpha engine for ERCOT.**

Built on [WeatherNext 2](https://deepmind.google/discover/blog/graphcast-ai-model-for-faster-and-more-accurate-global-weather-forecasting/) (Google DeepMind) × ERCOT market data × Claude AI. Generates 15-day electricity price forecasts and spike probability signals from raw weather physics — before the market prices it in.

🌐 **[Live Dashboard →](https://ashrarxthi.github.io/WeatherAlpha)**

---

## The Alpha Thesis

Electricity prices in ERCOT are primarily driven by **grid tightness** — the gap between demand and renewable generation. That gap is a function of weather: temperature drives demand (cooling/heating load), wind drives supply (35GW of wind capacity in Texas), and cloud cover drives solar output.

WeatherNext 2 gives us a **15-day ensemble weather forecast** with accuracy that beats traditional numerical models. By mapping weather signals → grid tightness → price, we can generate forward-looking price signals **before they're reflected in DAM settlement prices.**

```
WeatherNext 2 Forecast          ERCOT Historical Prices
(temp, wind, cloud, solar)      (2yr DAM + RTM, $/MWh)
         │                               │
         └──────────┬────────────────────┘
                    ▼
          Tightness Calibration Model
          LinearRegression → predicted $/MWh
          LogisticRegression → P(spike > 2× avg)
                    │
                    ▼
          15-Day Price Forecast
          + Spike Probability per Day
                    │
                    ▼
          Claude Trading Signal
          (thesis · specific trades · risk factors)
```

**Target users:** Energy hedge funds, large industrial power buyers, insurtech parametric trigger desks.

---

## Features

- 📡 **Live ERCOT data** — pulls 2 years of DAM + RTM settlement prices via ERCOT Public API
- 🌤️ **WeatherNext 2** — DeepMind's state-of-the-art 15-day ensemble forecast via BigQuery
- 🧠 **ML calibration** — time-series cross-validated regression trained on historical tightness-to-price relationships
- ⚡ **Spike detection** — logistic regression flags days with >2× average price probability
- 🤖 **Claude signal** — AI synthesizes weather + model output into a structured trading thesis
- 📊 **Auto-dashboard** — self-contained HTML dashboard opens on every run, published to GitHub Pages

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/ashrarxthi/WeatherAlpha.git
cd WeatherAlpha
pip install -r requirements.txt
cp config.example.py config.py
```

### 2. Configure credentials

Edit `config.py` with your API keys (see [Setup Guide](#setup-guide) below).

### 3. Run

```bash
# First run — fetches 2yr price history and trains the model (~5-10 min)
python main.py --train

# Subsequent runs — cached model, just fetches today's forecast (~30 sec)
python main.py

# Skip Claude signal
python main.py --no-signal

# Different ERCOT hub
python main.py --zone HB_HOUSTON
```

Dashboard opens automatically in your browser. To publish to GitHub Pages:
```bash
git add docs/
git commit -m "update dashboard"
git push
```

---

## Setup Guide

### Google Cloud (WeatherNext 2)

> ⚠️ WeatherNext 2 access requires manual approval from Google. Submit a request [here](https://console.cloud.google.com/bigquery/analytics-hub/exchanges/projects/871883017250/locations/us/dataExchanges/weathernext_19397e1bcb7/listings/weathernext_2_19a39fe59dd). Approval email comes from `weathernext@google.com` within a few days. The pipeline runs on a synthetic weather proxy until then.

1. Create a GCP project at [console.cloud.google.com](https://console.cloud.google.com)
2. Enable the **BigQuery API**
3. Create a Service Account with roles: `BigQuery Data Viewer` + `BigQuery Job User`
4. Download JSON key → save to `./credentials/google_service_account.json`
5. Once WeatherNext access is approved, subscribe via Analytics Hub and set:
```python
GCP_PROJECT_ID      = "your-project-id"
WEATHERNEXT_TABLE   = "your-project.weathernext_2.weathernext_2_0_0"
```

### ERCOT API

1. Register at [apiexplorer.ercot.com](https://apiexplorer.ercot.com) (free)
2. Subscribe to `NP4-190-CD` (DAM prices) and `NP6-905-CD` (RTM prices)
3. Copy your subscription key
```python
ERCOT_USERNAME     = "you@email.com"
ERCOT_PASSWORD     = "your-password"
ERCOT_SUBSCRIPTION = "your-subscription-key"
```

### Anthropic API

1. Get a key at [console.anthropic.com](https://console.anthropic.com)
```python
ANTHROPIC_API_KEY = "sk-ant-..."
```

---

## Output

| File | Contents |
|------|----------|
| `outputs/forecast_YYYYMMDD_HHMM.csv` | 15-day price + spike probability |
| `outputs/signal_YYYYMMDD_HHMM.json` | Claude trading signal (JSON) |
| `outputs/dashboard.html` | Interactive price dashboard |
| `docs/index.html` | GitHub Pages live dashboard |
| `outputs/model_metrics.json` | MAE, AUC, training stats |

---

## Extending

**Add natural gas as a feature** — Henry Hub futures are the marginal fuel for ERCOT pricing. Fetch via [EIA API](https://api.eia.gov) and add as a feature in `models/calibration.py`.

**Add PJM/CAISO support** — Swap the ERCOT bounding box in `data/weathernext.py` and replace API calls in `data/ercot.py` with PJM or CAISO equivalents.

**Automate it** — Run on a cron every 6 hours (WeatherNext updates 4×/day), push signals to Slack via webhook, store in Postgres for backtesting.

**Improve the model** — Try XGBoost instead of LinearRegression, add ERCOT load forecast (NP3-566-CD) as a feature, or incorporate futures curve data for forward price calibration.

---

## Architecture

```
weatheralpha/
├── config.py               ← credentials + settings (gitignored)
├── config.example.py       ← template for new users
├── main.py                 ← pipeline orchestrator + dashboard generator
├── requirements.txt
├── data/
│   ├── weathernext.py      ← BigQuery fetcher + ERCOT spatial filter
│   └── ercot.py            ← ERCOT API client (DAM + RTM)
├── models/
│   └── calibration.py      ← tightness→price ML model
├── signals/
│   └── generator.py        ← Claude signal synthesis
├── docs/
│   └── index.html          ← GitHub Pages live dashboard
└── outputs/                ← generated forecasts + signals
```

---

## Status

| Component | Status |
|-----------|--------|
| ERCOT DAM price history | ✅ Live |
| ERCOT RTM price history | ✅ Live |
| ML calibration model | ✅ Working |
| Claude trading signal | ✅ Working |
| Dashboard + GitHub Pages | ✅ Live |
| WeatherNext 2 (BigQuery) | ⏳ Pending Google approval |

---

## License

MIT
