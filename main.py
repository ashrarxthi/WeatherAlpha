"""
WeatherAlpha — Main Pipeline
─────────────────────────────
Run this file to execute the full pipeline:

  1. Fetch WeatherNext 2 15-day forecast from BigQuery
  2. Pull ERCOT historical prices (first run only — cached after)
  3. Train / load the calibrated tightness-to-price model
  4. Generate 15-day price + spike forecast
  5. Ask Claude to synthesise a trading signal
  6. Print report + save outputs

Usage:
    # First time (trains the model):
    python main.py --train

    # Subsequent runs (loads cached model):
    python main.py

    # Specific zone:
    python main.py --zone HB_HOUSTON

    # Skip Claude signal (just show the price forecast):
    python main.py --no-signal
"""

from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import DEFAULT_ZONE, FORECAST_DAYS
from data.weathernext import _get_client as get_bq_client, fetch_ercot_forecast, aggregate_to_daily
from data.ercot        import build_daily_price_history
from models.calibration import WeatherPriceModel, build_feature_matrix
from signals.generator  import generate_trading_signal, format_signal_report

CACHE_DIR  = Path(__file__).parent / ".cache"
OUTPUT_DIR = Path(__file__).parent / "outputs"
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def _synthetic_forecast(days: int = 15) -> pd.DataFrame:
    """
    Generate a synthetic 15-day weather forecast as a proxy for WeatherNext.
    Uses seasonal temperature curves for ERCOT/Texas.
    Replaced automatically once WeatherNext BigQuery access is approved.
    """
    import numpy as np
    print("[WeatherNext] ⚠️  Using synthetic forecast proxy (awaiting Google access approval)")
    today = datetime.utcnow().date()
    dates = [today + timedelta(days=i) for i in range(days)]
    doy   = [d.timetuple().tm_yday for d in dates]

    np.random.seed(42)
    temp_mean = [65 + 18 * np.sin((d - 80) * 2 * np.pi / 365) for d in doy]
    rows = []
    for i, date in enumerate(dates):
        t = temp_mean[i]
        rows.append({
            "date":               pd.Timestamp(date),
            "temp_f_mean":        round(t + np.random.randn() * 3, 1),
            "temp_f_max":         round(t + 8 + np.random.randn() * 2, 1),
            "temp_f_min":         round(t - 8 + np.random.randn() * 2, 1),
            "cdd":                round(max(0, t - 65), 1),
            "hdd":                round(max(0, 65 - t), 1),
            "wind_mph_mean":      round(12 + 4 * np.cos((doy[i] - 100) * 2 * np.pi / 365) + np.random.randn() * 2, 1),
            "cloud_pct":          round(np.clip(40 + 15 * np.random.randn(), 0, 100), 1),
            "demand_proxy":       round(40000 + max(0, t - 65) * 500 + max(0, 65 - t) * 400, 0),
            "solar_gen_proxy":    round(max(0, 1 - np.clip(40 + 15 * np.random.randn(), 0, 100) / 100) * 0.18 * 20000, 0),
            "wind_gen_proxy":     round((12 + 4 * np.cos((doy[i] - 100) * 2 * np.pi / 365)) / 30 * 35000, 0),
        })
    df = pd.DataFrame(rows)
    df["renewable_gen_proxy"] = df["wind_gen_proxy"] + df["solar_gen_proxy"]
    df["tightness"]           = df["demand_proxy"] - df["renewable_gen_proxy"]
    return df


# ── Step helpers ──────────────────────────────────────────────────────────────
def step_fetch_forecast(zone: str) -> pd.DataFrame:
    """Fetch + aggregate WeatherNext 2 forecast, with synthetic fallback."""
    print(f"\n{'='*60}")
    print(f"STEP 1: Fetching WeatherNext 2 forecast ({FORECAST_DAYS} days)")
    print("="*60)
    try:
        bq     = get_bq_client()
        hourly = fetch_ercot_forecast(client=bq)
        daily  = aggregate_to_daily(hourly)
    except Exception as e:
        if "403" in str(e) or "Access Denied" in str(e) or "does not exist" in str(e):
            print(f"[WeatherNext] BigQuery access not yet approved — using synthetic proxy")
            daily = _synthetic_forecast(days=FORECAST_DAYS)
        else:
            raise
    cache_path = CACHE_DIR / f"forecast_{datetime.utcnow().strftime('%Y%m%d')}.csv"
    daily.to_csv(cache_path, index=False)
    print(f"[Cache] Forecast saved to {cache_path}")
    return daily


def step_fetch_prices(zone: str) -> pd.DataFrame:
    """Fetch or load cached ERCOT price history."""
    print(f"\n{'='*60}")
    print(f"STEP 2: Building ERCOT price history (zone={zone})")
    print("="*60)
    cache_path = CACHE_DIR / f"prices_{zone}.csv"

    if cache_path.exists():
        age_days = (datetime.utcnow() - datetime.fromtimestamp(cache_path.stat().st_mtime)).days
        if age_days < 1:
            print(f"[Cache] Loading prices from cache ({age_days}d old)...")
            return pd.read_csv(cache_path, parse_dates=["date"])

    prices = build_daily_price_history(zone=zone)
    prices.to_csv(cache_path, index=False)
    print(f"[Cache] Prices saved to {cache_path}")
    return prices


def step_train_model(prices: pd.DataFrame) -> WeatherPriceModel:
    """
    Train calibration model using historical ERCOT prices.
    Since we don't have historical WeatherNext data yet, we reconstruct
    weather features from the price data itself using temperature proxies
    from NOAA's free public API (no auth needed).
    """
    print(f"\n{'='*60}")
    print("STEP 3: Calibrating tightness-to-price model")
    print("="*60)

    # Build a synthetic weather feature matrix from price history
    # using seasonal patterns as proxies until historical WeatherNext is available
    import numpy as np
    weather_proxy = _build_weather_proxy(prices)
    df = build_feature_matrix(weather_proxy, prices)
    print(f"[Model] Training on {len(df)} days of data...")
    model = WeatherPriceModel()
    metrics = model.fit(df)
    model.save()
    metrics_path = OUTPUT_DIR / "model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Model] Metrics saved to {metrics_path}")
    return model


def _build_weather_proxy(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Build synthetic weather features from price history dates.
    Uses seasonal temperature curves as proxies for real weather.
    This is a bootstrap approach — replaced by real WeatherNext
    reanalysis data once historical access is approved.
    """
    import numpy as np
    df = prices[["date"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    doy = df["date"].dt.dayofyear

    # ERCOT Texas: hot summers, mild winters
    # Approximate daily mean temp using sinusoidal seasonal curve
    df["temp_f_mean"] = 65 + 18 * np.sin((doy - 80) * 2 * np.pi / 365)
    df["temp_f_max"]  = df["temp_f_mean"] + 8 + 3 * np.random.randn(len(df))
    df["temp_f_min"]  = df["temp_f_mean"] - 8

    BALANCE = 65
    df["cdd"] = (df["temp_f_mean"] - BALANCE).clip(lower=0)
    df["hdd"] = (BALANCE - df["temp_f_mean"]).clip(lower=0)

    # Wind slightly higher in spring/fall
    df["wind_mph_mean"] = 12 + 4 * np.cos((doy - 100) * 2 * np.pi / 365) + np.random.randn(len(df)) * 2
    df["cloud_pct"]     = 40 + 15 * np.random.randn(len(df))

    # Demand and renewable proxies
    df["demand_proxy"]        = 40000 + df["cdd"] * 500 + df["hdd"] * 400
    df["solar_gen_proxy"]     = (1 - df["cloud_pct"] / 100).clip(0) * 0.18 * 20000
    df["wind_gen_proxy"]      = df["wind_mph_mean"] / 30 * 35000
    df["renewable_gen_proxy"] = df["wind_gen_proxy"] + df["solar_gen_proxy"]
    df["tightness"]           = df["demand_proxy"] - df["renewable_gen_proxy"]

    df["cloud_pct"] = df["cloud_pct"].clip(0, 100)
    return df


def step_load_model() -> WeatherPriceModel:
    """Load pre-trained model from disk."""
    print(f"\n{'='*60}")
    print("STEP 3: Loading cached model")
    print("="*60)
    return WeatherPriceModel.load()


def step_generate_forecast(model: WeatherPriceModel, forecast_weather: pd.DataFrame) -> pd.DataFrame:
    """Run model inference on the 15-day weather forecast."""
    print(f"\n{'='*60}")
    print("STEP 4: Generating price forecast")
    print("="*60)
    forecast = model.predict(forecast_weather)
    forecast_path = OUTPUT_DIR / f"forecast_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv"
    forecast.to_csv(forecast_path, index=False)
    print(f"\n15-Day Price Forecast:\n")
    print(forecast[["date","temp_f_max","tightness","predicted_price","spike_probability","day_signal"]].to_string(index=False))
    print(f"\n[Output] Forecast saved to {forecast_path}")
    return forecast


def step_generate_signal(forecast: pd.DataFrame, prices: pd.DataFrame, zone: str) -> dict:
    """Call Claude to generate trading signal."""
    print(f"\n{'='*60}")
    print("STEP 5: Generating AI trading signal")
    print("="*60)
    base_price    = float(prices["price_mean"].iloc[-30:].mean())
    rolling_mean  = float(prices["rolling_mean_30d"].dropna().iloc[-1])
    signal = generate_trading_signal(forecast, zone=zone, base_price=base_price, rolling_mean=rolling_mean)
    report = format_signal_report(signal, forecast, zone)
    print(report)
    signal_path = OUTPUT_DIR / f"signal_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.json"
    with open(signal_path, "w") as f:
        json.dump(signal, f, indent=2)
    print(f"[Output] Signal saved to {signal_path}")
    return signal


def step_generate_dashboard(forecast: pd.DataFrame, zone: str) -> None:
    """Generate self-contained HTML dashboard and open in browser."""
    import json, webbrowser

    print(f"\n{'='*60}")
    print("STEP 6: Generating dashboard")
    print("="*60)

    # Serialize forecast to JSON
    rows = []
    for _, r in forecast.iterrows():
        rows.append({
            "date":       str(r["date"])[:10] if hasattr(r["date"], "__str__") else r["date"],
            "temp":       round(float(r.get("temp_f_mean", 65)), 1),
            "tempMax":    round(float(r.get("temp_f_max", 70)), 1),
            "wind":       round(float(r.get("wind_mph_mean", 14)), 1),
            "cloud":      round(float(r.get("cloud_pct", 40)), 1),
            "demand":     int(r.get("demand_proxy", 40000)),
            "renewables": int(r.get("renewable_gen_proxy", 20000)),
            "tightness":  int(r.get("tightness", 20000)),
            "price":      round(float(r.get("predicted_price", 0)), 2),
            "spike":      round(float(r.get("spike_probability", 40)), 1),
            "signal":     str(r.get("day_signal", "MODERATE")),
        })

    data_json = json.dumps(rows)
    run_time  = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WeatherAlpha · {zone}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #07090f; --panel: #0c1018; --border: #151d28;
    --accent: #00d4ff; --elevated: #ff4444; --warn: #ffaa00;
    --text: #667788; --bright: #dde8f0; --green: #00c896;
    --grid: #0f1520;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: var(--bg); color: var(--bright); font-family: 'IBM Plex Mono', monospace; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }}

  /* TOP BAR */
  #topbar {{ background: var(--panel); border-bottom: 1px solid var(--border); height: 44px; display: flex; align-items: center; justify-content: space-between; padding: 0 20px; flex-shrink: 0; }}
  #topbar .logo {{ display: flex; align-items: center; gap: 10px; }}
  #topbar .dot {{ width: 7px; height: 7px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 10px var(--accent); }}
  #topbar .brand {{ color: var(--accent); font-weight: 700; font-size: 13px; letter-spacing: 0.18em; }}
  #topbar .sub {{ color: var(--text); font-size: 10px; margin-left: 14px; }}
  #topbar .meta {{ display: flex; gap: 20px; font-size: 10px; color: var(--text); }}
  .warn-text {{ color: var(--warn); }}

  /* LAYOUT */
  #main {{ display: flex; flex: 1; overflow: hidden; }}

  /* SIDEBAR */
  #sidebar {{ width: 175px; background: var(--panel); border-right: 1px solid var(--border); overflow-y: auto; flex-shrink: 0; }}
  #sidebar .sidebar-header {{ padding: 10px 14px 6px; font-size: 9px; letter-spacing: 0.1em; color: var(--text); border-bottom: 1px solid var(--border); }}
  .day-item {{ padding: 9px 14px; cursor: pointer; border-left: 2px solid transparent; border-bottom: 1px solid var(--grid); transition: all 0.1s; }}
  .day-item:hover {{ background: rgba(0,212,255,0.03); }}
  .day-item.active {{ background: rgba(0,212,255,0.05); border-left-color: var(--accent); }}
  .day-item .row1 {{ display: flex; justify-content: space-between; align-items: center; }}
  .day-item .dname {{ font-size: 11px; color: var(--text); }}
  .day-item.active .dname {{ color: var(--bright); font-weight: 600; }}
  .day-item .dsig {{ font-size: 10px; }}
  .day-item .row2 {{ display: flex; justify-content: space-between; margin-top: 2px; font-size: 10px; color: var(--text); }}

  /* CONTENT */
  #content {{ flex: 1; display: flex; flex-direction: column; overflow: hidden; }}

  /* KPI ROW */
  #kpis {{ display: flex; gap: 1px; padding: 1px; flex-shrink: 0; border-bottom: 1px solid var(--border); }}
  .kpi {{ background: var(--panel); border: 1px solid var(--border); padding: 11px 16px; flex: 1; }}
  .kpi .klabel {{ font-size: 9px; letter-spacing: 0.12em; color: var(--text); margin-bottom: 5px; }}
  .kpi .kvalue {{ font-size: 19px; font-weight: 700; }}
  .kpi .ksub {{ font-size: 9px; color: var(--text); margin-top: 3px; }}

  /* STATS ROW */
  #stats {{ display: flex; gap: 1px; padding: 1px 1px 0; flex-shrink: 0; border-bottom: 1px solid var(--border); }}
  .stat {{ background: var(--panel); padding: 8px 14px; flex: 1; }}
  .stat .slabel {{ font-size: 8px; letter-spacing: 0.1em; color: var(--text); margin-bottom: 4px; }}
  .stat .svalue {{ font-size: 13px; font-weight: 600; color: var(--bright); }}

  /* TABS */
  #tabs {{ display: flex; border-bottom: 1px solid var(--border); flex-shrink: 0; }}
  .tab {{ padding: 10px 18px; font-size: 10px; letter-spacing: 0.08em; cursor: pointer; color: var(--text); border-bottom: 2px solid transparent; transition: all 0.15s; font-family: 'IBM Plex Mono', monospace; background: none; border-top: none; border-left: none; border-right: none; }}
  .tab:hover {{ color: var(--accent); }}
  .tab.active {{ color: var(--accent); border-bottom-color: var(--accent); background: rgba(0,212,255,0.04); }}

  /* CHART AREA */
  #chart-area {{ flex: 1; padding: 16px 20px; overflow: hidden; display: flex; flex-direction: column; }}
  #chart-area .chart-title {{ font-size: 10px; color: var(--text); margin-bottom: 12px; letter-spacing: 0.08em; }}
  .chart-wrap {{ flex: 1; position: relative; min-height: 0; }}
  .chart-wrap canvas {{ max-height: 100% !important; }}

  /* RIGHT PANEL */
  #signal-panel {{ width: 215px; background: var(--panel); border-left: 1px solid var(--border); display: flex; flex-direction: column; flex-shrink: 0; }}
  #signal-panel .sp-header {{ padding: 10px 14px; border-bottom: 1px solid var(--border); font-size: 9px; letter-spacing: 0.1em; color: var(--text); }}
  #signal-badge {{ padding: 14px; border-bottom: 1px solid var(--border); }}
  #signal-badge .badge-inner {{ border-radius: 3px; padding: 12px 14px; text-align: center; }}
  #signal-badge .badge-val {{ font-size: 22px; font-weight: 700; letter-spacing: 0.08em; }}
  #signal-badge .badge-label {{ font-size: 9px; color: var(--text); margin-top: 4px; letter-spacing: 0.08em; }}
  #signal-metrics {{ padding: 12px 14px; border-bottom: 1px solid var(--border); display: flex; flex-direction: column; gap: 10px; }}
  .metric-row {{ display: flex; justify-content: space-between; align-items: center; }}
  .metric-row .ml {{ font-size: 9px; color: var(--text); letter-spacing: 0.08em; }}
  .metric-row .mv {{ font-size: 12px; font-weight: 600; color: var(--bright); }}
  #signal-thesis {{ padding: 12px 14px; flex: 1; overflow-y: auto; }}
  #signal-thesis .thesis-label {{ font-size: 9px; letter-spacing: 0.1em; color: var(--text); margin-bottom: 8px; }}
  #signal-thesis .thesis-body {{ font-size: 10px; color: var(--text); line-height: 1.8; }}
  #elevated-list {{ padding: 12px 14px; border-top: 1px solid var(--border); }}
  #elevated-list .el-header {{ font-size: 9px; letter-spacing: 0.1em; color: var(--text); margin-bottom: 8px; }}
  .el-row {{ display: flex; justify-content: space-between; padding: 5px 0; cursor: pointer; border-bottom: 1px solid var(--grid); }}
  .el-row .el-date {{ font-size: 10px; color: var(--elevated); }}
  .el-row .el-spike {{ font-size: 10px; color: var(--text); }}
</style>
</head>
<body>

<!-- TOP BAR -->
<div id="topbar">
  <div class="logo">
    <div class="dot"></div>
    <span class="brand">WEATHERALPHA</span>
    <span style="color:#151d28;font-size:18px;margin:0 4px">│</span>
    <span class="sub">ERCOT · {zone} · 15-DAY PRICE INTELLIGENCE</span>
  </div>
  <div class="meta">
    <span class="warn-text">⚠ SYNTHETIC PROXY · WEATHERNEXT PENDING</span>
    <span>RUN: <span style="color:var(--bright)">{run_time}</span></span>
  </div>
</div>

<div id="main">
  <!-- SIDEBAR -->
  <div id="sidebar">
    <div class="sidebar-header">FORECAST DAYS</div>
    <div id="day-list"></div>
  </div>

  <!-- CONTENT -->
  <div id="content">
    <div id="kpis"></div>
    <div id="stats"></div>
    <div id="tabs">
      <button class="tab active" onclick="switchTab('price', this)">PRICE FORECAST</button>
      <button class="tab" onclick="switchTab('spike', this)">SPIKE PROBABILITY</button>
      <button class="tab" onclick="switchTab('supply', this)">SUPPLY / DEMAND</button>
      <button class="tab" onclick="switchTab('weather', this)">WEATHER SIGNALS</button>
    </div>
    <div id="chart-area">
      <div class="chart-title" id="chart-title"></div>
      <div class="chart-wrap">
        <canvas id="main-chart"></canvas>
      </div>
    </div>
  </div>

  <!-- SIGNAL PANEL -->
  <div id="signal-panel">
    <div class="sp-header" id="sp-header">TRADING SIGNAL</div>
    <div id="signal-badge">
      <div class="badge-inner" id="badge-inner">
        <div class="badge-val" id="badge-val"></div>
        <div class="badge-label">RISK LEVEL</div>
      </div>
    </div>
    <div id="signal-metrics"></div>
    <div id="signal-thesis">
      <div class="thesis-label">ALPHA THESIS</div>
      <div class="thesis-body" id="thesis-body"></div>
    </div>
    <div id="elevated-list">
      <div class="el-header">ELEVATED DAYS</div>
      <div id="el-rows"></div>
    </div>
  </div>
</div>

<script>
const DATA = {data_json};
let activeDay = 0;
let activeTab = 'price';
let chart = null;

const C = {{
  accent: '#00d4ff', elevated: '#ff4444', warn: '#ffaa00',
  text: '#667788', bright: '#dde8f0', green: '#00c896',
  grid: '#0f1520', panel: '#0c1018',
}};

function sc(signal) {{ return signal === 'ELEVATED' ? C.elevated : C.accent; }}

function fmtDate(iso) {{
  const d = new Date(iso);
  return d.toLocaleDateString('en-US', {{ month: 'short', day: 'numeric' }});
}}

function buildSidebar() {{
  const el = document.getElementById('day-list');
  el.innerHTML = DATA.map((d, i) => `
    <div class="day-item ${{i === 0 ? 'active' : ''}}" onclick="selectDay(${{i}})" id="day-${{i}}">
      <div class="row1">
        <span class="dname">${{fmtDate(d.date)}}</span>
        <span class="dsig" style="color:${{sc(d.signal)}}">${{d.signal === 'ELEVATED' ? '● ELEV' : '·'}}</span>
      </div>
      <div class="row2">
        <span>${{d.temp}}°F</span>
        <span style="color:${{d.spike > 50 ? C.elevated : C.text}}">${{d.spike.toFixed(0)}}%</span>
      </div>
    </div>
  `).join('');
}}

function buildKPIs() {{
  const peakPrice = Math.max(...DATA.map(d => d.price));
  const peakDay   = DATA.find(d => d.price === peakPrice);
  const avgSpike  = (DATA.reduce((s, d) => s + d.spike, 0) / DATA.length).toFixed(1);
  const elevated  = DATA.filter(d => d.signal === 'ELEVATED');
  const d         = DATA[activeDay];

  document.getElementById('kpis').innerHTML = `
    <div class="kpi"><div class="klabel">PEAK FORECAST PRICE</div><div class="kvalue" style="color:${{C.elevated}}">$${{peakPrice.toFixed(0)}}/MWh</div><div class="ksub">${{fmtDate(peakDay.date)}} · highest in window</div></div>
    <div class="kpi"><div class="klabel">AVG SPIKE PROBABILITY</div><div class="kvalue" style="color:${{C.warn}}">${{avgSpike}}%</div><div class="ksub">across all 15 forecast days</div></div>
    <div class="kpi"><div class="klabel">ELEVATED RISK DAYS</div><div class="kvalue" style="color:${{C.elevated}}">${{elevated.length}}</div><div class="ksub">${{elevated.map(x => fmtDate(x.date)).join(' · ')}}</div></div>
    <div class="kpi"><div class="klabel">SELECTED · SIGNAL</div><div class="kvalue" style="color:${{sc(d.signal)}}">${{d.signal}}</div><div class="ksub">${{d.spike.toFixed(1)}}% spike · $${{d.price.toFixed(0)}}/MWh pred</div></div>
  `;
}}

function buildStats() {{
  const d = DATA[activeDay];
  const stats = [
    {{ label: 'PRED PRICE', value: `$${{d.price.toFixed(2)}}/MWh`, color: d.price > 100 ? C.elevated : C.green }},
    {{ label: 'DEMAND', value: `${{(d.demand/1000).toFixed(1)}}K MW` }},
    {{ label: 'RENEWABLES', value: `${{(d.renewables/1000).toFixed(1)}}K MW`, color: C.green }},
    {{ label: 'GRID TIGHTNESS', value: `${{(d.tightness/1000).toFixed(1)}}K MW` }},
    {{ label: 'WIND SPEED', value: `${{d.wind}} mph` }},
    {{ label: 'CLOUD COVER', value: `${{d.cloud}}%` }},
    {{ label: 'TEMP (MEAN/MAX)', value: `${{d.temp}}° / ${{d.tempMax}}°F` }},
  ];
  document.getElementById('stats').innerHTML = stats.map(s => `
    <div class="stat">
      <div class="slabel">${{s.label}}</div>
      <div class="svalue" style="color:${{s.color || C.bright}}">${{s.value}}</div>
    </div>
  `).join('');
}}

function buildSignal() {{
  const d = DATA[activeDay];
  const elevated = DATA.filter(x => x.signal === 'ELEVATED');
  const isElev   = d.signal === 'ELEVATED';
  const color     = sc(d.signal);

  document.getElementById('sp-header').textContent = `TRADING SIGNAL · ${{fmtDate(d.date).toUpperCase()}}`;

  const badge = document.getElementById('badge-inner');
  badge.style.background = isElev ? 'rgba(255,68,68,0.08)' : 'rgba(0,212,255,0.06)';
  badge.style.border      = `1px solid ${{color}}33`;
  document.getElementById('badge-val').textContent  = d.signal;
  document.getElementById('badge-val').style.color  = color;

  const metrics = [
    {{ label: 'SPIKE PROB', value: `${{d.spike.toFixed(1)}}%`, color: d.spike > 50 ? C.elevated : C.warn }},
    {{ label: 'PRED PRICE', value: `$${{d.price.toFixed(2)}}`, color: d.price > 100 ? C.elevated : C.green }},
    {{ label: 'TIGHTNESS',  value: `${{(d.tightness/1000).toFixed(1)}}K MW` }},
    {{ label: 'RENEWABLES', value: `${{(d.renewables/1000).toFixed(1)}}K MW`, color: C.green }},
    {{ label: 'WIND',       value: `${{d.wind}} mph` }},
    {{ label: 'CLOUD',      value: `${{d.cloud}}%` }},
  ];
  document.getElementById('signal-metrics').innerHTML = metrics.map(m => `
    <div class="metric-row">
      <span class="ml">${{m.label}}</span>
      <span class="mv" style="color:${{m.color || C.bright}}">${{m.value}}</span>
    </div>
  `).join('');

  const thesis = isElev
    ? `<span style="color:${{C.elevated}}">High cloud cover (${{d.cloud}}%)</span> suppressing solar output. Wind at ${{d.wind}} mph. Tightness at ${{(d.tightness/1000).toFixed(1)}}K MW. <span style="color:${{C.warn}}">Spike probability exceeds 50% threshold.</span> Watch for RTM/DAM divergence.`
    : `Wind at ${{d.wind}} mph delivering ${{(d.renewables/1000).toFixed(1)}}K MW renewables. Grid balanced at ${{(d.tightness/1000).toFixed(1)}}K MW tightness. <span style="color:${{C.green}}">DAM prices likely subdued</span> barring demand shock.`;
  document.getElementById('thesis-body').innerHTML = thesis;

  document.getElementById('el-rows').innerHTML = elevated.map((f, i) => `
    <div class="el-row" onclick="selectDay(${{DATA.findIndex(x => x.date === f.date)}})">
      <span class="el-date">${{fmtDate(f.date)}}</span>
      <span class="el-spike">${{f.spike.toFixed(0)}}% spike</span>
    </div>
  `).join('');
}}

function selectDay(i) {{
  document.querySelectorAll('.day-item').forEach(el => el.classList.remove('active'));
  document.getElementById(`day-${{i}}`).classList.add('active');
  activeDay = i;
  buildKPIs();
  buildStats();
  buildSignal();
  renderChart();
}}

function switchTab(tab, btn) {{
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
  activeTab = tab;
  renderChart();
}}

function renderChart() {{
  const titles = {{
    price:   'DAM SETTLEMENT POINT PRICE FORECAST  ·  $/MWh  ·  {zone}',
    spike:   'SPIKE PROBABILITY  ·  %  ·  LOGISTIC REGRESSION ON GRID TIGHTNESS',
    supply:  'DEMAND vs RENEWABLES vs GRID TIGHTNESS  ·  MW',
    weather: 'TEMPERATURE + WIND + CLOUD COVER  ·  15-DAY WEATHER SIGNALS',
  }};
  document.getElementById('chart-title').textContent = titles[activeTab];

  if (chart) {{ chart.destroy(); chart = null; }}
  const ctx = document.getElementById('main-chart').getContext('2d');
  const labels = DATA.map(d => fmtDate(d.date));

  const gridOpts = {{
    color: C.grid, drawBorder: false,
  }};
  const tickOpts = {{
    color: C.text, font: {{ family: 'IBM Plex Mono', size: 9 }},
  }};

  if (activeTab === 'price') {{
    chart = new Chart(ctx, {{
      type: 'line',
      data: {{
        labels,
        datasets: [{{
          label: 'Predicted Price',
          data: DATA.map(d => d.price),
          borderColor: DATA.map((d, i) => i === activeDay ? C.bright : C.accent),
          borderWidth: 2,
          backgroundColor: 'rgba(0,212,255,0.08)',
          fill: true,
          tension: 0.3,
          pointRadius: DATA.map((d, i) => i === activeDay ? 7 : d.signal === 'ELEVATED' ? 6 : 4),
          pointBackgroundColor: DATA.map(d => d.signal === 'ELEVATED' ? C.elevated : C.accent),
          pointBorderColor: C.panel,
          pointBorderWidth: 1.5,
        }}],
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        onClick: (e, els) => {{ if (els.length) selectDay(els[0].index); }},
        plugins: {{
          legend: {{ display: false }},
          tooltip: {{ callbacks: {{ label: ctx => `$${{ctx.parsed.y.toFixed(2)}}/MWh` }} }},
          annotation: {{}} ,
        }},
        scales: {{
          x: {{ grid: gridOpts, ticks: tickOpts, border: {{ color: C.grid }} }},
          y: {{ grid: gridOpts, ticks: {{ ...tickOpts, callback: v => `$${{v}}` }}, border: {{ display: false }} }},
        }},
      }},
    }});
  }}

  else if (activeTab === 'spike') {{
    chart = new Chart(ctx, {{
      type: 'bar',
      data: {{
        labels,
        datasets: [{{
          label: 'Spike Probability',
          data: DATA.map(d => d.spike),
          backgroundColor: DATA.map((d, i) => {{
            const base = d.spike > 50 ? C.elevated : d.spike > 45 ? C.warn : C.accent;
            return i === activeDay ? base : base + '99';
          }}),
          borderRadius: 2,
        }}],
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        onClick: (e, els) => {{ if (els.length) selectDay(els[0].index); }},
        plugins: {{
          legend: {{ display: false }},
          tooltip: {{ callbacks: {{ label: ctx => `${{ctx.parsed.y.toFixed(1)}}%` }} }},
        }},
        scales: {{
          x: {{ grid: gridOpts, ticks: tickOpts, border: {{ color: C.grid }} }},
          y: {{ min: 0, max: 100, grid: gridOpts, ticks: {{ ...tickOpts, callback: v => `${{v}}%` }}, border: {{ display: false }} }},
        }},
      }},
    }});
  }}

  else if (activeTab === 'supply') {{
    chart = new Chart(ctx, {{
      type: 'line',
      data: {{
        labels,
        datasets: [
          {{ label: 'Demand', data: DATA.map(d => d.demand), borderColor: C.warn, borderWidth: 1.5, backgroundColor: 'rgba(255,170,0,0.07)', fill: true, tension: 0.3, pointRadius: 0 }},
          {{ label: 'Renewables', data: DATA.map(d => d.renewables), borderColor: C.green, borderWidth: 1.5, backgroundColor: 'rgba(0,200,150,0.07)', fill: true, tension: 0.3, pointRadius: 0 }},
          {{ label: 'Tightness', data: DATA.map(d => d.tightness), borderColor: C.elevated, borderWidth: 2, borderDash: [4, 2], backgroundColor: 'transparent', tension: 0.3, pointRadius: 0 }},
        ],
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{
          legend: {{ labels: {{ color: C.text, font: {{ family: 'IBM Plex Mono', size: 9 }}, boxWidth: 12 }} }},
          tooltip: {{ callbacks: {{ label: ctx => `${{(ctx.parsed.y/1000).toFixed(1)}}K MW` }} }},
        }},
        scales: {{
          x: {{ grid: gridOpts, ticks: tickOpts, border: {{ color: C.grid }} }},
          y: {{ grid: gridOpts, ticks: {{ ...tickOpts, callback: v => `${{(v/1000).toFixed(0)}}K` }}, border: {{ display: false }} }},
        }},
      }},
    }});
  }}

  else if (activeTab === 'weather') {{
    chart = new Chart(ctx, {{
      type: 'line',
      data: {{
        labels,
        datasets: [
          {{ label: 'Temp Max °F', data: DATA.map(d => d.tempMax), borderColor: C.elevated, borderWidth: 1, backgroundColor: 'rgba(255,68,68,0.06)', fill: true, tension: 0.3, pointRadius: 0, yAxisID: 'temp' }},
          {{ label: 'Temp Mean °F', data: DATA.map(d => d.temp), borderColor: C.warn, borderWidth: 2, backgroundColor: 'transparent', tension: 0.3, pointRadius: 0, yAxisID: 'temp' }},
          {{ label: 'Wind mph', data: DATA.map(d => d.wind), borderColor: C.accent, borderWidth: 2, backgroundColor: 'transparent', tension: 0.3, pointRadius: 0, yAxisID: 'wind' }},
          {{ label: 'Cloud %', data: DATA.map(d => d.cloud), borderColor: C.text, borderWidth: 1, backgroundColor: 'rgba(102,119,136,0.15)', fill: true, tension: 0.3, pointRadius: 0, yAxisID: 'wind' }},
        ],
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{
          legend: {{ labels: {{ color: C.text, font: {{ family: 'IBM Plex Mono', size: 9 }}, boxWidth: 12 }} }},
        }},
        scales: {{
          x: {{ grid: gridOpts, ticks: tickOpts, border: {{ color: C.grid }} }},
          temp: {{ type: 'linear', position: 'left', grid: gridOpts, ticks: {{ ...tickOpts, callback: v => `${{v}}°` }}, border: {{ display: false }} }},
          wind: {{ type: 'linear', position: 'right', grid: {{ display: false }}, ticks: {{ ...tickOpts, callback: v => `${{v}}` }}, border: {{ display: false }} }},
        }},
      }},
    }});
  }}
}}

// Init
buildSidebar();
buildKPIs();
buildStats();
buildSignal();
renderChart();
</script>
</body>
</html>"""

    dashboard_path = OUTPUT_DIR / "dashboard.html"
    with open(dashboard_path, "w") as f:
        f.write(html)
    print(f"[Dashboard] Saved to {dashboard_path}")
    print(f"[Dashboard] Opening in browser...")
    webbrowser.open(f"file://{dashboard_path.absolute()}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="WeatherAlpha Energy Price Intelligence")
    parser.add_argument("--zone",      default=DEFAULT_ZONE, help=f"ERCOT hub (default: {DEFAULT_ZONE})")
    parser.add_argument("--train",     action="store_true",  help="Retrain model from scratch")
    parser.add_argument("--no-signal", action="store_true",  help="Skip AI signal generation")
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════╗
║           WEATHERALPHA  ·  Energy Price Intelligence     ║
║           Zone: {args.zone:<10}  Run: {datetime.utcnow().strftime("%Y-%m-%d %H:%M")} UTC     ║
╚══════════════════════════════════════════════════════════╝
""")

    # Step 1: Fetch live WeatherNext 2 forecast
    forecast_weather = step_fetch_forecast(zone=args.zone)

    # Step 2: Get historical prices for calibration
    prices = step_fetch_prices(zone=args.zone)

    # Step 3: Train or load model
    if args.train:
        model = step_train_model(prices)
    else:
        try:
            model = step_load_model()
        except FileNotFoundError:
            print("[Model] No cached model found — training from scratch (use --train to be explicit)")
            model = step_train_model(prices)

    # Step 4: Generate 15-day price forecast
    forecast = step_generate_forecast(model, forecast_weather)

    # Step 5: AI signal
    if not args.no_signal:
        step_generate_signal(forecast, prices, zone=args.zone)

    # Step 6: Generate dashboard
    step_generate_dashboard(forecast, zone=args.zone)

    print("\n✅ WeatherAlpha run complete. Outputs in ./outputs/\n")


if __name__ == "__main__":
    main()
