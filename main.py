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
    dashboard_path = OUTPUT_DIR / "dashboard.html"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WeatherAlpha · {zone}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #f8faf8;
    --panel: #ffffff;
    --border: #e4ebe4;
    --border2: #d0dbd0;
    --accent: #1a7a4a;
    --accent-light: #e8f5ee;
    --accent2: #2da65e;
    --elevated: #c0392b;
    --elevated-light: #fdf0ef;
    --warn: #d4841a;
    --warn-light: #fef6ec;
    --text: #6b7c6b;
    --text2: #3d4f3d;
    --bright: #1a2e1a;
    --grid: #eef3ee;
    --sidebar: #f2f7f2;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: var(--bg); color: var(--bright); font-family: 'Inter', sans-serif; height: 100vh; display: flex; flex-direction: column; overflow: hidden; font-size: 13px; }}

  /* TOP BAR */
  #topbar {{ background: var(--panel); border-bottom: 1px solid var(--border); height: 52px; display: flex; align-items: center; justify-content: space-between; padding: 0 24px; flex-shrink: 0; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }}
  #topbar .logo {{ display: flex; align-items: center; gap: 12px; }}
  #topbar .map-wrap {{ position: relative; display: flex; align-items: center; cursor: default; }}
  #topbar .map-wrap:hover .map-tooltip {{ opacity: 1; pointer-events: auto; }}
  .map-tooltip {{ position: absolute; top: 38px; left: 50%; transform: translateX(-50%); background: white; border: 1px solid var(--border2); border-radius: 8px; padding: 10px 14px; width: 220px; box-shadow: 0 4px 16px rgba(0,0,0,0.10); opacity: 0; pointer-events: none; transition: opacity 0.15s; z-index: 100; }}
  .map-tooltip::before {{ content: ''; position: absolute; top: -5px; left: 50%; transform: translateX(-50%); width: 8px; height: 8px; background: white; border-left: 1px solid var(--border2); border-top: 1px solid var(--border2); transform: translateX(-50%) rotate(45deg); }}
  .map-tooltip .mt-title {{ font-size: 11px; font-weight: 700; color: var(--bright); margin-bottom: 4px; }}
  .map-tooltip .mt-body {{ font-size: 10px; color: var(--text); line-height: 1.6; }}
  .map-tooltip .mt-stat {{ display: flex; justify-content: space-between; margin-top: 6px; padding-top: 6px; border-top: 1px solid var(--border); }}
  .map-tooltip .mt-stat span {{ font-size: 10px; color: var(--text); }}
  .map-tooltip .mt-stat strong {{ font-size: 10px; color: var(--bright); font-weight: 600; }}
  #topbar .brand-wrap {{ display: flex; align-items: center; gap: 8px; }}
  #topbar .brand-icon {{ width: 28px; height: 28px; background: var(--accent); border-radius: 6px; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 12px; letter-spacing: -0.5px; }}
  #topbar .brand {{ color: var(--bright); font-weight: 700; font-size: 15px; letter-spacing: -0.3px; }}
  #topbar .divider {{ width: 1px; height: 20px; background: var(--border2); margin: 0 4px; }}
  #topbar .sub {{ color: var(--text); font-size: 12px; }}
  #topbar .meta {{ display: flex; align-items: center; gap: 16px; font-size: 11px; color: var(--text); }}
  .badge {{ padding: 3px 8px; border-radius: 4px; font-size: 10px; font-weight: 600; letter-spacing: 0.03em; }}
  .badge-warn {{ background: var(--warn-light); color: var(--warn); }}
  .badge-ok {{ background: var(--accent-light); color: var(--accent); }}

  /* LAYOUT */
  #main {{ display: flex; flex: 1; overflow: hidden; }}

  /* SIDEBAR */
  #sidebar {{ width: 180px; background: var(--sidebar); border-right: 1px solid var(--border); overflow-y: auto; flex-shrink: 0; }}
  #sidebar .sidebar-header {{ padding: 12px 14px 8px; font-size: 10px; font-weight: 600; letter-spacing: 0.08em; color: var(--text); text-transform: uppercase; border-bottom: 1px solid var(--border); }}
  .day-item {{ padding: 10px 14px; cursor: pointer; border-left: 3px solid transparent; border-bottom: 1px solid var(--border); transition: all 0.12s; }}
  .day-item:hover {{ background: var(--accent-light); }}
  .day-item.active {{ background: var(--accent-light); border-left-color: var(--accent); }}
  .day-item .row1 {{ display: flex; justify-content: space-between; align-items: center; }}
  .day-item .dname {{ font-size: 12px; font-weight: 500; color: var(--text2); }}
  .day-item.active .dname {{ color: var(--accent); font-weight: 600; }}
  .day-item .dsig {{ font-size: 9px; font-weight: 600; letter-spacing: 0.05em; padding: 2px 5px; border-radius: 3px; }}
  .dsig-elevated {{ background: var(--elevated-light); color: var(--elevated); }}
  .dsig-moderate {{ color: var(--text); }}
  .day-item .row2 {{ display: flex; justify-content: space-between; margin-top: 3px; font-size: 11px; color: var(--text); }}
  .day-item .spike-bar-bg {{ height: 3px; background: var(--border2); border-radius: 2px; margin-top: 5px; overflow: hidden; }}
  .day-item .spike-bar-fg {{ height: 100%; border-radius: 2px; transition: width 0.2s; }}

  /* CONTENT */
  #content {{ flex: 1; display: flex; flex-direction: column; overflow: hidden; background: var(--bg); }}

  /* SPARKLINE STRIP */
  #sparkline-strip {{ flex-shrink: 0; background: var(--panel); border-bottom: 1px solid var(--border); padding: 12px 20px 10px; }}
  #sparkline-strip .strip-label {{ font-size: 10px; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase; color: var(--text); margin-bottom: 8px; display: flex; justify-content: space-between; }}
  #sparkline-wrap {{ position: relative; height: 90px; width: 100%; }}
  #sparkline-wrap canvas {{ position: absolute; top: 0; left: 0; cursor: pointer; }}

  /* DAY DETAIL */
  #day-detail {{ flex: 1; overflow-y: auto; padding: 16px 20px; display: flex; flex-direction: column; gap: 14px; }}

  /* Detail header */
  .detail-header {{ display: flex; align-items: center; justify-content: space-between; }}
  .detail-date {{ font-size: 22px; font-weight: 700; letter-spacing: -0.5px; color: var(--bright); }}
  .detail-signal {{ padding: 6px 14px; border-radius: 6px; font-size: 12px; font-weight: 700; letter-spacing: 0.05em; }}

  /* Big price card */
  .price-card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 18px 20px; display: flex; gap: 0; }}
  .pc-main {{ flex: 1; }}
  .pc-label {{ font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; color: var(--text); margin-bottom: 6px; }}
  .pc-price {{ font-size: 36px; font-weight: 700; letter-spacing: -1px; }}
  .pc-sub {{ font-size: 11px; color: var(--text); margin-top: 4px; }}
  .pc-divider {{ width: 1px; background: var(--border); margin: 0 20px; }}
  .pc-spike {{ text-align: center; display: flex; flex-direction: column; align-items: center; justify-content: center; min-width: 80px; }}
  .pc-spike-val {{ font-size: 28px; font-weight: 700; }}
  .pc-spike-label {{ font-size: 10px; color: var(--text); margin-top: 4px; font-weight: 500; }}

  /* Spike bar */
  .spike-bar-wrap {{ background: var(--border); border-radius: 4px; height: 6px; margin-top: 10px; overflow: hidden; width: 100%; }}
  .spike-bar-fill {{ height: 100%; border-radius: 4px; transition: width 0.4s; }}

  /* Driver grid */
  .driver-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
  .driver-card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; }}
  .driver-card.risk {{ border-color: rgba(192,57,43,0.3); background: rgba(192,57,43,0.03); }}
  .driver-card.ok {{ border-color: rgba(26,122,74,0.2); background: rgba(26,122,74,0.02); }}
  .dc-top {{ display: flex; align-items: center; gap: 7px; margin-bottom: 5px; }}
  .dc-icon {{ font-size: 15px; }}
  .dc-label {{ font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; }}
  .dc-value {{ font-size: 18px; font-weight: 700; font-family: 'DM Mono', monospace; }}
  .dc-sub {{ font-size: 10px; margin-top: 3px; }}

  /* Action box */
  .action-box {{ background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px 18px; }}
  .action-box.risk {{ border-color: rgba(192,57,43,0.3); background: rgba(192,57,43,0.03); }}
  .action-box.ok {{ border-color: rgba(26,122,74,0.2); background: rgba(26,122,74,0.02); }}
  .ab-title {{ font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }}
  .ab-body {{ font-size: 12px; line-height: 1.7; color: var(--text2); }}
  .ab-watches {{ margin-top: 10px; display: flex; flex-direction: column; gap: 4px; }}
  .ab-watch {{ font-size: 11px; color: var(--text2); padding-left: 14px; position: relative; }}
  .ab-watch::before {{ content: "→"; position: absolute; left: 0; font-weight: 700; }}

  /* RIGHT PANEL */
  #signal-panel {{ width: 260px; background: var(--panel); border-left: 1px solid var(--border); display: flex; flex-direction: column; flex-shrink: 0; overflow-y: auto; }}
  #signal-panel .sp-header {{ padding: 12px 16px; border-bottom: 1px solid var(--border); font-size: 10px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text); }}
  #signal-badge {{ padding: 16px; border-bottom: 1px solid var(--border); }}
  #signal-badge .badge-inner {{ border-radius: 8px; padding: 14px; text-align: center; }}
  #signal-badge .badge-val {{ font-size: 18px; font-weight: 700; letter-spacing: 0.05em; }}
  #signal-badge .badge-label {{ font-size: 10px; color: var(--text); margin-top: 4px; font-weight: 500; }}
  #signal-metrics {{ padding: 12px 16px; border-bottom: 1px solid var(--border); display: flex; flex-direction: column; gap: 10px; }}
  .metric-row {{ display: flex; justify-content: space-between; align-items: center; }}
  .metric-row .ml {{ font-size: 10px; font-weight: 500; color: var(--text); }}
  .metric-row .mv {{ font-size: 12px; font-weight: 600; font-family: 'DM Mono', monospace; color: var(--bright); }}
  #signal-thesis {{ padding: 14px 16px; flex: 1; overflow-y: auto; }}
  #signal-thesis .thesis-label {{ font-size: 10px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text); margin-bottom: 8px; }}
  #signal-thesis .thesis-body {{ font-size: 11px; color: var(--text2); line-height: 1.7; }}
  #elevated-list {{ padding: 12px 16px; border-top: 1px solid var(--border); }}
  #elevated-list .el-header {{ font-size: 10px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text); margin-bottom: 8px; }}
  .el-row {{ display: flex; justify-content: space-between; padding: 6px 0; cursor: pointer; border-bottom: 1px solid var(--border); }}
  .el-row:hover .el-date {{ color: var(--elevated); text-decoration: underline; }}
  .el-row .el-date {{ font-size: 11px; font-weight: 500; color: var(--elevated); }}
  .el-row .el-spike {{ font-size: 11px; color: var(--text); font-family: 'DM Mono', monospace; }}

  /* SIGNAL PANEL extras */
  .sp-section-header {{ padding: 10px 16px 6px; font-size: 10px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text); border-top: 1px solid var(--border); }}
  #summary-box {{ padding: 14px 16px; border-bottom: 1px solid var(--border); }}
  #summary-box .sum-headline {{ font-size: 13px; font-weight: 600; color: var(--bright); line-height: 1.4; margin-bottom: 8px; }}
  #summary-box .sum-body {{ font-size: 11px; color: var(--text); line-height: 1.7; }}
  #next-alert-box {{ padding: 12px 16px; border-bottom: 1px solid var(--border); }}
  #next-alert-box .nal-label {{ font-size: 9px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text); margin-bottom: 8px; }}
  .alert-row {{ display: flex; align-items: center; justify-content: space-between; padding: 7px 10px; border-radius: 6px; margin-bottom: 5px; cursor: pointer; transition: opacity 0.1s; }}
  .alert-row:hover {{ opacity: 0.8; }}
  .alert-row .ar-date {{ font-size: 12px; font-weight: 600; }}
  .alert-row .ar-spike {{ font-size: 11px; font-family: 'DM Mono', monospace; }}
  .alert-row .ar-days {{ font-size: 10px; padding: 2px 6px; border-radius: 3px; font-weight: 600; }}
  #signal-drivers {{ padding: 10px 16px; border-bottom: 1px solid var(--border); display: flex; flex-direction: column; gap: 8px; }}
  .driver-row {{ display: flex; align-items: center; gap: 8px; }}
  .driver-icon {{ width: 22px; height: 22px; border-radius: 5px; display: flex; align-items: center; justify-content: center; font-size: 12px; flex-shrink: 0; }}
  .driver-text {{ flex: 1; }}
  .driver-label {{ font-size: 10px; font-weight: 600; color: var(--text2); }}
  .driver-val {{ font-size: 11px; color: var(--text); }}
  #signal-watch {{ padding: 12px 16px; }}
  #signal-watch .watch-label {{ font-size: 9px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text); margin-bottom: 8px; }}
  #signal-watch .watch-item {{ font-size: 11px; color: var(--text2); line-height: 1.7; padding-left: 12px; position: relative; margin-bottom: 4px; }}
  #signal-watch .watch-item::before {{ content: "→"; position: absolute; left: 0; color: var(--warn); font-weight: 600; }}

  /* Scrollbar */
  ::-webkit-scrollbar {{ width: 4px; }}
  ::-webkit-scrollbar-track {{ background: transparent; }}
  ::-webkit-scrollbar-thumb {{ background: var(--border2); border-radius: 2px; }}
</style>
</head>
<body>

<!-- TOP BAR -->
<div id="topbar">
  <div class="logo">
    <div class="brand-wrap">
      <div class="brand-icon">WA</div>
      <span class="brand">WeatherAlpha</span>
    </div>
    <div class="divider"></div>
    <div class="map-wrap">
      <!-- Simplified Texas SVG outline with ERCOT shaded -->
      <svg width="44" height="36" viewBox="0 0 44 36" fill="none" xmlns="http://www.w3.org/2000/svg" style="display:block">
        <!-- Texas state outline (simplified) -->
        <path d="M4,2 L38,2 L40,4 L40,10 L38,10 L38,14 L36,16 L36,20 L32,24 L28,28 L24,32 L20,34 L16,32 L10,28 L6,22 L4,18 L2,14 L2,6 Z"
              fill="rgba(26,122,74,0.15)" stroke="#1a7a4a" stroke-width="1.2"/>
        <!-- ERCOT coverage label -->
        <text x="21" y="16" text-anchor="middle" font-size="5.5" font-family="Inter,sans-serif" font-weight="700" fill="#1a7a4a">ERCOT</text>
        <text x="21" y="22" text-anchor="middle" font-size="4" font-family="Inter,sans-serif" fill="#6b7c6b">~90% of Texas</text>
        <!-- Small star for Austin (ERCOT HQ) -->
        <circle cx="22" cy="26" r="1.5" fill="#1a7a4a"/>
      </svg>
      <div class="map-tooltip">
        <div class="mt-title">ERCOT Coverage Area</div>
        <div class="mt-body">The Electric Reliability Council of Texas manages the flow of electric power to about 90% of the state — roughly 26 million customers across most of Texas.</div>
        <div class="mt-stat">
          <span>Grid capacity</span><strong>~90 GW</strong>
        </div>
        <div class="mt-stat">
          <span>Customers</span><strong>~26M</strong>
        </div>
        <div class="mt-stat">
          <span>Monitoring zone</span><strong>{zone}</strong>
        </div>
      </div>
    </div>
    <div class="divider"></div>
    <span class="sub">Texas Grid (ERCOT) · {zone} · 15-Day Price Intelligence</span>
  </div>
  <div class="meta">
    <span class="badge badge-warn">⚠ Synthetic Proxy · WeatherNext Pending</span>
    <span>Updated: <strong>{run_time}</strong></span>
  </div>
</div>

<div id="main">
  <!-- SIDEBAR -->
  <div id="sidebar">
    <div class="sidebar-header">Forecast Days</div>
    <div id="day-list"></div>
  </div>

  <!-- CONTENT -->
  <div id="content">
    <div id="sparkline-strip">
      <div class="strip-label">
        <span>15-Day Price Forecast — click any day</span>
        <span id="strip-range" style="font-weight:400;color:var(--text)"></span>
      </div>
      <div id="sparkline-wrap"><canvas id="sparkline-chart"></canvas></div>
    </div>
    <div id="day-detail"></div>
  </div>

  <!-- SIGNAL PANEL -->
  <div id="signal-panel">
    <div class="sp-header">15-Day Outlook</div>

    <!-- Plain English Summary -->
    <div id="summary-box"></div>

    <!-- Next Alert -->
    <div id="next-alert-box"></div>

    <!-- Selected Day Detail -->
    <div class="sp-section-header" id="sp-day-header">Selected Day</div>
    <div id="signal-badge">
      <div class="badge-inner" id="badge-inner">
        <div class="badge-val" id="badge-val"></div>
        <div class="badge-label">Risk Level</div>
      </div>
    </div>
    <div id="signal-drivers"></div>
    <div id="signal-watch"></div>
  </div>
</div>

<script>
const DATA = {data_json};
let activeDay = 0;
let activeTab = 'price';
let chart = null;

const C = {{
  accent:  '#1a7a4a',
  accent2: '#2da65e',
  accentA: 'rgba(26,122,74,0.1)',
  elevated: '#c0392b',
  elevatedA: 'rgba(192,57,43,0.08)',
  warn:    '#d4841a',
  warnA:   'rgba(212,132,26,0.1)',
  text:    '#6b7c6b',
  bright:  '#1a2e1a',
  green2:  '#27ae60',
  grid:    '#eef3ee',
  panel:   '#ffffff',
}};

function sc(signal) {{ return signal === 'ELEVATED' ? C.elevated : C.accent; }}

function fmtDate(iso) {{
  const d = new Date(iso + 'T00:00:00');
  return d.toLocaleDateString('en-US', {{ month: 'short', day: 'numeric' }});
}}

function buildSidebar() {{
  const el = document.getElementById('day-list');
  el.innerHTML = DATA.map((d, i) => {{
    const barColor = d.spike > 50 ? C.elevated : d.spike > 42 ? C.warn : C.accent;
    return `
    <div class="day-item ${{i === 0 ? 'active' : ''}}" onclick="selectDay(${{i}})" id="day-${{i}}">
      <div class="row1">
        <span class="dname">${{fmtDate(d.date)}}</span>
        ${{d.signal === 'ELEVATED'
          ? `<span class="dsig dsig-elevated">ELEV</span>`
          : `<span class="dsig dsig-moderate"></span>`}}
      </div>
      <div class="row2">
        <span>${{d.temp}}°F</span>
        <span style="color:${{barColor}};font-weight:${{d.spike > 50 ? 600 : 400}}">${{d.spike.toFixed(0)}}%</span>
      </div>
      <div class="spike-bar-bg">
        <div class="spike-bar-fg" style="width:${{d.spike}}%;background:${{barColor}}"></div>
      </div>
    </div>`;
  }}).join('');
}}

function renderSparkline() {{
  const existing = Chart.getChart('sparkline-chart');
  if (existing) existing.destroy();

  const prices = DATA.map(d => d.price);
  const minP = Math.min(...prices), maxP = Math.max(...prices);
  document.getElementById('strip-range').textContent =
    `$${{minP.toFixed(0)}} – $${{maxP.toFixed(0)}}/MWh`;

  const ctx = document.getElementById('sparkline-chart').getContext('2d');
  new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels: DATA.map(d => fmtDate(d.date)),
      datasets: [{{
        data: prices,
        backgroundColor: DATA.map((d, i) => {{
          if (i === activeDay) return d.signal === 'ELEVATED' ? C.elevated : C.accent;
          return d.signal === 'ELEVATED' ? C.elevated + '99' : C.accent + '66';
        }}),
        borderRadius: 3,
        borderSkipped: false,
        barPercentage: 0.7,
        categoryPercentage: 0.85,
      }}],
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      onClick: (e, els) => {{ if (els.length) selectDay(els[0].index); }},
      plugins: {{ legend: {{ display: false }}, tooltip: {{
        callbacks: {{ label: c => `$${{c.parsed.y.toFixed(0)}}/MWh` }},
        backgroundColor: '#fff', titleColor: C.bright, bodyColor: C.text,
        borderColor: '#e4ebe4', borderWidth: 1,
      }} }},
      scales: {{
        x: {{ grid: {{ display: false }}, ticks: {{ color: C.text, font: {{ family: 'Inter', size: 9 }}, maxRotation: 0 }}, border: {{ display: false }} }},
        y: {{ display: false, min: Math.max(0, minP * 0.85) }},
      }},
    }},
  }});
}}

function renderDayDetail() {{
  const d = DATA[activeDay];
  const isElev = d.signal === 'ELEVATED';
  const col = sc(d.signal);

  const windOk  = d.wind >= 14;
  const cloudOk = d.cloud <= 45;
  const tempOk  = d.temp <= 72;
  const tightOk = d.tightness <= 20500;

  const actionText = isElev
    ? `Spike probability is <strong style="color:${{C.elevated}}">${{d.spike.toFixed(0)}}%</strong> — above the 50% threshold. Low wind and high cloud cover are squeezing renewables while demand stays elevated. Consider locking in forward contracts or reducing spot exposure on this day.`
    : `Grid looks well-supplied. Wind is delivering ${{(d.renewables/1000).toFixed(1)}}K MW of renewable capacity. DAM prices should remain subdued unless an unexpected demand event occurs.`;

  const watches = [];
  if (!windOk || d.wind < 16) watches.push(`Wind drops below 12 mph (currently ${{d.wind}} mph)`);
  if (!cloudOk || d.cloud > 40) watches.push(`Cloud cover spikes above 60% (currently ${{d.cloud}}%)`);
  if (!tempOk || d.temp > 68) watches.push(`Temp exceeds 85°F — cooling demand surge (currently ${{d.temp}}°F)`);
  watches.push(`RTM diverges more than 20% above DAM settlement`);

  document.getElementById('day-detail').innerHTML = `
    <div class="detail-header">
      <div class="detail-date">${{new Date(d.date + 'T00:00:00').toLocaleDateString('en-US', {{ weekday:'long', month:'long', day:'numeric' }})}}</div>
      <div class="detail-signal" style="background:${{col}}18;color:${{col}};border:1px solid ${{col}}44">${{d.signal}}</div>
    </div>

    <div class="price-card">
      <div class="pc-main">
        <div class="pc-label">Predicted DAM Price</div>
        <div class="pc-price" style="color:${{d.price > 100 ? C.elevated : C.accent}}">${{d.price > 100 ? '' : ''}}$${{d.price.toFixed(0)}}<span style="font-size:16px;font-weight:500">/MWh</span></div>
        <div class="pc-sub">Grid tightness: ${{(d.tightness/1000).toFixed(1)}}K MW · Renewables: ${{(d.renewables/1000).toFixed(1)}}K MW</div>
      </div>
      <div class="pc-divider"></div>
      <div class="pc-spike">
        <div class="pc-spike-val" style="color:${{d.spike > 50 ? C.elevated : d.spike > 40 ? C.warn : C.accent}}">${{d.spike.toFixed(0)}}%</div>
        <div class="pc-spike-label">Spike Prob</div>
        <div class="spike-bar-wrap" style="width:70px;margin-top:8px">
          <div class="spike-bar-fill" style="width:${{d.spike}}%;background:${{d.spike > 50 ? C.elevated : d.spike > 40 ? C.warn : C.accent}}"></div>
        </div>
      </div>
    </div>

    <div class="driver-grid">
      <div class="driver-card ${{windOk ? 'ok' : 'risk'}}">
        <div class="dc-top">
          <span class="dc-icon">💨</span>
          <span class="dc-label" style="color:${{windOk ? C.accent : C.elevated}}">Wind</span>
        </div>
        <div class="dc-value" style="color:${{windOk ? C.bright : C.elevated}}">${{d.wind}} <span style="font-size:13px">mph</span></div>
        <div class="dc-sub" style="color:${{windOk ? C.text : C.elevated}}">${{windOk ? '✓ adequate generation' : '⚠ below avg — less solar'}}</div>
      </div>
      <div class="driver-card ${{cloudOk ? 'ok' : 'risk'}}">
        <div class="dc-top">
          <span class="dc-icon">☁️</span>
          <span class="dc-label" style="color:${{cloudOk ? C.accent : C.elevated}}">Cloud Cover</span>
        </div>
        <div class="dc-value" style="color:${{cloudOk ? C.bright : C.elevated}}">${{d.cloud}}<span style="font-size:13px">%</span></div>
        <div class="dc-sub" style="color:${{cloudOk ? C.text : C.elevated}}">${{cloudOk ? '✓ solar unimpeded' : '⚠ solar suppressed'}}</div>
      </div>
      <div class="driver-card ${{tempOk ? 'ok' : 'risk'}}">
        <div class="dc-top">
          <span class="dc-icon">🌡️</span>
          <span class="dc-label" style="color:${{tempOk ? C.accent : C.elevated}}">Temperature</span>
        </div>
        <div class="dc-value" style="color:${{tempOk ? C.bright : C.elevated}}">${{d.temp}}°<span style="font-size:13px">F</span></div>
        <div class="dc-sub" style="color:${{tempOk ? C.text : C.elevated}}">${{tempOk ? '✓ moderate demand' : '⚠ elevated cooling load'}}</div>
      </div>
      <div class="driver-card ${{tightOk ? 'ok' : 'risk'}}">
        <div class="dc-top">
          <span class="dc-icon">⚡</span>
          <span class="dc-label" style="color:${{tightOk ? C.accent : C.elevated}}">Grid Tightness</span>
        </div>
        <div class="dc-value" style="color:${{tightOk ? C.bright : C.elevated}}">${{(d.tightness/1000).toFixed(1)}}<span style="font-size:13px">K MW</span></div>
        <div class="dc-sub" style="color:${{tightOk ? C.text : C.elevated}}">${{tightOk ? '✓ grid balanced' : '⚠ tight supply margin'}}</div>
      </div>
    </div>

    <div class="action-box ${{isElev ? 'risk' : 'ok'}}">
      <div class="ab-title" style="color:${{col}}">${{isElev ? '⚠ Action Recommended' : '✓ No Action Needed'}}</div>
      <div class="ab-body">${{actionText}}</div>
      <div class="ab-watches">
        <div style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;color:var(--text);margin-top:6px;margin-bottom:2px">Watch For</div>
        ${{watches.map(w => `<div class="ab-watch" style="--col:${{C.warn}}">${{w}}</div>`).join('')}}
      </div>
    </div>
  `;
}}

function buildSummary() {{
  const elevated = DATA.filter(d => d.signal === 'ELEVATED');
  const nextElev = elevated[0];
  const avgSpike = DATA.reduce((s, d) => s + d.spike, 0) / DATA.length;
  const maxSpike = Math.max(...DATA.map(d => d.spike));
  const maxSpikeDay = DATA.find(d => d.spike === maxSpike);
  const peakPrice = Math.max(...DATA.map(d => d.price));

  // Plain English headline
  let headline, body;
  if (elevated.length === 0) {{
    headline = "15-day window looks calm.";
    body = `Average spike probability is ${{avgSpike.toFixed(0)}}% — below the 50% alert threshold. No elevated risk days detected. Grid is well-supplied with renewables.`;
  }} else if (elevated.length === 1) {{
    headline = `One day to watch: ${{fmtDate(nextElev.date)}}.`;
    body = `${{nextElev.spike.toFixed(0)}}% spike probability driven by low wind and high cloud cover. All other days look manageable. Peak predicted price: $${{peakPrice.toFixed(0)}}/MWh.`;
  }} else {{
    headline = `${{elevated.length}} elevated risk days in the window.`;
    body = `Highest risk: ${{fmtDate(maxSpikeDay.date)}} at ${{maxSpike.toFixed(0)}}% spike probability. Consider hedging exposure on ${{elevated.map(e => fmtDate(e.date)).join(', ')}}.`;
  }}

  document.getElementById('summary-box').innerHTML = `
    <div class="sum-headline">${{headline}}</div>
    <div class="sum-body">${{body}}</div>
  `;

  // Next alert rows
  document.getElementById('next-alert-box').innerHTML = `
    <div class="nal-label">Spike Alerts</div>
    ${{elevated.length ? elevated.map((f, i) => {{
      const daysAway = DATA.findIndex(d => d.date === f.date);
      const bg = f.spike > 55 ? 'rgba(192,57,43,0.07)' : 'rgba(212,132,26,0.07)';
      const col = f.spike > 55 ? C.elevated : C.warn;
      return `<div class="alert-row" style="background:${{bg}}" onclick="selectDay(${{DATA.findIndex(d => d.date === f.date)}})">
        <div>
          <div class="ar-date" style="color:${{col}}">${{fmtDate(f.date)}}</div>
          <div class="ar-spike" style="color:var(--text)">${{f.spike.toFixed(0)}}% spike prob</div>
        </div>
        <div class="ar-days" style="background:${{col}}22;color:${{col}}">Day ${{daysAway + 1}}</div>
      </div>`;
    }}).join('') : '<div style="font-size:11px;color:var(--text);padding:4px 0">No alerts in window ✓</div>'}}
  `;
}}

function buildSignal() {{
  const d = DATA[activeDay];
  const isElev = d.signal === 'ELEVATED';
  const color  = sc(d.signal);

  document.getElementById('sp-day-header').textContent = `${{fmtDate(d.date)}} · Detail`;

  const badge = document.getElementById('badge-inner');
  badge.style.background = isElev ? 'rgba(192,57,43,0.08)' : 'rgba(26,122,74,0.08)';
  badge.style.border = `1px solid ${{color}}33`;
  document.getElementById('badge-val').textContent = d.signal;
  document.getElementById('badge-val').style.color = color;

  // Key drivers — what's actually causing this
  const windRisk  = d.wind < 14;
  const cloudRisk = d.cloud > 45;
  const heatRisk  = d.temp > 72;
  const drivers = [
    {{
      icon: '💨', label: 'Wind',
      val: `${{d.wind}} mph — ${{windRisk ? '⚠ below avg, less renewable output' : 'adequate wind generation'}}`,
      risk: windRisk,
    }},
    {{
      icon: '☁️', label: 'Cloud Cover',
      val: `${{d.cloud}}% — ${{cloudRisk ? '⚠ heavy cloud, solar suppressed' : 'good solar conditions'}}`,
      risk: cloudRisk,
    }},
    {{
      icon: '🌡️', label: 'Temperature',
      val: `${{d.temp}}°F mean — ${{heatRisk ? '⚠ elevated load expected' : 'moderate demand'}}`,
      risk: heatRisk,
    }},
    {{
      icon: '⚡', label: 'Grid Tightness',
      val: `${{(d.tightness/1000).toFixed(1)}}K MW gap between demand and supply`,
      risk: d.tightness > 20500,
    }},
  ];

  document.getElementById('signal-drivers').innerHTML = drivers.map(dr => `
    <div class="driver-row">
      <div class="driver-icon" style="background:${{dr.risk ? 'rgba(192,57,43,0.08)' : 'rgba(26,122,74,0.08)'}}">
        ${{dr.icon}}
      </div>
      <div class="driver-text">
        <div class="driver-label" style="color:${{dr.risk ? C.elevated : C.bright}}">${{dr.label}}</div>
        <div class="driver-val">${{dr.val}}</div>
      </div>
    </div>
  `).join('');

  // What to watch
  const watches = [];
  if (d.wind < 16) watches.push(`Wind drops below 12 mph — renewable shortfall risk`);
  if (d.cloud < 50) watches.push(`Cloud cover spikes above 60% — solar output drops`);
  if (d.temp > 68) watches.push(`Temp exceeds 80°F — cooling load surge`);
  watches.push(`RTM prices diverging more than 20% above DAM`);

  document.getElementById('signal-watch').innerHTML = `
    <div class="watch-label">Watch For</div>
    ${{watches.map(w => `<div class="watch-item">${{w}}</div>`).join('')}}
  `;
}}

function selectDay(i) {{
  document.querySelectorAll('.day-item').forEach(el => el.classList.remove('active'));
  document.getElementById(`day-${{i}}`).classList.add('active');
  activeDay = i;
  buildSignal();
  renderSparkline();
  renderDayDetail();
}}

buildSidebar();
buildSummary();
buildSignal();
renderSparkline();
renderDayDetail();
</script>
</body>
</html>"""

    with open(dashboard_path, "w") as f:
        f.write(html)
    print(f"[Dashboard] Saved to {dashboard_path}")

    # Auto-publish to docs/ for GitHub Pages
    import shutil
    docs_dir = Path(__file__).parent / "docs"
    docs_dir.mkdir(exist_ok=True)
    shutil.copy(dashboard_path, docs_dir / "index.html")
    print(f"[Dashboard] Published to docs/index.html")

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
