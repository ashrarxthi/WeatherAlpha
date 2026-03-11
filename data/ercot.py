"""
WeatherAlpha — ERCOT Historical Price Fetcher
─────────────────────────────────────────────
Pulls Day-Ahead Market (DAM) and Real-Time Market (RTM) settlement
point prices from the ERCOT public API.

API docs: https://developer.ercot.com/applications/pubapi/
Endpoint: NP4-190-CD  (DAM Settlement Point Prices)
Endpoint: NP6-905-CD  (RTM Settlement Point Prices - 15 min)

ERCOT hubs used:
  HB_BUSAVG  — load-weighted average across all buses (best overall signal)
  HB_HOUSTON — Houston hub
  HB_NORTH   — North Texas hub
  HB_SOUTH   — South Texas hub
  HB_WEST    — West Texas hub (wind-heavy, behaves differently)
"""

from __future__ import annotations
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    ERCOT_API_BASE,
    ERCOT_USERNAME,
    ERCOT_PASSWORD,
    ERCOT_SUBSCRIPTION,
    CALIBRATION_DAYS,
    DEFAULT_ZONE,
    SPIKE_MULTIPLIER,
)

# Correct endpoint URLs
ERCOT_DAM_BASE   = "https://api.ercot.com/api/public-reports/np4-190-cd"
ERCOT_RTM_BASE   = "https://api.ercot.com/api/public-reports/np6-905-cd"
ERCOT_TOKEN_URL  = "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"


# ── Auth token ───────────────────────────────────────────────────────────────
def _get_token() -> str:
    """Fetch a fresh Bearer token from ERCOT. Expires every hour."""
    resp = requests.post(ERCOT_TOKEN_URL, data={
        "username":      ERCOT_USERNAME,
        "password":      ERCOT_PASSWORD,
        "grant_type":    "password",
        "scope":         "openid fec253ea-0d06-4272-a5e6-b478baeecd70 offline_access",
        "client_id":     "fec253ea-0d06-4272-a5e6-b478baeecd70",
        "response_type": "id_token",
    }, timeout=15)
    resp.raise_for_status()
    return resp.json()["access_token"]


# ── HTTP session with retry logic ────────────────────────────────────────────
def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    token = _get_token()
    session.headers.update({
        "Ocp-Apim-Subscription-Key": ERCOT_SUBSCRIPTION,
        "Authorization": f"Bearer {token}",
    })
    return session


def fetch_dam_prices(
    start_date: Optional[datetime] = None,
    end_date:   Optional[datetime] = None,
    zone:       str = DEFAULT_ZONE,
    session:    Optional[requests.Session] = None,
) -> pd.DataFrame:
    """
    Fetch Day-Ahead Market hourly settlement prices for a given hub.

    Returns DataFrame with columns: datetime, zone, price_dam
    """
    if end_date is None:
        end_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    if start_date is None:
        start_date = end_date - timedelta(days=CALIBRATION_DAYS)

    session = session or _make_session()
    all_records = []
    cursor = start_date

    # ERCOT API returns max 31 days per request — paginate monthly
    while cursor < end_date:
        chunk_end = min(cursor + timedelta(days=28), end_date)
        params = {
            "deliveryDateFrom": cursor.strftime("%Y-%m-%d"),
            "deliveryDateTo":   chunk_end.strftime("%Y-%m-%d"),
            "settlementPoint":  zone,
            "size":             10_000,
        }
        print(f"[ERCOT DAM] Fetching {cursor.date()} → {chunk_end.date()} for {zone}...")
        resp = session.get(f"{ERCOT_DAM_BASE}/dam_stlmnt_pnt_prices", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        for row in data.get("data", []):
            # Response is list: [deliveryDate, hourEnding, settlementPoint, settlementPointPrice, DSTFlag]
            # ERCOT uses hour "24:00" to mean midnight end-of-day — convert to next day 00:00
            hour_str = row[1]
            if hour_str == "24:00":
                dt = datetime.strptime(row[0], "%Y-%m-%d") + timedelta(days=1)
            else:
                dt = datetime.strptime(f"{row[0]} {hour_str}", "%Y-%m-%d %H:%M")
            all_records.append({
                "datetime":  dt,
                "zone":      row[2],
                "price_dam": float(row[3]),
            })

        cursor = chunk_end + timedelta(days=1)
        time.sleep(0.3)  # be polite to the API

    df = pd.DataFrame(all_records)
    if df.empty:
        raise ValueError(f"No DAM price data returned for zone={zone}, check your API credentials.")

    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"[ERCOT DAM] Got {len(df):,} hourly records ({df['datetime'].min().date()} – {df['datetime'].max().date()})")
    return df


def fetch_rtm_prices(
    start_date: Optional[datetime] = None,
    end_date:   Optional[datetime] = None,
    zone:       str = DEFAULT_ZONE,
    session:    Optional[requests.Session] = None,
) -> pd.DataFrame:
    """
    Fetch Real-Time Market 15-min settlement prices and resample to hourly.
    RTM prices are noisier but capture actual spike events better.
    """
    if end_date is None:
        end_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    if start_date is None:
        start_date = end_date - timedelta(days=CALIBRATION_DAYS)

    session = session or _make_session()
    all_records = []
    cursor = start_date

    while cursor < end_date:
        chunk_end = min(cursor + timedelta(days=7), end_date)
        params = {
            "deliveryDateFrom": cursor.strftime("%Y-%m-%d"),
            "deliveryDateTo":   chunk_end.strftime("%Y-%m-%d"),
            "settlementPoint":  zone,
            "size":             50_000,
        }
        print(f"[ERCOT RTM] Fetching {cursor.date()} → {chunk_end.date()}...")
        resp = session.get(f"{ERCOT_RTM_BASE}/spp_node_zone_hub", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        for row in data.get("data", []):
            # Response: [deliveryDate, deliveryHour, deliveryInterval, settlementPoint, settlementPointType, settlementPointPrice, DSTFlag]
            hour = int(row[1])
            interval_minute = (int(row[2]) - 1) * 15
            if hour == 24:
                dt = datetime.strptime(row[0], "%Y-%m-%d") + timedelta(days=1)
            else:
                dt = datetime.strptime(f"{row[0]} {hour:02d}:{interval_minute:02d}", "%Y-%m-%d %H:%M")
            all_records.append({
                "datetime":  dt,
                "zone":      row[3],
                "price_rtm": float(row[5]),
            })

        cursor = chunk_end + timedelta(days=1)
        time.sleep(0.2)

    df = pd.DataFrame(all_records).sort_values("datetime")

    # Resample 15-min → hourly average
    df = df.set_index("datetime").resample("1h")["price_rtm"].mean().reset_index()
    df["zone"] = zone
    print(f"[ERCOT RTM] Got {len(df):,} hourly records (resampled from 15-min)")
    return df


def build_daily_price_history(
    zone: str = DEFAULT_ZONE,
) -> pd.DataFrame:
    """
    Combine DAM + RTM prices into a daily summary used for model calibration.

    Returns columns:
        date, zone,
        price_mean, price_max, price_p95,
        spike_count, spike_flag,
        rolling_mean_30d, rolling_mean_7d
    """
    dam = fetch_dam_prices(zone=zone)
    rtm = fetch_rtm_prices(zone=zone)

    # Merge on datetime
    merged = pd.merge(dam, rtm, on=["datetime", "zone"], how="outer")
    # Use RTM where available (more accurate for spikes), fallback to DAM
    merged["price"] = merged["price_rtm"].combine_first(merged["price_dam"])
    merged["date"]  = merged["datetime"].dt.date

    daily = merged.groupby("date").agg(
        price_mean  = ("price", "mean"),
        price_max   = ("price", "max"),
        price_p95   = ("price", lambda x: x.quantile(0.95)),
        price_min   = ("price", "min"),
        hour_count  = ("price", "count"),
    ).reset_index()

    daily["zone"] = zone

    # ── Rolling baselines ─────────────────────────────────────────────────────
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["rolling_mean_30d"] = daily["price_mean"].rolling(30, min_periods=7).mean()
    daily["rolling_mean_7d"]  = daily["price_mean"].rolling(7,  min_periods=3).mean()

    # ── Spike flag: max price > SPIKE_MULTIPLIER × 30-day rolling mean ────────
    daily["spike_flag"]  = daily["price_max"] > (daily["rolling_mean_30d"] * SPIKE_MULTIPLIER)
    daily["spike_count"] = daily["spike_flag"].astype(int)

    print(f"[ERCOT] Built daily price history: {len(daily)} days, "
          f"{daily['spike_flag'].sum()} spike days ({daily['spike_flag'].mean()*100:.1f}%)")
    return daily
