"""
WeatherAlpha — WeatherNext 2 Data Fetcher
─────────────────────────────────────────
Pulls 15-day ensemble forecasts from the WeatherNext 2 public BigQuery dataset.

WeatherNext 2 fields we care about:
  - 2m_temperature          → surface temp (Kelvin → convert to °F)
  - 10m_u_component_of_wind → east-west wind component (m/s)
  - 10m_v_component_of_wind → north-south wind component (m/s)
  - total_cloud_cover       → 0–1 fraction
  - total_precipitation     → mm/hr
  - surface_solar_radiation_downwards → W/m²

ERCOT grid centre-point:  lat 31.0, lon -99.0  (central Texas)
"""

from __future__ import annotations
import math
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

from config import (
    GCP_PROJECT_ID,
    GOOGLE_CREDENTIALS_PATH,
    WEATHERNEXT_TABLE,
    FORECAST_DAYS,
)


# ── ERCOT grid bounding box ──────────────────────────────────────────────────
ERCOT_BOUNDS = {
    "lat_min": 25.8, "lat_max": 36.5,
    "lon_min": -106.6, "lon_max": -93.5,
}

# ── WeatherNext grid resolution is ~0.25° ───────────────────────────────────
GRID_RES = 0.25


def _get_client() -> bigquery.Client:
    creds = service_account.Credentials.from_service_account_file(
        GOOGLE_CREDENTIALS_PATH,
        scopes=["https://www.googleapis.com/auth/bigquery"],
    )
    creds = creds.with_quota_project(GCP_PROJECT_ID)
    return bigquery.Client(project=GCP_PROJECT_ID, credentials=creds)

def _kelvin_to_fahrenheit(k: float) -> float:
    return (k - 273.15) * 9 / 5 + 32


def _wind_speed(u: float, v: float) -> float:
    """Combined wind speed from u/v components (m/s → mph)."""
    return math.sqrt(u**2 + v**2) * 2.237


def fetch_ercot_forecast(
    client: Optional[bigquery.Client] = None,
    forecast_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Query WeatherNext 2 for a 15-day hourly forecast over the ERCOT footprint.

    Returns a DataFrame with columns:
        valid_time, lat, lon,
        temp_f, wind_mph, cloud_cover, precip_mm, solar_w_m2
    """
    client = client or _get_client()
    if forecast_date is None:
        # Use the most recent 00Z run
        forecast_date = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    end_time = forecast_date + timedelta(days=FORECAST_DAYS)

    query = f"""
    SELECT
        init_time,
        valid_time,
        latitude   AS lat,
        longitude  AS lon,
        `2m_temperature`                          AS temp_k,
        `10m_u_component_of_wind`                 AS wind_u,
        `10m_v_component_of_wind`                 AS wind_v,
        total_cloud_cover                         AS cloud_cover,
        total_precipitation                       AS precip_mm,
        surface_solar_radiation_downwards         AS solar_w_m2
    FROM `{WEATHERNEXT_TABLE}`
    WHERE
        -- Most recent model run on or before our target date
        init_time = (
            SELECT MAX(init_time)
            FROM `{WEATHERNEXT_TABLE}`
            WHERE init_time <= @forecast_date
        )
        AND valid_time BETWEEN @forecast_date AND @end_time
        -- ERCOT spatial bounding box
        AND latitude  BETWEEN {ERCOT_BOUNDS["lat_min"]} AND {ERCOT_BOUNDS["lat_max"]}
        AND longitude BETWEEN {ERCOT_BOUNDS["lon_min"]} AND {ERCOT_BOUNDS["lon_max"]}
    ORDER BY valid_time, lat, lon
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("forecast_date", "TIMESTAMP", forecast_date),
            bigquery.ScalarQueryParameter("end_time",      "TIMESTAMP", end_time),
        ]
    )

    print(f"[WeatherNext] Fetching forecast from {forecast_date.date()} → {end_time.date()}...")
    df = client.query(query, job_config=job_config).to_dataframe()
    print(f"[WeatherNext] Retrieved {len(df):,} rows covering {df['lat'].nunique()} grid points.")

    # ── Unit conversions ─────────────────────────────────────────────────────
    df["temp_f"]    = df["temp_k"].apply(_kelvin_to_fahrenheit)
    df["wind_mph"]  = df.apply(lambda r: _wind_speed(r["wind_u"], r["wind_v"]), axis=1)
    df["cloud_pct"] = (df["cloud_cover"] * 100).clip(0, 100)

    return df[[
        "init_time", "valid_time", "lat", "lon",
        "temp_f", "wind_mph", "cloud_pct", "precip_mm", "solar_w_m2"
    ]]


def aggregate_to_daily(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse hourly grid-level data to daily ERCOT-wide averages.
    Also computes demand-pressure proxy: high temp + low wind/solar = tight grid.
    """
    hourly_df["date"] = pd.to_datetime(hourly_df["valid_time"]).dt.date

    daily = hourly_df.groupby("date").agg(
        temp_f_max   = ("temp_f",    "max"),
        temp_f_mean  = ("temp_f",    "mean"),
        temp_f_min   = ("temp_f",    "min"),
        wind_mph_mean= ("wind_mph",  "mean"),
        wind_mph_max = ("wind_mph",  "max"),
        cloud_pct    = ("cloud_pct", "mean"),
        precip_mm    = ("precip_mm", "sum"),
        solar_w_m2   = ("solar_w_m2","mean"),
    ).reset_index()

    # ── Renewable generation proxy ────────────────────────────────────────────
    # Wind capacity factor (ERCOT ~35GW nameplate wind)
    daily["wind_gen_proxy"] = daily["wind_mph_mean"] / 30.0 * 35_000  # MWh-ish

    # Solar capacity factor (ERCOT ~20GW nameplate solar)
    daily["solar_gen_proxy"] = (daily["solar_w_m2"] / 1000.0) * 0.18 * 20_000

    daily["renewable_gen_proxy"] = daily["wind_gen_proxy"] + daily["solar_gen_proxy"]

    # ── Demand proxy based on temperature (heating/cooling degree days) ───────
    BALANCE_POINT = 65  # °F — industry standard
    daily["hdd"] = (BALANCE_POINT - daily["temp_f_mean"]).clip(lower=0)
    daily["cdd"] = (daily["temp_f_mean"] - BALANCE_POINT).clip(lower=0)
    # ERCOT baseline demand ~40GW, each degree moves it ~500MW
    daily["demand_proxy"] = 40_000 + (daily["cdd"] * 500) + (daily["hdd"] * 400)

    # ── Grid tightness: demand minus available renewable ──────────────────────
    daily["tightness"] = daily["demand_proxy"] - daily["renewable_gen_proxy"]

    return daily
