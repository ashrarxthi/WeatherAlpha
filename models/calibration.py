"""
WeatherAlpha — Tightness-to-Price Calibration Model
─────────────────────────────────────────────────────
This is the core IP of WeatherAlpha.

It takes:
  - Historical ERCOT daily price data  (from ercot.py)
  - Historical weather data            (reconstructed from WeatherNext reanalysis)

And fits a model that learns:
  "Given tightness index X on day D, what is the expected price and
   probability that prices spike above 2× the rolling mean?"

We use two models:
  1. LinearRegression   → predict next-day mean price ($/MWh)
  2. LogisticRegression → predict P(spike) as a probability 0–1

Features used:
  - tightness         (demand_proxy − renewable_proxy)
  - temp_f_max        (extreme heat is highly nonlinear)
  - cdd / hdd         (cooling/heating degree days)
  - wind_mph_mean
  - cloud_pct
  - day_of_week       (weekends have lower industrial demand)
  - month             (summer/winter seasonality)
  - lag_price_1d      (yesterday's price — momentum signal)
  - lag_spike_3d      (did we spike recently? — mean reversion)
"""

from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, roc_auc_score

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "tightness", "temp_f_max", "cdd", "hdd",
    "wind_mph_mean", "cloud_pct",
    "day_of_week", "month",
    "lag_price_1d", "lag_spike_3d",
]


# ── Feature engineering ──────────────────────────────────────────────────────
def build_feature_matrix(
    weather_daily: pd.DataFrame,
    price_daily:   pd.DataFrame,
) -> pd.DataFrame:
    """
    Join weather and price data on date, engineer features, return clean matrix.
    Both DataFrames must have a 'date' column.
    """
    weather_daily = weather_daily.copy()
    price_daily   = price_daily.copy()
    weather_daily["date"] = pd.to_datetime(weather_daily["date"])
    price_daily["date"]   = pd.to_datetime(price_daily["date"])

    df = pd.merge(weather_daily, price_daily, on="date", how="inner")

    # Calendar features
    df["day_of_week"] = df["date"].dt.dayofweek   # 0=Mon, 6=Sun
    df["month"]       = df["date"].dt.month

    # Lag features (require sort by date)
    df = df.sort_values("date").reset_index(drop=True)
    df["lag_price_1d"] = df["price_mean"].shift(1)
    df["lag_spike_3d"] = df["spike_flag"].astype(int).rolling(3, min_periods=1).sum().shift(1)

    # Drop first few rows where lags are NaN
    df = df.dropna(subset=FEATURE_COLS + ["price_mean", "spike_flag"])
    return df


# ── Training ─────────────────────────────────────────────────────────────────
class WeatherPriceModel:
    """
    Wraps a price regression + spike classifier.
    Persists to disk so you don't have to retrain every run.
    """

    def __init__(self):
        self.scaler        = StandardScaler()
        self.price_model   = LinearRegression()
        self.spike_model   = LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced")
        self.trained       = False
        self.metrics: dict = {}

    def fit(self, df: pd.DataFrame) -> dict:
        """
        Train on historical joined weather+price DataFrame.
        Uses time-series cross-validation (no data leakage).
        Returns dict of evaluation metrics.
        """
        X = df[FEATURE_COLS].values
        y_price = df["price_mean"].values
        y_spike = df["spike_flag"].astype(int).values

        # Time-series CV (always train on past, test on future)
        tscv = TimeSeriesSplit(n_splits=5)
        price_maes, spike_aucs = [], []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_tr, X_te = X[train_idx], X[test_idx]
            yp_tr, yp_te = y_price[train_idx], y_price[test_idx]
            ys_tr, ys_te = y_spike[train_idx], y_spike[test_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            pm = LinearRegression().fit(X_tr_s, yp_tr)
            sm = LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced").fit(X_tr_s, ys_tr)

            price_maes.append(mean_absolute_error(yp_te, pm.predict(X_te_s)))
            if ys_te.sum() > 0:
                spike_aucs.append(roc_auc_score(ys_te, sm.predict_proba(X_te_s)[:, 1]))

            print(f"  Fold {fold+1}: price MAE=${price_maes[-1]:.2f}/MWh, "
                  f"spike AUC={spike_aucs[-1]:.3f}" if spike_aucs else f"  Fold {fold+1}: price MAE=${price_maes[-1]:.2f}/MWh")

        # Final fit on all data
        X_scaled = self.scaler.fit_transform(X)
        self.price_model.fit(X_scaled, y_price)
        self.spike_model.fit(X_scaled, y_spike)
        self.trained = True

        self.metrics = {
            "price_mae_mean":    float(np.mean(price_maes)),
            "price_mae_std":     float(np.std(price_maes)),
            "spike_auc_mean":    float(np.mean(spike_aucs)) if spike_aucs else None,
            "n_train":           int(len(df)),
            "spike_rate":        float(y_spike.mean()),
            "_train_price_mean": float(np.mean(y_price)),  # used as lag fallback in predict()
        }

        print(f"\n[Model] Training complete.")
        print(f"  Price MAE:  ${self.metrics['price_mae_mean']:.2f} ± {self.metrics['price_mae_std']:.2f} /MWh")
        if self.metrics["spike_auc_mean"]:
            print(f"  Spike AUC:  {self.metrics['spike_auc_mean']:.3f} (0.5=random, 1.0=perfect)")
        return self.metrics

    def predict(self, weather_forecast: pd.DataFrame) -> pd.DataFrame:
        """
        Given a 15-day weather forecast DataFrame (from weathernext.aggregate_to_daily),
        return a DataFrame with predicted price and spike probability for each day.

        The forecast won't have lag_price_1d / lag_spike_3d from history — we impute
        those from the model's training data mean (conservative fallback).
        """
        assert self.trained, "Call .fit() before .predict()"

        df = weather_forecast.copy()

        # Fill lag features with neutral values if missing (forecasting into future)
        if "lag_price_1d" not in df.columns:
            df["lag_price_1d"] = self.metrics.get("_train_price_mean", 45.0)
        if "lag_spike_3d" not in df.columns:
            df["lag_spike_3d"] = 0.0

        # Calendar
        df["date"]        = pd.to_datetime(df["date"])
        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"]       = df["date"].dt.month

        X = df[FEATURE_COLS].fillna(0).values
        X_scaled = self.scaler.transform(X)

        df["predicted_price"]     = self.price_model.predict(X_scaled).clip(min=0)
        df["spike_probability"]   = self.spike_model.predict_proba(X_scaled)[:, 1] * 100
        df["spike_probability"]   = df["spike_probability"].clip(0, 99)

        # Signal label per day
        df["day_signal"] = df["spike_probability"].apply(_label_signal)

        return df[[
            "date", "temp_f_max", "temp_f_mean", "wind_mph_mean", "cloud_pct",
            "demand_proxy", "renewable_gen_proxy", "tightness",
            "predicted_price", "spike_probability", "day_signal"
        ]]

    def save(self, path: Optional[Path] = None):
        path = path or MODEL_DIR / "weather_price_model.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[Model] Saved to {path}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "WeatherPriceModel":
        path = path or MODEL_DIR / "weather_price_model.pkl"
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"[Model] Loaded from {path}")
        return model


def _label_signal(spike_prob: float) -> str:
    if spike_prob >= 70:  return "HIGH RISK"
    if spike_prob >= 50:  return "ELEVATED"
    if spike_prob >= 30:  return "MODERATE"
    return "LOW RISK"
