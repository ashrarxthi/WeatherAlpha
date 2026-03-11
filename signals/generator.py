"""
WeatherAlpha — AI Signal Generator
────────────────────────────────────
Takes the calibrated 15-day price forecast and asks Claude to synthesise
a structured trading signal: direction, confidence, thesis, specific trades.

Output schema (JSON):
{
  "signal":       "STRONG BUY" | "BUY" | "NEUTRAL" | "SELL" | "STRONG SELL",
  "confidence":   0-100,
  "thesis":       "2-3 sentence thesis",
  "key_driver":   "one phrase",
  "price_target": float,
  "trades": [
    {
      "action":    "Buy" | "Sell",
      "instrument":"e.g. ERCOT DAM HB_BUSAVG forward",
      "timeframe": "e.g. Days 4-7",
      "rationale": "why"
    }
  ],
  "risks": ["risk1", "risk2"],
  "hedge": "optional hedge recommendation"
}
"""

from __future__ import annotations
import json
import re
from datetime import datetime
from typing import Optional

import anthropic
import pandas as pd

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL


def generate_trading_signal(
    forecast_df:   pd.DataFrame,
    zone:          str = "HB_BUSAVG",
    base_price:    float = 45.0,
    rolling_mean:  Optional[float] = None,
) -> dict:
    """
    Send the 15-day calibrated forecast to Claude and get back a structured signal.

    Args:
        forecast_df:  Output of WeatherPriceModel.predict() — one row per day
        zone:         ERCOT hub name (for context)
        base_price:   Current rolling average price ($/MWh)
        rolling_mean: 30-day rolling mean (if different from base_price)

    Returns:
        Parsed signal dict matching the schema above.
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # ── Summarise forecast into a compact text table ──────────────────────────
    rows = []
    for _, r in forecast_df.iterrows():
        rows.append(
            f"  {str(r['date'])[:10]}  temp={r['temp_f_max']:.0f}°F  "
            f"wind={r['wind_mph_mean']:.0f}mph  "
            f"tightness={r['tightness']:.0f}  "
            f"pred_price=${r['predicted_price']:.0f}/MWh  "
            f"spike_prob={r['spike_probability']:.0f}%  "
            f"signal={r['day_signal']}"
        )
    table = "\n".join(rows)

    high_risk_days = forecast_df[forecast_df["spike_probability"] >= 60]
    peak_price_day = forecast_df.loc[forecast_df["predicted_price"].idxmax()]

    prompt = f"""You are a senior quantitative energy trader specialising in ERCOT (Texas) 
electricity markets. You have been handed a 15-day weather-driven price forecast 
produced by a calibrated ML model trained on 2 years of historical ERCOT settlement 
prices and WeatherNext 2 ensemble weather data.

MARKET CONTEXT
  Zone:          {zone}
  Current base:  ${base_price:.0f}/MWh (30-day rolling mean: ${rolling_mean or base_price:.0f}/MWh)
  Today:         {datetime.utcnow().strftime("%Y-%m-%d")}

15-DAY FORECAST
{table}

HIGH-RISK WINDOW(S):
{high_risk_days[['date','predicted_price','spike_probability']].to_string(index=False) if not high_risk_days.empty else "  None identified"}

PEAK PRICE DAY: {str(peak_price_day['date'])[:10]} — ${peak_price_day['predicted_price']:.0f}/MWh 
(spike prob {peak_price_day['spike_probability']:.0f}%)

Generate a structured trading signal and return ONLY a valid JSON object — 
no markdown, no preamble, no explanation outside the JSON.

Required schema:
{{
  "signal":       "STRONG BUY" | "BUY" | "NEUTRAL" | "SELL" | "STRONG SELL",
  "confidence":   integer 0-100,
  "thesis":       "2-3 sentences summarising the key weather-to-price thesis",
  "key_driver":   "one concise phrase naming the single biggest price driver",
  "price_target": float (estimated peak price $/MWh during the window),
  "trades": [
    {{
      "action":     "Buy" or "Sell",
      "instrument": "specific ERCOT product (e.g. DAM HB_BUSAVG forward days 4-7)",
      "timeframe":  "date range or day labels",
      "rationale":  "one sentence"
    }}
  ],
  "risks": ["up to 3 downside risks to the thesis"],
  "hedge": "optional hedge trade to protect against the primary risk"
}}"""

    print("[Claude] Generating trading signal...")
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()

    # Strip any accidental markdown fences
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()

    try:
        signal = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[Claude] JSON parse error: {e}\nRaw response:\n{raw}")
        raise

    print(f"[Claude] Signal: {signal.get('signal')} (confidence {signal.get('confidence')}%)")
    return signal


def format_signal_report(signal: dict, forecast_df: pd.DataFrame, zone: str) -> str:
    """Pretty-print the signal as a terminal-friendly report."""
    sep = "─" * 60
    trades_txt = "\n".join(
        f"  [{t['action'].upper()}] {t['instrument']}\n"
        f"    Timeframe: {t['timeframe']}\n"
        f"    Rationale: {t['rationale']}"
        for t in signal.get("trades", [])
    )
    risks_txt = "\n".join(f"  ▲ {r}" for r in signal.get("risks", []))

    return f"""
{sep}
WEATHERALPHA — {zone} TRADING SIGNAL
{sep}
  SIGNAL:       {signal['signal']}
  CONFIDENCE:   {signal['confidence']}%
  PRICE TARGET: ${signal.get('price_target', 'N/A')}/MWh
  KEY DRIVER:   {signal.get('key_driver', '')}

THESIS
  {signal['thesis']}

RECOMMENDED TRADES
{trades_txt or '  None'}

RISK FACTORS
{risks_txt or '  None identified'}

HEDGE
  {signal.get('hedge', 'None recommended')}
{sep}
"""
