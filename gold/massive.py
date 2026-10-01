"""
Massive (formerly Polygon.io) gold provider — true XAU/USD spot with bid/ask.

WHY THIS EXISTS
    The previous Azure-compatible path was PAXG (a tokenised ounce) corrected by
    a constant offset. Two things killed it:
      * Binance geo-blocks Ontario, and the app runs in Azure Canada Central,
        so the primary PAXG source returns HTTP 451 there.
      * The offset drifted. Measured 2026-10-01: PAXG after correction sat
        $13.91 above true spot, against a calibrated p95 of $4.33. PAXG carries
        a crypto premium that moves; a frozen constant was never going to hold.

    Massive quotes XAU/USD directly, so there is no proxy and no offset.

WHAT THE SUBSCRIPTION ACTUALLY DELIVERS (measured 2026-10-01, Currencies Starter)
    last_quote   real bid/ask, refreshing every ~2s with genuine movement
    1/second     1-second aggregates, ~4 ticks each
    1/minute     ~145 ticks/minute underlying
    1/day        daily bars
    1/hour       RETURNS ZERO ROWS for forex — see fetch_hourly_spot below.

ENDPOINT QUIRK YOU WILL TRIP OVER
    `range/1/hour/...` and `range/60/minute/...` both return resultsCount 0 for
    C:XAUUSD, while 1/second, 1/minute and 1/day all work. So hourly bars are
    built by resampling minutes. 35 days of minutes is 35,424 rows in ONE
    request (under the 50,000 cap, no pagination) and yields 617 hourly bars —
    comfortably more than the 200 EMA200(hourly) needs.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests

log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
BASE = "https://api.polygon.io"
PAIR = "XAU/USD"
TICKER = "C:XAUUSD"

# Typical interbank spread on this feed, from 20 consecutive pulls on
# 2026-10-01 ~14:30 ET: min 0.45, mean 0.52, max 0.56.
#
# READ THIS BEFORE USING IT AS A COST INPUT. It is NOT what you pay. Your Exness
# account was measured at a $0.20 median (research.SPREAD_MEDIAN), and this feed
# is an aggregated interbank quote running ~2.6x that. It is used only as a
# STRESS RATIO: live / typical. When this feed's spread trebles, your broker's
# will widen too, and that is the signal worth acting on.
#
# Caveat: 20 samples inside one minute of the US afternoon is not a session
# profile. Recalibrate across London/NY/Asia before leaning on the ratio hard.
SPREAD_TYPICAL = 0.52
SPREAD_SAMPLE_NOTE = "20 pulls, 2026-10-01 14:30 ET"


def _key() -> str | None:
    try:
        from config import MASSIVE_API_KEY
        return MASSIVE_API_KEY or None
    except Exception:
        return None


def available() -> bool:
    return bool(_key())


def _get(path: str, **params) -> dict | None:
    key = _key()
    if not key:
        return None
    params["apiKey"] = key
    try:
        r = requests.get(f"{BASE}{path}", params=params, timeout=10)
        if r.status_code != 200:
            log.warning("massive %s -> HTTP %s", path, r.status_code)
            return None
        j = r.json()
        # A plan that does not cover this data answers 200 with NOT_AUTHORIZED,
        # so status must be checked explicitly rather than trusting the code.
        if j.get("status") in ("NOT_AUTHORIZED", "ERROR"):
            log.warning("massive %s -> %s: %s", path, j.get("status"), j.get("message"))
            return None
        return j
    except Exception as exc:
        log.warning("massive %s failed: %s", path, exc)
        return None


def last_quote() -> dict | None:
    """Live bid/ask. Returns None (never a guess) when unavailable."""
    j = _get(f"/v1/last_quote/currencies/{PAIR}")
    if not j or "last" not in j:
        return None
    last = j["last"]
    bid, ask = last.get("bid"), last.get("ask")
    if not bid or not ask or bid <= 0 or ask <= 0:
        return None
    spread = ask - bid
    ts = last.get("timestamp")
    return {
        "bid": float(bid),
        "ask": float(ask),
        "price": (float(bid) + float(ask)) / 2,
        "spread": float(spread),
        # Ratio against this feed's own typical spread. >1 means conditions are
        # wider than normal; the absolute number is not your broker's cost.
        "spread_x": round(spread / SPREAD_TYPICAL, 2) if SPREAD_TYPICAL else None,
        "ts": datetime.fromtimestamp(ts / 1000, tz=ET) if ts else datetime.now(ET),
    }


def snapshot() -> dict | None:
    """Session OHLC and today's change, straight from the venue."""
    j = _get(f"/v2/snapshot/locale/global/markets/forex/tickers/{TICKER}")
    t = (j or {}).get("ticker")
    if not t:
        return None
    day = t.get("day") or {}
    return {
        "open": day.get("o"), "high": day.get("h"),
        "low": day.get("l"), "close": day.get("c"),
        "change": t.get("todaysChange"),
        "change_pct": t.get("todaysChangePerc"),
    }


def fetch_hourly_spot(days: int = 35) -> pd.DataFrame:
    """Hourly XAU/USD SPOT bars, indexed in New York time.

    Built by resampling 1-minute bars because the hourly endpoint returns no
    rows for forex (see module docstring). This replaces the old GC=F futures
    series, which sat ~$62 above spot and made every printed level unfillable.
    """
    end = datetime.now(UTC).date()
    start = end - timedelta(days=days)
    j = _get(f"/v2/aggs/ticker/{TICKER}/range/1/minute/{start}/{end}", limit=50000)
    rows = (j or {}).get("results") or []
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.set_index("dt").sort_index()
    hourly = (
        df.resample("1h")
        .agg({"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"})
        .dropna()
    )
    hourly.columns = ["Open", "High", "Low", "Close", "Volume"]
    hourly.index = hourly.index.tz_convert(ET)
    return hourly
