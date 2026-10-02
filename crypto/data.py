"""
Crypto market data from Massive (Polygon), and the universe filter.

WHY A UNIVERSE FILTER IS NOT OPTIONAL HERE
    Massive lists 598 USD crypto pairs. Only 28 clear $10M/day and 95 clear
    $1M/day. A screen that surfaces a $50k/day altcoin is not a screen, it is a
    way to get filled 8% away from the quote. Equities have the same problem but
    an order of magnitude milder; crypto's long tail is most of the list.

STABLECOINS ARE EXCLUDED BY VOLATILITY, NOT BY NAME
    USDT, USDC and UST all rank near the top by dollar volume and would
    otherwise fill both the long and the short list with instruments pegged to
    $1. They are removed by a minimum-volatility floor rather than a hardcoded
    list, because a list goes stale the moment a new stablecoin launches and the
    property that disqualifies them is measurable: annualised vol near zero
    against crypto's typical 40-100%.

    Consequence worth stating: a DE-PEGGING stablecoin would be an excellent
    short and this filter hides it. That is a deliberate trade - de-peg events
    are rare, abrupt, and not what a daily trend screen is for.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE = "https://api.polygon.io"
BENCH = "X:BTCUSD"          # crypto's market factor, the analogue of SPY

# Liquidity floor, in median daily dollar volume over the lookback.
MIN_DOLLAR_VOLUME = 5_000_000.0
# Annualised vol below this is a peg, not an asset. Measured: majors run
# 40-100%, stablecoins under 2%.
MIN_ANNUAL_VOL_PCT = 10.0

# Reference asset for the real-world-asset filter. See drop_rwa_tokens().
GOLD = "C:XAUUSD"

# Massive's plan allows unlimited calls, so the only reason to serialise would be
# politeness. 8 is enough to cut a 300-ticker pull from ~5 minutes to well under
# one without hammering anything.
FETCH_WORKERS = 8


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
    for attempt in range(3):
        try:
            r = requests.get(f"{BASE}{path}", params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code != 200:
                return None
            j = r.json()
            if j.get("status") in ("NOT_AUTHORIZED", "ERROR"):
                log.warning("crypto %s -> %s", path, j.get("message"))
                return None
            return j
        except Exception as exc:
            if attempt == 2:
                log.warning("crypto %s failed: %s", path, exc)
    return None


def grouped_day(d: date) -> pd.DataFrame:
    """Every crypto ticker's OHLCV for one day, in a single request."""
    j = _get(f"/v2/aggs/grouped/locale/global/market/crypto/{d.isoformat()}")
    rows = (j or {}).get("results") or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).rename(columns={
        "T": "ticker", "o": "open", "h": "high", "l": "low",
        "c": "close", "v": "volume", "vw": "vwap", "n": "trades",
    })
    df["date"] = pd.to_datetime(d)
    keep = [c for c in ("ticker", "date", "open", "high", "low", "close",
                        "volume", "vwap", "trades") if c in df.columns]
    return df[keep]


def daily_bars(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Daily OHLCV for one ticker over a range (one request)."""
    j = _get(f"/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}", limit=50000)
    rows = (j or {}).get("results") or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).rename(columns={
        "o": "open", "h": "high", "l": "low", "c": "close",
        "v": "volume", "vw": "vwap", "n": "trades",
    })
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_localize(None)
    df["ticker"] = ticker
    # Thin pairs come back WITHOUT a vw field entirely, so the column must be
    # created rather than selected. Falling back to close is correct here: for a
    # bar with few trades, close IS approximately the volume-weighted price, and
    # the alternative (dropping the pair) would silently shrink the universe.
    if "vwap" not in df.columns:
        df["vwap"] = df["close"]
    df["vwap"] = df["vwap"].fillna(df["close"])
    for c in ("open", "high", "low", "close", "volume"):
        if c not in df.columns:
            return pd.DataFrame()          # genuinely unusable, skip the pair
    return df[["ticker", "date", "open", "high", "low", "close", "volume", "vwap"]]


def candidate_tickers(top_n: int = 150, on: date | None = None) -> list[str]:
    """USD pairs ranked by one day's dollar volume, as a shortlist to pull history for.

    SELECTION BIAS, STATED PLAINLY: ranking on a RECENT day means the shortlist
    is made of things that are liquid TODAY. Coins that died during the sample
    never appear. That flatters a long screen and starves a short screen of its
    best candidates, in exactly the way today's-survivors bias does for equities
    - only worse, because crypto's mortality rate is far higher.
    """
    on = on or (datetime.now(UTC).date() - timedelta(days=1))
    for back in range(0, 5):                       # step back over quiet days
        df = grouped_day(on - timedelta(days=back))
        if not df.empty:
            break
    if df.empty:
        return []
    df = df[df["ticker"].str.endswith("USD")].copy()
    df["dollar_volume"] = df["volume"] * df["vwap"].fillna(df["close"])
    df = df.sort_values("dollar_volume", ascending=False)
    return df.head(top_n)["ticker"].tolist()


def build_panel(tickers: list[str], start: date, end: date,
                progress: bool = False) -> pd.DataFrame:
    """Long-format panel of daily bars, fetched in parallel."""
    frames = []
    done = 0
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        for d in pool.map(lambda t: daily_bars(t, start, end), tickers):
            done += 1
            if progress and done % 50 == 0:
                print(f"  fetched {done}/{len(tickers)} ...", flush=True)
            if d is not None and not d.empty:
                frames.append(d)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])


def drop_rwa_tokens(panel: pd.DataFrame) -> pd.DataFrame:
    """Remove tokens that track a real-world asset rather than crypto.

    WHY THIS IS NOT A BLACKLIST, AND NOT A CORRELATION-TO-BTC FLOOR
        PAXG and XAUT (gold-backed) topped the SHORT list on the first run, for
        the uninteresting reason that gold rose less than crypto did — so their
        relative-strength-vs-BTC feature was deeply negative. They are not
        crypto shorts; they are gold, wearing a ticker.

        A BTC-correlation floor was tried first and FAILED: both correlate 0.36
        with BTC, which is inside the real-crypto range (HNT is 0.33). It would
        have excluded legitimate coins and kept these.

        What separates them cleanly is the asset they actually track. Measured
        over 400 days:
            PAXG  corr BTC 0.36 | corr GOLD 0.98
            XAUT  corr BTC 0.36 | corr GOLD 0.99
            ETH   corr BTC 0.90 | corr GOLD 0.33
            HNT   corr BTC 0.33 | corr GOLD 0.13
        So the rule is simply: if it tracks gold MORE than it tracks crypto, it
        is not a crypto asset. Self-maintaining — any future gold-backed token
        is caught without editing a list. The gold series is on the same
        subscription, so this costs one extra request.
    """
    if panel.empty or "ticker" not in panel:
        return panel
    dates = panel["date"]
    g = daily_bars(GOLD, dates.min().date(), dates.max().date())
    if g.empty:
        log.warning("gold reference unavailable - RWA filter skipped")
        return panel

    w = panel.pivot_table(index="date", columns="ticker", values="close").sort_index()
    w["__gold"] = g.set_index("date")["close"]
    r = w.pct_change()
    if BENCH not in r:
        return panel
    drop = []
    for t in panel["ticker"].unique():
        if t in (BENCH,) or t not in r:
            continue
        cb, cg = r[t].corr(r[BENCH]), r[t].corr(r["__gold"])
        if pd.notna(cg) and pd.notna(cb) and cg > cb:
            drop.append(t)
    if drop:
        log.info("RWA filter dropped: %s", drop)
    return panel[~panel["ticker"].isin(drop)]


def apply_universe_filter(panel: pd.DataFrame,
                          min_dollar_volume: float = MIN_DOLLAR_VOLUME,
                          min_vol_pct: float = MIN_ANNUAL_VOL_PCT) -> pd.DataFrame:
    """Drop illiquid pairs and pegs. Returns the panel restricted to survivors."""
    if panel.empty:
        return panel
    p = panel.copy()
    p["dollar_volume"] = p["volume"] * p["vwap"].fillna(p["close"])

    stats = p.groupby("ticker").agg(
        med_dv=("dollar_volume", "median"),
        n=("close", "size"),
    )
    ret = p.sort_values("date").groupby("ticker")["close"].pct_change()
    p["_ret"] = ret
    vol = p.groupby("ticker")["_ret"].std() * (365 ** 0.5) * 100
    stats["ann_vol"] = vol

    keep = stats[(stats["med_dv"] >= min_dollar_volume)
                 & (stats["ann_vol"] >= min_vol_pct)
                 & (stats["n"] >= 120)].index
    out = p[p["ticker"].isin(keep)].drop(columns=["_ret"])
    return drop_rwa_tokens(out)
