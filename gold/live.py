"""
Live gold data for the dashboard, and the forward paper record.

DESIGN NOTE — no state.
    The paper record is RECOMPUTED from price history each time rather than
    appended to a log. The 16:00 ET rule is deterministic, so replaying it over
    bars since the paper-test start date reproduces the record exactly. A log
    file would add a persistence layer that can desync, double-count on restart,
    or be lost when the App Service recycles — for no benefit.

    Consequence worth knowing: the record only extends as far back as yfinance
    serves hourly bars (~2 years), which comfortably covers a paper test begun
    2026-08-07.

SPREAD, AND WHY TWO NUMBERS ARE SHOWN
    Massive quotes a live interbank bid/ask (~$0.52 typical). That is NOT what
    you pay: your Exness account measured a $0.20 median. So the COST model
    keeps using the broker-measured constant from gold/research.py, while the
    live spread is reported as a market-condition RATIO (live / typical). When
    the interbank spread trebles, yours widens too — that is the actionable
    part, not the absolute figure.
"""
from __future__ import annotations
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from gold import massive
from gold.research import (SPREAD_MEDIAN, CONTRACT_OZ, VOL_PROFILE,
                           expected_move)

ET = ZoneInfo("America/New_York")
SYMBOL = "GC=F"


def fetch_hourly(period: str = "2y") -> pd.DataFrame:
    """Hourly gold bars, indexed in New York time — SPOT when obtainable.

    Massive first. The yfinance GC=F fallback is FUTURES: it tracks spot almost
    perfectly in hourly CHANGES (corr 0.9979) so trend and ATR survive, but it
    sat ~$62 above spot when measured, so absolute levels taken from it are not
    fillable. Callers that print price levels should check `is_spot_history()`.
    """
    if massive.available():
        d = massive.fetch_hourly_spot()
        if d is not None and not d.empty:
            return d

    d = yf.Ticker(SYMBOL).history(period=period, interval="1h")
    if d is None or d.empty:
        return pd.DataFrame()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d[["Open", "High", "Low", "Close", "Volume"]].dropna()
    idx = pd.to_datetime(d.index)
    d.index = idx.tz_convert(ET) if idx.tz is not None else idx.tz_localize("UTC").tz_convert(ET)
    return d


def atr(d: pd.DataFrame, n: int = 24) -> pd.Series:
    h, l, c = d["High"], d["Low"], d["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def current_state(d: pd.DataFrame) -> dict:
    """Live price plus the risk profile for the hour we are actually in."""
    now = datetime.now(ET)
    hour = now.hour
    halted = hour not in VOL_PROFILE          # 17:00 ET settlement halt
    ref_hour = 18 if halted else hour
    x = VOL_PROFILE[ref_hour]

    price = live_atr = chg = exp_move = None
    if not d.empty:
        price = float(d["Close"].iloc[-1])
        a = atr(d)
        live_atr = float(a.iloc[-1]) if len(a) else None
        if len(d) > 24:
            chg = price / float(d["Close"].iloc[-25]) - 1
        exp_move = expected_move(price, ref_hour)

    # Stop sized from LIVE ATR when available — the seasonal multiple tells you
    # how this hour compares, live ATR tells you what today actually looks like.
    stop = 1.5 * live_atr if live_atr else (1.5 * exp_move if exp_move else None)
    return {
        "now": now, "hour": hour, "ref_hour": ref_hour, "halted": halted,
        "vol_x": x, "expected_move": exp_move,
        "price": price, "live_atr": live_atr, "chg_24h": chg,
        "suggested_stop": stop, "spread": SPREAD_MEDIAN,
    }


MT5_SYMBOL = "XAUUSDm"
MT5_PATHS = (
    r"C:\Program Files\MetaTrader 5 (AMR)\terminal64.exe",
    r"C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe",
)


def get_spot() -> dict:
    """
    True XAU/USD spot with live bid/ask where obtainable, clearly labelled
    futures where not.

    ORDER, AND WHY
      1. Massive (Polygon) XAU/USD — real bid/ask refreshing ~every 2s. Quotes
         the metal directly, so there is no proxy and no offset to drift. Works
         from Azure. This is the default EVERYWHERE, including locally, so the
         dashboard you develop against shows the same number as the deployed one.
      2. gold-api.com — keyless, ~30s cadence, measured +$2.24 vs Massive spot
         on 2026-10-01. Mid only, no bid/ask. Good enough to keep the card alive.
      3. yfinance GC=F — FUTURES. Measured +$62.36 (+1.44%) against spot, and the
         basis drifts with rates and expiry (a fitted constant still left $19.50
         mean error over 30 days), so it is never called spot. Hourly CHANGES
         still correlate 0.9979, so structure survives even when the level does not.

    MT5 is deliberately NOT in this chain any more. It only ever worked on
    Windows, and having local read a different source from production is exactly
    how a dashboard starts lying about what production shows. Set
    GOLD_USE_MT5=1 to put your broker's own tick back at the front for
    comparison; it stays off by default.

    PAXG was removed. Binance geo-blocks Ontario (the app runs in Azure Canada
    Central) and the tokenised-ounce premium drifted to $13.91 against a
    calibrated p95 of $4.33.

    The returned dict always carries `is_spot` and `source`; the UI keys its
    colour and warnings off those rather than assuming.
    """
    import os

    if os.getenv("GOLD_USE_MT5") == "1":
        q = _mt5_tick()
        if q:
            return q

    # 1. Massive — the metal itself, with a real two-sided quote.
    if massive.available():
        try:
            q = massive.last_quote()
            if q:
                return {
                    "price": q["price"], "bid": q["bid"], "ask": q["ask"],
                    "spread": q["spread"], "spread_x": q.get("spread_x"),
                    "is_spot": True, "two_sided": True,
                    "source": "Massive XAU/USD — spot, live bid/ask",
                    "ts": q["ts"],
                }
        except Exception:
            pass

    # 2. gold-api.com — keyless mid. No bid/ask, so cost falls back to the
    #    broker-measured median rather than pretending to observe a spread.
    try:
        r = requests.get("https://api.gold-api.com/price/XAU", timeout=8)
        if r.status_code == 200:
            v = r.json().get("price")
            if v and float(v) > 0:
                return {
                    "price": float(v), "bid": None, "ask": None,
                    "spread": SPREAD_MEDIAN, "is_spot": True, "two_sided": False,
                    "source": "gold-api.com — spot mid (~30s, no bid/ask)",
                    "ts": datetime.now(ET),
                }
    except Exception:
        pass

    # 3. Futures — last resort, never presented as spot.
    try:
        q = yf.Ticker(SYMBOL).history(period="1d", interval="1m")
        if q is not None and not q.empty:
            return {
                "price": float(q["Close"].iloc[-1]), "bid": None, "ask": None,
                "spread": SPREAD_MEDIAN, "is_spot": False, "two_sided": False,
                "source": "yfinance GC=F — FUTURES, not spot",
                "ts": q.index[-1].tz_convert(ET),
            }
    except Exception:
        pass
    return {"price": None, "is_spot": False, "two_sided": False,
            "source": "unavailable", "ts": None}


def _mt5_tick() -> dict | None:
    """Opt-in only (GOLD_USE_MT5=1). Windows-only; absent on Azure Linux."""
    try:
        import MetaTrader5 as mt5                      # noqa: PLC0415
    except Exception:
        return None
    for path in MT5_PATHS:
        try:
            if not mt5.initialize(path=path, timeout=30_000):
                continue
            mt5.symbol_select(MT5_SYMBOL, True)
            t = mt5.symbol_info_tick(MT5_SYMBOL)
            mt5.shutdown()
            if t and t.bid > 0 and t.ask > 0:
                return {
                    "price": (t.bid + t.ask) / 2, "bid": t.bid, "ask": t.ask,
                    "spread": t.ask - t.bid, "is_spot": True, "two_sided": True,
                    "source": f"MT5 {MT5_SYMBOL} — your broker's tick",
                    "ts": datetime.fromtimestamp(t.time, tz=ET),
                }
        except Exception:
            try:
                mt5.shutdown()
            except Exception:
                pass
    return None


def is_spot_history() -> bool:
    """True when fetch_hourly() is returning real spot rather than GC=F futures."""
    return massive.available()


def market_status(now: datetime | None = None) -> dict:
    """
    Is the gold market actually open right now?

    Without this, a frozen price is indistinguishable from a broken feed — the
    single most confusing state a live dashboard can show. XAU/USD trades
    Sunday 18:00 ET through Friday 17:00 ET, with a daily halt 17:00-18:00 ET.
    """
    now = now or datetime.now(ET)
    dow, hour = now.weekday(), now.hour          # Mon=0 .. Sun=6

    if dow == 4 and hour >= 17:                  # Friday after the close
        opens = (now + timedelta(days=2)).replace(hour=18, minute=0, second=0, microsecond=0)
        return {"open": False, "reason": "weekend break, since Friday 17:00 ET",
                "reopens": opens}
    if dow == 5:                                 # Saturday
        opens = (now + timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)
        return {"open": False, "reason": "weekend break", "reopens": opens}
    if dow == 6 and hour < 18:                   # Sunday before the open
        opens = now.replace(hour=18, minute=0, second=0, microsecond=0)
        return {"open": False, "reason": "weekend break, reopening this evening",
                "reopens": opens}
    if hour == 17:                               # daily settlement halt
        return {"open": False, "reason": "daily settlement halt (17:00-18:00 ET)",
                "reopens": now.replace(hour=18, minute=0, second=0, microsecond=0)}
    return {"open": True, "reason": "", "reopens": None}


def signal_card(d: pd.DataFrame) -> dict:
    """
    Trade levels for the current bar.

    EVERY PARAMETER HERE CAME OUT OF THE TESTING, not convention:

    * DIRECTION from price vs EMA200. This is the only input that mattered
      consistently — in the matched control the EMA200 filter accounted for
      essentially all of the apparent edge, and the specific hour for almost
      none of it. So the card states a trend BIAS, not a timing signal.

    * STOP at 1.5 x ATR. The stop grid was monotonic: 0.25x ATR gave profit
      factor 0.96, 0.35x gave 0.93, 0.50x gave 0.98 — all losing — while 1.5x
      and wider were profitable. Gold's hourly noise is 20-45x the size of any
      drift found, so a tight stop harvests noise against you. A tighter stop
      is available below, clearly marked as tested-and-worse.

    * TARGETS at 2R and 3R. In the expectancy sweep, 3R returned +0.43R per
      trade against +0.27R at 2R and NEGATIVE expectancy at 0.25-0.5R, even
      though the tight targets won 73-81% of the time. Win rate and expectancy
      move in opposite directions here.

    Warnings are raised for the two conditions that produced the worst losses in
    testing: Friday entries (weekend gap) and the 17:00 ET settlement halt.
    """
    now = datetime.now(ET)
    if d.empty or len(d) < 210:
        return {"ok": False, "reason": "insufficient price history"}

    c = d["Close"]
    price = float(c.iloc[-1])
    ema200 = float(c.ewm(span=200, adjust=False).mean().iloc[-1])
    ema50 = float(c.ewm(span=50, adjust=False).mean().iloc[-1])
    a = float(atr(d).iloc[-1])

    if price > ema200 and price > ema50:
        bias, dirn, icon = "BULLISH", +1, "🟢"
    elif price < ema200 and price < ema50:
        bias, dirn, icon = "BEARISH", -1, "🔴"
    else:
        bias, dirn, icon = "MIXED", +1 if price > ema200 else -1, "🟡"

    risk = 1.5 * a
    stop = price - dirn * risk
    tp1 = price + dirn * 2.0 * risk
    tp2 = price + dirn * 3.0 * risk

    hour = now.hour
    warnings = []
    if now.weekday() == 4:
        warnings.append(
            "Friday — any position held into the close carries weekend gap risk. "
            "Friday entries produced 6 of the 10 worst losses in testing and no "
            "measurable edge (p=0.59)."
        )
    if hour == 17:
        warnings.append("17:00 ET settlement halt — no stop can fill until 18:00 ET.")
    if hour == 16:
        warnings.append(
            "16:00 ET — the next bar is 18:00 ET, spanning the settlement halt. "
            "A stop cannot protect a position held across it."
        )
    if tp1 - price != 0 and abs(tp1 - price) < SPREAD_MEDIAN / 0.05:
        warnings.append(
            f"Target 1 is under ${SPREAD_MEDIAN/0.05:.2f} — spread would take "
            f"more than 5% of the move."
        )

    vol_mult = VOL_PROFILE.get(hour, VOL_PROFILE.get(18, 1.0))
    return {
        "ok": True, "now": now, "price": price, "bias": bias, "dirn": dirn,
        "icon": icon, "ema50": ema50, "ema200": ema200, "atr": a,
        "entry": price, "stop": stop, "tp1": tp1, "tp2": tp2,
        "risk_usd": risk, "vol_mult": vol_mult,
        "tight_stop": price - dirn * 0.5 * a,       # shown as tested-and-worse
        "spread": SPREAD_MEDIAN, "warnings": warnings,
        "pct_from_ema200": price / ema200 - 1,
    }


def paper_record(d: pd.DataFrame, start: str) -> dict:
    """
    Replay the 16:00 ET rule over every session since `start`.

    Returns the trade list plus summary. Deliberately reports the losing side —
    worst trade, longest losing run, drawdown — because those decide whether a
    rule is executable, and they are what a signal feed usually omits.
    """
    empty = {"n": 0, "trades": pd.DataFrame(), "win": None, "net_oz": 0.0,
             "net_lot": 0.0, "maxdd_lot": 0.0, "streak": 0, "start": start}
    if d.empty:
        return empty
    since = pd.Timestamp(start, tz=ET)
    f = d[d.index >= since]
    if f.empty:
        return empty

    f = f.reset_index().rename(columns={f.reset_index().columns[0]: "ts"})
    f["hour"] = f["ts"].dt.hour
    f["dow"] = f["ts"].dt.dayofweek
    # Monday-Thursday only. Friday entries hold across the weekend (median 50h
    # vs 2h), contribute no measurable edge (p=0.59), and produced 6 of the 10
    # worst losses in the backtest — including a -$97.98 gap no stop could catch.
    idx = f.index[(f["hour"] == 16) & (f["dow"] <= 3)].to_numpy()
    idx = idx[idx < len(f) - 1]
    if len(idx) == 0:
        return empty

    t = pd.DataFrame({
        "date": f["ts"].to_numpy()[idx],
        "entry": f["Close"].to_numpy()[idx],
        "exit": f["Close"].to_numpy()[idx + 1],
    })
    t["net_oz"] = (t["exit"] - t["entry"]) - SPREAD_MEDIAN
    t["net_lot"] = t["net_oz"] * CONTRACT_OZ
    t["win"] = t["net_oz"] > 0

    eq = np.cumsum(t["net_lot"].to_numpy())
    dd = eq - np.maximum.accumulate(eq)
    run = best = 0
    for w in t["win"]:
        run = 0 if w else run + 1
        best = max(best, run)

    return {
        "n": len(t), "trades": t, "win": float(t["win"].mean()),
        "net_oz": float(t["net_oz"].sum()), "net_lot": float(eq[-1]),
        "maxdd_lot": float(dd.min()), "streak": int(best), "start": start,
    }
