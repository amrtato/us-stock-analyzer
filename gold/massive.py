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

import json
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

# Typical spread on this feed. Properly measured now: 7,200 raw quotes across
# all 24 hours of 2026-10-06 (measure_spread.py), median $0.550.
#
# This supersedes an earlier 0.52 taken from 20 pulls inside ONE minute of the
# US afternoon - close by luck, but a snapshot rather than a distribution, and
# spreads here are strongly session-dependent (Asian open runs a $0.715 median
# against London's $0.500).
#
# It is now also the COST BASIS in research.SPREAD_MEDIAN. Previously the page
# displayed this feed's bid/ask while costing trades at a different broker's
# $0.20 - two different instruments on one card.
SPREAD_TYPICAL = 0.550
SPREAD_SAMPLE_NOTE = "7,200 quotes across 24h, 2026-10-06"


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


# ─────────────────────────────────────────────────────────────────────────────
# Live tick stream
#
# WHAT THIS BUYS, AND WHAT IT DOES NOT
#     REST last_quote refreshes about every 2s and the UI polls on a timer, so a
#     printed price could be up to 5s old. This holds the WebSocket open in a
#     background thread and keeps the newest tick in memory, cutting staleness to
#     well under a second. Measured on this account: 1.6 ticks/sec on C.XAU/USD.
#
#     It does NOT make the card repaint per tick. Streamlit renders on a timer;
#     the browser has no push subscription. Per-tick repainting would need the
#     socket opened in the BROWSER, which means shipping this API key to every
#     visitor of a public site. That is why it is done here instead.
#
# ONE CONNECTION PER PROCESS
#     Streamlit re-runs the script constantly. Without the lock and the module
#     global below, every rerun would open another socket and the account would
#     be throttled for self-inflicted reasons.
# ─────────────────────────────────────────────────────────────────────────────
import threading
import time

WS_URL = "wss://socket.polygon.io/forex"
WS_SYMBOL = "C.XAU/USD"
# Past this age, while the market is open, the stream is treated as broken and
# callers fall back to REST. Generous versus 1.6 ticks/sec so a normal lull in
# liquidity does not trigger a pointless failover.
STREAM_STALE_SECONDS = 15.0

_stream = None
_stream_lock = threading.Lock()


class TickStream:
    """Background XAU/USD tick reader. Never raises into the caller."""

    def __init__(self, key: str, symbol: str = WS_SYMBOL):
        self._key = key
        self._symbol = symbol
        self._lock = threading.Lock()
        self._tick: dict | None = None
        self._ticks = 0
        self._reconnects = 0
        self._error: str | None = None
        self._connected = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="xauusd-ticks", daemon=True)

    # -- public ---------------------------------------------------------------
    def start(self) -> "TickStream":
        if not self._thread.is_alive():
            self._thread.start()
        return self

    def latest(self) -> dict | None:
        """Newest tick, or None when nothing has arrived yet."""
        with self._lock:
            return dict(self._tick) if self._tick else None

    def health(self) -> dict:
        with self._lock:
            age = (time.time() - self._tick["mono"]) if self._tick else None
            return {
                "connected": self._connected,
                "ticks": self._ticks,
                "reconnects": self._reconnects,
                "age": age,
                "error": self._error,
            }

    def stop(self) -> None:
        self._stop.set()

    # -- internals ------------------------------------------------------------
    def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                import websocket  # noqa: PLC0415 - optional dependency
            except Exception as exc:
                with self._lock:
                    self._error = f"websocket-client missing: {exc}"
                return                                    # no point retrying
            try:
                ws = websocket.create_connection(WS_URL, timeout=20)
                ws.send(json.dumps({"action": "auth", "params": self._key}))
                ws.send(json.dumps({"action": "subscribe", "params": self._symbol}))
                with self._lock:
                    self._connected, self._error = True, None
                backoff = 1.0                             # a good connect resets it
                self._pump(ws)
            except Exception as exc:
                with self._lock:
                    self._error = str(exc)[:200]
            finally:
                with self._lock:
                    self._connected = False
                try:
                    ws.close()
                except Exception:
                    pass
            if self._stop.is_set():
                return
            with self._lock:
                self._reconnects += 1
            # Capped backoff. The socket also drops on the weekend close, and
            # hammering a shut market helps nobody.
            self._stop.wait(backoff)
            backoff = min(backoff * 2, 60.0)

    def _pump(self, ws) -> None:
        import websocket                               # noqa: PLC0415
        while not self._stop.is_set():
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                # No data within the socket timeout. That is NORMAL: gold is shut
                # from Friday 17:00 to Sunday 18:00 ET, and a quiet market is not
                # a broken one. Treating it as a disconnect would reconnect every
                # ~20s all weekend - thousands of pointless handshakes - and reset
                # the backoff each time so it never backed off at all.
                continue
            if not raw:
                return
            try:
                msgs = json.loads(raw)
            except Exception:
                continue
            for m in msgs if isinstance(msgs, list) else [msgs]:
                if m.get("ev") != "C":
                    continue
                bid, ask = m.get("b"), m.get("a")
                if not bid or not ask or bid <= 0 or ask <= 0:
                    continue
                ts = m.get("t")
                with self._lock:
                    self._ticks += 1
                    self._tick = {
                        "bid": float(bid),
                        "ask": float(ask),
                        "price": (float(bid) + float(ask)) / 2,
                        "spread": float(ask) - float(bid),
                        "ts": (datetime.fromtimestamp(ts / 1000, tz=ET)
                               if ts else datetime.now(ET)),
                        # Monotonic clock for age: wall-clock can jump and would
                        # make a healthy stream look stale (or vice versa).
                        "mono": time.time(),
                    }


def get_stream() -> TickStream | None:
    """Process-wide singleton. Safe to call on every Streamlit rerun."""
    global _stream
    key = _key()
    if not key:
        return None
    with _stream_lock:
        if _stream is None:
            _stream = TickStream(key).start()
        return _stream


def stream_quote() -> dict | None:
    """Newest streamed tick, or None when absent/stale so callers fall back."""
    s = get_stream()
    if s is None:
        return None
    t = s.latest()
    if not t:
        return None
    age = time.time() - t["mono"]
    # While the market is shut there are no ticks by definition; the last traded
    # quote is the correct thing to show, so staleness is only a fault when open.
    from gold.live import market_status                   # noqa: PLC0415 - cycle
    if age > STREAM_STALE_SECONDS and market_status()["open"]:
        return None
    return {
        "bid": t["bid"], "ask": t["ask"], "price": t["price"],
        "spread": t["spread"],
        "spread_x": round(t["spread"] / SPREAD_TYPICAL, 2) if SPREAD_TYPICAL else None,
        "ts": t["ts"], "age": age,
    }
