"""
Market data fetcher using yfinance 1.2.0+.

Yahoo Finance has blocked yf.Ticker().info (quoteSummary endpoint) for
free/unauthenticated users.  We replace it entirely with three free endpoints:
  - fast_info          → price, market cap, 52-week range, moving averages
  - income_stmt        → revenue, margins, EPS, EPS growth
  - balance_sheet      → equity, debt, assets, current ratio
  - analyst_price_targets → mean/high/low analyst target prices

All fundamental ratios are derived from these sources, giving us identical
(or better) data without touching the blocked endpoint.
"""
import time
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings
from typing import Optional

warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np

from config import normalize_sector

log = logging.getLogger(__name__)

# ── Concurrency limit for yfinance .info-class calls ──────────────────────────
# Each fetch_fundamentals makes 4 HTTP calls (fast_info, income_stmt,
# balance_sheet, analyst_price_targets).
#
# This was Semaphore(1) because concurrent calls used to return "Invalid Crumb"
# 401s. RE-MEASURED 2026-10-01 on 20 tickers, counting how many came back with a
# real sector (only a successful call can supply one):
#     concurrency 1 -> 32.3s, 20/20 good
#     concurrency 3 -> 12.5s, 20/20 good   (2.6x)
#     concurrency 6 -> 10.4s, 20/20 good   (3.1x)
# No degradation at any level, so the serialisation was costing ~150s of the
# ~169s cold start for a hazard that no longer reproduces.
#
# Settled on 4: past ~3 the curve flattens, and a failure here is silent (a
# blocked call returns defaults rather than raising), so there is no reason to
# sit near the edge for a few seconds. Override with YF_FUNDAMENTALS_CONCURRENCY
# if Yahoo tightens again — a dial beats a redeploy.
#
# NOTE: measured against the locally installed yfinance; requirements.txt pins an
# older one for Azure. If sectors start coming back "Unknown" in production, set
# that env var to 1 first, then investigate.
_YF_CONCURRENCY = max(1, int(os.getenv("YF_FUNDAMENTALS_CONCURRENCY", "4")))
_YF_SEM = threading.Semaphore(_YF_CONCURRENCY)

# ── In-memory cache (4-hour TTL) ───────────────────────────────────────────────
_FUND_CACHE:      dict = {}
_FUND_CACHE_TIME: dict = {}
FUND_CACHE_TTL = 4 * 3600

# ── In-memory price-history cache (30-min TTL, matches Streamlit cache) ────────
# Keyed by (ticker, period) — NOT ticker alone.
#
# It was ticker-only, and that silently served the wrong window: the Daily
# Rankings tab prefetches "6mo" first, so a later request for "1y" found a
# fresh entry under the same key and got six months back while believing it had
# a year. Nothing errored; the Short Candidates screen simply scored EMA200 on
# ~126 unconverged bars. A cache key must contain everything that changes the
# value, or it is not a cache, it is a bug with a hit rate.
_PRICE_CACHE:      dict = {}
_PRICE_CACHE_TIME: dict = {}
PRICE_CACHE_TTL = 30 * 60    # 30 minutes


# ── Helpers ────────────────────────────────────────────────────────────────────

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse MultiIndex columns from yfinance into simple field names."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [col[0] for col in df.columns]
    return df


def _row(df: pd.DataFrame, *names) -> Optional[float]:
    """Extract the most-recent (first column) value of the first matching row."""
    for name in names:
        if name in df.index:
            val = df.loc[name].iloc[0]
            try:
                v = float(val)
                if not (v != v):   # not NaN
                    return v
            except Exception:
                pass
    return None


def _pct(a, b) -> Optional[float]:
    """Safe percentage change: (a - b) / |b|."""
    try:
        if b and b != 0:
            return (a - b) / abs(b)
    except Exception:
        pass
    return None


def _safe(val, default=None):
    """Return val unless it is None / NaN / inf."""
    try:
        if val is None:
            return default
        f = float(val)
        if f != f or abs(f) == float("inf"):
            return default
        return f
    except Exception:
        return default


# ── Price history ───────────────────────────────────────────────────────────────

def fetch_price_history(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    # ── In-memory cache hit (populated by prefetch_price_histories) ─────────────
    if interval == "1d":
        now = time.monotonic()
        ck = (ticker, period)
        if ck in _PRICE_CACHE and (now - _PRICE_CACHE_TIME.get(ck, 0)) < PRICE_CACHE_TTL:
            return _PRICE_CACHE[ck]

    for attempt in range(3):
        try:
            with _YF_SEM:
                time.sleep(0.1)
                df = yf.download(ticker, period=period, interval=interval,
                                 auto_adjust=True, progress=False)
            result = _flatten(df) if not df.empty else df
            # Populate price cache so repeated individual calls are instant
            if interval == "1d" and not result.empty:
                _PRICE_CACHE[(ticker, period)]      = result
                _PRICE_CACHE_TIME[(ticker, period)] = time.monotonic()
            return result
        except RuntimeError as exc:
            if "dictionary changed size" in str(exc) and attempt < 2:
                time.sleep(0.5 * (attempt + 1))
                continue
            log.error("Price fetch failed for %s: %s", ticker, exc)
            return pd.DataFrame()
        except Exception as exc:
            log.error("Price fetch failed for %s: %s", ticker, exc)
            return pd.DataFrame()
    return pd.DataFrame()


def prefetch_price_histories(tickers: list, period: str = "6mo") -> int:
    """Batch-download price history for all tickers in a SINGLE yfinance call.

    Populates _PRICE_CACHE so that subsequent fetch_price_history() calls
    (running inside ThreadPoolExecutor) become instant cache hits instead of
    N serialised HTTP calls through _YF_SEM.

    Returns the number of tickers successfully cached.
    """
    if not tickers:
        return 0
    now = time.monotonic()
    to_fetch = [
        t for t in tickers
        if (t, period) not in _PRICE_CACHE
        or (now - _PRICE_CACHE_TIME.get((t, period), 0)) >= PRICE_CACHE_TTL
    ]
    if not to_fetch:
        return 0

    for attempt in range(3):
        try:
            with _YF_SEM:
                time.sleep(0.1)
                if len(to_fetch) == 1:
                    raw    = yf.download(to_fetch[0], period=period, interval="1d",
                                         auto_adjust=True, progress=False)
                    result = _flatten(raw)
                    if not result.empty:
                        ts = time.monotonic()
                        _PRICE_CACHE[(to_fetch[0], period)]      = result
                        _PRICE_CACHE_TIME[(to_fetch[0], period)] = ts
                    return 1 if not result.empty else 0
                else:
                    raw = yf.download(list(to_fetch), period=period, interval="1d",
                                      auto_adjust=True, group_by="ticker", progress=False)

            if raw is None or raw.empty:
                return 0
            count = 0
            ts    = time.monotonic()
            for ticker in list(to_fetch):
                try:
                    df = _flatten(raw[ticker].dropna())
                    if not df.empty:
                        _PRICE_CACHE[(ticker, period)]      = df
                        _PRICE_CACHE_TIME[(ticker, period)] = ts
                        count += 1
                except Exception as e:
                    log.debug("Prefetch parse error %s: %s", ticker, e)
            return count

        except RuntimeError as exc:
            if "dictionary changed size" in str(exc) and attempt < 2:
                time.sleep(0.5 * (attempt + 1))
                continue
            log.error("prefetch_price_histories failed: %s", exc)
            return 0
        except Exception as exc:
            log.error("prefetch_price_histories failed: %s", exc)
            return 0
    return 0


def fetch_intraday(ticker: str, interval: str = "5m") -> pd.DataFrame:
    for attempt in range(3):
        try:
            with _YF_SEM:
                time.sleep(0.1)
                df = yf.download(ticker, period="5d", interval=interval,
                                 auto_adjust=True, progress=False)
            return _flatten(df)
        except RuntimeError as exc:
            if "dictionary changed size" in str(exc) and attempt < 2:
                time.sleep(0.5 * (attempt + 1))
                continue
            log.error("Intraday fetch failed for %s: %s", ticker, exc)
            return pd.DataFrame()
        except Exception as exc:
            log.error("Intraday fetch failed for %s: %s", ticker, exc)
            return pd.DataFrame()
    return pd.DataFrame()


# ── Batch quotes ────────────────────────────────────────────────────────────────

def fetch_batch_quotes(tickers: list) -> dict:
    """One call for all tickers → {ticker: {price, change_pct, volume, ...}}"""
    results = {}
    if not tickers:
        return results
    for attempt in range(3):
        try:
            with _YF_SEM:
                time.sleep(0.1)
                data = yf.download(list(tickers), period="2d", interval="1d",
                                   auto_adjust=True, group_by="ticker", progress=False)
            if data.empty:
                return results
            for ticker in list(tickers):  # iterate a copy to avoid mutation issues
                try:
                    df = data[ticker].dropna()
                    if df.empty or len(df) < 2:
                        continue
                    today, prev = df.iloc[-1], df.iloc[-2]
                    chg = (float(today["Close"]) - float(prev["Close"])) / float(prev["Close"]) * 100
                    results[ticker] = {
                        "price":      float(today["Close"]),
                        "open":       float(today["Open"]),
                        "high":       float(today["High"]),
                        "low":        float(today["Low"]),
                        "volume":     float(today["Volume"]),
                        "change_pct": float(chg),
                        "prev_close": float(prev["Close"]),
                    }
                except Exception as e:
                    log.debug("Quote parse failed for %s: %s", ticker, e)
            return results
        except RuntimeError as exc:
            if "dictionary changed size" in str(exc) and attempt < 2:
                time.sleep(0.5 * (attempt + 1))
                continue
            log.error("Batch quote failed: %s", exc)
            return results
        except Exception as exc:
            log.error("Batch quote failed: %s", exc)
            return results
    return results


# ── Fundamentals (no .info — uses free endpoints only) ─────────────────────────

def fetch_fundamentals(ticker: str) -> dict:
    """
    Build a complete fundamentals dict without using yf.Ticker().info
    (that endpoint is blocked by Yahoo Finance for free users).

    Sources used:
      fast_info             → price, mkt cap, 52w range, moving averages
      income_stmt           → revenue, margins, EPS, growth
      balance_sheet         → equity, debt, ratios
      analyst_price_targets → analyst mean/high/low target
    """
    now = time.monotonic()
    if ticker in _FUND_CACHE and (now - _FUND_CACHE_TIME.get(ticker, 0)) < FUND_CACHE_TTL:
        return _FUND_CACHE[ticker]

    result: dict = {}

    with _YF_SEM:
        time.sleep(0.15)   # 150 ms gap between tickers → ~7 req/s max to Yahoo
        try:
            t = yf.Ticker(ticker)

            # ── 1. fast_info (always works, no auth) ─────────────────────────
            fi = t.fast_info
            price      = _safe(fi.last_price)
            mkt_cap    = _safe(fi.market_cap)
            shares     = _safe(fi.shares)
            yr_high    = _safe(fi.year_high)
            yr_low     = _safe(fi.year_low)
            ma50       = _safe(fi.fifty_day_average)
            ma200      = _safe(fi.two_hundred_day_average)
            avg_vol_3m = _safe(fi.three_month_average_volume)

            result.update({
                "market_cap":  mkt_cap,
                "52w_high":    yr_high,
                "52w_low":     yr_low,
                "50d_avg":     ma50,
                "200d_avg":    ma200,
                "avg_volume":  avg_vol_3m,
                "currency":    getattr(fi, "currency", "USD"),
            })

            # ── 2. Income statement (annual) ──────────────────────────────────
            try:
                inc = t.income_stmt          # columns = fiscal years, newest first
                if inc is not None and not inc.empty:
                    rev0 = _row(inc, "Total Revenue", "Operating Revenue")
                    rev1 = _row(inc, "Total Revenue", "Operating Revenue") if inc.shape[1] < 2 else None
                    if inc.shape[1] >= 2:
                        rev1 = float(inc.loc["Total Revenue"].iloc[1]) if "Total Revenue" in inc.index else None

                    gp   = _row(inc, "Gross Profit")
                    oi   = _row(inc, "Operating Income", "Total Operating Income As Reported")
                    ni0  = _row(inc, "Net Income", "Net Income Common Stockholders")
                    ni1  = None
                    if inc.shape[1] >= 2:
                        try:
                            ni1 = float(inc.loc["Net Income"].iloc[1]) if "Net Income" in inc.index else None
                        except Exception:
                            pass

                    ebitda   = _row(inc, "EBITDA", "Normalized EBITDA")
                    eps_dil  = _row(inc, "Diluted EPS")
                    eps_basic = _row(inc, "Basic EPS")
                    eps      = eps_dil or eps_basic

                    # Margins = simple ratio (ni / rev), NOT _pct which is a change fn
                    def _ratio(a, b):
                        try:
                            return a / b if a is not None and b and b != 0 else None
                        except Exception:
                            return None

                    profit_margin  = _ratio(ni0, rev0)
                    op_margin      = _ratio(oi,  rev0)
                    gross_margin   = _ratio(gp,  rev0)
                    # Growth = YoY change — _pct is correct here
                    revenue_growth = _pct(rev0, rev1)  if rev0 and rev1 else None
                    earn_growth    = _pct(ni0,  ni1)   if ni0  and ni1  else None

                    # P/E from EPS
                    pe = (price / eps) if price and eps and eps > 0 else None
                    # P/S
                    ps = (mkt_cap / rev0) if mkt_cap and rev0 and rev0 > 0 else None
                    # EV/EBITDA (rough — needs debt/cash)
                    ev_ebitda = None   # refined below after balance sheet

                    result.update({
                        "pe_ratio":        pe,
                        "price_to_sales":  ps,
                        "profit_margin":   profit_margin,
                        "operating_margin": op_margin,
                        "gross_margin":    gross_margin,
                        "revenue_growth":  revenue_growth,
                        "earnings_growth": earn_growth,
                        "eps_trailing":    eps,
                        "ebitda":          ebitda,
                        "_revenue":        rev0,
                        "_net_income":     ni0,
                    })
            except Exception as exc:
                log.debug("Income stmt failed for %s: %s", ticker, exc)

            # ── 3. Balance sheet ──────────────────────────────────────────────
            try:
                bs = t.balance_sheet
                if bs is not None and not bs.empty:
                    equity     = _row(bs, "Stockholders Equity", "Common Stock Equity",
                                         "Total Equity Gross Minority Interest")
                    total_debt = _row(bs, "Total Debt")
                    total_assets = _row(bs, "Total Assets")
                    cash       = _row(bs, "Cash And Cash Equivalents",
                                         "Cash Cash Equivalents And Short Term Investments")
                    cur_assets  = _row(bs, "Current Assets")
                    cur_liab    = _row(bs, "Current Liabilities")
                    inventory   = _row(bs, "Inventory")

                    ni = result.get("_net_income")
                    # All of these are simple ratios, not YoY changes
                    de_ratio = (total_debt / equity)    if total_debt and equity and equity != 0 else None
                    roe      = (ni / equity)             if ni and equity and equity != 0 else None
                    roa      = (ni / total_assets)       if ni and total_assets and total_assets != 0 else None
                    cur_ratio   = (cur_assets / cur_liab) if cur_assets and cur_liab and cur_liab != 0 else None
                    quick_ratio = ((cur_assets - (inventory or 0)) / cur_liab) if cur_assets and cur_liab and cur_liab != 0 else None

                    # EV/EBITDA
                    ebitda = result.get("ebitda")
                    if mkt_cap and total_debt and cash and ebitda and ebitda != 0:
                        ev = mkt_cap + total_debt - cash
                        ev_ebitda = ev / ebitda

                    # Price/Book
                    pb = (price * (shares or 0)) / equity if price and shares and equity and equity > 0 else None

                    result.update({
                        "roe":            roe,
                        "roa":            roa,
                        "debt_to_equity": de_ratio,
                        "current_ratio":  cur_ratio,
                        "quick_ratio":    quick_ratio,
                        "ev_to_ebitda":   ev_ebitda,
                        "price_to_book":  pb,
                    })
            except Exception as exc:
                log.debug("Balance sheet failed for %s: %s", ticker, exc)

            # ── 4. Analyst targets ────────────────────────────────────────────
            try:
                apt = t.analyst_price_targets
                if apt and isinstance(apt, dict):
                    target_mean = _safe(apt.get("mean"))
                    # Convert mean target to a 1-5 "recommendation" proxy
                    # 1 = strong buy (target >> price), 5 = sell (target << price)
                    rec = None
                    if target_mean and price:
                        upside = (target_mean - price) / price
                        if   upside >  0.20: rec = 1.5   # strong buy
                        elif upside >  0.08: rec = 2.0   # buy
                        elif upside >  0.02: rec = 2.5   # moderate buy
                        elif upside > -0.05: rec = 3.0   # hold
                        elif upside > -0.15: rec = 3.5   # underperform
                        else:                rec = 4.5   # sell

                    result.update({
                        "target_price":  target_mean,
                        "target_high":   _safe(apt.get("high")),
                        "target_low":    _safe(apt.get("low")),
                        "recommendation": rec,
                    })
            except Exception as exc:
                log.debug("Analyst targets failed for %s: %s", ticker, exc)

            # ── 5. Company metadata ───────────────────────────────────────────
            # fast_info.quote_type is the *instrument* type ("EQUITY"), NOT the
            # GICS sector — reading it as one made every stock report
            # sector="EQUITY", which silently disabled the sector-rotation
            # sub-score, the per-sector P/E benchmark, and the recession /
            # geopolitical macro flags. basic_info has no sector or long_name
            # either (it is a fast_info shim), so both fell through to defaults.
            # .info is the only source that carries them; it is a heavier call
            # but this whole function is cached per ticker for FUND_CACHE_HRS.
            result.update({
                "sector":        "Unknown",
                "industry":      "Unknown",
                "company_name":  ticker,
                "beta":          None,
                "analyst_count": 0,
            })
            try:
                info = t.info or {}
                result["company_name"] = (
                    info.get("longName") or info.get("shortName") or ticker
                )
                result["sector"]   = normalize_sector(info.get("sector"))
                result["industry"] = info.get("industry") or "Unknown"
                # beta was never populated, so every stock scored the neutral
                # default and displayed a fabricated "1.00".
                result["beta"]     = _safe(info.get("beta"))
                # analyst_count is consumed by score_analyst_consensus() but was
                # never set, producing "Analyst consensus: Buy (0 analysts)".
                result["analyst_count"] = _safe(
                    info.get("numberOfAnalystOpinions"), 0
                ) or 0
            except Exception as exc:
                log.debug("Company info failed for %s: %s", ticker, exc)

            # Remove internal scratch keys
            result.pop("_revenue",   None)
            result.pop("_net_income", None)

        except Exception as exc:
            log.error("Fundamentals failed for %s: %s", ticker, exc)

    _FUND_CACHE[ticker]      = result
    _FUND_CACHE_TIME[ticker] = time.monotonic()
    return result


# ── Background cache warm-up ──────────────────────────────────────────────────
# Cold start is dominated by fundamentals: 101 tickers took ~153s of a ~169s
# total before concurrency was raised. With AlwaysOn the process is long-lived,
# so that cost is paid once per restart — and it was being paid by whoever
# happened to open the page first.
#
# This moves it off the request path: the work starts as soon as the process
# does, so by the time anyone arrives the caches are warm (or warming) and the
# page assembles from memory.
#
# It deliberately touches ONLY the plain-dict caches in this module, never a
# Streamlit API. st.cache_data requires a ScriptRunContext that a bare thread
# does not have, and calling into it from here would either throw or silently
# write to a context nobody reads.
_WARM_LOCK = threading.Lock()
_warm_thread: Optional[threading.Thread] = None
_warm_state: dict = {"started": None, "done": None, "funds": 0, "total": 0, "error": None}


def warm_status() -> dict:
    """Snapshot for the UI, so a warming app can say so instead of looking hung."""
    with _WARM_LOCK:
        return dict(_warm_state)


def start_cache_warm(tickers: list, periods: tuple = ("6mo", "1y")) -> None:
    """Kick off a one-shot background warm. Idempotent; safe on every rerun."""
    global _warm_thread
    with _WARM_LOCK:
        if _warm_thread is not None and _warm_thread.is_alive():
            return
        if _warm_state["done"] is not None:
            return                                   # already warmed this process
        _warm_state.update({"started": time.time(), "total": len(tickers),
                            "funds": 0, "done": None, "error": None})

    def _run():
        try:
            for period in periods:
                prefetch_price_histories(list(tickers), period=period)
            # Fundamentals are the expensive part; run them through the same
            # semaphore the request path uses so the two cannot combine to
            # exceed the concurrency limit.
            with ThreadPoolExecutor(max_workers=_YF_CONCURRENCY) as pool:
                for _ in pool.map(fetch_fundamentals, tickers):
                    with _WARM_LOCK:
                        _warm_state["funds"] += 1
        except Exception as exc:                     # never kill the app
            with _WARM_LOCK:
                _warm_state["error"] = str(exc)[:200]
        finally:
            with _WARM_LOCK:
                _warm_state["done"] = time.time()

    t = threading.Thread(target=_run, name="cache-warm", daemon=True)
    with _WARM_LOCK:
        _warm_thread = t
    t.start()
