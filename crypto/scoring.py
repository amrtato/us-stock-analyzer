"""
Crypto trend scoring — ONE coherent score, ranked from both ends.

WHY THIS IS ONE MODEL AND NOT TWO, UNLIKE THE EQUITY SHORT SCREEN
    `analyzers/bearish.py` is a separate model from the long score, and the
    docstring there explains why: the equity technical pillar blends
    mean-reversion scorers (RSI, Bollinger, Stochastic) with trend scorers (EMA,
    momentum, MACD) that partly cancel, so inverting it produces another
    incoherent score.

    Here both sides are built from scratch on a SINGLE thesis — trend
    continuation — so the top and bottom of one ranking are exactly the long and
    short candidates. Two models would be two chances to overfit, with no
    compensating benefit. The asymmetry in the equity case was a property of the
    score I inherited, not a general law.

SHORTING CRYPTO IS ACTUALLY SYMMETRIC, WHICH EQUITIES ARE NOT
    The equity short screen needed borrow-availability gates, a squeeze-risk
    model and half-size positions because shorting stock is structurally harder
    than buying it. Crypto perpetuals have no borrow constraint, no recall and
    no uptick rule, so a short is close to a mirror of a long. The gates here
    are about LIQUIDITY and VOLATILITY, not borrow.

EVERY FEATURE IS CROSS-SECTIONALLY Z-SCORED, EQUALLY WEIGHTED
    No fitted coefficients, so there is nothing to overfit. This is the same
    construction `compare_models.py` used for the equity candidates, and it is
    deliberate: a model with free parameters would score better in-sample and
    tell you nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BENCH = "X:BTCUSD"

FEATURES = ("trend", "mom", "rel_btc", "risk_adj_mom", "vol_trend")


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def compute_features(panel: pd.DataFrame, bench: str = BENCH) -> pd.DataFrame:
    """Per-(ticker, date) features. Strictly backward-looking at every row.

    Nothing here uses a future bar, so slicing the panel at date t reproduces
    exactly what the screen would have shown on t.
    """
    p = panel.sort_values(["ticker", "date"]).copy()
    g = p.groupby("ticker", group_keys=False)

    p["ema20"] = g["close"].transform(lambda s: _ema(s, 20))
    p["ema50"] = g["close"].transform(lambda s: _ema(s, 50))
    # EMA200 on daily crypto is ~7 months. Kept for structure but weighted via
    # `trend` alongside faster averages, because crypto regimes turn over much
    # faster than equity ones.
    p["ema100"] = g["close"].transform(lambda s: _ema(s, 100))

    p["roc7"] = g["close"].transform(lambda s: s.pct_change(7) * 100)
    p["roc30"] = g["close"].transform(lambda s: s.pct_change(30) * 100)
    p["vol30"] = g["close"].transform(
        lambda s: s.pct_change().rolling(30).std() * np.sqrt(365) * 100)

    # Benchmark return, aligned on date. Crypto is dominated by one factor: a
    # coin up 5% while BTC is up 8% is WEAK, and a raw momentum screen would
    # call it strong.
    b = (p[p["ticker"] == bench][["date", "close"]]
         .rename(columns={"close": "bench_close"}))
    p = p.merge(b, on="date", how="left")
    p["bench_roc30"] = p["bench_close"].pct_change(30) * 100

    # ── features ─────────────────────────────────────────────────────────────
    px = p["close"]
    p["trend"] = ((px / p["ema20"] - 1) + (px / p["ema50"] - 1)
                  + (px / p["ema100"] - 1)) * 100
    p["mom"] = p["roc7"] * 0.4 + p["roc30"] * 0.6
    p["rel_btc"] = p["roc30"] - p["bench_roc30"]
    # Momentum per unit of risk. Without this the ranking is just a volatility
    # ranking: the wildest coin has the biggest raw move in whichever direction
    # it happened to go.
    p["risk_adj_mom"] = p["roc30"] / p["vol30"].replace(0, np.nan)
    # Is volume confirming the move, or is it drifting on nothing?
    p["dollar_volume"] = p["volume"] * p["vwap"].fillna(p["close"])
    p["vol_trend"] = g["dollar_volume"].transform(
        lambda s: s.rolling(7).mean() / s.rolling(30).mean().replace(0, np.nan))

    return p


def score_cross_section(p: pd.DataFrame, features=FEATURES) -> pd.DataFrame:
    """Z-score each feature within each date, then average. Equal weights."""
    out = p.copy()
    parts = []
    for f in features:
        grp = out.groupby("date")[f]
        z = (out[f] - grp.transform("mean")) / grp.transform("std").replace(0, np.nan)
        out[f + "_z"] = z.clip(-3, 3)          # cap so one outlier cannot dominate
        parts.append(out[f + "_z"])
    out["score"] = pd.concat(parts, axis=1).mean(axis=1)
    return out


def rank_latest(scored: pd.DataFrame, n: int = 10,
                exclude: tuple = (BENCH,)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Top-n long and bottom-n short candidates on the most recent date.

    BTC is excluded by default: it IS the benchmark, so its relative-strength
    feature is identically zero and ranking it against itself is meaningless.
    """
    if scored.empty:
        return pd.DataFrame(), pd.DataFrame()
    last = scored["date"].max()
    cur = scored[(scored["date"] == last) & (~scored["ticker"].isin(exclude))]
    cur = cur.dropna(subset=["score"])
    longs = cur.nlargest(n, "score").copy()
    shorts = cur.nsmallest(n, "score").copy()
    return longs, shorts


def to_display(rows: pd.DataFrame, side: str) -> pd.DataFrame:
    """Flatten a ranked slice into the columns the UI shows."""
    if rows.empty:
        return pd.DataFrame()
    d = rows.copy()
    d["symbol"] = d["ticker"].str.replace("^X:", "", regex=True).str.replace("USD$", "", regex=True)
    d["side"] = side
    cols = ["symbol", "side", "close", "score", "roc7", "roc30", "rel_btc",
            "vol30", "dollar_volume"]
    return d[[c for c in cols if c in d.columns]]
