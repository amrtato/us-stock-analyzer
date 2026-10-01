"""
Bearish / short-candidate analysis engine.

WHY THIS IS NOT ``100 - technical_score``
    The long technical pillar is a *blend of two opposing theses*: RSI,
    Bollinger and Stochastic score mean reversion (oversold = good), while EMA
    alignment, momentum and MACD score trend following (rising = good). They
    partially cancel, which is a large part of why the composite measured
    IC ~ 0 in walk-forward testing.

    Inverting an incoherent score yields another incoherent score — it would
    rank "overbought inside a strong uptrend" (a squeeze waiting to happen)
    alongside "broken down below a falling EMA200" and call both of them
    shorts. So this module encodes ONE thesis: confirmed downtrend
    continuation, which is how short books are actually run. Shorting strength
    on valuation is the trade that gets squeezed; shorting confirmed weakness
    is the trade that has a stop you can define.

THE STRUCTURAL HEADWIND IS REAL AND IS PRICED IN HERE
    A long in this universe earns the equity drift for free; a short pays it.
    That is why relative weakness vs SPY carries 20% of the weight: a stock
    merely falling *less* than the market is not a short, it is a bad long.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from analyzers.technical import (
    _ema,
    compute_adx,
    compute_atr,
    compute_bollinger,
    compute_macd,
    compute_rsi,
    compute_stochastic,
)

log = logging.getLogger(__name__)

# Below this RSI the name is already washed out. Short entries here are
# historically where squeezes start: you are selling to the people covering.
# Gated rather than scored, so it surfaces as a reason instead of a number.
RSI_CAPITULATION = 25.0
# A short needs borrow. Dollar volume is the only borrow proxy a free data
# source offers; thin names are where recalls and fees bite.
MIN_DOLLAR_VOLUME = 20_000_000.0


def _score_trend_breakdown(price, ema21, ema50, ema200) -> float:
    """Full bear stack (price < EMA21 < EMA50 < EMA200) is the core thesis."""
    vals = [v for v in (price, ema21, ema50, ema200) if v and not np.isnan(v)]
    if len(vals) < 2:
        return 50.0
    pairs = list(zip(vals, vals[1:]))
    correct = sum(1 for a, b in pairs if a < b)       # descending = bearish
    score = 20 + (correct / len(pairs)) * 60
    if ema50 and ema200 and not np.isnan(ema50) and not np.isnan(ema200):
        if ema50 < ema200:
            score += 10                                # death-cross structure
    return float(min(100, score))


def _score_relative_weakness(stock_roc, bench_roc) -> float:
    """Underperformance vs the index in percentage points over 60 sessions.

    Weighted heavily on purpose: an outright short pays the drift the index
    earns, so "falls less than SPY" is not a short candidate at all.
    """
    if stock_roc is None or bench_roc is None:
        return 50.0
    gap = bench_roc - stock_roc           # positive = stock lagging = bearish
    return float(max(0, min(100, 50 + gap * 2.0)))


def _score_down_momentum(roc5, roc20) -> float:
    score = 50.0
    if roc20 < 0:
        score += min(20, abs(roc20) * 1.2)
    else:
        score -= min(20, roc20 * 1.2)
    if roc5 < 0:
        score += min(15, abs(roc5) * 1.5)
    else:
        score -= min(15, roc5 * 1.5)
    return float(max(0, min(100, score)))


def _score_macd_bear(macd, hist_now, hist_prev) -> float:
    score = 50.0
    if hist_prev > 0 > hist_now:
        score += 30                                    # fresh bearish cross
    elif hist_prev < 0 < hist_now:
        score -= 30
    if hist_now < hist_prev:
        score += 15                                    # histogram deteriorating
    else:
        score -= 10
    if macd < 0:
        score += 10
    else:
        score -= 10
    return float(max(0, min(100, score)))


def _score_distribution(close: pd.Series, volume: pd.Series) -> float:
    """Are the down days the heavy-volume days? That is institutional selling."""
    if len(close) < 25 or volume is None:
        return 50.0
    ret = close.pct_change().tail(20)
    vol = volume.tail(20)
    if vol.sum() <= 0 or ret.isna().all():
        return 50.0
    down_vol = float(vol[ret < 0].sum())
    up_vol = float(vol[ret > 0].sum())
    total = down_vol + up_vol
    if total <= 0:
        return 50.0
    share = down_vol / total                           # 0.5 = balanced
    return float(max(0, min(100, 50 + (share - 0.5) * 160)))


def _score_bear_adx(adx, di_minus_dominant: bool) -> float:
    """Trend STRENGTH only counts when the trend is pointing down."""
    if not di_minus_dominant:
        return 35.0
    if adx > 40:
        return 90.0
    if adx > 25:
        return 75.0
    if adx > 20:
        return 58.0
    return 42.0


def _score_rally_rejection(price, ema21, high_20d) -> float:
    """Lower highs: price failed at the EMA21 and sits off the 20-day high."""
    if not high_20d or high_20d <= 0:
        return 50.0
    off_high = (high_20d - price) / high_20d * 100
    score = 50 + min(30, off_high * 2.0)
    if ema21 and price < ema21:
        score += 12                                    # rallies capped by EMA21
    return float(max(0, min(100, score)))


def analyse_bearish(ticker: str, df: pd.DataFrame,
                    bench_close: pd.Series | None = None) -> dict:
    """Score a short-side (downtrend-continuation) thesis in [0, 100].

    ``bench_close`` is the index close series; when absent the relative-weakness
    component abstains at 50 rather than guessing, so the score degrades
    instead of lying.
    """
    if df is None or df.empty or len(df) < 60:
        return {"score": 50.0, "signals": ["Insufficient data"], "indicators": {},
                "subscores": {},
                "gates": {"tradeable": False, "reasons": ["insufficient history"]}}

    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    volume = df["Volume"].squeeze()

    rsi = compute_rsi(close)
    macd, sig, hist = compute_macd(close)
    bb_up, bb_mid, bb_lo = compute_bollinger(close)
    atr = compute_atr(high, low, close)
    stoch_k, stoch_d = compute_stochastic(high, low, close)
    adx = compute_adx(high, low, close)

    ema21 = _ema(close, 21)
    ema50 = _ema(close, 50)
    ema200 = _ema(close, 200)
    avg_vol = volume.rolling(20).mean()

    def _last(s, default):
        try:
            v = float(s.iloc[-1])
            return default if np.isnan(v) else v
        except Exception:
            return default

    price = float(close.iloc[-1])
    rsi_v = _last(rsi, 50.0)
    macd_v = _last(macd, 0.0)
    hist_v = _last(hist, 0.0)
    hist_p = float(hist.iloc[-2]) if len(hist) > 1 and not np.isnan(hist.iloc[-2]) else 0.0
    ema21_v = _last(ema21, price)
    ema50_v = _last(ema50, price)
    ema200_v = _last(ema200, price)
    atr_v = _last(atr, 0.0)
    adx_v = _last(adx, 20.0)
    stoch_v = _last(stoch_k, 50.0)
    vol_v = float(volume.iloc[-1])
    avgvol_v = _last(avg_vol, vol_v)
    bb_lo_v = _last(bb_lo, price)
    bb_up_v = _last(bb_up, price)

    roc5 = float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) > 5 else 0.0
    roc20 = float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) > 20 else 0.0
    roc60 = float((close.iloc[-1] / close.iloc[-61] - 1) * 100) if len(close) > 60 else None

    bench_roc60 = None
    if bench_close is not None and len(bench_close) > 60:
        try:
            b = bench_close.reindex(close.index).ffill().dropna()
            if len(b) > 60:
                bench_roc60 = float((b.iloc[-1] / b.iloc[-61] - 1) * 100)
        except Exception:
            bench_roc60 = None

    high_20d = float(high.tail(20).max()) if len(high) >= 20 else price

    # DI sign: is the directional movement actually pointing down?
    dn = -low.diff()
    up = high.diff()
    dm_plus = float(np.nansum(np.where((up > dn) & (up > 0), up, 0)[-14:]))
    dm_minus = float(np.nansum(np.where((dn > up) & (dn > 0), dn, 0)[-14:]))
    di_minus_dominant = dm_minus > dm_plus

    s_trend = _score_trend_breakdown(price, ema21_v, ema50_v, ema200_v)
    s_rel = _score_relative_weakness(roc60, bench_roc60)
    s_mom = _score_down_momentum(roc5, roc20)
    s_macd = _score_macd_bear(macd_v, hist_v, hist_p)
    s_dist = _score_distribution(close, volume)
    s_adx = _score_bear_adx(adx_v, di_minus_dominant)
    s_rej = _score_rally_rejection(price, ema21_v, high_20d)

    weights = {
        "trend": 0.26, "rel": 0.20, "mom": 0.18,
        "macd": 0.12, "dist": 0.10, "adx": 0.08, "rej": 0.06,
    }
    composite = (
        s_trend * weights["trend"] + s_rel * weights["rel"] +
        s_mom * weights["mom"] + s_macd * weights["macd"] +
        s_dist * weights["dist"] + s_adx * weights["adx"] +
        s_rej * weights["rej"]
    )

    # ── Tradeability gates (reasons, not score adjustments) ───────────────────
    reasons = []
    dollar_vol = price * avgvol_v
    if rsi_v < RSI_CAPITULATION:
        reasons.append(f"RSI {rsi_v:.0f} — already capitulated, squeeze risk")
    if dollar_vol < MIN_DOLLAR_VOLUME:
        reasons.append(f"thin: ${dollar_vol / 1e6:.0f}M/day — borrow may be hard")
    if price < bb_lo_v:
        reasons.append("below lower Bollinger — extended, poor entry")
    if ema50_v > ema200_v and price > ema50_v:
        reasons.append("still in an uptrend — this would be shorting strength")
    # Caught in testing: a name can score well on trend + momentum while still
    # OUTPERFORMING the index, because it fell less than SPY did. That is a bad
    # long, not a short — an outright short pays the drift, so beating the
    # market is disqualifying no matter how bearish the chart looks alone.
    if bench_roc60 is not None and roc60 is not None and roc60 > bench_roc60:
        reasons.append(
            f"outperforming SPY by {roc60 - bench_roc60:.1f}pp over 60d — "
            f"falling less than the market is not a short"
        )

    signals = []
    if price < ema21_v < ema50_v < ema200_v:
        signals.append("Full EMA bear alignment")
    if ema50_v < ema200_v:
        signals.append("Death-cross structure (EMA50 < EMA200)")
    if hist_p > 0 > hist_v:
        signals.append("MACD bearish crossover")
    if roc20 < -5:
        signals.append(f"20-day momentum {roc20:.1f}%")
    if bench_roc60 is not None and roc60 is not None and (bench_roc60 - roc60) > 5:
        signals.append(f"Lagging SPY by {bench_roc60 - roc60:.1f}pp over 60d")
    if s_dist > 62:
        signals.append("Distribution: down days carry the volume")
    if adx_v > 30 and di_minus_dominant:
        signals.append(f"Strong downtrend (ADX {adx_v:.0f})")
    if rsi_v > 60:
        signals.append(f"RSI {rsi_v:.0f} — bounce into resistance")
    if price > ema200_v:
        signals.append("Above EMA200 — counter-trend short")

    return {
        "score": round(composite, 2),
        "signals": signals,
        "indicators": {
            "rsi": round(rsi_v, 1), "macd_hist": round(hist_v, 4),
            "ema21": round(ema21_v, 2), "ema50": round(ema50_v, 2),
            "ema200": round(ema200_v, 2), "atr": round(atr_v, 2),
            "atr_pct": round(atr_v / price * 100 if price else 0, 2),
            "adx": round(adx_v, 1), "stoch_k": round(stoch_v, 1),
            "roc5": round(roc5, 2), "roc20": round(roc20, 2),
            "roc60": round(roc60, 2) if roc60 is not None else None,
            "bench_roc60": round(bench_roc60, 2) if bench_roc60 is not None else None,
            "rel_strength": round(roc60 - bench_roc60, 2)
                            if (roc60 is not None and bench_roc60 is not None) else None,
            "high_20d": round(high_20d, 2), "bb_lower": round(bb_lo_v, 2),
            "bb_upper": round(bb_up_v, 2),
            "vol_ratio": round(vol_v / avgvol_v if avgvol_v else 1, 2),
            "dollar_volume": round(dollar_vol, 0), "price": round(price, 2),
        },
        "subscores": {
            "trend": s_trend, "rel": s_rel, "mom": s_mom, "macd": s_macd,
            "dist": s_dist, "adx": s_adx, "rej": s_rej,
        },
        "gates": {"tradeable": not reasons, "reasons": reasons},
    }
