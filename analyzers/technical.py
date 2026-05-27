"""
Technical analysis engine.

Indicators computed:
  RSI(14), MACD(12,26,9), Bollinger Bands(20,2), EMA(9,21,50,200),
  ATR(14), Stochastic(14,3), Volume ratio, Price momentum (ROC),
  Support/Resistance levels, Trend strength (ADX-like).

Each indicator returns a sub-score in [0, 100].
Final technical score is their weighted average.
"""
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Indicator implementations (no external TA library needed) ──────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def compute_macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast   = _ema(close, fast)
    ema_slow   = _ema(close, slow)
    macd_line  = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger(close: pd.Series, period=20, std_dev=2):
    mid  = _sma(close, period)
    std  = close.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low   = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    pct_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-9)
    pct_d = pct_k.rolling(d_period).mean()
    return pct_k, pct_d


def compute_adx(high, low, close, period=14) -> pd.Series:
    """Simplified ADX (trend strength 0-100)."""
    up_move   = high.diff()
    down_move = -low.diff()
    dm_plus  = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    dm_minus = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    atr = compute_atr(high, low, close, period)
    di_plus  = 100 * pd.Series(dm_plus,  index=close.index).rolling(period).mean() / atr
    di_minus = 100 * pd.Series(dm_minus, index=close.index).rolling(period).mean() / atr
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-9)
    return dx.rolling(period).mean()


# ── Scoring functions (each returns 0-100) ─────────────────────────────────────

def _score_rsi(rsi: float) -> float:
    """Oversold → high score (bullish opportunity); neutral ~50."""
    if rsi <= 30:   return 90
    if rsi <= 40:   return 75
    if rsi <= 55:   return 55
    if rsi <= 65:   return 45
    if rsi <= 75:   return 30
    return 15  # very overbought


def _score_macd(macd, signal, hist_now, hist_prev) -> float:
    """Bullish cross or rising histogram = high score."""
    score = 50.0
    # Cross detection
    if hist_prev < 0 < hist_now:   score += 30  # fresh bullish cross
    elif hist_prev > 0 > hist_now: score -= 30  # fresh bearish cross
    # Histogram direction
    if hist_now > hist_prev:       score += 15
    else:                          score -= 10
    # MACD above zero line
    if macd > 0:                   score += 10
    else:                          score -= 10
    return max(0, min(100, score))


def _score_bollinger(price, upper, mid, lower) -> float:
    """Near lower band = oversold/buy opportunity; near upper = extended."""
    band_width = upper - lower
    if band_width < 1e-9:
        return 50.0
    pos = (price - lower) / band_width  # 0=at lower, 1=at upper
    if pos < 0.1:   return 85  # extreme oversold
    if pos < 0.25:  return 70
    if pos < 0.45:  return 60  # below mid
    if pos < 0.55:  return 55  # mid
    if pos < 0.75:  return 40  # above mid but not extended
    if pos < 0.90:  return 30
    return 15  # at or above upper band


def _score_ema_alignment(price, ema9, ema21, ema50, ema200) -> float:
    """Perfect bull alignment: price > EMA9 > EMA21 > EMA50 > EMA200."""
    score = 50.0
    vals  = [v for v in [price, ema9, ema21, ema50, ema200] if not np.isnan(v)]
    if len(vals) < 2:
        return score
    # Each correctly ordered pair adds points
    pairs = list(zip(vals, vals[1:]))
    correct = sum(1 for a, b in pairs if a > b)
    score = 20 + (correct / len(pairs)) * 60
    # Golden cross bonus
    if not np.isnan(ema50) and not np.isnan(ema200) and ema50 > ema200:
        score += 10
    return min(100, score)


def _score_volume(vol_today, avg_vol) -> float:
    """Volume surge on upday is bullish; score the magnitude."""
    if avg_vol <= 0:
        return 50.0
    ratio = vol_today / avg_vol
    if ratio > 3.0:  return 90
    if ratio > 2.0:  return 75
    if ratio > 1.5:  return 65
    if ratio > 1.0:  return 55
    if ratio > 0.7:  return 45
    return 30


def _score_momentum(roc_5, roc_20) -> float:
    """Rate of change on 5d and 20d windows."""
    score = 50.0
    if roc_5  > 0: score += min(15, roc_5  * 2)
    else:          score -= min(15, abs(roc_5) * 2)
    if roc_20 > 0: score += min(10, roc_20)
    else:          score -= min(10, abs(roc_20))
    return max(0, min(100, score))


def _score_stochastic(k, d) -> float:
    """Oversold stochastic with bullish cross = buy signal."""
    score = 50.0
    if k < 20 and k > d:   score = 80  # oversold bullish cross
    elif k < 20:           score = 65  # oversold
    elif k > 80 and k < d: score = 20  # overbought bearish cross
    elif k > 80:           score = 30  # overbought
    else:
        score = 40 + (50 - k) * 0.2   # linear in mid-range
    return max(0, min(100, score))


def _score_trend_strength(adx: float) -> float:
    """Strong trend = higher confidence in direction signals."""
    if adx > 40:   return 85
    if adx > 25:   return 70
    if adx > 20:   return 55
    return 40  # choppy, weak trend


# ── Main entry point ───────────────────────────────────────────────────────────

def analyse_technical(ticker: str, df: pd.DataFrame, intraday_df: pd.DataFrame = None) -> dict:
    """
    Compute all technical indicators and return a dict with:
      - 'score': composite technical score [0-100]
      - 'signals': list of human-readable signal strings
      - 'indicators': raw indicator values
      - 'timeframe_fit': {'scalp': bool, 'day': bool, 'swing': bool, 'invest': bool}
    """
    if df.empty or len(df) < 30:
        return {"score": 50.0, "signals": ["Insufficient data"], "indicators": {}, "timeframe_fit": {}}

    close  = df["Close"].squeeze()
    high   = df["High"].squeeze()
    low    = df["Low"].squeeze()
    volume = df["Volume"].squeeze()

    # ── Compute indicators ────────────────────────────────────────────────────
    rsi          = compute_rsi(close)
    macd, sig, hist = compute_macd(close)
    bb_up, bb_mid, bb_lo = compute_bollinger(close)
    atr          = compute_atr(high, low, close)
    stoch_k, stoch_d = compute_stochastic(high, low, close)
    adx          = compute_adx(high, low, close)

    ema9  = _ema(close, 9)
    ema21 = _ema(close, 21)
    ema50 = _ema(close, 50)
    ema200 = _ema(close, 200)
    avg_vol = volume.rolling(20).mean()

    # Latest values
    price     = float(close.iloc[-1])
    rsi_v     = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0
    macd_v    = float(macd.iloc[-1]) if not np.isnan(macd.iloc[-1]) else 0.0
    sig_v     = float(sig.iloc[-1])  if not np.isnan(sig.iloc[-1])  else 0.0
    hist_v    = float(hist.iloc[-1]) if not np.isnan(hist.iloc[-1]) else 0.0
    hist_prev = float(hist.iloc[-2]) if len(hist) > 1 and not np.isnan(hist.iloc[-2]) else 0.0
    bb_up_v   = float(bb_up.iloc[-1]) if not np.isnan(bb_up.iloc[-1]) else price
    bb_mid_v  = float(bb_mid.iloc[-1]) if not np.isnan(bb_mid.iloc[-1]) else price
    bb_lo_v   = float(bb_lo.iloc[-1]) if not np.isnan(bb_lo.iloc[-1]) else price
    ema9_v    = float(ema9.iloc[-1])  if not np.isnan(ema9.iloc[-1])  else price
    ema21_v   = float(ema21.iloc[-1]) if not np.isnan(ema21.iloc[-1]) else price
    ema50_v   = float(ema50.iloc[-1]) if not np.isnan(ema50.iloc[-1]) else price
    ema200_v  = float(ema200.iloc[-1]) if not np.isnan(ema200.iloc[-1]) else price
    atr_v     = float(atr.iloc[-1])   if not np.isnan(atr.iloc[-1])   else 0.0
    stoch_k_v = float(stoch_k.iloc[-1]) if not np.isnan(stoch_k.iloc[-1]) else 50.0
    stoch_d_v = float(stoch_d.iloc[-1]) if not np.isnan(stoch_d.iloc[-1]) else 50.0
    adx_v     = float(adx.iloc[-1])   if not np.isnan(adx.iloc[-1])   else 20.0
    vol_v     = float(volume.iloc[-1])
    avg_vol_v = float(avg_vol.iloc[-1]) if not np.isnan(avg_vol.iloc[-1]) else vol_v

    # Rate of change
    roc5  = float((close.iloc[-1] / close.iloc[-6]  - 1) * 100) if len(close) > 5  else 0.0
    roc20 = float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) > 20 else 0.0

    # ── Sub-scores ────────────────────────────────────────────────────────────
    s_rsi    = _score_rsi(rsi_v)
    s_macd   = _score_macd(macd_v, sig_v, hist_v, hist_prev)
    s_bb     = _score_bollinger(price, bb_up_v, bb_mid_v, bb_lo_v)
    s_ema    = _score_ema_alignment(price, ema9_v, ema21_v, ema50_v, ema200_v)
    s_vol    = _score_volume(vol_v, avg_vol_v)
    s_mom    = _score_momentum(roc5, roc20)
    s_stoch  = _score_stochastic(stoch_k_v, stoch_d_v)
    s_adx    = _score_trend_strength(adx_v)

    weights = {
        "rsi": 0.20, "macd": 0.18, "bb": 0.12, "ema": 0.18,
        "vol": 0.10, "mom": 0.10, "stoch": 0.07, "adx": 0.05,
    }
    composite = (
        s_rsi   * weights["rsi"]   +
        s_macd  * weights["macd"]  +
        s_bb    * weights["bb"]    +
        s_ema   * weights["ema"]   +
        s_vol   * weights["vol"]   +
        s_mom   * weights["mom"]   +
        s_stoch * weights["stoch"] +
        s_adx   * weights["adx"]
    )

    # ── Signal strings ────────────────────────────────────────────────────────
    signals = []
    if rsi_v < 35:      signals.append(f"RSI oversold ({rsi_v:.1f})")
    elif rsi_v > 65:    signals.append(f"RSI overbought ({rsi_v:.1f})")
    if hist_prev < 0 < hist_v:  signals.append("MACD bullish crossover")
    elif hist_prev > 0 > hist_v: signals.append("MACD bearish crossover")
    if price < bb_lo_v:          signals.append("Price below lower Bollinger Band")
    elif price > bb_up_v:        signals.append("Price above upper Bollinger Band")
    if ema50_v > ema200_v and float(ema50.iloc[-2]) <= float(ema200.iloc[-2]):
        signals.append("Golden Cross (EMA50 crossed above EMA200)")
    if price > ema9_v > ema21_v > ema50_v: signals.append("Full EMA bull alignment")
    if vol_v > avg_vol_v * 2:    signals.append(f"Volume surge ({vol_v/avg_vol_v:.1f}x avg)")
    if adx_v > 30:               signals.append(f"Strong trend (ADX {adx_v:.0f})")
    if stoch_k_v < 20:           signals.append(f"Stochastic oversold ({stoch_k_v:.0f})")
    if roc5 > 3:                 signals.append(f"5-day momentum +{roc5:.1f}%")

    # ── Timeframe fit ─────────────────────────────────────────────────────────
    atr_pct = atr_v / price * 100 if price else 0
    scalp  = (rsi_v < 35 or rsi_v > 70) and vol_v > avg_vol_v * 1.5 and atr_pct > 0.5
    day    = (composite > 62) and (adx_v > 20) and (vol_v > avg_vol_v * 1.2)
    swing  = (composite > 60) and (ema50_v > ema200_v) and (roc20 > 0)
    invest = (composite > 58) and (ema200_v > 0) and (price > ema200_v) and (adx_v > 20)

    return {
        "score": round(composite, 2),
        "signals": signals,
        "indicators": {
            "rsi":        round(rsi_v, 1),
            "macd":       round(macd_v, 4),
            "macd_hist":  round(hist_v, 4),
            "bb_upper":   round(bb_up_v, 2),
            "bb_lower":   round(bb_lo_v, 2),
            "ema9":       round(ema9_v, 2),
            "ema21":      round(ema21_v, 2),
            "ema50":      round(ema50_v, 2),
            "ema200":     round(ema200_v, 2),
            "atr":        round(atr_v, 2),
            "atr_pct":    round(atr_pct, 2),
            "stoch_k":    round(stoch_k_v, 1),
            "adx":        round(adx_v, 1),
            "vol_ratio":  round(vol_v / avg_vol_v if avg_vol_v else 1, 2),
            "roc5":       round(roc5, 2),
            "roc20":      round(roc20, 2),
            "price":      round(price, 2),
        },
        "subscores": {
            "rsi": s_rsi, "macd": s_macd, "bb": s_bb, "ema": s_ema,
            "vol": s_vol, "mom": s_mom, "stoch": s_stoch, "adx": s_adx,
        },
        "timeframe_fit": {
            "scalp":  scalp,
            "day":    day,
            "swing":  swing,
            "invest": invest,
        },
    }
