"""
Gold (GC=F) feature engine — Phase 1, swing timeframe.

DISCIPLINE (the reason the stock scorer measured zero edge):
  * every feature is CAUSAL — computed from bars <= t only
  * any normalisation uses an EXPANDING window, never whole-sample statistics.
    Ranking a feature against its own full history is look-ahead bias and is the
    single easiest way to manufacture a backtest that cannot be traded.
  * features are grouped by what they actually measure, so we can check whether
    a "confirmation" is really independent information or the same price series
    wearing a different hat. Cross-asset drivers (DXY, yields, risk) are the only
    genuinely orthogonal block here — every oscillator is a transform of price.
"""
from __future__ import annotations
import warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)

import numpy as np
import pandas as pd
import yfinance as yf

GOLD = "GC=F"
CROSS = {
    "dxy":  "DX-Y.NYB",   # dollar index      — gold's most direct inverse driver
    "tnx":  "^TNX",       # 10y nominal yield
    "fvx":  "^FVX",       # 5y nominal yield
    "tip":  "TIP",        # TIPS ETF — proxy for REAL yields (up = real yields down)
    "vix":  "^VIX",       # equity risk
    "spx":  "^GSPC",
    "slv":  "SLV",        # silver — metals-complex confirmation
    "cop":  "HG=F",       # copper — growth vs store-of-value
    "oil":  "CL=F",       # inflation impulse
}


def _flat(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance sometimes returns a column MultiIndex; flatten and drop tz."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df[~df.index.duplicated(keep="last")]


def load(period: str = "10y") -> pd.DataFrame:
    g = _flat(yf.Ticker(GOLD).history(period=period, interval="1d")).dropna()
    g = g[["Open", "High", "Low", "Close", "Volume"]]
    out = g.copy()
    for name, sym in CROSS.items():
        try:
            c = _flat(yf.Ticker(sym).history(period=period, interval="1d"))["Close"]
            out[f"x_{name}"] = c.reindex(out.index).ffill(limit=5)
        except Exception as exc:
            print(f"  warn: {sym} failed ({type(exc).__name__})")
    return out


# ── primitives ────────────────────────────────────────────────────────────────
def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _atr(h, l, c, n: int = 14) -> pd.Series:
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def _adx(h, l, c, n: int = 14) -> pd.Series:
    up, dn = h.diff(), -l.diff()
    plus  = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    pdi = 100 * pd.Series(plus,  index=h.index).ewm(alpha=1/n, adjust=False).mean() / atr
    mdi = 100 * pd.Series(minus, index=h.index).ewm(alpha=1/n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean()


def _swing_pivots(h: pd.Series, l: pd.Series, k: int = 3):
    """Fractal pivots confirmed k bars later — shifted so they are causal."""
    ph = h.rolling(2*k+1, center=True).max().eq(h)
    pl = l.rolling(2*k+1, center=True).min().eq(l)
    return ph.shift(k).fillna(False), pl.shift(k).fillna(False)


def build(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    f = pd.DataFrame(index=df.index)
    f["close"] = c

    # ── TREND ────────────────────────────────────────────────────────────────
    for n in (20, 50, 200):
        f[f"t_px_ema{n}"] = c / c.ewm(span=n, adjust=False).mean() - 1
    f["t_ema50_200"] = (c.ewm(span=50, adjust=False).mean()
                        / c.ewm(span=200, adjust=False).mean() - 1)
    f["t_adx"] = _adx(h, l, c)

    # ── MOMENTUM ─────────────────────────────────────────────────────────────
    for n in (5, 10, 20, 60):
        f[f"m_roc{n}"] = c.pct_change(n)

    # ── MEAN REVERSION ───────────────────────────────────────────────────────
    f["r_rsi14"] = _rsi(c)
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    f["r_bb_pos"] = (c - (ma20 - 2*sd20)) / (4*sd20).replace(0, np.nan)
    f["r_from_hi20"] = c / h.rolling(20).max() - 1
    f["r_from_lo20"] = c / l.rolling(20).min() - 1

    # ── VOLATILITY ───────────────────────────────────────────────────────────
    atr = _atr(h, l, c)
    f["v_atr_pct"] = atr / c
    f["v_rvol20"]  = c.pct_change().rolling(20).std()
    # expanding percentile of vol — causal regime label
    f["v_atr_rank"] = f["v_atr_pct"].expanding(250).apply(
        lambda w: (w.iloc[-1] > w[:-1]).mean() if len(w) > 30 else np.nan, raw=False)

    # ── STRUCTURE / ICT-STYLE ────────────────────────────────────────────────
    ph20, pl20 = h.rolling(20).max().shift(1), l.rolling(20).min().shift(1)
    # liquidity sweep: prior range extreme taken intrabar, then rejected on close
    f["s_sweep_hi"] = ((h > ph20) & (c < ph20)).astype(float)
    f["s_sweep_lo"] = ((l < pl20) & (c > pl20)).astype(float)
    # genuine breakout (closed beyond)
    f["s_bo_hi"] = (c > ph20).astype(float)
    f["s_bo_lo"] = (c < pl20).astype(float)
    # fair value gap (3-bar imbalance), causal: known at bar t
    f["s_fvg_up"] = (l > h.shift(2)).astype(float)
    f["s_fvg_dn"] = (h < l.shift(2)).astype(float)
    # break of structure vs last confirmed fractal pivot
    ph, pl = _swing_pivots(h, l, 3)
    last_ph = h.where(ph).ffill()
    last_pl = l.where(pl).ffill()
    f["s_bos_up"] = (c > last_ph).astype(float)
    f["s_bos_dn"] = (c < last_pl).astype(float)
    f["s_inside"] = ((h < h.shift()) & (l > l.shift())).astype(float)

    # ── CROSS-ASSET (the orthogonal block) ───────────────────────────────────
    for name in CROSS:
        col = f"x_{name}"
        if col not in df:
            continue
        s = df[col]
        if name in ("tnx", "fvx", "vix"):        # already rates/levels
            f[f"c_{name}_chg5"]  = s.diff(5)
            f[f"c_{name}_chg20"] = s.diff(20)
            f[f"c_{name}_lvl"]   = s
        else:
            f[f"c_{name}_roc5"]  = s.pct_change(5)
            f[f"c_{name}_roc20"] = s.pct_change(20)
    if "x_slv" in df:
        f["c_gold_silver"] = c / df["x_slv"]
        f["c_gs_roc20"] = f["c_gold_silver"].pct_change(20)
    # real-yield proxy: TIPS price momentum (TIP up => real yields down => gold tailwind)
    if "x_tip" in df:
        f["c_realyield_proxy"] = -df["x_tip"].pct_change(20)

    # ── SEASONALITY ──────────────────────────────────────────────────────────
    f["z_dow"]   = df.index.dayofweek
    f["z_month"] = df.index.month

    # ── TARGETS (forward — never used as inputs) ─────────────────────────────
    for hz in (1, 5, 10, 20):
        f[f"fwd{hz}"] = c.shift(-hz) / c - 1
    f["atr_at_t"] = atr
    return f


FEATURE_GROUPS = {
    "trend":       lambda n: n.startswith("t_"),
    "momentum":    lambda n: n.startswith("m_"),
    "meanrev":     lambda n: n.startswith("r_"),
    "volatility":  lambda n: n.startswith("v_"),
    "structure":   lambda n: n.startswith("s_"),
    "crossasset":  lambda n: n.startswith("c_"),
    "seasonal":    lambda n: n.startswith("z_"),
}


def feature_names(f: pd.DataFrame) -> list[str]:
    return [c for c in f.columns
            if any(fn(c) for fn in FEATURE_GROUPS.values())]
