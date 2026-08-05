"""
Per-indicator decomposition: which technical inputs (if any) carry signal?

backtest.py showed the blended technical score has rank IC ~= 0. This asks the
follow-up: is that because every input is noise, or because a real signal is
being diluted / cancelled by the blend?

Tests, per rebalance date, the cross-sectional Spearman rank IC of:
  1. the 8 SUBSCORES the app actually blends (these are what the weights apply to)
  2. a few raw/derived features, as a sanity reference

A useful equity signal typically shows |mean IC| >= 0.02-0.03 with an information
ratio (mean IC / sd IC) above ~0.2. Sign matters: a consistently NEGATIVE IC is
just as usable as a positive one (invert it), so what damns a feature is IC ~= 0,
not IC < 0.

Same point-in-time discipline and the same survivorship caveat as backtest.py.
"""
import sys, warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
sys.path.insert(0, r"C:\Users\AmrAbdelsalam\OneDrive - Trioshield Technologies Inc\Desktop\Claude Projects\XAUUSD_AurumPrime\us-stock-analyzer")

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

from config import ALL_STOCKS
from analyzers.technical import analyse_technical

PERIOD, WARMUP, STEP = "3y", 250, 5
HORIZONS = [5, 20]
CURRENT_W = {"rsi": .20, "macd": .18, "bb": .12, "ema": .18,
             "vol": .10, "mom": .10, "stoch": .07, "adx": .05}

print(f"downloading {len(ALL_STOCKS)} tickers ...")
raw = yf.download(ALL_STOCKS, period=PERIOD, interval="1d",
                  group_by="ticker", auto_adjust=True, progress=False, threads=True)
frames = {}
for t in ALL_STOCKS:
    try:
        d = raw[t].dropna()
        if len(d) >= WARMUP + max(HORIZONS) + 20:
            frames[t] = d
    except Exception:
        pass

cal = pd.DatetimeIndex(sorted(set().union(*[set(d.index) for d in frames.values()])))
dates = cal[WARMUP:len(cal) - max(HORIZONS) - 1:STEP]
print(f"{len(frames)} tickers, {len(dates)} rebalance dates "
      f"({dates[0].date()} to {dates[-1].date()})\n")

rows = []
for i, dt in enumerate(dates):
    if i % 20 == 0:
        print(f"  {i}/{len(dates)} ...", flush=True)
    for t, d in frames.items():
        hist = d.loc[:dt]
        fut  = d.loc[dt:]
        if len(hist) < WARMUP or len(fut) <= max(HORIZONS):
            continue
        try:
            r = analyse_technical(t, hist)
        except Exception:
            continue
        sub, ind = r.get("subscores", {}), r.get("indicators", {})
        if not sub or not ind:
            continue
        px0 = float(hist["Close"].iloc[-1])
        if px0 <= 0:
            continue
        rec = {"date": dt, "ticker": t, "SCORE(blended)": r["score"]}
        for k, v in sub.items():
            rec[f"sub:{k}"] = v
        ema50, ema200 = ind.get("ema50") or np.nan, ind.get("ema200") or np.nan
        bb_u, bb_l = ind.get("bb_upper") or np.nan, ind.get("bb_lower") or np.nan
        rec.update({
            "raw:rsi":        ind.get("rsi"),
            "raw:adx":        ind.get("adx"),
            "raw:roc5":       ind.get("roc5"),
            "raw:roc20":      ind.get("roc20"),
            "raw:vol_ratio":  ind.get("vol_ratio"),
            "raw:atr_pct":    ind.get("atr_pct"),
            "raw:px/ema50":   px0 / ema50 - 1 if ema50 and ema50 == ema50 else np.nan,
            "raw:px/ema200":  px0 / ema200 - 1 if ema200 and ema200 == ema200 else np.nan,
            "raw:bb_pos":     (px0 - bb_l) / (bb_u - bb_l) if bb_u and bb_l and bb_u > bb_l else np.nan,
        })
        for h in HORIZONS:
            rec[f"fwd{h}"] = float(fut["Close"].iloc[h]) / px0 - 1.0
        rows.append(rec)

df = pd.DataFrame(rows)
feats = [c for c in df.columns if c.startswith(("sub:", "raw:")) or c == "SCORE(blended)"]
print(f"\nobservations: {len(df):,}\n")


def ic_table(feature, col):
    """Per-date Spearman IC, then aggregate across dates."""
    ics = df.groupby("date").apply(
        lambda g: g[feature].corr(g[col], method="spearman")
        if g[feature].notna().sum() > 10 and g[feature].nunique() > 3 else np.nan
    ).dropna()
    if len(ics) < 10:
        return None
    t, p = stats.ttest_1samp(ics, 0.0)
    return {"mean": ics.mean(), "sd": ics.std(),
            "ir": ics.mean() / ics.std() if ics.std() else np.nan,
            "t": t, "p": p, "pos": (ics > 0).mean(), "n": len(ics)}


for h in HORIZONS:
    col = f"fwd{h}"
    print("=" * 82)
    print(f"RANK IC vs {h}-DAY FORWARD RETURN     (weight = current technical weight)")
    print("=" * 82)
    print(f"  {'feature':<20}{'wt':>6}{'mean IC':>10}{'IR':>8}{'t':>8}{'p':>8}{'pos%':>7}")
    print("  " + "-" * 78)
    res = []
    for f in feats:
        r = ic_table(f, col)
        if r:
            res.append((f, r))
    res.sort(key=lambda x: -abs(x[1]["mean"]))
    for f, r in res:
        key = f.split(":")[1] if f.startswith("sub:") else None
        wt = f"{CURRENT_W[key]:.2f}" if key in CURRENT_W else ""
        star = "  <<<" if r["p"] < 0.05 else ""
        print(f"  {f:<20}{wt:>6}{r['mean']:>+10.3f}{r['ir']:>+8.2f}"
              f"{r['t']:>+8.2f}{r['p']:>8.3f}{r['pos']*100:>6.0f}%{star}")
    print()
