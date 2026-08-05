"""
Candidate replacements for the technical pillar, tested on 5 years.

Why 5y and not the 3y used earlier: the decision to build a trend/momentum model
was informed by seeing price/EMA50 win on the 3-year out-of-sample window, so
that window is no longer a clean test of THIS design. Extending to 5 years adds
~2 years I have not looked at.

Two cadences are reported:
  STEP=5   96+ rebalance dates, but 20-day windows overlap 4x -> p-values optimistic
  STEP=20  non-overlapping windows -> fewer dates, but honest significance

Candidates (all z-scored cross-sectionally per date, EQUAL weights — no fitted
parameters, so there is nothing to overfit):
  ORIGINAL     the app's current 8-component blend
  CORRECTED    same, with rsi/bb/stoch signs flipped and macd/vol dropped
  PXEMA50      price vs EMA50 alone (the bar to beat)
  TREND3       px/ema50 + px/ema200 + roc20
  TREND3+ADX   TREND3 plus ADX (trend-strength confirmation)
  TREND2       px/ema50 + roc20
"""
import sys, os, warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd, yfinance as yf
from scipy import stats
from config import ALL_STOCKS
from analyzers.technical import analyse_technical

PERIOD, WARMUP, HORIZON = "5y", 250, 20
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".panel5y.csv")
SUBS  = ["rsi", "macd", "bb", "ema", "vol", "mom", "stoch", "adx"]
W     = {"rsi": .20, "macd": .18, "bb": .12, "ema": .18, "vol": .10,
         "mom": .10, "stoch": .07, "adx": .05}

if os.path.exists(CACHE):
    df = pd.read_csv(CACHE, parse_dates=["date"])
    print(f"loaded cached panel: {len(df):,} rows")
else:
    print(f"downloading {len(ALL_STOCKS)} tickers, {PERIOD} ...")
    raw = yf.download(ALL_STOCKS, period=PERIOD, interval="1d",
                      group_by="ticker", auto_adjust=True, progress=False, threads=True)
    frames = {}
    for t in ALL_STOCKS:
        try:
            d = raw[t].dropna()
            if len(d) >= WARMUP + HORIZON + 20:
                frames[t] = d
        except Exception:
            pass
    cal = pd.DatetimeIndex(sorted(set().union(*[set(d.index) for d in frames.values()])))
    dates = cal[WARMUP:len(cal) - HORIZON - 1:5]
    print(f"{len(frames)} tickers, {len(dates)} dates "
          f"({dates[0].date()} to {dates[-1].date()})")
    rows = []
    for i, dt in enumerate(dates):
        if i % 25 == 0:
            print(f"  scoring {i}/{len(dates)} ...", flush=True)
        for t, d in frames.items():
            hist, fut = d.loc[:dt], d.loc[dt:]
            if len(hist) < WARMUP or len(fut) <= HORIZON:
                continue
            try:
                r = analyse_technical(t, hist)
            except Exception:
                continue
            sub, ind = r.get("subscores", {}), r.get("indicators", {})
            if not sub:
                continue
            px = float(hist["Close"].iloc[-1])
            e50, e200 = ind.get("ema50"), ind.get("ema200")
            if px <= 0 or not e50 or not e200:
                continue
            rec = {"date": dt, "ticker": t, "orig": r["score"],
                   "pxema50": px / e50 - 1, "pxema200": px / e200 - 1,
                   "roc20": ind.get("roc20"), "adx": ind.get("adx"),
                   "fwd": float(fut["Close"].iloc[HORIZON]) / px - 1.0}
            for k in SUBS:
                rec[k] = sub.get(k, np.nan)
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(CACHE, index=False)
    print(f"cached {len(df):,} rows")

FEATS = ["pxema50", "pxema200", "roc20", "adx"]


def build(d):
    d = d.copy()
    for c in SUBS + FEATS:
        g = d.groupby("date")[c]
        d[c + "_z"] = (d[c] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    d["CORRECTED"] = (-d["rsi_z"] * .20 - d["bb_z"] * .12 - d["stoch_z"] * .07
                      + d["ema_z"] * .18 + d["mom_z"] * .10 + d["adx_z"] * .05)
    d["PXEMA50"]    = d["pxema50_z"]
    d["TREND2"]     = d["pxema50_z"] + d["roc20_z"]
    d["TREND3"]     = d["pxema50_z"] + d["pxema200_z"] + d["roc20_z"]
    d["TREND3+ADX"] = d["TREND3"] + d["adx_z"]
    d["ORIGINAL"]   = d["orig"]
    return d


df = build(df)
MODELS = ["ORIGINAL", "CORRECTED", "PXEMA50", "TREND2", "TREND3", "TREND3+ADX"]


def evaluate(frame, col):
    ics, edges = [], []
    for dt, g in frame.groupby("date"):
        g = g.dropna(subset=[col, "fwd"])
        if len(g) < 20 or g[col].nunique() < 4:
            continue
        ics.append(g[col].corr(g["fwd"], method="spearman"))
        k = max(1, int(round(len(g) * .10)))
        edges.append(g.nlargest(k, col)["fwd"].mean() - g["fwd"].mean())
    ics, edges = np.array(ics), np.array(edges)
    if len(ics) < 5:
        return None
    return {"ic": ics.mean(), "ir": ics.mean() / ics.std() if ics.std() else np.nan,
            "edge": edges.mean(), "t": stats.ttest_1samp(edges, 0)[0],
            "p": stats.ttest_1samp(edges, 0)[1], "hit": (edges > 0).mean(), "n": len(edges)}


all_dates = np.sort(df["date"].unique())
split = all_dates[len(all_dates) // 2]

for cadence, keep in (("STEP=5  (overlapping windows — p optimistic)", all_dates),
                      ("STEP=20 (non-overlapping — honest p)", all_dates[::4])):
    sub = df[df["date"].isin(keep)]
    print("\n" + "=" * 88)
    print(cadence)
    print("=" * 88)
    for label, frame in (("EARLY half (unseen by design)", sub[sub["date"] < split]),
                         ("LATE half", sub[sub["date"] >= split]),
                         ("FULL 5y", sub)):
        print(f"\n  --- {label} ---")
        print(f"  {'model':<14}{'IC':>8}{'IR':>7}{'edge':>9}{'t':>7}{'p':>8}{'hit':>7}{'n':>5}")
        print("  " + "-" * 63)
        for m in MODELS:
            r = evaluate(frame, m)
            if not r:
                continue
            star = " *" if r["p"] < 0.05 else ""
            print(f"  {m:<14}{r['ic']:>+8.3f}{r['ir']:>+7.2f}{r['edge']*100:>+8.2f}%"
                  f"{r['t']:>+7.2f}{r['p']:>8.3f}{r['hit']*100:>6.0f}%{r['n']:>5}{star}")
print(f"\nwindow: {pd.Timestamp(all_dates[0]).date()} to {pd.Timestamp(all_dates[-1]).date()}"
      f"   split at {pd.Timestamp(split).date()}")
