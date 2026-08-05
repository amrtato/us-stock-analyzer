"""
Walk-forward test: does a high TECHNICAL score precede positive forward returns?

SCOPE — read this before trusting the output.
  Only the technical pillar (38% of the composite) is tested. The fundamental,
  sentiment and macro pillars CANNOT be backtested with yfinance: it serves only
  a current snapshot of P/E, ROE, analyst counts and news. Replaying today's
  fundamentals onto a 2024 date is look-ahead bias and would manufacture a
  flattering result. analyse_technical() is a pure function of the daily OHLCV
  frame, so slicing the frame at date t reproduces exactly what the app would
  have computed on that date.

METHOD
  - universe: ALL_STOCKS (the app's Daily Top Picks universe)
  - rebalance every 5 trading days after a 250-bar warm-up (EMA200 needs history)
  - at each date, score every ticker using bars <= t only
  - forward return measured from t's close to t+h's close
  - significance: one observation per REBALANCE DATE (top-decile mean minus
    universe mean), so cross-sectional correlation between stocks on the same
    day cannot inflate the sample size

KNOWN BIAS
  Survivorship: the universe is today's 101 tickers, all of which survived.
  This lifts absolute returns for every bucket, but the top-vs-universe SPREAD
  is largely immune because all buckets share the bias.
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

PERIOD      = "3y"
WARMUP      = 250
STEP        = 5
HORIZONS    = [5, 20]
TOP_FRAC    = 0.10

print(f"downloading {len(ALL_STOCKS)} tickers, period={PERIOD} ...")
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
print(f"usable tickers: {len(frames)}")
if not frames:
    sys.exit("no data")

cal = sorted(set().union(*[set(d.index) for d in frames.values()]))
cal = pd.DatetimeIndex(sorted(cal))
dates = cal[WARMUP:len(cal) - max(HORIZONS) - 1:STEP]
print(f"rebalance dates: {len(dates)}  ({dates[0].date()} to {dates[-1].date()})\n")

rows = []
for i, dt in enumerate(dates):
    if i % 10 == 0:
        print(f"  scoring {i}/{len(dates)} ...", flush=True)
    for t, d in frames.items():
        hist = d.loc[:dt]
        if len(hist) < WARMUP:
            continue
        fut = d.loc[dt:]
        if len(fut) <= max(HORIZONS):
            continue
        try:
            score = analyse_technical(t, hist)["score"]
        except Exception:
            continue
        px0 = float(hist["Close"].iloc[-1])
        if px0 <= 0:
            continue
        rec = {"date": dt, "ticker": t, "score": score}
        for h in HORIZONS:
            rec[f"fwd{h}"] = float(fut["Close"].iloc[h]) / px0 - 1.0
        rows.append(rec)

df = pd.DataFrame(rows)
print(f"\nobservations: {len(df):,}  |  mean score {df['score'].mean():.1f} "
      f"(sd {df['score'].std():.1f}, range {df['score'].min():.0f}-{df['score'].max():.0f})\n")

for h in HORIZONS:
    col = f"fwd{h}"
    print("=" * 74)
    print(f"HORIZON: {h} trading days")
    print("=" * 74)

    # quintile monotonicity — is a higher score associated with a better return?
    df["q"] = df.groupby("date")["score"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False, duplicates="drop"))
    qt = df.groupby("q")[col].agg(["mean", "count"])
    print("  score quintile (0 = lowest):")
    for q, r in qt.iterrows():
        print(f"    Q{int(q)+1}  mean fwd {r['mean']*100:+6.2f}%   n={int(r['count']):,}")

    # per-date spread: top decile mean minus universe mean
    per_date = []
    for dt, g in df.groupby("date"):
        if len(g) < 20:
            continue
        k = max(1, int(round(len(g) * TOP_FRAC)))
        top = g.nlargest(k, "score")[col].mean()
        bot = g.nsmallest(k, "score")[col].mean()
        per_date.append({"date": dt, "top": top, "bot": bot, "uni": g[col].mean()})
    pdf = pd.DataFrame(per_date)
    pdf["edge"]   = pdf["top"] - pdf["uni"]
    pdf["spread"] = pdf["top"] - pdf["bot"]

    t_edge, p_edge = stats.ttest_1samp(pdf["edge"], 0.0)
    t_sprd, p_sprd = stats.ttest_1samp(pdf["spread"], 0.0)

    print(f"\n  top decile   mean fwd {pdf['top'].mean()*100:+6.2f}%")
    print(f"  universe     mean fwd {pdf['uni'].mean()*100:+6.2f}%")
    print(f"  bottom dec.  mean fwd {pdf['bot'].mean()*100:+6.2f}%")
    print(f"\n  EDGE (top - universe): {pdf['edge'].mean()*100:+.2f}%  "
          f"t={t_edge:+.2f}  p={p_edge:.3f}  hit-rate {(pdf['edge']>0).mean()*100:.0f}%")
    print(f"  SPREAD (top - bottom): {pdf['spread'].mean()*100:+.2f}%  "
          f"t={t_sprd:+.2f}  p={p_sprd:.3f}  hit-rate {(pdf['spread']>0).mean()*100:.0f}%")
    print(f"  (n = {len(pdf)} rebalance dates)")

    ic = df.groupby("date").apply(lambda g: g["score"].corr(g[col], method="spearman"))
    print(f"\n  rank IC: mean {ic.mean():+.3f}  sd {ic.std():.3f}  "
          f"IR {ic.mean()/ic.std() if ic.std() else float('nan'):+.2f}  "
          f"positive {(ic>0).mean()*100:.0f}% of dates\n")
