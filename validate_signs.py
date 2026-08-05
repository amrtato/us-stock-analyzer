"""
Split-sample validation of the sign-inversion hypothesis.

decompose.py found that sub:rsi, sub:bb and sub:stoch have rank ICs that mirror
their own raw indicators — i.e. the scoring functions appear to invert their
inputs. That was measured on the FULL sample, so acting on it would be fitting
to the same data that produced the hypothesis.

This script does it honestly:
  IN-SAMPLE   (first half)  -> decide each subscore's sign, and derive weights
  OUT-OF-SAMPLE (second half, untouched) -> evaluate the resulting composites

Nothing from the OOS window informs any sign or weight.

Composites compared, all evaluated OOS:
  ORIGINAL        the app's current blended technical score
  SIGN-CORRECTED  same 8 subscores, same weights, signs flipped per IS ICs
  IC-WEIGHTED     weights proportional to IS |IC| (more aggressive fitting)
  PX/EMA50        single best raw feature from decompose.py, as a reference bar

Subscores are z-scored cross-sectionally per date before blending so that the
differing scales/distributions of the 8 scorers cannot dominate the sum.
"""
import sys, os, warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

from config import ALL_STOCKS
from analyzers.technical import analyse_technical

PERIOD, WARMUP, STEP = "3y", 250, 5
HORIZON  = 20
CACHE    = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".scored_cache.csv")
SUBS     = ["rsi", "macd", "bb", "ema", "vol", "mom", "stoch", "adx"]
W        = {"rsi": .20, "macd": .18, "bb": .12, "ema": .18,
            "vol": .10, "mom": .10, "stoch": .07, "adx": .05}

# ── build (or reuse) the point-in-time scored panel ───────────────────────────
if os.path.exists(CACHE):
    df = pd.read_csv(CACHE, parse_dates=["date"])
    print(f"loaded cached panel: {len(df):,} rows")
else:
    print(f"downloading {len(ALL_STOCKS)} tickers ...")
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
    dates = cal[WARMUP:len(cal) - HORIZON - 1:STEP]
    rows = []
    for i, dt in enumerate(dates):
        if i % 20 == 0:
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
            px0 = float(hist["Close"].iloc[-1])
            ema50 = ind.get("ema50") or np.nan
            if px0 <= 0:
                continue
            rec = {"date": dt, "ticker": t, "orig": r["score"],
                   "pxema50": px0 / ema50 - 1 if ema50 == ema50 and ema50 else np.nan,
                   "fwd": float(fut["Close"].iloc[HORIZON]) / px0 - 1.0}
            for k in SUBS:
                rec[k] = sub.get(k, np.nan)
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(CACHE, index=False)
    print(f"cached {len(df):,} rows -> {os.path.basename(CACHE)}")

dates = np.sort(df["date"].unique())
split = dates[len(dates) // 2]
IS, OOS = df[df["date"] < split], df[df["date"] >= split]
print(f"\nIN-SAMPLE   {pd.Timestamp(dates[0]).date()} to {pd.Timestamp(split).date()}"
      f"   {IS['date'].nunique()} dates, {len(IS):,} obs")
print(f"OUT-SAMPLE  {pd.Timestamp(split).date()} to {pd.Timestamp(dates[-1]).date()}"
      f"   {OOS['date'].nunique()} dates, {len(OOS):,} obs\n")


def per_date_ic(frame, col, target="fwd"):
    return frame.groupby("date").apply(
        lambda g: g[col].corr(g[target], method="spearman")
        if g[col].notna().sum() > 10 and g[col].nunique() > 3 else np.nan).dropna()


def summarise(ics, label):
    if len(ics) < 5:
        return f"  {label:<16} insufficient dates"
    t, p = stats.ttest_1samp(ics, 0.0)
    return (f"  {label:<16} IC {ics.mean():+.3f}   IR {ics.mean()/ics.std():+.2f}   "
            f"t {t:+.2f}   p {p:.3f}   pos {(ics>0).mean()*100:.0f}%   n={len(ics)}")


# ── STEP 1: derive signs + IC weights from IN-SAMPLE ONLY ─────────────────────
print("STEP 1 — per-subscore IC, IN-SAMPLE vs OUT-OF-SAMPLE")
print(f"  {'subscore':<10}{'wt':>6}{'IS IC':>9}{'OOS IC':>9}{'sign held?':>13}")
print("  " + "-" * 50)
signs, is_ic = {}, {}
for k in SUBS:
    i_ic = per_date_ic(IS, k).mean()
    o_ic = per_date_ic(OOS, k).mean()
    signs[k] = -1.0 if i_ic < 0 else 1.0
    is_ic[k] = abs(i_ic)
    held = "yes" if (i_ic > 0) == (o_ic > 0) else "NO — flipped"
    print(f"  {k:<10}{W[k]:>6.2f}{i_ic:>+9.3f}{o_ic:>+9.3f}{held:>13}")

flipped = [k for k in SUBS if signs[k] < 0]
print(f"\n  signs inverted by IS data: {', '.join(flipped) if flipped else '(none)'}")
print(f"  combined weight of inverted components: {sum(W[k] for k in flipped):.2f}\n")


# ── STEP 2: build composites (IS-derived params only), evaluate OOS ───────────
def zscore(g, cols):
    out = g.copy()
    for c in cols:
        s = out[c]
        sd = s.std()
        out[c + "_z"] = (s - s.mean()) / sd if sd and sd == sd else 0.0
    return out


df2 = df.groupby("date", group_keys=False).apply(lambda g: zscore(g, SUBS))
zc = [k + "_z" for k in SUBS]

df2["corrected"] = sum(df2[k + "_z"] * signs[k] * W[k] for k in SUBS)
tot = sum(is_ic.values()) or 1.0
df2["icweighted"] = sum(df2[k + "_z"] * signs[k] * (is_ic[k] / tot) for k in SUBS)

OOS2 = df2[df2["date"] >= split]
IS2  = df2[df2["date"] < split]

print(f"STEP 2 — composite performance ({HORIZON}-day forward return)\n")
for label, frame in (("IN-SAMPLE", IS2), ("OUT-OF-SAMPLE", OOS2)):
    print(f"  --- {label} ---")
    print(summarise(per_date_ic(frame, "orig"),       "ORIGINAL"))
    print(summarise(per_date_ic(frame, "corrected"),  "SIGN-CORRECTED"))
    print(summarise(per_date_ic(frame, "icweighted"), "IC-WEIGHTED"))
    print(summarise(per_date_ic(frame, "pxema50"),    "PX/EMA50"))
    print()

# ── STEP 3: does it translate into a top-decile edge OOS? ─────────────────────
print("STEP 3 — OOS top-decile edge vs universe mean\n")
for col, label in (("orig", "ORIGINAL"), ("corrected", "SIGN-CORRECTED"),
                   ("icweighted", "IC-WEIGHTED"), ("pxema50", "PX/EMA50")):
    recs = []
    for dt, g in OOS2.groupby("date"):
        g = g.dropna(subset=[col, "fwd"])
        if len(g) < 20:
            continue
        k = max(1, int(round(len(g) * 0.10)))
        recs.append(g.nlargest(k, col)["fwd"].mean() - g["fwd"].mean())
    if len(recs) < 5:
        print(f"  {label:<16} insufficient dates"); continue
    a = np.array(recs)
    t, p = stats.ttest_1samp(a, 0.0)
    print(f"  {label:<16} edge {a.mean()*100:+.2f}%   t {t:+.2f}   p {p:.3f}   "
          f"hit {(a>0).mean()*100:.0f}%   n={len(a)}")
