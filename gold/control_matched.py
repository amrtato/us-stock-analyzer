"""
Matched random control for the trend-filtered 18:00 ET variant.

signal_engine.py section D ran its control with a 0.50xATR stop, but the variant
that actually performed (section C) used NO stop plus a price>EMA200 filter. A
control has to match the thing it is controlling, or it proves nothing.

This runs the exact configuration — no stop, same EMA200 filter, same trade count
— on RANDOM Mon-Thu hours. The filter matters enormously here: gold trended up
across the sample, so "only trade when price is above its 200-EMA" is itself a
bullish selector. If random hours under the same filter earn what 18:00 earns,
the hour contributes nothing and we are just measuring the trend filter.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy import stats

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "xauusd_h1.csv")
SPREAD, OZ = 0.20, 100.0


def prep():
    d = pd.read_csv(DATA, parse_dates=["dt"]).set_index("dt").sort_index()
    d.index = d.index.tz_convert("America/New_York")
    h, l, c = d["high"], d["low"], d["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 24, adjust=False).mean().shift(1)
    d["ema200"] = c.ewm(span=200, adjust=False).mean().shift(1)
    f = d.reset_index().rename(columns={"dt": "ts"})
    f["h"] = f["ts"].dt.hour
    f["dow"] = f["ts"].dt.dayofweek
    f = f.dropna(subset=["atr", "ema200"])
    f = f[(f["dow"] <= 3) & (f["atr"] > 0)]
    f["above200"] = f["close"] > f["ema200"]
    return f.reset_index(drop=True)


def net_nostop(s):
    """No stop: enter at open, exit at close, spread both sides."""
    return (s["close"].to_numpy() - SPREAD / 2) - (s["open"].to_numpy() + SPREAD / 2)


def main():
    f = prep()
    sig = f[(f["h"] == 18) & f["above200"]]
    actual = net_nostop(sig)
    print(f"18:00 ET, Mon-Thu, price>EMA200, no stop")
    print(f"  n={len(actual)}  mean ${actual.mean():+.4f}/oz  "
          f"total ${actual.sum()*OZ:+,.0f}/lot  win {(actual>0).mean()*100:.1f}%\n")

    # Control 1: random hours, SAME trend filter
    pool = f[f["above200"]]
    rng = np.random.default_rng(17)
    m1 = []
    for _ in range(1000):
        pick = pool.iloc[rng.choice(len(pool), size=len(actual), replace=False)]
        m1.append(net_nostop(pick).mean())
    m1 = np.array(m1)
    p1 = (m1 < actual.mean()).mean() * 100

    # Control 2: EVERY hour individually, same filter — where does 18:00 rank?
    rows = []
    for hh in sorted(f["h"].unique()):
        s = f[(f["h"] == hh) & f["above200"]]
        if len(s) < 200:
            continue
        v = net_nostop(s)
        t, p = stats.ttest_1samp(v, 0.0)
        rows.append({"hour": hh, "n": len(v), "mean": v.mean(), "t": t, "p": p})
    r = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)

    print("=" * 84)
    print("CONTROL 1 — random Mon-Thu hours, same EMA200 filter, no stop")
    print("=" * 84)
    print(f"  random mean ${m1.mean():+.4f}/oz   90% CI "
          f"[${np.percentile(m1,5):+.4f}, ${np.percentile(m1,95):+.4f}]")
    print(f"  18:00 ET   ${actual.mean():+.4f}/oz   -> {p1:.0f}th percentile")
    print(f"  verdict: {'BEATS random' if p1 > 95 else 'INDISTINGUISHABLE from random'}")

    print("\n" + "=" * 84)
    print("CONTROL 2 — every hour under the same filter, ranked")
    print("=" * 84)
    print(f"  {'rank':<6}{'hour':<8}{'n':>6}{'mean $/oz':>12}{'t':>8}{'p':>9}")
    print("  " + "-" * 50)
    for i, x in r.iterrows():
        mark = "   <-- 18:00 ET" if int(x["hour"]) == 18 else ""
        if i < 6 or int(x["hour"]) == 18:
            print(f"  {i+1:<6}{int(x['hour']):02d}:00{'':<3}{int(x['n']):>6}"
                  f"{x['mean']:>+12.4f}{x['t']:>+8.2f}{x['p']:>9.4f}{mark}")
    pos = int(r.index[r["hour"] == 18][0]) + 1
    n_pos = (r["mean"] > 0).sum()
    bonf = 0.05 / len(r)
    print(f"\n  18:00 ET ranks {pos} of {len(r)} hours")
    print(f"  hours with positive mean: {n_pos} of {len(r)}")
    print(f"  hours significant raw p<0.05: {(r['p']<0.05).sum()} "
          f"(expect ~{0.05*len(r):.0f} by chance)")
    print(f"  Bonferroni p<{bonf:.4f}: {(r['p']<bonf).sum()} survive")
    print(f"\n  mean across ALL hours under this filter: ${r['mean'].mean():+.4f}/oz")
    print(f"  18:00 excess over the average hour:      "
          f"${actual.mean()-r['mean'].mean():+.4f}/oz")


if __name__ == "__main__":
    main()
