"""
Is there ANY short-horizon predictability in gold, and is it bigger than the spread?

This is the decisive economic question for scalping, and it comes before any
setup design. Predictability is necessary but not sufficient: an edge that moves
price $0.05 in your favour is worthless against a $0.20 round-trip spread. So
every number here is reported in DOLLARS and as a MULTIPLE OF SPREAD, never as a
correlation alone — a statistically significant correlation that cannot pay the
spread is not a trading edge.

Three scans, on 4 years of real M1 with the broker's own per-bar spread:

  1. LEAD-LAG        does an N-minute move predict the next M minutes?
                     (positive = momentum/continuation, negative = reversal)
  2. TIME OF DAY     minute-resolution session structure, 4y, Bonferroni-corrected
  3. VOL-CONDITIONED does predictability appear only in certain volatility states?

TRAIN = first 3 years. The final year is held out and not read here.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
from scipy import stats

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "xauusd_m1.csv")
HOLDOUT_DAYS = 365


def load():
    d = pd.read_csv(DATA, parse_dates=["dt"])
    d["spread_usd"] = d["spread"] * 0.001
    d = d.set_index("dt").sort_index()
    return d


def main():
    d = load()
    split = d.index.max() - pd.Timedelta(days=HOLDOUT_DAYS)
    tr = d[d.index <= split]
    print(f"TRAIN {len(tr):,} M1 bars  {tr.index.min():%Y-%m-%d} -> {tr.index.max():%Y-%m-%d}")
    print(f"HOLDOUT after {split:%Y-%m-%d} — not read in this script")
    med_sp = tr["spread_usd"].median()
    print(f"median spread ${med_sp:.3f}  (round trip ${med_sp:.3f}, paid once entry->exit)\n")

    c = tr["close"]

    # ── 1. LEAD-LAG ──────────────────────────────────────────────────────────
    print("=" * 94)
    print("1) LEAD-LAG — past move vs next move.  'edge $' = mean forward move")
    print("   conditional on a 1-sigma past move.  Compare against spread.")
    print("=" * 94)
    print(f"  {'look':>5}{'fwd':>5}{'corr':>9}{'t':>8}{'p':>9}{'edge $':>10}{'vs spread':>11}  verdict")
    print("  " + "-" * 88)
    rows = []
    for lb in (1, 5, 15, 30, 60):
        past = c.pct_change(lb)
        for fw in (1, 5, 15, 30, 60):
            fwd = c.shift(-fw) / c - 1
            # non-overlapping: sample every max(lb,fw) bars
            step = max(lb, fw)
            s = pd.DataFrame({"p": past, "f": fwd}).dropna().iloc[::step]
            if len(s) < 500:
                continue
            r, p = stats.pearsonr(s["p"], s["f"])
            t = r * np.sqrt(len(s) - 2) / np.sqrt(max(1 - r**2, 1e-12))
            # economic size: forward $ move implied by a 1-sd past move
            sd_past = s["p"].std()
            edge_ret = r * (s["f"].std() / sd_past) * sd_past   # = r * sd_fwd
            edge_usd = abs(edge_ret) * float(c.mean())
            ratio = edge_usd / med_sp
            verdict = ("TRADEABLE" if ratio > 1.5 and p < 0.01
                       else "too small vs spread" if p < 0.01 else "no signal")
            rows.append(p)
            print(f"  {lb:>5}{fw:>5}{r:>+9.4f}{t:>+8.2f}{p:>9.3f}{edge_usd:>10.3f}"
                  f"{ratio:>10.2f}x  {verdict}")

    n = len(rows)
    print(f"\n  {n} tests; Bonferroni p<{0.05/n:.5f}; "
          f"raw p<0.05: {sum(1 for x in rows if x < 0.05)} (expect ~{0.05*n:.0f})")

    # ── 2. TIME OF DAY ───────────────────────────────────────────────────────
    print("\n" + "=" * 94)
    print("2) TIME OF DAY — mean 15-min forward move by ET hour (4y, minute data)")
    print("=" * 94)
    et = tr.tz_convert("America/New_York") if tr.index.tz else tr.tz_localize("UTC").tz_convert("America/New_York")
    fwd15 = (et["close"].shift(-15) / et["close"] - 1)
    hh = et.index.hour
    res = []
    for h in range(24):
        m = (hh == h) & fwd15.notna()
        v = fwd15[m].iloc[::15]        # non-overlapping within the hour
        if len(v) < 200:
            continue
        t, p = stats.ttest_1samp(v, 0.0)
        res.append({"h": h, "n": len(v), "usd": v.mean() * float(c.mean()), "t": t, "p": p})
    r = pd.DataFrame(res).sort_values("t", key=abs, ascending=False)
    bonf = 0.05 / len(r)
    print(f"  {'hour':<7}{'n':>7}{'mean $':>10}{'vs spread':>11}{'t':>8}{'p':>9}   Bonferroni p<{bonf:.4f}")
    print("  " + "-" * 76)
    for _, x in r.head(8).iterrows():
        mark = "  ** SURVIVES" if x["p"] < bonf else ("  *" if x["p"] < 0.05 else "")
        print(f"  {int(x['h']):02d}:00 {int(x['n']):>7}{x['usd']:>+10.3f}"
              f"{abs(x['usd'])/med_sp:>10.2f}x{x['t']:>+8.2f}{x['p']:>9.3f}{mark}")
    print(f"\n  survive Bonferroni: {(r['p'] < bonf).sum()} of {len(r)} hours")

    # ── 3. VOL-CONDITIONED REVERSAL ──────────────────────────────────────────
    print("\n" + "=" * 94)
    print("3) VOL-CONDITIONED — 15m reversal edge within volatility terciles")
    print("=" * 94)
    rv = c.pct_change().rolling(60).std()
    rank = rv.rolling(5000, min_periods=1000).rank(pct=True)
    past5 = c.pct_change(5)
    fwd15b = c.shift(-15) / c - 1
    s = pd.DataFrame({"p": past5, "f": fwd15b, "v": rank}).dropna().iloc[::15]
    print(f"  {'vol regime':<16}{'n':>8}{'corr':>9}{'t':>8}{'p':>9}{'edge $':>10}{'vs spread':>11}")
    print("  " + "-" * 74)
    for lab, lo, hi in (("low (<33%)", 0, .33), ("mid", .33, .67), ("high (>67%)", .67, 1.01)):
        m = s[(s["v"] >= lo) & (s["v"] < hi)]
        if len(m) < 300:
            continue
        r_, p_ = stats.pearsonr(m["p"], m["f"])
        t_ = r_ * np.sqrt(len(m) - 2) / np.sqrt(max(1 - r_**2, 1e-12))
        e = abs(r_ * m["f"].std()) * float(c.mean())
        print(f"  {lab:<16}{len(m):>8}{r_:>+9.4f}{t_:>+8.2f}{p_:>9.3f}{e:>10.3f}{e/med_sp:>10.2f}x")


if __name__ == "__main__":
    main()
