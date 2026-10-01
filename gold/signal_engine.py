"""
A stoppable gold signal, walk-forward validated.

WHAT CHANGED FROM THE EARLIER RULE
    The original 16:00 ET rule held from the 16:00 close to the 18:00 close,
    spanning the CME settlement halt — and on Fridays, the entire weekend. Its
    edge split into a GAP component (unstoppable, market shut) and a BAR
    component (stoppable). Friday entries carried 60% of the worst losses while
    contributing no measurable edge.

    This engine trades only the BAR: enter at the 18:00 ET open, exit at the
    18:00 close or on a stop, Monday-Thursday only. Nothing is held through a
    halt or a weekend, so a stop can actually do its job.

VALIDATION — WALK-FORWARD, NOT A HOLDOUT
    The earlier holdout is spent, so re-using it would be a second look at data
    already consulted. Instead: for each year Y, parameters are chosen using
    ONLY years < Y, then applied to year Y untouched. Every reported trade is
    out-of-sample with respect to its own parameter choice. This cannot be
    gamed by picking the best stop in hindsight, which is exactly the failure
    mode that makes most published "tight stop" systems unreproducible.

STOP SIMULATION
    H1 bars give only the hour's high/low, so intrabar path is unknown. The
    conservative assumption is used throughout: if the bar's range touched the
    stop, the stop filled — even when the bar closed profitably. This
    understates results rather than flattering them.

Costs: broker-measured spread charged on entry and exit.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy import stats

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "xauusd_h1.csv")
SPREAD = 0.20
OZ = 100.0
ENTRY_HOUR = 18
STOPS = [0.25, 0.35, 0.50, 0.75, 1.00, 1.50, None]   # × ATR; None = no stop


def load():
    d = pd.read_csv(DATA, parse_dates=["dt"]).set_index("dt").sort_index()
    d.index = d.index.tz_convert("America/New_York")
    h, l, c = d["high"], d["low"], d["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 24, adjust=False).mean().shift(1)   # shift: causal
    d["ema200"] = c.ewm(span=200, adjust=False).mean().shift(1)
    f = d.reset_index().rename(columns={"dt": "ts"})
    f["h"] = f["ts"].dt.hour
    f["dow"] = f["ts"].dt.dayofweek
    sig = f[(f["h"] == ENTRY_HOUR) & (f["dow"] <= 3)].copy()        # Mon-Thu
    sig = sig.dropna(subset=["atr", "ema200"])
    sig = sig[sig["atr"] > 0]
    sig["year"] = sig["ts"].dt.year
    sig["above200"] = sig["close"] > sig["ema200"]
    return sig.reset_index(drop=True)


def run(sig: pd.DataFrame, stop_atr, trend_filter=False) -> np.ndarray:
    """Net $/oz per signal. Conservative: stop assumed hit if the range touched it."""
    s = sig[sig["above200"]] if trend_filter else sig
    o, hi, lo, c, a = (s["open"].to_numpy(), s["high"].to_numpy(), s["low"].to_numpy(),
                       s["close"].to_numpy(), s["atr"].to_numpy())
    entry = o + SPREAD / 2
    if stop_atr is None:
        exitp = c - SPREAD / 2
        return exitp - entry
    stop = entry - stop_atr * a
    hit = lo <= stop
    exitp = np.where(hit, stop, c) - SPREAD / 2
    return exitp - entry


def summarise(net: np.ndarray) -> dict:
    if len(net) < 10:
        return {}
    eq = np.cumsum(net * OZ)
    dd = eq - np.maximum.accumulate(eq)
    run_ = best = 0
    for x in net:
        run_ = 0 if x > 0 else run_ + 1
        best = max(best, run_)
    g, l = net[net > 0].sum(), -net[net < 0].sum()
    t, p = stats.ttest_1samp(net, 0.0)
    return {"n": len(net), "win": (net > 0).mean(), "mean_oz": net.mean(),
            "total_lot": eq[-1], "pf": g / l if l > 0 else np.inf,
            "maxdd_lot": dd.min(), "streak": best, "worst_lot": net.min() * OZ,
            "t": t, "p": p,
            "calmar": eq[-1] / abs(dd.min()) if dd.min() < 0 else np.inf}


def main():
    sig = load()
    print(f"{len(sig)} Mon-Thu 18:00 ET signals  "
          f"{sig['ts'].min():%Y-%m-%d} -> {sig['ts'].max():%Y-%m-%d}\n")

    # ── in-sample stop grid, for orientation only ────────────────────────────
    print("=" * 104)
    print("A) STOP GRID (whole sample — orientation only, NOT the result)")
    print("=" * 104)
    print(f"  {'stop':<10}{'n':>6}{'win%':>7}{'$/oz':>9}{'total/lot':>12}{'PF':>6}"
          f"{'maxDD/lot':>12}{'worst/lot':>12}{'streak':>8}{'t':>7}{'p':>8}")
    print("  " + "-" * 98)
    for st in STOPS:
        r = summarise(run(sig, st))
        if not r:
            continue
        lab = "none" if st is None else f"{st:.2f}×ATR"
        print(f"  {lab:<10}{r['n']:>6}{r['win']*100:>6.1f}%{r['mean_oz']:>+9.3f}"
              f"{r['total_lot']:>+12,.0f}{r['pf']:>6.2f}{r['maxdd_lot']:>+12,.0f}"
              f"{r['worst_lot']:>+12,.0f}{r['streak']:>8}{r['t']:>+7.2f}{r['p']:>8.4f}")

    # ── walk-forward: choose the stop on prior years only ────────────────────
    print("\n" + "=" * 104)
    print("B) WALK-FORWARD — stop chosen from PRIOR years only, applied to the next")
    print("=" * 104)
    years = sorted(sig["year"].unique())
    rows, oos = [], []
    for y in years[2:]:                       # need >=2 years of history to choose
        past = sig[sig["year"] < y]
        fut = sig[sig["year"] == y]
        if len(past) < 200 or len(fut) < 30:
            continue
        # choose the stop maximising Calmar (return / drawdown) on PAST data only
        best, best_st = -np.inf, None
        for st in STOPS:
            r = summarise(run(past, st))
            if r and r["total_lot"] > 0 and r["calmar"] > best:
                best, best_st = r["calmar"], st
        net = run(fut, best_st)
        r = summarise(net)
        oos.append(net)
        lab = "none" if best_st is None else f"{best_st:.2f}×ATR"
        rows.append({"year": y, "stop": lab, "n": r["n"], "win": r["win"],
                     "total": r["total_lot"], "t": r["t"]})
        print(f"  {y}  stop chosen from {sig[sig['year'] < y]['year'].min()}-{y-1}: "
              f"{lab:<9} -> n={r['n']:<4} win {r['win']*100:4.1f}%  "
              f"net ${r['total_lot']:+9,.0f}/lot")

    allnet = np.concatenate(oos)
    R = summarise(allnet)
    print("\n  " + "-" * 98)
    print(f"  COMBINED OUT-OF-SAMPLE: n={R['n']}  win {R['win']*100:.1f}%  "
          f"PF {R['pf']:.2f}  net ${R['total_lot']:+,.0f}/lot")
    print(f"  max drawdown ${R['maxdd_lot']:+,.0f}/lot   worst trade ${R['worst_lot']:+,.0f}   "
          f"longest losing streak {R['streak']}")
    print(f"  t={R['t']:+.2f}  p={R['p']:.4f}  Calmar {R['calmar']:.2f}")
    prof = sum(1 for r in rows if r["total"] > 0)
    print(f"  profitable years: {prof} of {len(rows)}")

    # ── trend filter, same walk-forward discipline ───────────────────────────
    print("\n" + "=" * 104)
    print("C) SAME, WITH TREND FILTER (only when price > EMA200)")
    print("=" * 104)
    oos2, rows2 = [], []
    for y in years[2:]:
        past, fut = sig[sig["year"] < y], sig[sig["year"] == y]
        if len(past) < 200 or len(fut) < 30:
            continue
        best, best_st = -np.inf, None
        for st in STOPS:
            r = summarise(run(past, st, trend_filter=True))
            if r and r["total_lot"] > 0 and r["calmar"] > best:
                best, best_st = r["calmar"], st
        net = run(fut, best_st, trend_filter=True)
        if len(net) < 10:
            continue
        oos2.append(net)
        rows2.append(summarise(net)["total_lot"])
    if oos2:
        a2 = np.concatenate(oos2)
        R2 = summarise(a2)
        print(f"  n={R2['n']}  win {R2['win']*100:.1f}%  PF {R2['pf']:.2f}  "
              f"net ${R2['total_lot']:+,.0f}/lot  maxDD ${R2['maxdd_lot']:+,.0f}")
        print(f"  worst ${R2['worst_lot']:+,.0f}  streak {R2['streak']}  "
              f"t={R2['t']:+.2f}  p={R2['p']:.4f}  Calmar {R2['calmar']:.2f}")
        print(f"  profitable years: {sum(1 for x in rows2 if x > 0)} of {len(rows2)}")

    # ── random control ───────────────────────────────────────────────────────
    print("\n" + "=" * 104)
    print("D) RANDOM CONTROL — same stop rules, random Mon-Thu hours")
    print("=" * 104)
    d = pd.read_csv(DATA, parse_dates=["dt"]).set_index("dt").sort_index()
    d.index = d.index.tz_convert("America/New_York")
    h, l, c = d["high"], d["low"], d["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 24, adjust=False).mean().shift(1)
    f = d.reset_index().rename(columns={"dt": "ts"})
    f["dow"] = f["ts"].dt.dayofweek
    pool = f[(f["dow"] <= 3) & f["atr"].notna() & (f["atr"] > 0)].reset_index(drop=True)
    rng = np.random.default_rng(3)
    means = []
    for _ in range(500):
        pick = pool.iloc[rng.choice(len(pool), size=len(sig), replace=False)]
        means.append(run(pick, 0.50).mean())
    means = np.array(means)
    actual = run(sig, 0.50).mean()
    pct = (means < actual).mean() * 100
    print(f"  random Mon-Thu hours, 0.50×ATR stop: mean ${means.mean():+.4f}/oz "
          f"(90% CI ${np.percentile(means,5):+.3f} to ${np.percentile(means,95):+.3f})")
    print(f"  18:00 ET signal:                     mean ${actual:+.4f}/oz  "
          f"-> {pct:.0f}th percentile")
    print(f"  verdict: {'BEATS random' if pct > 95 else 'indistinguishable from random'}")


if __name__ == "__main__":
    main()
