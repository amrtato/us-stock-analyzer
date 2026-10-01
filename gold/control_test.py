"""
Two controls that decide whether the event-study numbers mean anything.

CONTROL 1 — RANDOM ENTRY BASELINE.
  In event_study.py every LONG setup showed positive expectancy and every SHORT
  setup negative. Gold trended up strongly across the train window, so the
  obvious competing explanation is drift: longs won because gold rose, not
  because the setup selected anything. The test is to enter LONG on random bars
  with identical stop/target/cost rules. If random longs earn what fvg_up earns,
  the setup adds nothing. Any claimed edge must beat this bar, not beat zero.

CONTROL 2 — WIN RATE vs REWARD:RISK.
  The 70-80% win rate target is achievable on demand: move the target close
  enough to the entry and the hit rate rises toward 100%. What it does to
  expectancy is the whole question. This sweeps target distance and reports win
  rate AND expectancy together, so the trade-off is visible rather than asserted.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats

from gold.features import load, build
from gold.event_study import simulate, de_overlap

HOLDOUT_YEARS = 3
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".gold_panel.csv")
RNG = np.random.default_rng(42)


def main():
    f = (pd.read_csv(CACHE, index_col=0, parse_dates=True)
         if os.path.exists(CACHE) else build(load("10y")))
    raw = load("10y")
    f["High"], f["Low"] = raw["High"].reindex(f.index), raw["Low"].reindex(f.index)
    split = f.index.max() - pd.DateOffset(years=HOLDOUT_YEARS)
    tr = f[f.index <= split].reset_index(drop=True)

    ann = (tr["close"].iloc[-1] / tr["close"].iloc[0]) ** (252/len(tr)) - 1
    print(f"TRAIN {len(tr)} bars | gold annualised drift over window: {ann*100:+.1f}%/yr\n")

    # ── CONTROL 1 ────────────────────────────────────────────────────────────
    print("=" * 92)
    print("CONTROL 1 — random-entry baseline (1.5xATR stop, 2R target, costs charged)")
    print("=" * 92)
    valid = np.where(tr["atr_at_t"].notna() & (tr["atr_at_t"] > 0))[0]
    valid = valid[valid < len(tr) - 25]

    for direction, dname in ((+1, "LONG"), (-1, "SHORT")):
        means = []
        for _ in range(400):
            pick = np.sort(RNG.choice(valid, size=70, replace=False))
            pick = de_overlap(pick, 20)
            s = simulate(tr, pick, direction)
            if len(s) >= 10:
                means.append((s["R"].mean(), s["win"].mean()))
        m = np.array(means)
        lo, hi = np.percentile(m[:, 0], [5, 95])
        print(f"  random {dname:<6} avg R {m[:,0].mean():+.3f}   win {m[:,1].mean()*100:.0f}%   "
              f"90% CI of avg R: [{lo:+.2f}, {hi:+.2f}]")

    rnd_long = np.array([x[0] for x in means])  # last loop = SHORT; recompute LONG
    means_long = []
    for _ in range(400):
        pick = de_overlap(np.sort(RNG.choice(valid, size=70, replace=False)), 20)
        s = simulate(tr, pick, +1)
        if len(s) >= 10:
            means_long.append(s["R"].mean())
    means_long = np.array(means_long)

    print(f"\n  {'setup':<28}{'avg R':>8}{'vs random long':>16}{'percentile':>12}")
    print("  " + "-" * 66)
    for label, col, d in (("fvg_up", "s_fvg_up", +1), ("bos_up", "s_bos_up", +1),
                          ("breakout_hi", "s_bo_hi", +1), ("sweep_lo", "s_sweep_lo", +1)):
        ev = de_overlap(np.where(tr[col] == 1)[0], 20)
        s = simulate(tr, ev, d)
        if len(s) < 12:
            continue
        r = s["R"].mean()
        pct = (means_long < r).mean() * 100
        verdict = "beats random" if pct > 95 else ("indistinguishable" if pct > 5 else "worse")
        print(f"  {label:<28}{r:>+8.2f}{r - means_long.mean():>+16.2f}{pct:>11.0f}%   {verdict}")

    # ── CONTROL 2 ────────────────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("CONTROL 2 — win rate vs expectancy (fvg_up entries, 1.5xATR stop fixed)")
    print("=" * 92)
    print(f"  {'target':<10}{'R:R':>7}{'win%':>8}{'avg R':>9}{'PF':>7}{'total R':>10}  verdict")
    print("  " + "-" * 74)
    ev = de_overlap(np.where(tr["s_fvg_up"] == 1)[0], 20)
    for rr in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        s = simulate(tr, ev, +1, atr_stop=1.5, rr=rr)
        if len(s) < 12:
            continue
        R = s["R"]
        g, l = R[R > 0].sum(), -R[R < 0].sum()
        pf = g / l if l > 0 else np.inf
        w = s["win"].mean() * 100
        verdict = ("high win rate, LOSES money" if w >= 65 and R.mean() <= 0
                   else "high win rate, marginal" if w >= 65
                   else "")
        print(f"  {rr:.2f}xRisk{'':<2}{rr:>7.2f}{w:>8.0f}{R.mean():>+9.2f}{pf:>7.2f}{R.sum():>+10.1f}  {verdict}")

    print("\n  Note: win rate is a dial, not an achievement. Tightening the target raises it")
    print("  monotonically while expectancy falls — the two move in opposite directions.")


if __name__ == "__main__":
    main()
