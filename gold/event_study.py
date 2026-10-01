"""
Event study for the structural / ICT-style setups.

WHY THIS EXISTS: explore.py sampled every h-th bar to avoid overlapping forward
windows. That is right for continuous features but destroys sparse EVENTS — a
setup firing on 5% of bars leaves ~4 occurrences in an 88-sample non-overlapping
series, below any usable threshold, so every structure feature was skipped. The
absence of ICT features from that output was an artefact of my sampling, not
evidence about ICT.

Correct handling: keep EVERY occurrence, then de-overlap greedily — walk forward
and keep an event only if it is at least h bars after the last kept one. That
preserves event count while ensuring no two measured windows share bars.

Two views per setup:
  1. FORWARD RETURN  — does the setup predict direction?
  2. TRADE SIMULATION — with an ATR stop and target, what actually happens?
     This is the question a signal service has to answer. A setup can have no
     directional edge yet still be tradeable if its payoff is asymmetric, and it
     can have a high hit rate yet lose money if the stop is far and target near.
     Both are reported so neither can flatter the other.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats

from gold.features import load, build

HOLDOUT_YEARS = 3
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".gold_panel.csv")

SETUPS = {
    "sweep_hi (bearish stop-run)": ("s_sweep_hi", -1),
    "sweep_lo (bullish stop-run)": ("s_sweep_lo", +1),
    "breakout_hi":                 ("s_bo_hi",    +1),
    "breakout_lo":                 ("s_bo_lo",    -1),
    "fvg_up":                      ("s_fvg_up",   +1),
    "fvg_dn":                      ("s_fvg_dn",   -1),
    "bos_up":                      ("s_bos_up",   +1),
    "bos_dn":                      ("s_bos_dn",   -1),
    "inside_bar":                  ("s_inside",   +1),
}


def de_overlap(idx_positions: np.ndarray, gap: int) -> np.ndarray:
    keep, last = [], -10**9
    for i in idx_positions:
        if i - last >= gap:
            keep.append(i); last = i
    return np.array(keep, dtype=int)


def simulate(df: pd.DataFrame, entries: np.ndarray, direction: int,
             atr_stop: float = 1.5, rr: float = 2.0, max_bars: int = 20,
             cost_atr: float = 0.05):
    """
    Bar-by-bar walk: stop and target from ATR at entry, whichever is hit first.
    Conservative tie-break — if both levels fall inside the same bar's range we
    assume the STOP filled, since we cannot see intrabar order without tick data.
    Costs charged as a fraction of ATR (spread + slippage).
    """
    h, l, c = df["High"].to_numpy(), df["Low"].to_numpy(), df["close"].to_numpy()
    atr = df["atr_at_t"].to_numpy()
    out = []
    for i in entries:
        if i + 1 >= len(c) or not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        entry = c[i]
        risk  = atr_stop * atr[i]
        stop  = entry - direction * risk
        tgt   = entry + direction * risk * rr
        res, bars = None, 0
        for j in range(i + 1, min(i + 1 + max_bars, len(c))):
            bars = j - i
            hit_stop = (l[j] <= stop) if direction > 0 else (h[j] >= stop)
            hit_tgt  = (h[j] >= tgt)  if direction > 0 else (l[j] <= tgt)
            if hit_stop and hit_tgt:
                res = -1.0; break          # conservative: assume stop first
            if hit_stop:
                res = -1.0; break
            if hit_tgt:
                res = rr;  break
        if res is None:                     # timed out — mark to market in R
            res = direction * (c[min(i + max_bars, len(c) - 1)] - entry) / risk
        res -= cost_atr * atr[i] / risk * 2  # entry + exit cost, in R
        out.append({"R": res, "bars": bars, "win": res > 0})
    return pd.DataFrame(out)


def main():
    if os.path.exists(CACHE):
        f = pd.read_csv(CACHE, index_col=0, parse_dates=True)
    else:
        f = build(load("10y")); f.to_csv(CACHE)

    raw = load("10y")
    f["High"], f["Low"] = raw["High"].reindex(f.index), raw["Low"].reindex(f.index)

    split = f.index.max() - pd.DateOffset(years=HOLDOUT_YEARS)
    tr = f[f.index <= split].reset_index(drop=True)
    print(f"TRAIN {len(tr)} bars (holdout untouched)\n")

    print("=" * 100)
    print("A) DIRECTIONAL EDGE — forward return after setup, de-overlapped")
    print("=" * 100)
    print(f"  {'setup':<28}{'dir':>4}{'h':>4}{'n':>5}{'evt ret':>10}{'base':>9}{'diff':>9}{'t':>7}{'p':>7}")
    print("  " + "-" * 94)
    rowsA = []
    for label, (col, d) in SETUPS.items():
        if col not in tr:
            continue
        for hz in (5, 10, 20):
            tgt = f"fwd{hz}"
            valid = tr[col].notna() & tr[tgt].notna()
            ev = np.where(valid & (tr[col] == 1))[0]
            nv = np.where(valid & (tr[col] == 0))[0]
            ev, nv = de_overlap(ev, hz), de_overlap(nv, hz)
            if len(ev) < 12:
                continue
            a = d * tr[tgt].to_numpy()[ev]
            b = d * tr[tgt].to_numpy()[nv]
            t, p = stats.ttest_ind(a, b, equal_var=False)
            rowsA.append({"setup": label, "h": hz, "n": len(ev), "p": p})
            star = " *" if p < 0.05 else ""
            print(f"  {label:<28}{d:>+4}{hz:>4}{len(ev):>5}{a.mean()*100:>+9.2f}%"
                  f"{b.mean()*100:>+8.2f}%{(a.mean()-b.mean())*100:>+8.2f}%{t:>+7.2f}{p:>7.3f}{star}")
    print()

    print("=" * 100)
    print("B) TRADE SIMULATION — 1.5xATR stop, 2R target, 20-bar max, costs charged")
    print("=" * 100)
    print(f"  {'setup':<28}{'n':>5}{'win%':>7}{'avg R':>8}{'total R':>9}{'PF':>7}{'exp':>8}{'t':>7}{'p':>7}")
    print("  " + "-" * 94)
    for label, (col, d) in SETUPS.items():
        if col not in tr:
            continue
        ev = de_overlap(np.where(tr[col] == 1)[0], 20)
        if len(ev) < 12:
            print(f"  {label:<28}{len(ev):>5}   too few events")
            continue
        sim = simulate(tr, ev, d)
        if len(sim) < 12:
            continue
        R = sim["R"]
        gains, losses = R[R > 0].sum(), -R[R < 0].sum()
        pf = gains / losses if losses > 0 else np.inf
        t, p = stats.ttest_1samp(R, 0.0)
        star = " *" if p < 0.05 else ""
        print(f"  {label:<28}{len(sim):>5}{sim['win'].mean()*100:>6.0f}%{R.mean():>+8.2f}"
              f"{R.sum():>+9.1f}{pf:>7.2f}{R.mean():>+8.2f}{t:>+7.2f}{p:>7.3f}{star}")

    print()
    n_tests = len(rowsA) + len(SETUPS)
    print(f"MULTIPLE COMPARISONS: ~{n_tests} tests here. Expect ~{0.05*n_tests:.0f} false "
          f"positives at p<0.05. Bonferroni threshold p < {0.05/max(n_tests,1):.4f}")


if __name__ == "__main__":
    main()
