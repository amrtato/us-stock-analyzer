"""
Phase 1 step 1: does ANY individual feature predict gold's forward return?

Single-asset testing differs from the cross-sectional stock work. There is no
universe to rank against, so "IC" here is a TIME-SERIES correlation between a
feature and the forward return, and two traps have to be handled explicitly:

  1. OVERLAP. 20-day forward returns sampled daily share 19 of 20 days. That
     inflates every t-stat roughly 4-5x. Headline numbers below are computed on
     NON-OVERLAPPING samples (every h-th bar), which is why n looks small — it
     is the honest n.

  2. LOOK-AHEAD IN NORMALISATION. Bucketing a feature by its whole-sample
     quintiles leaks the future. Buckets here use an EXPANDING percentile, so
     each day is ranked only against days that preceded it.

Nothing is combined yet. Combining before knowing which parts carry signal is
exactly how the stock scorer ended up blending a mean-reversion half against a
momentum half and netting zero.

TRAIN window only. The last 3 years are held out and not touched here.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats

from gold.features import load, build, feature_names, FEATURE_GROUPS

HOLDOUT_YEARS = 3
HORIZONS = [5, 10, 20]
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".gold_panel.csv")


def expanding_pct_rank(s: pd.Series, min_obs: int = 250) -> pd.Series:
    """Percentile of each value against ONLY prior values (causal)."""
    out = np.full(len(s), np.nan)
    vals = s.to_numpy(dtype=float)
    for i in range(min_obs, len(s)):
        past = vals[:i]
        past = past[~np.isnan(past)]
        if len(past) >= min_obs and not np.isnan(vals[i]):
            out[i] = (vals[i] > past).mean()
    return pd.Series(out, index=s.index)


def main():
    if os.path.exists(CACHE):
        f = pd.read_csv(CACHE, index_col=0, parse_dates=True)
        print(f"loaded cached panel: {f.shape[0]} bars, {f.shape[1]} cols")
    else:
        print("downloading gold + cross-asset drivers ...")
        raw = load("10y")
        f = build(raw)
        f.to_csv(CACHE)
        print(f"built panel: {f.shape[0]} bars, {f.shape[1]} cols")

    split = f.index.max() - pd.DateOffset(years=HOLDOUT_YEARS)
    train = f[f.index <= split]
    print(f"TRAIN  {train.index.min().date()} to {train.index.max().date()}  "
          f"({len(train)} bars)")
    print(f"HOLDOUT {split.date()} onward — NOT touched in this script\n")

    feats = [c for c in feature_names(train) if c not in ("z_dow", "z_month")]
    results = []

    for hz in HORIZONS:
        tgt = f"fwd{hz}"
        sub = train.dropna(subset=[tgt])
        # non-overlapping sample
        nol = sub.iloc[::hz]
        for name in feats:
            x = nol[name]
            y = nol[tgt]
            ok = x.notna() & y.notna()
            if ok.sum() < 40 or x[ok].nunique() < 4:
                continue
            xv, yv = x[ok], y[ok]
            # binary/flag features -> compare event vs non-event
            if set(np.unique(xv.to_numpy())) <= {0.0, 1.0}:
                a, b = yv[xv == 1], yv[xv == 0]
                if len(a) < 15 or len(b) < 15:
                    continue
                t, p = stats.ttest_ind(a, b, equal_var=False)
                results.append({"h": hz, "feature": name, "kind": "event",
                                "n": int(len(a)), "effect": a.mean() - b.mean(),
                                "t": t, "p": p,
                                "hit": float((a > 0).mean()),
                                "base": float((b > 0).mean())})
            else:
                rho, p = stats.spearmanr(xv, yv)
                # top vs bottom tercile by causal expanding rank
                rk = expanding_pct_rank(sub[name]).reindex(xv.index)
                hi, lo = yv[rk >= 0.7], yv[rk <= 0.3]
                eff = (hi.mean() - lo.mean()) if len(hi) > 10 and len(lo) > 10 else np.nan
                results.append({"h": hz, "feature": name, "kind": "cont",
                                "n": int(ok.sum()), "effect": eff,
                                "t": rho * np.sqrt(max(ok.sum() - 2, 1)) / np.sqrt(max(1 - rho**2, 1e-9)),
                                "p": p, "hit": np.nan, "base": np.nan, "rho": rho})

    res = pd.DataFrame(results)
    res.to_csv(os.path.join(os.path.dirname(CACHE), ".gold_feature_scan.csv"), index=False)

    for hz in HORIZONS:
        r = res[res.h == hz].copy()
        r["absS"] = r["t"].abs()
        r = r.sort_values("absS", ascending=False).head(14)
        print("=" * 92)
        print(f"HORIZON {hz}d — top features by |t| (TRAIN, non-overlapping n)")
        print("=" * 92)
        print(f"  {'feature':<24}{'kind':<7}{'n':>5}{'effect':>10}{'rho':>8}{'t':>8}{'p':>8}{'hit%':>7}{'base%':>7}")
        print("  " + "-" * 86)
        for _, x in r.iterrows():
            rho = f"{x.get('rho', np.nan):+.3f}" if pd.notna(x.get("rho", np.nan)) else "   —  "
            eff = f"{x['effect']*100:+.2f}%" if pd.notna(x["effect"]) else "    —"
            hit = f"{x['hit']*100:.0f}" if pd.notna(x["hit"]) else "  —"
            bse = f"{x['base']*100:.0f}" if pd.notna(x["base"]) else "  —"
            star = " *" if x["p"] < 0.05 else ""
            print(f"  {x['feature']:<24}{x['kind']:<7}{x['n']:>5}{eff:>10}{rho:>8}"
                  f"{x['t']:>+8.2f}{x['p']:>8.3f}{hit:>7}{bse:>7}{star}")
        print()

    # how many would we expect by chance?
    n_tests = len(res)
    sig = (res["p"] < 0.05).sum()
    print("=" * 92)
    print(f"MULTIPLE-COMPARISON CHECK: {n_tests} tests run, {sig} significant at p<0.05.")
    print(f"  Expected by chance alone: ~{0.05*n_tests:.0f}.")
    print(f"  Bonferroni threshold for this many tests: p < {0.05/max(n_tests,1):.5f}")
    strong = res[res["p"] < 0.05/max(n_tests, 1)]
    if len(strong):
        print("  Survives Bonferroni:")
        for _, x in strong.iterrows():
            print(f"    {x['feature']:<24} h={x['h']:<3} p={x['p']:.2e}")
    else:
        print("  NOTHING survives Bonferroni correction.")
    print("=" * 92)


if __name__ == "__main__":
    main()
