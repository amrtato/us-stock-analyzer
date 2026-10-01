"""
HOLDOUT TEST — single pre-registered hypothesis. Run once.

HYPOTHESIS (fixed before looking at holdout data, derived from TRAIN only):
    Gold has a positive drift during the 16:00-17:00 ET hour — the last hour
    before the daily settlement break. Long entry at 16:00 ET, exit at 17:00 ET.

EVIDENCE FROM TRAIN (2018-08 to 2024-08, H1):
    +$0.329/hr, t=+3.64, survives Bonferroni across all 23 hours, 1.64x spread.
    Independently flagged on M1 data (+$0.146 over 15min, t=+3.98).

WHY IT IS PLAUSIBLE RATHER THAN JUST A SURVIVING p-VALUE:
    16:00-17:00 ET precedes the CME daily halt. End-of-session positioning and
    settlement flow is a real mechanism, and the hour is one of the LOWEST
    volatility hours (0.55x) — a drift showing up in a quiet window has better
    signal-to-noise than one riding a volatility spike.

RULES — no variants, no optimisation, no second look:
    * entry: first H1 bar stamped 16:00 ET, at close, paying half the real spread
    * exit:  next bar's close (17:00 ET), paying half the real spread
    * no stop, no target — this tests the DRIFT itself, not a trade dressed on top
    * report mean $, t-stat, hit rate, and the same figure net of cost

A single test. Whatever it says is the answer.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy import stats

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "xauusd_h1.csv")
HOLDOUT_YEARS = 2
HOUR = 16


def main():
    d = pd.read_csv(DATA, parse_dates=["dt"]).set_index("dt").sort_index()
    d.index = d.index.tz_convert("America/New_York")
    d["spread_usd"] = d["spread"] * 0.001

    split = d.index.max() - pd.DateOffset(years=HOLDOUT_YEARS)
    tr = d[d.index <= split]
    ho = d[d.index > split]

    def measure(frame, label):
        f = frame.reset_index().rename(columns={"dt": "ts"})
        f["hour"] = f["ts"].dt.hour
        idx = f.index[f["hour"] == HOUR].to_numpy()
        idx = idx[idx < len(f) - 1]
        if len(idx) < 30:
            print(f"  {label}: too few observations"); return None
        entry = f["close"].to_numpy()[idx]
        exit_ = f["close"].to_numpy()[idx + 1]
        sp = f["spread_usd"].to_numpy()[idx]
        gross = exit_ - entry
        net = gross - sp                      # half-spread each side = one full spread
        t, p = stats.ttest_1samp(net, 0.0)
        tg, pg = stats.ttest_1samp(gross, 0.0)
        print(f"  {label}")
        print(f"    n={len(net)}   {f['ts'].iloc[idx[0]]:%Y-%m-%d} -> {f['ts'].iloc[idx[-1]]:%Y-%m-%d}")
        print(f"    GROSS  mean ${gross.mean():+.4f}  t={tg:+.2f}  p={pg:.4f}  "
              f"hit {(gross>0).mean()*100:.1f}%")
        print(f"    NET    mean ${net.mean():+.4f}  t={t:+.2f}  p={p:.4f}  "
              f"hit {(net>0).mean()*100:.1f}%")
        print(f"    median spread ${np.median(sp):.3f}   "
              f"total net ${net.sum():+,.2f} per 1.0 lot-equivalent unit")
        return {"n": len(net), "net_mean": net.mean(), "t": t, "p": p,
                "hit": (net > 0).mean(), "net": net}

    print("=" * 78)
    print("TRAIN (already seen — shown only for comparison)")
    print("=" * 78)
    a = measure(tr, "2018-08 to 2024-08")

    print("\n" + "=" * 78)
    print("HOLDOUT (first and only look)")
    print("=" * 78)
    b = measure(ho, f"{split:%Y-%m} to {d.index.max():%Y-%m}")

    if a and b:
        print("\n" + "=" * 78)
        print("VERDICT")
        print("=" * 78)
        same_sign = np.sign(a["net_mean"]) == np.sign(b["net_mean"])
        print(f"  train net ${a['net_mean']:+.4f}   holdout net ${b['net_mean']:+.4f}"
              f"   sign held: {same_sign}")
        print(f"  holdout t={b['t']:+.2f}  p={b['p']:.4f}  n={b['n']}")
        if b["p"] < 0.05 and same_sign and b["net_mean"] > 0:
            print("\n  -> SURVIVES out of sample, net of real spread.")
        elif same_sign and b["net_mean"] > 0:
            print("\n  -> Same direction and positive, but NOT significant out of sample.")
            print("     Consistent with a real-but-small effect, or with chance. Not")
            print("     sufficient on its own to trade.")
        else:
            print("\n  -> FAILS out of sample. The train result was noise or regime-specific.")


if __name__ == "__main__":
    main()
