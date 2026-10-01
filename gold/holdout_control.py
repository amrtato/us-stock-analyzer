"""
Drift control for the 16:00 ET holdout result.

holdout_1600et.py reported +$2.01/hr net, t=+2.41, p=0.016 — and my script
printed "SURVIVES". Two problems with accepting that:

  1. The TRAIN net figure was +$0.0864, t=+0.94, p=0.345 — NOT significant once
     the real spread is charged. The "1.64x spread" headline came from the GROSS
     number. So the hypothesis never actually cleared costs in-sample.

  2. The holdout effect is 23x the train effect. Gold went from ~$2,400 to
     ~$4,250 across that window. In a strong bull market EVERY hour drifts up,
     so a long-only bias at any hour will look profitable.

The control: compare 16:00 ET against ALL OTHER HOURS in the same holdout window.
If 16:00 is merely one of 23 rising hours, it has no special property — the
"edge" is just being long gold in 2024-2026.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy import stats

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "xauusd_h1.csv")
HOLDOUT_YEARS = 2


def main():
    d = pd.read_csv(DATA, parse_dates=["dt"]).set_index("dt").sort_index()
    d.index = d.index.tz_convert("America/New_York")
    d["spread_usd"] = d["spread"] * 0.001
    split = d.index.max() - pd.DateOffset(years=HOLDOUT_YEARS)
    ho = d[d.index > split].reset_index().rename(columns={"dt": "ts"})
    ho["hour"] = ho["ts"].dt.hour

    ann = (ho["close"].iloc[-1] / ho["close"].iloc[0]) ** (1 / HOLDOUT_YEARS) - 1
    print(f"HOLDOUT {ho['ts'].iloc[0]:%Y-%m-%d} -> {ho['ts'].iloc[-1]:%Y-%m-%d}")
    print(f"gold ${ho['close'].iloc[0]:,.0f} -> ${ho['close'].iloc[-1]:,.0f}  "
          f"({ann*100:+.1f}%/yr)\n")

    rows = []
    c = ho["close"].to_numpy(); sp = ho["spread_usd"].to_numpy()
    for h in range(24):
        idx = ho.index[ho["hour"] == h].to_numpy()
        idx = idx[idx < len(ho) - 1]
        if len(idx) < 100:
            continue
        net = (c[idx + 1] - c[idx]) - sp[idx]
        t, p = stats.ttest_1samp(net, 0.0)
        rows.append({"hour": h, "n": len(net), "net": net.mean(),
                     "t": t, "p": p, "hit": (net > 0).mean()})
    r = pd.DataFrame(rows).sort_values("net", ascending=False)

    print("=" * 78)
    print("EVERY HOUR, long-only, net of real spread — SAME holdout window")
    print("=" * 78)
    print(f"  {'rank':<6}{'hour':<8}{'n':>6}{'net $':>10}{'t':>8}{'p':>9}{'hit%':>8}")
    print("  " + "-" * 62)
    for i, (_, x) in enumerate(r.iterrows(), 1):
        mark = "   <-- our hypothesis" if int(x["hour"]) == 16 else ""
        print(f"  {i:<6}{int(x['hour']):02d}:00{'':<3}{int(x['n']):>6}{x['net']:>+10.3f}"
              f"{x['t']:>+8.2f}{x['p']:>9.3f}{x['hit']*100:>7.1f}%{mark}")

    rank16 = int((r["hour"] == 16).idxmax())
    pos = list(r["hour"]).index(16) + 1
    n_pos = (r["net"] > 0).sum()
    n_sig = (r["p"] < 0.05).sum()
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  16:00 ET ranks {pos} of {len(r)} hours by net return")
    print(f"  hours with POSITIVE net return: {n_pos} of {len(r)}")
    print(f"  hours significant at raw p<0.05: {n_sig} (expect ~{0.05*len(r):.0f} by chance)")
    print(f"  Bonferroni threshold: p < {0.05/len(r):.4f}  -> "
          f"{(r['p'] < 0.05/len(r)).sum()} hour(s) survive")
    mean_all = r["net"].mean()
    print(f"\n  mean net across ALL hours: ${mean_all:+.3f}")
    print(f"  16:00 ET net:              ${r.loc[r['hour']==16,'net'].iloc[0]:+.3f}")
    print(f"  excess over the average hour: "
          f"${r.loc[r['hour']==16,'net'].iloc[0] - mean_all:+.3f}")


if __name__ == "__main__":
    main()
