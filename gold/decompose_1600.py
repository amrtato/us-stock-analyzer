"""
Where does the 16:00 ET edge actually live — in the gap, or in the bar?

The rule enters at the 16:00 ET close and exits at the 18:00 ET close, because
there is no 17:00 bar (CME settlement halt). That total return has two parts:

    GAP  =  18:00 open / 16:00 close - 1     price moves while the market is shut
    BAR  =  18:00 close / 18:00 open - 1     price moves while you can trade

This distinction decides whether a stop is even possible:

  * If the edge is in the GAP, no stop can protect you — the market is closed
    while the move happens, and on Fridays the "gap" is the whole weekend. That
    is unmanaged overnight risk wearing a signal's clothing.

  * If the edge is in the BAR, it is capturable with a real stop, and a tight
    stop becomes a design choice rather than an impossibility.

Also splits out Friday entries, which hold across the entire weekend.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy import stats

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "xauusd_h1.csv")
SPREAD = 0.20
OZ = 100.0


def build():
    d = pd.read_csv(DATA, parse_dates=["dt"]).set_index("dt").sort_index()
    d.index = d.index.tz_convert("America/New_York")
    f = d.reset_index().rename(columns={"dt": "ts"})
    f["h"] = f["ts"].dt.hour
    i = f.index[f["h"] == 16].to_numpy()
    i = i[i < len(f) - 1]
    t = pd.DataFrame({
        "entry_ts": f["ts"].to_numpy()[i],
        "exit_ts": f["ts"].to_numpy()[i + 1],
        "c16": f["close"].to_numpy()[i],
        "o18": f["open"].to_numpy()[i + 1],
        "h18": f["high"].to_numpy()[i + 1],
        "l18": f["low"].to_numpy()[i + 1],
        "c18": f["close"].to_numpy()[i + 1],
    })
    t["dow"] = pd.to_datetime(t["entry_ts"]).dt.dayofweek        # 4 = Friday
    t["hold_h"] = (pd.to_datetime(t["exit_ts"]) - pd.to_datetime(t["entry_ts"])) / pd.Timedelta("1h")
    t["gap"] = t["o18"] - t["c16"]
    t["bar"] = t["c18"] - t["o18"]
    t["total"] = t["c18"] - t["c16"]
    return t


def block(x: np.ndarray, label: str, cost: float):
    net = x - cost
    tt, p = stats.ttest_1samp(net, 0.0)
    print(f"  {label:<34}n={len(net):<6}mean ${net.mean():+7.3f}/oz "
          f"= ${net.mean()*OZ:+8.2f}/lot  t={tt:+5.2f}  p={p:.4f}  "
          f"hit {(net>0).mean()*100:4.1f}%")
    return net


def main():
    t = build()
    print(f"{len(t)} signals  {t['entry_ts'].min():%Y-%m-%d} -> {t['entry_ts'].max():%Y-%m-%d}\n")

    fri = t["dow"] == 4
    print(f"Friday entries (hold across the weekend): {fri.sum()} of {len(t)} "
          f"({fri.mean()*100:.0f}%)")
    print(f"  median hold, Mon-Thu {t.loc[~fri,'hold_h'].median():.0f}h | "
          f"Friday {t.loc[fri,'hold_h'].median():.0f}h\n")

    print("=" * 96)
    print("A) DECOMPOSITION — all signals (spread charged once to the total only)")
    print("=" * 96)
    block(t["gap"].to_numpy(), "GAP  16:00 close -> 18:00 open", 0.0)
    block(t["bar"].to_numpy(), "BAR  18:00 open -> 18:00 close", 0.0)
    block(t["total"].to_numpy(), "TOTAL (net of spread)", SPREAD)

    print("\n" + "=" * 96)
    print("B) EXCLUDING FRIDAY (no weekend holds)")
    print("=" * 96)
    m = ~fri
    block(t.loc[m, "gap"].to_numpy(), "GAP", 0.0)
    block(t.loc[m, "bar"].to_numpy(), "BAR", 0.0)
    block(t.loc[m, "total"].to_numpy(), "TOTAL (net of spread)", SPREAD)

    print("\n" + "=" * 96)
    print("C) FRIDAY ONLY (the weekend-gap trades)")
    print("=" * 96)
    block(t.loc[fri, "gap"].to_numpy(), "GAP (across the weekend)", 0.0)
    block(t.loc[fri, "bar"].to_numpy(), "BAR", 0.0)
    block(t.loc[fri, "total"].to_numpy(), "TOTAL (net of spread)", SPREAD)

    print("\n" + "=" * 96)
    print("D) RISK CONCENTRATION — where do the worst losses come from?")
    print("=" * 96)
    t["total_net"] = t["total"] - SPREAD
    worst = t.nsmallest(10, "total_net")
    n_fri = (worst["dow"] == 4).sum()
    gap_share = (worst["gap"].abs() / (worst["gap"].abs() + worst["bar"].abs())).mean()
    print(f"  10 worst signals: {n_fri} are Friday entries "
          f"({n_fri/10*100:.0f}%, vs {fri.mean()*100:.0f}% of all signals)")
    print(f"  in those, the GAP accounts for {gap_share*100:.0f}% of the move on average")
    print(f"  worst single: ${worst['total_net'].iloc[0]*OZ:+,.2f}/lot on "
          f"{pd.to_datetime(worst['entry_ts'].iloc[0]):%Y-%m-%d (%a)}, "
          f"gap ${worst['gap'].iloc[0]:+.2f} / bar ${worst['bar'].iloc[0]:+.2f}")
    print()
    for lab, mask in (("all", slice(None)), ("Mon-Thu", ~fri), ("Friday", fri)):
        v = t.loc[mask, "total_net"].to_numpy() * OZ
        print(f"  {lab:<9} worst ${v.min():+10,.2f}   5th pct ${np.percentile(v,5):+9,.2f}   "
              f"sd ${v.std():+9,.2f}")


if __name__ == "__main__":
    main()
