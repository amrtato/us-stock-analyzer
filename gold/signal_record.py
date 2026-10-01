"""
The complete signal record for everything the research actually supports.

This is not a curated highlight reel. It is every signal the 16:00 ET rule would
have fired across 8 years, with the equity curve, the drawdowns, and the losing
streaks — because those are what decide whether a rule is survivable, and they
are exactly what gets omitted from signal-service marketing.

RULE (fixed, no parameters to tune):
    enter LONG at the close of the 16:00 ET bar
    exit  at the close of the 17:00 ET bar
    charge the broker's real per-bar spread
    one signal per trading day, no stop, no target, no discretion

Reported: full period, train vs holdout separately, equity, max drawdown,
longest losing run, and the trade-by-trade feed for the most recent signals.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy import stats

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "xauusd_h1.csv")
HOUR = 16
CONTRACT = 100.0          # XAUUSDm: 1.00 lot = 100 oz


def build():
    d = pd.read_csv(DATA, parse_dates=["dt"]).set_index("dt").sort_index()
    d.index = d.index.tz_convert("America/New_York")
    d["spread_usd"] = d["spread"] * 0.001
    f = d.reset_index().rename(columns={"dt": "ts"})
    f["hour"] = f["ts"].dt.hour
    idx = f.index[f["hour"] == HOUR].to_numpy()
    idx = idx[idx < len(f) - 1]
    t = pd.DataFrame({
        "ts": f["ts"].to_numpy()[idx],
        "entry": f["close"].to_numpy()[idx],
        "exit": f["close"].to_numpy()[idx + 1],
        "spread": f["spread_usd"].to_numpy()[idx],
    })
    t["gross"] = t["exit"] - t["entry"]
    t["net"] = t["gross"] - t["spread"]
    t["usd_per_lot"] = t["net"] * CONTRACT
    t["win"] = t["net"] > 0
    return t


def stats_block(t: pd.DataFrame, label: str):
    n = len(t)
    if n < 20:
        print(f"  {label}: too few"); return
    net = t["net"].to_numpy()
    eq = np.cumsum(t["usd_per_lot"].to_numpy())
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    # longest losing run
    run = best = 0
    for w in t["win"]:
        run = 0 if w else run + 1
        best = max(best, run)
    tstat, p = stats.ttest_1samp(net, 0.0)
    gains, losses = net[net > 0].sum(), -net[net < 0].sum()
    pf = gains / losses if losses > 0 else np.inf
    ann = eq[-1] / max((t["ts"].iloc[-1] - t["ts"].iloc[0]).days / 365.25, .01)
    print(f"  {label}")
    print(f"    signals {n:<6} {t['ts'].iloc[0]:%Y-%m-%d} -> {t['ts'].iloc[-1]:%Y-%m-%d}")
    print(f"    win rate      {t['win'].mean()*100:>6.1f}%     profit factor {pf:>5.2f}")
    print(f"    mean net      ${net.mean():>+6.3f}/oz   = ${net.mean()*CONTRACT:>+8.2f} per 1.0 lot")
    print(f"    total         ${eq[-1]:>+9,.0f} per lot   (${ann:>+8,.0f}/yr)")
    print(f"    MAX DRAWDOWN  ${dd.min():>+9,.0f} per lot   ({abs(dd.min())/max(eq[-1],1)*100:>5.0f}% of total gain)")
    print(f"    worst trade   ${net.min()*CONTRACT:>+9,.2f}   best ${net.max()*CONTRACT:>+9,.2f}")
    print(f"    longest losing streak: {best} consecutive")
    print(f"    t={tstat:+.2f}  p={p:.4f}")
    print()


def main():
    t = build()
    split = t["ts"].max() - pd.DateOffset(years=2)
    print("=" * 78)
    print("SIGNAL RECORD — 16:00 ET long, exit 17:00 ET, real spread")
    print("=" * 78)
    print(f"contract size {CONTRACT:.0f} oz  |  all $ figures per 1.00 lot\n")

    stats_block(t, "FULL PERIOD (8 years)")
    stats_block(t[t["ts"] <= split], "TRAIN 2018-08 -> 2024-08")
    stats_block(t[t["ts"] > split], "HOLDOUT 2024-08 -> 2026-08")

    # yearly
    print("=" * 78)
    print("YEAR BY YEAR (per 1.0 lot, net)")
    print("=" * 78)
    t["yr"] = t["ts"].dt.year
    yr = t.groupby("yr").agg(n=("net", "size"), win=("win", "mean"),
                             usd=("usd_per_lot", "sum"))
    print(f"  {'year':<7}{'signals':>9}{'win%':>8}{'net $/lot':>13}")
    print("  " + "-" * 38)
    for y, r in yr.iterrows():
        flag = "" if r["usd"] > 0 else "   <-- losing year"
        print(f"  {y:<7}{int(r['n']):>9}{r['win']*100:>7.1f}%{r['usd']:>+13,.0f}{flag}")

    # most recent signal feed
    print("\n" + "=" * 78)
    print("MOST RECENT 15 SIGNALS (what the feed would look like)")
    print("=" * 78)
    print(f"  {'date':<12}{'entry':>10}{'exit':>10}{'spread':>8}{'net $/oz':>11}{'per lot':>11}  result")
    print("  " + "-" * 72)
    for _, r in t.tail(15).iterrows():
        res = "WIN " if r["win"] else "LOSS"
        print(f"  {r['ts']:%Y-%m-%d}{r['entry']:>10.2f}{r['exit']:>10.2f}"
              f"{r['spread']:>8.3f}{r['net']:>+11.3f}{r['usd_per_lot']:>+11.2f}  {res}")

    # what a realistic account would experience
    print("\n" + "=" * 78)
    print("REALITY CHECK — 0.10 lot on a $10,000 account")
    print("=" * 78)
    ho = t[t["ts"] > split]
    lot = 0.10
    eq = np.cumsum(ho["usd_per_lot"].to_numpy() * lot)
    peak = np.maximum.accumulate(eq); dd = eq - peak
    notional = ho["entry"].iloc[-1] * CONTRACT * lot
    print(f"  notional per trade      ${notional:,.0f}  (leverage on $10k: {notional/10000:.1f}x)")
    print(f"  2-year net              ${eq[-1]:+,.0f}   ({eq[-1]/10000*100:+.1f}% of account)")
    print(f"  max drawdown            ${dd.min():+,.0f}   ({abs(dd.min())/10000*100:.1f}% of account)")
    print(f"  signals                 {len(ho)} over 2 years (~{len(ho)/24:.0f}/month)")


if __name__ == "__main__":
    main()
