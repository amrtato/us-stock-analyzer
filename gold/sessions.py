"""
Intraday session / killzone structure in gold — 2 years of hourly GC=F.

Two DIFFERENT questions, deliberately reported apart, because conflating them is
how session analysis usually oversells itself:

  VOLATILITY BY HOUR — almost certainly real. Intraday vol seasonality is one of
  the most robust effects in markets. It is NOT a directional edge; it tells you
  when to size down and where to put stops, not which way to bet.

  RETURN BY HOUR — the tradeable claim, and a far higher bar. Tested against
  Bonferroni across all 23 hours, because scanning hours and reporting the best
  one is guaranteed to produce a "winner" from noise alone.

Then the ICT-style session setups (Asian-range breakout, killzone bias) are run
as actual trades with costs, each against a random-entry control matched on
direction — the control that dissolved every daily ICT setup in control_test.py.

Sessions (ET, matching common ICT definitions):
  Asian range     19:00-02:00     London killzone 02:00-05:00
  NY killzone     07:00-10:00     London close    10:00-12:00
"""
from __future__ import annotations
import sys, os, warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".gold_1h.csv")
RNG = np.random.default_rng(7)
COST_ATR = 0.05          # spread+slippage as fraction of ATR, per side


def load_hourly() -> pd.DataFrame:
    if os.path.exists(CACHE):
        d = pd.read_csv(CACHE, index_col=0, parse_dates=True)
        d.index = pd.to_datetime(d.index, utc=True).tz_convert("America/New_York")
        return d
    d = yf.Ticker("GC=F").history(period="2y", interval="1h")
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d[["Open", "High", "Low", "Close", "Volume"]].dropna()
    d.index = d.index.tz_convert("America/New_York")
    d.to_csv(CACHE)
    return d


def atr(df: pd.DataFrame, n: int = 24) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def simulate(df: pd.DataFrame, entries, direction, atr_stop=1.5, rr=2.0, max_bars=12):
    h, l, c, a = (df["High"].to_numpy(), df["Low"].to_numpy(),
                  df["Close"].to_numpy(), df["atr"].to_numpy())
    out = []
    for i in entries:
        if i + 1 >= len(c) or not np.isfinite(a[i]) or a[i] <= 0:
            continue
        e = c[i]; risk = atr_stop * a[i]
        stop = e - direction * risk; tgt = e + direction * risk * rr
        res = None
        for j in range(i + 1, min(i + 1 + max_bars, len(c))):
            hs = (l[j] <= stop) if direction > 0 else (h[j] >= stop)
            ht = (h[j] >= tgt)  if direction > 0 else (l[j] <= tgt)
            if hs:                     # conservative on same-bar ambiguity
                res = -1.0; break
            if ht:
                res = rr; break
        if res is None:
            res = direction * (c[min(i + max_bars, len(c) - 1)] - e) / risk
        out.append(res - 2 * COST_ATR * a[i] / risk)
    return np.array(out)


def random_control(df, valid, direction, n, reps=400):
    means = []
    for _ in range(reps):
        pick = np.sort(RNG.choice(valid, size=min(n, len(valid)), replace=False))
        r = simulate(df, pick, direction)
        if len(r) >= 10:
            means.append(r.mean())
    return np.array(means)


def main():
    d = load_hourly()
    d["atr"] = atr(d)
    d["ret"] = d["Close"].pct_change()
    d["fwd1"] = d["Close"].shift(-1) / d["Close"] - 1
    d["hour"] = d.index.hour
    d["dow"] = d.index.dayofweek
    d["date"] = d.index.date
    d = d.dropna(subset=["ret"])
    print(f"{len(d)} hourly bars  {d.index[0].date()} to {d.index[-1].date()}\n")

    # ── 1. VOLATILITY BY HOUR ────────────────────────────────────────────────
    print("=" * 88)
    print("1) VOLATILITY BY HOUR (ET) — expected to be real; NOT a directional edge")
    print("=" * 88)
    g = d.groupby("hour")["ret"]
    vol = (g.std() * 100).sort_index()
    overall = d["ret"].std() * 100
    print(f"  overall hourly sigma {overall:.3f}%   (bar = relative to overall)")
    for h, v in vol.items():
        if h == 17:
            continue
        bar = "#" * int(round(v / overall * 22))
        print(f"    {h:02d}:00  sigma {v:.3f}%  {v/overall:>5.2f}x  {bar}")

    # ── 2. RETURN BY HOUR ────────────────────────────────────────────────────
    print("\n" + "=" * 88)
    print("2) MEAN RETURN BY HOUR — the tradeable claim (Bonferroni across 23 hours)")
    print("=" * 88)
    rows = []
    for h, grp in d.groupby("hour"):
        if h == 17 or len(grp) < 60:
            continue
        t, p = stats.ttest_1samp(grp["ret"].dropna(), 0.0)
        rows.append({"hour": h, "n": len(grp), "mean_bp": grp["ret"].mean() * 1e4,
                     "t": t, "p": p})
    r = pd.DataFrame(rows).sort_values("t", key=abs, ascending=False)
    bonf = 0.05 / len(r)
    print(f"  {'hour':<7}{'n':>6}{'mean(bp)':>11}{'t':>8}{'p':>9}   Bonferroni p<{bonf:.4f}")
    print("  " + "-" * 66)
    for _, x in r.head(8).iterrows():
        mark = " ** SURVIVES" if x["p"] < bonf else (" *" if x["p"] < 0.05 else "")
        print(f"  {int(x['hour']):02d}:00 {int(x['n']):>6}{x['mean_bp']:>+11.2f}"
              f"{x['t']:>+8.2f}{x['p']:>9.3f}{mark}")
    surv = (r["p"] < bonf).sum()
    print(f"\n  hours significant raw p<0.05: {(r['p']<0.05).sum()} of {len(r)} "
          f"(expect ~{0.05*len(r):.0f} by chance)")
    print(f"  hours surviving Bonferroni:   {surv}")

    # ── 3. ASIAN RANGE BREAKOUT ──────────────────────────────────────────────
    print("\n" + "=" * 88)
    print("3) ASIAN-RANGE BREAKOUT — break of 19:00-02:00 range during London/NY")
    print("=" * 88)
    ts = d.index                      # keep timestamps before dropping the index
    d = d.reset_index(drop=True)
    d["ts"] = ts
    d["sess_date"] = (d["ts"] - pd.Timedelta(hours=18)).dt.date   # session day starts 18:00 ET
    asian = d[(d["hour"] >= 19) | (d["hour"] <= 2)]
    rng = asian.groupby("sess_date").agg(a_hi=("High", "max"), a_lo=("Low", "min"),
                                         a_n=("High", "size"))
    rng = rng[rng["a_n"] >= 5]
    d = d.merge(rng, left_on="sess_date", right_index=True, how="left")

    valid_all = np.where(d["atr"].notna() & (d["atr"] > 0))[0]
    valid_all = valid_all[valid_all < len(d) - 15]

    print(f"  {'window':<22}{'dir':>5}{'n':>6}{'win%':>7}{'avg R':>8}{'PF':>7}"
          f"{'rand':>8}{'pct':>6}  verdict")
    print("  " + "-" * 80)
    for wname, lo, hi in (("London 03:00-06:00", 3, 6), ("NY 08:00-11:00", 8, 11)):
        win = d[(d["hour"] >= lo) & (d["hour"] <= hi) & d["a_hi"].notna()]
        for dirn, label, cond in ((+1, "break up", win["Close"] > win["a_hi"]),
                                  (-1, "break dn", win["Close"] < win["a_lo"])):
            idx = win[cond].index.to_numpy()
            # one trade per session day
            keep, seen = [], set()
            for i in idx:
                sd = d.loc[i, "sess_date"]
                if sd not in seen:
                    keep.append(i); seen.add(sd)
            keep = np.array(keep, dtype=int)
            if len(keep) < 20:
                print(f"  {wname+' '+label:<22}{dirn:>+5}{len(keep):>6}   too few")
                continue
            R = simulate(d, keep, dirn)
            if len(R) < 20:
                continue
            gains, losses = R[R > 0].sum(), -R[R < 0].sum()
            pf = gains / losses if losses > 0 else np.inf
            ctrl = random_control(d, valid_all, dirn, len(R))
            pct = (ctrl < R.mean()).mean() * 100
            verdict = "BEATS random" if pct > 95 else "indistinguishable"
            print(f"  {wname+' '+label:<22}{dirn:>+5}{len(R):>6}{(R>0).mean()*100:>6.0f}%"
                  f"{R.mean():>+8.2f}{pf:>7.2f}{ctrl.mean():>+8.2f}{pct:>5.0f}%  {verdict}")

    # ── 4. KILLZONE BIAS ─────────────────────────────────────────────────────
    print("\n" + "=" * 88)
    print("4) KILLZONE DIRECTIONAL BIAS — trades taken at killzone open")
    print("=" * 88)
    print(f"  {'killzone':<22}{'dir':>5}{'n':>6}{'win%':>7}{'avg R':>8}{'PF':>7}"
          f"{'rand':>8}{'pct':>6}  verdict")
    print("  " + "-" * 80)
    for kname, hr in (("London KZ 02:00", 2), ("NY KZ 07:00", 7), ("London close 10:00", 10)):
        idx = d.index[d["hour"] == hr].to_numpy()
        idx = idx[idx < len(d) - 15]
        for dirn in (+1, -1):
            R = simulate(d, idx, dirn)
            if len(R) < 30:
                continue
            gains, losses = R[R > 0].sum(), -R[R < 0].sum()
            pf = gains / losses if losses > 0 else np.inf
            ctrl = random_control(d, valid_all, dirn, len(R))
            pct = (ctrl < R.mean()).mean() * 100
            verdict = "BEATS random" if pct > 95 else "indistinguishable"
            print(f"  {kname:<22}{dirn:>+5}{len(R):>6}{(R>0).mean()*100:>6.0f}%"
                  f"{R.mean():>+8.2f}{pf:>7.2f}{ctrl.mean():>+8.2f}{pct:>5.0f}%  {verdict}")


if __name__ == "__main__":
    main()
