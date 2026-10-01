"""
Session / killzone retest on 8 years of MT5 H1 — 4x the data of the yfinance run.

The earlier nulls (gold/sessions.py) used 2 years and ~500 observations per hour.
An underpowered null is weak evidence, so this repeats the same tests on 47,161
bars spanning 2018-2026, and charges the broker's OWN per-bar spread instead of
an assumed cost.

Timezone: MT5 stamps are server time. Verified as UTC by two independent checks —
the daily break lands at 21:00 (= 17:00 ET) and the volatility peak at raw hour 14
maps to 10:00 ET, matching the yfinance profile exactly.

TRAIN = 2018-08 to 2024-08. Final 2 years held out, not read here.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy import stats

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "xauusd_h1.csv")
HOLDOUT_YEARS = 2
RNG = np.random.default_rng(11)


def load():
    d = pd.read_csv(DATA, parse_dates=["dt"]).set_index("dt").sort_index()
    d.index = d.index.tz_convert("America/New_York")   # server=UTC, verified
    d["spread_usd"] = d["spread"] * 0.001
    h, l, c = d["high"], d["low"], d["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1/24, adjust=False).mean()
    return d.dropna(subset=["atr"])


def simulate(df, entries, direction, atr_stop=1.5, rr=2.0, max_bars=12):
    """Real per-bar spread charged on entry and exit. Stop assumed first on ties."""
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    a, sp = df["atr"].to_numpy(), df["spread_usd"].to_numpy()
    out = []
    for i in entries:
        if i + 1 >= len(c) or not np.isfinite(a[i]) or a[i] <= 0:
            continue
        risk = atr_stop * a[i]
        e = c[i] + direction * sp[i] / 2          # pay half-spread entering
        stop = e - direction * risk
        tgt = e + direction * risk * rr
        res = None
        for j in range(i + 1, min(i + 1 + max_bars, len(c))):
            hs = (l[j] <= stop) if direction > 0 else (h[j] >= stop)
            ht = (h[j] >= tgt) if direction > 0 else (l[j] <= tgt)
            if hs:
                res = -1.0; break
            if ht:
                res = rr; break
        if res is None:
            res = direction * (c[min(i + max_bars, len(c) - 1)] - e) / risk
        res -= (sp[i] / 2) / risk                  # pay half-spread exiting
        out.append(res)
    return np.array(out)


def control(df, valid, direction, n, reps=300):
    m = []
    for _ in range(reps):
        pick = np.sort(RNG.choice(valid, size=min(n, len(valid)), replace=False))
        r = simulate(df, pick, direction)
        if len(r) >= 10:
            m.append(r.mean())
    return np.array(m)


def main():
    d = load()
    split = d.index.max() - pd.DateOffset(years=HOLDOUT_YEARS)
    tr = d[d.index <= split]
    print(f"TRAIN {len(tr):,} H1 bars  {tr.index.min():%Y-%m-%d} -> {tr.index.max():%Y-%m-%d}")
    print(f"HOLDOUT after {split:%Y-%m-%d} — not read here")
    print(f"median spread ${tr['spread_usd'].median():.3f}\n")

    tr = tr.reset_index().rename(columns={"index": "ts", "dt": "ts"})
    tr["hour"] = tr["ts"].dt.hour
    tr["ret"] = tr["close"].pct_change()
    tr["fwd1"] = tr["close"].shift(-1) / tr["close"] - 1
    px = float(tr["close"].mean())
    msp = tr["spread_usd"].median()

    # ── 1. return by hour ────────────────────────────────────────────────────
    print("=" * 88)
    print("1) MEAN 1H FORWARD RETURN BY ET HOUR (8y) — Bonferroni across 23 hours")
    print("=" * 88)
    rows = []
    for h, g in tr.groupby("hour"):
        v = g["fwd1"].dropna()
        if len(v) < 200:
            continue
        t, p = stats.ttest_1samp(v, 0.0)
        rows.append({"h": h, "n": len(v), "usd": v.mean() * px, "t": t, "p": p})
    r = pd.DataFrame(rows).sort_values("t", key=abs, ascending=False)
    bonf = 0.05 / len(r)
    print(f"  {'hour':<7}{'n':>7}{'mean $':>10}{'vs spread':>11}{'t':>8}{'p':>9}  Bonf p<{bonf:.4f}")
    print("  " + "-" * 74)
    for _, x in r.head(8).iterrows():
        mark = "  ** SURVIVES" if x["p"] < bonf else ("  *" if x["p"] < 0.05 else "")
        print(f"  {int(x['h']):02d}:00 {int(x['n']):>7}{x['usd']:>+10.3f}"
              f"{abs(x['usd'])/msp:>10.2f}x{x['t']:>+8.2f}{x['p']:>9.3f}{mark}")
    print(f"\n  raw p<0.05: {(r['p']<0.05).sum()} of {len(r)} (expect ~{0.05*len(r):.0f})"
          f" | survive Bonferroni: {(r['p']<bonf).sum()}")

    # ── 2. killzones, real spread, random control ────────────────────────────
    valid = np.where(tr["atr"].notna() & (tr["atr"] > 0))[0]
    valid = valid[valid < len(tr) - 15]
    print("\n" + "=" * 88)
    print("2) KILLZONE TRADES (8y, real spread, vs random-entry control)")
    print("=" * 88)
    print(f"  {'killzone':<20}{'dir':>5}{'n':>6}{'win%':>7}{'avg R':>8}{'PF':>7}{'rand':>8}{'pct':>6}  verdict")
    print("  " + "-" * 78)
    for kname, hr in (("London KZ 02:00", 2), ("NY KZ 07:00", 7),
                      ("NY open 09:00", 9), ("London close 10:00", 10),
                      ("COMEX settle 13:00", 13)):
        idx = tr.index[tr["hour"] == hr].to_numpy()
        idx = idx[idx < len(tr) - 15]
        for dirn in (+1, -1):
            R = simulate(tr, idx, dirn)
            if len(R) < 50:
                continue
            g_, l_ = R[R > 0].sum(), -R[R < 0].sum()
            pf = g_ / l_ if l_ > 0 else np.inf
            ctrl = control(tr, valid, dirn, len(R))
            pct = (ctrl < R.mean()).mean() * 100
            v = "BEATS random" if pct > 95 and R.mean() > 0 else (
                "beats random but NEGATIVE exp" if pct > 95 else "indistinguishable")
            print(f"  {kname:<20}{dirn:>+5}{len(R):>6}{(R>0).mean()*100:>6.0f}%"
                  f"{R.mean():>+8.2f}{pf:>7.2f}{ctrl.mean():>+8.2f}{pct:>5.0f}%  {v}")

    # ── 3. volatility profile (the thing that DID replicate) ────────────────
    print("\n" + "=" * 88)
    print("3) VOLATILITY BY HOUR (8y) — for the risk tool")
    print("=" * 88)
    ov = tr["ret"].std()
    prof = tr.groupby("hour")["ret"].agg(["std", "count"])
    prof["x"] = prof["std"] / ov
    prof["atr_usd"] = tr.groupby("hour")["atr"].median()
    out = os.path.join(os.path.dirname(DATA), "vol_profile.csv")
    prof.to_csv(out)
    for h, row in prof.iterrows():
        bar = "#" * int(round(row["x"] * 20))
        print(f"  {h:02d}:00 ET  {row['x']:>5.2f}x  median ATR ${row['atr_usd']:>6.2f}  {bar}")
    print(f"\n  saved {out}")


if __name__ == "__main__":
    main()
