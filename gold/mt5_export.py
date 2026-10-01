"""
Download gold history from MT5 into local files for backtesting.

copy_rates_from_pos rejects large counts on this build ("Invalid params"), so
history is pulled with copy_rates_range in monthly chunks walking backwards —
which also reveals where the broker's archive actually ends.

Outputs (gold/data/):
    xauusd_m1.csv    minute bars, full available depth
    xauusd_m5.csv    5-minute bars
    xauusd_h1.csv    hourly bars (8y — deeper than yfinance's 2y)
    spread_stats.txt measured spread distribution

Read-only against MT5: rates and ticks only.
"""
from __future__ import annotations
import os, time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

TERMINAL = r"C:\Program Files\MetaTrader 5 (AMR)\terminal64.exe"
SYM = "XAUUSDm"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT, exist_ok=True)

TFS = {"m1": mt5.TIMEFRAME_M1, "m5": mt5.TIMEFRAME_M5, "h1": mt5.TIMEFRAME_H1}


def fetch_chunked(sym: str, tf, start: datetime, end: datetime,
                  chunk_days: int = 30) -> pd.DataFrame:
    frames, cur, empty_streak = [], end, 0
    while cur > start:
        prev = max(cur - timedelta(days=chunk_days), start)
        r = mt5.copy_rates_range(sym, tf, prev, cur)
        if r is not None and len(r):
            frames.append(pd.DataFrame(r)); empty_streak = 0
        else:
            empty_streak += 1
            if empty_streak >= 4:      # four consecutive empty months = archive end
                break
        cur = prev
        time.sleep(0.05)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset="time")
    df["dt"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.sort_values("dt").reset_index(drop=True)


def main():
    if not mt5.initialize(path=TERMINAL, timeout=60_000):
        print("init failed:", mt5.last_error()); return
    mt5.symbol_select(SYM, True)
    si = mt5.symbol_info(SYM)
    now = datetime.now(timezone.utc)
    print(f"{SYM}  digits={si.digits}  point={si.point}  contract={si.trade_contract_size}\n")

    summary = []
    for name, tf in TFS.items():
        back = 8 * 365 if name == "h1" else 4 * 365
        print(f"fetching {name.upper()} back to {back//365}y ...", flush=True)
        t0 = time.time()
        df = fetch_chunked(SYM, tf, now - timedelta(days=back), now,
                           chunk_days=15 if name == "m1" else 60)
        if df.empty:
            print(f"  {name}: NO DATA\n"); continue
        path = os.path.join(OUT, f"xauusd_{name}.csv")
        keep = df[["dt", "open", "high", "low", "close", "tick_volume", "spread"]]
        keep.to_csv(path, index=False)
        span = (df["dt"].iloc[-1] - df["dt"].iloc[0]).days / 365.25
        print(f"  {name.upper()}: {len(df):,} bars  {df['dt'].iloc[0]:%Y-%m-%d} -> "
              f"{df['dt'].iloc[-1]:%Y-%m-%d}  ({span:.2f}y)  [{time.time()-t0:.0f}s]")
        print(f"    saved {path}")
        # spread carried on each bar, in points
        sp = df["spread"].to_numpy() * si.point
        print(f"    bar spread: median ${np.median(sp):.3f}  p90 ${np.percentile(sp,90):.3f}\n")
        summary.append((name, len(df), span, float(np.median(sp))))

    with open(os.path.join(OUT, "spread_stats.txt"), "w") as fh:
        fh.write(f"symbol={SYM} digits={si.digits} point={si.point}\n")
        fh.write(f"contract={si.trade_contract_size} tick_value={si.trade_tick_value}\n")
        for name, n, span, sp in summary:
            fh.write(f"{name}: {n} bars, {span:.2f}y, median spread ${sp:.3f}\n")

    print("=" * 70)
    for name, n, span, sp in summary:
        print(f"  {name.upper():<4} {n:>9,} bars  {span:>5.2f}y  median spread ${sp:.3f}")
    print("=" * 70)
    mt5.shutdown()


if __name__ == "__main__":
    main()
