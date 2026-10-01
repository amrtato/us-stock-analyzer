"""
How deep does M1 and tick history actually go on this broker?

copy_rates_from_pos(..., 5000) only asks for 5,000 bars, so "deep range empty"
in mt5_sync.py measured my request size, not the broker's archive. Here we ask
for a very large count to force a full download, then probe tick availability
backwards month by month until it runs dry.

Read-only: rates and ticks only.
"""
from __future__ import annotations
import time
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5

TERMINAL = r"C:\Program Files\MetaTrader 5 (AMR)\terminal64.exe"
SYM = "XAUUSDm"


def main():
    if not mt5.initialize(path=TERMINAL, timeout=60_000):
        print("init failed:", mt5.last_error()); return
    mt5.symbol_select(SYM, True)
    si = mt5.symbol_info(SYM)
    now = datetime.now(timezone.utc)

    print(f"{SYM}  digits={si.digits} point={si.point} "
          f"spread={si.spread}pts = ${si.spread*si.point:.3f}")
    print(f"contract={si.trade_contract_size} tick_value={si.trade_tick_value}\n")

    # ── deep M1: request a very large count ─────────────────────────────────
    print("=" * 76)
    print("M1 DEPTH — requesting large bar counts (each triggers a deeper pull)")
    print("=" * 76)
    for count in (100_000, 500_000, 1_500_000, 3_000_000):
        t0 = time.time()
        r = mt5.copy_rates_from_pos(SYM, mt5.TIMEFRAME_M1, 0, count)
        if r is None or len(r) == 0:
            print(f"  request {count:>9,}: none ({mt5.last_error()})"); continue
        d0 = datetime.fromtimestamp(int(r[0]["time"]), tz=timezone.utc)
        d1 = datetime.fromtimestamp(int(r[-1]["time"]), tz=timezone.utc)
        yrs = (d1 - d0).days / 365.25
        print(f"  request {count:>9,}: got {len(r):>9,} bars  "
              f"{d0:%Y-%m-%d} -> {d1:%Y-%m-%d}  ({yrs:.2f}y)  [{time.time()-t0:.1f}s]")
        if len(r) < count:
            print(f"    -> broker archive exhausted at {len(r):,} M1 bars")
            break

    # ── tick depth: walk backwards a month at a time ────────────────────────
    print("\n" + "=" * 76)
    print("TICK DEPTH — 24h sample at increasing age")
    print("=" * 76)
    for months in (0, 1, 3, 6, 12, 24, 36):
        end = now - timedelta(days=30 * months)
        start = end - timedelta(days=1)
        t0 = time.time()
        tk = mt5.copy_ticks_range(SYM, start, end, mt5.COPY_TICKS_ALL)
        n = 0 if tk is None else len(tk)
        status = f"{n:>9,} ticks" if n else "none"
        print(f"  {months:>2} months back ({start:%Y-%m-%d}): {status}  [{time.time()-t0:.1f}s]")
        if n == 0 and months > 0:
            print("    -> tick archive ends before this point")
            break

    # ── real spread distribution from ticks ─────────────────────────────────
    print("\n" + "=" * 76)
    print("REAL SPREAD (from ticks, last 3 days) — the economics of scalping")
    print("=" * 76)
    tk = mt5.copy_ticks_range(SYM, now - timedelta(days=3), now, mt5.COPY_TICKS_ALL)
    if tk is not None and len(tk):
        import numpy as np
        sp = tk["ask"] - tk["bid"]
        sp = sp[(sp > 0) & np.isfinite(sp)]
        px = float(np.median(tk["bid"][tk["bid"] > 0]))
        print(f"  n={len(sp):,}  median ${np.median(sp):.3f}  mean ${sp.mean():.3f}  "
              f"p90 ${np.percentile(sp,90):.3f}  p99 ${np.percentile(sp,99):.3f}")
        print(f"  gold ~${px:,.2f}")
        for target in (1.0, 2.0, 5.0, 10.0):
            pct = np.median(sp) / target * 100
            print(f"    a ${target:>5.2f} scalp target gives up {pct:>5.1f}% to spread")
    mt5.shutdown()


if __name__ == "__main__":
    main()
