"""
Force MT5 to sync gold history, then report what is genuinely retrievable.

copy_rates_range() returns nothing for a symbol the terminal has never cached —
MT5 fetches history lazily, normally when a chart is opened. copy_rates_from_pos()
triggers that fetch, so we poll it until the server responds, then re-ask for the
full range.

Still read-only: rates and ticks only, no trading calls anywhere.
"""
from __future__ import annotations
import time
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5

TERMINALS = {
    "AMR (demo, 356 syms)": r"C:\Program Files\MetaTrader 5 (AMR)\terminal64.exe",
    "EXNESS (real)":        r"C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe",
}
CANDIDATES = ["XAUUSDm", "XAUUSDc", "XAUUSD247m", "XAUUSD"]

TFS = [("M1", mt5.TIMEFRAME_M1), ("M5", mt5.TIMEFRAME_M5),
       ("M15", mt5.TIMEFRAME_M15), ("H1", mt5.TIMEFRAME_H1)]


def sync(sym: str, tf, tries: int = 12):
    """Poll copy_rates_from_pos until the terminal has pulled history."""
    for i in range(tries):
        r = mt5.copy_rates_from_pos(sym, tf, 0, 5000)
        if r is not None and len(r) > 100:
            return r
        time.sleep(2.0)
    return r if r is not None else None


def main():
    for tname, path in TERMINALS.items():
        print("=" * 84)
        print(f"TERMINAL: {tname}")
        print("=" * 84)
        if not mt5.initialize(path=path, timeout=60_000):
            print(f"  init failed: {mt5.last_error()}\n"); continue

        syms = {s.name for s in (mt5.symbols_get() or [])}
        targets = [c for c in CANDIDATES if c in syms]
        if not targets:
            print("  no gold symbol matched\n"); mt5.shutdown(); continue

        now = datetime.now(timezone.utc)
        for sym in targets:
            mt5.symbol_select(sym, True)
            si = mt5.symbol_info(sym)
            if si:
                pt = si.point
                sp_price = si.spread * pt
                print(f"\n  {sym}: digits={si.digits} point={pt} "
                      f"spread={si.spread}pts (~${sp_price:.3f}) "
                      f"tick_value={getattr(si,'trade_tick_value',None)}")
            for tfname, tf in TFS:
                r = sync(sym, tf, tries=6 if tfname == "M1" else 3)
                if r is None or len(r) == 0:
                    print(f"    {tfname:<4} no data"); continue
                first = datetime.fromtimestamp(int(r[0]["time"]), tz=timezone.utc)
                last = datetime.fromtimestamp(int(r[-1]["time"]), tz=timezone.utc)
                # now that it is cached, ask for the deep range
                deep = mt5.copy_rates_range(sym, tf, now - timedelta(days=365*8), now)
                dn = 0 if deep is None else len(deep)
                if dn:
                    d0 = datetime.fromtimestamp(int(deep[0]["time"]), tz=timezone.utc)
                    d1 = datetime.fromtimestamp(int(deep[-1]["time"]), tz=timezone.utc)
                    yrs = (d1 - d0).days / 365.25
                    print(f"    {tfname:<4} recent {len(r):>6,} | DEEP {dn:>9,} bars "
                          f"{d0:%Y-%m-%d} -> {d1:%Y-%m-%d}  ({yrs:.1f}y)")
                else:
                    print(f"    {tfname:<4} recent {len(r):>6,} "
                          f"{first:%Y-%m-%d} -> {last:%Y-%m-%d} | deep range empty")
            # tick availability
            tk = mt5.copy_ticks_range(sym, now - timedelta(days=5), now, mt5.COPY_TICKS_ALL)
            print(f"    ticks last 5d: {0 if tk is None else len(tk):,}")
        mt5.shutdown()
        print()


if __name__ == "__main__":
    main()
