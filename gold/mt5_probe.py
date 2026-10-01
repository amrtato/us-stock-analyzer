"""
Probe the MT5 terminals: what gold symbols exist, and how much M1/tick history
is actually retrievable.

READ-ONLY BY CONSTRUCTION. This module calls only:
    initialize / login-less attach, account_info, symbols_get,
    symbol_info, copy_rates_range, copy_ticks_range
It never imports or calls order_send, order_check, positions_*, or anything else
that can place, modify or close a trade.

Account identifiers are masked in output — the broker and demo/real flag are
enough to know which terminal answered.
"""
from __future__ import annotations
import sys
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5

TERMINALS = {
    "EXNESS": r"C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe",
    "AMR":    r"C:\Program Files\MetaTrader 5 (AMR)\terminal64.exe",
}
GOLD_HINTS = ("XAU", "GOLD")


def mask(n) -> str:
    s = str(n)
    return f"{'*' * max(len(s) - 3, 0)}{s[-3:]}" if s and s != "None" else "n/a"


def probe(name: str, path: str) -> bool:
    print("=" * 78)
    print(f"TERMINAL: {name}")
    print("=" * 78)
    if not mt5.initialize(path=path, timeout=60_000):
        print(f"  initialize failed: {mt5.last_error()}")
        return False

    ti = mt5.terminal_info()
    ai = mt5.account_info()
    if ti:
        print(f"  build {ti.build} | connected={ti.connected} | trade_allowed={ti.trade_allowed}")
        print(f"  data path: {ti.data_path}")
    if ai:
        kind = {0: "DEMO", 1: "CONTEST", 2: "REAL"}.get(ai.trade_mode, "?")
        print(f"  broker: {ai.company} | server: {ai.server} | {kind} | "
              f"login {mask(ai.login)} | {ai.currency}")
    else:
        print("  account_info unavailable (terminal may not be logged in)")

    syms = mt5.symbols_get() or []
    gold = [s.name for s in syms if any(h in s.name.upper() for h in GOLD_HINTS)]
    print(f"  {len(syms)} symbols total; gold-like: {gold[:12]}{' ...' if len(gold) > 12 else ''}")

    now = datetime.now(timezone.utc)
    for sym in gold[:6]:
        if not mt5.symbol_select(sym, True):
            print(f"    {sym:<12} could not select")
            continue
        si = mt5.symbol_info(sym)
        spread = f"{si.spread} pts" if si else "?"
        digits = si.digits if si else "?"
        # how far back does M1 go?
        m1 = mt5.copy_rates_range(sym, mt5.TIMEFRAME_M1,
                                  now - timedelta(days=365 * 6), now)
        if m1 is None or len(m1) == 0:
            print(f"    {sym:<12} spread {spread:<9} M1: none")
            continue
        first = datetime.fromtimestamp(int(m1[0]["time"]), tz=timezone.utc)
        last = datetime.fromtimestamp(int(m1[-1]["time"]), tz=timezone.utc)
        yrs = (last - first).days / 365.25
        # ticks for a recent 2-day window (cheap sample)
        tk = mt5.copy_ticks_range(sym, now - timedelta(days=2), now, mt5.COPY_TICKS_ALL)
        ntk = 0 if tk is None else len(tk)
        print(f"    {sym:<12} spread {spread:<9} digits {digits} | "
              f"M1 {len(m1):>9,} bars  {first:%Y-%m-%d} -> {last:%Y-%m-%d} ({yrs:.1f}y) | "
              f"ticks/2d {ntk:,}")
    mt5.shutdown()
    print()
    return True


if __name__ == "__main__":
    any_ok = False
    for nm, p in TERMINALS.items():
        try:
            any_ok |= probe(nm, p)
        except Exception as exc:
            print(f"  {nm}: {type(exc).__name__}: {exc}\n")
    if not any_ok:
        print("No terminal responded. Launch MT5 manually, log in, then re-run.")
        sys.exit(1)
