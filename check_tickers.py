"""Liveness check for every configured ticker: does it still trade, and is data fresh?"""
import sys, warnings, logging
from datetime import datetime, timezone, timedelta
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
sys.path.insert(0, r"C:\Users\AmrAbdelsalam\OneDrive - Trioshield Technologies Inc\Desktop\Claude Projects\XAUUSD_AurumPrime\us-stock-analyzer")

import yfinance as yf
from config import ALL_STOCKS, SECTOR_STOCKS, SECTOR_MAP

universe = {}
for t in ALL_STOCKS:
    universe.setdefault(t, set()).add("ALL_STOCKS")
for sec, lst in SECTOR_STOCKS.items():
    for t in lst:
        universe.setdefault(t, set()).add(f"SECTOR_STOCKS/{sec}")
for sec, lst in SECTOR_MAP.items():
    for t in lst:
        universe.setdefault(t, set()).add(f"SECTOR_MAP/{sec}")

print(f"checking {len(universe)} unique tickers...\n")
dead, stale, ok = [], [], 0
now = datetime.now(timezone.utc)

for t in sorted(universe):
    try:
        df = yf.Ticker(t).history(period="1mo", interval="1d")
        if df is None or df.empty:
            dead.append((t, "no price history", sorted(universe[t]))); continue
        last = df.index[-1].to_pydatetime()
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        age = (now - last).days
        if age > 10:
            stale.append((t, f"last bar {age}d old ({last.date()})", sorted(universe[t])))
        else:
            ok += 1
    except Exception as e:
        dead.append((t, f"error: {type(e).__name__}", sorted(universe[t])))

print(f"OK: {ok}   STALE: {len(stale)}   DEAD: {len(dead)}\n")
for label, rows in (("DEAD / NO DATA", dead), ("STALE", stale)):
    if rows:
        print(f"--- {label} ---")
        for t, why, where in rows:
            print(f"  {t:<6} {why:<32} in: {', '.join(where)}")
        print()
