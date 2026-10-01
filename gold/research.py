"""
Frozen research constants for the Gold tab.

WHY THESE ARE CONSTANTS, NOT LIVE QUERIES
    The volatility profile and spread statistics were measured from MT5 exports
    (47,161 H1 bars / 1,411,140 M1 bars). MetaTrader5's Python API needs a local
    Windows terminal, so it cannot run on Azure Linux. These are also not the kind
    of numbers that should be recomputed per page load — an 8-year seasonal profile
    is a research finding with a provenance, not a live quote. Re-derive them by
    re-running gold/sessions_h1_mt5.py after a fresh export, then update here.

    Live price and ATR come from yfinance at request time (see gold/live.py).
    Spread cannot be observed on Azure, so the measured broker median is used and
    labelled as such wherever it drives a number the user might act on.

EVERY FIGURE BELOW IS MEASURED. Nothing is estimated or illustrative.
"""
from __future__ import annotations

MEASURED_AT = "2026-08-07"
# Provenance of the RESEARCH constants below. The LIVE feed is a separate
# thing (Massive XAU/USD — see gold/massive.py); this string must not be
# read as naming the current price source, so it says "research" outright.
SOURCE = "research: MT5 · Exness XAUUSDm"
H1_BARS, H1_YEARS = 47_161, 8.0
M1_BARS, M1_YEARS = 1_411_140, 4.0

# Broker-measured spread, XAUUSDm. digits=3, point=0.001, contract=100 oz.
SPREAD_MEDIAN = 0.20
SPREAD_P90 = 0.24
SPREAD_P99 = 0.364
SPREAD_MAX = 4.00
CONTRACT_OZ = 100.0

# Hourly volatility profile, New York time, 8 years of H1.
# Value = that hour's return sigma RELATIVE to gold's average hour.
#
# Stored as a multiple rather than a dollar ATR on purpose. Gold's median close
# over the measurement window was $1,883 and it now trades near $4,400, so any
# dollar figure from that period is stale — a "$4.47 median ATR" next to a live
# $20 ATR reads as a 4x anomaly when it is only a price-level change. A relative
# multiple is regime-independent, and dollars are derived from LIVE price below.
#
# ATR(24) was rejected as the per-hour measure: it averages 24 bars, so it barely
# moves across the day (0.232%-0.263%) and carries almost no hourly information.
# Per-hour return sigma ranges 0.53x-1.71x — that is where the structure lives.
# 17:00 ET absent — CME daily settlement halt.
BASE_HOURLY_SIGMA_PCT = 0.221      # gold's average hour, % of price, 8y H1

VOL_PROFILE = {
    0: 0.58, 1: 0.74, 2: 0.93, 3: 0.98, 4: 0.83, 5: 0.79, 6: 0.91, 7: 0.88,
    8: 1.70, 9: 1.62, 10: 1.71, 11: 1.19, 12: 0.94, 13: 0.86, 14: 1.03,
    15: 0.82, 16: 0.55, 18: 1.05, 19: 0.64, 20: 0.79, 21: 1.01, 22: 0.64,
    23: 0.53,
}


def expected_move(price: float, hour: int) -> float:
    """
    Typical 1-sigma move for this hour, in dollars, at the CURRENT price.

    Derived from live price rather than a stored dollar figure, so it stays
    correct as gold reprices.
    """
    mult = VOL_PROFILE.get(hour, 1.0)
    return price * (BASE_HOURLY_SIGMA_PCT / 100.0) * mult

SESSIONS = [
    ("🌏 Asian session", [18, 19, 20, 21, 22, 23, 0, 1, 2]),
    ("🇬🇧 London", [3, 4, 5, 6, 7]),
    ("🔥 London / New York overlap", [8, 9, 10, 11]),
    ("🇺🇸 New York afternoon", [12, 13, 14, 15, 16]),
]

# ── The one monitored hypothesis ─────────────────────────────────────────────
# Long at 16:00 ET close, exit 17:00 ET close. No stop, no target.
HYPOTHESIS = {
    "name": "16:00 ET settlement-hour drift",
    # CORRECTED 2026-08-07. This previously read "exit at the 17:00 ET close",
    # which was wrong: there is no 17:00 bar (CME settlement halt), so the exit
    # lands on the 18:00 bar in 1,975 of 1,993 cases. The position is therefore
    # held ACROSS a market halt — and on Fridays, across the whole weekend
    # (median hold 50h vs 2h Mon-Thu). That is why the worst signal lost $15,013:
    # a -$97.98 weekend gap that no stop could have caught.
    "rule": ("Long at the 16:00 ET close, exit at the 18:00 ET close "
             "(no 17:00 bar — settlement halt). Monday–Thursday only."),
    "status": "PAPER TEST — NOT TRADEABLE",
    "paper_start": "2026-08-07",          # forward record is computed from here
    # Friday entries were dropped from the forward test on the evidence below.
    # The recorded backtest figures further down INCLUDE Fridays, because that
    # is the rule as originally registered — they are not restated to flatter it.
    "friday_finding": (
        "Friday entries are 20% of signals, contribute no measurable edge "
        "(p=0.59), and produced 6 of the 10 worst losses with 2.2× the standard "
        "deviation ($1,639 vs $735). Excluding them cuts worst-case loss from "
        "-$15,017 to -$7,328 and lifts the remaining signal from t=2.70 to t=3.43."
    ),
    "decomposition": (
        "The edge splits into a GAP component (16:00 close → 18:00 open, market "
        "shut, mean +$0.449/oz Mon-Thu, t=+6.10) and a BAR component (18:00 open "
        "→ close, market open, +$0.381/oz, t=+2.56). The larger half is captured "
        "while the market is closed, so no stop can protect it."
    ),
    "full": dict(n=1993, win=0.501, pf=1.39, mean_oz=0.566, total_lot=112_813,
                 maxdd_lot=-18_597, streak=29, worst_lot=-15_012.90, t=2.57, p=0.0101),
    "train": dict(n=1496, win=0.476, pf=1.11, mean_oz=0.086, total_lot=12_920,
                  maxdd_lot=-18_597, streak=29, worst_lot=-6_176.60, t=0.94, p=0.3453),
    "holdout": dict(n=497, win=0.575, pf=1.59, mean_oz=2.010, total_lot=99_894,
                    maxdd_lot=-15_012, streak=8, worst_lot=-15_012.90, t=2.41, p=0.0164),
    "years": {2018: 865, 2019: 937, 2020: 10_568, 2021: -10_258, 2022: -4_300,
              2023: 11_702, 2024: 6_540, 2025: 33_838, 2026: 62_922},
    "caveats": [
        "Win rate over the full 8 years is 50.1% — a coin flip.",
        "29 consecutive losing signals at worst. Almost nobody keeps executing through that.",
        "In training the drawdown reached 144% of everything the rule earned.",
        "2025-26 alone are 86% of the 8-year total; 2021 and 2022 both lost money.",
        r"No stop loss. Worst single signal was -\$15,013 per 1.0 lot.",
        "Train result was not significant net of spread (p=0.345).",
    ],
}

# ── What was tested and did not survive ──────────────────────────────────────
LEDGER = [
    ("ICT setups on daily bars — FVG, order blocks, liquidity sweeps, BOS",
     "Random entry", False),
    ("40 features × 120 directional tests, 5–20 day horizons",
     "2 hit, ~6 expected", False),
    ("Asian-range breakout into London and New York", "No edge", False),
    ("Killzone directional bias — London, NY, London close", "No edge", False),
    ("M1 short-horizon predictability, 4 years",
     "Real but 0.44× spread", False),
    ("Elliott Wave", "Not codifiable", False),
    ("Volatility seasonality by hour", "Replicated ✓", True),
    ("16:00 ET settlement drift", "Survived holdout ✓", True),
]

# Largest short-horizon predictability found anywhere in the M1 work, in USD.
# Reported next to the spread because that comparison is the whole finding.
BEST_M1_EDGE_USD = 0.088

# ── Spot price sourcing ──────────────────────────────────────────────────────
# There is no free spot XAU/USD feed. yfinance returns nothing for XAUUSD=X,
# XAU=X or XAUUSD; GOLD and ^XAU are mining equities, not metal. GC=F is
# FUTURES and sat +$62.36 (+1.44%) above this broker's spot when measured —
# levels on that scale simply do not fill.
#
# PAXG (Paxos Gold) is the workable substitute: each token is one fine troy
# ounce held in London vaults and redeemable, so it tracks spot closely. It
# trades on Binance with a keyless public API.
#
# Calibration, 686 matched hourly bars vs MT5 XAUUSDm (2026-06-27 to 08-07):
#     PAXG - spot : mean -$3.66, sd $2.19, corr of hourly changes 0.9951
#     after subtracting the median offset below:
#         mean |error| $1.74, p95 $4.33, max $6.82
#     (GC=F futures for comparison: mean |error| $19.50, max $56.14)
#
# RETIRED 2026-10-01 — NOTHING READS THESE ANY MORE.
# Kept as a record of a measurement, not as live configuration. The PAXG tier was
# removed from gold/live.py for two independent reasons:
#   1. Binance geo-blocks Ontario, and the app runs in Azure Canada Central, so
#      the primary PAXG source answers HTTP 451 in production.
#   2. The warning below came true. Re-measured on 2026-10-01 against Massive
#      XAU/USD, PAXG-after-offset sat $13.91 above spot — 3.2x its own p95, with
#      the offset making the error WORSE rather than better. The premium moved,
#      exactly as predicted here, and a frozen constant could not track it.
# Live spot now comes from Massive (gold/massive.py), which quotes the metal
# directly, so there is no proxy and no offset to go stale.
PAXG_OFFSET = -3.68            # PAXG minus spot, median. spot ≈ PAXG - OFFSET
PAXG_CALIBRATED = "2026-08-07"
PAXG_ERR_MEAN = 1.74
PAXG_ERR_P95 = 4.33
PAXG_ERR_MAX = 6.82

# Known caveat: PAXG trades 24/7 while the gold market closes at weekends. When
# gold is shut, PAXG can drift on crypto flow alone and the calibration above
# does not apply — the UI flags this rather than silently showing a stale proxy.


def vol_state(x: float) -> tuple[str, str, str]:
    """(css colour token, label, dot) for a volatility multiple."""
    if x < 0.85:
        return "#00ff88", "Calm", "🟢"
    if x <= 1.30:
        return "#ffd700", "Active", "🟡"
    return "#ff4444", "Extreme", "🔴"


def min_viable_target(spread: float = SPREAD_MEDIAN, max_share: float = 0.05) -> float:
    """Smallest target that keeps spread under `max_share` of the move."""
    return spread / max_share
