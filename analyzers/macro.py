"""
Macro & political context scorer.

Considers:
  - Sector rotation signals (which sectors are leading vs lagging)
  - Beta-adjusted risk (high-beta = higher reward/risk in bull markets)
  - 52-week position (near highs vs lows)
  - Market cap quality (mega-cap stability vs small-cap growth)
  - Dividend safety (income signal)

Political/macro context is passed in as a free-text snapshot fetched
once per session by the AI report generator and stored in a shared dict.
"""
import logging

log = logging.getLogger(__name__)

# Sector performance scores updated daily (filled by run_macro_update)
# Keys: sector name → momentum score [0-100]
SECTOR_MOMENTUM: dict[str, float] = {}

# Macroeconomic regime flags (set externally from AI analysis)
MACRO_FLAGS = {
    "fed_hawkish":       False,  # rate hike mode → punishes growth stocks
    "recession_risk":    False,  # risk-off → defensives outperform
    "strong_dollar":     False,  # hurts multinationals
    "earnings_season":   False,  # higher volatility, opportunity
    "geopolitical_risk": False,  # energy + defense benefit
}


def update_sector_momentum(sector_returns: dict[str, float]):
    """Call with {sector: pct_return} to update sector momentum cache."""
    for sector, ret in sector_returns.items():
        # Normalize return to [0,100] score
        if ret > 3:       SECTOR_MOMENTUM[sector] = 85
        elif ret > 1.5:   SECTOR_MOMENTUM[sector] = 70
        elif ret > 0:     SECTOR_MOMENTUM[sector] = 58
        elif ret > -1.5:  SECTOR_MOMENTUM[sector] = 42
        elif ret > -3:    SECTOR_MOMENTUM[sector] = 30
        else:             SECTOR_MOMENTUM[sector] = 15


def _score_52w_position(price: float, high52: float, low52: float) -> float:
    """
    Score based on where price sits in its 52-week range.
    Near low = potential value/reversal; mid-range = neutral; near high = momentum.
    """
    if high52 <= low52 or high52 <= 0:
        return 50.0
    pos = (price - low52) / (high52 - low52)  # 0=at low, 1=at high

    # U-shaped preference: both extreme oversold and strong momentum score well
    if pos < 0.15:   return 78  # deep in range — reversal candidate
    if pos < 0.30:   return 65
    if pos < 0.50:   return 55
    if pos < 0.70:   return 60
    if pos < 0.85:   return 68  # approaching highs — momentum
    return 75                    # near 52w high — breakout territory


def _score_beta(beta: float) -> float:
    """
    In bull market: higher beta (up to ~1.5) is rewarded.
    In bear/uncertain: lower beta preferred.
    Neutral: 1.0 scores 60, very high or negative scores lower.
    """
    if beta is None or beta != beta:  # None or NaN
        return 50.0
    if beta < 0:      return 30
    if beta < 0.5:    return 45  # too defensive
    if beta < 0.8:    return 55
    if beta < 1.2:    return 65
    if beta < 1.5:    return 72
    if beta < 2.0:    return 60  # volatile but tradeable
    return 40                    # very high beta = speculative


def _score_market_cap(market_cap_usd: float) -> float:
    """Mega-caps (>$200B) get quality premium; micro-caps penalized."""
    if not market_cap_usd:
        return 50.0
    b = market_cap_usd / 1e9  # billions
    if b > 500:  return 75   # mega-cap (AAPL, MSFT, NVDA…)
    if b > 100:  return 68   # large-cap
    if b > 10:   return 60   # mid-cap
    if b > 2:    return 50   # small-cap
    return 38                 # micro-cap


def _score_sector_rotation(sector: str) -> float:
    """Use cached sector momentum; default neutral if unknown."""
    return SECTOR_MOMENTUM.get(sector, 55.0)


def _apply_macro_flags(base_score: float, sector: str, fund: dict) -> tuple[float, list[str]]:
    """Adjust base macro score based on active macro regime flags."""
    adj    = 0.0
    flags  = []
    pe     = fund.get("pe_ratio", 20) or 20
    beta   = fund.get("beta", 1.0) or 1.0
    div    = fund.get("dividend_yield", 0) or 0

    if MACRO_FLAGS["fed_hawkish"]:
        if pe > 30:
            adj -= 8
            flags.append("Fed hawkish: high-growth/high-PE stocks penalized")
        if div > 0.03:
            adj += 5
            flags.append("Fed hawkish: dividend stocks resilient")

    if MACRO_FLAGS["recession_risk"]:
        defensive = sector in ("Healthcare", "Utilities", "Consumer")
        if defensive:
            adj += 10
            flags.append("Recession risk: defensive sector premium")
        elif sector == "Technology" and beta > 1.3:
            adj -= 8
            flags.append("Recession risk: high-beta tech penalized")

    if MACRO_FLAGS["strong_dollar"]:
        # Domestic/small companies benefit; multinationals hurt
        mkcap = fund.get("market_cap", 1e12) or 1e12
        if mkcap > 200e9:
            adj -= 5
            flags.append("Strong USD: multinational revenue headwind")

    if MACRO_FLAGS["earnings_season"]:
        adj += 3
        flags.append("Earnings season: elevated opportunity")

    if MACRO_FLAGS["geopolitical_risk"]:
        if sector in ("Energy", "Industrials"):
            adj += 8
            flags.append("Geopolitical risk: energy/defense tailwind")

    return base_score + adj, flags


def analyse_macro(ticker: str, fund: dict, quote: dict) -> dict:
    """
    Compute macro/political score and return structured result.
    fund: fundamentals dict from fetcher
    quote: {price, change_pct, volume, ...}
    """
    price    = quote.get("price", 0) if quote else 0
    high52   = fund.get("52w_high", price) or price
    low52    = fund.get("52w_low",  price) or price
    beta     = fund.get("beta")
    mktcap   = fund.get("market_cap")
    sector   = fund.get("sector", "Unknown")

    s_52w    = _score_52w_position(price, high52, low52)
    s_beta   = _score_beta(beta)
    s_mktcap = _score_market_cap(mktcap)
    s_sector = _score_sector_rotation(sector)

    base = s_52w * 0.25 + s_beta * 0.20 + s_mktcap * 0.20 + s_sector * 0.35

    adjusted, macro_flags = _apply_macro_flags(base, sector, fund)
    composite = max(0, min(100, adjusted))

    signals = []
    pct_from_high = (price - high52) / high52 * 100 if high52 else 0
    pct_from_low  = (price - low52)  / low52  * 100 if low52  else 0
    if pct_from_high > -5:  signals.append(f"Near 52-week high ({pct_from_high:.1f}% below)")
    if pct_from_low < 20:   signals.append(f"Near 52-week low ({pct_from_low:.1f}% above)")
    if beta and beta > 1.4: signals.append(f"High-beta play (β={beta:.2f})")
    if beta and beta < 0.6: signals.append(f"Low-beta defensive (β={beta:.2f})")
    signals.extend(macro_flags)

    return {
        "score": round(composite, 2),
        "signals": signals,
        "subscores": {
            "52w_position":    round(s_52w, 1),
            "beta":            round(s_beta, 1),
            "market_cap":      round(s_mktcap, 1),
            "sector_momentum": round(s_sector, 1),
        },
        "sector": sector,
    }
