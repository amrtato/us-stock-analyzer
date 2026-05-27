"""
Fundamental analysis scorer.

Evaluates valuation, profitability, growth, balance sheet health,
and analyst consensus. Returns a composite score [0-100].
"""
import logging
import numpy as np

log = logging.getLogger(__name__)

# Sector-adjusted P/E benchmarks (rough fair-value P/E by sector)
SECTOR_PE_BENCHMARKS = {
    "Technology":        28,
    "Consumer":          22,
    "Financials":        14,
    "Healthcare":        20,
    "Industrials":       18,
    "Energy":            12,
    "Communication":     20,
    "Materials":         16,
    "Utilities":         18,
    "Real Estate":       30,
    "Unknown":           20,
}


def _safe(val, default=0.0):
    if val is None or val != val:  # None or NaN
        return default
    try:
        f = float(val)
        return default if (np.isinf(f) or np.isnan(f)) else f
    except (TypeError, ValueError):
        return default


# ── Individual component scores (each 0-100) ──────────────────────────────────

def score_valuation(fund: dict) -> float:
    """Lower P/E vs sector + low PEG = better value."""
    sector   = fund.get("sector", "Unknown")
    bench_pe = SECTOR_PE_BENCHMARKS.get(sector, 20)
    pe       = _safe(fund.get("pe_ratio"))
    fpe      = _safe(fund.get("forward_pe"))
    peg      = _safe(fund.get("peg_ratio"))
    ptb      = _safe(fund.get("price_to_book"))

    if pe <= 0 and fpe <= 0:
        return 50.0  # no P/E data

    use_pe = fpe if fpe > 0 else pe

    # P/E vs benchmark
    pe_ratio = use_pe / bench_pe
    if pe_ratio < 0.7:    pe_score = 90
    elif pe_ratio < 0.9:  pe_score = 75
    elif pe_ratio < 1.1:  pe_score = 60
    elif pe_ratio < 1.3:  pe_score = 45
    elif pe_ratio < 1.6:  pe_score = 30
    else:                 pe_score = 15

    # PEG
    if 0 < peg < 1.0:     peg_score = 85
    elif 1.0 <= peg < 1.5: peg_score = 65
    elif 1.5 <= peg < 2.0: peg_score = 45
    elif peg >= 2.0:       peg_score = 25
    else:                  peg_score = 50

    # Price-to-book
    if ptb <= 0:           ptb_score = 50
    elif ptb < 1.5:        ptb_score = 80
    elif ptb < 3.0:        ptb_score = 60
    elif ptb < 5.0:        ptb_score = 45
    else:                  ptb_score = 30

    return pe_score * 0.5 + peg_score * 0.35 + ptb_score * 0.15


def score_profitability(fund: dict) -> float:
    """Margin and ROE/ROA quality."""
    profit_margin  = _safe(fund.get("profit_margin"))
    op_margin      = _safe(fund.get("operating_margin"))
    roe            = _safe(fund.get("roe"))
    roa            = _safe(fund.get("roa"))

    def margin_score(m):
        if m > 0.25:  return 90
        if m > 0.15:  return 75
        if m > 0.08:  return 60
        if m > 0.03:  return 45
        if m > 0:     return 30
        return 10

    def roe_score(r):
        if r > 0.30:  return 90
        if r > 0.20:  return 75
        if r > 0.12:  return 60
        if r > 0.05:  return 45
        if r > 0:     return 30
        return 10

    pm_s  = margin_score(profit_margin)
    om_s  = margin_score(op_margin)
    roe_s = roe_score(roe)
    roa_s = roe_score(roa * 2.5)  # ROA is typically lower; scale up

    return pm_s * 0.30 + om_s * 0.25 + roe_s * 0.30 + roa_s * 0.15


def score_growth(fund: dict) -> float:
    """Revenue and earnings growth momentum."""
    rev_growth = _safe(fund.get("revenue_growth"))
    earn_growth = _safe(fund.get("earnings_growth"))
    eps_t = _safe(fund.get("eps_trailing"))
    eps_f = _safe(fund.get("eps_forward"))

    def growth_score(g):
        if g > 0.30:  return 95
        if g > 0.20:  return 85
        if g > 0.10:  return 70
        if g > 0.05:  return 58
        if g > 0:     return 48
        if g > -0.05: return 35
        return 20

    rev_s  = growth_score(rev_growth)
    earn_s = growth_score(earn_growth)

    # Forward EPS improvement
    if eps_t > 0 and eps_f > eps_t:
        eps_imp = (eps_f - eps_t) / eps_t
        eps_s = growth_score(eps_imp)
    else:
        eps_s = 50

    return rev_s * 0.35 + earn_s * 0.40 + eps_s * 0.25


def score_balance_sheet(fund: dict) -> float:
    """Debt burden and liquidity."""
    dte   = _safe(fund.get("debt_to_equity"))   # lower is better
    curr  = _safe(fund.get("current_ratio"), 1)
    quick = _safe(fund.get("quick_ratio"), 1)

    # D/E score
    if dte <= 0:     dte_score = 60   # no data
    elif dte < 0.5:  dte_score = 90
    elif dte < 1.0:  dte_score = 75
    elif dte < 2.0:  dte_score = 55
    elif dte < 3.0:  dte_score = 35
    else:            dte_score = 15

    # Current ratio
    if curr > 2.5:   cr_score = 85
    elif curr > 1.5: cr_score = 70
    elif curr > 1.0: cr_score = 55
    else:            cr_score = 30

    # Quick ratio
    if quick > 1.5:  qr_score = 85
    elif quick > 1.0: qr_score = 70
    elif quick > 0.7: qr_score = 50
    else:            qr_score = 30

    return dte_score * 0.50 + cr_score * 0.30 + qr_score * 0.20


def score_analyst_consensus(fund: dict) -> float:
    """Analyst rating 1=strong buy → 5=sell. Normalize to 0-100."""
    rec   = _safe(fund.get("recommendation"))  # 1-5 scale
    n     = _safe(fund.get("analyst_count"), 0)
    price = _safe(fund.get("price_today"))
    tp    = _safe(fund.get("target_price"))

    if rec <= 0:
        rec_score = 50.0
    else:
        # Invert: 1→90, 2→70, 3→50, 4→30, 5→10
        rec_score = max(10, min(90, 100 - rec * 20))

    # Upside to target price
    if price > 0 and tp > 0:
        upside = (tp - price) / price
        if upside > 0.20:   tp_score = 90
        elif upside > 0.10: tp_score = 75
        elif upside > 0:    tp_score = 60
        elif upside > -0.10: tp_score = 40
        else:               tp_score = 20
    else:
        tp_score = 50

    # Conviction: more analysts = more reliable
    if n >= 20:   cov_score = 80
    elif n >= 10: cov_score = 65
    elif n >= 5:  cov_score = 55
    else:         cov_score = 40

    return rec_score * 0.50 + tp_score * 0.35 + cov_score * 0.15


# ── Main entry point ───────────────────────────────────────────────────────────

def analyse_fundamental(ticker: str, fund: dict, current_price: float = 0) -> dict:
    """
    Compute composite fundamental score and return structured result.
    """
    if not fund:
        return {"score": 50.0, "signals": ["No fundamental data"], "subscores": {}}

    # Inject current price for target-price upside calc
    fund["price_today"] = current_price

    val_s   = score_valuation(fund)
    prof_s  = score_profitability(fund)
    grow_s  = score_growth(fund)
    bs_s    = score_balance_sheet(fund)
    ana_s   = score_analyst_consensus(fund)

    weights = {
        "valuation": 0.25,
        "profitability": 0.22,
        "growth": 0.28,
        "balance_sheet": 0.12,
        "analyst": 0.13,
    }

    composite = (
        val_s  * weights["valuation"]  +
        prof_s * weights["profitability"] +
        grow_s * weights["growth"] +
        bs_s   * weights["balance_sheet"] +
        ana_s  * weights["analyst"]
    )

    # ── Human-readable signals ────────────────────────────────────────────────
    signals = []
    pe = _safe(fund.get("pe_ratio"))
    if pe > 0:
        bench = SECTOR_PE_BENCHMARKS.get(fund.get("sector", "Unknown"), 20)
        if pe < bench * 0.8:   signals.append(f"Undervalued P/E ({pe:.1f} vs {bench} sector avg)")
        elif pe > bench * 1.5: signals.append(f"Premium P/E ({pe:.1f} vs {bench} sector avg)")

    peg = _safe(fund.get("peg_ratio"))
    if 0 < peg < 1.0:   signals.append(f"Attractive PEG ratio ({peg:.2f})")

    roe = _safe(fund.get("roe"))
    if roe > 0.20:   signals.append(f"High ROE ({roe*100:.1f}%)")

    rg = _safe(fund.get("revenue_growth"))
    if rg > 0.15:    signals.append(f"Strong revenue growth ({rg*100:.1f}% YoY)")
    elif rg < -0.05: signals.append(f"Revenue declining ({rg*100:.1f}% YoY)")

    eg = _safe(fund.get("earnings_growth"))
    if eg > 0.20:    signals.append(f"Strong earnings growth ({eg*100:.1f}%)")

    rec = _safe(fund.get("recommendation"))
    if rec > 0:
        labels = {1: "Strong Buy", 2: "Buy", 3: "Hold", 4: "Underperform", 5: "Sell"}
        label  = labels.get(round(rec), f"{rec:.1f}")
        n      = int(_safe(fund.get("analyst_count")))
        signals.append(f"Analyst consensus: {label} ({n} analysts)")

    return {
        "score":    round(composite, 2),
        "signals":  signals,
        "subscores": {
            "valuation":    round(val_s, 1),
            "profitability": round(prof_s, 1),
            "growth":       round(grow_s, 1),
            "balance_sheet": round(bs_s, 1),
            "analyst":      round(ana_s, 1),
        },
        "metrics": {
            "pe":             round(pe, 2) if pe else None,
            "peg":            round(peg, 2) if peg else None,
            "roe":            f"{roe*100:.1f}%" if roe else None,
            "revenue_growth": f"{rg*100:.1f}%" if rg else None,
            "earnings_growth": f"{eg*100:.1f}%" if eg else None,
            "debt_equity":    round(_safe(fund.get("debt_to_equity")), 2),
            "market_cap_b":   round(_safe(fund.get("market_cap")) / 1e9, 1),
        },
    }
