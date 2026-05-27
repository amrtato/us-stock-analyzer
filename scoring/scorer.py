"""
Composite scoring engine.

Merges technical, fundamental, sentiment, and macro scores into
a single ranked list with trade-timeframe tagging and risk metrics.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from config import WEIGHTS

log = logging.getLogger(__name__)


@dataclass
class StockScore:
    ticker:       str
    name:         str
    price:        float
    change_pct:   float

    # Component scores [0-100]
    tech_score:   float = 50.0
    fund_score:   float = 50.0
    sent_score:   float = 50.0
    macro_score:  float = 50.0

    # Composite
    total_score:  float = 0.0

    # Timeframe tags
    scalp:   bool = False
    day:     bool = False
    swing:   bool = False
    invest:  bool = False

    # Risk metrics
    atr:     float = 0.0
    atr_pct: float = 0.0
    beta:    float = 1.0

    # Signals and context
    signals:    list[str] = field(default_factory=list)
    indicators: dict      = field(default_factory=dict)
    fund_metrics: dict    = field(default_factory=dict)
    sector:     str       = "Unknown"

    # Suggested levels (calculated from ATR)
    entry:       float = 0.0
    stop_loss:   float = 0.0
    target_1:    float = 0.0
    target_2:    float = 0.0
    risk_reward: float = 0.0

    def compute_total(self) -> None:
        self.total_score = round(
            self.tech_score  * WEIGHTS["technical"]  +
            self.fund_score  * WEIGHTS["fundamental"] +
            self.sent_score  * WEIGHTS["sentiment"]   +
            self.macro_score * WEIGHTS["macro"],
            2
        )

    def compute_levels(self) -> None:
        """ATR-based entry, stop, and target levels."""
        if self.price <= 0 or self.atr <= 0:
            self.entry = self.price
            return
        self.entry     = round(self.price, 2)
        self.stop_loss = round(self.price - 1.5 * self.atr, 2)
        self.target_1  = round(self.price + 2.0 * self.atr, 2)
        self.target_2  = round(self.price + 3.5 * self.atr, 2)
        risk = self.price - self.stop_loss
        reward = self.target_1 - self.price
        self.risk_reward = round(reward / risk, 2) if risk > 0 else 0.0

    @property
    def timeframes(self) -> list[str]:
        tags = []
        if self.scalp:  tags.append("SCALP")
        if self.day:    tags.append("DAY")
        if self.swing:  tags.append("SWING")
        if self.invest: tags.append("INVEST")
        return tags or ["WATCH"]

    @property
    def grade(self) -> str:
        s = self.total_score
        if s >= 78: return "A+"
        if s >= 72: return "A"
        if s >= 66: return "B+"
        if s >= 60: return "B"
        if s >= 54: return "C+"
        if s >= 48: return "C"
        return "D"


def build_stock_score(
    ticker: str,
    tech_result:  dict,
    fund_result:  dict,
    sent_result:  dict,
    macro_result: dict,
    fund_data:    dict,
    quote:        dict,
) -> StockScore:
    """
    Assemble a StockScore from the four analyser outputs.
    """
    price      = quote.get("price", 0) if quote else 0
    change_pct = quote.get("change_pct", 0) if quote else 0
    name       = fund_data.get("company_name", ticker)
    beta       = fund_data.get("beta", 1.0) or 1.0
    sector     = fund_data.get("sector", "Unknown")

    ind = tech_result.get("indicators", {})
    atr     = ind.get("atr", 0)
    atr_pct = ind.get("atr_pct", 0)

    # Merge all signals
    all_signals = (
        tech_result.get("signals", []) +
        fund_result.get("signals", []) +
        sent_result.get("signals", []) +
        macro_result.get("signals", [])
    )
    # Deduplicate while preserving order
    seen = set()
    unique_signals = []
    for s in all_signals:
        if s not in seen:
            seen.add(s)
            unique_signals.append(s)

    tf = tech_result.get("timeframe_fit", {})

    ss = StockScore(
        ticker=ticker,
        name=name,
        price=price,
        change_pct=change_pct,
        tech_score=tech_result.get("score", 50),
        fund_score=fund_result.get("score", 50),
        sent_score=sent_result.get("score", 50),
        macro_score=macro_result.get("score", 50),
        scalp=tf.get("scalp", False),
        day=tf.get("day", False),
        swing=tf.get("swing", False),
        invest=tf.get("invest", False),
        atr=atr,
        atr_pct=atr_pct,
        beta=beta,
        signals=unique_signals,
        indicators=ind,
        fund_metrics=fund_result.get("metrics", {}),
        sector=sector,
    )
    ss.compute_total()
    ss.compute_levels()
    return ss


def rank_stocks(scores: list[StockScore], top_n: int = 10) -> list[StockScore]:
    """
    Sort by total_score descending and return top N.
    Applies a diversity filter so no more than 3 stocks from the same sector.
    """
    sorted_all = sorted(scores, key=lambda s: s.total_score, reverse=True)

    sector_count: dict[str, int] = {}
    result: list[StockScore] = []
    overflow: list[StockScore] = []

    for s in sorted_all:
        sec = s.sector
        if sector_count.get(sec, 0) < 3:
            result.append(s)
            sector_count[sec] = sector_count.get(sec, 0) + 1
        else:
            overflow.append(s)
        if len(result) == top_n:
            break

    # Fill remaining spots from overflow if needed
    remaining = top_n - len(result)
    result.extend(overflow[:remaining])

    return result[:top_n]
