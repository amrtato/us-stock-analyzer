"""
Composite scoring engine.

Merges technical, fundamental, sentiment, and macro scores into
a single ranked list with trade-timeframe tagging, risk metrics,
market-status classification, and actionable trade advice.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from config import WEIGHTS

log = logging.getLogger(__name__)


@dataclass
class StockScore:
    ticker:     str
    name:       str
    price:      float
    change_pct: float

    # Component scores [0-100]
    tech_score:  float = 50.0
    fund_score:  float = 50.0
    sent_score:  float = 50.0
    macro_score: float = 50.0

    # Composite
    total_score: float = 0.0

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
    signals:      list   = field(default_factory=list)
    indicators:   dict   = field(default_factory=dict)
    fund_metrics: dict   = field(default_factory=dict)
    sector:       str    = "Unknown"

    # Trade levels
    entry:          float = 0.0
    entry_zone_low: float = 0.0
    entry_zone_high: float = 0.0
    stop_loss:      float = 0.0
    target_1:       float = 0.0
    target_2:       float = 0.0
    risk_reward:    float = 0.0

    # Market classification (computed)
    market_status: str = "Neutral"   # Strong Bullish / Bullish / Neutral / Bearish / Strong Bearish
    risk_rating:   str = "Medium"    # Low / Medium / High
    trade_advice:  str = ""          # Actionable multi-line advice string

    # ── Computed properties ────────────────────────────────────────────────────

    def compute_total(self) -> None:
        self.total_score = round(
            self.tech_score  * WEIGHTS["technical"]  +
            self.fund_score  * WEIGHTS["fundamental"] +
            self.sent_score  * WEIGHTS["sentiment"]   +
            self.macro_score * WEIGHTS["macro"],
            2
        )

    def compute_levels(self) -> None:
        """
        Smarter entry / stop / target calculation.

        Entry is anchored to the nearest EMA support below current price,
        adjusted for RSI regime (overbought → wait for pullback,
        oversold → buy near current price).
        Stop is placed below the next meaningful support with ATR buffer.
        Targets are 2× and 3.5× the entry-to-stop risk.
        """
        if self.price <= 0:
            self.entry = self.price
            return

        price = self.price
        atr   = max(self.atr, price * 0.005)   # minimum 0.5% ATR floor
        ind   = self.indicators

        rsi    = ind.get("rsi",      50.0)
        ema9   = ind.get("ema9",     price)
        ema21  = ind.get("ema21",    price)
        ema50  = ind.get("ema50",    price)
        ema200 = ind.get("ema200",   price)
        bb_lo  = ind.get("bb_lower", price - 2 * atr)
        bb_up  = ind.get("bb_upper", price + 2 * atr)

        # Collect EMA/BB levels below current price (potential supports)
        supports_below = sorted(
            [v for v in [ema9, ema21, ema50, bb_lo] if 0 < v < price * 0.99],
            reverse=True,   # closest first
        )
        nearest_support = supports_below[0] if supports_below else (price - 1.5 * atr)

        # ── Entry zone ────────────────────────────────────────────────────────
        if rsi > 68:
            # Overbought — recommend waiting for pullback to EMA21/EMA9
            pullback = min(ema21, ema9) if ema21 < price and ema9 < price else nearest_support
            self.entry          = round(pullback * 1.003, 2)    # just above support
            self.entry_zone_low  = round(pullback * 0.995, 2)
            self.entry_zone_high = round(pullback * 1.012, 2)
        elif rsi < 32:
            # Oversold — already at or near support, buy near current
            self.entry          = round(price * 1.002, 2)
            self.entry_zone_low  = round(price * 0.993, 2)
            self.entry_zone_high = round(price * 1.010, 2)
        else:
            # Neutral RSI — enter between nearest support and current price
            midpoint = (price + nearest_support) / 2
            self.entry          = round(midpoint, 2)
            self.entry_zone_low  = round(nearest_support * 0.997, 2)
            self.entry_zone_high = round(price * 1.003, 2)

        # ── Stop loss (below the next support down with ATR buffer) ───────────
        deeper_supports = sorted(
            [v for v in [ema50, ema200, bb_lo] if v < self.entry * 0.995],
            reverse=True,
        )
        if deeper_supports:
            stop_base = deeper_supports[0]
            raw_stop  = stop_base - 0.4 * atr
        else:
            raw_stop = self.entry - 1.5 * atr

        # Clamp: stop must be between 1% and 9% below entry
        self.stop_loss = round(
            max(self.entry * 0.91, min(self.entry * 0.99, raw_stop)), 2
        )

        # ── Targets (risk-multiple based) ─────────────────────────────────────
        risk = max(self.entry - self.stop_loss, atr * 0.5)
        self.target_1    = round(self.entry + 2.0 * risk, 2)
        self.target_2    = round(self.entry + 3.5 * risk, 2)
        self.risk_reward = round((self.target_1 - self.entry) / risk, 2) if risk > 0 else 0.0

    def compute_market_status(self) -> None:
        """
        Classify market direction using a weighted signal count.
        Checks: trend alignment (EMAs), momentum (RSI, MACD), composite score.
        """
        ind   = self.indicators
        price = self.price
        score = self.total_score

        rsi       = ind.get("rsi",      50.0)
        ema50     = ind.get("ema50",    price)
        ema200    = ind.get("ema200",   price)
        macd_hist = ind.get("macd_hist", 0.0)
        adx       = ind.get("adx",      20.0)

        bull = 0
        bear = 0

        # Trend alignment (strongest signal — weight 2)
        if price > ema50 > ema200:  bull += 2
        elif price < ema50 < ema200: bear += 2
        elif price > ema50:          bull += 1
        elif price < ema50:          bear += 1

        # RSI regime
        if rsi < 35:   bull += 2   # oversold = long opportunity
        elif rsi < 50: bull += 1
        elif rsi > 65: bear += 1   # overbought = extended / caution
        elif rsi > 75: bear += 2

        # MACD momentum
        if macd_hist > 0:  bull += 1
        else:              bear += 1

        # Composite score
        if score >= 68:    bull += 2
        elif score >= 58:  bull += 1
        elif score < 48:   bear += 2
        elif score < 55:   bear += 1

        # Trend strength bonus
        if adx > 30:
            if bull > bear: bull += 1
            else:           bear += 1

        net = bull - bear
        if   net >= 5:  self.market_status = "Strong Bullish"
        elif net >= 2:  self.market_status = "Bullish"
        elif net >= -1: self.market_status = "Neutral"
        elif net >= -4: self.market_status = "Bearish"
        else:           self.market_status = "Strong Bearish"

    def compute_risk_rating(self) -> None:
        """
        Classify trade risk as Low / Medium / High.
        Factors: ATR%, beta, risk:reward ratio, score certainty.
        """
        risk_pts = 0

        # Volatility (ATR as % of price)
        if self.atr_pct > 4.0:   risk_pts += 3
        elif self.atr_pct > 2.5: risk_pts += 2
        elif self.atr_pct > 1.5: risk_pts += 1

        # Beta
        if self.beta > 2.0:   risk_pts += 2
        elif self.beta > 1.4: risk_pts += 1

        # Poor risk:reward
        if self.risk_reward < 1.2: risk_pts += 2
        elif self.risk_reward < 1.8: risk_pts += 1

        # Low score = uncertain direction
        if self.total_score < 48: risk_pts += 2
        elif self.total_score < 55: risk_pts += 1

        # Bearish trend penalty
        if self.market_status in ("Bearish", "Strong Bearish"): risk_pts += 2

        if risk_pts >= 6:   self.risk_rating = "High"
        elif risk_pts >= 3: self.risk_rating = "Medium"
        else:               self.risk_rating = "Low"

    def compute_trade_advice(self) -> None:
        """
        Generate a concise, actionable trade advice string covering:
        - Entry rationale (why this level)
        - Stop loss rationale
        - Target rationale
        - Position sizing guidance
        """
        ind    = self.indicators
        price  = self.price
        rsi    = ind.get("rsi",    50.0)
        ema21  = ind.get("ema21",  price)
        ema50  = ind.get("ema50",  price)
        ema200 = ind.get("ema200", price)
        adx    = ind.get("adx",   20.0)
        stop   = self.stop_loss
        t1     = self.target_1
        t2     = self.target_2
        entry  = self.entry

        lines = []

        # ── Overall verdict ───────────────────────────────────────────────────
        status_map = {
            "Strong Bullish": "presents a strong bullish setup",
            "Bullish":        "is showing a bullish signal",
            "Neutral":        "is in a neutral/consolidation phase",
            "Bearish":        "is under bearish pressure — trade with caution",
            "Strong Bearish": "is in a strong downtrend — high caution required",
        }
        verdict = status_map.get(self.market_status, "shows a mixed picture")
        lines.append(f"{self.ticker} {verdict} with a composite score of {self.total_score:.1f}/100.")

        # ── Entry rationale ───────────────────────────────────────────────────
        if rsi > 68:
            lines.append(
                f"RSI is elevated at {rsi:.1f} — the stock is overbought short-term. "
                f"Wait for a pullback toward the EMA21 (${ema21:.2f}) before entering "
                f"to get a better entry and reduce downside risk."
            )
        elif rsi < 32:
            lines.append(
                f"RSI is oversold at {rsi:.1f} — the stock is near a support floor. "
                f"Current price levels offer a favourable entry with high reward potential."
            )
        else:
            nearest_ema = ema21 if abs(price - ema21) < abs(price - ema50) else ema50
            lines.append(
                f"RSI at {rsi:.1f} is in the neutral zone. "
                f"Enter near the EMA support at ${nearest_ema:.2f} for the best risk/reward."
            )

        # ── Entry zone ────────────────────────────────────────────────────────
        lines.append(
            f"Optimal Entry Zone: ${self.entry_zone_low:.2f} – ${self.entry_zone_high:.2f}  "
            f"(reference entry: ${entry:.2f})"
        )

        # ── Trend context ─────────────────────────────────────────────────────
        if price > ema50 > ema200:
            lines.append(
                f"Long-term trend is bullish — price is above both EMA50 (${ema50:.2f}) "
                f"and EMA200 (${ema200:.2f}). Trend is your friend here."
            )
        elif price < ema50 and ema50 < ema200:
            lines.append(
                f"Caution: stock is in a downtrend — below EMA50 (${ema50:.2f}) "
                f"and EMA200 (${ema200:.2f}). Only trade short-term bounces with tight stops."
            )
        elif price > ema200:
            lines.append(
                f"Price is above the long-term EMA200 (${ema200:.2f}), "
                f"suggesting the broader trend is intact."
            )

        if adx > 30:
            lines.append(f"ADX at {adx:.0f} confirms a strong, well-defined trend.")
        elif adx < 20:
            lines.append(f"ADX at {adx:.0f} — weak trend; expect choppy price action.")

        # ── Stop & target ─────────────────────────────────────────────────────
        stop_pct = (entry - stop) / entry * 100 if entry > 0 else 0
        t1_pct   = (t1 - entry)  / entry * 100 if entry > 0 else 0
        t2_pct   = (t2 - entry)  / entry * 100 if entry > 0 else 0
        lines.append(
            f"Stop Loss: ${stop:.2f}  (-{stop_pct:.1f}% from entry, "
            f"below key support)."
        )
        lines.append(
            f"Target 1: ${t1:.2f}  (+{t1_pct:.1f}%)  |  "
            f"Target 2: ${t2:.2f}  (+{t2_pct:.1f}%)  |  "
            f"Risk:Reward: {self.risk_reward:.1f}x"
        )

        # ── Position sizing ────────────────────────────────────────────────────
        if self.risk_rating == "High":
            lines.append(
                f"High volatility (ATR {self.atr_pct:.1f}% / Beta {self.beta:.2f}) — "
                f"limit position to 1-2% of capital. Consider a scaled entry."
            )
        elif self.risk_rating == "Medium":
            lines.append(
                f"Moderate risk (ATR {self.atr_pct:.1f}%) — "
                f"standard position size of 2-4% of capital is appropriate."
            )
        else:
            lines.append(
                f"Low volatility (ATR {self.atr_pct:.1f}%) — "
                f"position size of 3-5% of capital acceptable."
            )

        self.trade_advice = "\n\n".join(lines)

    @property
    def timeframes(self) -> list:
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

    @property
    def status_emoji(self) -> str:
        return {
            "Strong Bullish": "🟢",
            "Bullish":        "🟩",
            "Neutral":        "⬜",
            "Bearish":        "🟥",
            "Strong Bearish": "🔴",
        }.get(self.market_status, "⬜")

    @property
    def risk_emoji(self) -> str:
        return {"Low": "✅", "Medium": "⚠️", "High": "🚨"}.get(self.risk_rating, "⚠️")

    @property
    def status_color(self) -> str:
        return {
            "Strong Bullish": "#00ff88",
            "Bullish":        "#7dff7d",
            "Neutral":        "#aaaaaa",
            "Bearish":        "#ff8c66",
            "Strong Bearish": "#ff4444",
        }.get(self.market_status, "#aaaaaa")

    @property
    def risk_color(self) -> str:
        return {"Low": "#00ff88", "Medium": "#ffd700", "High": "#ff4444"}.get(self.risk_rating, "#ffd700")


def build_stock_score(
    ticker:      str,
    tech_result:  dict,
    fund_result:  dict,
    sent_result:  dict,
    macro_result: dict,
    fund_data:    dict,
    quote:        dict,
) -> StockScore:
    """Assemble a StockScore from the four analyser outputs."""
    price      = quote.get("price", 0) if quote else 0
    change_pct = quote.get("change_pct", 0) if quote else 0
    name       = fund_data.get("company_name", ticker)
    beta       = fund_data.get("beta", 1.0) or 1.0
    sector     = fund_data.get("sector", "Unknown")

    ind     = tech_result.get("indicators", {})
    atr     = ind.get("atr", 0)
    atr_pct = ind.get("atr_pct", 0)

    # Merge all signals
    all_signals = (
        tech_result.get("signals", []) +
        fund_result.get("signals", []) +
        sent_result.get("signals", []) +
        macro_result.get("signals", [])
    )
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
    ss.compute_market_status()
    ss.compute_risk_rating()
    ss.compute_trade_advice()
    return ss


def rank_stocks(scores: list, top_n: int = 10) -> list:
    """
    Sort by total_score descending and return top N.
    Applies a diversity filter so no more than 3 stocks from the same sector.
    """
    sorted_all = sorted(scores, key=lambda s: s.total_score, reverse=True)

    sector_count = {}
    result   = []
    overflow = []

    for s in sorted_all:
        sec = s.sector
        if sector_count.get(sec, 0) < 3:
            result.append(s)
            sector_count[sec] = sector_count.get(sec, 0) + 1
        else:
            overflow.append(s)
        if len(result) == top_n:
            break

    remaining = top_n - len(result)
    result.extend(overflow[:remaining])
    return result[:top_n]
