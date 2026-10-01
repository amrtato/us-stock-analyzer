"""
Short-side scoring: trade levels, risk classification and advice.

EVERY LEVEL IS MIRRORED, AND THAT MIRROR IS NOT SYMMETRIC
    A long's stop sits below entry and its loss is bounded at -100%. A short's
    stop sits ABOVE entry and its loss is unbounded — a takeover bid or a
    squeeze can gap straight through the stop, and the position grows as it
    moves against you (losing shorts get bigger, losing longs get smaller).
    So this module deliberately differs from ``StockScore.compute_levels`` in
    three ways rather than just flipping the arithmetic:

      1. Entry is anchored to RESISTANCE ABOVE price, not support below.
         Shorting into a bounce gives a definable stop; shorting a stock that
         has already fallen out of bed gives you the worst fill and the
         tightest squeeze risk.
      2. Stop distance is CAPPED TIGHTER (max 7% vs the long side's 9%),
         because the tail beyond the stop is unbounded.
      3. Position sizing is roughly half the long-side guidance at equal
         volatility, and gap risk is quantified rather than mentioned.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# A short entry anchored more than this above spot stops describing a
# tradeable bounce and becomes a wish.
MAX_BOUNCE_HEIGHT = 0.06       # 6%
# Stops are capped tighter than the long side's 9% — see module docstring.
MAX_STOP_PCT = 0.07


@dataclass
class ShortScore:
    ticker: str
    name: str
    price: float
    change_pct: float

    bear_score: float = 50.0
    subscores: dict = field(default_factory=dict)

    atr: float = 0.0
    atr_pct: float = 0.0
    beta: Optional[float] = None
    sector: str = "Unknown"

    signals: list = field(default_factory=list)
    indicators: dict = field(default_factory=dict)

    tradeable: bool = True
    gate_reasons: list = field(default_factory=list)

    # Trade levels (entry ABOVE or at price; stop ABOVE entry; targets BELOW)
    entry: float = 0.0
    entry_zone_low: float = 0.0
    entry_zone_high: float = 0.0
    stop_loss: float = 0.0
    target_1: float = 0.0
    target_2: float = 0.0
    risk_reward: float = 0.0
    entry_is_high_bounce: bool = False

    squeeze_risk: str = "Medium"       # Low / Medium / High
    # Describes how closely the CHART matches the bearish template -- NOT
    # confidence that the trade pays. Walk-forward testing found no edge,
    # so "Strong/Moderate" would have asserted a forecast the data refuses.
    setup_quality: str = "Weak"        # Weak / Partial / Textbook
    trade_advice: str = ""

    # ── Levels ────────────────────────────────────────────────────────────────
    def compute_levels(self) -> None:
        if self.price <= 0:
            self.entry = self.price
            return

        price = self.price
        atr = max(self.atr, price * 0.005)
        ind = self.indicators

        rsi = ind.get("rsi", 50.0)
        ema21 = ind.get("ema21", price)
        ema50 = ind.get("ema50", price)
        ema200 = ind.get("ema200", price)
        bb_up = ind.get("bb_upper", price + 2 * atr)
        bb_lo = ind.get("bb_lower", price - 2 * atr)

        # ── Entry: short into a bounce toward the nearest resistance above ────
        resistances = sorted(v for v in (ema21, ema50, bb_up) if v and v > price * 1.01)

        if rsi < 35:
            # Already washed out. The honest answer is that there is no good
            # short entry here; anchor to the first resistance above and say so.
            anchor = resistances[0] if resistances else price * 1.03
            ceiling = price * (1.0 + MAX_BOUNCE_HEIGHT)
            if anchor > ceiling:
                anchor = ceiling
                self.entry_is_high_bounce = True
            self.entry = round(anchor * 0.997, 2)
            self.entry_zone_low = round(anchor * 0.988, 2)
            self.entry_zone_high = round(anchor * 1.005, 2)
        elif rsi > 62:
            # Bouncing into resistance — the best short entry this model finds.
            self.entry = round(price * 0.998, 2)
            self.entry_zone_low = round(price * 0.990, 2)
            self.entry_zone_high = round(price * 1.007, 2)
        else:
            nearest = resistances[0] if resistances else price * 1.02
            ceiling = price * (1.0 + MAX_BOUNCE_HEIGHT)
            if nearest > ceiling:
                nearest = ceiling
                self.entry_is_high_bounce = True
            midpoint = (price + nearest) / 2
            self.entry = round(midpoint, 2)
            self.entry_zone_low = round(price * 0.997, 2)
            self.entry_zone_high = round(nearest * 1.003, 2)

        # ── Stop: ABOVE the next resistance up, with an ATR buffer ────────────
        higher = sorted(v for v in (ema50, ema200, bb_up) if v and v > self.entry * 1.005)
        raw_stop = (higher[0] + 0.4 * atr) if higher else (self.entry + 1.5 * atr)

        # Clamp between 1% and MAX_STOP_PCT above entry.
        self.stop_loss = round(
            min(self.entry * (1 + MAX_STOP_PCT), max(self.entry * 1.01, raw_stop)), 2
        )

        # ── Targets: BELOW entry, anchored on real support ────────────────────
        risk = max(self.stop_loss - self.entry, atr * 0.5)
        supports = sorted(
            (v for v in (bb_lo, ema200) if v and v < self.entry * 0.99), reverse=True
        )
        t1_structural = supports[0] if supports else None
        t1_risk_based = self.entry - 2.0 * risk

        if t1_structural and t1_structural <= self.entry - 0.8 * risk:
            self.target_1 = round(t1_structural, 2)
        else:
            self.target_1 = round(t1_risk_based, 2)

        # A target above the price you can short at right now is not a target.
        if self.target_1 >= price:
            self.target_1 = round(min(price - 1.2 * risk, price * 0.98), 2)
        self.target_1 = max(self.target_1, 0.01)

        t1_dist = self.entry - self.target_1
        self.target_2 = round(max(self.entry - max(1.75 * t1_dist, 3.0 * risk), 0.01), 2)
        self.risk_reward = round(t1_dist / risk, 2) if risk > 0 else 0.0

    # ── Risk ──────────────────────────────────────────────────────────────────
    def compute_squeeze_risk(self) -> None:
        """How likely is this to rip against you before it works?

        Squeezes feed on names that are already extended below their averages
        (everyone short, nobody left to sell), thin, or oversold.
        """
        pts = 0
        ind = self.indicators
        rsi = ind.get("rsi", 50.0)
        price = self.price or 1.0
        ema21 = ind.get("ema21") or price
        dollar_vol = ind.get("dollar_volume") or 0.0

        if rsi < 25:
            pts += 3
        elif rsi < 32:
            pts += 2

        stretch = (ema21 - price) / price * 100 if price else 0
        if stretch > 8:
            pts += 3                       # far below EMA21 = rubber-banded
        elif stretch > 4:
            pts += 1

        if dollar_vol and dollar_vol < 20_000_000:
            pts += 3
        elif dollar_vol and dollar_vol < 75_000_000:
            pts += 1

        if self.atr_pct > 5:
            pts += 2
        elif self.atr_pct > 3:
            pts += 1

        if self.beta is not None and self.beta > 1.8:
            pts += 1

        self.squeeze_risk = "High" if pts >= 6 else ("Medium" if pts >= 3 else "Low")

    def compute_setup_quality(self) -> None:
        """How completely does this chart match the bearish template?

        Deliberately phrased as pattern-match, not conviction: a textbook
        breakdown and a profitable short are different claims, and only the
        first one is measurable here.
        """
        s = self.bear_score
        if s >= 70 and self.tradeable and self.squeeze_risk != "High":
            self.setup_quality = "Textbook"
        elif s >= 60 and self.tradeable:
            self.setup_quality = "Partial"
        else:
            self.setup_quality = "Weak"

    # ── Advice ────────────────────────────────────────────────────────────────
    def compute_trade_advice(self) -> None:
        ind = self.indicators
        price = self.price
        rsi = ind.get("rsi", 50.0)
        ema50 = ind.get("ema50", price)
        ema200 = ind.get("ema200", price)
        adx = ind.get("adx", 20.0)
        rel = ind.get("rel_strength")
        entry, stop = self.entry, self.stop_loss
        lines = []

        lines.append(
            f"{self.ticker} scores {self.bear_score:.1f}/100 on the downtrend-continuation "
            f"model — setup: {self.setup_quality}. That measures how closely the chart "
            f"matches a breakdown, not the odds it keeps falling: this model showed no "
            f"measurable edge in walk-forward testing."
        )

        if ema50 and ema200 and ema50 < ema200 and price < ema50:
            lines.append(
                f"Structure is bearish: price ${price:,.2f} sits below EMA50 "
                f"(${ema50:,.2f}) which is below EMA200 (${ema200:,.2f}). "
                f"This is trend continuation, not a call on valuation."
            )
        elif price > ema200:
            lines.append(
                f"⚠️ Price is still ABOVE the EMA200 (${ema200:,.2f}). This is a "
                f"counter-trend short — the lower-probability version of this trade."
            )

        if rel is not None:
            if rel < 0:
                lines.append(
                    f"Relative weakness: lagging SPY by {abs(rel):.1f}pp over 60 days. "
                    f"That gap matters more than the absolute fall — a short pays the "
                    f"market's drift, so underperformance is what you are actually selling."
                )
            else:
                lines.append(
                    f"⚠️ This name is OUTPERFORMING SPY by {rel:.1f}pp over 60 days. "
                    f"Shorting relative strength is how short books bleed."
                )

        if rsi > 62:
            lines.append(
                f"RSI {rsi:.1f} — bouncing into resistance. This is the entry the "
                f"model prefers: you are selling a rally inside a downtrend."
            )
        elif rsi < 32:
            lines.append(
                f"RSI {rsi:.1f} — already oversold. Entering here means selling to "
                f"the people covering. Wait for a bounce toward ${entry:,.2f}."
            )

        if adx > 30:
            lines.append(f"ADX {adx:.0f} confirms the downtrend is well-defined.")
        elif adx < 20:
            lines.append(f"ADX {adx:.0f} — weak/choppy trend; breakdowns tend to fail.")

        if self.entry_is_high_bounce:
            lines.append(
                f"⚠️ These levels assume a bounce to ${entry:,.2f}; {self.ticker} trades "
                f"at ${price:,.2f} now. There is no entry at today's price."
            )

        stop_pct = (stop - entry) / entry * 100 if entry else 0
        t1_pct = (entry - self.target_1) / entry * 100 if entry else 0
        t2_pct = (entry - self.target_2) / entry * 100 if entry else 0
        lines.append(
            f"Entry ${entry:,.2f} (zone ${self.entry_zone_low:,.2f}–${self.entry_zone_high:,.2f})  |  "
            f"Stop ${stop:,.2f} (+{stop_pct:.1f}% ABOVE entry — a short's stop is above)."
        )
        lines.append(
            f"Target 1: ${self.target_1:,.2f} (-{t1_pct:.1f}%)  |  "
            f"Target 2: ${self.target_2:,.2f} (-{t2_pct:.1f}%)  |  R:R {self.risk_reward:.1f}x"
        )

        # Gap risk, quantified rather than hand-waved.
        gap_loss = stop_pct * 2.5
        lines.append(
            f"🚨 Unbounded downside: the stop caps a normal loss at {stop_pct:.1f}%, but an "
            f"overnight gap — earnings, a bid, an FDA headline — can fill far above it. "
            f"A 2.5x gap-through would cost roughly {gap_loss:.0f}%. Size for that, not for the stop."
        )

        if self.squeeze_risk == "High":
            lines.append(
                "🚨 HIGH squeeze risk (oversold / extended / thin). Position at 0.5–1% of "
                "capital, or skip — this is the profile that rips 20% in a session."
            )
        elif self.squeeze_risk == "Medium":
            lines.append(
                f"⚠️ Moderate squeeze risk (ATR {self.atr_pct:.1f}%). "
                "Position 1–2% of capital with a hard stop, not a mental one."
            )
        else:
            lines.append(
                f"Squeeze risk low (ATR {self.atr_pct:.1f}%, liquid). "
                "Position 2–3% of capital — still below the long-side equivalent."
            )

        if self.gate_reasons:
            lines.append("Gate warnings: " + "; ".join(self.gate_reasons) + ".")

        self.trade_advice = "\n\n".join(lines)

    # ── Display helpers ───────────────────────────────────────────────────────
    @property
    def grade(self) -> str:
        s = self.bear_score
        if s >= 78: return "A+"
        if s >= 72: return "A"
        if s >= 66: return "B+"
        if s >= 60: return "B"
        if s >= 54: return "C+"
        if s >= 48: return "C"
        return "D"

    @property
    def setup_emoji(self) -> str:
        return {"Textbook": "🔴", "Partial": "🟠", "Weak": "⬜"}.get(self.setup_quality, "⬜")

    @property
    def squeeze_emoji(self) -> str:
        return {"Low": "✅", "Medium": "⚠️", "High": "🚨"}.get(self.squeeze_risk, "⚠️")

    @property
    def setup_color(self) -> str:
        return {"Textbook": "#ff4444", "Partial": "#ff8c66", "Weak": "#aaaaaa"}.get(
            self.setup_quality, "#aaaaaa")

    @property
    def squeeze_color(self) -> str:
        return {"Low": "#00ff88", "Medium": "#ffd700", "High": "#ff4444"}.get(
            self.squeeze_risk, "#ffd700")


def build_short_score(ticker: str, bear_result: dict, fund_data: dict, quote: dict) -> ShortScore:
    """Assemble a ShortScore from an ``analyse_bearish`` result."""
    price = quote.get("price", 0) if quote else 0
    if not price:
        price = bear_result.get("indicators", {}).get("price", 0) or 0
    change_pct = quote.get("change_pct", 0) if quote else 0
    name = (fund_data or {}).get("company_name", ticker)
    raw_beta = (fund_data or {}).get("beta")
    beta = float(raw_beta) if raw_beta not in (None, "") else None

    ind = bear_result.get("indicators", {})
    gates = bear_result.get("gates", {})

    ss = ShortScore(
        ticker=ticker,
        name=name,
        price=price,
        change_pct=change_pct,
        bear_score=bear_result.get("score", 50.0),
        subscores=bear_result.get("subscores", {}),
        atr=ind.get("atr", 0) or 0,
        atr_pct=ind.get("atr_pct", 0) or 0,
        beta=beta,
        sector=(fund_data or {}).get("sector", "Unknown"),
        signals=bear_result.get("signals", []),
        indicators=ind,
        tradeable=bool(gates.get("tradeable", True)),
        gate_reasons=list(gates.get("reasons", [])),
    )
    ss.compute_levels()
    ss.compute_squeeze_risk()
    ss.compute_setup_quality()
    ss.compute_trade_advice()
    return ss


def rank_shorts(shorts: list, top_n: int = 10, require_tradeable: bool = True) -> list:
    """Highest bear score first, max 3 per sector, gated names demoted not dropped."""
    pool = sorted(shorts, key=lambda s: s.bear_score, reverse=True)
    if require_tradeable:
        pool = [s for s in pool if s.tradeable] + [s for s in pool if not s.tradeable]

    sector_count: dict[str, int] = {}
    result, overflow = [], []
    for s in pool:
        if sector_count.get(s.sector, 0) < 3:
            result.append(s)
            sector_count[s.sector] = sector_count.get(s.sector, 0) + 1
        else:
            overflow.append(s)
        if len(result) == top_n:
            break
    result.extend(overflow[: max(0, top_n - len(result))])
    return result[:top_n]
