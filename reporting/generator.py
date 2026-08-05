"""
Report generator.

Produces:
  1. Terminal-rendered rich report (colour-coded, tables)
  2. Markdown file saved to reports/
  3. AI narrative via Claude API (optional but recommended)
"""
from __future__ import annotations

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from scoring.scorer import StockScore
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, REPORT_DIR

log = logging.getLogger(__name__)
console = Console()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _score_color(score: float) -> str:
    if score >= 72: return "bold green"
    if score >= 60: return "yellow"
    if score >= 48: return "orange1"
    return "red"


def _chg_color(pct: float) -> str:
    return "green" if pct >= 0 else "red"


def _tf_badges(stock: StockScore) -> str:
    tags = stock.timeframes
    badge_map = {"SCALP": "⚡", "DAY": "📅", "SWING": "📈", "INVEST": "💼", "WATCH": "👁"}
    return " ".join(badge_map.get(t, t) for t in tags)


# ── Terminal report ────────────────────────────────────────────────────────────

def print_terminal_report(ranked: list[StockScore], date_str: str) -> None:
    console.rule(f"[bold cyan]US STOCK DAILY REPORT — {date_str}[/bold cyan]")
    console.print()

    table = Table(
        title="[bold]Top 10 by Composite Score[/bold]",
        box=box.DOUBLE_EDGE,
        show_lines=True,
        highlight=True,
        caption=f"Weights: Tech {40}% | Fund {30}% | Sentiment {18}% | Macro {12}%",
    )

    table.add_column("Rank",    style="bold", justify="center", width=4)
    table.add_column("Ticker",  style="bold cyan", width=7)
    table.add_column("Name",    width=22)
    table.add_column("Price",   justify="right", width=8)
    table.add_column("Chg%",    justify="right", width=7)
    table.add_column("Score",   justify="center", width=7)
    table.add_column("Grade",   justify="center", width=6)
    table.add_column("Tech",    justify="center", width=5)
    table.add_column("Fund",    justify="center", width=5)
    table.add_column("Sent",    justify="center", width=5)
    table.add_column("Macro",   justify="center", width=5)
    table.add_column("Frames",  width=16)
    table.add_column("Sector",  width=15)

    for i, stock in enumerate(ranked, 1):
        chg_str  = f"[{_chg_color(stock.change_pct)}]{stock.change_pct:+.1f}%[/]"
        sc_style = _score_color(stock.total_score)

        table.add_row(
            str(i),
            stock.ticker,
            stock.name[:22],
            f"${stock.price:,.2f}",
            chg_str,
            f"[{sc_style}]{stock.total_score:.1f}[/]",
            f"[{sc_style}]{stock.grade}[/]",
            f"{stock.tech_score:.0f}",
            f"{stock.fund_score:.0f}",
            f"{stock.sent_score:.0f}",
            f"{stock.macro_score:.0f}",
            _tf_badges(stock),
            stock.sector[:15],
        )

    console.print(table)
    console.print()

    # Detail cards for each stock
    for i, stock in enumerate(ranked, 1):
        _print_stock_card(i, stock)


def _print_stock_card(rank: int, stock: StockScore) -> None:
    ind = stock.indicators
    color = _score_color(stock.total_score)

    lines = [
        f"[bold]{rank}. {stock.ticker}[/bold] — {stock.name}",
        f"  Price: [bold]${stock.price:,.2f}[/bold]  ({stock.change_pct:+.2f}%)  "
        f"Beta: {f'{stock.beta:.2f}' if stock.beta is not None else '—'}  "
        f"ATR: ${stock.atr:.2f} ({stock.atr_pct:.1f}%)",
        "",
        f"  [bold]Levels:[/bold]  Entry: ${stock.entry:,.2f}  "
        f"Stop: ${stock.stop_loss:,.2f}  "
        f"T1: ${stock.target_1:,.2f}  T2: ${stock.target_2:,.2f}  "
        f"R:R {stock.risk_reward:.1f}x",
        "",
        f"  [bold]Indicators:[/bold]  RSI {ind.get('rsi','—')}  |  "
        f"MACD hist {ind.get('macd_hist','—')}  |  "
        f"Stoch {ind.get('stoch_k','—')}  |  "
        f"EMA50 {'>' if stock.price > ind.get('ema50', stock.price) else '<'} price  |  "
        f"ADX {ind.get('adx','—')}",
        "",
    ]

    if stock.signals:
        lines.append("  [bold]Signals:[/bold]")
        for sig in stock.signals[:6]:
            lines.append(f"    • {sig}")
        lines.append("")

    if stock.fund_metrics:
        m = stock.fund_metrics
        parts = []
        if m.get("pe"):           parts.append(f"P/E {m['pe']}")
        if m.get("peg"):          parts.append(f"PEG {m['peg']}")
        if m.get("roe"):          parts.append(f"ROE {m['roe']}")
        if m.get("revenue_growth"): parts.append(f"Rev ↑{m['revenue_growth']}")
        if m.get("market_cap_b"): parts.append(f"MCap ${m['market_cap_b']}B")
        if parts:
            lines.append(f"  [bold]Fundamentals:[/bold]  " + "  |  ".join(parts))

    console.print(Panel(
        "\n".join(lines),
        border_style=color,
        expand=False,
        padding=(0, 1),
    ))
    console.print()


# ── Markdown report ────────────────────────────────────────────────────────────

def save_markdown_report(
    ranked:    list[StockScore],
    date_str:  str,
    ai_narrative: str = "",
) -> Path:
    os.makedirs(REPORT_DIR, exist_ok=True)
    filename = Path(REPORT_DIR) / f"report_{date_str.replace('-', '')}.md"

    lines = [
        f"# US Stock Daily Report — {date_str}",
        "",
        f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} ET  ",
        f"> Universe: DJIA + NASDAQ-100 + S&P500 top components  ",
        f"> Scoring weights: Technical 38% | Fundamental 30% | Sentiment 18% | Macro 14%",
        "",
        "---",
        "",
    ]

    if ai_narrative:
        lines += [
            "## AI Market Context",
            "",
            ai_narrative,
            "",
            "---",
            "",
        ]

    lines += ["## Top 10 by Composite Score", "",
              "_A screening rank, not a forecast — the composite has no "
              "demonstrated relationship with forward returns._", ""]

    # Summary table
    lines += [
        "| # | Ticker | Name | Price | Chg% | Score | Grade | Timeframes | Sector |",
        "|---|--------|------|-------|------|-------|-------|------------|--------|",
    ]
    for i, s in enumerate(ranked, 1):
        tfs = " ".join(s.timeframes)
        lines.append(
            f"| {i} | **{s.ticker}** | {s.name[:25]} | ${s.price:,.2f} | "
            f"{s.change_pct:+.1f}% | **{s.total_score:.1f}** | {s.grade} | {tfs} | {s.sector} |"
        )

    lines += ["", "---", ""]

    # Detail sections
    for i, s in enumerate(ranked, 1):
        ind = s.indicators
        m   = s.fund_metrics
        lines += [
            f"## {i}. {s.ticker} — {s.name}",
            "",
            f"**Score: {s.total_score:.1f} ({s.grade})** | "
            f"Tech: {s.tech_score:.0f} | Fund: {s.fund_score:.0f} | "
            f"Sent: {s.sent_score:.0f} | Macro: {s.macro_score:.0f}",
            "",
            f"**Price:** ${s.price:,.2f} ({s.change_pct:+.2f}%) | "
            f"**Beta:** {f'{s.beta:.2f}' if s.beta is not None else '—'} | "
            f"**ATR:** ${s.atr:.2f} ({s.atr_pct:.1f}%)",
            "",
            "### Trade Levels",
            f"| Entry | Stop Loss | Target 1 | Target 2 | Risk:Reward |",
            f"|-------|-----------|----------|----------|-------------|",
            f"| ${s.entry:,.2f} | ${s.stop_loss:,.2f} | ${s.target_1:,.2f} | ${s.target_2:,.2f} | {s.risk_reward:.1f}x |",
            "",
            "### Technical Indicators",
            f"| RSI | MACD Hist | Stoch K | ADX | EMA50 | EMA200 | Vol Ratio | ROC5d |",
            f"|-----|-----------|---------|-----|-------|--------|-----------|-------|",
            (
                f"| {ind.get('rsi','—')} | {ind.get('macd_hist','—')} | "
                f"{ind.get('stoch_k','—')} | {ind.get('adx','—')} | "
                f"{ind.get('ema50','—')} | {ind.get('ema200','—')} | "
                f"{ind.get('vol_ratio','—')}x | {ind.get('roc5','—')}% |"
            ),
            "",
        ]

        if m:
            parts = []
            for k, v in m.items():
                if v is not None:
                    parts.append(f"**{k.replace('_',' ').title()}:** {v}")
            if parts:
                lines += ["### Fundamentals", " | ".join(parts), ""]

        if s.signals:
            lines += ["### Signals & Catalysts"]
            for sig in s.signals:
                lines.append(f"- {sig}")
            lines.append("")

        lines += ["---", ""]

    lines += [
        "## Disclaimer",
        "",
        "> This report is generated by an automated algorithmic system for informational "
        "and educational purposes only. It does **not** constitute financial advice. "
        "Past signals do not guarantee future performance. Always apply your own due "
        "diligence and risk management before trading.",
        "",
    ]

    filename.write_text("\n".join(lines), encoding="utf-8")
    log.info("Markdown report saved → %s", filename)
    return filename


# ── AI Narrative via Claude API ────────────────────────────────────────────────

def generate_ai_narrative(ranked: list[StockScore]) -> str:
    """
    Call Claude to generate a 300-500 word market context and stock narrative.
    Falls back gracefully if no API key is set.
    """
    if not ANTHROPIC_API_KEY:
        return "_Set ANTHROPIC_API_KEY in .env for AI-generated market narrative._"

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        # Build a concise summary to feed Claude
        stock_lines = []
        for i, s in enumerate(ranked, 1):
            tfs = ", ".join(s.timeframes)
            sigs = "; ".join(s.signals[:3])
            stock_lines.append(
                f"{i}. {s.ticker} ({s.name}) — Score {s.total_score:.1f}, "
                f"Price ${s.price:.2f} ({s.change_pct:+.1f}%), "
                f"Timeframes: {tfs}, "
                f"Sector: {s.sector}, "
                f"Key signals: {sigs}"
            )

        prompt = f"""You are a professional US equity analyst. Today is {datetime.now().strftime('%B %d, %Y')}.

Based on the following algorithmic stock scores and signals, write a concise market context and trading brief:

TOP 10 STOCKS TODAY:
{chr(10).join(stock_lines)}

The composite score is a SCREENING RANK. Walk-forward testing (101 stocks,
2022-2026, non-overlapping windows) found no reliable relationship between it
and forward returns. Describe what the indicators currently show; do not imply
these are predictions or recommendations, and do not invent a rationale for why
a stock will rise.

Write a 350-450 word narrative covering:
1. **Market Context** (2-3 sentences on macro/sector rotation backdrop)
2. **What Stands Out** — 3-4 stocks whose indicator readings are notable, and why
3. **Risk Factors** — key risks worth watching today
4. **Timeframe Summary** — which stocks suit scalping vs swing vs longer-term

Be specific and data-driven. Describe, don't predict. Use plain text with
markdown headers."""

        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=700,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    except Exception as exc:
        log.error("AI narrative generation failed: %s", exc)
        return f"_AI narrative unavailable: {exc}_"
