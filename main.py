"""
US Stock Analyzer — Main Orchestrator

Usage:
  python main.py              # run once immediately
  python main.py --schedule   # run daily at 07:30 ET (pre-market)
  python main.py --tickers AAPL MSFT NVDA  # analyse specific tickers only
  python main.py --no-ai      # skip Claude narrative (faster / no API key needed)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import schedule

from config import ALL_STOCKS, TOP_N, WEIGHTS
from data.fetcher import fetch_price_history, fetch_fundamentals, fetch_batch_quotes
from data.news_fetcher import fetch_rss_news, fetch_newsapi, aggregate_ticker_news
from analyzers.technical   import analyse_technical
from analyzers.fundamental import analyse_fundamental
from analyzers.sentiment   import analyse_sentiment
from analyzers.macro       import analyse_macro, update_sector_momentum
from scoring.scorer        import build_stock_score, rank_stocks, StockScore
from reporting.generator   import (
    print_terminal_report,
    save_markdown_report,
    generate_ai_narrative,
)

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")


# ── Sector ETF proxies for sector-rotation scoring ─────────────────────────────
SECTOR_ETFS = {
    "Technology":   "XLK",
    "Financials":   "XLF",
    "Healthcare":   "XLV",
    "Consumer":     "XLY",
    "Industrials":  "XLI",
    "Energy":       "XLE",
    "Communication": "XLC",
    "Materials":    "XLB",
    "Utilities":    "XLU",
    "Real Estate":  "XLRE",
}


def compute_sector_returns() -> dict[str, float]:
    """Fetch 1-day returns for each sector ETF and update macro module."""
    etf_tickers = list(SECTOR_ETFS.values())
    quotes = fetch_batch_quotes(etf_tickers)
    returns = {}
    for sector, etf in SECTOR_ETFS.items():
        if etf in quotes:
            returns[sector] = quotes[etf]["change_pct"]
    return returns


def analyse_one(ticker: str, news_map: dict, quotes: dict) -> StockScore | None:
    """Full analysis pipeline for a single ticker."""
    try:
        # 1. Price history
        df = fetch_price_history(ticker, period="1y", interval="1d")
        if df.empty:
            log.warning("Skipping %s — no price data", ticker)
            return None

        intra = fetch_price_history(ticker, period="5d", interval="5m")

        # 2. Fundamentals
        fund_data = fetch_fundamentals(ticker)
        quote     = quotes.get(ticker, {})
        price     = quote.get("price", float(df["Close"].iloc[-1]))

        # 3. Run the four analysers
        tech_result  = analyse_technical(ticker, df, intra)
        fund_result  = analyse_fundamental(ticker, fund_data, price)
        sent_result  = analyse_sentiment(
            ticker,
            aggregate_ticker_news(ticker, news_map),
            quote,
        )
        macro_result = analyse_macro(ticker, fund_data, quote)

        # 4. Assemble composite score
        stock_score = build_stock_score(
            ticker, tech_result, fund_result, sent_result, macro_result,
            fund_data, quote,
        )
        return stock_score

    except Exception as exc:
        log.error("Analysis failed for %s: %s", ticker, exc)
        return None


def run_analysis(tickers: list[str], use_ai: bool = True) -> list[StockScore]:
    """
    Full pipeline: fetch → analyse → rank → report.
    Returns ranked top-N StockScore list.
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    log.info("=== US Stock Analyzer — %s ===", date_str)
    log.info("Universe: %d tickers", len(tickers))

    # ── Step 1: Sector rotation context ──────────────────────────────────────
    log.info("Computing sector momentum...")
    sector_returns = compute_sector_returns()
    if sector_returns:
        update_sector_momentum(sector_returns)
        best_sector  = max(sector_returns, key=sector_returns.get)
        worst_sector = min(sector_returns, key=sector_returns.get)
        log.info(
            "Sector rotation — Leading: %s (%.1f%%)  Lagging: %s (%.1f%%)",
            best_sector,  sector_returns[best_sector],
            worst_sector, sector_returns[worst_sector],
        )

    # ── Step 2: Batch quotes (single network call) ────────────────────────────
    log.info("Fetching batch quotes...")
    quotes = fetch_batch_quotes(tickers)
    log.info("Got quotes for %d/%d tickers", len(quotes), len(tickers))

    # ── Step 3: News (single batch fetch) ────────────────────────────────────
    log.info("Fetching news...")
    news_map = fetch_rss_news(tickers)
    from config import NEWS_API_KEY
    if NEWS_API_KEY:
        api_news = fetch_newsapi(tickers)
        for t, articles in api_news.items():
            news_map.setdefault(t, []).extend(articles)
    log.info("News fetched for universe")

    # ── Step 4: Per-ticker analysis (parallel) ────────────────────────────────
    log.info("Analysing %d tickers (parallel)...", len(tickers))
    scores: list[StockScore] = []
    max_workers = min(20, len(tickers))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(analyse_one, ticker, news_map, quotes): ticker
            for ticker in tickers
        }
        done = 0
        for future in as_completed(futures):
            result = future.result()
            if result:
                scores.append(result)
            done += 1
            if done % 20 == 0:
                log.info("  Progress: %d/%d done", done, len(tickers))

    log.info("Analysis complete: %d stocks scored", len(scores))

    # ── Step 5: Rank and select top N ────────────────────────────────────────
    ranked = rank_stocks(scores, TOP_N)

    # ── Step 6: AI narrative ──────────────────────────────────────────────────
    ai_narrative = ""
    if use_ai:
        log.info("Generating AI narrative (Claude)...")
        ai_narrative = generate_ai_narrative(ranked)

    # ── Step 7: Output ────────────────────────────────────────────────────────
    print_terminal_report(ranked, date_str)

    report_path = save_markdown_report(ranked, date_str, ai_narrative)
    log.info("Report saved → %s", report_path)

    # Print AI narrative to terminal
    if ai_narrative and not ai_narrative.startswith("_Set"):
        from rich.console import Console
        from rich.panel import Panel
        Console().print(Panel(
            ai_narrative,
            title="[bold cyan]AI Market Narrative[/bold cyan]",
            border_style="cyan",
        ))

    return ranked


def main():
    parser = argparse.ArgumentParser(description="US Stock Daily Analyzer")
    parser.add_argument(
        "--schedule", action="store_true",
        help="Run on a daily schedule at 07:30 ET"
    )
    parser.add_argument(
        "--tickers", nargs="+",
        help="Analyse specific tickers instead of the full universe"
    )
    parser.add_argument(
        "--no-ai", action="store_true",
        help="Skip Claude AI narrative (faster, no API key needed)"
    )
    parser.add_argument(
        "--top", type=int, default=TOP_N,
        help=f"Number of top stocks to show (default: {TOP_N})"
    )
    args = parser.parse_args()

    tickers  = args.tickers if args.tickers else ALL_STOCKS
    use_ai   = not args.no_ai

    if args.schedule:
        from rich.console import Console
        Console().print(
            "[bold cyan]Scheduled mode: running daily at 07:30 ET[/bold cyan]\n"
            "Press Ctrl+C to stop."
        )
        schedule.every().day.at("07:30").do(run_analysis, tickers=tickers, use_ai=use_ai)
        # Also run immediately
        run_analysis(tickers, use_ai)
        while True:
            schedule.run_pending()
            time.sleep(30)
    else:
        run_analysis(tickers, use_ai)


if __name__ == "__main__":
    main()
