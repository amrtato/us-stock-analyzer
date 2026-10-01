# Daily Market Analyzer

Covers two instruments, with deliberately different treatments.

**US Stocks** — multi-factor screener for DJIA, NASDAQ-100 and S&P500. Scores ~100
tickers across technical, fundamental, sentiment and macro dimensions and outputs a
ranked list with trade levels.

**Gold (XAUUSD)** — risk and volatility only. **No score, no ranking, no direction.**
Walk-forward testing over 2018–2026 across daily, hourly and minute data — including
ICT setups, cross-asset drivers and session/killzone structure — produced no tradeable
directional signal, so the gold tab reports what *did* survive: hourly volatility
seasonality and trade-cost arithmetic. See [Research findings](#research-findings).

> **A note on the stock ranking.** Walk-forward testing found no reliable relationship
> between the composite score and forward returns (rank IC ≈ 0 across 2022–2026 with
> non-overlapping windows). Treat it as a screening rank for your own analysis, not a
> buy list. `backtest.py`, `decompose.py`, `validate_signs.py` and `compare_models.py`
> reproduce that work.

## Quick Start

```bash
# 1. Install dependencies
pip3 install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY (required for AI narrative)
# NEWS_API_KEY and ALPHA_VANTAGE_KEY are optional

# 3. Run (full universe, with AI narrative)
python3 main.py

# 4. Run on specific tickers only (fast test)
python3 main.py --tickers AAPL NVDA MSFT GOOGL META AMZN

# 5. Run without AI narrative (no API key needed)
python3 main.py --no-ai

# 6. Schedule daily runs at 07:30 ET (pre-market)
python3 main.py --schedule
```

## Output

- **Terminal**: colour-coded ranked table + per-stock detail cards
- **Markdown**: `reports/report_YYYYMMDD.md` — copy into Notion, Obsidian, etc.

## Scoring Model

| Dimension | Weight | What it measures |
|-----------|--------|-----------------|
| Technical | 38% | RSI, MACD, Bollinger Bands, EMA alignment, ATR, ADX, Stochastic, Volume |
| Fundamental | 30% | P/E vs sector, PEG, ROE, revenue/earnings growth, debt/equity, analyst consensus |
| Sentiment | 18% | News headline sentiment, article volume, day's price action |
| Macro | 14% | Sector rotation, beta, 52-week position, market cap quality |

Scores: 0-100 per dimension → composite → A+/A/B+/B/C+/C/D grade

## Trade Levels (ATR-based)

| Level | Formula |
|-------|---------|
| Entry | Current price |
| Stop Loss | Entry − 1.5 × ATR(14) |
| Target 1 | Entry + 2.0 × ATR(14) |
| Target 2 | Entry + 3.5 × ATR(14) |
| Risk:Reward | (T1 − Entry) / (Entry − SL) |

## Timeframe Tags

| Tag | Criteria |
|-----|---------|
| ⚡ SCALP | RSI extreme + volume surge + ATR > 0.5% |
| 📅 DAY | Score > 62 + ADX > 20 + volume above avg |
| 📈 SWING | Score > 60 + EMA50 > EMA200 + positive 20d momentum |
| 💼 INVEST | Score > 58 + price above EMA200 + strong trend |

## Macro Flags

Edit `analyzers/macro.py → MACRO_FLAGS` to activate regime adjustments:
- `fed_hawkish`: penalises high-PE growth stocks, rewards dividend payers
- `recession_risk`: rewards defensives (Healthcare, Utilities, Consumer staples)
- `earnings_season`: adds a small opportunity premium across the board
- `geopolitical_risk`: boosts Energy and Industrials

## Gold Risk tab

Reports measured risk, not direction.

| Panel | Source |
|-------|--------|
| Hourly volatility profile | 47,161 MT5 H1 bars (8y), New York time |
| Typical move this hour | live price × that hour's σ multiple |
| Suggested stop | 1.5 × live ATR |
| Cost check | broker-measured spread (median $0.20, p99 $0.364) |
| Monitored hypothesis | 16:00 ET drift — paper test, **not tradeable** |

Constants live in `gold/research.py`. Re-derive them by re-running the analysis
scripts against a fresh MT5 export — `gold/mt5_export.py` needs a local Windows
terminal with **Tools → Options → Charts → Max bars in chart → Unlimited**, or M1
history silently truncates at 100,000 bars.

Live price on the deployed app comes from yfinance (`GC=F`) because the MetaTrader5
package cannot run on Azure Linux. Spread is not observable there, so the measured
broker median is used and labelled as such.

## Research findings

Every claim below is reproducible from the scripts in `gold/`.

| Tested | Result |
|--------|--------|
| ICT setups on daily bars (FVG, order blocks, liquidity sweeps, BOS) | Indistinguishable from random entry |
| 40 features × 120 directional tests, 5–20 day horizons | 2 significant, ~6 expected by chance |
| Asian-range breakout into London / New York | No edge over random |
| Killzone directional bias | No edge over random |
| M1 short-horizon predictability (4y, real spread) | Real (t = −5.01) but **0.44× spread** |
| Elliott Wave | Excluded — not objectively codifiable |
| **Hourly volatility seasonality** | **Replicated across two providers (2.6× range)** |
| **16:00 ET settlement drift** | **Survived holdout; 50.1% win over 8y — paper test only** |

The recurring lesson: several effects are statistically real and still smaller than
the spread. A backtest that assumes zero or flat costs will find "edges" that cannot
be traded. Costs are charged from broker-measured spread throughout.

## Data Sources

- **Price & Fundamentals**: Yahoo Finance via `yfinance` (free, no key needed)
- **News**: RSS feeds (Yahoo Finance, MarketWatch, Reuters) — always free
- **NewsAPI** (optional): richer per-stock news, 100 req/day free tier
- **AI Narrative**: Claude Sonnet via Anthropic API

## File Structure

```
US_Stock_Analyzer/
├── main.py                 # Orchestrator + CLI
├── config.py               # Stock universe, weights, settings
├── data/
│   ├── fetcher.py          # yfinance price + fundamentals
│   └── news_fetcher.py     # RSS + NewsAPI
├── analyzers/
│   ├── technical.py        # RSI, MACD, BB, EMA, ATR, ADX, Stoch
│   ├── fundamental.py      # Valuation, profitability, growth, balance sheet
│   ├── sentiment.py        # News sentiment + price action
│   └── macro.py            # Sector rotation, beta, 52w position, regime
├── scoring/
│   └── scorer.py           # Composite scorer + sector diversity filter
├── reporting/
│   └── generator.py        # Terminal (rich) + Markdown + Claude AI narrative
└── reports/                # Saved daily reports
```

## Disclaimer

For informational and educational purposes only. Not financial advice.
Always apply your own due diligence and risk management before trading.
