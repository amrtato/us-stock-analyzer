# US Stock Analyzer

Daily multi-factor stock screener for DJIA, NASDAQ-100, and S&P500.
Scores ~100 top US stocks across technical, fundamental, sentiment, and macro dimensions — outputs a ranked top-10 with trade levels every morning.

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
