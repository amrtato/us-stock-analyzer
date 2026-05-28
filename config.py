"""
Central configuration: stock universe, weights, thresholds, and settings.
Works locally (reads .env) and on Streamlit Cloud (reads st.secrets).
"""
import os
from dotenv import load_dotenv

load_dotenv()   # no-op on Streamlit Cloud, works locally

def _secret(key: str, default: str = "") -> str:
    """Read from st.secrets first (Streamlit Cloud), fall back to env var."""
    try:
        import streamlit as st
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        return os.getenv(key, default)

# ── API Keys ──────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = _secret("ANTHROPIC_API_KEY")
NEWS_API_KEY      = _secret("NEWS_API_KEY")
ALPHA_VANTAGE_KEY = _secret("ALPHA_VANTAGE_KEY")

# ── Stock Universe ─────────────────────────────────────────────────────────────

DJIA_STOCKS = [
    "AAPL", "MSFT", "JPM", "V", "JNJ", "WMT", "PG", "UNH", "HD", "CVX",
    "MRK", "AMGN", "CAT", "BA", "GS", "MMM", "MCD", "AXP", "IBM", "TRV",
    "CRM", "HON", "INTC", "VZ", "DIS", "KO", "NKE", "DOW", "CSCO", "SHW",  # WBA delisted 2024
]

NASDAQ_TOP = [
    "NVDA", "META", "GOOGL", "AMZN", "TSLA", "AVGO", "NFLX", "COST",
    "AMD", "ADBE", "QCOM", "AMAT", "MRVL", "PANW", "SNPS", "CDNS", "LRCX",
    "KLAC", "ORCL", "CRWD", "ABNB", "DDOG", "ZS", "MELI", "BKNG",
    "REGN", "GILD", "VRTX", "TTD", "FAST", "PCAR", "PAYX",
]

SP500_TOP = [
    "BRK-B", "LLY", "SPGI", "ACN", "LIN", "RTX", "PLD", "LOW", "SBUX",
    "GE", "MDT", "BLK", "DE", "NOW", "SHW", "ZTS", "ISRG", "ELV", "CI",
    "NEE", "APD", "ECL", "MDLZ", "TMUS", "ADI", "MCHP", "HCA",
    "WFC", "BAC", "MS", "COF", "MA", "PYPL", "FIS", "FISV",
    "TGT", "SO", "DUK", "STZ", "USB",
]

# Deduplicated full universe
ALL_STOCKS = sorted(set(DJIA_STOCKS + NASDAQ_TOP + SP500_TOP))

# ── Sector Stocks for Browser (20 curated stocks per sector) ──────────────────
SECTOR_STOCKS = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "AMD",  "AVGO", "QCOM", "AMAT", "INTC", "ADBE", "CRM",
        "NOW",  "CRWD", "ORCL", "IBM",  "CSCO", "ADI",  "MRVL", "PANW", "TTD",  "KLAC",
    ],
    "Consumer": [
        "AMZN", "TSLA", "COST", "WMT",  "HD",   "MCD",  "SBUX", "NKE",  "TGT",  "NFLX",
        "BKNG", "ABNB", "MELI", "KO",   "PG",   "MDLZ", "LOW",  "DG",   "YUM",  "CMG",
    ],
    "Financials": [
        "JPM",  "GS",   "MS",   "BAC",  "WFC",  "USB",  "COF",  "AXP",  "V",    "MA",
        "PYPL", "BLK",  "SPGI", "TRV",  "AFL",  "C",    "BK",   "SCHW", "ICE",  "CME",
    ],
    "Healthcare": [
        "JNJ",  "UNH",  "LLY",  "MRK",  "AMGN", "GILD", "REGN", "VRTX", "ISRG", "MDT",
        "ZTS",  "ELV",  "CI",   "HCA",  "BMY",  "PFE",  "ABT",  "CVS",  "IDXX", "SYK",
    ],
    "Industrials": [
        "CAT",  "BA",   "HON",  "GE",   "RTX",  "DE",   "MMM",  "UPS",  "FDX",  "LMT",
        "NOC",  "GD",   "ITW",  "ETN",  "EMR",  "CSX",  "NSC",  "FAST", "PCAR", "PAYX",
    ],
    "Energy": [
        "CVX",  "XOM",  "COP",  "EOG",  "SLB",  "MPC",  "VLO",  "PSX",  "OXY",  "HAL",
        "BKR",  "HES",  "DVN",  "MRO",  "WMB",  "KMI",  "LNG",  "APA",  "CTRA", "MTDR",
    ],
    "Communication": [
        "GOOGL","META", "DIS",  "VZ",   "TMUS", "NFLX", "T",    "CMCSA","EA",   "TTWO",
        "WBD",  "SPOT", "PINS", "SNAP", "RBLX", "IPG",  "OMC",  "FOXA", "LYV",  "ZM",
    ],
    "Materials": [
        "LIN",  "APD",  "ECL",  "SHW",  "NEM",  "FCX",  "AA",   "VMC",  "MLM",  "DOW",
        "DD",   "PPG",  "NUE",  "STLD", "CF",   "MOS",  "ALB",  "IFF",  "EMN",  "PKG",
    ],
    "Utilities": [
        "NEE",  "SO",   "DUK",  "D",    "AEP",  "EXC",  "SRE",  "XEL",  "WEC",  "DTE",
        "AES",  "PPL",  "ETR",  "PNW",  "AWK",  "FE",   "LNT",  "NI",   "CMS",  "EVRG",
    ],
    "Real Estate": [
        "PLD",  "AMT",  "CCI",  "EQIX", "SPG",  "O",    "WELL", "DLR",  "PSA",  "EQR",
        "AVB",  "MAA",  "UDR",  "CPT",  "ESS",  "VTR",  "ARE",  "BXP",  "KIM",  "REG",
    ],
}

# Sector mapping for rotation analysis
SECTOR_MAP = {
    "Technology": ["AAPL", "MSFT", "NVDA", "AMD", "INTC", "QCOM", "AVGO", "AMAT", "LRCX",
                   "KLAC", "SNPS", "CDNS", "ADBE", "CRM", "NOW", "CRWD", "PANW", "DDOG",
                   "ZS", "ORCL", "IBM", "CSCO", "ADI", "MCHP", "MRVL", "TTD"],
    "Consumer": ["AMZN", "TSLA", "COST", "WMT", "HD", "LOW", "MCD", "SBUX", "NKE",
                 "TGT", "NFLX", "BKNG", "ABNB", "MELI", "ETSY", "KO", "PG", "MDLZ"],
    "Financials": ["JPM", "GS", "MS", "BAC", "WFC", "USB", "COF", "AXP", "V", "MA",
                   "PYPL", "FIS", "FISV", "BLK", "SPGI", "TRV", "AFL", "MET"],
    "Healthcare": ["JNJ", "UNH", "LLY", "MRK", "AMGN", "GILD", "REGN", "VRTX", "ISRG",
                   "MDT", "ZTS", "ELV", "CI", "HCA", "IDXX"],
    "Industrials": ["CAT", "BA", "HON", "GE", "RTX", "DE", "MMM", "FAST", "PCAR", "PAYX", "DOW"],
    "Energy": ["CVX", "XOM"],
    "Communication": ["GOOGL", "META", "DIS", "VZ", "TMUS"],
    "Materials": ["LIN", "APD", "ECL", "SHW"],
    "Utilities": ["NEE", "SO", "DUK"],
    "Real Estate": ["PLD"],
}

# ── Scoring Weights ────────────────────────────────────────────────────────────
WEIGHTS = {
    "technical":    0.38,
    "fundamental":  0.30,
    "sentiment":    0.18,
    "macro":        0.14,
}

# ── Technical Thresholds ───────────────────────────────────────────────────────
RSI_OVERSOLD   = 35
RSI_OVERBOUGHT = 65
VOLUME_SURGE   = 1.5      # 1.5× average = notable volume

# ── Output Settings ───────────────────────────────────────────────────────────
TOP_N           = 10      # stocks in daily report
REPORT_DIR      = "reports"
DATA_CACHE_MINS = 30      # cache yfinance data for N minutes

# ── AI Model ──────────────────────────────────────────────────────────────────
CLAUDE_MODEL = "claude-sonnet-4-6"

# ── News RSS Feeds ────────────────────────────────────────────────────────────
NEWS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline",
    "https://www.marketwatch.com/rss/marketpulse",
    "https://feeds.reuters.com/reuters/businessNews",
    "https://www.investing.com/rss/news_25.rss",   # US stocks
    "https://seekingalpha.com/feed.xml",
]
