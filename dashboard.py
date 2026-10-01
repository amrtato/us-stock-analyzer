"""
Daily Market Analyzer — Streamlit Dashboard

Covers two instruments with deliberately different treatments:

  US STOCKS — scored and ranked on four weighted pillars.
  GOLD      — risk and volatility only. No score, no ranking, no direction.
              Walk-forward testing over 2018-2026 (daily, hourly and minute
              data, including ICT setups and cross-asset drivers) produced no
              tradeable directional signal, so the gold tab does not emit one.

Tabs:
  1. Daily Rankings   — ranked top-N from the selected stock universe
  2. Stock Search     — analyse any ticker or company name on demand
  3. Sector Browser   — compare top 20 stocks in each market sector
  4. Watch List       — per-browser saved tickers with live prices
  5. Gold Risk        — session volatility, trade-cost check, monitored hypothesis
  6. Short Candidates — bearish screen for short selling (a separate model, NOT an
                        inversion of the long score — see analyzers/bearish.py)

Launch:
    streamlit run dashboard.py
"""
import os
import re
import sys
import json
import time
import uuid
import warnings
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
from datetime import datetime
sys.path.insert(0, ".")

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

from config import (
    ALL_STOCKS, DJIA_STOCKS, NASDAQ_TOP, SP500_TOP,
    SECTOR_STOCKS,
)
from data.fetcher import fetch_price_history, fetch_fundamentals, fetch_batch_quotes, prefetch_price_histories
from data.news_fetcher import fetch_rss_news, aggregate_ticker_news
from analyzers.technical import analyse_technical
from analyzers.fundamental import analyse_fundamental
from analyzers.sentiment import analyse_sentiment
from analyzers.macro import analyse_macro, update_sector_momentum, MACRO_FLAGS
from scoring.scorer import build_stock_score, rank_stocks, StockScore
from shorts.tab import show_short_tab

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Daily Market Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ─────────────────────────────────────────────────────────────────────
# Single source of truth in styles.py so dashboard.py and gold_only.py stay in
# sync — a missing class renders unstyled HTML with no error to catch it.
from styles import inject as _inject_css
_inject_css()

# ── Constants ──────────────────────────────────────────────────────────────────
SECTOR_ETFS = {
    "Technology": "XLK", "Financials": "XLF", "Healthcare": "XLV",
    "Consumer": "XLY",   "Industrials": "XLI", "Energy": "XLE",
    "Communication": "XLC", "Materials": "XLB", "Utilities": "XLU",
}

SECTOR_ICONS = {
    "Technology": "💻", "Consumer": "🛍️",  "Financials": "🏦",
    "Healthcare": "💊", "Industrials": "⚙️", "Energy": "⛽",
    "Communication": "📡", "Materials": "🪨", "Utilities": "⚡",
    "Real Estate": "🏢",
}

# ── Animated loading screen ────────────────────────────────────────────────────
LOADING_MSGS = [
    "🤖 Teaching AI to read balance sheets so you don't have to...",
    "📊 Crunching 847 financial ratios (only 846 of them matter)...",
    "☕ Markets never sleep, but our analysts needed a coffee break...",
    "🎲 Consulting the Magic 8-Ball... just kidding, it's actual math...",
    "🦉 Warren Buffett would hold — we're figuring out if you should too...",
    "📉 Teaching robots why panic-selling is always the wrong move...",
    "🔮 Crystal ball buffering... please stand by for technicals...",
    "💰 Computing your future gains... assuming you bought the dip...",
    "🚀 To the moon? Let's check the RSI and MACD first...",
    "📰 Reading every financial headline (yes, even the clickbait ones)...",
    "🧮 P/E ratios, EPS growth, RSI — not alphabet soup, just alpha...",
    "🎯 Finding optimal entry points unlike your last three trades...",
    "🐂 Politely asking 100 stocks whether they're bullish or bearish...",
    "⏰ Patience is a virtue. So is a 3:1 risk-reward ratio...",
    "🔬 Running 7 indicators because 6 felt overconfident...",
    "🎰 This is NOT gambling — we have a Sharpe ratio AND spreadsheets!",
]

FUN_FACTS = [
    "💡 The NYSE was founded in 1792 under a buttonwood tree in Manhattan.",
    "💡 Warren Buffett bought his first stock at age 11 for $38.",
    "💡 The word 'salary' comes from Roman soldiers being paid in salt.",
    "💡 The world's first stock exchange opened in Amsterdam in 1602.",
    "💡 Black Monday (Oct 19, 1987) — Dow fell 22.6% in a single session.",
    "💡 Apple became the first US company to reach a $1 trillion market cap in 2018.",
    "💡 The S&P 500 has returned ~10% per year on average since 1957.",
    "💡 The term 'blue chip' comes from poker — blue chips hold the highest value.",
    "💡 NASDAQ stands for National Association of Securities Dealers Automated Quotations.",
    "💡 There are ~58,000 publicly traded companies worldwide right now.",
]

_LOADING_TEMPLATE = """\
<!DOCTYPE html><html><head>
<style>
body{margin:0;background:#0e1117;color:#fafafa;font-family:'Segoe UI',Helvetica,sans-serif;overflow:hidden}
.wrap{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:490px;padding:16px 20px}
.tape{width:100%;overflow:hidden;background:#13131f;border-radius:6px;padding:7px 0;margin-bottom:22px;border:1px solid #252540}
.tape-inner{display:inline-block;white-space:nowrap;color:#00aaff;font-size:.82em;font-weight:700;animation:scroll 24s linear infinite}
@keyframes scroll{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}
.icon{font-size:3.6em;margin:4px 0;animation:pulse 1.6s ease-in-out infinite}
@keyframes pulse{0%,100%{transform:scale(1)}50%{transform:scale(1.13)}}
.title{font-size:1.45em;font-weight:700;color:#e8e8e8;margin:8px 0 3px;text-align:center}
.sub{color:#666;font-size:.84em;margin-bottom:14px;text-align:center;letter-spacing:.3px}
.pbar-wrap{width:82%;max-width:520px;background:#1a1a2a;border-radius:999px;height:9px;overflow:hidden;margin:10px 0 4px;border:1px solid #2a2a3a}
.pbar{height:100%;background:linear-gradient(90deg,#005fcc,#00aaff,#00ff88);border-radius:999px;width:2%;animation:grow %%SECS%%s ease-out forwards}
@keyframes grow{from{width:2%}to{width:93%}}
.cdown-row{color:#888;font-size:.86em;margin:6px 0 16px}
.cnum{color:#00aaff;font-weight:700;font-size:1.1em;font-variant-numeric:tabular-nums}
.joke-box{background:#13182a;border:1px solid #212840;border-radius:10px;padding:13px 22px;margin:6px 0;text-align:center;max-width:580px;min-height:48px;display:flex;align-items:center;justify-content:center}
.joke{color:#c8c8d8;font-size:.91em;font-style:italic;transition:opacity .4s}
.fact{color:#444;font-size:.77em;margin-top:11px;max-width:500px;text-align:center;transition:opacity .4s}
.dots span{animation:blink 1.4s infinite;color:#00aaff}
.dots span:nth-child(2){animation-delay:.2s}
.dots span:nth-child(3){animation-delay:.4s}
@keyframes blink{0%,80%,100%{opacity:0}40%{opacity:1}}
</style></head><body>
<div class="wrap">
  <div class="tape"><div class="tape-inner">%%TAPE%%</div></div>
  <div class="icon">📊</div>
  <div class="title">Analysing %%N%% stocks<span class="dots"><span>.</span><span>.</span><span>.</span></span></div>
  <div class="sub">Technical &nbsp;·&nbsp; Fundamental &nbsp;·&nbsp; Sentiment &nbsp;·&nbsp; Macro</div>
  <div class="pbar-wrap"><div class="pbar" id="pb"></div></div>
  <div class="cdown-row">Ready in approximately <span class="cnum" id="cd">%%SECS%%</span> seconds</div>
  <div class="joke-box"><div class="joke" id="jk">🤔 Warming up the analysis engine...</div></div>
  <div class="fact" id="ft"></div>
</div>
<script>
var J=%%JOKES%%,F=%%FACTS%%,ji=0,fi=0,s=%%SECS%%;
document.getElementById('jk').textContent=J[0];
document.getElementById('ft').textContent=F[0];
setInterval(function(){s=Math.max(0,s-1);document.getElementById('cd').textContent=s;},1000);
setInterval(function(){ji=(ji+1)%J.length;var e=document.getElementById('jk');e.style.opacity=0;setTimeout(function(){e.textContent=J[ji];e.style.opacity=1;},400);},4500);
setTimeout(function(){setInterval(function(){fi=(fi+1)%F.length;var e=document.getElementById('ft');e.style.opacity=0;setTimeout(function(){e.textContent=F[fi];e.style.opacity=1;},400);},7000);},3500);
</script>
</body></html>"""


def _make_loading_html(n_tickers: int, est_secs: int) -> str:
    """Return a self-contained animated HTML loading page."""
    tape_syms = [
        "AAPL","NVDA","MSFT","GOOGL","AMZN","TSLA","META","NFLX",
        "AMD","JPM","V","JNJ","WMT","DIS","BKNG","GS","MA",
        "BAC","COST","AVGO","ORCL","CRM","ADBE","INTC","HON",
    ]
    tape_inner = " &nbsp;·&nbsp; ".join(
        '<span style="color:#00ff88">' + t + '</span>' for t in tape_syms * 2
    )
    return (
        _LOADING_TEMPLATE
        .replace("%%JOKES%%", json.dumps(LOADING_MSGS))
        .replace("%%FACTS%%", json.dumps(FUN_FACTS))
        .replace("%%SECS%%",  str(est_secs))
        .replace("%%N%%",     str(n_tickers))
        .replace("%%TAPE%%",  tape_inner)
    )


def _needs_loading(cache_key: str) -> bool:
    """True when this ticker set hasn't been loaded in the current browser session."""
    return cache_key not in st.session_state.get("_loaded_keys", set())


def _mark_loaded(cache_key: str) -> None:
    if "_loaded_keys" not in st.session_state:
        st.session_state["_loaded_keys"] = set()
    st.session_state["_loaded_keys"].add(cache_key)


# ── Watch List file persistence ────────────────────────────────────────────────
# Stored in the user's home directory so it survives Azure zip-deploys.
# In Azure App Service the /home directory is on Azure Files and IS persistent.
# A single shared path here meant every visitor to the deployed app read and
# wrote ONE watch list — open the site in a second browser and you saw (and
# could delete) someone else's stocks. Streamlit has no user auth, so scope the
# file by a per-browser id carried in the URL query string: it survives reloads
# for that browser, and no two visitors collide.
_WATCHLIST_DIR = os.path.join(os.path.expanduser("~"), ".us_stock_watchlists")


def _watchlist_id() -> str:
    """Stable per-browser id, persisted in the URL so reloads keep the list."""
    wid = st.query_params.get("wl")
    if not wid or not re.fullmatch(r"[A-Za-z0-9]{8,32}", str(wid)):
        wid = uuid.uuid4().hex[:16]
        st.query_params["wl"] = wid
    return str(wid)


def _watchlist_path() -> str:
    os.makedirs(_WATCHLIST_DIR, exist_ok=True)
    return os.path.join(_WATCHLIST_DIR, f"{_watchlist_id()}.json")


def _load_watchlist_file() -> list:
    """Return this browser's persisted tickers. [] if absent/invalid."""
    try:
        with open(_watchlist_path()) as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(t).upper().strip() for t in data if str(t).strip()]
    except Exception:
        pass
    return []


def _save_watchlist_file(tickers: list) -> None:
    """Write this browser's tickers to disk (survives page refreshes)."""
    try:
        with open(_watchlist_path(), "w") as f:
            json.dump([str(t).upper() for t in tickers], f)
    except Exception as exc:
        logging.warning("watchlist save failed: %s", exc)


# ── Cached data fetchers ───────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def get_sector_returns():
    etfs   = list(SECTOR_ETFS.values())
    quotes = fetch_batch_quotes(etfs)
    return {
        sec: quotes[etf]["change_pct"]
        for sec, etf in SECTOR_ETFS.items() if etf in quotes
    }


@st.cache_data(ttl=1800, show_spinner=False)
def get_batch_quotes(tickers: tuple):
    return fetch_batch_quotes(list(tickers))


@st.cache_data(ttl=1800, show_spinner=False)
def get_news(tickers: tuple):
    return fetch_rss_news(list(tickers))


@st.cache_data(ttl=3600, show_spinner=False)
def get_price_history(ticker: str, period: str = "6mo"):
    return fetch_price_history(ticker, period=period)


@st.cache_data(ttl=3600, show_spinner=False)
def get_fundamentals(ticker: str):
    return fetch_fundamentals(ticker)


@st.cache_data(ttl=60, show_spinner=False)
def get_watchlist_quotes(tickers: tuple) -> dict:
    """Batch quotes with a 60-second cache — keeps watch-list prices fresh."""
    if not tickers:
        return {}
    return fetch_batch_quotes(list(tickers))


def analyse_ticker(ticker, news_map, quotes):
    try:
        df        = get_price_history(ticker)
        fund_data = get_fundamentals(ticker)
        quote     = quotes.get(ticker, {})
        price     = quote.get("price", float(df["Close"].iloc[-1]) if not df.empty else 0)

        tech_r  = analyse_technical(ticker, df)
        fund_r  = analyse_fundamental(ticker, fund_data, price)
        sent_r  = analyse_sentiment(ticker, aggregate_ticker_news(ticker, news_map), quote)
        macro_r = analyse_macro(ticker, fund_data, quote)

        return build_stock_score(ticker, tech_r, fund_r, sent_r, macro_r, fund_data, quote)
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def run_full_analysis(tickers: tuple, macro_flags: tuple) -> list:
    flags_dict = dict(macro_flags)
    for k, v in flags_dict.items():
        MACRO_FLAGS[k] = v

    sector_returns = get_sector_returns()
    update_sector_momentum(sector_returns)

    quotes   = get_batch_quotes(tickers)
    news_map = get_news(tickers)

    # ── Batch-prefetch all 6-month price histories in ONE yfinance call ─────────
    # This turns N serialised HTTP calls inside the ThreadPool into instant cache
    # hits, cutting price-fetch time from ~10 s (100 tickers) to ~4 s.
    prefetch_price_histories(list(tickers))

    scores = []
    with ThreadPoolExecutor(max_workers=15) as pool:
        futures = {pool.submit(analyse_ticker, t, news_map, quotes): t for t in tickers}
        for future in as_completed(futures):
            result = future.result()
            if result:
                scores.append(result)
    return scores


@st.cache_data(ttl=3600, show_spinner=False)
def resolve_ticker(query: str):
    """Resolve a company name or ticker symbol → (ticker, full_name) or (None, None)."""
    q     = query.strip()
    upper = q.upper()

    # 1. Try as a direct ticker symbol
    try:
        t     = yf.Ticker(upper)
        fi    = t.fast_info
        price = fi.last_price
        if price and float(price) > 0:
            # basic_info has no long_name (it is a fast_info shim), so this
            # always fell through to the bare symbol — which is why searching
            # "Palantir" returned "Palantir Technologies Inc." but searching
            # "TSLA" returned "TSLA — TSLA". .info carries the real name.
            name = upper
            try:
                info = t.info or {}
                name = info.get("longName") or info.get("shortName") or upper
            except Exception:
                pass
            return upper, name
    except Exception:
        pass

    # 2. Fall back to yfinance full-text search
    try:
        results = yf.Search(q, news_count=0, max_results=10).quotes
        if results:
            for r in results:
                if r.get("quoteType", "") in ("EQUITY", "ETF"):
                    sym  = r.get("symbol", "")
                    name = r.get("longname") or r.get("shortname") or sym
                    return sym, name
            # Fallback: first result regardless of type
            r    = results[0]
            sym  = r.get("symbol", "")
            name = r.get("longname") or r.get("shortname") or sym
            return sym, name
    except Exception:
        pass

    return None, None


@st.cache_data(ttl=1800, show_spinner=False)
def analyse_single_ticker(ticker: str, macro_flags: tuple):
    """Run the full multi-factor analysis pipeline on a single ticker."""
    flags_dict = dict(macro_flags)
    for k, v in flags_dict.items():
        MACRO_FLAGS[k] = v
    sector_returns = get_sector_returns()
    update_sector_momentum(sector_returns)
    quotes   = get_batch_quotes((ticker,))
    news_map = get_news((ticker,))
    return analyse_ticker(ticker, news_map, quotes)


# ── Visual helpers ─────────────────────────────────────────────────────────────

def grade_color(grade: str) -> str:
    if "A" in grade: return "#00ff88"
    if "B" in grade: return "#ffd700"
    if "C" in grade: return "#ff8c00"
    return "#ff4444"


def score_color(score: float) -> str:
    if score >= 72: return "#00ff88"
    if score >= 60: return "#ffd700"
    if score >= 48: return "#ff8c00"
    return "#ff4444"


def tf_badges_html(stock: StockScore) -> str:
    style = {"SCALP": "tf-scalp", "DAY": "tf-day", "SWING": "tf-swing",
             "INVEST": "tf-invest", "WATCH": "tf-watch"}
    icons = {"SCALP": "⚡", "DAY": "📅", "SWING": "📈", "INVEST": "💼", "WATCH": "👁"}
    parts = []
    for t in stock.timeframes:
        css  = style.get(t, "tf-watch")
        icon = icons.get(t, "")
        parts.append(f'<span class="tf-badge {css}">{icon} {t}</span>')
    return " ".join(parts)


def colour_score(val):
    try:
        v = float(val)
        if v >= 72: return "color: #00ff88; font-weight: bold"
        if v >= 60: return "color: #ffd700; font-weight: bold"
        if v >= 48: return "color: #ff8c00"
        return "color: #ff4444"
    except Exception:
        return ""


def colour_chg(val):
    try:
        v = float(str(val).replace("%", ""))
        return "color: #00ff88" if v >= 0 else "color: #ff4444"
    except Exception:
        return ""


def show_status_banner(stock: StockScore):
    """Prominent coloured banner: status pill + risk badge + score + quick metrics."""
    sc = stock.status_color
    rc = stock.risk_color
    st.markdown(
        f"""
        <div class="status-banner" style="background: linear-gradient(135deg, {sc}18, #1e1e2e);">
            <div>
                <span class="status-pill" style="background:{sc}33; color:{sc}; border:1.5px solid {sc};">
                    {stock.status_emoji} {stock.market_status.upper()}
                </span>
            </div>
            <div>
                <span class="status-pill" style="background:{rc}22; color:{rc}; border:1.5px solid {rc};">
                    {stock.risk_emoji} {stock.risk_rating.upper()} RISK
                </span>
            </div>
            <div style="flex:1; text-align:right; color:#aaa; font-size:0.9em;">
                Score&nbsp;<b style="color:{sc}; font-size:1.3em;">{stock.total_score:.1f}</b>
                &nbsp;/&nbsp;100&nbsp;&nbsp;|&nbsp;&nbsp;
                Grade&nbsp;<b style="color:{sc};">{stock.grade}</b>
                &nbsp;&nbsp;|&nbsp;&nbsp;
                R:R&nbsp;<b style="color:#00aaff;">{stock.risk_reward:.1f}x</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_trade_advice(stock: StockScore):
    """Render the AI-generated trade advice in a styled box."""
    border_color = stock.status_color
    # Replace newlines with HTML breaks for the markdown box
    advice_html = stock.trade_advice.replace("\n\n", "<br><br>").replace("\n", "<br>")
    # Highlight key numbers in the advice
    st.markdown(
        f'<div class="advice-box" style="border-left-color:{border_color};">'
        f"📋 <b>Trade Advice</b><br><br>{advice_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Chart builders ─────────────────────────────────────────────────────────────

def build_price_chart(ticker: str, df: pd.DataFrame) -> go.Figure:
    from analyzers.technical import _ema, compute_bollinger

    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()

    ema9,  ema21  = _ema(close, 9),  _ema(close, 21)
    ema50, ema200 = _ema(close, 50), _ema(close, 200)
    bb_up, bb_mid, bb_lo = compute_bollinger(close)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"].squeeze(), high=high, low=low, close=close,
        name=ticker,
        increasing_line_color="#00ff88", decreasing_line_color="#ff4444",
    ))
    fig.add_trace(go.Scatter(x=df.index, y=bb_up, name="BB Upper",
        line=dict(color="rgba(100,150,255,0.4)", width=1, dash="dot"), showlegend=False))
    fig.add_trace(go.Scatter(x=df.index, y=bb_lo, name="BB Lower",
        fill="tonexty", fillcolor="rgba(100,150,255,0.05)",
        line=dict(color="rgba(100,150,255,0.4)", width=1, dash="dot"), showlegend=False))
    fig.add_trace(go.Scatter(x=df.index, y=ema9,   name="EMA9",   line=dict(color="#ff9500", width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=ema21,  name="EMA21",  line=dict(color="#ffcc00", width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=ema50,  name="EMA50",  line=dict(color="#00aaff", width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=ema200, name="EMA200", line=dict(color="#ff4444", width=1.5)))
    fig.update_layout(
        template="plotly_dark", height=420,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=dict(text=f"{ticker} — 6-Month Price Chart", font=dict(size=14)),
    )
    return fig


def build_rsi_chart(ticker: str, df: pd.DataFrame) -> go.Figure:
    from analyzers.technical import compute_rsi
    rsi = compute_rsi(df["Close"].squeeze())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=rsi, name="RSI", line=dict(color="#00aaff", width=2)))
    fig.add_hline(y=70, line_dash="dash", line_color="#ff4444", annotation_text="70")
    fig.add_hline(y=30, line_dash="dash", line_color="#00ff88", annotation_text="30")
    fig.add_hline(y=50, line_dash="dot",  line_color="#888",   line_width=1)
    fig.update_layout(
        template="plotly_dark", height=160,
        margin=dict(l=0, r=0, t=20, b=0),
        yaxis=dict(range=[0, 100], title="RSI"),
        showlegend=False,
        title=dict(text="RSI (14)", font=dict(size=12)),
    )
    return fig


def build_score_bar(scores: list, height: int = 300) -> go.Figure:
    top = sorted(scores, key=lambda s: s.total_score, reverse=True)[:30]
    fig = go.Figure(go.Bar(
        x=[s.ticker for s in top],
        y=[s.total_score for s in top],
        marker_color=[score_color(s.total_score) for s in top],
        text=[f"{s.total_score:.1f}" for s in top],
        textposition="outside",
        customdata=[[s.grade, s.sector, ", ".join(s.timeframes)] for s in top],
        hovertemplate=(
            "<b>%{x}</b><br>Score: %{y:.1f}<br>"
            "Grade: %{customdata[0]}<br>Sector: %{customdata[1]}<br>"
            "TF: %{customdata[2]}<extra></extra>"
        ),
    ))
    fig.update_layout(
        template="plotly_dark", height=height,
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(range=[0, 105], title="Score"),
        xaxis=dict(tickangle=-45),
        showlegend=False,
    )
    return fig


def build_radar(stock: StockScore) -> go.Figure:
    cats = ["Technical", "Fundamental", "Sentiment", "Macro"]
    vals = [stock.tech_score, stock.fund_score, stock.sent_score, stock.macro_score]
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=cats + [cats[0]],
        fill="toself", fillcolor="rgba(0,170,255,0.2)",
        line=dict(color="#00aaff", width=2),
    ))
    fig.update_layout(
        template="plotly_dark", height=260,
        margin=dict(l=20, r=20, t=20, b=20),
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
    )
    return fig


def build_sector_chart(sector_returns: dict) -> go.Figure:
    sectors = list(sector_returns.keys())
    returns = [sector_returns[s] for s in sectors]
    fig = go.Figure(go.Bar(
        x=[f"{r:+.1f}%" for r in returns], y=sectors,
        orientation="h",
        marker_color=["#00ff88" if r >= 0 else "#ff4444" for r in returns],
        text=[f"{r:+.1f}%" for r in returns], textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark", height=280,
        margin=dict(l=0, r=40, t=10, b=0),
        xaxis=dict(title="1-Day Return %"), showlegend=False,
    )
    return fig


# ── Shared stock-detail view (reused by all four tabs) ────────────────────────

def show_stock_analysis(stock: StockScore, context: str = "main"):
    """Render the complete analysis panel for any StockScore.

    `context` must be unique per call-site (e.g. "daily", "search", "sector",
    "watchlist") so that Streamlit can assign distinct element IDs to each
    plotly_chart even when multiple tabs render simultaneously.
    """
    # ── Status banner + trade advice (always shown first) ─────────────────────
    show_status_banner(stock)
    if stock.trade_advice:
        show_trade_advice(stock)
    st.divider()

    pfx    = f"{context}_{stock.ticker}"   # unique prefix for chart keys
    d_left, d_right = st.columns([2, 1])

    with d_left:
        df_chart = get_price_history(stock.ticker, "6mo")
        if not df_chart.empty:
            st.plotly_chart(
                build_price_chart(stock.ticker, df_chart),
                key=f"price_{pfx}",
            )
            st.plotly_chart(
                build_rsi_chart(stock.ticker, df_chart),
                key=f"rsi_{pfx}",
            )

    with d_right:
        st.plotly_chart(
            build_radar(stock),
            key=f"radar_{pfx}",
        )
        st.markdown(f"### {stock.ticker} — {stock.grade}")
        ind = stock.indicators
        m   = st.columns(2)
        m[0].metric("Price", f"${stock.price:,.2f}", f"{stock.change_pct:+.2f}%")
        m[1].metric("Score", f"{stock.total_score:.1f}", stock.grade)
        # Beta is genuinely absent for some tickers — show "—" rather than the
        # old hardcoded 1.00, which looked like a real market-neutral reading.
        beta_txt = f"{stock.beta:.2f}" if stock.beta is not None else "—"
        st.markdown(
            f"**Sector:** {stock.sector}  |  "
            f"**Beta:** {beta_txt}  |  "
            f"**ATR:** ${stock.atr:.2f} ({stock.atr_pct:.1f}%)"
        )
        st.markdown("**Timeframes:**")
        st.markdown(tf_badges_html(stock), unsafe_allow_html=True)

    st.markdown("#### 📐 Trade Levels")
    entry = stock.entry or stock.price
    # Entry zone highlight
    if stock.entry_zone_low and stock.entry_zone_high:
        st.markdown(
            f'<div style="background:#00aaff18; border:1px solid #00aaff44; border-radius:8px; '
            f'padding:8px 16px; margin-bottom:8px; font-size:0.9em;">'
            f'🎯 <b>Optimal Entry Zone:</b>&nbsp; '
            f'<span style="color:#00aaff; font-size:1.1em; font-weight:bold;">'
            f'${stock.entry_zone_low:,.2f} – ${stock.entry_zone_high:,.2f}</span>'
            f'&nbsp;&nbsp;(reference entry: ${entry:,.2f})'
            f'</div>',
            unsafe_allow_html=True,
        )
    l1, l2, l3, l4, l5 = st.columns(5)
    l1.metric("Entry",       f"${entry:,.2f}")
    l2.metric("Stop Loss",   f"${stock.stop_loss:,.2f}",
              f"-{(entry - stock.stop_loss) / entry * 100:.1f}%" if entry else "")
    l3.metric("Target 1",    f"${stock.target_1:,.2f}",
              f"+{(stock.target_1 - entry) / entry * 100:.1f}%" if entry else "")
    l4.metric("Target 2",    f"${stock.target_2:,.2f}",
              f"+{(stock.target_2 - entry) / entry * 100:.1f}%" if entry else "")
    l5.metric("Risk:Reward", f"{stock.risk_reward:.1f}x")

    st.markdown("#### 📡 Technical Indicators")
    t1, t2, t3, t4, t5, t6, t7, t8 = st.columns(8)
    t1.metric("RSI (14)",  ind.get("rsi",      "—"))
    t2.metric("ADX",       ind.get("adx",      "—"))
    t3.metric("Stoch K",   ind.get("stoch_k",  "—"))
    t4.metric("Vol Ratio", f"{ind.get('vol_ratio', '—')}x")
    t5.metric("EMA 50",    f"${ind.get('ema50',   '—')}")
    t6.metric("EMA 200",   f"${ind.get('ema200',  '—')}")
    t7.metric("ROC 5d",    f"{ind.get('roc5',     '—')}%")
    t8.metric("ROC 20d",   f"{ind.get('roc20',    '—')}%")

    fm = stock.fund_metrics
    if fm:
        st.markdown("#### 💼 Fundamentals")
        f1, f2, f3, f4, f5, f6 = st.columns(6)
        f1.metric("P/E Ratio",  fm.get("pe",             "—"))
        f2.metric("PEG Ratio",  fm.get("peg",            "—"))
        f3.metric("ROE",        fm.get("roe",            "—"))
        f4.metric("Rev Growth", fm.get("revenue_growth", "—"))
        f5.metric("EPS Growth", fm.get("earnings_growth","—"))
        f6.metric("Mkt Cap",    f"${fm.get('market_cap_b', '—')}B")

    if stock.signals:
        st.markdown("#### 🚦 Signals & Catalysts")
        sig_cols = st.columns(2)
        for i, sig in enumerate(stock.signals[:10]):
            sig_cols[i % 2].markdown(f"• {sig}")


# ── Helper: styled scores dataframe ───────────────────────────────────────────

def make_scores_df(scores: list, extra_cols: bool = False) -> pd.DataFrame:
    ranked = sorted(scores, key=lambda s: s.total_score, reverse=True)
    rows   = []
    for i, s in enumerate(ranked, 1):
        row = {
            "Rank":    i,
            "Ticker":  s.ticker,
            "Name":    s.name[:22],
            "Price":   f"${s.price:,.2f}",
            "Chg %":   f"{s.change_pct:+.2f}%",
            "Status":  f"{s.status_emoji} {s.market_status}",
            "Risk":    f"{s.risk_emoji} {s.risk_rating}",
            "Score":   round(s.total_score, 1),
            "Grade":   s.grade,
            # These were floats rounded to 0dp, which pandas still renders as
            # "59.000000" in the grid. They are whole numbers — store them as
            # ints so they display as "59".
            "Tech":    int(round(s.tech_score)),
            "Fund":    int(round(s.fund_score)),
            "Sent":    int(round(s.sent_score)),
            "Macro":   int(round(s.macro_score)),
            "TFs":     " | ".join(s.timeframes),
            "Entry":   f"${s.entry:,.2f}",
            "Entry Zone": (
                f"${s.entry_zone_low:,.2f}–${s.entry_zone_high:,.2f}"
                if s.entry_zone_low else "—"
            ),
            "Stop":    f"${s.stop_loss:,.2f}",
            "T1":      f"${s.target_1:,.2f}",
            "T2":      f"${s.target_2:,.2f}",
            "R:R":     f"{s.risk_reward:.1f}x",
        }
        if extra_cols:
            row["Sector"] = s.sector
        rows.append(row)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Daily Rankings
# ═══════════════════════════════════════════════════════════════════════════════

def show_daily_tab(tickers, top_n, macro_flags):
    tickers_t  = tuple(tickers)
    cache_key  = f"daily_{hash(tickers_t)}_{hash(macro_flags)}"
    n          = len(tickers)
    est_secs   = max(10, min(32, n // 5 + 10))   # ~10s for 30, ~30s for 100

    if _needs_loading(cache_key):
        loading_slot = st.empty()
        with loading_slot:
            components.html(_make_loading_html(n, est_secs), height=520, scrolling=False)
        scores = run_full_analysis(tickers_t, macro_flags)
        loading_slot.empty()
        _mark_loaded(cache_key)
    else:
        scores = run_full_analysis(tickers_t, macro_flags)

    if not scores:
        st.error("No data returned. Check your internet connection.")
        return

    ranked        = rank_stocks(scores, top_n)
    sector_returns = get_sector_returns()
    best_sec  = max(sector_returns, key=sector_returns.get) if sector_returns else "—"
    worst_sec = min(sector_returns, key=sector_returns.get) if sector_returns else "—"
    avg_score = sum(s.total_score for s in scores) / len(scores)
    top_stock = ranked[0]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Stocks Analysed",    len(scores))
    k2.metric("Universe Avg Score", f"{avg_score:.1f}")
    # "#1 Pick" claimed a recommendation. Walk-forward testing over 5 years found
    # the composite has no stable rank correlation with forward returns, so the
    # honest label is what this actually is: the highest-scoring name today.
    k3.metric(f"Highest Score: {top_stock.ticker}",
              f"{top_stock.total_score:.1f} ({top_stock.grade})",
              f"{top_stock.change_pct:+.1f}%")
    k4.metric("Leading Sector", best_sec, f"{sector_returns.get(best_sec, 0):+.1f}%")
    k5.metric("Lagging Sector", worst_sec, f"{sector_returns.get(worst_sec, 0):+.1f}%")

    st.divider()

    left, right = st.columns([3, 1])
    with left:
        st.subheader("📊 All Stocks by Score (Top 30)")
        st.plotly_chart(build_score_bar(scores), key="score_bar_daily")
    with right:
        if sector_returns:
            st.subheader("🔄 Sector Rotation (1d)")
            st.plotly_chart(build_sector_chart(sector_returns), key="sector_rot_daily")

    st.subheader(f"📊 Top {top_n} by Composite Score")
    st.caption(
        "A screening rank, not a forecast. Walk-forward testing (101 stocks, "
        "2022–2026, non-overlapping windows) found no reliable relationship "
        "between this score and forward returns — use it to shortlist names "
        "for your own analysis, not as a buy list."
    )
    df_top = make_scores_df(ranked[:top_n], extra_cols=True)
    styled = (
        df_top.style
        .map(colour_score, subset=["Score", "Tech", "Fund", "Sent", "Macro"])
        .map(colour_chg,   subset=["Chg %"])
        .format({"Score": "{:.1f}"})
    )
    st.dataframe(styled, width='stretch', hide_index=True, height=420)

    st.divider()
    st.subheader("🔍 Stock Deep Dive")
    choices = [
        f"{i}. {s.ticker} — {s.name[:30]} (Score: {s.total_score:.1f})"
        for i, s in enumerate(ranked, 1)
    ]
    pick  = st.selectbox("Select a stock for full analysis:", choices, key="daily_pick")
    stock = ranked[choices.index(pick)]
    show_stock_analysis(stock, context="daily")

    st.divider()
    st.caption("⚠️ For informational purposes only. Not financial advice. Always apply your own due diligence.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Stock Search
# ═══════════════════════════════════════════════════════════════════════════════

def show_search_tab(macro_flags):
    st.markdown("## 🔍 Stock Search & Deep Analysis")
    st.caption(
        "Search **any** US or international stock by ticker symbol or company name — "
        "not limited to the pre-built universe."
    )

    col_in, col_btn = st.columns([5, 1])
    with col_in:
        query = st.text_input(
            "",
            placeholder="e.g.  AAPL   or   Apple Inc   or   Palantir   or   BRK-B",
            label_visibility="collapsed",
            key="search_query",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("🔍 Analyse", type="primary", width='stretch', key="search_btn")

    if not query:
        st.info(
            "💡 **Tips:**\n"
            "- Ticker symbols work best: `NVDA`, `TSLA`, `BRK-B`, `MELI`\n"
            "- Company names also work: `Nvidia`, `Tesla`, `Berkshire`\n"
            "- International stocks: `BABA`, `TSM`, `ASML`\n"
            "- ETFs: `SPY`, `QQQ`, `GLD`"
        )
        return

    with st.spinner(f"Looking up **'{query}'**…"):
        ticker, name = resolve_ticker(query)

    if not ticker:
        st.error(
            f"❌ No stock found for **'{query}'**. "
            "Try the ticker symbol directly (e.g. `AAPL`) or check the spelling."
        )
        return

    st.success(f"✅ Found: **{ticker}** — {name}")
    st.divider()

    with st.spinner(f"Running full multi-factor analysis for **{ticker}**…"):
        stock = analyse_single_ticker(ticker, macro_flags)

    if not stock:
        st.warning("⚠️ Analysis unavailable — insufficient market data for this ticker.")
        return

    show_stock_analysis(stock, context="search")
    st.divider()
    st.caption("⚠️ For informational purposes only. Not financial advice.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Sector Browser
# ═══════════════════════════════════════════════════════════════════════════════

def show_sector_browser(macro_flags):
    st.markdown("## 🏭 Sector Browser — 20 Stocks per Sector")
    st.caption(
        "Instantly compare and rank the 20 leading stocks in any market sector. "
        "Scores cached for 30 min — click **Refresh Data** in the sidebar to force reload."
    )

    sectors       = list(SECTOR_STOCKS.keys())
    sector_labels = [f"{SECTOR_ICONS.get(s, '📊')} {s}" for s in sectors]

    sel_label = st.radio(
        "Sector:",
        sector_labels,
        horizontal=True,
        label_visibility="collapsed",
        key="sector_radio",
    )
    selected_sector = sectors[sector_labels.index(sel_label)]
    sector_tickers  = tuple(SECTOR_STOCKS[selected_sector])
    icon            = SECTOR_ICONS.get(selected_sector, "📊")

    # ── Sector ETF quick stat ─────────────────────────────────────────────────
    etf = SECTOR_ETFS.get(selected_sector)
    if etf:
        etf_q   = get_batch_quotes((etf,))
        etf_d   = etf_q.get(etf, {})
        etf_chg = etf_d.get("change_pct", 0)
        etf_px  = etf_d.get("price", 0)
        st.metric(
            f"{icon} {selected_sector} ETF ({etf})",
            f"${etf_px:,.2f}",
            f"{etf_chg:+.2f}%",
            delta_color="normal" if etf_chg >= 0 else "inverse",
        )

    st.divider()

    # ── Run analysis ──────────────────────────────────────────────────────────
    sec_cache_key = f"sector_{hash(sector_tickers)}_{hash(macro_flags)}"
    sec_n         = len(sector_tickers)
    sec_est       = max(8, min(18, sec_n // 3 + 6))   # ~13s for 20 stocks

    if _needs_loading(sec_cache_key):
        sec_slot = st.empty()
        with sec_slot:
            components.html(
                _make_loading_html(sec_n, sec_est), height=520, scrolling=False
            )
        sector_scores = run_full_analysis(sector_tickers, macro_flags)
        sec_slot.empty()
        _mark_loaded(sec_cache_key)
    else:
        sector_scores = run_full_analysis(sector_tickers, macro_flags)

    if not sector_scores:
        st.error("No data returned. Check your connection.")
        return

    ranked_sector = sorted(sector_scores, key=lambda s: s.total_score, reverse=True)

    # ── KPI row ───────────────────────────────────────────────────────────────
    avg_score = sum(s.total_score for s in sector_scores) / len(sector_scores)
    top_s     = ranked_sector[0]
    bot_s     = ranked_sector[-1]
    bullish   = sum(1 for s in sector_scores if s.total_score >= 60)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Stocks Analysed",     len(sector_scores))
    k2.metric("Sector Avg Score",    f"{avg_score:.1f}")
    # "Best"/"Worst" read as verdicts; these are simply the score extremes.
    k3.metric(f"Highest: {top_s.ticker}",
              f"{top_s.total_score:.1f} ({top_s.grade})", f"{top_s.change_pct:+.1f}%")
    k4.metric(f"Lowest:  {bot_s.ticker}",
              f"{bot_s.total_score:.1f} ({bot_s.grade})", f"{bot_s.change_pct:+.1f}%")
    k5.metric("Bullish (≥ 60)",      f"{bullish} / {len(sector_scores)}")

    st.divider()

    # ── Score comparison bar chart ────────────────────────────────────────────
    st.subheader(f"📊 {icon} {selected_sector} — Score Comparison")
    st.plotly_chart(
        build_score_bar(sector_scores, height=320),
        key=f"score_bar_sector_{selected_sector}",
    )

    st.divider()

    # ── Full comparison table ─────────────────────────────────────────────────
    st.subheader(f"📋 {selected_sector} — All {len(ranked_sector)} Stocks Ranked by Score")

    df_sector = make_scores_df(ranked_sector)
    styled = (
        df_sector.style
        .map(colour_score, subset=["Score", "Tech", "Fund", "Sent", "Macro"])
        .map(colour_chg,   subset=["Chg %"])
        .format({"Score": "{:.1f}"})
    )
    st.dataframe(styled, width='stretch', hide_index=True, height=500)

    st.divider()

    # ── Deep-dive within sector ───────────────────────────────────────────────
    st.subheader(f"🔍 {selected_sector} — Deep Dive")
    choices = [
        f"{i}. {s.ticker} — {s.name[:30]} (Score: {s.total_score:.1f})"
        for i, s in enumerate(ranked_sector, 1)
    ]
    pick  = st.selectbox(
        "Pick a stock for the full analysis:",
        choices,
        key=f"sector_pick_{selected_sector}",
    )
    show_stock_analysis(ranked_sector[choices.index(pick)], context=f"sector_{selected_sector}")
    st.divider()
    st.caption("⚠️ For informational purposes only. Not financial advice.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Watch List
# ═══════════════════════════════════════════════════════════════════════════════

def show_watchlist_tab(macro_flags: tuple):
    st.markdown("## 📋 My Watch List")
    st.caption(
        "Add any stock by ticker or company name — live prices update every 60 s, "
        "full analysis refreshes every 30 min."
    )

    # ── Initialise session state from persistent file ──────────────────────────
    if "watchlist" not in st.session_state:
        st.session_state["watchlist"] = _load_watchlist_file()
    watchlist: list = st.session_state["watchlist"]

    # ── Add stock ──────────────────────────────────────────────────────────────
    # A form with clear_on_submit is the only legal way to empty this box:
    # assigning st.session_state["wl_input"] after the widget exists raises
    # StreamlitAPIException. Without the clear, the previous ticker stayed in
    # the field and the next entry was appended to it ("NVDA" + "Berkshire"
    # → "NVDABerkshire"), so every user's second add failed.
    with st.form("wl_add_form", clear_on_submit=True, border=False):
        add_col, btn_col = st.columns([5, 1])
        with add_col:
            new_q = st.text_input(
                "Add to watch list",
                placeholder="Ticker or company name — e.g. NVDA  ·  Palantir  ·  BRK-B",
                label_visibility="collapsed",
                key="wl_input",
            )
        with btn_col:
            add_clicked = st.form_submit_button(
                "➕ Add", type="primary", width='stretch'
            )

    if add_clicked and new_q.strip():
        with st.spinner(f"Looking up '{new_q.strip()}'…"):
            sym, name = resolve_ticker(new_q.strip())
        if sym:
            if sym not in watchlist:
                watchlist.append(sym)
                st.session_state["watchlist"] = watchlist
                _save_watchlist_file(watchlist)
                st.success(f"✅ Added **{sym}** — {name}")
                st.rerun()
            else:
                st.info(f"**{sym}** is already in your watch list.")
        else:
            st.error(
                f"❌ Could not find **'{new_q}'**. "
                "Try the exact ticker symbol (e.g. `NVDA`)."
            )

    # ── Empty-state placeholder ────────────────────────────────────────────────
    if not watchlist:
        st.markdown(
            '<div style="text-align:center;padding:70px 20px;color:#555;">'
            '<div style="font-size:3.2em;margin-bottom:14px;">📭</div>'
            '<div style="font-size:1.1em;color:#888;">Your watch list is empty.</div>'
            '<div style="font-size:.88em;margin-top:8px;">Add stocks above to start tracking them.</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Management bar: header · remove selector · refresh/remove button ───────
    st.divider()
    hdr_col, rm_col, act_col = st.columns([2, 4, 1])
    with hdr_col:
        st.markdown(
            f"<div style='padding-top:8px;font-weight:600;font-size:1.05em;'>"
            f"👁&nbsp; Watching {len(watchlist)} "
            f"stock{'s' if len(watchlist) != 1 else ''}</div>",
            unsafe_allow_html=True,
        )
    with rm_col:
        to_remove = st.multiselect(
            "remove",
            watchlist,
            placeholder="Select tickers to remove…",
            label_visibility="collapsed",
            key="wl_remove",
        )
    with act_col:
        btn_label = "🗑️ Remove" if to_remove else "🔄 Refresh"
        act_btn   = st.button(btn_label, width='stretch', key="wl_action")

    if act_btn and to_remove:
        for t in to_remove:
            if t in watchlist:
                watchlist.remove(t)
        st.session_state["watchlist"] = watchlist
        _save_watchlist_file(watchlist)
        # Drop cached "loaded" marker for this watchlist so analysis re-runs
        if "_loaded_keys" in st.session_state:
            st.session_state["_loaded_keys"] = {
                k for k in st.session_state["_loaded_keys"]
                if not k.startswith("watchlist_")
            }
        st.rerun()
    elif act_btn and not to_remove:
        # Refresh: invalidate watchlist analysis marker so loading screen re-fires
        if "_loaded_keys" in st.session_state:
            st.session_state["_loaded_keys"] = {
                k for k in st.session_state["_loaded_keys"]
                if not k.startswith("watchlist_")
            }
        st.rerun()

    if not watchlist:
        return

    # ── Live quotes (60-s cache) ───────────────────────────────────────────────
    live_q = get_watchlist_quotes(tuple(watchlist))

    # ── Full analysis — with animated loading screen on first/refresh hit ──────
    wl_t         = tuple(watchlist)
    wl_cache_key = f"watchlist_{hash(wl_t)}_{hash(macro_flags)}"
    wl_n         = len(wl_t)
    wl_est       = max(8, min(20, wl_n // 3 + 6))

    if _needs_loading(wl_cache_key):
        wl_slot = st.empty()
        with wl_slot:
            components.html(_make_loading_html(wl_n, wl_est), height=520, scrolling=False)
        wl_scores = run_full_analysis(wl_t, macro_flags)
        wl_slot.empty()
        _mark_loaded(wl_cache_key)
    else:
        wl_scores = run_full_analysis(wl_t, macro_flags)

    if not wl_scores:
        st.warning(
            "⚠️ Analysis unavailable — check your connection and press **🔄 Refresh**."
        )
        return

    score_map = {s.ticker: s for s in wl_scores}

    # ── KPI strip ──────────────────────────────────────────────────────────────
    avg_sc  = sum(s.total_score for s in wl_scores) / len(wl_scores)
    bull_n  = sum(1 for s in wl_scores if "Bullish" in s.market_status)
    bear_n  = sum(1 for s in wl_scores if "Bearish" in s.market_status)
    top_s   = max(wl_scores, key=lambda s: s.total_score)
    top_chg = live_q.get(top_s.ticker, {}).get("change_pct", top_s.change_pct)

    best_ticker = max(
        watchlist,
        key=lambda t: live_q.get(t, {}).get("change_pct",
                                             score_map.get(t, top_s).change_pct),
    )
    best_chg = live_q.get(best_ticker, {}).get(
        "change_pct", score_map.get(best_ticker, top_s).change_pct
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Avg Score",              f"{avg_sc:.1f}")
    k2.metric("🐂 Bullish",              f"{bull_n} / {len(wl_scores)}")
    k3.metric("🐻 Bearish",              f"{bear_n} / {len(wl_scores)}")
    k4.metric(f"🏆 Best Score: {top_s.ticker}",
              f"{top_s.total_score:.1f} ({top_s.grade})", f"{top_chg:+.1f}%")
    k5.metric(f"📈 Today's Leader: {best_ticker}",
              f"{live_q.get(best_ticker,{}).get('price', score_map.get(best_ticker,top_s).price):,.2f}",
              f"{best_chg:+.2f}%")

    st.divider()

    # ── Live price cards ───────────────────────────────────────────────────────
    st.subheader("💰 Live Prices")
    cols_per_row = min(5, len(watchlist))
    chunks       = [
        watchlist[i: i + cols_per_row]
        for i in range(0, len(watchlist), cols_per_row)
    ]
    for chunk in chunks:
        card_cols = st.columns(cols_per_row)
        for ci, ticker in enumerate(chunk):
            q   = live_q.get(ticker, {})
            s   = score_map.get(ticker)
            px  = q.get("price",      s.price      if s else 0.0)
            ch  = q.get("change_pct", s.change_pct if s else 0.0)
            vol = q.get("volume",     0)
            with card_cols[ci]:
                st.metric(
                    label=f"{s.status_emoji + ' ' if s else ''}**{ticker}**",
                    value=f"${px:,.2f}",
                    delta=f"{ch:+.2f}%",
                    delta_color="normal" if ch >= 0 else "inverse",
                )
                if s:
                    sc  = s.total_score
                    clr = score_color(sc)
                    st.markdown(
                        f'<div style="height:5px;border-radius:3px;'
                        f'background:#1a1a2e;margin:-6px 0 4px;">'
                        f'<div style="width:{sc:.0f}%;height:100%;'
                        f'background:{clr};border-radius:3px;"></div></div>'
                        f'<div style="font-size:.74em;color:#777;text-align:center;">'
                        f'{s.risk_emoji}&nbsp;{s.risk_rating}&nbsp;·&nbsp;'
                        f'Score&nbsp;<b style="color:{clr};">{sc:.0f}</b></div>',
                        unsafe_allow_html=True,
                    )
                    if vol:
                        st.caption(f"Vol: {vol/1e6:.1f}M")

    st.caption("Prices update every 60 s · Analysis refreshes every 30 min")
    st.divider()

    # ── Score comparison chart ─────────────────────────────────────────────────
    st.subheader("📊 Score Comparison")
    st.plotly_chart(
        build_score_bar(wl_scores, height=max(240, wl_n * 24)),
        key="score_bar_watchlist",
    )
    st.divider()

    # ── Detailed table ─────────────────────────────────────────────────────────
    st.subheader("📋 Detailed Overview")
    rows = []
    for ticker in watchlist:
        q  = live_q.get(ticker, {})
        s  = score_map.get(ticker)
        px = q.get("price",      s.price      if s else 0.0)
        ch = q.get("change_pct", s.change_pct if s else 0.0)
        rows.append({
            "Ticker":     ticker,
            "Name":       s.name[:22] if s else ticker,
            "Price":      f"${px:,.2f}",
            "Chg %":      f"{ch:+.2f}%",
            "Status":     f"{s.status_emoji} {s.market_status}" if s else "—",
            "Risk":       f"{s.risk_emoji} {s.risk_rating}"     if s else "—",
            "Score":      round(s.total_score, 1)               if s else 0,
            "Grade":      s.grade                               if s else "—",
            "Tech":       round(s.tech_score,  0)               if s else 0,
            "Fund":       round(s.fund_score,  0)               if s else 0,
            "Entry":      f"${s.entry:,.2f}"                    if s else "—",
            "Entry Zone": (
                f"${s.entry_zone_low:,.2f}–${s.entry_zone_high:,.2f}"
                if s and s.entry_zone_low else "—"
            ),
            "Stop":       f"${s.stop_loss:,.2f}"                if s else "—",
            "T1":         f"${s.target_1:,.2f}"                 if s else "—",
            "T2":         f"${s.target_2:,.2f}"                 if s else "—",
            "R:R":        f"{s.risk_reward:.1f}x"               if s else "—",
        })

    df_wl  = pd.DataFrame(rows)
    styled = (
        df_wl.style
        .map(colour_score, subset=["Score", "Tech", "Fund"])
        .map(colour_chg,   subset=["Chg %"])
        .format({"Score": "{:.1f}"})
    )
    st.dataframe(styled, width='stretch', hide_index=True,
                 height=min(600, 60 + len(rows) * 38))
    st.divider()

    # ── Individual deep dive ───────────────────────────────────────────────────
    st.subheader("🔍 Deep Dive")
    valid = sorted(
        [t for t in watchlist if t in score_map],
        key=lambda t: score_map[t].total_score,
        reverse=True,
    )
    if valid:
        choices = [
            f"{score_map[t].ticker} — {score_map[t].name[:30]}  "
            f"(Score: {score_map[t].total_score:.1f})"
            for t in valid
        ]
        pick = st.selectbox("Select a stock for full analysis:", choices, key="wl_pick")
        sym  = pick.split(" — ")[0].strip()
        if sym in score_map:
            show_stock_analysis(score_map[sym], context="watchlist")

    st.divider()
    st.caption("⚠️ For informational purposes only. Not financial advice.")


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("⚙️ Settings")
        st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")

        universe_choice = st.selectbox(
            "Stock Universe (Daily Rankings tab)",
            ["Full Universe (~100)", "DJIA (30)", "NASDAQ Top (33)", "S&P500 Top (40)", "Custom"],
        )
        if universe_choice == "Full Universe (~100)":
            tickers = ALL_STOCKS
        elif universe_choice == "DJIA (30)":
            tickers = DJIA_STOCKS
        elif universe_choice == "NASDAQ Top (33)":
            tickers = NASDAQ_TOP
        elif universe_choice == "S&P500 Top (40)":
            tickers = SP500_TOP
        else:
            custom  = st.text_input("Tickers (comma-separated)", "AAPL,NVDA,MSFT,GOOGL,META,AMZN")
            tickers = [t.strip().upper() for t in custom.split(",") if t.strip()]

        top_n = st.slider("Top N stocks to show", 5, 20, 10)

        st.divider()
        st.subheader("📡 Macro Regime Flags")
        fed_hawkish     = st.toggle("Fed Hawkish (rate hike mode)", False)
        recession_risk  = st.toggle("Recession Risk",               False)
        strong_dollar   = st.toggle("Strong USD",                   False)
        earnings_season = st.toggle("Earnings Season",              True)
        geo_risk        = st.toggle("Geopolitical Risk",            False)

        macro_flags = (
            ("fed_hawkish",       fed_hawkish),
            ("recession_risk",    recession_risk),
            ("strong_dollar",     strong_dollar),
            ("earnings_season",   earnings_season),
            ("geopolitical_risk", geo_risk),
        )

        st.divider()
        if st.button("🔄 Refresh Data", width='stretch'):
            st.cache_data.clear()
            st.rerun()
        st.caption("Data cached 30 min · Fundamentals 4 hr")

    # ── Header ────────────────────────────────────────────────────────────────
    col_title, col_date = st.columns([3, 1])
    with col_title:
        st.title("📊 Daily Market Analyzer")
        # Two instruments, two treatments — stated plainly rather than implying
        # gold gets the same scoring the stocks do. It does not: eight years of
        # testing produced no tradeable directional signal for gold, so that tab
        # reports risk only.
        st.caption(
            "**US Stocks** · DJIA · NASDAQ-100 · S&P500 — multi-factor scoring "
            "(Technical 38% | Fundamental 30% | Sentiment 18% | Macro 14%)  \n"
            "**Short Candidates** - downtrend-continuation model, scored separately from the long score  \n"
            "**Gold** · XAUUSD — risk &amp; volatility, no directional signal"
        )
    with col_date:
        st.metric("Today", datetime.now().strftime("%b %d, %Y"))

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab6, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Daily Rankings",
        "📉 Short Candidates",
        "🔍 Stock Search",
        "🏭 Sector Browser",
        "📋 Watch List",
        "🥇 Gold Risk",
    ])

    # EXECUTION ORDER MATTERS, TAB ORDER DOES NOT.
    # st.tabs is not lazy: every `with tabN:` block runs server-side on each
    # rerun, and the browser merely shows/hides them. Streamlit streams elements
    # as the script executes, so whatever runs FIRST appears first — while the
    # visual position stays fixed by the st.tabs([...]) list above.
    #
    # Gold needs ~3s (two yfinance calls); the stock universe needs 3-4 minutes
    # cold. Rendering gold first means it is usable immediately instead of
    # waiting behind 101 tickers it has nothing to do with.
    with tab5:
        # Risk/volatility only — the gold research produced no tradeable
        # directional signal, so this tab deliberately emits none.
        from gold.tab import show_gold_tab
        show_gold_tab()

    with tab1:
        show_daily_tab(tickers, top_n, macro_flags)

    with tab6:
        # Runs after tab1 on purpose: both screens share the batch-quote and
        # fundamentals caches, so by the time this executes they are warm and
        # the short screen costs only its own 1y price history.
        show_short_tab(tickers, top_n)

    with tab2:
        show_search_tab(macro_flags)

    with tab3:
        show_sector_browser(macro_flags)

    with tab4:
        show_watchlist_tab(macro_flags)


if __name__ == "__main__":
    main()
