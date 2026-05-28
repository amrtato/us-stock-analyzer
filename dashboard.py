"""
US Stock Analyzer — Streamlit Dashboard

Tabs:
  1. Daily Top Picks  — ranked top-N from the selected universe
  2. Stock Search     — analyse any ticker or company name on demand
  3. Sector Browser   — compare top 20 stocks in each market sector

Launch:
    streamlit run dashboard.py
"""
import sys
import warnings
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
from datetime import datetime
sys.path.insert(0, ".")

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

from config import (
    ALL_STOCKS, DJIA_STOCKS, NASDAQ_TOP, SP500_TOP,
    SECTOR_MAP, SECTOR_STOCKS,
)
from data.fetcher import fetch_price_history, fetch_fundamentals, fetch_batch_quotes
from data.news_fetcher import fetch_rss_news, aggregate_ticker_news
from analyzers.technical import analyse_technical
from analyzers.fundamental import analyse_fundamental
from analyzers.sentiment import analyse_sentiment
from analyzers.macro import analyse_macro, update_sector_momentum, MACRO_FLAGS
from scoring.scorer import build_stock_score, rank_stocks, StockScore

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="US Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e; border-radius: 10px;
        padding: 16px 20px; margin: 4px 0;
    }
    .grade-A { color: #00ff88; font-weight: bold; font-size: 1.2em; }
    .grade-B { color: #ffd700; font-weight: bold; font-size: 1.2em; }
    .grade-C { color: #ff8c00; font-weight: bold; font-size: 1.2em; }
    .grade-D { color: #ff4444; font-weight: bold; font-size: 1.2em; }
    .tf-badge {
        display: inline-block; padding: 2px 8px; border-radius: 4px;
        font-size: 0.75em; font-weight: bold; margin: 1px;
    }
    .tf-scalp  { background: #ff4444; color: white; }
    .tf-day    { background: #ff8c00; color: white; }
    .tf-swing  { background: #00aaff; color: white; }
    .tf-invest { background: #00cc66; color: white; }
    .tf-watch  { background: #555; color: #ccc; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem !important; }
</style>
""", unsafe_allow_html=True)

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
            name = upper
            try:
                bi = t.basic_info
                if bi:
                    name = getattr(bi, "long_name", upper) or upper
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


# ── Shared stock-detail view (reused by all three tabs) ────────────────────────

def show_stock_analysis(stock: StockScore):
    """Render the complete analysis panel for any StockScore."""
    d_left, d_right = st.columns([2, 1])

    with d_left:
        df_chart = get_price_history(stock.ticker, "6mo")
        if not df_chart.empty:
            st.plotly_chart(build_price_chart(stock.ticker, df_chart), width='stretch')
            st.plotly_chart(build_rsi_chart(stock.ticker, df_chart), width='stretch')

    with d_right:
        st.plotly_chart(build_radar(stock), width='stretch')
        st.markdown(f"### {stock.ticker} — {stock.grade}")
        ind = stock.indicators
        m   = st.columns(2)
        m[0].metric("Price", f"${stock.price:,.2f}", f"{stock.change_pct:+.2f}%")
        m[1].metric("Score", f"{stock.total_score:.1f}", stock.grade)
        st.markdown(
            f"**Sector:** {stock.sector}  |  "
            f"**Beta:** {stock.beta:.2f}  |  "
            f"**ATR:** ${stock.atr:.2f} ({stock.atr_pct:.1f}%)"
        )
        st.markdown("**Timeframes:**")
        st.markdown(tf_badges_html(stock), unsafe_allow_html=True)

    st.markdown("#### 📐 Trade Levels")
    l1, l2, l3, l4, l5 = st.columns(5)
    entry = stock.entry or stock.price
    l1.metric("Entry",       f"${entry:,.2f}")
    l2.metric("Stop Loss",   f"${stock.stop_loss:,.2f}",
              f"-{(entry - stock.stop_loss) / entry * 100:.1f}%" if entry else "")
    l3.metric("Target 1",    f"${stock.target_1:,.2f}",
              f"+{(stock.target_1 - entry) / entry * 100:.1f}%" if entry else "")
    l4.metric("Target 2",    f"${stock.target_2:,.2f}",
              f"+{(stock.target_2 - entry) / entry * 100:.1f}%" if entry else "")
    l5.metric("Risk:Reward", f"{stock.risk_reward}x")

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
            "Rank":   i,
            "Ticker": s.ticker,
            "Name":   s.name[:26],
            "Price":  f"${s.price:,.2f}",
            "Chg %":  f"{s.change_pct:+.2f}%",
            "Score":  round(s.total_score, 1),
            "Grade":  s.grade,
            "Tech":   round(s.tech_score,  0),
            "Fund":   round(s.fund_score,  0),
            "Sent":   round(s.sent_score,  0),
            "Macro":  round(s.macro_score, 0),
            "TFs":    " | ".join(s.timeframes),
            "Entry":  f"${s.entry:,.2f}",
            "Stop":   f"${s.stop_loss:,.2f}",
            "T1":     f"${s.target_1:,.2f}",
            "T2":     f"${s.target_2:,.2f}",
            "R:R":    f"{s.risk_reward}x",
        }
        if extra_cols:
            row["Sector"] = s.sector
        rows.append(row)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Daily Top Picks
# ═══════════════════════════════════════════════════════════════════════════════

def show_daily_tab(tickers, top_n, macro_flags):
    with st.spinner(f"Analysing {len(tickers)} stocks… (cached after first run)"):
        scores = run_full_analysis(tuple(tickers), macro_flags)

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
    k3.metric(f"#1 Pick: {top_stock.ticker}",
              f"{top_stock.total_score:.1f} ({top_stock.grade})",
              f"{top_stock.change_pct:+.1f}%")
    k4.metric("Leading Sector", best_sec, f"{sector_returns.get(best_sec, 0):+.1f}%")
    k5.metric("Lagging Sector", worst_sec, f"{sector_returns.get(worst_sec, 0):+.1f}%")

    st.divider()

    left, right = st.columns([3, 1])
    with left:
        st.subheader("📊 All Stocks by Score (Top 30)")
        st.plotly_chart(build_score_bar(scores), width='stretch')
    with right:
        if sector_returns:
            st.subheader("🔄 Sector Rotation (1d)")
            st.plotly_chart(build_sector_chart(sector_returns), width='stretch')

    st.subheader(f"🏆 Top {top_n} Trading Opportunities")
    df_top = make_scores_df(ranked[:top_n], extra_cols=True)
    # re-add Rank and extra columns present in old code
    styled = (
        df_top.style
        .map(colour_score, subset=["Score", "Tech", "Fund", "Sent", "Macro"])
        .map(colour_chg,   subset=["Chg %"])
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
    show_stock_analysis(stock)

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
        st.button("🔍 Analyse", type="primary", use_container_width=True, key="search_btn")

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

    show_stock_analysis(stock)
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
    with st.spinner(
        f"Analysing {len(sector_tickers)} {selected_sector} stocks… "
        "(cached — fast after first load)"
    ):
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
    k3.metric(f"Best:  {top_s.ticker}",
              f"{top_s.total_score:.1f} ({top_s.grade})", f"{top_s.change_pct:+.1f}%")
    k4.metric(f"Worst: {bot_s.ticker}",
              f"{bot_s.total_score:.1f} ({bot_s.grade})", f"{bot_s.change_pct:+.1f}%")
    k5.metric("Bullish (≥ 60)",      f"{bullish} / {len(sector_scores)}")

    st.divider()

    # ── Score comparison bar chart ────────────────────────────────────────────
    st.subheader(f"📊 {icon} {selected_sector} — Score Comparison")
    st.plotly_chart(build_score_bar(sector_scores, height=320), width='stretch')

    st.divider()

    # ── Full comparison table ─────────────────────────────────────────────────
    st.subheader(f"📋 {selected_sector} — All {len(ranked_sector)} Stocks Ranked by Score")

    df_sector = make_scores_df(ranked_sector)
    styled = (
        df_sector.style
        .map(colour_score, subset=["Score", "Tech", "Fund", "Sent", "Macro"])
        .map(colour_chg,   subset=["Chg %"])
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
    show_stock_analysis(ranked_sector[choices.index(pick)])
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
            "Stock Universe (Daily Top Picks tab)",
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
        st.title("📈 US Stock Daily Analyzer")
        st.caption(
            "DJIA · NASDAQ-100 · S&P500 — "
            "Multi-factor scoring: Technical 38% | Fundamental 30% | Sentiment 18% | Macro 14%"
        )
    with col_date:
        st.metric("Today", datetime.now().strftime("%b %d, %Y"))

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📈 Daily Top Picks",
        "🔍 Stock Search",
        "🏭 Sector Browser",
    ])

    with tab1:
        show_daily_tab(tickers, top_n, macro_flags)

    with tab2:
        show_search_tab(macro_flags)

    with tab3:
        show_sector_browser(macro_flags)


if __name__ == "__main__":
    main()
