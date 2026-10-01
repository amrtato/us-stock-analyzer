"""
Short Candidates tab — bearish screen for short-selling.

Mirrors the Daily Rankings tab in layout and theme, but the numbers underneath
are a different model (``analysers.bearish``), not an inversion of the long
score. See that module's docstring for why inverting was rejected.

The evidence line rendered at the top of this tab is filled from
``EVIDENCE`` below, which is set from ``backtest_short.py`` output. If the
model is ever retuned, re-run that script and update this constant — a screen
that quotes stale validation numbers is worse than one that quotes none.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analyzers.bearish import analyse_bearish
from data.fetcher import (
    fetch_batch_quotes,
    fetch_fundamentals,
    fetch_price_history,
    prefetch_price_histories,
)
from scoring.short_scorer import ShortScore, build_short_score, rank_shorts

BENCH = "SPY"

# Set from backtest_short.py + gate_check_short.py (run 2026-10-01).
# These are the measured numbers, not estimates. Do not soften them.
EVIDENCE = {
    "headline": (
        "⚠️ **This screen is not a validated signal.** Walk-forward testing "
        "(101 stocks, 2022–2026, non-overlapping 20-day windows) found shorting "
        "the top decile of this score lost **−1.81% per 20 days** — and shorting "
        "a *randomly chosen* stock lost less (−1.41%). The ranking carried no "
        "information: market-neutral edge was **−0.06% (p=0.93)**. Use this to "
        "find weak charts worth your own analysis, not as a short list."
    ),
    "detail": """
**Test setup** — 101 tickers, 2022-09-29 → 2026-08-27, point-in-time slicing
(each date scored using only bars available then), 20-day forward returns,
**non-overlapping** windows (n=50), 0.08%/trade cost for borrow + spread.

**The drift is the whole story.** The universe returned **+1.76% per 20 days**
and was positive on **70%** of windows. A short pays that. Every model tested
lost money outright, including the random control — which is what losing to the
drift looks like, not what a bad model looks like.

| model | outright | p | market-neutral | p | rank IC |
|---|---|---|---|---|---|
| **bear score** | −1.81% | 0.090 | −0.06% | 0.931 | +0.015 |
| inverted long score | −2.01% | 0.031 | −0.26% | 0.654 | +0.015 |
| trend component only | −2.10% | 0.013 | −0.35% | 0.459 | +0.009 |
| **random control** | **−1.41%** | 0.034 | **+0.34%** | 0.312 | −0.016 |

**Read the random row before anything else.** It beat the model on both
measures. The "significant" p-values in the outright column are the drift
showing up in every row — the control has them too.

**The rank IC has the wrong sign.** A working short model would show *negative*
IC (higher bear score → lower forward return). It measured **+0.015**: the
most bearish-looking charts went on to do marginally *better* than average.

**The gates are the one hopeful signal, and it is not significant.** Filtering
to names that pass the borrow/squeeze/relative-strength gates flips IC to
−0.030 (right sign) and market-neutral to **+0.49% with a 60% hit rate** —
but **p=0.60**, on 50 windows. That is indistinguishable from chance. It is
flagged here as a hypothesis to monitor, not a result.

**Honest summary:** the screen identifies genuinely weak charts. It has not
been shown that weak charts keep falling.
""",
    "tested": "2026-10-01 · backtest_short.py · gate_check_short.py",
}


# ── Data plumbing (cached; mirrors the main dashboard's TTLs) ─────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _history(ticker: str, period: str = "1y") -> pd.DataFrame:
    return fetch_price_history(ticker, period)


@st.cache_data(ttl=3600, show_spinner=False)
def _fundamentals(ticker: str) -> dict:
    return fetch_fundamentals(ticker)


@st.cache_data(ttl=1800, show_spinner=False)
def _quotes(tickers: tuple) -> dict:
    return fetch_batch_quotes(list(tickers))


@st.cache_data(ttl=1800, show_spinner=False)
def _bench_close(period: str = "1y") -> pd.Series | None:
    """SPY closes for the relative-weakness component.

    Returned as None on failure rather than a fabricated series: the analyser
    abstains at a neutral 50 when the benchmark is missing, which is the
    correct degradation. Silently substituting zeros would invent
    outperformance for every name in the universe.
    """
    try:
        df = fetch_price_history(BENCH, period)
        if df is None or df.empty:
            return None
        return df["Close"].squeeze()
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def run_short_analysis(tickers: tuple) -> list:
    prefetch_price_histories(list(tickers), period="1y")
    bench = _bench_close()
    quotes = _quotes(tickers)

    out = []
    for t in tickers:
        try:
            df = _history(t)
            if df is None or df.empty:
                continue
            result = analyse_bearish(t, df, bench)
            fund = _fundamentals(t)
            out.append(build_short_score(t, result, fund, quotes.get(t, {})))
        except Exception:
            continue
    return out


# ── Rendering helpers ────────────────────────────────────────────────────────
def _shorts_df(shorts: list) -> pd.DataFrame:
    rows = []
    for i, s in enumerate(shorts, 1):
        ind = s.indicators
        rel = ind.get("rel_strength")
        rows.append({
            "Rank": i,
            "Ticker": s.ticker,
            "Name": s.name[:22],
            "Price": f"${s.price:,.2f}",
            "Chg %": f"{s.change_pct:+.2f}%",
            "Setup": f"{s.setup_emoji} {s.setup_quality}",
            "Squeeze": f"{s.squeeze_emoji} {s.squeeze_risk}",
            "Bear": round(s.bear_score, 1),
            "Grade": s.grade,
            "vs SPY": f"{rel:+.1f}pp" if rel is not None else "—",
            "RSI": ind.get("rsi", "—"),
            "ADX": ind.get("adx", "—"),
            "Trend": int(round(s.subscores.get("trend", 50))),
            "Mom": int(round(s.subscores.get("mom", 50))),
            "Entry": f"${s.entry:,.2f}",
            "Stop": f"${s.stop_loss:,.2f}",
            "T1": f"${s.target_1:,.2f}",
            "T2": f"${s.target_2:,.2f}",
            "R:R": f"{s.risk_reward:.1f}x",
            "Sector": s.sector,
        })
    return pd.DataFrame(rows)


def _colour_bear(val):
    """Higher bear score = redder. Deliberately the opposite ramp to the long
    table, so a glance at the colour cannot be misread across the two tabs."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    if v >= 70: return "background-color: #ff444455; color: #fff; font-weight: bold;"
    if v >= 60: return "background-color: #ff8c6633; color: #ffb399;"
    if v >= 50: return "color: #ffd700;"
    return "color: #888;"


def _colour_chg(val):
    try:
        v = float(str(val).replace("%", "").replace("+", ""))
    except (TypeError, ValueError):
        return ""
    return "color: #00ff88;" if v > 0 else ("color: #ff4444;" if v < 0 else "")


def _short_banner(s: ShortScore) -> None:
    cc, qc = s.setup_color, s.squeeze_color
    st.markdown(
        f"""
        <div class="status-banner" style="background: linear-gradient(135deg, {cc}18, #1e1e2e);">
            <div>
                <span class="status-pill" style="background:{cc}33; color:{cc}; border:1.5px solid {cc};">
                    {s.setup_emoji} {s.setup_quality.upper()} BEAR SETUP
                </span>
            </div>
            <div>
                <span class="status-pill" style="background:{qc}22; color:{qc}; border:1.5px solid {qc};">
                    {s.squeeze_emoji} {s.squeeze_risk.upper()} SQUEEZE RISK
                </span>
            </div>
            <div style="flex:1; text-align:right; color:#aaa; font-size:0.9em;">
                Bear&nbsp;<b style="color:{cc}; font-size:1.3em;">{s.bear_score:.1f}</b>
                &nbsp;/&nbsp;100&nbsp;&nbsp;|&nbsp;&nbsp;
                Grade&nbsp;<b style="color:{cc};">{s.grade}</b>
                &nbsp;&nbsp;|&nbsp;&nbsp;
                R:R&nbsp;<b style="color:#00aaff;">{s.risk_reward:.1f}x</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _short_chart(ticker: str, df: pd.DataFrame, s: ShortScore) -> go.Figure:
    """Candles with the short levels drawn the right way up: stop ABOVE, targets BELOW."""
    fig = go.Figure()
    d = df.tail(120)
    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
        name=ticker, increasing_line_color="#00cc66", decreasing_line_color="#ff4444",
    ))
    for lbl, val, colour in (
        ("Stop", s.stop_loss, "#ff4444"),
        ("Entry", s.entry, "#00aaff"),
        ("T1", s.target_1, "#00cc66"),
        ("T2", s.target_2, "#00ff88"),
    ):
        if val:
            fig.add_hline(y=val, line_dash="dash", line_color=colour, line_width=1.4,
                          annotation_text=f"{lbl} ${val:,.2f}",
                          annotation_position="right",
                          annotation_font_color=colour)
    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font_color="#ffffff", xaxis_rangeslider_visible=False,
        showlegend=False, title=f"{ticker} — short levels",
    )
    fig.update_xaxes(gridcolor="#1e1e2e")
    fig.update_yaxes(gridcolor="#1e1e2e")
    return fig


def _subscore_radar(s: ShortScore) -> go.Figure:
    labels = {"trend": "Trend<br>breakdown", "rel": "Relative<br>weakness",
              "mom": "Down<br>momentum", "macd": "MACD", "dist": "Distribution",
              "adx": "Trend<br>strength", "rej": "Rally<br>rejection"}
    keys = list(labels)
    vals = [s.subscores.get(k, 50) for k in keys]
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=[labels[k] for k in keys] + [labels[keys[0]]],
        # Plotly rejects 8-digit hex (#rrggbbaa) for fillcolor even though CSS
        # accepts it and the rest of this app uses it — rgba() is the form it takes.
        fill="toself", line_color="#ff4444", fillcolor="rgba(255,68,68,0.20)",
    ))
    fig.update_layout(
        height=330, margin=dict(l=40, r=40, t=30, b=30),
        paper_bgcolor="#0e1117", font_color="#ffffff", showlegend=False,
        polar=dict(bgcolor="#141822",
                   radialaxis=dict(visible=True, range=[0, 100], gridcolor="#1e1e2e")),
        title="Where the bearish score comes from",
    )
    return fig


# ── Main tab ─────────────────────────────────────────────────────────────────
def show_short_tab(tickers, top_n: int = 10) -> None:
    st.markdown("## 📉 Short Candidates — Bearish Screen")

    if EVIDENCE.get("headline"):
        st.warning(EVIDENCE["headline"])
        if EVIDENCE.get("detail"):
            with st.expander("What the backtest actually found"):
                st.markdown(EVIDENCE["detail"])

    shorts = run_short_analysis(tuple(tickers))
    if not shorts:
        st.error("No data returned. Check your internet connection.")
        return

    ranked = rank_shorts(shorts, top_n)
    avg = sum(s.bear_score for s in shorts) / len(shorts)
    textbook = [s for s in shorts if s.setup_quality == "Textbook"]
    gated = [s for s in shorts if not s.tradeable]
    top = ranked[0]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Stocks Screened", len(shorts))
    k2.metric("Universe Avg Bear", f"{avg:.1f}")
    k3.metric(f"Most Bearish: {top.ticker}", f"{top.bear_score:.1f} ({top.grade})",
              f"{top.change_pct:+.1f}%")
    k4.metric("Textbook Setups", len(textbook),
              help="Bear score >=70, passed the tradeability gates, squeeze risk not High. "
                   "Measures chart pattern-match only - testing found no edge in it.")
    k5.metric("Gated Out", len(gated),
              help="Failed a borrow/squeeze/extension gate — shown but demoted")

    st.divider()
    st.subheader(f"📉 Top {top_n} Short Candidates")
    st.caption(
        "Ranked by a downtrend-continuation model — confirmed weakness, not "
        "overvaluation. **Stop sits ABOVE entry**; a short's loss is unbounded, "
        "so every row assumes a hard stop and half the position size you would "
        "use long."
    )
    df = _shorts_df(ranked)
    styled = (df.style
              .map(_colour_bear, subset=["Bear", "Trend", "Mom"])
              .map(_colour_chg, subset=["Chg %"])
              .format({"Bear": "{:.1f}"}))
    st.dataframe(styled, width='stretch', hide_index=True, height=420)

    if gated:
        with st.expander(f"⚠️ {len(gated)} name(s) failed a tradeability gate"):
            for s in gated[:15]:
                st.markdown(f"**{s.ticker}** (bear {s.bear_score:.1f}) — "
                            + "; ".join(s.gate_reasons))

    st.divider()
    st.subheader("🔍 Short Setup Deep Dive")
    choices = [f"{i}. {s.ticker} — {s.name[:30]} (Bear: {s.bear_score:.1f})"
               for i, s in enumerate(ranked, 1)]
    pick = st.selectbox("Select a candidate:", choices, key="short_pick")
    s = ranked[choices.index(pick)]

    _short_banner(s)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"${s.price:,.2f}", f"{s.change_pct:+.2f}%")
    rel = s.indicators.get("rel_strength")
    c2.metric("vs SPY (60d)", f"{rel:+.1f}pp" if rel is not None else "—",
              help="Negative = lagging the index. This is what a short actually sells.")
    c3.metric("ATR %", f"{s.atr_pct:.2f}%")
    c4.metric("Beta", f"{s.beta:.2f}" if s.beta is not None else "—")

    st.markdown("#### 📐 Short Trade Levels")
    if s.entry_zone_low and s.entry_zone_high:
        st.markdown(
            f'<div style="background:#ff444418; border:1px solid #ff444444; border-radius:8px; '
            f'padding:8px 16px; margin-bottom:8px; font-size:0.9em;">'
            f'🎯 <b>Short Entry Zone:</b>&nbsp; '
            f'<span style="color:#ff8c66; font-size:1.1em; font-weight:bold;">'
            f'${s.entry_zone_low:,.2f} – ${s.entry_zone_high:,.2f}</span>'
            f'&nbsp;&nbsp;(reference entry: ${s.entry:,.2f})</div>',
            unsafe_allow_html=True,
        )
    e = s.entry or s.price
    l1, l2, l3, l4, l5 = st.columns(5)
    l1.metric("Entry (sell)", f"${e:,.2f}")
    l2.metric("Stop (buy back)", f"${s.stop_loss:,.2f}",
              f"+{(s.stop_loss - e) / e * 100:.1f}%" if e else "",
              delta_color="inverse",
              help="ABOVE entry — the direction that loses money on a short.")
    l3.metric("Target 1", f"${s.target_1:,.2f}",
              f"-{(e - s.target_1) / e * 100:.1f}%" if e else "")
    l4.metric("Target 2", f"${s.target_2:,.2f}",
              f"-{(e - s.target_2) / e * 100:.1f}%" if e else "")
    l5.metric("Risk:Reward", f"{s.risk_reward:.1f}x")

    g1, g2 = st.columns([3, 2])
    with g1:
        df_hist = _history(s.ticker)
        if df_hist is not None and not df_hist.empty:
            st.plotly_chart(_short_chart(s.ticker, df_hist, s),
                            width='stretch', key=f"short_chart_{s.ticker}")
    with g2:
        st.plotly_chart(_subscore_radar(s), width='stretch',
                        key=f"short_radar_{s.ticker}")

    advice_html = s.trade_advice.replace("\n\n", "<br><br>").replace("\n", "<br>")
    st.markdown(
        f'<div class="advice-box" style="border-left-color:{s.setup_color};">'
        f"📋 <b>Short Thesis & Risk</b><br><br>{advice_html}</div>",
        unsafe_allow_html=True,
    )

    if s.signals:
        st.markdown("#### 🚦 Bearish Signals")
        st.markdown("  \n".join(f"- {x}" for x in s.signals))

    st.divider()
    st.caption(
        "⚠️ Short selling carries unbounded loss, borrow cost and recall risk, and "
        "requires a margin account. For informational purposes only. Not financial "
        "advice."
    )
