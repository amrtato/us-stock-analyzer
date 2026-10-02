"""
Crypto Long/Short tab — daily trend ranking across liquid USD pairs.

The ranking shown here is EXACTLY the one `backtest_crypto.py` tested: the same
`compute_features` and `score_cross_section`, applied to the latest slice of the
same panel. That is deliberate. The quickest way to ship a screen that behaves
differently from the thing you validated is to let the research path and the
live path compute the score separately, so they share one implementation.

The EVIDENCE block below carries the measured outcome. Re-run the backtest and
update it if the model is ever changed — a screen quoting stale validation
numbers is worse than one quoting none.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from crypto import data as cd
from crypto.scoring import BENCH, compute_features, rank_latest, score_cross_section

# From backtest_crypto.py, 2026-10-02. Do not soften these.
EVIDENCE = {
    "headline": (
        "⚠️ **This ranking showed no measurable edge.** Walk-forward testing "
        "(61 liquid pairs, 2023–2026, non-overlapping 14-day windows, costs "
        "included) found the long/short spread at **−1.00% (p=0.59)** — while a "
        "**random** ranking scored **+0.28%**. The long leg's +2.90% is the "
        "market's own drift (+2.36%), not selection. Use this to find liquid "
        "names with a clear trend, not as a trade list."
    ),
    "detail": """
**Test setup** — 300 candidate USD pairs narrowed to **61** by liquidity
(≥$1M/day median) and a volatility floor that removes stablecoins; 2023-10 →
2026-09; point-in-time scoring; **non-overlapping** 14-day windows (n=78);
20bp per leg round-trip cost.

| | long leg | short leg | **long/short** | rank IC |
|---|---|---|---|---|
| **score** | +2.90% (p=0.13) | −3.90% (p=0.10) | **−1.00% (p=0.59)** | −0.013 |
| **random control** | +1.42% | −1.15% | **+0.28%** | −0.001 |

**Read the random row first.** It beat the model on the only column that
isolates skill. The long leg looks healthy purely because crypto drifted
**+2.36% per 14 days** over this window — a coin flip inherits that too.

**The rank IC has the wrong sign** (−0.013): higher-scoring coins went on to do
marginally *worse*. And neither half of the sample rescues it — early L/S
+0.57% (p=0.84), late −2.58% (p=0.28).

**Checked for an underpowered test, not just a bad one.** A first run with a
$5M/day floor left only 29 pairs, i.e. three names per decile, which is three
coin flips rather than a portfolio. Widening to 61 pairs changed nothing:
L/S −2.04% → −1.00%, both indistinguishable from random. The conclusion is not
an artefact of universe size.

**Survivorship cuts hard here.** The universe is pairs liquid *today*, so coins
that died during the sample never appear. That flatters the long leg and robs
the short leg of its best candidates — worse than the equivalent equity bias,
because crypto mortality is far higher.

**What the data does support:** only 61 of 300 pairs are liquid enough to trade,
and the market was positive on just 47% of windows despite a +2.36% mean — the
returns come from rare large moves, which is a position-sizing fact, not a
signal.
""",
}

LOOKBACK_DAYS = 220          # enough for EMA100 plus the 30-day features


@st.cache_data(ttl=1800, show_spinner=False)
def _panel(top_candidates: int = 300) -> pd.DataFrame:
    """Daily bars for the liquid universe. One grouped call + N range calls."""
    end = datetime.now(UTC).date()
    start = end - timedelta(days=LOOKBACK_DAYS)
    tickers = cd.candidate_tickers(top_n=top_candidates)
    if not tickers:
        return pd.DataFrame()
    if BENCH not in tickers:
        tickers.append(BENCH)
    panel = cd.build_panel(tickers, start, end)
    if panel.empty:
        return panel
    return cd.apply_universe_filter(panel, min_dollar_volume=1_000_000.0)


@st.cache_data(ttl=1800, show_spinner=False)
def _scored(top_candidates: int = 300) -> pd.DataFrame:
    p = _panel(top_candidates)
    if p.empty:
        return p
    return score_cross_section(compute_features(p))


def _px(v: float) -> str:
    """Price with precision that scales to magnitude.

    A fixed 4dp renders SHIB (~0.00001) as "$0.0000" - the screen would show a
    row with no price on it. Crypto spans eight orders of magnitude in this
    universe, so the decimals have to follow the number.
    """
    if v is None or pd.isna(v):
        return "—"
    v = float(v)
    if v >= 1000:   return f"${v:,.2f}"
    if v >= 1:      return f"${v:,.3f}"
    if v >= 0.01:   return f"${v:,.5f}"
    if v >= 0.0001: return f"${v:,.7f}"
    return f"${v:,.9f}"


def _fmt(rows: pd.DataFrame, side: str) -> pd.DataFrame:
    out = []
    for i, (_, r) in enumerate(rows.iterrows(), 1):
        sym = str(r["ticker"]).replace("X:", "").replace("USD", "")
        px = r["close"]
        out.append({
            "#": i,
            "Asset": sym,
            "Price": _px(px),
            "Score": round(float(r["score"]), 2),
            "7d %": f"{r['roc7']:+.1f}%" if pd.notna(r["roc7"]) else "—",
            "30d %": f"{r['roc30']:+.1f}%" if pd.notna(r["roc30"]) else "—",
            "vs BTC": f"{r['rel_btc']:+.1f}pp" if pd.notna(r["rel_btc"]) else "—",
            "Ann vol": f"{r['vol30']:.0f}%" if pd.notna(r["vol30"]) else "—",
            # A realistic stop has to scale with the asset's own volatility; a
            # fixed percentage is meaningless across names that range 40-200%.
            "1σ / 14d": (_px(px * (r["vol30"] / 100) * (14 / 365) ** 0.5)
                         if pd.notna(r["vol30"]) else "—"),
            "$ vol/day": f"${r['dollar_volume'] / 1e6:,.1f}M",
        })
    return pd.DataFrame(out)


def _colour_side(side: str):
    pos, neg = ("#00ff88", "#ff4444")
    def _f(val):
        try:
            v = float(str(val).replace("%", "").replace("pp", "").replace("+", ""))
        except (TypeError, ValueError):
            return ""
        return f"color: {pos};" if v > 0 else (f"color: {neg};" if v < 0 else "")
    return _f


def _spread_chart(longs: pd.DataFrame, shorts: pd.DataFrame) -> go.Figure:
    def sym(df):
        return [str(t).replace("X:", "").replace("USD", "") for t in df["ticker"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=sym(longs)[::-1], x=longs["score"].tolist()[::-1],
                         orientation="h", marker_color="#00cc66", name="Long"))
    fig.add_trace(go.Bar(y=sym(shorts)[::-1], x=shorts["score"].tolist()[::-1],
                         orientation="h", marker_color="#ff4444", name="Short"))
    fig.update_layout(
        height=max(320, 22 * (len(longs) + len(shorts))),
        margin=dict(l=10, r=10, t=36, b=10),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="#ffffff",
        title="Trend score — both ends of one ranking", barmode="relative",
        legend=dict(orientation="h", y=1.08, x=0),
    )
    fig.update_xaxes(gridcolor="#1e1e2e", zerolinecolor="#444")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    return fig


def show_crypto_tab(n: int = 10) -> None:
    st.markdown("## ₿ Crypto Long / Short — Daily Candidates")

    if not cd.available():
        st.error("MASSIVE_API_KEY is not configured — the crypto feed needs it.")
        return

    st.warning(EVIDENCE["headline"])
    with st.expander("What the backtest actually found"):
        st.markdown(EVIDENCE["detail"])

    scored = _scored()
    if scored.empty:
        st.error("No crypto data returned. Check the Massive subscription and connectivity.")
        return

    longs, shorts = rank_latest(scored, n=n)
    if longs.empty:
        st.error("Scoring produced no ranked candidates.")
        return

    asof = pd.Timestamp(scored["date"].max()).date()
    universe = scored[scored["date"] == scored["date"].max()]["ticker"].nunique()
    drift = scored[scored["date"] == scored["date"].max()]["roc30"].median()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Liquid universe", universe,
              help="USD pairs clearing $1M/day median volume, excluding pegs")
    k2.metric("As of", str(asof))
    k3.metric("Median 30d move", f"{drift:+.1f}%" if pd.notna(drift) else "—",
              help="The market's own drift. A long list inherits this for free.")
    k4.metric("Benchmark", "BTC",
              help="Relative strength is measured against BTC, which dominates crypto returns")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"### 🟢 Long candidates — top {n}")
        st.caption("Strongest trend + relative strength vs BTC, volatility-adjusted.")
        df = _fmt(longs, "long")
        st.dataframe(df.style.map(_colour_side("long"), subset=["7d %", "30d %", "vs BTC"]),
                     width="stretch", hide_index=True, height=min(620, 45 + 35 * len(df)))
    with c2:
        st.markdown(f"### 🔴 Short candidates — bottom {n}")
        st.caption("Weakest trend + underperformance vs BTC. Perpetuals have no "
                   "borrow constraint, so a short here is a true mirror of a long.")
        df = _fmt(shorts, "short")
        st.dataframe(df.style.map(_colour_side("short"), subset=["7d %", "30d %", "vs BTC"]),
                     width="stretch", hide_index=True, height=min(620, 45 + 35 * len(df)))

    st.plotly_chart(_spread_chart(longs, shorts), key="crypto_spread")

    st.markdown(
        '<div class="advice-box" style="border-left-color:#ff8c00;">'
        "<b style='color:#e8e8e8;'>Read this before acting on either list.</b><br><br>"
        "<b>The 1σ / 14d column is the number to size from.</b> It is the one-standard-"
        "deviation move over the holding period the model was tested on, from each "
        "asset's own 30-day volatility. Crypto vol ranges roughly 40–200% annualised "
        "across this universe, so a fixed percentage stop is meaningless: the same "
        "stop that is wide on BTC is noise on a small-cap altcoin.<br><br>"
        "<b>The market was positive on only 47% of 14-day windows</b> despite averaging "
        "+2.36%. The returns come from rare large moves, not consistency — so "
        "position size for a long run of losing windows even when the mean is positive."
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.caption(
        "⚠️ Crypto trades 24/7 with no circuit breakers and venue-dependent liquidity. "
        "For informational purposes only. Not financial advice."
    )
