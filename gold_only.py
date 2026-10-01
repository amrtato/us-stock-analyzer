"""
Gold Risk & Volatility — standalone entry point.

WHY THIS EXISTS
    dashboard.py runs the full stock pipeline on first load: ~101 tickers, five
    HTTP calls each, 3-4 minutes cold. st.tabs is not lazy, so that work happens
    whether or not you ever open a stock tab. Reordering the tab blocks means
    gold no longer WAITS for it, but the fetching still runs in the background
    and still costs bandwidth and rate-limit budget.

    This entry point imports only the gold modules. No stock universe, no
    scoring, no news fetch — roughly 3 seconds to first paint. Use it when you
    want to sit and watch gold levels during a session.

Launch:
    streamlit run gold_only.py --server.port 8512

Both entry points share styles.py and gold/, so there is nothing to keep in
sync by hand.
"""
from __future__ import annotations
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st

st.set_page_config(
    page_title="Gold Risk & Volatility",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from styles import inject as _inject_css          # noqa: E402
from gold.tab import show_gold_tab                # noqa: E402

_inject_css()

ET = ZoneInfo("America/New_York")

col_t, col_d = st.columns([3, 1])
with col_t:
    st.title("🥇 Gold Risk & Volatility")
with col_d:
    st.metric("Now", datetime.now(ET).strftime("%H:%M ET"),
              datetime.now(ET).strftime("%a %d %b"))

st.divider()

show_gold_tab()

st.divider()
st.caption(
    "Standalone gold view — no stock data is loaded here. "
    "For the full app including US stock rankings, run `streamlit run dashboard.py`."
)
