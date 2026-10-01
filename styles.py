"""
Shared CSS for every entry point.

Extracted from dashboard.py so that dashboard.py and gold_only.py cannot drift
apart. The gold tab styles itself with .advice-box / .status-pill /
.status-banner / .metric-card; if those lived in only one entry point, the other
would render unstyled HTML with no error to warn you.

Palette matches .streamlit/config.toml — #0e1117 ground, #1e1e2e cards,
#00aaff accent — plus the grade colours the app already uses for severity.
"""
from __future__ import annotations
import streamlit as st

APP_CSS = """
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
    .status-banner {
        border-radius: 10px; padding: 18px 24px; margin: 10px 0 16px 0;
        display: flex; align-items: center; gap: 20px; flex-wrap: wrap;
    }
    .status-pill {
        display: inline-block; padding: 5px 16px; border-radius: 20px;
        font-weight: bold; font-size: 1em; letter-spacing: 0.5px;
    }
    .advice-box {
        background: #1a1f2e; border-left: 4px solid #00aaff;
        border-radius: 0 8px 8px 0; padding: 16px 20px; margin: 10px 0;
        font-size: 0.92em; line-height: 1.7;
    }
    div[data-testid="stMetricValue"] { font-size: 1.6rem !important; }
</style>
"""


def inject() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)
