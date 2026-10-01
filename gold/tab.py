"""
Gold Risk & Volatility tab.

Reports MEASURED RISK, not direction. Eight years of testing across daily,
hourly and minute data produced no tradeable directional signal for gold, so
this tab does not emit one. What it does provide — how far gold moves at this
hour, what a trade costs, whether a target can clear the spread — is the part
that survived testing.

The single surviving candidate (16:00 ET drift) appears as a MONITORED
HYPOTHESIS with its full record on display, including a 50.1% eight-year win
rate and a 29-signal losing streak. Showing the failures next to the headline
is the point: a tab that displayed only the holdout would read as a 57.5%
system, which is exactly the misreading the research warns against.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gold import massive
from gold import research as R
from gold.live import (fetch_hourly, current_state, paper_record,
                       signal_card, get_spot, market_status)


@st.cache_data(ttl=1800, show_spinner=False)
def _hourly():
    return fetch_hourly("2y")


def _stream_note(q) -> str:
    """Describe the transport honestly, including a degraded socket.

    A stalled stream silently falling back to polling would look identical to a
    healthy one, so the reason is printed rather than hidden.
    """
    if q.get("streamed"):
        age = q.get("age")
        return f"streaming (tick {age:.1f}s old)" if age is not None else "streaming"
    st_ = massive.get_stream()
    if st_ is None:
        return "polled every 1s"
    h = st_.health()
    if h.get("error"):
        return f"polled - stream down ({h['error'][:40]})"
    if h.get("reconnects"):
        return f"polled - stream reconnecting ({h['reconnects']}x)"
    return "polled every 1s - stream starting"


@st.cache_resource(show_spinner=False)
def _ensure_stream():
    """Open the tick socket once per server process.

    cache_resource (not cache_data) because the value is a live thread, not a
    serialisable result, and it must be shared by every viewer. Streamlit reruns
    this module constantly; without the cache each rerun would open another
    socket and get the account throttled for self-inflicted reasons.
    """
    return massive.get_stream()


@st.cache_data(ttl=1, show_spinner=False)
def _spot():
    """1s cache matching the fragment timer.

    When the stream is up this is nearly free — get_spot() reads the newest tick
    straight out of memory and makes no network call at all. The TTL only
    throttles the REST fallback used while the socket is down or the market shut.
    """
    return get_spot()


def _vol_chart(sel_hour: int) -> go.Figure:
    hrs = sorted(R.VOL_PROFILE)
    xs = [R.VOL_PROFILE[h] for h in hrs]
    cols = [R.vol_state(x)[0] for x in xs]
    lines = ["#ffffff" if h == sel_hour else "rgba(0,0,0,0)" for h in hrs]
    fig = go.Figure(go.Bar(
        x=xs, y=[f"{h:02d}:00" for h in hrs], orientation="h",
        marker=dict(color=cols, line=dict(color=lines, width=1.5)),
        hovertemplate="%{y} ET<br>%{x:.2f}× average<extra></extra>",
    ))
    fig.add_vline(x=1.0, line=dict(color="#888", width=1, dash="dot"))
    fig.update_layout(
        height=520, margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#aaa", size=11),
        xaxis=dict(title="σ relative to gold's average hour",
                   gridcolor="#252540", zeroline=False),
        yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"),
        showlegend=False, bargap=0.25,
    )
    return fig


@st.fragment(run_every="1s")
def _live_block(d):
    """
    Live price + trade levels, re-running on a 1s timer.

    st.fragment reruns ONLY this function, not the whole page. Without it the
    price would sit frozen until the user clicked something, and a full-page
    autorefresh would re-execute the stock pipeline every second and reset the
    cost slider constantly.

    WHY 1s AND NOT FASTER
        The feed delivers ~1.6 ticks/sec, so the price itself is never more than
        a fraction of a second old once the WebSocket is up — reading it costs
        nothing, it is already in memory. The limit is Streamlit: each rerun
        re-renders the fragment server-side and ships it to the browser, so
        sub-second timers buy flicker rather than information. True per-tick
        repainting would require the socket in the BROWSER, which on a public
        site means handing the API key to every visitor.
    """
    _ensure_stream()
    sc = signal_card(d)
    q = _spot()
    if sc.get("ok"):
        # Re-anchor every level onto the tradeable price. The hourly frame is
        # GC=F futures, which sits ~$62 above spot — levels left on the futures
        # scale would be unfillable. Trend and stop DISTANCE survive the shift
        # (futures/spot hourly changes correlate 0.9979); only the level moves.
        if q.get("price"):
            shift = q["price"] - sc["price"]
            for kk in ("price", "entry", "stop", "tp1", "tp2", "tight_stop"):
                sc[kk] += shift
            sc["ema50"] += shift
            sc["ema200"] += shift
            sc["basis_shift"] = shift
        sc["quote"] = q

        bc = "#00ff88" if sc["bias"] == "BULLISH" else "#ff4444" if sc["bias"] == "BEARISH" else "#ffd700"
        side = "LONG" if sc["dirn"] > 0 else "SHORT"

        st.subheader("🥇 XAU/USD — live")
        is_spot = q.get("is_spot", False)
        two_sided = q.get("two_sided", False)
        # Three tiers, three colours — the user should never have to guess which
        # feed produced the number their orders are based on.
        if is_spot and two_sided and q.get("streamed"):
            src_col, src_label = "#00ff88", "XAU/USD SPOT · STREAMING BID/ASK"
        elif is_spot and two_sided:
            src_col, src_label = "#00ff88", "XAU/USD SPOT · LIVE BID/ASK (POLLED)"
        elif is_spot:
            src_col, src_label = "#00aaff", "XAU/USD SPOT · MID ONLY"
        else:
            src_col, src_label = "#ff8c00", "GC=F FUTURES — NOT SPOT"

        # Rendered as the sides you actually transact on: you BUY at the ask and
        # SELL at the bid. A single mid hides a cost charged on every round trip.
        if two_sided and q.get("bid"):
            sx = q.get("spread_x")
            sx_html = ""
            if sx:
                sx_col = "#ff4444" if sx >= 2.5 else "#ffd700" if sx >= 1.5 else "#00ff88"
                sx_html = (f'<span style="color:{sx_col};font-weight:700;">'
                           f'&nbsp;({sx:.1f}x typical)</span>')
            bidask = (
                f'<div style="display:flex;gap:14px;margin-top:6px;">'
                f'<div style="flex:1;background:#ff444414;border:1px solid #ff444433;'
                f'border-radius:6px;padding:5px 9px;">'
                f'<div style="font-size:.68em;color:#ff8c66;letter-spacing:.5px;">SELL (BID)</div>'
                f'<div style="font-size:1.05em;font-weight:700;color:#ff8c66;">'
                f'${q["bid"]:,.2f}</div></div>'
                f'<div style="flex:1;background:#00ff8814;border:1px solid #00ff8833;'
                f'border-radius:6px;padding:5px 9px;">'
                f'<div style="font-size:.68em;color:#00cc77;letter-spacing:.5px;">BUY (ASK)</div>'
                f'<div style="font-size:1.05em;font-weight:700;color:#00cc77;">'
                f'${q["ask"]:,.2f}</div></div></div>'
                f'<div style="font-size:.75em;color:#888;margin-top:5px;">'
                f'live spread <b style="color:#e8e8e8;">${q["spread"]:.3f}</b>{sx_html}</div>'
            )
        else:
            bidask = ('<div style="font-size:.78em;color:#888;margin-top:5px;">'
                      'mid price - no bid/ask on this feed</div>')
        tstr = q["ts"].strftime("%H:%M:%S ET") if q.get("ts") is not None else "—"

        # "checked" is wall-clock and changes on every fragment run, so the user
        # can tell a LIVE-BUT-FLAT market from a STALLED app. Quote time alone
        # cannot make that distinction — the most confusing state a live
        # dashboard can show is a frozen number with no explanation.
        now_et = datetime.now(ZoneInfo("America/New_York"))
        checked = now_et.strftime("%H:%M:%S")
        mkt = market_status(now_et)
        if not mkt["open"]:
            reopen = mkt["reopens"]
            mins = int((reopen - now_et).total_seconds() // 60) if reopen else 0
            st.info(
                f"🌙 **Gold market closed** — {mkt['reason']}. Price is the last traded "
                f"quote, not a stalled feed; the page is still polling every 5s. "
                f"Reopens {reopen:%a %H:%M ET}"
                + (f" (in {mins//60}h {mins%60}m)." if mins > 0 else ".")
            )

        # Tick delta against the previous refresh, so movement is visible rather
        # than inferred from a number that silently changes.
        prev = st.session_state.get("gold_prev_px")
        st.session_state["gold_prev_px"] = sc["price"]
        if prev is not None and abs(sc["price"] - prev) > 1e-9:
            dv = sc["price"] - prev
            tick_col = "#00ff88" if dv > 0 else "#ff4444"
            tick = (f'<span style="color:{tick_col};font-size:.42em;font-weight:700;'
                    f'vertical-align:middle;margin-left:10px;">'
                    f'{"▲" if dv > 0 else "▼"} {abs(dv):,.2f}</span>')
        else:
            tick = ('<span style="color:#555;font-size:.42em;vertical-align:middle;'
                    'margin-left:10px;">—</span>')

        p1, p2 = st.columns([1, 2])
        with p1:
            st.markdown(
                f'<style>@keyframes gpulse{{0%,100%{{opacity:1}}50%{{opacity:.25}}}}</style>'
                f'<div style="background:#1e1e2e;border-radius:10px;padding:16px 20px;'
                f'border-left:4px solid {src_col};">'
                f'<div style="font-size:.75em;color:{src_col};text-transform:uppercase;'
                f'letter-spacing:.6px;font-weight:700;">'
                f'<span style="display:inline-block;width:7px;height:7px;border-radius:50%;'
                f'background:{src_col};margin-right:7px;'
                f'animation:gpulse 1.6s ease-in-out infinite;"></span>{src_label}</div>'
                f'<div style="font-size:2.3em;font-weight:700;color:#e8e8e8;'
                f'line-height:1.1;">${sc["price"]:,.2f}{tick}</div>'
                f'{bidask}'
                f'<div style="font-size:.75em;color:#666;margin-top:3px;">'
                f'{q["source"]}<br>quote {tstr} · checked {checked} · {_stream_note(q)}</div></div>',
                unsafe_allow_html=True,
            )
        with p2:
            st.markdown(
                f'<div class="status-banner" style="background:linear-gradient(135deg,{bc}18,#1e1e2e);">'
                f'<span class="status-pill" style="background:{bc}22;color:{bc};'
                f'border:1.5px solid {bc};">{sc["icon"]} {sc["bias"]} BIAS · {side}</span>'
                f'<span style="color:#aaa;margin-left:14px;font-size:.9em;">'
                f'{sc["pct_from_ema200"]*100:+.2f}% vs EMA200 · '
                f'EMA50 ${sc["ema50"]:,.2f} · EMA200 ${sc["ema200"]:,.2f}</span></div>',
                unsafe_allow_html=True,
            )

        lv = st.columns(5)
        lv[0].metric("Entry", f"${sc['entry']:,.2f}", "market")
        lv[1].metric("Stop loss", f"${sc['stop']:,.2f}",
                     f"{-abs(sc['risk_usd']):.2f} · 1.5×ATR", delta_color="inverse")
        lv[2].metric("Take profit 1", f"${sc['tp1']:,.2f}",
                     f"+{abs(sc['tp1']-sc['entry']):.2f} · 2R")
        lv[3].metric("Take profit 2", f"${sc['tp2']:,.2f}",
                     f"+{abs(sc['tp2']-sc['entry']):.2f} · 3R")
        lv[4].metric("Risk / 0.01 lot", f"${sc['risk_usd']:.2f}",
                     f"vol {sc['vol_mult']:.2f}×")

        if not is_spot:
            st.error(
                "**Levels are on the FUTURES scale, not spot.** Both spot sources failed, "
                "so this fell back to `GC=F`. Measured basis against a real broker feed "
                "was **+$62.36 (+1.44%)** — an order at these prices would not fill. Trend "
                "direction and stop *distance* remain valid (hourly changes correlate "
                "0.9979); the absolute levels do not."
            )
        elif not two_sided:
            st.info(
                "**Spot mid only — no live bid/ask on this feed.** The primary "
                "(Massive XAU/USD) was unavailable, so this is `gold-api.com`, which "
                "updates about every 30s and measured **+$2.24** against Massive spot on "
                "2026-10-01. Levels are sound; the spread in the cost panel below is the "
                "broker-measured median, not an observed one."
            )
        elif q.get("spread_x") and q["spread_x"] >= 2.5:
            st.warning(
                f"⚠️ **Spread is {q['spread_x']:.1f}× its normal width right now** "
                f"(${q['spread']:.3f} vs ~${massive.SPREAD_TYPICAL:.2f} typical). Interbank "
                f"spreads widen on news and thin liquidity, and your broker's widens with "
                f"them — research measured yours reaching ${R.SPREAD_MAX:.2f} against a "
                f"${R.SPREAD_MEDIAN:.2f} median. The cost figures below assume the median, "
                f"so they are optimistic until this settles."
            )
        for w in sc["warnings"]:
            st.warning(f"⚠️ {w}")

        st.markdown(
            '<div class="advice-box" style="border-left-color:#ff8c00;">'
            '<b style="color:#e8e8e8;">Read this before acting on the card above.</b><br>'
            '<span style="color:#aaa;font-size:.9em;">'
            'These are <b>risk-managed levels for a trend bias</b>, not a validated entry '
            'signal. Direction comes from price vs EMA200 — the only input that survived '
            'testing, and in a matched control it accounted for essentially all of the '
            'apparent edge while the specific hour accounted for almost none. Expect a '
            '<b>~48% win rate</b> and losing streaks of <b>15+</b>. The 1.5×ATR stop is '
            'deliberately not tight: at 0.25–0.50×ATR the same rule had profit factor '
            '0.93–0.98 (i.e. it lost money), because gold\'s hourly noise is 20–45× larger '
            'than any drift measured.</span></div>',
            unsafe_allow_html=True,
        )
        with st.expander("Why not a tighter stop? (tested)"):
            st.markdown(
                f"A {0.5:.2f}×ATR stop would sit at **${sc['tight_stop']:,.2f}** "
                f"(risk ${abs(sc['entry']-sc['tight_stop']):.2f}). Tested over 1,642 "
                f"signals across 8 years:"
            )
            st.dataframe(pd.DataFrame([
                {"Stop": "0.25×ATR", "Win %": "39.3%", "Profit factor": "0.96", "Verdict": "loses"},
                {"Stop": "0.35×ATR", "Win %": "42.6%", "Profit factor": "0.93", "Verdict": "loses"},
                {"Stop": "0.50×ATR", "Win %": "44.9%", "Profit factor": "0.98", "Verdict": "loses"},
                {"Stop": "1.00×ATR", "Win %": "46.6%", "Profit factor": "1.15", "Verdict": "profitable"},
                {"Stop": "1.50×ATR", "Win %": "46.8%", "Profit factor": "1.15", "Verdict": "profitable ✓ used"},
            ]), hide_index=True, width='stretch')
            st.caption(
                "Tightening the stop lowers the win rate *and* the profit factor here — "
                "the stop is hit by noise before the drift arrives. Tight stops need an "
                "edge that is large relative to noise; gold at this horizon has the opposite."
            )
        st.divider()



def show_gold_tab() -> None:
    st.caption(
        f"XAUUSD · {R.SOURCE} · {R.H1_BARS:,} hourly bars ({R.H1_YEARS:.0f}y) · "
        f"{R.M1_BARS:,} minute bars ({R.M1_YEARS:.0f}y) · measured {R.MEASURED_AT}"
    )

    st.markdown(
        '<div class="advice-box"><p style="margin:0">This tab reports '
        '<b style="color:#e8e8e8">measured risk, not direction</b>. Eight years of testing '
        'produced no tradeable directional signal for gold — the failures are listed at the '
        'bottom of this page rather than hidden.</p></div>',
        unsafe_allow_html=True,
    )

    d = _hourly()
    s = current_state(d)
    col, label, dot = R.vol_state(s["vol_x"])

    _live_block(d)

    # ── live state ───────────────────────────────────────────────────────────
    st.subheader("📊 Right now")
    if s["halted"]:
        st.info("17:00 ET — CME daily settlement halt. Figures shown for the 18:00 ET reopen.")

    st.markdown(
        f'<div class="status-banner" style="background:linear-gradient(135deg,{col}18,#1e1e2e);">'
        f'<span class="status-pill" style="background:{col}22;color:{col};border:1.5px solid {col};">'
        f'{dot} {label.upper()} VOLATILITY</span>'
        f'<span style="color:#aaa;margin-left:14px;">{s["now"]:%H:%M} ET · '
        f'{s["vol_x"]:.2f}× the average hour</span></div>',
        unsafe_allow_html=True,
    )

    k = st.columns(5)
    k[0].metric("Gold", f"${s['price']:,.2f}" if s["price"] else "—",
                f"{s['chg_24h']*100:+.2f}% 24h" if s["chg_24h"] is not None else None)
    k[1].metric("Hourly volatility", f"{s['vol_x']:.2f}×", label)
    k[2].metric("Typical move this hour",
                f"${s['expected_move']:.2f}" if s["expected_move"] else "—",
                f"1σ · {s['vol_x']:.2f}× average")
    k[3].metric("Suggested stop",
                f"${s['suggested_stop']:.2f}" if s["suggested_stop"] else "—",
                "1.5 × live ATR")
    k[4].metric("Min viable target", f"${R.min_viable_target():.2f}", "spread ≤ 5%")

    st.divider()

    # ── volatility profile ───────────────────────────────────────────────────
    st.subheader("🕐 Volatility by hour")
    st.caption(
        "Relative to gold's own average hour, from 8 years of data. The 2.6× spread between "
        "the quietest and busiest hour replicated across two independent data providers — "
        "it is the most reliable finding in this research. A stop that is sane at 23:00 gets "
        "swept at 10:00. 17:00 ET is absent: CME settlement halt."
    )
    left, right = st.columns([2, 1])
    with left:
        st.plotly_chart(_vol_chart(s["hour"]), key="gold_vol")
    with right:
        for name, hrs in R.SESSIONS:
            avg = sum(R.VOL_PROFILE[h] for h in hrs if h in R.VOL_PROFILE) / \
                  max(len([h for h in hrs if h in R.VOL_PROFILE]), 1)
            c2, l2, d2 = R.vol_state(avg)
            st.markdown(
                f'<div style="background:#1e1e2e;border-radius:10px;padding:10px 14px;'
                f'margin-bottom:8px;"><div style="font-size:.78em;color:#888;">{name}</div>'
                f'<div style="font-size:1.3em;font-weight:700;color:{c2};">{avg:.2f}× {d2}</div>'
                f'<div style="font-size:.75em;color:#666;">avg across '
                f'{len([h for h in hrs if h in R.VOL_PROFILE])} hours</div></div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ── cost check ───────────────────────────────────────────────────────────
    st.subheader("💰 Cost check")
    st.caption(
        # NB: raw f-strings. Streamlit parses $...$ as LaTeX math, so every literal
        # dollar sign in prose must reach markdown as \$ or the numbers get eaten.
        rf"Broker-measured spread: median \${R.SPREAD_MEDIAN:.2f}, p90 \${R.SPREAD_P90:.2f}, "
        rf"p99 \${R.SPREAD_P99:.3f}, max \${R.SPREAD_MAX:.2f} on news. This is the constraint "
        rf"that ended the scalping research — gold's largest measurable short-horizon edge "
        rf"was \${R.BEST_M1_EDGE_USD:.3f}, well under a single spread."
    )
    c1, c2 = st.columns([1, 2])
    with c1:
        tgt = st.slider("Target size ($)", 0.5, 20.0, 4.0, 0.25, key="gold_tgt")
        share = R.SPREAD_MEDIAN / tgt * 100
        if share > 15:
            st.error(f"**{share:.1f}%** of this target goes to spread. You must be right far "
                     f"more often than chance simply to break even.")
        elif share > 5:
            st.warning(f"**{share:.1f}%** goes to spread. Viable only with a genuine edge — "
                       f"and none was found at this horizon.")
        else:
            st.success(f"**{share:.1f}%** goes to spread. Cost is no longer the binding constraint.")
    with c2:
        rows = []
        for t in (1, 2, 4, 5, 10, 20):
            p = R.SPREAD_MEDIAN / t * 100
            rows.append({"Target": f"${t:.2f}", "Spread": f"${R.SPREAD_MEDIAN:.2f}",
                         "Share lost": f"{p:.1f}%",
                         "Assessment": "Spread dominates" if p > 15 else
                                       "Marginal" if p > 5 else "Workable"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')

    st.divider()

    # ── monitored hypothesis ─────────────────────────────────────────────────
    h = R.HYPOTHESIS
    st.subheader("🔬 Monitored hypothesis")
    st.markdown(
        f'<span class="status-pill" style="background:#ff8c0033;color:#ff8c00;'
        f'border:1.5px solid #ff8c00;">⚠️ {h["status"]}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="advice-box" style="border-left-color:#ff8c00;margin-top:10px;">'
        f'<b style="color:#e8e8e8;">{h["name"]}</b><br>'
        f'<span style="color:#aaa;font-size:.92em;">{h["rule"]}</span></div>',
        unsafe_allow_html=True,
    )

    p1, p2, p3 = st.columns(3)
    for cc, key, ttl in ((p1, "full", "Full 8 years"), (p2, "train", "Train 2018–24"),
                         (p3, "holdout", "Holdout 2024–26")):
        b = h[key]
        cc.markdown(
            f'<div style="background:#1e1e2e;border-radius:10px;padding:14px 16px;">'
            f'<div style="font-size:.75em;color:#888;text-transform:uppercase;'
            f'letter-spacing:.6px;font-weight:600;">{ttl}</div>'
            f'<div style="font-size:1.5em;font-weight:700;color:'
            f'{"#00ff88" if b["win"] >= .55 else "#ffd700" if b["win"] >= .5 else "#ff4444"};">'
            f'{b["win"]*100:.1f}% win</div>'
            f'<div style="font-size:.82em;color:#aaa;margin-top:6px;">'
            f'{b["n"]:,} signals · PF {b["pf"]:.2f}<br>'
            f'max DD <b style="color:#ff4444;">${b["maxdd_lot"]:,.0f}</b>/lot<br>'
            f'losing streak <b style="color:#ff4444;">{b["streak"]}</b><br>'
            f'p = {b["p"]:.4f}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("**Why this is not presented as a signal**")
    for c in h["caveats"]:
        st.markdown(f"- {c}")

    yr = pd.DataFrame({"Year": list(h["years"]), "Net $/lot": list(h["years"].values())})
    fig = go.Figure(go.Bar(
        x=yr["Year"], y=yr["Net $/lot"],
        marker_color=["#ff4444" if v < 0 else "#00aaff" for v in yr["Net $/lot"]],
        hovertemplate="%{x}: $%{y:,.0f}<extra></extra>"))
    fig.update_layout(height=230, margin=dict(l=8, r=8, t=8, b=8),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#aaa", size=11),
                      xaxis=dict(gridcolor="rgba(0,0,0,0)"),
                      yaxis=dict(title="Net $ per 1.0 lot", gridcolor="#252540"),
                      showlegend=False)
    st.plotly_chart(fig, key="gold_years")
    st.caption(
        "2025–26 alone are 86% of the eight-year total, while 2021 and 2022 both lost money. "
        "This pays in strongly trending gold and bleeds when gold ranges — which is leveraged "
        "long exposure with extra steps, not a stable edge."
    )

    # ── forward paper record ─────────────────────────────────────────────────
    st.markdown("**📝 Forward paper record**")
    rec = paper_record(d, h["paper_start"])
    if rec["n"] == 0:
        st.info(
            f"Paper test starts {h['paper_start']}. No signals recorded yet — the first will "
            f"appear after the next 16:00 ET close. This record is recomputed from price "
            f"history on every load, so it cannot drift or double-count."
        )
    else:
        q = st.columns(5)
        q[0].metric("Signals", rec["n"])
        q[1].metric("Win rate", f"{rec['win']*100:.1f}%")
        q[2].metric("Net", f"${rec['net_lot']:+,.0f}/lot")
        q[3].metric("Max drawdown", f"${rec['maxdd_lot']:+,.0f}/lot")
        q[4].metric("Longest losing run", rec["streak"])
        t = rec["trades"].copy()
        t["Date"] = pd.to_datetime(t["date"]).dt.strftime("%Y-%m-%d")
        t["Entry"] = t["entry"].map("${:,.2f}".format)
        t["Exit"] = t["exit"].map("${:,.2f}".format)
        t["Net $/lot"] = t["net_lot"].map("{:+,.2f}".format)
        t["Result"] = t["win"].map({True: "WIN", False: "LOSS"})
        st.dataframe(t[["Date", "Entry", "Exit", "Net $/lot", "Result"]].iloc[::-1],
                     hide_index=True, width='stretch', height=280)
        st.caption(
            "Forward results only. Compare the win rate above against the 50.1% eight-year "
            "figure, not against the 57.5% holdout — the holdout sat entirely inside gold's "
            "run from $2,386 to $4,336."
        )

    st.divider()

    # ── ledger ───────────────────────────────────────────────────────────────
    st.subheader("🧪 What was tested and failed")
    st.caption(
        "Kept visible on purpose. Knowing where the edge is not saves you from paying to "
        "rediscover it."
    )
    for name, verdict, ok in R.LEDGER:
        c = "#00ff88" if ok else "#ff4444"
        bg = "#00ff8822" if ok else "#ff444422"
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;gap:14px;'
            f'padding:9px 14px;border-bottom:1px solid #1b1b2b;">'
            f'<span style="color:#aaa;font-size:.92em;">{name}</span>'
            f'<span style="background:{bg};color:{c};border:1px solid {c}55;padding:2px 10px;'
            f'border-radius:4px;font-size:.74em;font-weight:700;white-space:nowrap;">{verdict}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.caption(
        "⚠️ For informational purposes only. Not financial advice. Spread is broker-specific — "
        "a raw-spread account changes every cost figure here and is the highest-leverage "
        "variable on this page."
    )
