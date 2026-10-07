"""
Latched gold signal — fixed levels from the moment of issue until it resolves.

THE BUG THIS EXISTS TO FIX
    The card used to recompute entry/stop/TP from the live price on every tick,
    so all four numbers drifted continuously. Look away for five minutes and the
    "signal" had different levels, with nothing marking that it had changed.
    That is not a signal, it is a moving quote wearing a signal's clothes, and
    it is actively misleading: you cannot tell whether a level was hit, because
    the level moved.

    A signal has to be a FACT about a moment: issued at time T, at price P, with
    a stop and targets fixed at issue. After that it can only resolve — target
    hit, stop hit, or invalidated by the bias flipping. Those are the only
    events that may change the numbers, and each one issues a NEW signal rather
    than editing the old one.

WHY THIS MODULE HOLDS STATE WHEN gold/live.py DELIBERATELY DOES NOT
    live.paper_record() is stateless on purpose: the 16:00 ET rule is
    deterministic, so replaying it over history reproduces the record exactly,
    and a log file would only add something that can desync.

    A latched live signal is the opposite. It depends on WHEN someone was
    watching — the entry is the price at the instant the bias established, which
    no amount of history can reconstruct. That genuinely requires persistence.

PERSISTENCE LOCATION
    Home directory, not the project. On Azure App Service /home survives both
    restarts AND deploys, while wwwroot is replaced on every deploy — a signal
    that vanished on each deploy would silently re-issue at a new price and
    reintroduce the drift this module removes.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
MAX_HISTORY = 20


def _default_path() -> Path:
    """Where to persist. Azure first, home second, env override always wins.

    On App Service, /home is the persistent share and the documented place for
    app data; Path.home() does NOT reliably resolve there. Getting this wrong
    was observed in production: the file silently failed to persist, so every
    Streamlit rerun re-issued a signal at the current price - the precise drift
    this module exists to eliminate, just at a coarser interval.
    """
    import os
    env = os.getenv("GOLD_SIGNAL_PATH")
    if env:
        return Path(env)
    azure = Path("/home/data")
    try:
        if Path("/home").is_dir():
            azure.mkdir(parents=True, exist_ok=True)
            return azure / "aurum_gold_signal.json"
    except OSError:
        pass
    return Path.home() / ".aurum_gold_signal.json"


STATE_PATH = _default_path()

_lock = threading.Lock()

# IN-PROCESS STATE IS THE PRIMARY STORE; THE FILE IS ONLY DURABILITY.
#
# A file that cannot be written must not cause a re-issue, because a re-issue
# silently restores floating levels. Holding the state in memory means the
# signal survives every rerun for the life of the process no matter what the
# filesystem does; the file then adds survival across restarts, best-effort.
_mem: dict | None = None
_persist_ok: bool | None = None

# Terminal states. ACTIVE is the only one whose levels are still in play.
ACTIVE, TP1, TP2, STOPPED, FLIPPED = "ACTIVE", "TP1_HIT", "TP2_HIT", "STOPPED", "FLIPPED"


def _now() -> datetime:
    return datetime.now(ET)


def _load() -> dict:
    global _mem
    if _mem is not None:                       # memory wins; see note above
        return _mem
    try:
        if STATE_PATH.exists():
            with STATE_PATH.open(encoding="utf-8") as fh:
                d = json.load(fh)
            d.setdefault("active", None)
            d.setdefault("history", [])
            _mem = d
            return d
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("gold signal state unreadable (%s) — starting fresh", exc)
    _mem = {"active": None, "history": []}
    return _mem


def _save(state: dict) -> None:
    global _mem, _persist_ok
    state["history"] = state.get("history", [])[-MAX_HISTORY:]
    _mem = state                               # always, even if the file fails
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE_PATH.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=1)
        tmp.replace(STATE_PATH)        # atomic, so a crash cannot truncate it
        _persist_ok = True
    except OSError as exc:
        if _persist_ok is not False:
            log.warning("gold signal not durable (%s) — in-memory only, "
                        "a restart will re-issue", exc)
        _persist_ok = False


def persistence() -> dict:
    """So the UI can say whether the signal survives a restart, rather than
    implying durability it does not have."""
    return {"path": str(STATE_PATH), "durable": _persist_ok}


# Minutes to wait before re-issuing in the SAME direction after a stop-out.
# Gold's hourly noise is 20-45x the measured drift, so a stop is usually noise
# rather than information; re-entering instantly just pays the spread again.
STOP_COOLDOWN_MIN = 60.0


def _blocked_by_cooldown(state: dict, dirn: int) -> bool:
    """True if the last signal was stopped recently in this same direction."""
    for h in reversed(state.get("history", [])):
        if h.get("status") != STOPPED:
            continue
        if int(h.get("dirn", 0)) != int(dirn):
            return False                     # other side: not blocked
        try:
            when = datetime.fromisoformat(h["resolved_at"])
        except (TypeError, ValueError, KeyError):
            return False
        mins = (_now() - when).total_seconds() / 60.0
        return mins < STOP_COOLDOWN_MIN
    return False


def _issue(bias: str, dirn: int, price: float, risk: float, atr: float) -> dict:
    """Freeze a new signal at the current price. Levels never change after this."""
    return {
        "bias": bias,
        "dirn": int(dirn),
        "entry": round(price, 2),
        "stop": round(price - dirn * risk, 2),
        "tp1": round(price + dirn * 2.0 * risk, 2),
        "tp2": round(price + dirn * 3.0 * risk, 2),
        "risk": round(risk, 2),
        "atr": round(atr, 2),
        "issued_at": _now().isoformat(),
        "status": ACTIVE,
        "resolved_at": None,
        "mfe": 0.0,          # best excursion in favour, in dollars
        "mae": 0.0,          # worst excursion against
    }


def _resolve(sig: dict, price: float) -> str:
    """Has this signal hit a level? Checked against the FIXED levels only."""
    d = sig["dirn"]
    if d > 0:
        if price <= sig["stop"]:
            return STOPPED
        if price >= sig["tp2"]:
            return TP2
        if price >= sig["tp1"] and sig["status"] == ACTIVE:
            return TP1
    else:
        if price >= sig["stop"]:
            return STOPPED
        if price <= sig["tp2"]:
            return TP2
        if price <= sig["tp1"] and sig["status"] == ACTIVE:
            return TP1
    return sig["status"]


def update(bias: str, dirn: int, price: float, risk: float, atr: float,
           market_open: bool = True) -> dict:
    """Advance the latched signal and return it.

    Call on every refresh. Returns a dict with the FIXED levels plus live
    context (unrealised move, distance to each level, age).

    The only events that change the levels are: no signal yet, the previous one
    resolved, or the bias flipped. Everything else leaves them alone — that is
    the entire point.
    """
    if not price or price <= 0 or risk <= 0:
        return {"ok": False, "reason": "no live price"}

    with _lock:
        state = _load()
        sig = state.get("active")
        events = []

        # 1. An existing signal first gets checked against ITS OWN fixed levels.
        #    This must include TP1_HIT, not just ACTIVE. Guarding on ACTIVE
        #    alone meant that once target 1 printed, _resolve() was never called
        #    again: TP2 could not register AND THE STOP COULD NOT TRIGGER, so a
        #    trade that reversed all the way through its stop still displayed
        #    "TP1 HIT". Caught by the lifecycle test.
        if sig and sig.get("status") in (ACTIVE, TP1):
            new_status = _resolve(sig, price)
            if new_status != sig["status"]:
                sig["status"] = new_status
                if new_status in (TP2, STOPPED):
                    sig["resolved_at"] = _now().isoformat()
                events.append(new_status)
            # TP1 leaves the signal live (runner to TP2); TP2/stop close it.
            if sig["status"] in (TP2, STOPPED):
                state["history"].append(sig)
                state["active"] = sig = None

        # 2. A bias flip invalidates whatever is open. It does NOT edit the old
        #    signal's numbers - it closes it and issues a new one, so the record
        #    shows what actually happened.
        if sig and sig.get("dirn") != int(dirn) and bias != "MIXED":
            sig["status"] = FLIPPED
            sig["resolved_at"] = _now().isoformat()
            state["history"].append(sig)
            state["active"] = sig = None
            events.append(FLIPPED)

        # 3. Issue only when there is nothing live and the market is open. A
        #    signal issued into a closed market would latch a stale print as its
        #    entry and look tradeable on Sunday morning.
        #
        #    COOLDOWN AFTER A STOP: without it, being stopped out re-issued the
        #    SAME direction at the stop price on the very next tick, because the
        #    bias had not changed. In a chop that is an infinite re-entry loop,
        #    each one entered at the worst available price. A flip still issues
        #    immediately - that is new information; being stopped is not.
        if sig is None and market_open and _blocked_by_cooldown(state, dirn):
            sig = None
        elif sig is None and market_open:
            sig = _issue(bias, dirn, price, risk, atr)
            state["active"] = sig
            events.append("ISSUED")

        # 4. Track excursions against the fixed levels, for honesty about how
        #    close it came rather than only where it ended.
        if sig and sig.get("status") in (ACTIVE, TP1):
            move = (price - sig["entry"]) * sig["dirn"]
            sig["mfe"] = round(max(sig.get("mfe", 0.0), move), 2)
            sig["mae"] = round(min(sig.get("mae", 0.0), move), 2)
            state["active"] = sig

        _save(state)

    if not sig:
        reason = ("market closed - no signal issued" if not market_open
                  else f"cooling off after a stop-out (up to {STOP_COOLDOWN_MIN:.0f}m, "
                       f"or until the bias flips)")
        return {"ok": False, "reason": reason,
                "history": state.get("history", [])[::-1]}

    issued = datetime.fromisoformat(sig["issued_at"])
    age_min = (_now() - issued).total_seconds() / 60.0
    move = (price - sig["entry"]) * sig["dirn"]
    return {
        "ok": True,
        **sig,
        "events": events,
        "age_min": age_min,
        "issued_dt": issued,
        "live_price": price,
        "move": round(move, 2),
        "move_r": round(move / sig["risk"], 2) if sig["risk"] else 0.0,
        "to_stop": round(abs(price - sig["stop"]), 2),
        "to_tp1": round(abs(sig["tp1"] - price), 2),
        "to_tp2": round(abs(sig["tp2"] - price), 2),
        "history": state.get("history", [])[::-1],
    }


def reset() -> None:
    """Drop the active signal (not the history). For manual intervention."""
    with _lock:
        global _mem
        state = _load()
        if state.get("active"):
            s = state["active"]
            s["status"] = FLIPPED
            s["resolved_at"] = _now().isoformat()
            state["history"].append(s)
        state["active"] = None
        _save(state)
