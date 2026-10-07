"""
Latched signals — fixed levels from the moment of issue until they resolve.

Shared by the gold card (one instrument) and the crypto long/short lists
(tens of instruments at once), keyed per instrument. It started gold-only;
duplicating it for crypto was rejected because this module has already
produced four distinct bugs under test, and two copies means finding the
fifth one twice.

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
    # Filename says "gold" for historical reasons - it predates crypto sharing
    # this store. Deliberately NOT renamed: the path is the identity of the file,
    # and changing it would orphan whatever signal is live at deploy time, which
    # is exactly the silent re-issue this module exists to prevent. A cosmetic
    # name is not worth dropping a position the page has told someone about.
    return Path.home() / ".aurum_gold_signal.json"


STATE_PATH = _default_path()

# Default key, so the single-instrument gold caller needs no key argument and
# the pre-existing state file keeps working.
GOLD = "XAUUSD"

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


def _migrate(d: dict) -> dict:
    """Old single-instrument shape {active, history} -> {signals:{key:…}, history}.

    Done in place on read so a deploy does not drop a live gold signal and
    silently re-issue it at a new price - the exact failure this module exists
    to prevent, and one that already bit once in production.
    """
    d.setdefault("history", [])
    if "signals" not in d:
        d["signals"] = {}
        if d.get("active"):
            d["signals"][GOLD] = d["active"]
        d.pop("active", None)
    return d


def _load() -> dict:
    global _mem
    if _mem is not None:                       # memory wins; see note above
        return _mem
    try:
        if STATE_PATH.exists():
            with STATE_PATH.open(encoding="utf-8") as fh:
                d = _migrate(json.load(fh))
            _mem = d
            return d
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("signal state unreadable (%s) — starting fresh", exc)
    _mem = {"signals": {}, "history": []}
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


def _blocked_by_cooldown(state: dict, key: str, dirn: int, cooldown_min: float) -> bool:
    """True if THIS instrument was stopped recently in this same direction.

    Keyed per instrument: a stop on one crypto pair must not mute every other
    pair, which an un-keyed history scan would have done the moment this module
    went multi-instrument.
    """
    for h in reversed(state.get("history", [])):
        if h.get("key", GOLD) != key or h.get("status") != STOPPED:
            continue
        if int(h.get("dirn", 0)) != int(dirn):
            return False                     # other side: not blocked
        try:
            when = datetime.fromisoformat(h["resolved_at"])
        except (TypeError, ValueError, KeyError):
            return False
        return (_now() - when).total_seconds() / 60.0 < cooldown_min
    return False


def _r(v: float) -> float:
    """Round to a precision that scales with magnitude.

    This was a flat round(v, 2), which is fine for gold in the thousands and
    destroys crypto: TRX at 0.3366 became 0.34, and SHIB at 0.0000098 would
    become 0.00 - a stop of zero. Anything priced under a dollar needs the
    decimals to follow the number, and most of the crypto universe is.
    """
    a = abs(v)
    if a >= 1000: return round(v, 2)
    if a >= 1:    return round(v, 4)
    if a >= 0.01: return round(v, 6)
    return round(v, 10)


def _issue(key: str, bias: str, dirn: int, price: float,
           risk: float, atr: float) -> dict:
    """Freeze a new signal at the current price. Levels never change after this."""
    return {
        "key": key,
        "bias": bias,
        "dirn": int(dirn),
        "entry": _r(price),
        "stop": _r(price - dirn * risk),
        "tp1": _r(price + dirn * 2.0 * risk),
        "tp2": _r(price + dirn * 3.0 * risk),
        "risk": _r(risk),
        "atr": _r(atr),
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


def _advance(state: dict, key: str, bias: str, dirn: int, price: float,
             risk: float, atr: float, market_open: bool,
             cooldown_min: float):
    """Advance ONE instrument inside an already-loaded state. No I/O.

    Split out of update() so a caller with twenty instruments does one load and
    one save instead of twenty of each - at a 1s refresh that is the difference
    between a cache and a disk thrash.
    """
    sig = state["signals"].get(key)
    events = []

    # 1. Check an existing signal against ITS OWN fixed levels. Must include
    #    TP1, not only ACTIVE: guarding on ACTIVE alone meant that once target 1
    #    printed, _resolve() was never called again, so TP2 could not register
    #    AND THE STOP COULD NOT TRIGGER.
    if sig and sig.get("status") in (ACTIVE, TP1):
        new_status = _resolve(sig, price)
        if new_status != sig["status"]:
            sig["status"] = new_status
            if new_status in (TP2, STOPPED):
                sig["resolved_at"] = _now().isoformat()
            events.append(new_status)
        if sig["status"] in (TP2, STOPPED):
            state["history"].append(sig)
            state["signals"].pop(key, None)
            sig = None

    # 2. A bias flip closes the old signal and issues a new one. It never edits
    #    the old numbers, so the record shows what actually happened.
    if sig and sig.get("dirn") != int(dirn) and bias != "MIXED":
        sig["status"] = FLIPPED
        sig["resolved_at"] = _now().isoformat()
        state["history"].append(sig)
        state["signals"].pop(key, None)
        sig = None
        events.append(FLIPPED)

    # 3. Issue only with nothing live, the market open, and no cooldown running.
    #    Being stopped is not new information; a bias flip is, so flips skip the
    #    cooldown by having already cleared the slot above.
    if sig is None and market_open:
        if _blocked_by_cooldown(state, key, dirn, cooldown_min):
            return None, events
        sig = _issue(key, bias, dirn, price, risk, atr)
        state["signals"][key] = sig
        events.append("ISSUED")

    # 4. Excursions against the FIXED levels - how far it ran for and against,
    #    because an outcome alone hides how close the other side came.
    if sig and sig.get("status") in (ACTIVE, TP1):
        move = (price - sig["entry"]) * sig["dirn"]
        sig["mfe"] = _r(max(sig.get("mfe", 0.0), move))
        sig["mae"] = _r(min(sig.get("mae", 0.0), move))
        state["signals"][key] = sig

    return sig, events


def _hist_for(state: dict, key: str):
    return [h for h in state.get("history", []) if h.get("key", GOLD) == key][::-1]


def _view(sig, price: float, events: list, state: dict, key: str,
          market_open: bool, cooldown_min: float) -> dict:
    """Fixed levels plus live context. The levels are never recomputed here."""
    if not sig:
        reason = ("market closed - no signal issued" if not market_open
                  else f"cooling off after a stop-out (up to {cooldown_min:.0f}m, "
                       f"or until the bias flips)")
        return {"ok": False, "key": key, "reason": reason, "events": events,
                "history": _hist_for(state, key)}
    issued = datetime.fromisoformat(sig["issued_at"])
    move = (price - sig["entry"]) * sig["dirn"]
    return {
        "ok": True, **sig, "events": events,
        "age_min": (_now() - issued).total_seconds() / 60.0,
        "issued_dt": issued, "live_price": price,
        "move": _r(move),
        "move_r": round(move / sig["risk"], 2) if sig["risk"] else 0.0,
        "move_pct": round(move / sig["entry"] * 100, 2) if sig["entry"] else 0.0,
        "to_stop": _r(abs(price - sig["stop"])),
        "to_tp1": _r(abs(sig["tp1"] - price)),
        "to_tp2": _r(abs(sig["tp2"] - price)),
        "history": _hist_for(state, key),
    }


def update(bias: str, dirn: int, price: float, risk: float, atr: float,
           market_open: bool = True, key: str = GOLD,
           cooldown_min: float = STOP_COOLDOWN_MIN) -> dict:
    """Advance one instrument's latched signal and return it."""
    if not price or price <= 0 or risk <= 0:
        return {"ok": False, "key": key, "reason": "no live price"}
    with _lock:
        state = _load()
        sig, events = _advance(state, key, bias, dirn, price, risk, atr,
                               market_open, cooldown_min)
        _save(state)
        return _view(sig, price, events, state, key, market_open, cooldown_min)


def update_many(items, market_open: bool = True,
                cooldown_min: float = STOP_COOLDOWN_MIN) -> dict:
    """Advance many instruments under ONE load/save.

    Each item: {key, bias, dirn, price, risk, atr}. Returns {key: view}.
    """
    out = {}
    with _lock:
        state = _load()
        for it in items:
            k = it["key"]
            if not it.get("price") or it["price"] <= 0 or not it.get("risk"):
                out[k] = {"ok": False, "key": k, "reason": "no live price"}
                continue
            sig, events = _advance(
                state, k, it["bias"], int(it["dirn"]), float(it["price"]),
                float(it["risk"]), float(it.get("atr", 0.0)),
                market_open, cooldown_min)
            out[k] = _view(sig, float(it["price"]), events, state, k,
                           market_open, cooldown_min)
        _save(state)
    return out


def open_signals(prefix=None) -> dict:
    """Every live signal, optionally filtered by key prefix.

    Needed because a crypto pair can fall out of the top-N while its signal is
    still open. Dropping it from the screen would quietly abandon a position the
    page had already told you about.
    """
    with _lock:
        sigs = _load().get("signals", {})
        return {k: dict(v) for k, v in sigs.items()
                if prefix is None or k.startswith(prefix)}


def reset(key=None) -> None:
    """Close live signals (history kept). None = all. For manual intervention."""
    with _lock:
        state = _load()
        for k in ([key] if key else list(state.get("signals", {}))):
            sig = state["signals"].pop(k, None)
            if sig:
                sig["status"] = FLIPPED
                sig["resolved_at"] = _now().isoformat()
                state["history"].append(sig)
        _save(state)
