from datetime import datetime


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _handoff(result, facts):
    signs = [name for name, s in result.get("signs", {}).items() if s.get("flag")]
    return {
        "time_noticed": None,
        "signs": {name: result["signs"][name].get("p") for name in signs},
        "facts": facts,
        "timeline": [],
        "text": "Possible stroke signs detected: " + (", ".join(signs) or "none") + ". 911 called. This is an automated warning, not a diagnosis.",
    }


def start(result, language="en"):
    state = {"session": _now(), "language": language, "result": result, "facts": {}, "timer_started_at": _now(), "history": []}
    return {
        "say": "Possible stroke signs. Call 911 now. When did you last see them acting normal?",
        "actions": [{"tool": "ask", "args": {"field": "last_known_normal"}, "t": _now()}],
        "handoff": _handoff(result, {}),
        "state": state,
    }


def reply(text, state, language="en"):
    state = dict(state)
    state["history"] = state.get("history", []) + [{"role": "family", "text": text}]
    return {
        "say": "Thank you. Stay with them and keep the door unlocked for the paramedics.",
        "actions": [{"tool": "record_fact", "args": {"field": "last_known_normal", "value": text}, "t": _now()}],
        "handoff": _handoff(state.get("result", {}), {"last_known_normal": text}),
        "state": state,
    }
