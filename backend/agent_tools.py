import re
from datetime import datetime
from pathlib import Path

from backend import safety

GUIDANCE = Path(__file__).with_name("guidance.md")

FIELDS = [
    "last_known_normal",
    "blood_thinners",
    "recent_fall_or_head_injury",
    "diabetes",
    "seizure",
    "current_medications",
    "allergies",
]

NO_MATCH = {
    "en": "Ask the 911 dispatcher.",
    "es": "Pregunte al operador del 911.",
    "vi": "Hãy hỏi nhân viên tổng đài 911.",
}

STOPWORDS = set("a an the is are am do does did can could should would i we he she they it to of in on for and or my his her their you your with be this that what when how why there any".split())


def now():
    return datetime.now().isoformat(timespec="seconds")


def _say(state, text, kind):
    if re.search(r"<\||\w+\s*\{", text or ""):
        return {"ok": False, "error": "text must be plain words for the family, not a tool call."}
    check = safety.check(text)
    if check["ok"]:
        state["said"].append(text)
        state["timeline"].append({"t": now(), "event": f"{kind}: {text}"})
        return {"ok": True, "said": text}
    line = safety.replacement(check["reason"], state["language"])
    state["said"].append(line)
    state["blocked"].append({"t": now(), "tool": kind, "text": text, "reason": check["reason"], "replaced_with": line})
    state["timeline"].append({"t": now(), "event": f"{kind} (safety replaced): {line}"})
    return {"ok": True, "said": line, "blocked": check["reason"], "note": "Your text was blocked by the safety layer and replaced."}


def speak(state, text):
    return _say(state, text, "speak")


def ask(state, field, question):
    if field not in FIELDS:
        return {"ok": False, "error": f"Unknown field. Use one of: {', '.join(FIELDS)}"}
    if field not in state["asked"]:
        state["asked"].append(field)
    return _say(state, question, f"ask {field}")


def _norm(text):
    return re.sub(r"[^\w\s:]", " ", (text or "").lower()).split()


def _said_by_family(state, value):
    words = _norm(value)
    if not words:
        return False
    for message in state["family"]:
        said = _norm(message)
        for i in range(len(said) - len(words) + 1):
            if said[i:i + len(words)] == words:
                return True
    return False


def record_fact(state, field, value):
    if field not in FIELDS:
        return {"ok": False, "error": f"Unknown field. Use one of: {', '.join(FIELDS)}"}
    if not _said_by_family(state, value):
        return {"ok": False, "error": "Value must be the family's exact words, copied from their message. Do not infer or rephrase."}
    state["facts"][field] = value
    state["timeline"].append({"t": now(), "event": f"recorded {field}: {value}"})
    return {"ok": True, "field": field, "value": value}


def start_timer(state, last_known_normal):
    if not _said_by_family(state, last_known_normal):
        return {"ok": False, "error": "Use the family's exact words for the time."}
    corrected = state["timer"]["last_known_normal"] is not None
    state["timer"] = {"last_known_normal": last_known_normal, "set_at": now()}
    state["facts"]["last_known_normal"] = last_known_normal
    state["timeline"].append({"t": now(), "event": f"timer {'corrected' if corrected else 'started'}: last known normal {last_known_normal}"})
    return {"ok": True, "last_known_normal": last_known_normal, "corrected": corrected}


def guidance_sections():
    sections = []
    for block in GUIDANCE.read_text(encoding="utf-8").split("\n## ")[1:]:
        lines = block.strip().splitlines()
        keywords = next((l[len("Keywords:"):] for l in lines if l.startswith("Keywords:")), "")
        source = next((l for l in lines if l.startswith("Source:")), "")
        body = " ".join(l for l in lines[1:] if not l.startswith(("Keywords:", "Source:")))
        sections.append({"name": lines[0].strip(), "keywords": keywords.lower(), "text": body.strip(), "source": source})
    return sections


def lookup_guidance(state, question):
    q = [w for w in _norm(question) if w not in STOPWORDS]
    best, best_score = None, 0
    for s in guidance_sections():
        keys = [k.strip() for k in s["keywords"].split(",")]
        score = sum(2 for k in keys if k and k in " ".join(q) and (" " in k or k in q))
        score += sum(1 for w in q if w in _norm(s["text"]))
        if score > best_score:
            best, best_score = s, score
    state["timeline"].append({"t": now(), "event": f"lookup_guidance: {question} -> {best['name'] if best and best_score >= 2 else 'no match'}"})
    if not best or best_score < 2:
        return {"ok": True, "section": None, "text": NO_MATCH.get(state["language"], NO_MATCH["en"])}
    return {"ok": True, "section": best["name"], "text": best["text"], "source": best["source"]}


def notify_contacts(state, update):
    if update in state.setdefault("notified", []):
        return {"ok": True, "note": "Already sent. Do not send it again."}
    check = safety.check(update)
    if not check["ok"]:
        state["blocked"].append({"t": now(), "tool": "notify_contacts", "text": update, "reason": check["reason"], "replaced_with": None})
        return {"ok": False, "error": f"Blocked by safety layer ({check['reason']}). Send only facts, e.g. '911 has been called.'"}
    try:
        from backend import alerts

        sent = alerts.notify(state["user"], update)
    except Exception as error:
        sent = [{"ok": False, "error": str(error)}]
    state["notified"].append(update)
    state["timeline"].append({"t": now(), "event": f"notified contacts: {update}"})
    return {"ok": True, "sent": sent}


def build_handoff(state):
    signs = state["result"].get("signs", {})
    flagged = {k: v.get("p") for k, v in signs.items() if v.get("flag")}
    facts = {f: state["facts"].get(f) for f in FIELDS}
    lines = [
        "PARAMEDIC HANDOFF (automated warning, not a diagnosis)",
        f"Check time: {state['started_at']}",
        f"Last known normal (family's words): {state['timer']['last_known_normal'] or state['facts'].get('last_known_normal') or 'not given'}",
        "Signs flagged by Stroke Check: " + (", ".join(f"{k} (p={v:.2f})" if isinstance(v, (int, float)) else k for k, v in flagged.items()) or "none"),
    ]
    lines += [f"{f.replace('_', ' ').capitalize()}: {v if v is not None else 'not asked/answered'}" for f, v in facts.items() if f != "last_known_normal"]
    lines.append("Timeline:")
    lines += [f"  {e['t'][11:]} {e['event']}" for e in state["timeline"]]
    return {
        "time_noticed": state["timer"]["last_known_normal"] or state["facts"].get("last_known_normal"),
        "check_time": state["started_at"],
        "signs": flagged,
        "facts": facts,
        "timeline": state["timeline"],
        "text": "\n".join(lines),
    }


def write_handoff(state):
    if not state["handoff_written"]:
        state["timeline"].append({"t": now(), "event": "handoff written"})
    state["handoff_written"] = True
    return {"ok": True, "handoff": build_handoff(state)["text"],
            "next": "Handoff is ready. Now speak one short sentence telling the family to show it to the paramedics, then call finish_turn."}


def finish_turn(state):
    if len(state["said"]) <= state.get("turn_said_start", 0):
        return {"ok": False, "error": "You have not said anything to the family this turn. Call speak or ask first, then finish_turn."}
    return {"ok": True}


RUN = {
    "speak": speak,
    "ask": ask,
    "record_fact": record_fact,
    "start_timer": start_timer,
    "lookup_guidance": lookup_guidance,
    "notify_contacts": notify_contacts,
    "write_handoff": write_handoff,
    "finish_turn": finish_turn,
}


def _tool(name, description, props=None, required=None):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": props or {}, "required": required or list((props or {}).keys())}}}


TOOLS = [
    _tool("speak", "Say one short, calm sentence to the family, in their language.", {"text": {"type": "string"}}),
    _tool("ask", "Ask the family for one handoff fact. One question at a time.",
          {"field": {"type": "string", "enum": FIELDS}, "question": {"type": "string"}}),
    _tool("record_fact", "Save a fact the family just said. value must be copied word for word from their message. Never infer.",
          {"field": {"type": "string", "enum": FIELDS}, "value": {"type": "string"}}),
    _tool("start_timer", "Start or correct the last-known-normal timer, using the family's exact words for the time.",
          {"last_known_normal": {"type": "string"}}),
    _tool("lookup_guidance", "Find approved guidance to answer a family question. Write the question in English. Answer only with the returned text.",
          {"question": {"type": "string"}}),
    _tool("notify_contacts", "Send a short factual update to the user's trusted contacts, e.g. '911 has been called.'",
          {"update": {"type": "string"}}),
    _tool("write_handoff", "Build the paramedic handoff from everything recorded so far."),
    _tool("finish_turn", "End your turn and wait for the family to reply."),
]
