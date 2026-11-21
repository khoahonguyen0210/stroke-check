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
STATUS_FIELDS = ["someone_present", "called_911", "destination_hospital"]
ALL_FIELDS = STATUS_FIELDS + FIELDS

NO_MATCH = {
    "en": "Ask the 911 dispatcher.",
    "es": "Pregunte al operador del 911.",
    "vi": "Hãy hỏi nhân viên tổng đài 911.",
}

LINES = {
    "call_911": {"en": "Call 911 now.", "es": "Llame al 911 ahora.", "vi": "Hãy gọi 911 ngay."},
    "follow_paramedics": {"en": "Follow the paramedics' instructions.", "es": "Siga las instrucciones de los paramédicos.",
                          "vi": "Hãy làm theo hướng dẫn của nhân viên cấp cứu."},
    "help_coming": {"en": "Help is on the way. Your contacts have been alerted.",
                    "es": "La ayuda está en camino. Sus contactos fueron avisados.",
                    "vi": "Người giúp đang đến. Người thân của bạn đã được báo."},
}

SIGN_NAMES = {
    "en": {"face": "face droop", "arms": "arm weakness", "speech": "slurred speech"},
    "es": {"face": "cara caída", "arms": "debilidad en el brazo", "speech": "habla arrastrada"},
    "vi": {"face": "méo mặt", "arms": "yếu tay", "speech": "nói ngọng"},
}

SCRIPT_911 = {
    "en": ["I'm calling about {who}. A stroke warning check found possible stroke signs: {signs}.",
           "They were last seen acting normal: {lkn}.",
           "Blood thinners: {bt}.",
           "Location: {loc}."],
    "es": ["Llamo por {who}. Una prueba de alerta de derrame encontró posibles señales: {signs}.",
           "La última vez que estaba normal: {lkn}.",
           "Anticoagulantes: {bt}.",
           "Ubicación: {loc}."],
    "vi": ["Tôi gọi về {who}. Bài kiểm tra cảnh báo đột quỵ thấy dấu hiệu có thể là đột quỵ: {signs}.",
           "Lần cuối thấy người đó bình thường: {lkn}.",
           "Thuốc chống đông máu: {bt}.",
           "Địa điểm: {loc}."],
}
SOMEONE = {"en": "my family member", "es": "mi familiar", "vi": "người nhà tôi"}

STOPWORDS = set("a an the is are am do does did can could should would i we he she they it to of in on for and or my his her their you your with be this that what when how why there any".split())

YES = r"\b(yes|yeah|yep|called|calling|done|they're coming|on the way|sí|si|llamé|ya|rồi|có|đã gọi|đang gọi)\b"
NO = r"\b(no|not yet|haven'?t|nope|nobody|no one|alone|todavía no|nadie|sola?|chưa|không|một mình)\b"


def now():
    return datetime.now().isoformat(timespec="seconds")


def lang(state):
    return state["language"] if state["language"] in LINES["call_911"] else "en"


def answer_is(value, kind):
    text = (value or "").lower()
    if kind == "yes":
        return bool(re.search(YES, text)) and not re.search(r"\b(not yet|haven'?t|todavía no|chưa)\b", text)
    return bool(re.search(NO, text))


def called_911(state):
    return answer_is(state["facts"].get("called_911"), "yes")


def alone(state):
    value = state["facts"].get("someone_present")
    return value is not None and answer_is(value, "no")


COVER = {
    "told_911": r"911",
    "no_food_drink_medicine": r"food|drink|medicine|comida|bebida|medicina|ăn|uống|thuốc",
    "stay_and_unlock": r"door|unlock|puerta|cửa",
    "contacts_alerted": r"contact|alerted|contactos|avisad|người thân",
    "safe_position": r"sit|lie down|siéntese|acuéstese|ngồi|nằm",
}


def mark_covered(state, text):
    for key, pattern in COVER.items():
        if re.search(pattern, text.lower()) and key not in state["covered"]:
            state["covered"].append(key)


def next_step(state):
    covered = state["covered"]
    lkn = state["timer"]["last_known_normal"] or state["facts"].get("last_known_normal")
    if state.get("paramedics_here"):
        return ("the paramedics are with the person. If the family says which hospital they are going to, call "
                "record_fact(field='destination_hospital'). Otherwise speak one short sentence: follow the paramedics' instructions")
    if state.get("no_answer", 0) == 1:
        return "the family did not answer: ask your last question again, in fewer words"
    if state.get("no_answer", 0) >= 2:
        return "still no answer: speak 'help is on the way', call notify_contacts, then finish_turn"
    if "someone_present" not in state["asked"] and "someone_present" not in state["facts"]:
        return "ask(field='someone_present'): Is someone with you right now?"
    if alone(state) and "contacts_alerted" not in covered:
        return "speak: your trusted contacts are being alerted"
    if alone(state) and "safe_position" not in covered:
        return "speak: unlock the door, then sit or lie down somewhere safe"
    if "called_911" not in state["asked"] and not called_911(state):
        return "ask(field='called_911'): have you called 911 yet?"
    if not lkn and "last_known_normal" not in state["asked"]:
        return "ask(field='last_known_normal'): when was the person last seen acting normal?"
    if "no_food_drink_medicine" not in covered:
        return "speak: do not give any food, drink or medicine"
    if not alone(state) and "stay_and_unlock" not in covered:
        return "speak: stay with the person and unlock the door for the paramedics"
    for field in FIELDS[1:]:
        if field not in state["asked"] and field not in state["facts"]:
            simple = " Use a simple yes/no question." if alone(state) else ""
            return f"ask(field='{field}').{simple}"
    return "speak: stay where you are until the paramedics arrive"


def _say(state, text, kind):
    if re.search(r"<\||\w+\s*\{", text or ""):
        return {"ok": False, "error": "text must be plain words for the family, not a tool call."}
    check = safety.check(text)
    if check["ok"]:
        mark_covered(state, text)
        state["said"].append(text)
        state["timeline"].append({"t": now(), "event": f"{kind}: {text}"})
        return {"ok": True, "said": text}
    line = safety.replacement(check["reason"], state["language"])
    mark_covered(state, line)
    state["said"].append(line)
    state["blocked"].append({"t": now(), "tool": kind, "text": text, "reason": check["reason"], "replaced_with": line})
    state["timeline"].append({"t": now(), "event": f"{kind} (safety replaced): {line}"})
    return {"ok": True, "said": line, "blocked": check["reason"], "replaced_with": line,
            "note": "Your text was blocked by the safety layer and replaced."}


def speak(state, text):
    if state.get("asked_this_turn"):
        return {"ok": False, "error": "You already asked a question this turn. Call finish_turn and wait for the answer."}
    return _say(state, text, "speak")


def ask(state, field, question):
    if field not in ALL_FIELDS:
        return {"ok": False, "error": f"Unknown field. Use one of: {', '.join(ALL_FIELDS)}"}
    if state.get("asked_this_turn"):
        return {"ok": False, "error": "Only one question per turn. Call finish_turn now."}
    if state["facts"].get(field) and not state.get("no_answer"):
        return {"ok": False, "error": f"{field} is already recorded. Follow STATE.next_step instead."}
    if field not in state["asked"]:
        state["asked"].append(field)
    state["asked_this_turn"] = True
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
    if field not in ALL_FIELDS:
        return {"ok": False, "error": f"Unknown field. Use one of: {', '.join(ALL_FIELDS)}"}
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
        return {"ok": False, "blocked": check["reason"], "error": f"Blocked by safety layer ({check['reason']}). Send only facts, e.g. '911 has been called.'"}
    try:
        from backend import alerts

        sent = alerts.notify(state["user"], update)
    except Exception as error:
        sent = [{"ok": False, "error": str(error)}]
    state["notified"].append(update)
    state["timeline"].append({"t": now(), "event": f"notified contacts: {update}"})
    return {"ok": True, "sent": sent}


def clean(value):
    return (value or "").strip().rstrip(".!?;, ")


def write_911_script(state):
    lkn = clean(state["timer"]["last_known_normal"] or state["facts"].get("last_known_normal"))
    if not lkn:
        return {"ok": False, "error": "Ask for last_known_normal first."}
    l = lang(state)
    names = SIGN_NAMES[l]
    signs = [names.get(k, k) for k, v in state["result"].get("signs", {}).items() if v.get("flag")]
    t = SCRIPT_911[l]
    who = state["result"].get("name") or (state["user"] if state["user"] not in ("user", "eval") else SOMEONE[l])
    lines = [t[0].format(who=who, signs=", ".join(signs) or "-"), t[1].format(lkn=lkn)]
    if state["facts"].get("blood_thinners"):
        lines.append(t[2].format(bt=clean(state["facts"]["blood_thinners"])))
    if state.get("location"):
        lines.append(t[3].format(loc=state["location"]))
    script = " ".join(lines)
    if script != state.get("script_911"):
        state["script_911"] = script
        state["timeline"].append({"t": now(), "event": "911 script written"})
    return {"ok": True, "script_911": script}


def build_handoff(state):
    signs = state["result"].get("signs", {})
    flagged = {k: v.get("p") for k, v in signs.items() if v.get("flag")}
    facts = {f: state["facts"].get(f) for f in FIELDS}
    lines = [
        "PARAMEDIC HANDOFF (automated warning, not a diagnosis)",
        f"Check time: {state['started_at']}",
        f"Last known normal (family's words): {facts['last_known_normal'] or 'not given'}",
        "Signs flagged by Stroke Check: " + (", ".join(f"{k} (p={v:.2f})" if isinstance(v, (int, float)) else k for k, v in flagged.items()) or "none"),
    ]
    lines += [f"{f.replace('_', ' ').capitalize()}: {v if v is not None else 'not asked/answered'}" for f, v in facts.items() if f != "last_known_normal"]
    if state.get("removed_facts"):
        lines.append("Removed by the Nemotron Super check (no supporting line): " + ", ".join(f"{r['field']}={r['value']}" for r in state["removed_facts"]))
    lines.append("Timeline:")
    lines += [f"  {e['t'][11:]} {e['event']}" for e in state["timeline"]]
    return {
        "time_noticed": facts["last_known_normal"],
        "check_time": state["started_at"],
        "signs": flagged,
        "facts": facts,
        "someone_present": state["facts"].get("someone_present"),
        "called_911": state["facts"].get("called_911"),
        "timeline": state["timeline"],
        "verified": state.get("verified"),
        "removed_facts": state.get("removed_facts", []),
        "text": "\n".join(lines),
    }


def spoken_summary(state):
    h = build_handoff(state)
    names = SIGN_NAMES["en"]
    parts = ["Paramedic summary. This is an automated warning, not a diagnosis.",
             "Possible stroke signs: " + (", ".join(names.get(k, k) for k in h["signs"]) or "none") + "."]
    if h["facts"]["last_known_normal"]:
        parts.append(f"Last seen normal: {clean(h['facts']['last_known_normal'])}.")
    for field in FIELDS[1:]:
        if h["facts"].get(field):
            parts.append(f"{field.replace('_', ' ').capitalize()}: {clean(h['facts'][field])}.")
    return " ".join(parts)


def write_handoff(state):
    if not state["handoff_written"]:
        state["timeline"].append({"t": now(), "event": "handoff written"})
    state["handoff_written"] = True
    return {"ok": True, "handoff": build_handoff(state)["text"],
            "next": "Handoff is ready. The app will read the summary out loud. Call finish_turn."}


def finish_turn(state):
    if len(state["said"]) <= state.get("turn_said_start", 0) and not state.get("paramedics_here"):
        return {"ok": False, "error": "You have not said anything to the family this turn. Call speak or ask first, then finish_turn."}
    return {"ok": True}


RUN = {
    "speak": speak,
    "ask": ask,
    "record_fact": record_fact,
    "start_timer": start_timer,
    "lookup_guidance": lookup_guidance,
    "notify_contacts": notify_contacts,
    "write_911_script": write_911_script,
    "write_handoff": write_handoff,
    "finish_turn": finish_turn,
}


def _tool(name, description, props=None, required=None):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": props or {}, "required": required or list((props or {}).keys())}}}


TOOLS = [
    _tool("speak", "Say one short, calm sentence to the family, in their language.", {"text": {"type": "string"}}),
    _tool("ask", "Ask the family one question for one field. At most one question per turn.",
          {"field": {"type": "string", "enum": ALL_FIELDS}, "question": {"type": "string"}}),
    _tool("record_fact", "Save a fact the family just said. value must be copied word for word from their message. Never infer.",
          {"field": {"type": "string", "enum": ALL_FIELDS}, "value": {"type": "string"}}),
    _tool("start_timer", "Start or correct the last-known-normal timer, using the family's exact words for the time.",
          {"last_known_normal": {"type": "string"}}),
    _tool("lookup_guidance", "Find approved guidance to answer a family question. Write the question in English. Answer only with the returned text.",
          {"question": {"type": "string"}}),
    _tool("notify_contacts", "Send a short factual update to the user's trusted contacts, e.g. '911 has been called.'",
          {"update": {"type": "string"}}),
    _tool("write_911_script", "Write the 3-4 sentences the caller reads to the 911 dispatcher, from recorded facts only. Call it once last_known_normal is known."),
    _tool("write_handoff", "Build the paramedic handoff from everything recorded so far. Call it when the paramedics are here."),
    _tool("finish_turn", "End your turn and wait for the family to reply."),
]
