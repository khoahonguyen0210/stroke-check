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

PRELIMINARY = {
    "someone_present": ("Is someone with you right now?", "¿Hay alguien con usted ahora?", "Bây giờ có ai ở bên bạn không?"),
    "called_911": ("Has anyone called 911 yet?", "¿Ya llamó alguien al 911?", "Đã có ai gọi 911 chưa?"),
    "last_known_normal": ("When were they last seen acting as usual?", "¿Cuándo fue la última vez que se comportó como de costumbre?", "Lần cuối người đó được thấy vẫn bình thường là khi nào?"),
    "blood_thinners": ("Do they take blood thinners? If you know, tell me the name.", "¿Toma anticoagulantes? Si sabe el nombre, dígamelo.", "Họ có dùng thuốc chống đông máu không? Hãy cho biết tên nếu bạn biết."),
    "current_medications": ("What other medications do they take?", "¿Qué otros medicamentos toma?", "Họ dùng những thuốc nào khác?"),
    "allergies": ("Do they have any medication allergies?", "¿Tiene alergias a algún medicamento?", "Người đó có dị ứng với thuốc nào không?"),
    "recent_fall_or_head_injury": ("Have they recently fallen or hit their head?", "¿Se ha caído o golpeado la cabeza recientemente?", "Gần đây người đó có bị ngã hoặc va đập đầu không?"),
    "diabetes": ("Do they have diabetes?", "¿Tiene diabetes?", "Người đó có bị tiểu đường không?"),
    "seizure": ("Have they had a seizure during this episode?", "¿Ha tenido una convulsión durante este episodio?", "Người đó có bị co giật trong lần này không?"),
}

DISPATCH_NUMBER = r"(?:9[ -]?1[ -]?1|nine one one)"
CALL_NEGATIVE = r"\b(?:not|never|haven['’]?t|hasn['’]?t|didn['’]?t|no one|nobody|chưa|không)\b|\bno (?:he |she |han |ha )?llam"
CALL_CONFIRMED = re.compile(
    rf"\b(?:called|dialed|llam(?:é|ó|amos|aron)|đã gọi|gọi)\s+(?:al\s+)?{DISPATCH_NUMBER}\b"
    rf"|\b{DISPATCH_NUMBER}(?:[’']s been|\s+(?:has (?:already )?been |was (?:already )?|is |ya fue )?)\s*(?:called|contacted|llamado)\b"
    rf"|\b(?:llam(?:é|ó|amos|aron) ya al|đã gọi đến)\s+{DISPATCH_NUMBER}\b"
    rf"|\b(?:đã gọi\s+{DISPATCH_NUMBER}|gọi\s+{DISPATCH_NUMBER}\s+rồi)\b",
    re.I,
)
CALL_REMINDER = re.compile(
    rf"\b(?:call|dial|phone|ring|contact|llam(?:a|e|en)|gọi)\s+(?:(?:al|the)\s+)?(?:{DISPATCH_NUMBER}|emergency services)\b"
    rf"|\bmake sure [^.]*called {DISPATCH_NUMBER}\b", re.I,
)
SHORT_YES = r"(?:yes|yes indeed|yeah|yep|sí|si|có|vâng|rồi)"
SHORT_NO = r"(?:no|nope|none|ninguna?|không)"
UNKNOWN = r"(?:skip(?: this)?|(?:i )?don['’]?t know|(?:i(?:['’]m| am) )?not sure|unknown|no sé|không biết)"

NO_MATCH = {
    "en": "Ask the 911 dispatcher.",
    "es": "Pregunte al operador del 911.",
    "vi": "Hãy hỏi nhân viên tổng đài 911.",
}

CARE_ANSWERS = {
    "call_911": ("Tell the dispatcher which signs you saw and when they started.", "Dígale al operador qué señales vio y cuándo empezaron.", "Hãy báo cho tổng đài biết những dấu hiệu bạn thấy và thời điểm bắt đầu."),
    "symptoms_go_away": ("These symptoms still need urgent care if they improve. Tell the dispatcher what changed.", "Aunque mejoren los síntomas, sigue siendo urgente. Dígale al operador qué cambió.", "Dù triệu chứng giảm, người đó vẫn cần được cấp cứu. Hãy báo cho tổng đài biết điều gì đã thay đổi."),
    "note_the_time": ("Tell the dispatcher when the signs started or when the person was last seen acting as usual.", "Dígale al operador cuándo empezaron las señales o cuándo vio a la persona comportarse como siempre por última vez.", "Hãy báo cho tổng đài khi dấu hiệu bắt đầu hoặc lần cuối người đó vẫn bình thường."),
    "do_not_drive": ("Do not drive to the hospital. The ambulance team can begin care on the way.", "No los lleve al hospital en carro. El equipo de la ambulancia puede comenzar la atención en el camino.", "Đừng tự lái xe đưa người đó đến bệnh viện. Đội cấp cứu có thể chăm sóc họ trên đường đi."),
    "hospital_questions": ("Tell paramedics what changed, when the person was last seen as usual, and which medications they take, if known.", "Dígale a los paramédicos qué cambió, cuándo vio a la persona normal por última vez y qué medicamentos toma, si lo sabe.", "Hãy cho nhân viên cấp cứu biết điều gì đã thay đổi, lần cuối người đó còn bình thường và danh sách thuốc nếu biết."),
    "food_drink_medicine": ("Do not give food, drink, or medicine, including aspirin. Follow the dispatcher’s instructions.", "No le dé alimentos, bebidas ni medicamentos, incluida la aspirina. Siga las instrucciones del operador.", "Không cho người đó ăn, uống hay dùng thuốc, kể cả aspirin. Hãy làm theo hướng dẫn của tổng đài."),
    "while_waiting": ("Stay with the person and follow the dispatcher’s instructions. Have their medication list ready if you know it.", "Quédese con la persona y siga las instrucciones del operador. Tenga a mano su lista de medicamentos si la conoce.", "Hãy ở bên người đó và làm theo hướng dẫn của tổng đài. Chuẩn bị danh sách thuốc nếu bạn biết."),
    "other_symptoms": ("Tell the dispatcher about any sudden change in speech, vision, balance, or strength.", "Dígale al operador si nota cambios repentinos en el habla, la visión, el equilibrio o la fuerza.", "Hãy báo cho tổng đài nếu bạn nhận thấy thay đổi đột ngột về lời nói, thị lực, thăng bằng hoặc sức lực."),
}
CARE_ANSWERS["be_fast"] = CARE_ANSWERS["other_symptoms"]
CARE_ALONE = ("Follow the dispatcher’s instructions while help is on the way.", "Siga las instrucciones del operador mientras llega la ayuda.", "Hãy làm theo hướng dẫn của tổng đài trong khi chờ người đến giúp.")
CARE_FOLLOW_UP = r"^(?:why(?: is that)?|why (?:not|shouldn['’]?t (?:i|we))|can you explain(?: that)?|¿?por qué|tại sao)[?.!]*$"
CARE_IMPROVED = r"\b(?:feel(?:s|ing)? better|symptoms (?:stopped|are gone)|seems? better|mejor|đỡ rồi)\b"
CARE_DRIVE_REASON = ("Paramedics can start care during transport and alert the hospital before arrival.", "Los paramédicos pueden empezar a atenderlo durante el traslado y avisar al hospital antes de llegar.", "Nhân viên cấp cứu có thể chăm sóc người đó trên đường và báo trước cho bệnh viện.")
CARE_WORRY = r"\b(?:scared|worried|panicking|afraid|asustad[oa]|miedo|lo lắng|sợ)\b"

LINES = {
    "call_911": {"en": "Call 911 now.", "es": "Llame al 911 ahora.", "vi": "Hãy gọi 911 ngay."},
    "follow_dispatcher": {"en": "Follow the dispatcher’s instructions.", "es": "Siga las instrucciones del operador.", "vi": "Hãy làm theo hướng dẫn của nhân viên tổng đài."},
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


def confirmed_call_statement(value):
    for clause in re.split(r"[.!?,;]+|\bbut\b", value, flags=re.I):
        if re.match(r"^\s*(have you|has anyone|did|should|can|has 911|was 911)\b", clause, re.I):
            continue
        if CALL_CONFIRMED.search(clause) and not re.search(CALL_NEGATIVE, clause, re.I):
            return True
    return False


def confirmed_call_answer(value):
    answer = value.lower().strip().rstrip(".!")
    if confirmed_call_statement(value):
        return True
    if re.search(CALL_NEGATIVE, value, re.I):
        return False
    return bool(
        re.match(rf"^{SHORT_YES}(?:[,\s]|$)", answer)
        or re.fullmatch(r"(?:we |i |someone |they )?(?:already )?called(?: already)?|(?:they['’]re|the ambulance is) on the way|ya llamé|đã gọi", answer)
    )


def called_911(state):
    return state.get("call_confirmed", False) or confirmed_call_answer(state["facts"].get("called_911") or "")


def pending_preliminary(state):
    if state.get("paramedics_here"):
        return None
    skipped = state.get("preliminary_skipped", [])
    return next((
        field for field in PRELIMINARY
        if field not in skipped and (
            not called_911(state) if field == "called_911" else not state["facts"].get(field)
        )
    ), None)


def preliminary_prompt(state, field):
    return PRELIMINARY[field][("en", "es", "vi").index(lang(state))]


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
    field = pending_preliminary(state)
    if field:
        return f"ask(field='{field}'): {preliminary_prompt(state, field)}"
    if "no_food_drink_medicine" not in covered:
        return "speak: do not give any food, drink or medicine"
    if "stay_and_unlock" not in covered:
        return "speak: stay with the person and unlock the door for the paramedics"
    return "speak: stay where you are until the paramedics arrive"


def _say(state, text, kind):
    if called_911(state) and CALL_REMINDER.search(text or ""):
        return {"ok": False, "error": "The call to 911 is confirmed. Do not repeat the call instruction."}
    if re.search(r"<\||\w+\s*\{", text or ""):
        return {"ok": False, "error": "text must be plain words for the family, not a tool call."}
    check = safety.check(text)
    if check["ok"]:
        mark_covered(state, text)
        state["said"].append(text)
        state["timeline"].append({"t": now(), "event": f"{kind}: {text}"})
        return {"ok": True, "said": text}
    line = safety.replacement(check["reason"], state["language"])
    if called_911(state) and CALL_REMINDER.search(line):
        line = LINES["follow_dispatcher"][lang(state)]
    mark_covered(state, line)
    state["said"].append(line)
    state["blocked"].append({"t": now(), "tool": kind, "text": text, "reason": check["reason"], "replaced_with": line})
    state["timeline"].append({"t": now(), "event": f"{kind} (safety replaced): {line}"})
    return {"ok": True, "said": line, "blocked": check["reason"], "replaced_with": line,
            "note": "Your text was blocked by the safety layer and replaced."}


def speak(state, text):
    if state.get("asked_this_turn"):
        return {"ok": False, "error": "You already asked a question this turn. Call finish_turn and wait for the answer."}
    if state.get("care_question") and not state.get("guided"):
        guidance = state.get("turn_guidance")
        if guidance is None:
            return {"ok": False, "error": "Use lookup_guidance first for this care question."}
        return speak_guidance(state, guidance)
    return _say(state, text, "speak")


def ask(state, field, question):
    if field not in ALL_FIELDS:
        return {"ok": False, "error": f"Unknown field. Use one of: {', '.join(ALL_FIELDS)}"}
    if state.get("asked_this_turn"):
        return {"ok": False, "error": "Only one question per turn. Call finish_turn now."}
    if not state.get("guided"):
        pending = pending_preliminary(state)
        if state.get("care_question"):
            return {"ok": False, "error": "Answer the family's care question first. Resume the handoff on the next turn."}
        if pending and pending != field:
            return {"ok": False, "error": f"Ask the outstanding question for {pending} first."}
        if field == "called_911" and called_911(state):
            return {"ok": False, "error": "The call to 911 is already confirmed."}
        if field in PRELIMINARY:
            question = preliminary_prompt(state, field)
    if state["facts"].get(field) and field != "called_911" and not state.get("no_answer"):
        return {"ok": False, "error": f"{field} is already recorded. Follow STATE.next_step instead."}
    result = _say(state, question, f"ask {field}")
    if result.get("ok") and not result.get("blocked"):
        if field not in state["asked"]:
            state["asked"].append(field)
        state["pending_question"] = field
        state["asked_this_turn"] = True
    return result


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
    if field == "called_911":
        if called_911(state):
            return {"ok": True, "field": field, "value": state["facts"][field]}
        latest = state["family"][-1] if state["family"] else ""
        if confirmed_call_statement(latest) or (
            state.get("pending_question") == field and confirmed_call_answer(latest)
        ):
            state["call_confirmed"] = True
        else:
            return {"ok": False, "error": "The family has not confirmed a completed call to 911."}
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


def guidance_match(question, sections):
    q = [word for word in _norm(question) if word not in STOPWORDS]

    def score(section):
        keys = [key.strip() for key in section["keywords"].split(",")]
        keywords = sum(2 for key in keys if key and key in " ".join(q) and (" " in key or key in q))
        return keywords + sum(word in _norm(section["text"]) for word in q)

    best = max(sections, key=score, default=None)
    return best if best and score(best) >= 2 else None


def lookup_guidance(state, question):
    sections = guidance_sections()
    latest = state["family"][-1] if state.get("care_question") and state["family"] else question
    best = None
    follow_up = bool(re.fullmatch(CARE_FOLLOW_UP, latest.strip(), re.I))
    if follow_up:
        previous = state.get("last_guidance_section")
        best = next((section for section in sections if section["name"] == previous), None)
    if re.search(CARE_IMPROVED, latest, re.I):
        best = next((section for section in sections if section["name"] == "symptoms_go_away"), None)
    best = best or guidance_match(latest, sections)
    if not best and question != latest:
        best = guidance_match(question, sections)
    if not best and re.search(CARE_WORRY, latest, re.I):
        best = next((section for section in sections if section["name"] == "while_waiting"), None)
    name = best["name"] if best else None
    state["timeline"].append({"t": now(), "event": f"lookup_guidance: {latest} -> {name or 'no match'}"})
    result = (
        {"ok": True, "section": name, "text": best["text"], "source": best["source"], "follow_up": follow_up}
        if best else {"ok": True, "section": None, "text": NO_MATCH[lang(state)]}
    )
    state["turn_guidance"] = result
    if name:
        state["last_guidance_section"] = name
    return result


def speak_guidance(state, guidance):
    section = guidance["section"]
    language = lang(state)
    index = ("en", "es", "vi").index(language)
    answers = CARE_ANSWERS.get(section)
    if section == "while_waiting" and alone(state):
        answers = CARE_ALONE
    if section == "do_not_drive" and guidance.get("follow_up"):
        answers = CARE_DRIVE_REASON
    text = answers[index] if answers else NO_MATCH[language]
    last = state["family"][-1] if state["family"] else ""
    if re.search(CARE_WORRY, last, re.I):
        text = ("I hear you. ", "Le entiendo. ", "Tôi hiểu. ")[index] + text
    return _say(state, text, "speak")


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


def write_handoff(state):
    if not state["handoff_written"]:
        state["timeline"].append({"t": now(), "event": "handoff written"})
    state["handoff_written"] = True
    return {"ok": True, "handoff": build_handoff(state)["text"],
            "next": "Handoff is ready to display. Call finish_turn."}


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
