FIELDS = [
    "last_known_normal",
    "blood_thinners",
    "recent_fall_or_head_injury",
    "diabetes",
    "seizure",
    "current_medications",
    "allergies",
]


def speak(state, text):
    return {"ok": True, "text": text}


def ask(state, field, question):
    return {"ok": True, "field": field, "question": question}


def record_fact(state, field, value):
    return {"ok": True, "field": field, "value": value}


def start_timer(state, last_known_normal):
    return {"ok": True, "last_known_normal": last_known_normal}


def lookup_guidance(state, question):
    return {"ok": True, "section": None, "text": "Ask the 911 dispatcher."}


def notify_contacts(state, update):
    return {"ok": True, "sent": []}


def write_handoff(state):
    return {"ok": True, "handoff": {}}


def finish_turn(state):
    return {"ok": True}
