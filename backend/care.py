import json
from uuid import uuid4

from backend import agent, agent_tools

CALL_911 = "Call 911 now and follow the dispatcher's instructions."
QUESTIONS = {
    "last_known_normal": "When was the person last seen acting as usual?",
    "blood_thinners": "Do they take any blood thinners? If you know, enter the name.",
    "recent_fall_or_head_injury": "Have they recently fallen or hit their head?",
    "current_medications": "What other medications do they take?",
    "allergies": "Do you know of any medication allergies?",
}


def action(tool, args, result):
    return {"tool": tool, "args": args, "result": result, "t": agent_tools.now()}


def speak(state, line, actions):
    detail = agent_tools.speak(state, line)
    actions.append(action("speak", {"text": line}, detail))
    return detail.get("said") or CALL_911


def respond(state, actions, line):
    state["conversation"].append({"role": "agent", "text": line})
    response = agent.respond(state, actions, line)
    field = state.get("pending_question")
    response["question"] = {
        "field": field, "prompt": QUESTIONS[field], "number": list(QUESTIONS).index(field) + 1, "total": len(QUESTIONS),
    } if field else None
    return response


def ask_next(state, actions):
    field = next((name for name in QUESTIONS if name not in state["asked"]), None)
    state["pending_question"] = field
    if field is None:
        actions.append(action("write_handoff", {}, agent_tools.write_handoff(state)))
        line = speak(state, "The handoff is ready to share with the paramedics.", actions)
    else:
        prompt = QUESTIONS[field]
        detail = agent_tools.ask(state, field, prompt)
        actions.append(action("ask", {"field": field}, detail))
        line = detail.get("said") or CALL_911
    return respond(state, actions, line)


def start(result):
    state = agent.new_state(result, "en")
    state["session"] += f"-{uuid4().hex[:8]}"
    state["guided"] = True
    actions = []
    speak(state, CALL_911, actions)
    return ask_next(state, actions)


def answer(state, text, skip):
    field = state["pending_question"]
    actions = []
    if skip:
        state["timeline"].append({"t": agent_tools.now(), "event": f"{field} skipped by the family"})
    else:
        state["family"].append(text)
        state["conversation"].append({"role": "family", "text": text})
        if field == "last_known_normal":
            actions.append(action("start_timer", {"last_known_normal": text}, agent_tools.start_timer(state, text)))
        else:
            actions.append(action("record_fact", {"field": field, "value": text}, agent_tools.record_fact(state, field, text)))
    return ask_next(state, actions)


def question(state, text):
    sections = {section["name"]: section["text"] for section in agent_tools.guidance_sections()}
    local = agent_tools.lookup_guidance(state, text)
    topic = local["section"]
    source = "local"
    try:
        client, model = agent.client_for("bolt")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Choose one topic from the provided stroke first-aid guidance for the user's question. Reply with only the topic ID, or none if no topic applies. A question about driving to care selects do_not_drive. Do not give advice yourself."},
                {"role": "user", "content": json.dumps({"question": text, "previous_care_question": state.get("previous_care_question"), "topics": sections})},
            ],
            temperature=0,
            max_tokens=30,
            timeout=4,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        selected = (response.choices[0].message.content or "").strip().lower()
        if selected in sections or selected == "none":
            topic = selected if selected in sections else None
            source = "nemotron"
    except Exception:
        pass
    guidance = sections.get(topic, agent_tools.NO_MATCH["en"])
    actions = [action("lookup_guidance", {"question": text}, {"ok": True, "section": topic, "source": source})]
    state["conversation"].append({"role": "family", "text": text})
    state["previous_care_question"] = {"text": text, "topic": topic}
    return respond(state, actions, speak(state, guidance, actions))
