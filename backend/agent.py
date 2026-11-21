import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path

from backend import agent_tools, safety

ROOT = Path(__file__).resolve().parents[1]
HANDOFFS = ROOT / "data" / "handoffs"
MAX_STEPS = 6
TIMEOUT_S = 6
VERIFY_TIMEOUT_S = 8

MODELS = {
    "bolt": {"base_url": "https://integrate.api.nvidia.com/v1", "key": "NVIDIA_API_KEY", "model": "nvidia/nemotron-3.5-bolt-e4b"},
    "super": {"base_url": "https://integrate.api.nvidia.com/v1", "key": "NVIDIA_API_KEY", "model": "nvidia/nemotron-3.5-super-vl-120b-a12b"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "key": "GEMINI_API_KEY", "model": "gemini-3.5-flash"},
}

LANGUAGES = {"en": "English", "es": "Spanish", "vi": "Vietnamese"}
NO_ANSWER = r"^\(?\s*no answer\s*\)?$"
PARAMEDICS = r"^\s*paramedics(?: are)? here[.!]?$|\b(?:paramedics|ems|(?:the )?ambulance) (?:are here|is here|(?:have |has )?arrived)\b|\bparamédicos (?:llegaron|están aquí)\b|\bcấp cứu đã (?:đến|tới)\b"

PROMPT_VERSION = "v8"
SYSTEM = """You are the Stroke Check emergency assistant. A stroke warning check just found possible stroke signs. The app has already shown "Call 911 now."
You talk with the person, or with the helper next to them, until the paramedics arrive. You act only through tools.

Every turn, in this order:
1. If the last message states a fact, call record_fact (or start_timer for the time) first. Fields someone_present and called_911 are facts too.
2. If the last message asks for help or raises a care concern, call lookup_guidance using the latest concern. Call speak to deliver the grounded answer, then wait.
3. Otherwise listen to the person's words. Briefly acknowledge what they just told you when it helps, or ask one short clarifying question. Continue with STATE.next_step when a handoff question is pending. Speak naturally in {language}.
   When the handoff is complete, respond to their latest input. Point to the written handoff if useful. Keep the paramedic summary in writing. Avoid repeating a static script or adding new care instructions.
4. Call finish_turn().

Conversation rules:
- Your first question is "Is someone with you right now?" (field someone_present).
- If the person is alone: tell them their contacts are being alerted, to unlock the door and sit or lie down somewhere safe, then ask the handoff questions as simple yes/no questions.
- If someone is with them: talk to that helper about the person.
- Ask whether 911 has been called only when STATE.call_confirmed is false. If it is true, acknowledge it briefly when relevant and do not ask again or tell them to call.
- Finish the preliminary handoff fields in STATE.next_step order before open-ended conversation. Ask about blood thinners, other medicines, allergies, head injury, diabetes and seizures one at a time. If the family asks a care question early, answer it without consuming their pending handoff question.
- If the answer is short or unclear, ask again in simpler words instead of guessing.
- Once last_known_normal is known, call write_911_script.
- When 911 has been called, call notify_contacts("911 has been called.") once.
- If they say which hospital the paramedics are taking the person to, call record_fact(field="destination_hospital") with the hospital name exactly as they said it.

Rules:
- record_fact and start_timer values must be copied word for word from the message. Never guess, infer or rephrase. If they refuse or do not know, record nothing and move on.
- If they correct the time, call start_timer again with the new words.
- Never diagnose. Never say they are fine or okay, that it is not a stroke, or that they can wait. Never suggest driving. Never give medicine or dose advice.
- Every speak and ask text must be in {language}. lookup_guidance questions are in English.
- No tool names or JSON in spoken text."""

VERIFY_PROMPT = """You check a paramedic handoff against the conversation it came from.
FAMILY LINES (numbered):
{lines}

HANDOFF FACTS:
{facts}

For each fact, give the number of the family line that states this value for this field. Use null if no line supports it, or if the line is about a different field.
Answer with JSON only: {{"field_name": line_number_or_null, ...}}"""


def load_env():
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1)
            if v and not os.environ.get(k):
                os.environ[k] = v


def location_text(location):
    if isinstance(location, dict):
        lat = location.get("lat", location.get("latitude"))
        lon = location.get("lon", location.get("lng", location.get("longitude")))
        if lat is not None and lon is not None:
            return f"https://maps.google.com/?q={lat},{lon}"
    return location if isinstance(location, str) and location else None


def new_state(result, language):
    return {
        "session": time.strftime("%Y%m%d-%H%M%S"),
        "user": result.get("user") or result.get("name") or "user",
        "language": language if language in LANGUAGES else "en",
        "result": {"signs": result.get("signs", {}), "overall": result.get("overall"), "name": result.get("name")},
        "location": location_text(result.get("location")),
        "started_at": agent_tools.now(),
        "timer": {"last_known_normal": None, "set_at": None},
        "facts": {},
        "call_confirmed": False,
        "pending_question": None,
        "preliminary_skipped": [],
        "care_question": False,
        "asked": [],
        "family": [],
        "conversation": [],
        "said": [],
        "blocked": [],
        "timeline": [{"t": agent_tools.now(), "event": "check found possible stroke signs; app showed 'Call 911 now'"}],
        "script_pos": 0,
        "handoff_written": False,
        "notified": [],
        "covered": [],
        "turn_said_start": 0,
        "no_answer": 0,
        "paramedics_here": False,
        "script_911": None,
        "verified": None,
        "removed_facts": [],
        "incident": result.get("incident") if isinstance(result.get("incident"), str) else None,
    }


def context(state):
    return "STATE\n" + json.dumps({
        "signs": state["result"]["signs"],
        "last_known_normal": state["timer"]["last_known_normal"],
        "facts_recorded": state["facts"],
        "call_confirmed": agent_tools.called_911(state),
        "fields_already_asked": state["asked"],
        "already_covered": state["covered"],
        "script_911_written": bool(state["script_911"]),
        "next_step": agent_tools.next_step(state),
    }, ensure_ascii=False)


def messages_for(state):
    lines = [f"{'Family' if t['role'] == 'family' else 'You'}: {t['text']}" for t in state["conversation"][-12:]]
    transcript = "\n".join(lines) if lines else "(no messages yet: this is your first turn)"
    user = f"CONVERSATION SO FAR\n{transcript}\n\n{context(state)}\n\nTake your next turn now using tool calls."
    return [{"role": "system", "content": SYSTEM.replace("{language}", LANGUAGES[state["language"]])},
            {"role": "user", "content": user}]


def text_tool_calls(content):
    calls = []
    pattern = r"\b(\w+)\s*(\{.*?\}|\(\))(?=\s*(?:\w+\s*[{(]|$))"
    for name, body in re.findall(pattern, content.replace("<|\"|>", '"'), re.S):
        if name not in agent_tools.RUN:
            continue
        body = "{}" if body == "()" else re.sub(r'([{,])\s*(\w+)\s*:', r'\1"\2":', body)
        try:
            calls.append((name, json.loads(body)))
        except ValueError:
            continue
    return calls


CLIENTS = {}


def client_for(name):
    from openai import OpenAI

    if name not in CLIENTS:
        load_env()
        cfg = MODELS[name]
        CLIENTS[name] = OpenAI(base_url=cfg["base_url"], api_key=os.environ.get(cfg["key"], "missing"), timeout=TIMEOUT_S, max_retries=0)
    return CLIENTS[name], MODELS[name]["model"]


POOL = ThreadPoolExecutor(max_workers=16)


class DeadlineError(Exception):
    pass


def call_model(client, model, msgs, reasoning, name, timeout=TIMEOUT_S, tools=True):
    extra, kw = {}, {}
    if name in ("bolt", "super"):
        extra = {"chat_template_kwargs": {"enable_thinking": bool(reasoning)}}
    if name == "gemini":
        kw = {"reasoning_effort": "low" if reasoning else "none"}
    if tools:
        kw.update(tools=agent_tools.TOOLS, tool_choice="required")
    future = POOL.submit(
        client.chat.completions.create,
        model=model, messages=msgs, temperature=0.2, max_tokens=2000 if reasoning else 400,
        extra_body=extra or None, timeout=timeout, **kw,
    )
    try:
        return future.result(timeout=timeout)
    except FutureTimeout as error:
        raise DeadlineError(f"no reply within {timeout:.1f} s") from error


def act(actions, tool, args, result, by="model", ms=None, reasoning=False, model=None, note=None):
    actions.append({"tool": tool, "args": args, "result": result, "t": agent_tools.now(), "ms": ms,
                    "reasoning": reasoning, "by": by, "model": model, "note": note})


def fallback(state, actions, reason):
    script = safety.fixed_script(state["language"])
    covered = state["covered"]
    lkn = state["timer"]["last_known_normal"] or state["facts"].get("last_known_normal")
    if state["paramedics_here"]:
        line = agent_tools.LINES["follow_paramedics"][agent_tools.lang(state)]
    elif state.get("care_question"):
        text = state["family"][-1]
        guidance = state.get("turn_guidance") or agent_tools.lookup_guidance(state, text)
        if not any(item["tool"] == "lookup_guidance" for item in actions):
            act(actions, "lookup_guidance", {"question": text}, guidance, by="code")
        act(actions, "fixed_script", {"reason": reason}, agent_tools.speak_guidance(state, guidance), by="code")
        return
    elif not state["said"] and not agent_tools.called_911(state):
        line = script[0] + " " + script[1]
    elif "told_911" not in covered and not agent_tools.called_911(state):
        line = script[1]
    elif not lkn and state["script_pos"] < 1:
        line = script[2]
    elif "no_food_drink_medicine" not in covered:
        line = script[3]
    else:
        line = script[4]
    state["script_pos"] += 1
    agent_tools.mark_covered(state, line)
    state["said"].append(line)
    state["timeline"].append({"t": agent_tools.now(), "event": f"fixed script ({reason}): {line}"})
    act(actions, "fixed_script", {"reason": reason}, {"said": line}, by="code")


def verify_handoff(state, actions):
    facts = {f: v for f, v in state["facts"].items() if f in agent_tools.FIELDS and v}
    lines = state["family"]
    t0 = time.time()
    _, model = client_for("super")
    if not facts:
        state["verified"] = True
        act(actions, "verify_handoff", {"facts": 0}, {"ok": True, "removed": []}, by="code", ms=0, reasoning=True, model=model)
        return
    prompt = VERIFY_PROMPT.format(lines="\n".join(f"{i + 1}. {l}" for i, l in enumerate(lines)),
                                  facts=json.dumps(facts, ensure_ascii=False))
    try:
        client, model = client_for("super")
        r = call_model(client, model, [{"role": "user", "content": prompt}], True, "super", timeout=VERIFY_TIMEOUT_S, tools=False)
        text = r.choices[0].message.content or ""
        support = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    except Exception as error:
        state["verified"] = False
        act(actions, "verify_handoff", {"facts": len(facts)}, {"ok": False, "error": type(error).__name__},
            by="code", ms=round((time.time() - t0) * 1000), reasoning=True, model=model)
        return
    state["facts_before_check"] = dict(facts)
    removed = []
    for field, value in facts.items():
        n = support.get(field)
        line = lines[n - 1] if isinstance(n, int) and 1 <= n <= len(lines) else ""
        if not agent_tools._said_by_family({"family": [line]}, value):
            removed.append({"field": field, "value": value, "reason": "no supporting line"})
            state["facts"].pop(field, None)
            if field == "last_known_normal":
                state["timer"]["last_known_normal"] = None
    state["removed_facts"] += removed
    state["verified"] = True
    state["timeline"].append({"t": agent_tools.now(), "event": f"handoff checked by Nemotron Super: {len(facts) - len(removed)} supported, {len(removed)} removed"})
    act(actions, "verify_handoff", {"facts": len(facts)}, {"ok": True, "removed": removed},
        by="code", ms=round((time.time() - t0) * 1000), reasoning=True, model=model)


def enforce(state, actions, said_before, handoff_turn):
    l = agent_tools.lang(state)
    if handoff_turn:
        if not state["handoff_written"]:
            act(actions, "write_handoff", {}, agent_tools.write_handoff(state), by="code")
        if state["verified"] is None:
            verify_handoff(state, actions)
            if state["script_911"]:
                agent_tools.write_911_script(state)
        return
    if state["paramedics_here"]:
        return
    if state["no_answer"] >= 2:
        who = state["result"].get("name") or state["user"]
        update = f"{who} is not answering after a stroke warning. Please call them now, or call 911 if you can't reach them."
        if update not in state["notified"]:
            act(actions, "notify_contacts", {"update": update}, agent_tools.notify_contacts(state, update), by="code")
        if not any("help" in s.lower() or "ayuda" in s.lower() or "giúp" in s.lower() for s in state["said"][said_before:]):
            act(actions, "speak", {"text": agent_tools.LINES["help_coming"][l]}, agent_tools.speak(state, agent_tools.LINES["help_coming"][l]), by="code")
    if state["timer"]["last_known_normal"] or state["facts"].get("last_known_normal"):
        before = state["script_911"]
        res = agent_tools.write_911_script(state)
        if res.get("ok") and res["script_911"] != before and not any(a["tool"] == "write_911_script" for a in actions):
            act(actions, "write_911_script", {}, res, by="code")
    field = agent_tools.pending_preliminary(state)
    if not field and state.get("pending_question") in agent_tools.PRELIMINARY:
        state["pending_question"] = None
    if field and not state["asked_this_turn"] and not state.get("care_question") and state["no_answer"] < 2:
        args = {"field": field, "question": agent_tools.preliminary_prompt(state, field)}
        act(actions, "ask", args, agent_tools.ask(state, **args), by="code")
    if not agent_tools.called_911(state) and not any(agent_tools.CALL_REMINDER.search(s) for s in state["said"][said_before:]):
        line = agent_tools.LINES["call_911"][l]
        state["said"].insert(said_before, line)
        agent_tools.mark_covered(state, line)
        act(actions, "speak", {"text": line}, {"ok": True, "said": line}, by="code", note="repeated until 911 is called")


def run_turn(state, model_name="bolt", reasoning=False, failover=True, initial_actions=None):
    from openai import InternalServerError, RateLimitError

    actions = list(initial_actions or [])
    said_before = len(state["said"])
    state["turn_said_start"] = said_before
    state["asked_this_turn"] = False
    state["turn_guidance"] = None
    turn_start = time.time()
    used = model_name
    handoff_turn = state["paramedics_here"] and not state["handoff_written"]
    if not handoff_turn:
        try:
            msgs = messages_for(state)
            for step in range(MAX_STEPS):
                left = TIMEOUT_S - (time.time() - turn_start)
                if left <= 0.3:
                    if len(state["said"]) == said_before:
                        fallback(state, actions, "turn over 6 s")
                    break
                t0 = time.time()
                client, model = client_for(used)
                try:
                    r = call_model(client, model, msgs, reasoning, used, timeout=left)
                except (RateLimitError, InternalServerError) as error:
                    if used != "bolt" or not failover:
                        raise
                    act(actions, "switch_model", {"from": model, "to": MODELS["super"]["model"]}, {"ok": True, "reason": type(error).__name__}, by="code")
                    used = "super"
                    continue
                ms = round((time.time() - t0) * 1000)
                m = r.choices[0].message
                calls = [(c.id, c.function.name, c.function.arguments) for c in m.tool_calls or []]
                if not calls:
                    for name, args in text_tool_calls((m.content or "").strip()):
                        try:
                            res = agent_tools.RUN[name](state, **args)
                        except TypeError as error:
                            res = {"ok": False, "error": f"Bad arguments: {error}"}
                        act(actions, name, args, res, ms=ms, reasoning=reasoning, model=model, note="parsed from text")
                    break
                msgs.append({"role": "assistant", "content": m.content or "", "tool_calls": [
                    {"id": i, "type": "function", "function": {"name": n, "arguments": a}} for i, n, a in calls]})
                finished = False
                for call_id, name, raw in calls:
                    try:
                        args = {} if name in ("finish_turn", "write_handoff", "write_911_script") else json.loads(raw or "{}")
                        res = agent_tools.RUN[name](state, **args) if name in agent_tools.RUN else {"ok": False, "error": f"Unknown tool {name}"}
                    except (TypeError, ValueError) as error:
                        args, res = {"raw": raw}, {"ok": False, "error": f"Bad arguments: {error}"}
                    act(actions, name, args, res, ms=ms, reasoning=reasoning, model=model)
                    msgs.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(res, ensure_ascii=False)})
                    finished = finished or (name == "finish_turn" and res.get("ok"))
                if finished or state["asked_this_turn"]:
                    break
        except Exception as error:
            if len(state["said"]) == said_before:
                fallback(state, actions, f"{type(error).__name__}")
    enforce(state, actions, said_before, handoff_turn)
    if not handoff_turn and len(state["said"]) == said_before:
        fallback(state, actions, "model said nothing")
    say = " ".join(state["said"][said_before:])
    if say:
        state["conversation"].append({"role": "agent", "text": say})
    return say, actions, MODELS[used]["model"]


def short_result(result):
    if result.get("error"):
        return "error: " + str(result["error"])[:80]
    for key in ("said", "script_911", "section", "value", "last_known_normal"):
        if result.get(key):
            return str(result[key])[:120]
    if "removed" in result:
        return f"{len(result['removed'])} facts removed"
    if "sent" in result:
        return f"sent to {len(result['sent'])} contacts"
    return "ok"


def steps(actions):
    return [{
        "tool": a["tool"], "args": a["args"], "result": short_result(a["result"]), "ms": a.get("ms"),
        "reasoning": a.get("reasoning", False), "by": a.get("by", "model"), "model": a.get("model"),
        "blocked_by_safety": bool(a["result"].get("blocked")), "replaced_with": a["result"].get("replaced_with"),
    } for a in actions]


def respond(state, actions, say, model=None):
    state["asked_this_turn"] = False
    handoff = agent_tools.build_handoff(state)
    if state.get("incident"):
        try:
            from backend import incident

            incident.update_from_agent(state["incident"], state, handoff)
        except Exception:
            pass
    HANDOFFS.mkdir(parents=True, exist_ok=True)
    (HANDOFFS / f"{state['session']}.txt").write_text(handoff["text"], encoding="utf-8")
    return {"say": say, "actions": actions, "steps": steps(actions), "model": model, "handoff": handoff, "state": state,
            "verified": state.get("verified"), "removed_facts": state.get("removed_facts", []), "script_911": state.get("script_911")}


def start(result, language="en", model_name="bolt", reasoning=False, failover=True):
    state = new_state(result, language)
    say, actions, model = run_turn(state, model_name, reasoning, failover)
    return respond(state, actions, say, model)


def capture_preliminary_reply(state, text, actions):
    field = state.get("pending_question")
    if not agent_tools.called_911(state) and (
        agent_tools.confirmed_call_statement(text)
        or (field == "called_911" and agent_tools.confirmed_call_answer(text))
    ):
        args = {"field": "called_911", "value": text}
        act(actions, "record_fact", args, agent_tools.record_fact(state, **args), by="code")
        return
    if not field or state.get("care_question") or state.get("paramedics_here"):
        return
    normalized = text.lower().strip().rstrip(".!")
    if re.fullmatch(agent_tools.UNKNOWN, normalized):
        if field not in state["preliminary_skipped"]:
            state["preliminary_skipped"].append(field)
        return
    yes = re.fullmatch(agent_tools.SHORT_YES, normalized)
    no = re.fullmatch(agent_tools.SHORT_NO, normalized)
    if field == "last_known_normal" and re.search(r"\b\d{1,2}:\d{2}\b", text):
        act(actions, "start_timer", {"last_known_normal": text}, agent_tools.start_timer(state, text), by="code")
    elif field not in ("called_911", "last_known_normal") and (no or (yes and field != "current_medications")):
        args = {"field": field, "value": text}
        act(actions, "record_fact", args, agent_tools.record_fact(state, **args), by="code")



def reply(text, state, language=None, model_name="bolt", reasoning=False, failover=True):
    if language in LANGUAGES:
        state["language"] = language
    text = (text or "").strip()
    actions = []
    state["care_question"] = (
        text.endswith("?") or text.startswith("¿")
        or bool(re.match(r"^(?:what|where|when|why|how|(?:can|should|could) (?:i|we|they|he|she)|tôi có nên)\b", text, re.I))
        or bool(re.search(agent_tools.CARE_IMPROVED, text, re.I))
        or bool(re.search(agent_tools.CARE_WORRY, text, re.I))
    )
    if re.fullmatch(NO_ANSWER, text.lower()):
        state["no_answer"] = state.get("no_answer", 0) + 1
        state["conversation"].append({"role": "family", "text": "(no answer)"})
        state["timeline"].append({"t": agent_tools.now(), "event": "no answer"})
    else:
        state["no_answer"] = 0
        state["family"].append(text)
        state["conversation"].append({"role": "family", "text": text})
        state["timeline"].append({"t": agent_tools.now(), "event": f"family: {text}"})
        if not state["care_question"] and re.search(PARAMEDICS, text.lower()):
            state["paramedics_here"] = True
        capture_preliminary_reply(state, text, actions)
    say, actions, model = run_turn(state, model_name, reasoning, failover, actions)
    return respond(state, actions, say, model)


def warm(model_name="bolt"):
    try:
        client, model = client_for(model_name)
        client.chat.completions.create(model=model, messages=[{"role": "user", "content": "ok"}], max_tokens=1, timeout=20)
        return True
    except Exception:
        return False


def handoff_text(session):
    path = HANDOFFS / f"{session}.txt"
    return path.read_text(encoding="utf-8") if path.exists() else "No handoff available."
