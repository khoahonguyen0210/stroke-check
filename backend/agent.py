import json
import os
import re
import time
from pathlib import Path

from backend import agent_tools, safety

ROOT = Path(__file__).resolve().parents[1]
HANDOFFS = ROOT / "data" / "handoffs"
MAX_STEPS = 6
TIMEOUT_S = 6

MODELS = {
    "bolt": {"base_url": "https://integrate.api.nvidia.com/v1", "key": "NVIDIA_API_KEY", "model": "nvidia/nemotron-3.5-bolt-e4b"},
    "super": {"base_url": "https://integrate.api.nvidia.com/v1", "key": "NVIDIA_API_KEY", "model": "nvidia/nemotron-3.5-super-vl-120b-a12b"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "key": "GEMINI_API_KEY", "model": "gemini-3.5-flash"},
}

LANGUAGES = {"en": "English", "es": "Spanish", "vi": "Vietnamese"}

PROMPT_VERSION = "v2"
SYSTEM = """You are the Stroke Check emergency assistant. A stroke warning check just found possible stroke signs. The app has already shown "Call 911 now."
You help the family in the minutes before the ambulance arrives. You act only through tools.

Every turn, in this order:
1. If the family's last message states a fact, call record_fact (or start_timer for the time) first.
2. If the family's last message asks a question, call lookup_guidance first, then speak the answer using only the returned text.
3. Say exactly one thing with speak() or ask one question with ask(). Use ask() for every question.
4. Call finish_turn().

What to say, in order (skip what is done; see STATE):
- First turn: speak "Possible stroke signs were found. Call 911 now." (in {language}).
- Ask when the person was last seen acting normal: ask(field="last_known_normal").
- Tell them not to give any food, drink or medicine.
- Tell them to stay with the person and unlock the door.
- Ask, one per turn: blood_thinners, recent_fall_or_head_injury, diabetes, seizure, current_medications, allergies.
- If the family says they have not called 911, or hesitates, tell them to call 911 now before anything else.
- When the family says 911 is called, call notify_contacts("911 has been called.") once.
- When the family says the ambulance or paramedics are here, call write_handoff, then speak one sentence telling them to show the handoff to the paramedics.

Rules:
- record_fact and start_timer values must be copied word for word from the family's message. Never guess, infer or rephrase. If they refuse or do not know, record nothing and move on.
- If the family corrects the time, call start_timer again with the new words.
- Never diagnose. Never say they are fine or okay, that it is not a stroke, or that they can wait. Never suggest driving. Never give medicine or dose advice.
- Every speak and ask text must be in {language}, even if the family writes in another language. lookup_guidance questions are in English.
- One short, calm sentence per speak. No tool names or JSON in spoken text."""


def load_env():
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1)
            if v and not os.environ.get(k):
                os.environ[k] = v


def new_state(result, language):
    return {
        "session": time.strftime("%Y%m%d-%H%M%S"),
        "user": result.get("user") or result.get("name") or "user",
        "language": language if language in LANGUAGES else "en",
        "result": {"signs": result.get("signs", {}), "overall": result.get("overall")},
        "started_at": agent_tools.now(),
        "timer": {"last_known_normal": None, "set_at": None},
        "facts": {},
        "asked": [],
        "family": [],
        "conversation": [],
        "said": [],
        "blocked": [],
        "timeline": [{"t": agent_tools.now(), "event": "check found possible stroke signs; app showed 'Call 911 now'"}],
        "script_pos": 0,
        "handoff_written": False,
        "notified": [],
        "turn_said_start": 0,
    }


def context(state):
    signs = {k: v for k, v in state["result"]["signs"].items()}
    return "STATE\n" + json.dumps({
        "signs": signs,
        "last_known_normal": state["timer"]["last_known_normal"],
        "facts_recorded": state["facts"],
        "fields_already_asked": state["asked"],
        "turns_so_far": len(state["family"]),
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


def call_model(client, model, msgs, reasoning, name, timeout=TIMEOUT_S):
    extra, kw = {}, {}
    if name in ("bolt", "super"):
        extra = {"chat_template_kwargs": {"enable_thinking": bool(reasoning)}}
    if name == "gemini":
        kw = {"reasoning_effort": "low" if reasoning else "none"}
    return client.chat.completions.create(
        model=model, messages=msgs, tools=agent_tools.TOOLS, tool_choice="required",
        temperature=0.2, max_tokens=1024 if reasoning else 400, extra_body=extra or None, timeout=timeout, **kw,
    )


def fallback(state, actions, reason):
    script = safety.fixed_script(state["language"])
    line = script[min(state["script_pos"], len(script) - 1)]
    state["script_pos"] += 1
    state["said"].append(line)
    state["timeline"].append({"t": agent_tools.now(), "event": f"fixed script ({reason}): {line}"})
    actions.append({"tool": "fixed_script", "args": {"reason": reason}, "result": {"said": line}, "t": agent_tools.now()})


def run_turn(state, model_name="bolt", reasoning=False):
    actions = []
    said_before = len(state["said"])
    state["turn_said_start"] = said_before
    turn_start = time.time()
    try:
        client, model = client_for(model_name)
        msgs = messages_for(state)
        for step in range(MAX_STEPS):
            left = TIMEOUT_S - (time.time() - turn_start)
            if left <= 0.3:
                if len(state["said"]) == said_before:
                    fallback(state, actions, "turn over 6 s")
                break
            t0 = time.time()
            r = call_model(client, model, msgs, reasoning, model_name, timeout=left)
            latency = round((time.time() - t0) * 1000)
            m = r.choices[0].message
            calls = m.tool_calls or []
            if not calls:
                content = (m.content or "").strip()
                parsed = text_tool_calls(content)
                for name, args in parsed:
                    try:
                        res = agent_tools.RUN[name](state, **args)
                    except TypeError as error:
                        res = {"ok": False, "error": f"Bad arguments: {error}"}
                    actions.append({"tool": name, "args": args, "result": res, "t": agent_tools.now(), "latency_ms": latency, "note": "parsed from text"})
                if not parsed and content and not re.search(r"[{}<>|]", content):
                    res = agent_tools.speak(state, content)
                    actions.append({"tool": "speak", "args": {"text": content}, "result": res, "t": agent_tools.now(), "latency_ms": latency, "note": "plain text reply"})
                break
            msgs.append({"role": "assistant", "content": m.content or "", "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]})
            finished = False
            for c in calls:
                name = c.function.name
                try:
                    args = json.loads(c.function.arguments or "{}")
                    if name in ("finish_turn", "write_handoff"):
                        args = {}
                    res = agent_tools.RUN[name](state, **args) if name in agent_tools.RUN else {"ok": False, "error": f"Unknown tool {name}"}
                except (TypeError, ValueError) as error:
                    args, res = {"raw": c.function.arguments}, {"ok": False, "error": f"Bad arguments: {error}"}
                actions.append({"tool": name, "args": args, "result": res, "t": agent_tools.now(), "latency_ms": latency})
                msgs.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(res, ensure_ascii=False)})
                finished = finished or (name == "finish_turn" and res.get("ok"))
            if finished:
                break
    except Exception as error:
        if len(state["said"]) == said_before:
            fallback(state, actions, f"{type(error).__name__}")
    if len(state["said"]) == said_before:
        fallback(state, actions, "model said nothing")
    say = " ".join(state["said"][said_before:])
    state["conversation"].append({"role": "agent", "text": say})
    return say, actions


def respond(state, actions, say):
    handoff = agent_tools.build_handoff(state)
    HANDOFFS.mkdir(parents=True, exist_ok=True)
    (HANDOFFS / f"{state['session']}.txt").write_text(handoff["text"], encoding="utf-8")
    return {"say": say, "actions": actions, "handoff": handoff, "state": state}


def start(result, language="en", model_name="bolt", reasoning=False):
    state = new_state(result, language)
    say, actions = run_turn(state, model_name, reasoning)
    return respond(state, actions, say)


def reply(text, state, language=None, model_name="bolt", reasoning=False):
    if language in LANGUAGES:
        state["language"] = language
    state["family"].append(text)
    state["conversation"].append({"role": "family", "text": text})
    state["timeline"].append({"t": agent_tools.now(), "event": f"family: {text}"})
    say, actions = run_turn(state, model_name, reasoning)
    return respond(state, actions, say)


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
