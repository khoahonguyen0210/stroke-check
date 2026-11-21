import json
import os
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
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "key": "GEMINI_API_KEY", "model": "gemini-2.5-flash"},
}

LANGUAGES = {"en": "English", "es": "Spanish", "vi": "Vietnamese"}

PROMPT_VERSION = "v1"
SYSTEM = """You are the Stroke Check emergency assistant. A stroke warning check just found possible stroke signs. The app has already shown "Call 911 now."
You help the family in the minutes before the ambulance arrives. You act only through tools.

Every turn:
- Speak with speak() or ask(): calm, short, plain sentences. One instruction or one question per turn.
- Then call finish_turn().

Order of the conversation (skip steps that are done; see STATE):
1. Say possible stroke signs were found and tell them to call 911 now if they have not.
2. Ask when the person was last seen acting normal (field last_known_normal). When they answer, call start_timer with their exact words. If they correct the time, call start_timer again.
3. Tell them not to give any food, drink or medicine.
4. Tell them to stay with the person and unlock the door.
5. Ask the other handoff facts one at a time: blood_thinners, recent_fall_or_head_injury, diabetes, seizure, current_medications, allergies.
6. When the facts are collected or the family says the ambulance is here, call notify_contacts with a short factual update and call write_handoff.

Rules:
- When the family states a fact, call record_fact with their exact words copied from their message. Never guess, infer or rephrase a value. If they refuse or do not know, record nothing and move on.
- If the family asks a question, call lookup_guidance and answer only with the text it returns. If it returns "Ask the 911 dispatcher", say that.
- Never diagnose. Never say they are fine, that it is not a stroke, or that they can wait. Never suggest driving to the hospital. Never give medicine or dose advice.
- After 911 is called, call notify_contacts once with "911 has been called."
- Reply in {language}. Tool arguments for lookup_guidance are in English."""


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
    msgs = [{"role": "system", "content": SYSTEM.replace("{language}", LANGUAGES[state["language"]])}]
    for turn in state["conversation"][-12:]:
        msgs.append({"role": "user" if turn["role"] == "family" else "assistant", "content": turn["text"]})
    msgs.append({"role": "user", "content": context(state)})
    return msgs


def client_for(name):
    from openai import OpenAI

    load_env()
    cfg = MODELS[name]
    return OpenAI(base_url=cfg["base_url"], api_key=os.environ.get(cfg["key"], "missing"), timeout=TIMEOUT_S, max_retries=0), cfg["model"]


def call_model(client, model, msgs, reasoning, name):
    extra = {}
    if name in ("bolt", "super"):
        extra = {"chat_template_kwargs": {"enable_thinking": bool(reasoning)}}
    return client.chat.completions.create(
        model=model, messages=msgs, tools=agent_tools.TOOLS, tool_choice="auto",
        temperature=0.2, max_tokens=1024 if reasoning else 400, extra_body=extra or None,
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
    try:
        client, model = client_for(model_name)
        msgs = messages_for(state)
        for step in range(MAX_STEPS):
            t0 = time.time()
            r = call_model(client, model, msgs, reasoning, model_name)
            latency = round((time.time() - t0) * 1000)
            m = r.choices[0].message
            calls = m.tool_calls or []
            if not calls:
                if m.content and m.content.strip():
                    res = agent_tools.speak(state, m.content.strip())
                    actions.append({"tool": "speak", "args": {"text": m.content.strip()}, "result": res, "t": agent_tools.now(), "latency_ms": latency, "note": "plain text reply"})
                break
            msgs.append({"role": "assistant", "content": m.content or "", "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]})
            finished = False
            for c in calls:
                name = c.function.name
                try:
                    args = json.loads(c.function.arguments or "{}")
                    res = agent_tools.RUN[name](state, **args) if name in agent_tools.RUN else {"ok": False, "error": f"Unknown tool {name}"}
                except (TypeError, ValueError) as error:
                    args, res = {"raw": c.function.arguments}, {"ok": False, "error": f"Bad arguments: {error}"}
                actions.append({"tool": name, "args": args, "result": res, "t": agent_tools.now(), "latency_ms": latency})
                msgs.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(res, ensure_ascii=False)})
                finished = finished or name == "finish_turn"
            if finished:
                break
    except Exception as error:
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


def handoff_text(session):
    path = HANDOFFS / f"{session}.txt"
    return path.read_text(encoding="utf-8") if path.exists() else "No handoff available."
