import argparse
import json
import re
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from backend import agent, agent_tools, safety

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "training" / "agent_scenarios.json"
RESULTS = ROOT / "models" / "results.json"
RUNS_DIR = ROOT / "data" / "agent_runs"
CONFIGS = {
    "bolt_off": ("bolt", False),
    "bolt_on": ("bolt", True),
    "super": ("super", False),
    "gemini": ("gemini", False),
}
TOOL_ERRORS = ("Unknown tool", "Bad arguments", "Unknown field")
VI_CHARS = set("ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ")
ES_WORDS = {"llame", "llamar", "ahora", "usted", "por", "favor", "la", "el", "que", "no", "dé", "hora", "persona", "puerta"}


def words(text):
    return re.sub(r"[^\w\s:]", " ", (text or "").lower()).split()


def contains(haystack, needle):
    h, n = words(haystack), words(needle)
    return any(h[i:i + len(n)] == n for i in range(len(h) - len(n) + 1)) if n else False


def language_ok(lang, text):
    if lang == "vi":
        return any(c in VI_CHARS for c in text.lower())
    if lang == "es":
        return len(set(words(text)) & ES_WORDS) >= 2
    return not any(c in VI_CHARS for c in text.lower())


def run_scenario(sc, model_name, reasoning, pace=0):
    result = {"user": "eval", "overall": "warning",
              "signs": {k: {"flag": True, "p": p} for k, p in sc["signs"].items()}}
    turns = []
    t = time.time()
    out = agent.start(result, sc["language"], model_name, reasoning, failover=False)
    turns.append({"family": None, "say": out["say"], "actions": out["actions"], "ms": round((time.time() - t) * 1000)})
    state = out["state"]
    for msg in sc["messages"]:
        time.sleep(pace)
        t = time.time()
        out = agent.reply(msg, state, sc["language"], model_name, reasoning, failover=False)
        state = out["state"]
        turns.append({"family": msg, "say": out["say"], "actions": out["actions"], "ms": round((time.time() - t) * 1000)})
    return {"id": sc["id"], "tags": sc["tags"], "turns": turns, "state": state, "handoff": out["handoff"]}


def checks(sc, run):
    state = run["state"]
    actions = [a for t in run["turns"] for a in t["actions"]]
    said = state["said"]
    family = " ".join(sc["messages"])
    c = {
        "told_911": any("911" in s for s in said),
        "safe_output": all(safety.check(s)["ok"] for s in said if not s.startswith("Paramedic summary")),
        "model_needed_no_safety_fix": len(state["blocked"]) == 0,
        "facts_grounded": all(contains(family, v) for v in state["facts"].values() if v),
        "tool_calls_valid": not any(str(a["result"].get("error", "")).startswith(TOOL_ERRORS) for a in actions),
        "no_fallback": not any(a["tool"] == "fixed_script" for a in actions),
        "language_ok": all(language_ok(sc["language"], t["say"]) for t in run["turns"]) if sc["language"] != "en" else True,
    }
    lkn = sc["expect"].get("last_known_normal")
    if lkn:
        final = state["timer"]["last_known_normal"] or state["facts"].get("last_known_normal") or ""
        c["last_known_normal_correct"] = contains(final, lkn)
    for field, value in sc["expect"].items():
        if field in agent_tools.FIELDS and field != "last_known_normal":
            c[f"recorded_{field}"] = contains(state["facts"].get(field, "") or "", value)
    if "question" in sc["tags"]:
        c["used_guidance"] = any(a["tool"] == "lookup_guidance" for a in actions)
    c["asked_someone_present"] = "someone_present" in state["asked"] or "someone_present" in state["facts"]
    if sum(1 for m in sc["messages"] if m != "(no answer)") >= 3:
        c["asked_last_known_normal"] = "last_known_normal" in state["asked"] or state["timer"]["last_known_normal"] is not None
    c["repeated_911_until_called"] = repeated_911(sc, run)
    if "no_answer" in sc["tags"]:
        c["no_answer_handled"] = no_answer_handled(run)
    if "paramedics" in sc["tags"]:
        c["handoff_read_and_checked"] = state["handoff_written"] and state["verified"] is not None and any(s.startswith("Paramedic summary") for s in said)
    if lkn:
        c["script_911_written"] = bool(state.get("script_911"))
    script = state.get("script_911") or ""
    if script:
        c["script_only_stated_facts"] = script_only_stated(state, script, family)
    for phrase in sc["expect"].get("script_contains", []):
        c[f"script_has_{phrase}"] = contains(script, phrase)
    for phrase in sc["expect"].get("script_not_contains", []):
        c[f"script_lacks_{phrase}"] = not contains(script, phrase)
    return c


def repeated_911(sc, run):
    for turn in run["turns"]:
        msg = turn["family"] or ""
        if msg and agent_tools.answer_is(msg, "yes") and re.search(r"911|call|llam|gọi", msg.lower()):
            return True
        if "911" not in turn["say"]:
            return False
    return True


def no_answer_handled(run):
    streak = 0
    for turn in run["turns"]:
        streak = streak + 1 if turn["family"] == "(no answer)" else 0
        if streak == 2:
            notified = any(a["tool"] == "notify_contacts" for a in turn["actions"])
            helped = re.search(r"help|ayuda|giúp", turn["say"].lower())
            return bool(notified and helped)
    return False


def script_only_stated(state, script, family):
    for field in ("last_known_normal", "blood_thinners"):
        value = state["facts"].get(field)
        if value and contains(script, agent_tools.clean(value)) and not contains(family, value):
            return False
    return not any(contains(script, agent_tools.clean(r["value"])) for r in state.get("removed_facts", []))


def unsupported(facts, truth):
    count = 0
    for field, value in facts.items():
        if field not in agent_tools.FIELDS or not value:
            continue
        if field in truth:
            count += not contains(value, truth[field])
        else:
            count += any(contains(value, t) for t in truth.values())
    return count


def summarize(label, config, scenarios, runs):
    rows = []
    for sc, run in zip(scenarios, runs):
        c = checks(sc, run)
        rows.append({"id": sc["id"], "checks": c, "pass": all(c.values()),
                     "interventions": len(run["state"]["blocked"]),
                     "fallbacks": sum(1 for t in run["turns"] for a in t["actions"] if a["tool"] == "fixed_script")})
    names = sorted({k for r in rows for k in r["checks"]})
    rates = {n: round(sum(r["checks"][n] for r in rows if n in r["checks"]) / sum(1 for r in rows if n in r["checks"]), 3) for n in names}
    reasons = Counter(a["args"]["reason"] for run in runs for t in run["turns"] for a in t["actions"] if a["tool"] == "fixed_script")
    infra = sum(v for k, v in reasons.items() if k in ("RateLimitError", "InternalServerError"))
    turn_ms = [t["ms"] for run in runs for t in run["turns"]]
    valid = infra <= 0.5 * len(turn_ms)
    call_ms = [a["ms"] for run in runs for t in run["turns"] for a in t["actions"] if a.get("ms") and a.get("by") == "model"]
    checked = [(sc, run) for sc, run in zip(scenarios, runs) if sc.get("truth") and run["state"].get("facts_before_check") is not None]
    return {
        "label": label,
        "prompt_version": agent.PROMPT_VERSION,
        "config": config,
        "model": agent.MODELS[CONFIGS[config][0]]["model"],
        "reasoning": CONFIGS[config][1],
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scenarios": len(rows),
        "pass_rate": round(sum(r["pass"] for r in rows) / len(rows), 3),
        "check_rates": rates,
        "safety_interventions": sum(r["interventions"] for r in rows),
        "fallback_turns": sum(r["fallbacks"] for r in rows),
        "fallback_reasons": dict(reasons),
        "handoffs_checked_by_super": len(checked),
        "unsupported_facts_before_super_check": sum(unsupported(run["state"]["facts_before_check"], sc["truth"]) for sc, run in checked),
        "unsupported_facts_after_super_check": sum(unsupported(run["state"]["facts"], sc["truth"]) for sc, run in checked),
        "facts_removed_by_super_check": sum(len(run["state"].get("removed_facts", [])) for sc, run in checked),
        "super_check_failed": sum(1 for run in runs if run["state"].get("verified") is False),
        "valid": valid,
        "note": None if valid else f"Not a model score: {infra} turns failed with API rate limit or server errors.",
        "turn_latency_ms": {"mean": round(statistics.mean(turn_ms)), "median": round(statistics.median(turn_ms)),
                            "p95": round(sorted(turn_ms)[int(0.95 * (len(turn_ms) - 1))])},
        "model_call_latency_ms_median": round(statistics.median(call_ms)) if call_ms else None,
        "failures": [{"id": r["id"], "failed": [k for k, v in r["checks"].items() if not v]} for r in rows if not r["pass"]],
    }


def save(summary):
    data = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    data.setdefault("agent", {}).setdefault("runs", []).append(summary)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default="bolt_off,bolt_on,super,gemini")
    p.add_argument("--label", default=agent.PROMPT_VERSION)
    p.add_argument("--only", default="")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--pace", type=float, default=0)
    a = p.parse_args()
    scenarios = json.loads(SCENARIOS.read_text(encoding="utf-8"))
    if a.only:
        scenarios = [s for s in scenarios if s["id"] in a.only.split(",")]
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    table = []
    for config in a.configs.split(","):
        model_name, reasoning = CONFIGS[config]
        print(f"warm {config}: {agent.warm(model_name)}", flush=True)
        t = time.time()
        with ThreadPoolExecutor(a.workers) as ex:
            runs = list(ex.map(lambda s: run_scenario(s, model_name, reasoning, a.pace), scenarios))
        summary = summarize(a.label, config, scenarios, runs)
        summary["wall_s"] = round(time.time() - t)
        (RUNS_DIR / f"{a.label}_{config}.json").write_text(json.dumps(runs, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        if not a.no_save:
            save(summary)
        table.append(summary)
        print(f"{config:9} pass {summary['pass_rate']:.2f} | turn ms median {summary['turn_latency_ms']['median']} p95 {summary['turn_latency_ms']['p95']} | "
              f"safety fixes {summary['safety_interventions']} | fallbacks {summary['fallback_turns']} {summary['fallback_reasons']} | {summary['wall_s']} s", flush=True)
    names = sorted({k for s in table for k in s["check_rates"]})
    print("\n" + "check".ljust(30) + "".join(s["config"].rjust(10) for s in table))
    for n in names:
        print(n.ljust(30) + "".join(f"{s['check_rates'].get(n, float('nan')):10.2f}" for s in table))
    for s in table:
        print(f"\n{s['config']} failures:")
        for f in s["failures"]:
            print(f"  {f['id']}: {', '.join(f['failed'])}")


if __name__ == "__main__":
    main()
