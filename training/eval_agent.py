import argparse
import json
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from backend import agent, safety

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


def run_scenario(sc, model_name, reasoning):
    result = {"user": "eval", "overall": "warning",
              "signs": {k: {"flag": True, "p": p} for k, p in sc["signs"].items()}}
    turns = []
    t = time.time()
    out = agent.start(result, sc["language"], model_name, reasoning)
    turns.append({"family": None, "say": out["say"], "actions": out["actions"], "ms": round((time.time() - t) * 1000)})
    state = out["state"]
    for msg in sc["messages"]:
        t = time.time()
        out = agent.reply(msg, state, sc["language"], model_name, reasoning)
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
        "safe_output": all(safety.check(s)["ok"] for s in said),
        "model_needed_no_safety_fix": len(state["blocked"]) == 0,
        "asked_last_known_normal": "last_known_normal" in state["asked"] or state["timer"]["last_known_normal"] is not None,
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
        if field != "last_known_normal":
            c[f"recorded_{field}"] = contains(state["facts"].get(field, "") or "", value)
    if "question" in sc["tags"]:
        c["used_guidance"] = any(a["tool"] == "lookup_guidance" for a in actions)
    return c


def summarize(label, config, scenarios, runs):
    rows = []
    for sc, run in zip(scenarios, runs):
        c = checks(sc, run)
        rows.append({"id": sc["id"], "checks": c, "pass": all(c.values()),
                     "interventions": len(run["state"]["blocked"]),
                     "fallbacks": sum(1 for t in run["turns"] for a in t["actions"] if a["tool"] == "fixed_script")})
    names = sorted({k for r in rows for k in r["checks"]})
    rates = {n: round(sum(r["checks"][n] for r in rows if n in r["checks"]) / sum(1 for r in rows if n in r["checks"]), 3) for n in names}
    turn_ms = [t["ms"] for run in runs for t in run["turns"]]
    call_ms = [a["latency_ms"] for run in runs for t in run["turns"] for a in t["actions"] if a.get("latency_ms")]
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
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-save", action="store_true")
    a = p.parse_args()
    scenarios = json.loads(SCENARIOS.read_text(encoding="utf-8"))
    if a.only:
        scenarios = [s for s in scenarios if s["id"] in a.only.split(",")]
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    table = []
    for config in a.configs.split(","):
        model_name, reasoning = CONFIGS[config]
        t = time.time()
        with ThreadPoolExecutor(a.workers) as ex:
            runs = list(ex.map(lambda s: run_scenario(s, model_name, reasoning), scenarios))
        summary = summarize(a.label, config, scenarios, runs)
        summary["wall_s"] = round(time.time() - t)
        (RUNS_DIR / f"{a.label}_{config}.json").write_text(json.dumps(runs, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        if not a.no_save:
            save(summary)
        table.append(summary)
        print(f"{config:9} pass {summary['pass_rate']:.2f} | turn ms median {summary['turn_latency_ms']['median']} p95 {summary['turn_latency_ms']['p95']} | "
              f"safety fixes {summary['safety_interventions']} | fallbacks {summary['fallback_turns']} | {summary['wall_s']} s", flush=True)
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
