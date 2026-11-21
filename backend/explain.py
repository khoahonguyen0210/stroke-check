import json
import re
from math import isfinite

from backend import agent, safety

TIMEOUT_S = 4
SIGN_NAMES = {"face": "face droop", "arms": "arm drift", "speech": "slurred speech"}
SYSTEM = """You explain the result of a stroke warning check in 2 or 3 plain sentences for the person who just did it.
Use only the measurements in the JSON. Name which measurements drove each flagged sign, using the numbers exactly as given.
Units: face values are mouth-corner lift in eye-widths; arm values are wrist drop in shoulder-widths during the 10-second hold.
Never diagnose. Never say "not a stroke", "you're fine" or anything reassuring. Never mention medicine.
If any sign is flagged, the last sentence must be exactly: Call 911 now."""


def numbers(values, prefix=""):
    out = {}
    for key, value in values.items():
        if isinstance(value, dict):
            out.update(numbers(value, f"{prefix}{key}."))
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value):
            out[f"{prefix}{key}"] = round(float(value), 2)
    return out


def allowed(values):
    ok = {"911"}
    for value in values.values():
        for v in (value, abs(value), value * 100, abs(value) * 100):
            for digits in (0, 1, 2):
                text = f"{round(v, digits):.{digits}f}"
                ok.add(text)
                ok.add(text.rstrip("0").rstrip(".") if "." in text else text)
    return ok


def grounded(text, values):
    found = re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))
    good = allowed(values)
    return all(n in good or n.lstrip("0") in good or ("0" + n) in good for n in found)


def flagged(result):
    return [name for name, sign in result.get("signs", {}).items() if isinstance(sign, dict) and sign.get("flag")]


def template(result, features):
    face = features.get("face", {})
    arms = features.get("arms", {})
    parts = []
    signs = flagged(result)
    if "face" in signs and face:
        parts.append(f"Your left mouth corner lifted {face.get('smile_left', 0):.2f} and your right {face.get('smile_right', 0):.2f} eye-widths, "
                     f"an asymmetry of {face.get('smile_asymmetry', 0):.2f}.")
    if "arms" in signs and arms:
        parts.append(f"During the hold your left wrist dropped {arms.get('left_drop', 0):.2f} and your right {arms.get('right_drop', 0):.2f} shoulder-widths.")
    if signs:
        parts.append("Call 911 now.")
    else:
        parts.append("Your face and arm measurements did not cross the warning thresholds. This is a warning tool, not a diagnosis.")
    return " ".join(parts)


def write(result, features):
    values = numbers({"signs": {k: v.get("p") for k, v in result.get("signs", {}).items() if isinstance(v, dict)}, **features})
    signs = flagged(result)
    readable = {k.replace(".", " ").replace("_", " "): v for k, v in values.items()}
    payload = {"flagged": [SIGN_NAMES.get(s, s) for s in signs], "overall": result.get("overall"), "measurements": readable}
    _, model = agent.client_for("bolt")
    try:
        client, model = agent.client_for("bolt")
        r = agent.call_model(client, model, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(payload)}],
                             False, "bolt", timeout=TIMEOUT_S, tools=False)
        text = re.sub(r"\s+", " ", r.choices[0].message.content or "").strip()
    except Exception:
        text = ""
    if text and signs and not text.endswith("Call 911 now."):
        text = text.rstrip(". ") + ". Call 911 now."
    if text and len(text) < 700 and safety.check(text)["ok"] and grounded(text, values):
        return {"text": text, "grounded": True, "model": model}
    return {"text": template(result, features), "grounded": False, "model": None}
