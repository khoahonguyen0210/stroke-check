import argparse
import io
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from backend import agent, storage

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "models" / "results.json"
MODEL = "gemini-3.5-flash"
SIGNS = ["face_droop", "arm_drift", "slurred"]
PROMPT = """You are helping evaluate a stroke warning screening tool. You get:
- face landmark renderings (MediaPipe face mesh, image y axis points down) for a neutral face, a big smile, and raised eyebrows;
- body pose renderings for the start and end of a 10-second hold with both arms raised straight out;
- an audio recording of the person reading: "{sentence}".
Estimate the probability (0 to 1) of each warning sign: face_droop (one side of the mouth or face lower or weaker), arm_drift (one arm drifts down), slurred (slurred or strange speech).
Answer only with JSON: {{"face_droop": p, "arm_drift": p, "slurred": p}}"""


def render_points(points, title, edges=None):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=80)
    p = np.array(points)
    ax.scatter(p[:, 0], p[:, 1], s=2, c="black")
    for a, b in edges or []:
        ax.plot([p[a, 0], p[b, 0]], [p[a, 1], p[b, 1]], c="red")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def face_images(face):
    frames = [f for f in face["frames"] if f.get("landmarks")]
    out = []
    for phase in ("relax", "smile", "brows"):
        ph = [f for f in frames if f.get("phase") == phase]
        if ph:
            out.append(render_points(ph[len(ph) * 2 // 3]["landmarks"], f"face: {phase}"))
    return out


def arm_images(arms):
    frames = [f for f in arms["frames"] if f.get("pose")]
    edges = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
    out = []
    for name, f in (("arms: start", frames[len(frames) // 10] if frames else None), ("arms: end", frames[-1] if frames else None)):
        if f:
            out.append(render_points([p[:2] for p in f["pose"]], name, edges))
    return out


def ask_gemini(client, session):
    from google.genai import types

    path = session["path"]
    face = json.loads((path / "face.json").read_text())
    arms = json.loads((path / "arms.json").read_text())
    parts = [types.Part.from_bytes(data=img, mime_type="image/png") for img in face_images(face) + arm_images(arms)]
    parts.append(types.Part.from_bytes(data=(path / "speech.webm").read_bytes(), mime_type="audio/webm"))
    parts.append(PROMPT.format(sentence=session["sentence"]))
    t = time.time()
    r = client.models.generate_content(model=MODEL, contents=parts, config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0))
    latency = round((time.time() - t) * 1000)
    return json.loads(r.text), latency


def main():
    import os

    from google import genai
    from sklearn.metrics import accuracy_score, f1_score

    p = argparse.ArgumentParser()
    p.add_argument("--pace", type=float, default=6)
    a = p.parse_args()
    agent.load_env()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    rows = []
    for s in storage.load_sessions():
        try:
            probs, latency = ask_gemini(client, s)
            rows.append({"labels": s["labels"], "probs": probs, "latency_ms": latency})
            print(f"{s['person']}/{s['take_id']} {s['labels']} -> {probs} {latency} ms", flush=True)
        except Exception as error:
            rows.append({"labels": s["labels"], "error": f"{type(error).__name__}: {str(error)[:120]}"})
            print(f"{s['person']}/{s['take_id']} error {type(error).__name__}", flush=True)
        time.sleep(a.pace)
    ok = [r for r in rows if "probs" in r]
    out = {"model": MODEL, "input": "landmark renderings (3 face, 2 arms) + speech audio, no training",
           "sessions": len(rows), "answered": len(ok), "errors": len(rows) - len(ok), "signs": {}}
    for sign in SIGNS:
        y = [int(sign in r["labels"]) for r in ok]
        pred = [int(float(r["probs"].get(sign, 0)) >= 0.6) for r in ok]
        if ok:
            out["signs"][sign] = {"n": len(ok), "positives": sum(y), "accuracy": round(accuracy_score(y, pred), 3),
                                  "f1": round(f1_score(y, pred, zero_division=0), 3)}
    if ok:
        out["median_latency_ms"] = round(float(np.median([r["latency_ms"] for r in ok])))
    print(json.dumps(out, indent=2))
    data = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    data["gemini_detection"] = out
    RESULTS.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
