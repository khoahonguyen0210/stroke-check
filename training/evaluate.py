import json
import time
from math import isfinite
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from backend import fusion, recording, storage
from training.train_fusion import classifier

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "models" / "results.json"
CACHE = ROOT / "data" / "eval_features.json"
FLAG = 0.6


def numeric(values):
    return {k: float(v) for k, v in values.items() if isinstance(v, (int, float)) and isfinite(v)}


def build_rows():
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    rows = []
    for s in storage.load_sessions():
        key = str(s["path"])
        if key not in cache:
            face = json.loads((s["path"] / "face.json").read_text())
            arms = json.loads((s["path"] / "arms.json").read_text())
            t = time.time()
            features, quality, latency = recording.analyze(face, arms, webm_path=s["path"] / "speech.webm", sentence=s["sentence"])
            cache[key] = {"features": features, "quality": quality, "latency": latency, "seconds": round(time.time() - t, 2)}
            CACHE.write_text(json.dumps(cache, default=str))
            print(f"analyzed {s['person']}/{s['take_id']} in {cache[key]['seconds']} s", flush=True)
        c = cache[key]
        rows.append({
            "person": s["person"], "labels": s["labels"], "glasses": bool(s.get("glasses")), "lighting": s.get("lighting", "good"),
            "ok": all(q.get("ok") for q in c["quality"].values()),
            "inputs": {
                "face": {f"face.{k}": v for k, v in numeric(c["features"].get("face", {})).items()},
                "arms": {f"arms.{k}": v for k, v in numeric(c["features"].get("arms", {})).items()},
                "speech": {f"speech.{k}": v for k, v in numeric(c["features"].get("speech", {})).items()},
                "fusion": fusion.model_input(c["features"], c["quality"]),
            },
            "latency": c["latency"],
        })
    return rows


def lopo(rows, model_input, label):
    people = sorted({r["person"] for r in rows})
    y = np.array([int(label in r["labels"]) for r in rows])
    p = np.full(len(rows), np.nan)
    for person in people:
        test = np.array([r["person"] == person for r in rows])
        train_y = y[~test]
        if min((train_y == 0).sum(), (train_y == 1).sum()) < 2:
            return None, y, f"leaving out {person} leaves fewer than 2 takes of one class"
        model = classifier().fit([r["inputs"][model_input] for r, t in zip(rows, test) if not t], train_y)
        p[test] = model.predict_proba([r["inputs"][model_input] for r, t in zip(rows, test) if t])[:, 1]
    return p, y, None


def metrics(y, p):
    pred = (p >= FLAG).astype(int)
    return {
        "n": int(len(y)), "positives": int(y.sum()),
        "accuracy": round(float(accuracy_score(y, pred)), 3),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 3),
        "uncertain_rate": round(float(((p >= 0.4) & (p < FLAG)).mean()), 3),
    }


def subsets(rows):
    return {
        "glasses": [r["glasses"] for r in rows],
        "no_glasses": [not r["glasses"] for r in rows],
        "good_light": [r["lighting"] == "good" for r in rows],
        "low_light": [r["lighting"] != "good" for r in rows],
    }


def main():
    all_rows = build_rows()
    rows = [r for r in all_rows if r["ok"]]
    print(f"{len(all_rows)} sessions, {len(rows)} pass quality, people: {sorted({r['person'] for r in rows})}", flush=True)
    detection = {
        "sessions": len(all_rows), "usable_sessions": len(rows),
        "people": sorted({r["person"] for r in rows}),
        "method": "leave-one-person-out; flag at p >= 0.6; same classifier as training/train_fusion.py",
        "signs": {}, "robustness": {},
    }
    plan = {"face_droop": ["face", "fusion"], "arm_drift": ["arms", "fusion"], "slurred": ["speech", "fusion"]}
    for label, inputs in plan.items():
        detection["signs"][label] = {}
        for model_input in inputs:
            p, y, error = lopo(rows, model_input, label)
            if error:
                detection["signs"][label][model_input] = {"skipped": error}
                print(f"{label:11} {model_input:7} skipped: {error}", flush=True)
                continue
            m = metrics(y, p)
            detection["signs"][label][model_input] = m
            print(f"{label:11} {model_input:7} {m}", flush=True)
            if model_input == "fusion":
                detection["robustness"][label] = {
                    name: metrics(y[np.array(mask)], p[np.array(mask)]) for name, mask in subsets(rows).items() if sum(mask) > 0
                }
    latencies = {k: round(float(np.median([r["latency"][k] for r in all_rows if k in r["latency"]])), 1)
                 for k in ("face", "arms", "speech") if any(k in r["latency"] for r in all_rows)}
    detection["median_latency_ms"] = latencies
    data = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    if "speech" in data:
        detection["speech_model_torgo"] = {"held_out_speakers": data["speech"].get("held_out"), "loso_all_speakers": data["speech"].get("loso_all_speakers")}
    data["detection"] = detection
    RESULTS.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"robustness: {json.dumps(detection['robustness'])}")
    print(f"median latency ms: {latencies}")
    print(f"saved {RESULTS}")


if __name__ == "__main__":
    main()
