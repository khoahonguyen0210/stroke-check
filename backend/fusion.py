from functools import lru_cache
from math import isfinite
from pathlib import Path


MODEL = Path(__file__).resolve().parents[1] / "models" / "fusion.joblib"
NAMES = {"face": "face_droop", "arms": "arm_drift", "speech": "slurred"}


def model_input(features, quality):
    return {
        f"{group}.{key}": float(value)
        for group, values in {**features, **{f"{name}_quality": item for name, item in quality.items()}}.items()
        if group not in {"arms", "arms_quality"}
        for key, value in values.items()
        if isinstance(value, (int, float)) and isfinite(value)
    }


@lru_cache(maxsize=1)
def load_models(modified_at):
    import joblib

    return joblib.load(MODEL)


def predict(features, quality):
    scores = {
        "face": min(1.0, features.get("face", {}).get("smile_asymmetry", 0) * 1.2),
        "arms": min(1.0, abs(features.get("arms", {}).get("drop_difference", 0)) * 1.5),
        "speech": features.get("speech", {}).get("p_slurred"),
    }
    if MODEL.is_file():
        values = model_input(features, quality)
        for name, model in load_models(MODEL.stat().st_mtime_ns).items():
            scores[name] = float(model.predict_proba([values])[0, 1])
    signs = {}
    for name, score in scores.items():
        valid = quality.get(name, {}).get("ok") and isinstance(score, (int, float)) and isfinite(score)
        signs[name] = {"flag": bool(valid and score >= 0.6), "p": score if valid else None}
    reason = next((item.get("reason") for item in quality.values() if not item.get("ok")), None)
    if not reason and any(item["p"] is None for item in signs.values()):
        reason = "One or more checks could not be completed."
    if not reason and any(item["p"] is not None and 0.4 <= item["p"] < 0.6 for item in signs.values()):
        reason = "One or more scores are uncertain."
    overall = "warning" if any(item["flag"] for item in signs.values()) else "retake" if reason else "clear"
    return {"overall": overall, "retake_reason": reason, "signs": signs, "features": features, "latency_ms": {}}
