from functools import lru_cache
from math import isfinite
from pathlib import Path


MODEL = Path(__file__).resolve().parents[1] / "models" / "fusion.joblib"
NAMES = {"face": "face_droop", "arms": "arm_drift", "speech": "slurred"}


def model_input(features, quality):
    return {
        f"{group}.{key}": float(value)
        for group, values in {**features, **{f"{name}_quality": item for name, item in quality.items()}}.items()
        for key, value in values.items()
        if isinstance(value, (int, float)) and isfinite(value)
    }


@lru_cache(maxsize=1)
def load_models(modified_at):
    import joblib

    return joblib.load(MODEL)


def predict(features, quality):
    speech_score = features.get("speech", {}).get("p_slurred")
    scores = {
        "face": min(1.0, features.get("face", {}).get("smile_asymmetry", 0) * 1.2),
        "arms": min(1.0, abs(features.get("arms", {}).get("drop_difference", 0)) * 1.5),
        "speech": speech_score,
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
    speech = signs["speech"]
    if (
        speech["p"] is not None
        and isinstance(speech_score, (int, float))
        and isfinite(speech_score)
        and speech["flag"] != (speech_score >= 0.6)
    ):
        speech.update(flag=False, uncertain=True)
        disagreement = "Speech scores disagree. Repeat the sentence close to the microphone."
        reason = f"{reason} {disagreement}" if reason else disagreement
    if not reason and any(item["p"] is None for item in signs.values()):
        reason = "One or more checks could not be completed."
    if not reason and any(item["p"] is not None and 0.4 <= item["p"] < 0.6 for item in signs.values()):
        reason = "One or more scores are uncertain."
    overall = "warning" if any(item["flag"] for item in signs.values()) else "retake" if reason else "clear"
    return {"overall": overall, "retake_reason": reason, "signs": signs, "features": features, "latency_ms": {}}



def predict_vision(features, quality):
    face = features["face"]
    arms = features["arms"]
    face_threshold = 0.5
    arm_threshold = 0.35
    left_drop = arms.get("left_drop", 0)
    right_drop = arms.get("right_drop", 0)
    scores = {
        "face": face.get("smile_asymmetry") if max(face.get("smile_left", 0), face.get("smile_right", 0)) >= 0.05 else None,
        "arms": abs(max(0, left_drop) - max(0, right_drop)) if arms else None,
    }
    thresholds = {"face": face_threshold, "arms": arm_threshold}
    reasons = {
        "face": "Smile fully with your face centered and well lit.",
        "arms": "Keep both wrists in view for the full hold.",
    }
    signs = {}
    for name, score in scores.items():
        valid = quality[name]["ok"] and score is not None and isfinite(score)
        flag = valid and score >= thresholds[name]
        signs[name] = {
            "flag": flag,
            "p": None,
            "status": "possible_sign" if flag else "not_observed" if valid else "unavailable",
            "measurement": score if valid else None,
        }
    warning = any(sign["flag"] for sign in signs.values())
    incomplete = any(sign["status"] == "unavailable" for sign in signs.values())
    notes = [quality[name].get("reason") or reasons[name] for name, sign in signs.items() if sign["status"] == "unavailable"]
    return {
        "overall": "warning" if warning else "retake" if incomplete else "clear",
        "retake_reason": " ".join(notes),
        "signs": signs,
        "features": features,
    }
