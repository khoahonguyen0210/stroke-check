def predict(features, quality):
    return {
        "overall": "retake",
        "retake_reason": "Analysis is unavailable.",
        "signs": {name: {"flag": False, "p": None} for name in ("face", "arms", "speech")},
        "features": features,
        "latency_ms": {},
    }
