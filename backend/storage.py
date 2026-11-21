import json
import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4

DATA = Path(__file__).resolve().parents[1] / "data"
SENTENCE = "The quick brown fox jumps over the lazy dog near the river bank."
LABELS = {"normal", "face_droop", "arm_drift", "slurred"}


def user_key(user):
    key = re.sub(r"[^a-z0-9]+", "-", user.strip().lower()).strip("-")
    if not key or len(key) > 80:
        raise ValueError("Enter a valid name.")
    return key


def save_session(meta, face, arms, audio, label):
    labels = list(dict.fromkeys(label.split(",")))
    if set(labels) - LABELS or ("normal" in labels and len(labels) > 1):
        raise ValueError("Invalid labels.")
    person = user_key(meta["name"])
    now = datetime.now().astimezone()
    take_id = f"{now:%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
    path = DATA / "sessions" / person / take_id
    path.mkdir(parents=True)
    details = {
        "person": person, "take_id": take_id, "labels": labels,
        "glasses": bool(meta.get("glasses")), "lighting": meta.get("lighting", "good"),
        "sentence": SENTENCE, "created_at": now.isoformat(),
    }
    for name, value in (("face", face), ("arms", arms)):
        (path / f"{name}.json").write_text(json.dumps(value))
    (path / "speech.webm").write_bytes(audio)
    (path / "meta.json").write_text(json.dumps(details))
    return path


def load_sessions():
    return [
        {**json.loads(path.read_text()), "path": path.parent}
        for path in sorted((DATA / "sessions").glob("*/*/meta.json"))
    ]


def contacts(user):
    return []


def save_contacts(user, contacts):
    return contacts
