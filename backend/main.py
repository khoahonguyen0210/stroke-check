import json
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from backend import alerts, fusion, recording, storage
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI()


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/check")
async def check(meta: str = Form(), face: str = Form(), arms: str = Form(), audio: UploadFile = File()):
    try:
        face_data, arms_data = json.loads(face), json.loads(arms)
        if not isinstance(face_data, dict) or not isinstance(arms_data, dict):
            raise ValueError("Expected landmark objects.")
        content = await audio.read()
        features, quality, latency = await run_in_threadpool(recording.analyze, face_data, arms_data, content)
    except (ValueError, TypeError, KeyError, IndexError) as error:
        raise HTTPException(422, "Invalid recording.") from error
    result = await run_in_threadpool(fusion.predict, features, quality)
    result["latency_ms"] = latency
    return result


@app.post("/api/collect")
async def collect(meta: str = Form(), face: str = Form(), arms: str = Form(), audio: UploadFile = File(), label: str = Form()):
    try:
        path = storage.save_session(json.loads(meta), json.loads(face), json.loads(arms), await audio.read(), label)
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(422, "Invalid session data.") from error
    return {"path": str(path.relative_to(ROOT))}


@app.post("/api/baseline/{user}")
async def baseline(user: str, meta: str = Form(), face: str = Form(), arms: str = Form(), audio: UploadFile = File()):
    return {"saved": False}


@app.post("/api/agent/start")
@app.post("/api/agent/message")
def agent_stub(payload: dict):
    return {"say": "Call 911 now and follow the dispatcher's instructions.", "actions": [], "handoff": {}, "state": {}}


@app.get("/api/agent/handoff/{session}")
def handoff(session: str):
    from fastapi.responses import PlainTextResponse

    return PlainTextResponse("No handoff available.")


CONTACT_LOCK = Lock()


class Contact(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    relationship: str = Field(min_length=1, max_length=80)
    email: str = Field(max_length=254, pattern=r"^[^\s@,;<>]+@[^\s@,;<>]+\.[^\s@,;<>]+$")
    phone: str | None = Field(default=None, max_length=40)
    consent: bool = False


class AlertRequest(BaseModel):
    user: str
    result: dict
    detected_at: str
    location: dict | None = None


def validate_user(user):
    try:
        storage.user_key(user)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@app.get("/api/contacts/{user}")
def get_contacts(user: str):
    validate_user(user)
    return storage.contacts(user)


@app.post("/api/contacts/{user}")
def post_contacts(user: str, payload: Contact):
    validate_user(user)
    if not payload.consent:
        raise HTTPException(422, "Contact consent is required.")
    with CONTACT_LOCK:
        contacts = storage.contacts(user)
        contacts = [item for item in contacts if item["email"].casefold() != payload.email.casefold()]
        if len(contacts) >= 3:
            raise HTTPException(409, "You can save up to three contacts.")
        return storage.save_contacts(user, contacts + [payload.model_dump()])


@app.post("/api/alert")
def send_alert(payload: AlertRequest):
    validate_user(payload.user)
    return {"sent": alerts.send(payload.user, payload.result, payload.detected_at, payload.location)}


@app.post("/api/alert/test/{user}")
def test_alert(user: str):
    validate_user(user)
    return {"sent": alerts.test(user)}


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True))
