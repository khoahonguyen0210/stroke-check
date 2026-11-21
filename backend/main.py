import json
import re
from copy import deepcopy
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from backend import alerts, fusion, recording, storage
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI()


@app.get("/api/health")
def health():
    return {"ok": True}


async def read_landmarks(part: UploadFile | str):
    return json.loads(part if isinstance(part, str) else await part.read())


@app.post("/api/check")
async def check(meta: str = Form(), face: UploadFile | str = File(), arms: UploadFile | str = File(), audio: UploadFile = File()):
    try:
        face_data, arms_data = await read_landmarks(face), await read_landmarks(arms)
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
async def collect(meta: str = Form(), face: UploadFile | str = File(), arms: UploadFile | str = File(), audio: UploadFile = File(), label: str = Form()):
    try:
        path = storage.save_session(json.loads(meta), await read_landmarks(face), await read_landmarks(arms), await audio.read(), label)
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(422, "Invalid session data.") from error
    return {"path": (Path("data") / path.relative_to(storage.DATA)).as_posix()}


@app.post("/api/baseline/{user}")
async def baseline(user: str, meta: str = Form(), face: UploadFile | str = File(), arms: UploadFile | str = File(), audio: UploadFile = File()):
    return {"saved": False}


AGENT_SESSIONS = {}
ALERTED_SESSIONS = set()


def agent_response(operation, *args):
    session = str(args[1].get("session", "")) if operation == "reply" else ""
    token = alerts.FOLLOWUP_ALLOWED.set(session in ALERTED_SESSIONS)
    try:
        from backend import agent

        response = getattr(agent, operation)(*args)
    except (ImportError, AttributeError):
        state = args[1] if operation == "reply" else {}
        response = {"say": "Call 911 now and follow the dispatcher's instructions.", "actions": [], "handoff": {}, "state": state}
    finally:
        alerts.FOLLOWUP_ALLOWED.reset(token)
    session = response["state"].get("session")
    if session is not None:
        AGENT_SESSIONS[str(session)] = (deepcopy(response["state"]), response["handoff"])
    return response


@app.post("/api/agent/start")
def agent_start(payload: dict):
    if not isinstance(payload.get("result"), dict):
        raise HTTPException(422, "Check result is required.")
    return agent_response("start", payload["result"], payload.get("language", "en"))


@app.post("/api/agent/message")
def agent_message(payload: dict):
    if not isinstance(payload.get("state"), dict) or not isinstance(payload.get("text"), str):
        raise HTTPException(422, "Message and agent state are required.")
    session = str(payload["state"].get("session", ""))
    if session not in AGENT_SESSIONS:
        raise HTTPException(404, "Assistant session has expired.")
    state, _ = AGENT_SESSIONS[session]
    return agent_response("reply", payload["text"], deepcopy(state), payload.get("language", "en"))


@app.get("/api/agent/handoff/{session}")
def handoff(session: str):
    from fastapi.responses import PlainTextResponse

    if session not in AGENT_SESSIONS:
        if re.fullmatch(r"[0-9]{8}-[0-9]{6}", session):
            path = storage.DATA / "handoffs" / f"{session}.txt"
            if path.is_file():
                return PlainTextResponse(path.read_text())
        raise HTTPException(404, "Handoff not found.")
    _, details = AGENT_SESSIONS[session]
    if details.get("timeline") and details.get("text"):
        return PlainTextResponse(details["text"])
    facts = details.get("facts", {})
    signs = details.get("signs", {})
    lines = ["Stroke Check paramedic handoff"]
    if details.get("time_noticed"):
        lines.append(f"Time noticed: {details['time_noticed']}")
    lines.extend(f"{name.replace('_', ' ')}: {value}" for name, value in signs.items())
    lines.extend(f"{name.replace('_', ' ')}: {value}" for name, value in facts.items())
    return PlainTextResponse("\n".join(lines))


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
    session: str | None = None


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
    sent = alerts.send(payload.user, payload.result, payload.detected_at, payload.location)
    session = AGENT_SESSIONS.get(payload.session)
    if session and session[0].get("user") == payload.user and any(item["ok"] for item in sent):
        ALERTED_SESSIONS.add(payload.session)
    return {"sent": sent}


@app.post("/api/alert/test/{user}")
def test_alert(user: str):
    validate_user(user)
    return {"sent": alerts.test(user)}


@app.get("/api/benchmarks")
def benchmarks():
    path = ROOT / "models" / "results.json"
    if not path.is_file():
        raise HTTPException(404, "Benchmark results are not available yet.")
    return FileResponse(path, media_type="application/json", headers={"Cache-Control": "no-store"})


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True))
