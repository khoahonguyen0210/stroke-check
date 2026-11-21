import json
import os
import re
from copy import deepcopy
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from backend import alerts, fusion, incident, recording, storage
from dotenv import load_dotenv
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI()


@app.get("/api/health")
def health():
    return {"ok": True}


async def read_landmarks(part: UploadFile | str):
    return json.loads(part if isinstance(part, str) else await part.read())


@app.post("/api/check")
async def check(meta: str = Form(), face: UploadFile | str = File(), arms: UploadFile | str = File()):
    try:
        face_data, arms_data = await read_landmarks(face), await read_landmarks(arms)
        if not isinstance(face_data, dict) or not isinstance(arms_data, dict):
            raise ValueError("Expected landmark objects.")
        features, quality, latency = await run_in_threadpool(recording.analyze_vision, face_data, arms_data)
    except (ValueError, TypeError, KeyError, IndexError) as error:
        raise HTTPException(422, "Invalid recording.") from error
    result = fusion.predict_vision(features, quality)
    result["latency_ms"] = latency
    return result


@app.post("/api/collect")
async def collect(meta: str = Form(), face: UploadFile | str = File(), arms: UploadFile | str = File(), label: str = Form()):
    try:
        path = storage.save_session(json.loads(meta), await read_landmarks(face), await read_landmarks(arms), None, label)
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(422, "Invalid session data.") from error
    return {"path": (Path("data") / path.relative_to(storage.DATA)).as_posix()}


@app.post("/api/baseline/{user}")
async def baseline(user: str, meta: str = Form(), face: UploadFile | str = File(), arms: UploadFile | str = File()):
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
    return save_agent_response(response)


def save_agent_response(response):
    session = response["state"].get("session")
    if session is not None:
        AGENT_SESSIONS[str(session)] = (deepcopy(response["state"]), response["handoff"])
    return response

@app.post("/api/agent/start")
def agent_start(payload: dict):
    if not isinstance(payload.get("result"), dict):
        raise HTTPException(422, "Check result is required.")
    if payload.get("guided") is True:
        from backend import care

        return save_agent_response(care.start(payload["result"]))
    return agent_response("start", payload["result"], payload.get("language", "en"))


@app.post("/api/agent/message")
def agent_message(payload: dict):
    if not isinstance(payload.get("state"), dict):
        raise HTTPException(422, "Message and agent state are required.")
    session = str(payload["state"].get("session", ""))
    if session not in AGENT_SESSIONS:
        raise HTTPException(404, "Assistant session has expired.")
    state, _ = AGENT_SESSIONS[session]
    text = payload.get("text", "")
    if not isinstance(text, str) or len(text) > 1500:
        raise HTTPException(422, "Enter a message of up to 1500 characters.")
    text = text.strip()
    if state.get("guided"):
        from backend import care

        next_state = deepcopy(state)
        kind = payload.get("kind")
        if kind == "question" and text:
            return save_agent_response(care.question(next_state, text))
        if kind in ("answer", "skip") and (text or kind == "skip"):
            if not state["pending_question"] or payload.get("field") != state["pending_question"]:
                raise HTTPException(409, "This handoff question has changed.")
            return save_agent_response(care.answer(next_state, text, kind == "skip"))
        raise HTTPException(422, "Enter an answer, skip the question, or ask a care question.")
    if not text:
        raise HTTPException(422, "Enter a message.")
    return agent_response("reply", text, deepcopy(state), payload.get("language", "en"))


@app.get("/api/agent/voice/{session}")
def agent_voice(session: str):
    from backend import voice

    entry = AGENT_SESSIONS.get(session)
    if not entry:
        raise HTTPException(404, "Voice session not found.")
    turns = entry[0]["conversation"]
    audio = voice.speech(turns[-1]["text"]) if turns and turns[-1]["role"] == "agent" else None
    if not audio:
        raise HTTPException(503, "ElevenLabs voice is unavailable.")
    return Response(audio, media_type="audio/mpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/agent/handoff/{session}")
def handoff(session: str):
    from fastapi.responses import PlainTextResponse

    if session not in AGENT_SESSIONS:
        if re.fullmatch(r"[0-9]{8}-[0-9]{6}(?:-[0-9a-f]{8})?", session):
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
    incident: str | None = None


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


def public_base(request):
    load_dotenv(ROOT / ".env")
    return (os.getenv("PUBLIC_BASE_URL") or str(request.base_url)).rstrip("/")


@app.post("/api/alert")
def send_alert(payload: AlertRequest, request: Request):
    validate_user(payload.user)
    link = f"{public_base(request)}/family/{payload.incident}" if payload.incident and incident.load(payload.incident) else None
    sent = alerts.send(payload.user, payload.result, payload.detected_at, payload.location, link)
    if link:
        incident.contacts_alerted(payload.incident, sent)
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


class IncidentRequest(BaseModel):
    user: str
    name: str = Field(default="", max_length=80)
    phone: str | None = Field(default=None, max_length=40)
    result: dict


class LocationUpdate(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)
    accuracy: float | None = Field(default=None, ge=0)


@app.post("/api/incident")
def create_incident(payload: IncidentRequest):
    validate_user(payload.user)
    phone = re.sub(r"[^0-9+]", "", payload.phone or "") or None
    return {"id": incident.create(payload.user, payload.name.strip(), phone, payload.result)}


@app.post("/api/incident/{incident_id}/location")
def incident_location(incident_id: str, payload: LocationUpdate):
    if not incident.set_location(incident_id, payload.lat, payload.lng, payload.accuracy):
        raise HTTPException(404, "This link has expired.")
    return {"ok": True}


@app.get("/api/incident/{incident_id}")
def incident_view(incident_id: str):
    view = incident.public(incident_id)
    if not view:
        raise HTTPException(404, "This link has expired.")
    return view


@app.get("/family/{incident_id}")
def family_page(incident_id: str):
    return FileResponse(ROOT / "frontend" / "family.html", headers={"Cache-Control": "no-store"})


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True))
