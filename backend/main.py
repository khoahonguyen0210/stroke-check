import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from backend import fusion, recording, storage

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


@app.get("/api/contacts/{user}")
def get_contacts(user: str):
    return []


@app.post("/api/contacts/{user}")
def post_contacts(user: str, payload: dict):
    return []


@app.post("/api/alert")
@app.post("/api/alert/test/{user}")
def alert_stub(user: str = "", payload: dict | None = None):
    return {"sent": []}


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True))
