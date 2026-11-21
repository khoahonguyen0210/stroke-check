import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from backend import fusion, storage

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI()


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/check")
async def check(meta: str = Form(), face: str = Form(), arms: str = Form(), audio: UploadFile = File()):
    return fusion.predict({}, {})


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
