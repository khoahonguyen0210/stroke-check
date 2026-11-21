import base64
import json
import os
import subprocess
import wave
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from backend import fusion, storage

ROOT = Path(__file__).resolve().parents[1]
MODEL = "gemini-3.5-flash-lite"
STATES = ("possible_sign", "insufficient_evidence", "sign_not_observed")


def review(features, audio):
    load_dotenv(ROOT / ".env")
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return {"status": "unavailable"}
    try:
        pcm = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-t", "8", "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"],
            input=audio, capture_output=True, check=True, timeout=3,
        ).stdout
        buffer = BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            wav.writeframes(pcm)
        measurements = fusion.model_input(features, {})
        parts = [
            {"text": (
                "Review these measurements and spoken audio from a stroke-check demo for observable signs. "
                "For face, assess one-sided smile weakness. For arms, assess one-sided downward drift. "
                "For speech, review articulation in the audio. A transcription mismatch alone is insufficient. "
                "Use insufficient_evidence whenever the evidence is ambiguous. Return only the specified categories. "
                f"The expected sentence is {storage.SENTENCE}\nMeasurements: {json.dumps(measurements)}"
            )},
            {"inlineData": {"mimeType": "audio/wav", "data": base64.b64encode(buffer.getvalue()).decode("ascii")}},
        ]
        schema = {
            "type": "object",
            "properties": {name: {"type": "string", "enum": list(STATES)} for name in fusion.NAMES},
            "required": list(fusion.NAMES),
        }
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "responseMimeType": "application/json", "responseJsonSchema": schema, "temperature": 0,
                "thinkingConfig": {"thinkingLevel": "minimal"},
            },
        }
        request = Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "x-goog-api-key": key},
        )
        with urlopen(request, timeout=8) as response:
            candidate = json.load(response)["candidates"][0]
        text = "".join(part.get("text", "") for part in candidate["content"]["parts"] if not part.get("thought"))
        outcome = json.loads(text)
        if isinstance(outcome, dict) and all(outcome.get(name) in STATES for name in fusion.NAMES):
            return {"status": "reviewed", "signs": {name: outcome[name] for name in fusion.NAMES}}
    except (OSError, subprocess.SubprocessError, KeyError, IndexError, TypeError, ValueError):
        pass
    return {"status": "unavailable"}
