import json
import os
import re
from functools import lru_cache
from pathlib import Path
from urllib.request import Request, urlopen

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VOICE = "JBFqnCBsd6RMkjVDRZzb"


@lru_cache(maxsize=32)
def generate(key, voice, text):
    request = Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice}?output_format=mp3_22050_32",
        data=json.dumps({"text": text.replace("911", "nine one one"), "model_id": "eleven_flash_v2_5"}).encode(),
        headers={"Content-Type": "application/json", "Accept": "audio/mpeg", "xi-api-key": key},
    )
    with urlopen(request, timeout=4) as response:
        return response.read(1_000_000)


def speech(text):
    load_dotenv(ROOT / ".env")
    key = os.getenv("ELEVENLABS_API_KEY")
    voice = os.getenv("ELEVENLABS_VOICE_ID") or DEFAULT_VOICE
    if not key or not re.fullmatch(r"[a-zA-Z0-9_-]+", voice) or not text or len(text) > 600:
        return None
    try:
        return generate(key, voice, text)
    except OSError:
        return None
