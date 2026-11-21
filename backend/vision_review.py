import base64
import binascii
import json
from time import perf_counter

from backend import agent

JPEG_PREFIX = "data:image/jpeg;base64,"
OBSERVATIONS = {"asymmetry_seen", "not_seen", "inconclusive"}
SYSTEM = (
    "Compare mouth-corner motion in three photos of the same face. The first is neutral. "
    "The other two show the person trying to smile. Compare lift relative to the neutral photo. "
    "Report only visible movement. If the face is obscured, the pose changes, or movement cannot be compared, use inconclusive. "
    "Use left or right from the viewer's perspective in the image. Do not diagnose or give advice. "
    'Return JSON only: {"observation":"asymmetry_seen|not_seen|inconclusive","weaker_side":"image_left|image_right|unknown"}. '
    'When observation is not asymmetry_seen, use unknown for weaker_side.'
)


def valid_frame(frame):
    if not frame.startswith(JPEG_PREFIX):
        return False
    try:
        raw = base64.b64decode(frame[len(JPEG_PREFIX):], validate=True)
    except (ValueError, binascii.Error):
        return False
    return 100 <= len(raw) <= 96_000 and raw.startswith(b"\xff\xd8\xff") and raw.endswith(b"\xff\xd9")


def review(frames):
    start = perf_counter()
    client, model = agent.client_for("super")
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": "Neutral face, first smile frame, second smile frame, in that order."},
            *({"type": "image_url", "image_url": {"url": frame}} for frame in frames),
        ]},
    ]
    response = agent.call_model(client, model, messages, False, "super", timeout=10, tools=False)
    text = (response.choices[0].message.content or "").strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.removeprefix("```").removeprefix("json").removesuffix("```").strip()
    result = json.loads(text)
    observation = result["observation"]
    side = result.get("weaker_side")
    if observation not in OBSERVATIONS:
        raise ValueError("Unexpected vision observation")
    if observation != "asymmetry_seen" or side not in ("image_left", "image_right"):
        side = None
    return {"observation": observation, "weaker_side": side, "latency_ms": round((perf_counter() - start) * 1000)}
