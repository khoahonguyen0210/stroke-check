import json
import math
import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from tempfile import NamedTemporaryFile
from threading import Lock, Thread
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from backend import storage

DIR = storage.DATA / "incidents"
LOCK = Lock()
TTL = timedelta(hours=24)
OVERPASS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
CACHE = storage.DATA / "hospital_cache.json"
ID = r"[A-Za-z0-9_-]{24,64}"
COMMON = {"hospital", "health", "medical", "center", "centre", "the", "regional", "of", "and", "chi", "&"}


def now():
    return datetime.now(timezone.utc)


def stamp():
    return now().isoformat(timespec="seconds")


def path(incident_id):
    return DIR / f"{incident_id}.json" if re.fullmatch(ID, incident_id or "") else None


def load(incident_id):
    p = path(incident_id)
    if not p or not p.is_file():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return data if datetime.fromisoformat(data["expires_at"]) > now() else None


def save(data):
    DIR.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=DIR, delete=False, encoding="utf-8", suffix=".tmp") as file:
        json.dump(data, file, ensure_ascii=False)
    os.replace(file.name, path(data["id"]))


def add_event(data, key, text):
    if any(e["key"] == key for e in data["timeline"]):
        return
    data["timeline"].append({"key": key, "t": stamp(), "event": text})


def create(user, name, phone, result):
    incident_id = secrets.token_urlsafe(24)
    signs = [k for k, v in (result.get("signs") or {}).items() if isinstance(v, dict) and v.get("flag")]
    data = {
        "id": incident_id, "user": user, "name": name or user, "phone": phone or None,
        "created_at": stamp(), "expires_at": (now() + TTL).isoformat(timespec="seconds"),
        "signs": signs, "timeline": [], "location": None, "hospitals": None,
        "destination_hospital": None, "handoff": None,
    }
    add_event(data, "warning", "Possible stroke signs detected: " + (", ".join({"face": "face droop", "arms": "arm drift", "speech": "slurred speech"}.get(s, s) for s in signs) or "warning"))
    with LOCK:
        save(data)
    return incident_id


def distance_km(a_lat, a_lng, b_lat, b_lng):
    r = 6371.0
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp, dl = p2 - p1, math.radians(b_lng - a_lng)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def query_overpass(lat, lng):
    query = (f'[out:json][timeout:40];(node["amenity"="hospital"](around:15000,{lat},{lng});'
             f'way["amenity"="hospital"](around:15000,{lat},{lng});relation["amenity"="hospital"](around:15000,{lat},{lng}););out center tags;')
    for url in OVERPASS:
        try:
            request = Request(url, data=urlencode({"data": query}).encode(), headers={"User-Agent": "stroke-check-hackathon"})
            with urlopen(request, timeout=45) as response:
                return json.load(response)["elements"]
        except Exception:
            continue
    raise OSError("Overpass is unavailable")


def cache_key(lat, lng):
    return f"{round(lat, 1)},{round(lng, 1)}"


def cached_hospitals(lat, lng):
    if not CACHE.is_file():
        return None
    return json.loads(CACHE.read_text(encoding="utf-8")).get(cache_key(lat, lng))


def fetch_hospitals(lat, lng):
    cached = cached_hospitals(lat, lng)
    if cached is not None:
        elements = [{"tags": {"name": h["name"]}, "lat": h["lat"], "lon": h["lng"]} for h in cached]
    else:
        elements = query_overpass(lat, lng)
    hospitals = []
    for e in elements:
        point = e.get("center", e)
        name = e.get("tags", {}).get("name")
        if name and "lat" in point and "lon" in point:
            hospitals.append({"name": name, "lat": point["lat"], "lng": point["lon"],
                              "km": round(distance_km(lat, lng, point["lat"], point["lon"]), 2)})
    hospitals = sorted(hospitals, key=lambda h: h["km"])[:30]
    with LOCK:
        cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.is_file() else {}
        cache[cache_key(lat, lng)] = [{"name": h["name"], "lat": h["lat"], "lng": h["lng"]} for h in hospitals]
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return hospitals


def load_hospitals(incident_id, lat, lng):
    try:
        hospitals = fetch_hospitals(lat, lng)
    except Exception:
        hospitals = []
    with LOCK:
        data = load(incident_id)
        if data:
            data["hospitals"] = hospitals
            if data["destination_hospital"] and data["destination_hospital"].get("lat") is None:
                data["destination_hospital"] = match_hospital(data["destination_hospital"]["said"], data)
                name = data["destination_hospital"]["name"]
                add_event(data, f"destination:{name}", f"Paramedics are going to {name}")
            save(data)


def set_location(incident_id, lat, lng, accuracy):
    with LOCK:
        data = load(incident_id)
        if not data:
            return False
        data["location"] = {"lat": lat, "lng": lng, "accuracy": accuracy, "at": stamp()}
        start = data["hospitals"] is None
        if start:
            data["hospitals"] = "loading"
        save(data)
    if start:
        Thread(target=load_hospitals, args=(incident_id, lat, lng), daemon=True).start()
    return True


def name_words(name):
    text = re.sub(r"\bst\b\.?", "saint", (name or "").lower())
    return {w for w in re.findall(r"[a-z0-9]+", text) if w not in COMMON}


def match_hospital(said, data):
    hospitals = data["hospitals"] if isinstance(data["hospitals"], list) else []
    wanted = name_words(said)
    best, best_score = None, 0.0
    for h in hospitals:
        overlap = len(wanted & name_words(h["name"]))
        if not overlap:
            continue
        score = overlap + SequenceMatcher(None, " ".join(sorted(wanted)), " ".join(sorted(name_words(h["name"])))).ratio() - h["km"] / 1000
        if score > best_score:
            best, best_score = h, score
    if not best:
        return {"name": said, "said": said, "lat": None, "lng": None}
    return {**best, "said": said}


def contacts_alerted(incident_id, sent):
    count = sum(1 for s in sent if s.get("ok"))
    if not count:
        return
    with LOCK:
        data = load(incident_id)
        if data:
            add_event(data, "contacts", f"Trusted contacts alerted ({count})")
            save(data)


def update_from_agent(incident_id, state, handoff):
    from backend import agent_tools

    with LOCK:
        data = load(incident_id)
        if not data:
            return
        if agent_tools.called_911(state):
            add_event(data, "911", "911 called")
        for update in state.get("notified", []):
            add_event(data, f"notify:{update}", f"Contacts updated: {update}")
        if state.get("paramedics_here"):
            add_event(data, "paramedics", "Paramedics are here")
        said = state["facts"].get("destination_hospital")
        if said and (not data["destination_hospital"] or data["destination_hospital"].get("said") != said):
            match = match_hospital(said, data)
            if match:
                data["destination_hospital"] = match
                add_event(data, f"destination:{match['name']}", f"Paramedics are going to {match['name']}")
        data["handoff"] = {
            "time_noticed": handoff.get("time_noticed"),
            "signs": list(handoff.get("signs", {}).keys()),
            "facts": {k: v for k, v in handoff.get("facts", {}).items() if v},
            "verified": handoff.get("verified"),
        }
        save(data)


def public(incident_id):
    data = load(incident_id)
    if not data:
        return None
    return {
        "name": data["name"], "phone": data["phone"],
        "status_timeline": [{"t": e["t"], "event": e["event"]} for e in data["timeline"]],
        "location": data["location"], "destination_hospital": data["destination_hospital"],
        "handoff": data["handoff"], "hospitals": data["hospitals"] if isinstance(data["hospitals"], list) else [],
        "hospitals_loading": data["hospitals"] == "loading", "expires_at": data["expires_at"],
    }
