import argparse
import json
import time
from urllib.request import Request, urlopen

ROUTE = [
    [30.61183, -96.34085], [30.61313, -96.33893], [30.61463, -96.33828], [30.61593, -96.33636], [30.61747, -96.33495],
    [30.61945, -96.33474], [30.62046, -96.33674], [30.62131, -96.33857], [30.62259, -96.3405], [30.62405, -96.34225],
    [30.62552, -96.34401], [30.62698, -96.34576], [30.62845, -96.34751], [30.62991, -96.34927], [30.63137, -96.351],
    [30.63296, -96.3526], [30.63458, -96.35416], [30.63627, -96.35562], [30.63801, -96.35697], [30.63978, -96.3581],
    [30.64091, -96.35605], [30.64254, -96.35466], [30.64395, -96.35286], [30.64548, -96.35117], [30.64698, -96.34946],
    [30.64863, -96.34804], [30.65065, -96.34797], [30.65274, -96.34824], [30.65476, -96.34843], [30.6556, -96.34686],
]


def post(base, incident_id, lat, lng):
    body = json.dumps({"lat": lat, "lng": lng, "accuracy": 15}).encode()
    request = Request(f"{base}/api/incident/{incident_id}/location", data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=10) as response:
        return response.status


def main():
    p = argparse.ArgumentParser(description="Dev tool: move an incident's location from the MSC along a College Station route.")
    p.add_argument("incident_id")
    p.add_argument("--base", default="http://localhost:8000")
    p.add_argument("--every", type=float, default=2.0)
    a = p.parse_args()
    for i, (lat, lng) in enumerate(ROUTE, 1):
        status = post(a.base.rstrip("/"), a.incident_id, lat, lng)
        print(f"{i:2d}/{len(ROUTE)} {lat:.5f}, {lng:.5f} -> {status}", flush=True)
        if i < len(ROUTE):
            time.sleep(a.every)


if __name__ == "__main__":
    main()
