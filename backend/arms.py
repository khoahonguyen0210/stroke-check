import numpy as np


SIDES = {"Left": (11, 13, 15), "Right": (12, 14, 16)}


def visible(frame):
    pose = frame.get("pose", [])
    return len(pose) >= 33 and all(
        len(pose[index]) >= 4 and pose[index][3] >= 0.5
        for indices in SIDES.values() for index in indices
    )


def quality(recording):
    frames = recording.get("frames", [])
    valid = [frame for frame in frames if visible(frame)]
    fraction = len(valid) / max(1, len(frames))
    duration = valid[-1]["t"] - valid[0]["t"] if valid else 0.0
    ok = fraction >= 0.75 and duration >= 7
    return {"ok": ok, "reason": None if ok else "Keep both arms visible for the full hold.", "visible_fraction": fraction}


def features(recording):
    measurements = {side: [] for side in SIDES}
    for frame in recording.get("frames", []):
        if not visible(frame):
            continue
        pose = np.asarray(frame["pose"])[:, :3]
        scale = np.linalg.norm(pose[11] - pose[12])
        if scale < 0.01:
            continue
        hands = {hand["side"]: np.asarray(hand["landmarks"]) for hand in frame.get("hands", [])}
        for side, (s, e, w) in SIDES.items():
            shoulder, elbow, wrist = pose[[s, e, w]]
            upper, lower = shoulder - elbow, wrist - elbow
            denominator = np.linalg.norm(upper) * np.linalg.norm(lower)
            bend = float(np.pi - np.arccos(np.clip(np.dot(upper, lower) / denominator, -1, 1))) if denominator > 0 else 0.0
            hand = hands.get(side)
            normal = None
            if hand is not None and len(hand) >= 21:
                cross = np.cross(hand[5] - hand[0], hand[17] - hand[0])
                length = np.linalg.norm(cross)
                if length > 0:
                    normal = cross / length
            measurements[side].append((frame["t"], (wrist[1] - shoulder[1]) / scale, bend, normal))
    output = {}
    for side, rows in measurements.items():
        if len(rows) < 2:
            return {}
        t, height, bend, normals = zip(*rows)
        size = max(1, len(rows) // 5)
        palm = [normal for normal in normals if normal is not None]
        rotation = np.arccos(np.clip(np.dot(palm[0], palm[-1]), -1, 1)) if palm else 0.0
        prefix = side.lower()
        output[f"{prefix}_drop"] = float(np.mean(height[-size:]) - np.mean(height[:size]))
        output[f"{prefix}_slope"] = float(np.polyfit(t, height, 1)[0]) if t[-1] > t[0] else 0.0
        output[f"{prefix}_bend"] = float(np.mean(bend))
        output[f"{prefix}_rotation"] = float(rotation)
        output[f"{prefix}_hand_fraction"] = len(palm) / len(rows)
    output["drop_difference"] = output["left_drop"] - output["right_drop"]
    output["rotation_difference"] = output["left_rotation"] - output["right_rotation"]
    return output


def trace(recording):
    rows = []
    for frame in recording.get("frames", []):
        if not visible(frame):
            continue
        pose = np.asarray(frame["pose"])[:, :3]
        scale = np.linalg.norm(pose[11] - pose[12])
        if scale < 0.01:
            continue
        rows.append((frame["t"], (pose[15][1] - pose[11][1]) / scale, (pose[16][1] - pose[12][1]) / scale))
    if len(rows) < 2:
        return {}
    step = max(1, len(rows) // 150)
    rows = rows[::step]
    t0 = rows[0][0]
    return {
        "t": [round(r[0] - t0, 2) for r in rows],
        "left": [round(float(r[1] - rows[0][1]), 4) for r in rows],
        "right": [round(float(r[2] - rows[0][2]), 4) for r in rows],
    }
