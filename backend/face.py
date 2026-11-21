from math import atan2, degrees, hypot
from statistics import mean

SIDES = ((291, 263, 334, 386, 374), (61, 33, 105, 159, 145))


def measurements(recording):
    for frame in recording.get("frames", []):
        points = frame.get("landmarks", [])
        if len(points) < 468:
            continue
        right, left = points[33], points[263]
        dx, dy = left[0] - right[0], left[1] - right[1]
        scale = hypot(dx, dy)
        if scale < 0.01:
            continue

        def height(index):
            x, y = points[index][:2]
            return ((x - right[0]) * dy - (y - right[1]) * dx) / scale**2

        sides = [
            (height(mouth), height(brow) - height(eye), abs(height(top) - height(bottom)))
            for mouth, eye, brow, top, bottom in SIDES
        ]
        yield frame["phase"], frame["t"], degrees(atan2(dy, dx)), sides


def quality(recording):
    rows = list(measurements(recording))
    fraction = len(rows) / max(1, len(recording.get("frames", [])))
    tilt = [row[2] for row in rows]
    tilt_range = max(tilt) - min(tilt) if tilt else 0.0
    phases = {phase: sum(row[0] == phase for row in rows) for phase in ("relax", "smile", "brows")}
    ok = fraction >= 0.75 and tilt_range <= 20 and min(phases.values()) >= 5
    return {"ok": ok, "reason": None if ok else "Keep your face centered and improve the lighting.", "detected_fraction": fraction, "head_tilt_range": tilt_range}


def features(recording):
    rows = list(measurements(recording))
    groups = {phase: [row for row in rows if row[0] == phase] for phase in ("relax", "smile", "brows")}
    if not all(groups.values()):
        return {}
    amplitudes, onsets, brows, eyes = [], [], [], []
    for side in range(2):
        neutral = mean(row[3][side][0] for row in groups["relax"])
        smile = [(row[1], row[3][side][0] - neutral) for row in groups["smile"]]
        amplitude = max(0.0, max(lift for _, lift in smile))
        amplitudes.append(amplitude)
        onsets.append(next((t for t, lift in smile if lift >= 0.04), 6.0))
        brows.append(mean(row[3][side][1] for row in groups["brows"]) - mean(row[3][side][1] for row in groups["relax"]))
        eyes.append(mean(row[3][side][2] for row in groups["relax"]))
    return {
        "smile_left": amplitudes[0], "smile_right": amplitudes[1],
        "smile_asymmetry": abs(amplitudes[0] - amplitudes[1]) / max(0.05, *amplitudes),
        "smile_onset_difference": abs(onsets[0] - onsets[1]),
        "brow_asymmetry": abs(brows[0] - brows[1]), "eye_asymmetry": abs(eyes[0] - eyes[1]),
    }


def mirror_pairs(neutral):
    import numpy as np

    top, chin = neutral[10, :2], neutral[152, :2]
    along = (chin - top) / max(np.linalg.norm(chin - top), 1e-6)
    normal = np.array([-along[1], along[0]])
    offset = (neutral[:, :2] - top) @ normal
    left_sign = np.sign(offset[263]) or 1.0
    reflected = neutral[:, :2] - 2 * np.outer(offset, normal)
    left = np.where(offset * left_sign > 0.002)[0]
    right = np.where(offset * left_sign < -0.002)[0]
    if not len(left) or not len(right):
        return []
    return [(int(i), int(right[np.argmin(np.linalg.norm(neutral[right, :2] - reflected[i], axis=1))])) for i in left]


def overlay(recording):
    import numpy as np

    frames = [f for f in recording.get("frames", []) if len(f.get("landmarks", [])) >= 468]
    relax = [np.asarray(f["landmarks"], dtype=float)[:478, :3] for f in frames if f.get("phase") == "relax"]
    smiles = [np.asarray(f["landmarks"], dtype=float)[:478, :3] for f in frames if f.get("phase") == "smile"]
    if not relax or not smiles or any(len(s) != len(relax[0]) for s in smiles):
        return {}
    neutral = np.mean(relax, axis=0)
    eye = max(float(np.linalg.norm(neutral[263, :2] - neutral[33, :2])), 1e-3)

    def moved(points):
        return ((points[:, :2] - points[168, :2]) - (neutral[:, :2] - neutral[168, :2])) / eye

    def corner_lift(points):
        shift = moved(points)
        return -shift[291, 1] - shift[61, 1]

    peak = max(smiles, key=corner_lift)
    displacement = np.linalg.norm(moved(peak), axis=1)
    asymmetry = np.zeros(len(neutral))
    for i, j in mirror_pairs(neutral):
        value = abs(displacement[i] - displacement[j]) / max(displacement[i], displacement[j], 0.02)
        asymmetry[i] = max(asymmetry[i], value)
        asymmetry[j] = max(asymmetry[j], value)
    values = features(recording)
    left, right = values.get("smile_left", 0.0), values.get("smile_right", 0.0)
    weaker = "left" if left < right else "right"
    stronger = max(left, right)
    return {
        "neutral": np.round(neutral, 4).tolist(),
        "smile": np.round(peak, 4).tolist(),
        "asymmetry": np.round(np.clip(asymmetry, 0, 1), 3).tolist(),
        "lift_left": round(float(left), 4),
        "lift_right": round(float(right), 4),
        "weaker": weaker,
        "less_percent": round(100 * (1 - min(left, right) / stronger)) if stronger > 0 else 0,
    }
