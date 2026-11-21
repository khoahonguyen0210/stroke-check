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
