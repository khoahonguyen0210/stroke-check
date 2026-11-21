import argparse
import json
from tempfile import NamedTemporaryFile
from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backend import fusion, recording, storage


def classifier():
    return make_pipeline(
        DictVectorizer(sparse=False),
        StandardScaler(),
        CalibratedClassifierCV(LogisticRegression(class_weight="balanced"), cv=2),
    )


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate fusion on labeled takes.")
    parser.add_argument("--exclude-person", action="append", default=[], help="Skip this person's recordings. Repeat to skip multiple people.")
    parser.add_argument("--output", type=Path, default=fusion.MODEL, help="Save the trained model to this path.")
    args = parser.parse_args()

    excluded = {storage.user_key(person) for person in args.exclude_person}
    sessions = [
        session for session in storage.load_sessions()
        if session["person"] not in excluded and (session["path"] / "speech.webm").is_file()
    ]
    if len({session["person"] for session in sessions}) < 2:
        raise SystemExit("The legacy three-sign trainer needs recordings with audio from at least two people.")

    rows, labels, people = [], [], []
    for session in sessions:
        path = session["path"]
        face = json.loads((path / "face.json").read_text())
        arms = json.loads((path / "arms.json").read_text())
        features, quality, _ = recording.analyze(face, arms, webm_path=path / "speech.webm", sentence=session["sentence"])
        if all(item["ok"] for item in quality.values()):
            rows.append(fusion.model_input(features, quality))
            labels.append(session["labels"])
            people.append(session["person"])

    if len(set(people)) < 2:
        raise SystemExit("Training requires usable recordings from at least two people.")
    models = {}
    splits = list(LeaveOneGroupOut().split(rows, groups=people))
    for sign, label in fusion.NAMES.items():
        targets = [int(label in take) for take in labels]
        actual, predicted = [], []
        for train, test in splits:
            train_targets = [targets[i] for i in train]
            if min(train_targets.count(0), train_targets.count(1)) < 2:
                raise SystemExit(f"{label} needs at least two positive and two negative takes in each training fold.")
            model = classifier().fit([rows[i] for i in train], train_targets)
            actual.extend(targets[i] for i in test)
            predicted.extend(int(p >= 0.6) for p in model.predict_proba([rows[i] for i in test])[:, 1])
        print(f"{label} accuracy={accuracy_score(actual, predicted):.3f} F1={f1_score(actual, predicted, zero_division=0):.3f} n={len(actual)}")
        models[sign] = classifier().fit(rows, targets)

    with NamedTemporaryFile(dir=args.output.parent, delete=False) as file:
        path = Path(file.name)
    try:
        joblib.dump(models, path)
        path.replace(args.output)
    finally:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
