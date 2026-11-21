import csv
import io
import random
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import soundfile as sf
import torch
import torchaudio

EMB_DIR = Path("data/torgo_embeddings")
CHUNK = 250
MAX_SECONDS = 8
PER_SPEAKER = 400
SR = 16000


def parse_name(path):
    parts = Path(path).stem.split("_")
    return {"speaker": parts[0], "session": parts[1], "mic": parts[2], "utt": parts[3]}


def load_torgo():
    import datasets

    d = datasets.load_dataset("abnerh/TORGO-database", split="train")
    return d.cast_column("audio", datasets.Audio(decode=False))


def audio_paths(d):
    return pc.struct_field(d.data.table.column("audio"), "path").to_pylist()


def select_rows(paths):
    best = {}
    for i, p in enumerate(paths):
        n = parse_name(p)
        key = (n["speaker"], n["session"], n["utt"])
        if key not in best or (n["mic"] == "arrayMic" and parse_name(paths[best[key]])["mic"] != "arrayMic"):
            best[key] = i
    by_speaker = {}
    for key, i in sorted(best.items()):
        by_speaker.setdefault(key[0], []).append(i)
    rng = random.Random(0)
    chosen = []
    for spk in sorted(by_speaker):
        rows = by_speaker[spk]
        rng.shuffle(rows)
        chosen += rows[:PER_SPEAKER]
    rng.shuffle(chosen)
    return chosen


def write_meta(d, paths):
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    cols = d.select_columns(["speech_status", "severity", "gender", "duration", "transcription"]).to_dict()
    with open(EMB_DIR / "meta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "speaker", "session", "mic", "speech_status", "severity", "gender", "duration", "path", "transcription"])
        for i, p in enumerate(paths):
            n = parse_name(p)
            w.writerow([i, n["speaker"], n["session"], n["mic"], cols["speech_status"][i], cols["severity"][i],
                        cols["gender"][i], cols["duration"][i], p, cols["transcription"][i]])


def read_audio(audio):
    if audio.get("bytes"):
        wav, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=True)
    else:
        wav, sr = sf.read(audio["path"], dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.mean(axis=1))
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    wav = wav[: SR * MAX_SECONDS].numpy()
    if len(wav) < SR // 10:
        wav = np.pad(wav, (0, SR // 10 - len(wav)))
    return wav


def embed(worker=0, workers=1):
    from transformers import AutoFeatureExtractor, AutoModel

    torch.set_num_threads(max(1, torch.get_num_threads() // workers))
    d = load_torgo()
    paths = audio_paths(d)
    selected = select_rows(paths)
    if worker == 0:
        write_meta(d, paths)
        np.save(EMB_DIR / "selected.npy", np.array(selected))
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    fe = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    model = AutoModel.from_pretrained("facebook/wav2vec2-base").eval()
    chunks = [selected[c:c + CHUNK] for c in range(0, len(selected), CHUNK)]
    mine = [k for k in range(len(chunks)) if k % workers == worker]
    print(f"worker {worker}/{workers}: {len(selected)} selected clips, {len(chunks)} chunks, {len(mine)} mine, threads {torch.get_num_threads()}", flush=True)
    start = time.time()
    done = 0
    for n_done, k in enumerate(mine):
        out = EMB_DIR / f"chunk_{k:03d}.npz"
        if out.exists():
            continue
        embs = []
        for i in chunks[k]:
            x = fe(read_audio(d[i]["audio"]), sampling_rate=SR, return_tensors="pt").input_values
            with torch.inference_mode():
                hs = model(x, output_hidden_states=True).hidden_states
            embs.append(torch.stack([h[0].mean(dim=0) for h in hs]).numpy().astype(np.float16))
        np.savez(out, idx=np.array(chunks[k]), emb=np.stack(embs))
        done += len(chunks[k])
        rate = done / (time.time() - start)
        left = sum(len(chunks[j]) for j in mine[n_done + 1:])
        print(f"{time.strftime('%H:%M:%S')} worker {worker} saved {out.name} | {rate:.2f} clips/s | ~{left / rate / 60:.0f} min left", flush=True)
    print(f"EMBED DONE worker {worker}", flush=True)


TEST_SPEAKERS = ["F03", "M05", "FC03", "MC04"]
LAYERS = [3, 6, 8, 10, 12]
MODEL_PATH = Path("models/speech.joblib")
RESULTS = Path("models/results.json")


def load_embeddings():
    import pandas as pd

    meta = pd.read_csv(EMB_DIR / "meta.csv")
    idx, emb = [], []
    for f in sorted(EMB_DIR.glob("chunk_*.npz")):
        z = np.load(f)
        idx.append(z["idx"])
        emb.append(z["emb"])
    idx = np.concatenate(idx)
    emb = np.concatenate(emb).astype(np.float32)
    rows = meta.set_index("idx").loc[idx].reset_index()
    y = (rows["speech_status"] == "dysarthria").astype(int).to_numpy()
    return rows, emb, y


def make_model(kind):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5, n_jobs=8, eval_metric="logloss")


def scores(y, p):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    pred = (p >= 0.5).astype(int)
    out = {"n": int(len(y)), "accuracy": round(float(accuracy_score(y, pred)), 3), "f1": round(float(f1_score(y, pred)), 3)}
    if len(set(y)) > 1:
        out["auc"] = round(float(roc_auc_score(y, p)), 3)
    return out


def loso(X, y, groups, kind):
    p = np.zeros(len(y))
    for g in sorted(set(groups)):
        test = groups == g
        m = make_model(kind)
        m.fit(X[~test], y[~test])
        p[test] = m.predict_proba(X[test])[:, 1]
    return p


def train():
    import json

    import joblib

    rows, emb, y = load_embeddings()
    groups = rows["speaker"].to_numpy()
    print(f"{len(y)} clips from {len(set(groups))} speakers, {y.sum()} dysarthric", flush=True)
    test = np.isin(groups, TEST_SPEAKERS)
    print(f"held-out test speakers {TEST_SPEAKERS}: {test.sum()} clips; train speakers: {sorted(set(groups[~test]))}", flush=True)
    selection = {}
    for layer in LAYERS:
        p = loso(emb[~test, layer], y[~test], groups[~test], "logreg")
        selection[f"logreg_layer{layer}"] = scores(y[~test], p)
        print(f"  select logreg layer {layer:2d}: {selection[f'logreg_layer{layer}']}", flush=True)
    best_layer = max(LAYERS, key=lambda l: selection[f"logreg_layer{l}"]["f1"])
    p = loso(emb[~test, best_layer], y[~test], groups[~test], "xgboost")
    selection[f"xgboost_layer{best_layer}"] = scores(y[~test], p)
    print(f"  select xgboost layer {best_layer:2d}: {selection[f'xgboost_layer{best_layer}']}", flush=True)
    best_kind = max(["logreg", "xgboost"], key=lambda k: selection[f"{k}_layer{best_layer}"]["f1"])
    print(f"chosen: {best_kind}, layer {best_layer}", flush=True)

    held = {}
    for kind in ["logreg", "xgboost"]:
        m = make_model(kind)
        m.fit(emb[~test, best_layer], y[~test])
        p = m.predict_proba(emb[test, best_layer])[:, 1]
        held[kind] = scores(y[test], p)
        held[kind]["per_speaker_accuracy"] = {s: round(float(((p[groups[test] == s] >= 0.5) == y[test][groups[test] == s]).mean()), 3) for s in TEST_SPEAKERS}
        print(f"held-out {kind}: {held[kind]}", flush=True)
    loso_all = scores(y, loso(emb[:, best_layer], y, groups, best_kind))
    print(f"leave-one-speaker-out over all 15 speakers ({best_kind}): {loso_all}", flush=True)

    final = make_model(best_kind)
    final.fit(emb[:, best_layer], y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "kind": best_kind, "layer": best_layer, "max_seconds": MAX_SECONDS}, MODEL_PATH)
    data = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    data["speech"] = {
        "dataset": "TORGO (abnerh/TORGO-database), one mic per utterance, up to 400 clips per speaker",
        "clips": int(len(y)), "speakers": int(len(set(groups))), "dysarthric_clips": int(y.sum()),
        "features": f"wav2vec2-base layer {best_layer} mean-pooled",
        "model": best_kind,
        "test_speakers": TEST_SPEAKERS,
        "held_out": held[best_kind],
        "held_out_both_models": held,
        "loso_all_speakers": loso_all,
        "selection_cv_on_train_speakers": selection,
        "note": "Dysarthria in TORGO is a proxy for stroke slurring; TORGO prompts are mostly single words and short phrases.",
    }
    RESULTS.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {MODEL_PATH} and {RESULTS}", flush=True)


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "train"
    if step == "embed":
        embed(int(sys.argv[2]) if len(sys.argv) > 2 else 0, int(sys.argv[3]) if len(sys.argv) > 3 else 1)
    if step == "train":
        train()
