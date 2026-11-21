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


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step == "embed":
        embed(int(sys.argv[2]) if len(sys.argv) > 2 else 0, int(sys.argv[3]) if len(sys.argv) > 3 else 1)
