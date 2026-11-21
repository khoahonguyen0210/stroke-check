import csv
import io
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

EMB_DIR = Path("data/torgo_embeddings")
CHUNK = 500
MAX_SECONDS = 10
SR = 16000


def speaker_of(path):
    return Path(path).name.split("_")[0]


def load_torgo():
    import datasets

    d = datasets.load_dataset("abnerh/TORGO-database", split="train")
    return d.cast_column("audio", datasets.Audio(decode=False))


def read_audio(audio):
    if audio.get("bytes"):
        wav, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=True)
    else:
        wav, sr = sf.read(audio["path"], dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.mean(axis=1))
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav[: SR * MAX_SECONDS].numpy()


def write_meta(d):
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    rows = d.remove_columns("audio").to_list()
    paths = [a["path"] for a in d["audio"]]
    with open(EMB_DIR / "meta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "speaker", "speech_status", "severity", "gender", "duration", "path", "transcription"])
        for i, (r, p) in enumerate(zip(rows, paths)):
            w.writerow([i, speaker_of(p), r["speech_status"], r["severity"], r["gender"], r["duration"], p, r["transcription"]])


def embed():
    from transformers import AutoFeatureExtractor, AutoModel

    d = load_torgo()
    write_meta(d)
    fe = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    model = AutoModel.from_pretrained("facebook/wav2vec2-base").eval()
    n = len(d)
    start = time.time()
    done = 0
    for c0 in range(0, n, CHUNK):
        out = EMB_DIR / f"chunk_{c0:05d}.npz"
        if out.exists():
            continue
        idx = list(range(c0, min(c0 + CHUNK, n)))
        embs = []
        for i in idx:
            wav = read_audio(d[i]["audio"])
            if len(wav) < SR // 10:
                wav = np.pad(wav, (0, SR // 10 - len(wav)))
            x = fe(wav, sampling_rate=SR, return_tensors="pt").input_values
            with torch.inference_mode():
                hs = model(x, output_hidden_states=True).hidden_states
            embs.append(torch.stack([h[0].mean(dim=0) for h in hs]).numpy().astype(np.float16))
        np.savez(out, idx=np.array(idx), emb=np.stack(embs))
        done += len(idx)
        rate = done / (time.time() - start)
        left = sum(1 for c in range(c0 + CHUNK, n, CHUNK) if not (EMB_DIR / f"chunk_{c:05d}.npz").exists()) * CHUNK
        print(f"{time.strftime('%H:%M:%S')} saved {out.name} rows {c0}-{idx[-1]} | {rate:.1f} clips/s | ~{left / rate / 60:.0f} min left", flush=True)
    print("EMBED DONE", flush=True)


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("embed", "all"):
        embed()
