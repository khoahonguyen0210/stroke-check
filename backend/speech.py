import re
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "speech.joblib"
SR = 16000
FRAME = 400
HOP = 160


def to_wav(webm_path):
    wav_path = Path(webm_path).with_suffix(".wav")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(webm_path), "-ac", "1", "-ar", str(SR), str(wav_path)],
        check=True,
    )
    return str(wav_path)


def load(wav_path):
    wav, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SR:
        import torch
        import torchaudio

        wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, SR).numpy()
    return wav


def frame_db(wav):
    if len(wav) < FRAME:
        wav = np.pad(wav, (0, FRAME - len(wav)))
    n = 1 + (len(wav) - FRAME) // HOP
    frames = np.stack([wav[i * HOP:i * HOP + FRAME] for i in range(n)])
    return 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)


def voiced(wav):
    db = frame_db(wav)
    threshold = max(-45.0, np.percentile(db, 95) - 25)
    return db > threshold


def words(text):
    return re.sub(r"[^a-z0-9' ]", " ", text.lower()).split()


def quality(wav_path):
    wav = load(wav_path)
    duration = len(wav) / SR
    rms_db = float(20 * np.log10(np.sqrt((wav ** 2).mean()) + 1e-9)) if len(wav) else -120.0
    clipping = float((np.abs(wav) >= 0.99).mean()) if len(wav) else 0.0
    voiced_s = float(voiced(wav).sum() * HOP / SR) if len(wav) else 0.0
    reason = None
    if duration < 1.5:
        reason = "The recording was too short. Please read the whole sentence."
    elif voiced_s < 0.8 or rms_db < -50:
        reason = "We could not hear you. Move closer to the microphone and speak up."
    elif clipping > 0.01:
        reason = "The recording was too loud. Move a little away from the microphone."
    return {"ok": reason is None, "reason": reason, "duration_s": round(duration, 2), "loudness_db": round(rms_db, 1),
            "clipping": round(clipping, 4), "voiced_s": round(voiced_s, 2)}


@lru_cache(maxsize=1)
def wav2vec():
    from transformers import AutoFeatureExtractor, AutoModel

    return AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base"), AutoModel.from_pretrained("facebook/wav2vec2-base").eval()


@lru_cache(maxsize=1)
def classifier():
    import joblib

    return joblib.load(MODEL) if MODEL.exists() else None


@lru_cache(maxsize=1)
def whisper():
    from faster_whisper import WhisperModel

    return WhisperModel("base", device="cpu", compute_type="int8")


def warm():
    wav2vec()
    classifier()
    whisper()
    return True


def embedding(wav, layer, max_seconds):
    import torch

    fe, model = wav2vec()
    wav = wav[: SR * max_seconds]
    x = fe(wav, sampling_rate=SR, return_tensors="pt").input_values
    with torch.inference_mode():
        hs = model(x, output_hidden_states=True).hidden_states
    return hs[layer][0].mean(dim=0).numpy()


def p_slurred(wav):
    bundle = classifier()
    if bundle is None:
        return float("nan")
    emb = embedding(wav, bundle["layer"], bundle["max_seconds"])
    return float(bundle["model"].predict_proba(emb.reshape(1, -1))[0, 1])


def transcribe(wav_path):
    segments, _ = whisper().transcribe(str(wav_path), language="en", beam_size=1, vad_filter=False)
    segments = list(segments)
    text = " ".join(s.text.strip() for s in segments).strip()
    logprob = float(np.mean([s.avg_logprob for s in segments])) if segments else -5.0
    return text, logprob


def features(wav_path, sentence):
    import jiwer

    wav = load(wav_path)
    v = voiced(wav)
    idx = np.flatnonzero(v)
    if len(idx):
        span = v[idx[0]:idx[-1] + 1]
        span_s = len(span) * HOP / SR
        pause_ratio = float(1 - span.mean())
    else:
        span_s, pause_ratio = 0.0, 1.0
    text, logprob = transcribe(wav_path)
    target = " ".join(words(sentence))
    heard = " ".join(words(text))
    wer = float(jiwer.wer(target, heard)) if heard else 1.0
    return {
        "p_slurred": p_slurred(wav),
        "wer": round(wer, 3),
        "rate": round(len(target.split()) / span_s, 3) if span_s > 0 else 0.0,
        "pause_ratio": round(pause_ratio, 3),
        "speech_span_s": round(span_s, 2),
        "asr_logprob": round(logprob, 3),
        "transcript": text,
    }
