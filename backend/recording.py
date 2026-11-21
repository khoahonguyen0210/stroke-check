from pathlib import Path
from subprocess import CalledProcessError
from tempfile import TemporaryDirectory
from time import perf_counter

from backend import arms, face, storage


def analyze(face_json, arms_json, audio=None, webm_path=None, sentence=storage.SENTENCE):
    features, quality, latency = {}, {}, {}
    for name, module, recording in (("face", face, face_json), ("arms", arms, arms_json)):
        start = perf_counter()
        features[name] = module.features(recording)
        quality[name] = module.quality(recording)
        latency[name] = round((perf_counter() - start) * 1000, 2)

    start = perf_counter()
    try:
        from backend import speech

        with TemporaryDirectory() as directory:
            temporary = Path(directory) / "speech.webm"
            temporary.write_bytes(audio if audio is not None else Path(webm_path).read_bytes())
            wav = speech.to_wav(temporary)
            quality["speech"] = speech.quality(wav)
            features["speech"] = speech.features(wav, sentence) if quality["speech"]["ok"] else {}
    except (ImportError, OSError, CalledProcessError, ValueError, RuntimeError):
        quality["speech"] = {"ok": False, "reason": "Speech analysis is unavailable or the recording could not be read."}
        features["speech"] = {}
    latency["speech"] = round((perf_counter() - start) * 1000, 2)
    return features, quality, latency
