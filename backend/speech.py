import subprocess
from pathlib import Path


def to_wav(webm_path):
    wav_path = Path(webm_path).with_suffix(".wav")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(webm_path), "-ac", "1", "-ar", "16000", str(wav_path)],
        check=True,
    )
    return str(wav_path)


def features(wav_path, sentence):
    return {"p_slurred": 0.1, "wer": 0.0, "rate": 2.5, "pause_ratio": 0.2, "transcript": sentence}


def quality(wav_path):
    return {"ok": True, "reason": None, "duration_s": 4.0, "loudness_db": -20.0, "clipping": 0.0}
