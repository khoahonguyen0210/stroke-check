# Stroke Check

A 30-second stroke warning check that runs in a laptop browser with a webcam and microphone.

**This is a warning tool, not a diagnosis. If you think someone is having a stroke, call 911 right away.**

## What it does

The user follows three prompts:

1. **Face:** relax, smile big, raise eyebrows (9 s).
2. **Arms:** raise both arms straight out, palms up, hold still (10 s).
3. **Speech:** read "The quick brown fox jumps over the lazy dog near the river bank." out loud.

The app checks the three classic warning signs (face droop, arm drift, slurred speech) and shows either "No warning signs" or "Possible stroke signs: call 911 now."

When signs are found:

- "Call 911 now" appears immediately as fixed text. It never waits for a model.
- An **emergency assistant powered by NVIDIA Nemotron** guides the family. It starts a last-known-normal timer, answers questions only from approved CDC/NHS/MedlinePlus guidance, collects the facts paramedics need word for word, and writes a paramedic handoff.
- After a 20-second "Cancel — false alarm" window, the user's **trusted contacts** get an email with what was detected, when, and where (if location was shared). This is in addition to 911, never instead of it.

## How it works

| Part | How |
| --- | --- |
| Face | MediaPipe Face Landmarker in the browser. Mouth-corner lift, eyebrow and eye-opening asymmetry, normalized by inter-eye distance (`backend/face.py`). |
| Arms | MediaPipe Pose + Hand Landmarker. Wrist drop and palm rotation, left vs right, normalized by shoulder width (`backend/arms.py`). |
| Speech | ffmpeg → 16 kHz wav. wav2vec2-base embeddings → classifier trained on TORGO (dysarthric vs typical speech), split by speaker. Whisper (faster-whisper base) transcript → word error rate. Speaking rate, pause ratio, and recording quality (`backend/speech.py`, `training/train_speech.py`). |
| Decision | One calibrated classifier per sign, trained on our own recordings, leave-one-person-out. Asks for a retake when quality fails or a score is uncertain (`backend/fusion.py`, `training/train_fusion.py`). |
| Emergency agent | Nemotron 3.5 Bolt E4B with OpenAI-style tool calling: `speak`, `ask`, `record_fact`, `start_timer`, `lookup_guidance`, `notify_contacts`, `write_handoff`, `finish_turn` (`backend/agent.py`, `backend/agent_tools.py`). |
| Safety layer | Code, not the model. Every sentence the family hears and every contact update goes through `backend/safety.py`. It blocks reassurance ("you're fine"), "not a stroke", "wait and see", medicine or dose advice, and diagnoses, in English, Spanish and Vietnamese, and swaps in a fixed-script line. On API error or a timeout over 6 s, the agent falls back to the fixed script. |
| Alerts | Gmail SMTP email to trusted contacts who agreed to receive alerts, logged to `data/alerts.log` (`backend/alerts.py`). |

All results (speech model on held-out TORGO speakers, leave-one-person-out detection, agent eval runs) are in `models/results.json`. Every number we show comes from that file.

## Run it

Requirements: Python 3.11, ffmpeg on PATH, Chrome.

```
python -m venv .venv
.venv\Scripts\activate            (Windows)   or   source .venv/bin/activate
pip install mediapipe opencv-python numpy scipy scikit-learn xgboost fastapi uvicorn python-multipart python-dotenv google-genai openai transformers faster-whisper jiwer datasets soundfile
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
copy .env.example .env             then fill in NVIDIA_API_KEY, SMTP_USER, SMTP_APP_PASSWORD (GEMINI_API_KEY only for the eval)
uvicorn backend.main:app --port 8000
```

Open http://localhost:8000 in Chrome and allow the camera and microphone.

Training and evaluation:

```
python -m training.train_speech embed 0 1     wav2vec2 embeddings for TORGO, cached in data/torgo_embeddings/
python -m training.train_speech train         speech classifier, writes models/speech.joblib and results.json "speech"
python -m training.train_fusion               per-sign classifiers from data/sessions/
python -m training.evaluate                   leave-one-person-out + robustness, results.json "detection"
python -m training.eval_agent --label v2      30 agent scenarios on Nemotron Bolt (reasoning off/on), Nemotron Super and Gemini, results.json "agent"
```

## Credits

Everything in this repository was written during tidalBYTE '26. Outside sources:

**Models and APIs**
- NVIDIA Nemotron 3.5 Bolt E4B (`nvidia/nemotron-3.5-bolt-e4b`) and Nemotron 3.5 Super VL 120B-A12B (`nvidia/nemotron-3.5-super-vl-120b-a12b`), via build.nvidia.com.
- Google Gemini, via the Gemini API's OpenAI-compatible endpoint. Used only as a comparison baseline in the agent eval, never in the product.
- `facebook/wav2vec2-base` (Meta AI), via Hugging Face Transformers.
- Whisper `base` (OpenAI), via faster-whisper / CTranslate2 (`Systran/faster-whisper-base`).
- MediaPipe Tasks Vision: Face Landmarker, Pose Landmarker (lite), Hand Landmarker (Google).

**Dataset**
- TORGO database of dysarthric speech (Rudzicz, Namasivayam and Wolff, University of Toronto), via the Hugging Face copy `abnerh/TORGO-database`.

**Guidance sources** (for `backend/guidance.md`)
- CDC, "Signs and Symptoms of Stroke": https://www.cdc.gov/stroke/signs-symptoms/index.html
- CDC, "Treatment and Intervention for Stroke": https://www.cdc.gov/stroke/treatment/index.html
- CDC, "About Stroke": https://www.cdc.gov/stroke/about/index.html
- MedlinePlus, "Stroke": https://medlineplus.gov/ency/article/000726.htm
- NHS, "Symptoms of a stroke": https://www.nhs.uk/conditions/stroke/symptoms/
- The "food, drink and medicine" and "while you wait" sections are our team's safety rules, marked as such in the file.

**Maps (family live view)**
- MapLibre GL JS (loaded from unpkg) for the map.
- OpenFreeMap map styles and tiles (`liberty` and `dark`): https://openfreemap.org
- Map data © OpenStreetMap contributors (ODbL).
- Overpass API (and its public mirrors) to find hospitals near the person.
- OSRM demo server (router.project-osrm.org): used once to generate the fixed test route in `scripts/simulate_route.py`.

**Libraries**
FastAPI, Starlette, Uvicorn, Pydantic, python-multipart, python-dotenv, NumPy, SciPy, scikit-learn, XGBoost, PyTorch, torchaudio, Hugging Face Transformers and Datasets, PyArrow, pandas, faster-whisper, jiwer, soundfile, joblib, the `openai` Python package, google-genai, ffmpeg. MediaPipe Tasks Vision is loaded from jsDelivr. The Pixelify Sans font comes from Google Fonts.

**AI tools**
- Claude Code (Anthropic) was used as a coding assistant.
