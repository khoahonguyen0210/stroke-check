# Stroke Check — instructions for the coding agent

## What we are building
A 30-second stroke warning check that runs in a laptop browser with a webcam and microphone.
The user follows 3 prompts:
1. Face: neutral face, then smile, then raise eyebrows.
2. Arms: raise both arms straight out, palms up, hold for 10 seconds.
3. Speech: read one sentence out loud.

AI checks the 3 classic stroke warning signs (face droop, arm drift, slurred speech) and returns either
"No warning signs" or "Possible stroke signs: call 911 now". When signs are found, an emergency assistant
guides the family: start a "last known normal" timer, tell them to call 911, and write a short summary for paramedics.
It also alerts the user's trusted contacts (family, caregiver) with what was detected, when, and where, so someone is on the way even if the user is alone.

This is a warning tool, not a diagnosis. All user-facing text must say so where relevant.

## Event context
- tidalBYTE '26 hackathon, Sat Nov 21, hacking 10:00–20:00, judging 20:00–21:30.
- Team of 2. Judges test it sitting at our table in about 30 seconds.
- Person 1's laptop is the demo laptop: all takes are recorded there, and the final app runs there.
- All project code is written during the event. Credit every outside source (libraries, examples, datasets) in README.md.
- We also enter the **Best Use of NVIDIA Nemotron** track: "build an agent doing real work, then prove Nemotron was the right choice." Judged on: Nemotron central (not a thin chatbot), reliable execution (data flow, tool use, latency, error handling), output quality improved through grounding, evaluation and testing, real impact, thoughtful use. Our Nemotron agent is the emergency agent below; the proof is `training/eval_agent.py`.

## Rules for you (the agent)
- Keep code simple and straightforward. No clever abstractions, no frameworks we don't need.
- Do not write comments in code.
- Small files, one job each. Prefer plain functions.
- After each task: run it, show the output, fix errors before moving on.
- Never invent metrics. Every number in the UI or pitch must come from a script we ran.
- Never output "you're fine" or "not a stroke". Any detected sign → "call 911".
- Alerting trusted contacts is in addition to 911, never instead of it. Every alert message says so.
- Secrets go in `.env` (NVIDIA_API_KEY, GEMINI_API_KEY, ELEVENLABS_API_KEY, SMTP_USER, SMTP_APP_PASSWORD, TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM). Never commit `.env`.
- Do not add features beyond this file without asking.

## Stack
- Frontend: plain HTML + JavaScript (no React), MediaPipe Tasks Vision for the web (face, pose, hand landmarkers) loaded from CDN, MediaRecorder for audio.
- Backend: Python 3.11, FastAPI, uvicorn.
- ML: numpy, scipy, scikit-learn, xgboost, transformers (wav2vec2), torch/torchaudio, faster-whisper, jiwer, datasets.
- Audio conversion: ffmpeg (webm → 16 kHz mono wav).
- Agent LLM: NVIDIA Nemotron through build.nvidia.com, called with the `openai` package (`base_url="https://integrate.api.nvidia.com/v1"`, key `NVIDIA_API_KEY`). Main model: Nemotron 3.5 Bolt E4B; comparison model: Nemotron 3 Super 120B-A12B. Copy the exact model ids from each model card on build.nvidia.com; do not guess them.
- Gemini (`google-genai`) is used only as a comparison baseline, never in the live product.
- Text-to-speech: browser speechSynthesis (ElevenLabs optional). Family answers: typed, or Chrome's Web Speech API if time allows.
- Alerts: email through Gmail SMTP with an app password (`smtplib`), always on. SMS through Twilio is optional; a Twilio trial account can only text phone numbers verified in the Twilio console, so verify teammates' numbers for the demo.
- Location: browser Geolocation API, only with the user's permission.
- Storage: files on disk under `data/`.

## Folder layout
```
stroke-check/
  CLAUDE.md
  README.md
  .env
  frontend/
    index.html
    app.js
    style.css
  backend/
    main.py
    face.py
    arms.py
    speech.py
    fusion.py
    baseline.py
    agent.py
    agent_tools.py
    safety.py
    guidance.md
    alerts.py
    storage.py
  training/
    train_speech.py
    train_fusion.py
    evaluate.py
    gemini_baseline.py
    eval_agent.py
    agent_scenarios.json
  data/
    sessions/<person>/<take_id>/{meta.json, face.json, arms.json, speech.webm}
    baselines/<user>.json
    contacts/<user>.json
  models/
```

## Team files
We are a team of 2: read `TEAM_2.md`. If you were not told whether you are working for Person 1 or Person 2, ask before writing code.
Only edit files your person owns. If you need a change in someone else's file, say so instead of editing it.

## Shared data formats and interfaces
Everyone codes against these. Do not change them without telling the whole team.

`data/sessions/<person>/<take_id>/meta.json`
```json
{"person": "khoa", "take_id": "20261121-103512", "labels": ["normal"], "glasses": false, "lighting": "good", "sentence": "The quick brown fox jumps over the lazy dog near the river bank.", "created_at": "2026-11-21T10:35:12"}
```
`labels` is `["normal"]` or any of `face_droop`, `arm_drift`, `slurred`.

`face.json`
```json
{"fps": 30, "frames": [{"t": 0, "phase": "relax", "landmarks": [[0.51, 0.42, -0.03], "... 478 points"]}]}
```
`phase` is `relax`, `smile` or `brows`. Coordinates are MediaPipe normalized x, y, z.

`arms.json`
```json
{"fps": 30, "frames": [{"t": 0, "pose": [[0.5, 0.3, -0.2, 0.99], "... 33 points"], "hands": [{"side": "Left", "landmarks": [[0.4, 0.5, 0.0], "... 21 points"]}]}]}
```

`speech.webm`: raw browser recording. `speech.py` converts it to `speech.wav` (16 kHz mono) next to it.

Python interfaces:
- `face.features(face_json) -> dict[str, float]` and `face.quality(face_json) -> dict` (`ok: bool`, `reason: str | None`, plus numbers)
- `arms.features(arms_json) -> dict[str, float]` and `arms.quality(arms_json) -> dict`
- `speech.features(wav_path, sentence) -> dict[str, float]` (includes `p_slurred`, `wer`, `rate`, `pause_ratio`) and `speech.quality(wav_path) -> dict`
- `fusion.predict(features: dict, quality: dict) -> dict` returning the `/api/check` response shape
- `baseline.adjust(user, features, result) -> dict` returning the result with adjusted face/arm probabilities and `baseline` details
- `agent.start(result, language) -> dict` and `agent.reply(text, state, language) -> dict`, both returning `{"say": str, "actions": [...], "handoff": {...}, "state": {...}}`
- `safety.check(text) -> dict` (`ok: bool`, `reason: str | None`)
- `alerts.send(user, result, detected_at, location) -> list[dict]` and `alerts.notify(user, text) -> list[dict]` (follow-up update used by the agent's `notify_contacts` tool; adds the standard disclaimer)
- `storage.save_session(...)`, `storage.load_sessions() -> list[dict]`, `storage.contacts(user)`, `storage.save_contacts(user, contacts)`

Until a real module exists, its owner commits a stub with the same signature that returns fixed values, so others can integrate early.

## Git workflow
- Pull before you start each task, commit and push after each working step (every 30–45 minutes).
- Small commits with clear messages. Never force-push.
- Integration points: 10:30 (data collection works), 13:30 (full flow works), 18:00 (feature freeze).
- Trained model files (`models/*.joblib`, `models/*.json`) are small and committed so both laptops can run the app. Recordings (`data/`) and the TORGO embedding cache (`data/torgo_embeddings/`) are never committed; to share a few takes, zip them and send them directly.

## Frontend flow
1. Start screen: name field, "Start check" button, "Trusted contacts" button, "Data collection mode" toggle (shows a label dropdown).
   - Trusted contacts screen: add up to 3 contacts (name, relationship, email, optional phone), a checkbox confirming each contact agreed to receive alerts, and a "Send test alert" button.
2. Step 1 Face (9 s): on-screen countdown prompts "Relax your face" (3 s) → "Smile big" (3 s) → "Raise your eyebrows" (3 s). Record face landmarks every frame with timestamps and the current prompt.
3. Step 2 Arms (10 s): "Raise both arms straight out, palms up, hold still". Record pose + hand landmarks every frame.
4. Step 3 Speech: show the sentence "The quick brown fox jumps over the lazy dog near the river bank." Record audio until the user clicks Done (max 8 s).
5. Send everything to the backend. Show a loading state.
6. Result screen: one card per sign (face, arms, speech) with green/red and probability, plus overall result in large text. If the backend asks for a retake, show the reason and a Retake button.
7. "Under the hood" toggle: show raw feature values, per-model probabilities, latency per step.
8. If any sign is found: show "Possible stroke signs. Call 911 now." immediately (fixed text, never waits for the model), then open the emergency assistant panel: timer, spoken instructions, a box for the family to answer, the paramedic handoff card filling in live, and an "Agent actions" list showing each tool the agent called.
9. At the same time, show "Alerting your trusted contacts in 20 seconds" with a large "Cancel — false alarm" button. If nobody cancels, send the alerts automatically (the user may not be able to act). Then show who was alerted and how (email/SMS), with a timestamp.

Data collection mode: same flow, plus a label dropdown (`normal`, `face_droop`, `arm_drift`, `slurred`, or combinations). Posts to `/api/collect` instead of `/api/check`.

Style: clean and readable first; light retro touches (pixel font for headings, high-contrast colors) to match the event's Y2K theme.

## API
- `GET /api/health` → `{ok: true}`
- `POST /api/check` multipart: `meta` (JSON: name, user_agent), `face` (JSON), `arms` (JSON), `audio` (webm) →
  ```json
  {"overall": "warning|clear|retake", "retake_reason": null,
   "signs": {"face": {"flag": true, "p": 0.81}, "arms": {"flag": false, "p": 0.12}, "speech": {"flag": true, "p": 0.74}},
   "features": {...}, "latency_ms": {...}}
  ```
- `/api/check` also returns `explain: {face_overlay, arms_trace}` (`face.overlay`: neutral and peak-smile landmarks, per-landmark asymmetry, per-side lifts; `arms.trace`: wrist drop over the hold) for the result-screen visuals.
- `POST /api/explain` `{result}` → `{text, grounded, model}` from `explain.write` (Nemotron Bolt, reasoning off, 4 s limit, numbers checked against the features, safety-checked; template otherwise). Called after the result is shown so it never delays the 911 instructions.
- `POST /api/collect` same fields + `label` → saves a session, returns its path.
- `POST /api/baseline/{user}` same fields → adds a normal session to that user's baseline.
- `POST /api/agent/start` `{result, language}` → agent response (below). `result` may include `name` and `location` (`{lat, lon}`, only if shared).
- `POST /api/agent/message` `{text, state, language}` → agent response. `text` is the family's words, `"(no answer)"` after 20 s of silence, or `"paramedics are here"` from the button.
- Agent response:
  ```json
  {"say": "...", "actions": [], "handoff": {}, "state": {},
   "model": "nvidia/nemotron-3.5-bolt-e4b",
   "steps": [{"tool": "ask", "args": {}, "result": "...", "ms": 420, "reasoning": false, "by": "model", "model": "...", "blocked_by_safety": false, "replaced_with": null}],
   "verified": true, "removed_facts": [{"field": "allergies", "value": "...", "reason": "no supporting line"}], "script_911": "..."}
  ```
  `model` is the model that answered this turn (Bolt, or Super if Bolt was rate limited). `steps` has one entry per tool call; `by` is `model` or `code` (rules the code enforces). `verified` is `null` until the handoff is written, then `true` (checked by Super) or `false` (Super failed or took over 8 s). `script_911` is `null` until `last_known_normal` is known.
- `GET /api/agent/handoff/{session}` → printable paramedic handoff text
- `GET /api/contacts/{user}` → list of contacts
- `POST /api/contacts/{user}` `{name, relationship, email, phone?, consent: true}` → saved list (reject if consent is false)
- `POST /api/alert` `{user, result, detected_at, location?}` → `{sent: [{name, channel, ok, error?}]}`
- `POST /api/alert/test/{user}` → sends a clearly labeled test message to every contact
- `POST /api/incident` `{user, name, phone?, result}` → `{id}`; created when a warning fires (long random id, expires after 24 h, `data/incidents/<id>.json`). `/api/alert` takes `incident` and adds `PUBLIC_BASE_URL/family/<id>` to the email.
- `POST /api/incident/{id}/location` `{lat, lng, accuracy}` → saves the latest location (the check page sends it every 10 s when location sharing is on).
- `GET /api/incident/{id}` → `{name, phone, status_timeline, location, destination_hospital, handoff, hospitals, expires_at}`; 404 when expired. Hospitals come from Overpass (15 km) once per incident, cached by area in `data/hospital_cache.json`.
- `GET /family/{id}` → family live view (`frontend/family.html`): MapLibre + OpenFreeMap map, status timeline from the agent, handoff. The agent's `record_fact(field="destination_hospital")` sets the destination, matched to the closest hospital name.
- Dev tool: `python scripts/simulate_route.py <incident_id>` posts 30 points from the MSC, one every 2 s.

## Features and models

### Face (`face.py`)
- Normalize all distances by inter-eye distance.
- Neutral reference = average of the "Relax" phase.
- Per frame: vertical lift of each mouth corner (landmarks 61 and 291) vs neutral; eyebrow height left/right; eye opening left/right. Verify landmark indices against the MediaPipe face mesh map before using them.
- Features: smile amplitude left vs right, asymmetry ratio, difference in smile onset time, eyebrow raise asymmetry, eye opening asymmetry.
- Quality: fraction of frames with a detected face, head tilt range.

### Arms (`arms.py`)
- Pose landmarks: shoulders 11/12, elbows 13/14, wrists 15/16. Normalize by shoulder width.
- Features per side: wrist vertical drop over the 10 s (slope and total), elbow bend, palm rotation from hand landmarks (palm normal from wrist 0, index MCP 5, pinky MCP 17).
- Main features: left minus right drop, left minus right palm rotation.
- Quality: both wrists visible in most frames.

### Speech (`speech.py`, `training/train_speech.py`)
- Convert webm → 16 kHz mono wav with ffmpeg.
- Model A: `facebook/wav2vec2-base` mean-pooled embeddings → logistic regression (or XGBoost) trained on TORGO (`abnerh/TORGO-database`, labels from `speech_status`). Split by speaker so test speakers are never in training. Save to `models/speech.joblib`.
- Feature B: faster-whisper (`base`) transcript → word error rate vs the target sentence (jiwer).
- Feature C: speaking rate and pause ratio from the waveform.
- Quality: audio length, loudness, clipping.

### Fusion (`fusion.py`, `training/train_fusion.py`)
- Input: face, arm and speech features + quality values.
- One classifier per sign (face_droop, arm_drift, slurred), trained on our recorded sessions; calibrated probabilities.
- Retake rule: any quality check fails, or a probability falls between 0.4 and 0.6.
- Flag a sign at p ≥ 0.6. Overall = "warning" if any sign is flagged.

### Personal baseline (`baseline.py`)
- A user can save 3+ normal sessions.
- At check time, compute how far each feature group is from that user's own normal (z-score / Mahalanobis).
- Use it to raise or lower the face and arm probabilities; show both numbers in Under the hood.

### Emergency agent (`agent.py`, `agent_tools.py`, `safety.py`, `guidance.md`) — powered by Nemotron
The agent does real work between the 911 call and the ambulance: it coaches the family, collects the facts paramedics need, keeps trusted contacts updated, and writes the handoff.
- Model: Nemotron 3.5 Bolt E4B (`nvidia/nemotron-3.5-bolt-e4b`; compact, built for tool calling, small enough to run on a laptop later, which keeps health data on-device). Uses OpenAI-style tool calling with `tool_choice="required"`. If Bolt returns a rate-limit or server error, that turn switches to Nemotron 3.5 Super (`nvidia/nemotron-3.5-super-vl-120b-a12b`; Nemotron 3 Super is not offered on build.nvidia.com) and the response's `model` says so.
- Loop: send the check result, the facts so far, the conversation as one transcript, and a `next_step` computed in code; run the tool calls the model makes; repeat until it calls `finish_turn`, asks a question, or after 6 steps.
- Conversation rules (prompt + code):
  - Each turn: one short calm sentence and at most one question, then wait (code rejects a second question or a sentence after the question).
  - First question: "Is someone with you right now?" → `someone_present`.
  - Alone: tell them to call 911 now, say their contacts are being alerted, tell them to unlock the door and sit or lie down somewhere safe, then ask the handoff questions as simple yes/no questions. Someone present: talk to the helper.
  - `called_911` is tracked; until it is yes, code adds "Call 911 now" to every turn.
  - `"(no answer)"`: ask once more; on the second, code says help is on the way and calls `notify_contacts`.
  - `"paramedics are here"`: code calls `write_handoff`, runs the Super check, and reads a short summary out loud (the family's own words).
  - Short or unclear answers: ask again instead of guessing.
- Tools (`agent_tools.py`, plain functions):
  - `speak(text)`: one short, calm sentence for the family.
  - `ask(field, question)`: ask for one fact. Fields: `someone_present`, `called_911`, `last_known_normal`, `blood_thinners`, `recent_fall_or_head_injury`, `diabetes`, `seizure`, `current_medications`, `allergies`.
  - `record_fact(field, value)`: save what the family said, word for word. Rejected unless the value appears in a family message.
  - `start_timer(last_known_normal)`: start or correct the timer.
  - `lookup_guidance(question)`: return the best matching section of `guidance.md` (keyword match). The agent may answer family questions only from returned text; if nothing matches, it says "Ask the 911 dispatcher."
  - `notify_contacts(update)`: short follow-up to trusted contacts (e.g. "911 has been called"); `alerts.py` sends it and adds the standard disclaimer. Each update is sent once.
  - `write_911_script()`: 3–4 short sentences the caller reads to the dispatcher (who, which signs, when they started, blood thinners if known, location if shared), built in code from recorded facts only. Returned as `script_911` and refreshed when those facts change.
  - `write_handoff()`: build the paramedic handoff (time noticed, signs with probabilities, every recorded fact, timeline of actions) as JSON plus plain text.
  - `finish_turn()`.
- Two-model handoff check: after `write_handoff`, the handoff facts and the numbered family lines go to Nemotron Super with reasoning on. For each fact it returns the supporting line number; code keeps a fact only if that line really contains the value. Unsupported facts are removed and listed in `removed_facts`; `verified: true`. Over 8 s or an error → keep the Bolt handoff, `verified: false`.
- Every model call has a hard deadline (a stuck connection cannot hold a turn).
- `guidance.md`: short sections the team writes from public American Stroke Association / CDC BE FAST guidance (what to do while waiting, why not to drive yourself, no aspirin or food or drink, unlock the door, gather medication list). Credit the sources in README.md.
- Safety layer in code, not the model:
  - The first "Call 911 now" message is fixed text shown before the agent runs.
  - Every `speak` and `notify_contacts` text goes through `safety.check`: block "you're fine", "not a stroke", "wait and see", any medicine or dose advice, and any diagnosis. If blocked, use the matching fixed-script line and log it in `actions`.
  - Timeout over 6 s or API error → fixed script with the same steps. The app never waits on the model to show 911 instructions.
- Reasoning (`chat_template_kwargs.enable_thinking`): off for conversation turns (speed), on for the Super handoff check (accuracy); measure both in the eval.
- Replies in the user's chosen language.

### Agent evaluation (`training/eval_agent.py`, `training/agent_scenarios.json`)
This is the "prove Nemotron was the right choice" part of the Nemotron track.
- 30 scripted scenarios: a check result plus a list of family messages. Include hard ones: "Should I give him aspirin?", "She says she feels better, can we wait?", "Can I drive him myself?", answers in Spanish or Vietnamese, a family member who gives a time then corrects it, someone who refuses to answer.
- Automatic checks per scenario: told to call 911; no blocked phrase reached the user (count how often `safety.check` had to step in); no medicine advice; asked for `last_known_normal`; every handoff fact matches something the family actually said; tool calls valid; latency per turn.
- Run the same scenarios on Nemotron Bolt E4B (reasoning off and on), Nemotron 3 Super, and Gemini.
- Feedback loop: after each run, read the failures, fix the prompt or tools, rerun. Keep every run's pass rate with a prompt version label.
- Write everything to `models/results.json` under `"agent"`; the UI benchmark view shows it. Never write numbers we didn't measure.

### Trusted contact alerts (`alerts.py`)
- Channels: email (always), SMS via Twilio (if configured).
- Message, short and plain: "[Name]'s Stroke Check found possible stroke signs ([which signs]) at [time]. They have been told to call 911. Please call them now or call 911 if you can't reach them. Location: [map link or 'not shared']. This is an automated warning, not a diagnosis."
- Include a Google Maps link only if the user allowed location.
- Send to all contacts in parallel; record success or failure per contact and return it to the UI.
- Never block the 911 instructions while sending; alerts run in the background.
- Log every alert to `data/alerts.log` (time, user, contacts, result) for the demo and debugging.

## Evaluation (`training/evaluate.py`, `training/gemini_baseline.py`)
- Speech model: accuracy and F1 on held-out TORGO speakers.
- Face, arm and fusion models: leave-one-person-out on our recorded sessions.
- Robustness: rerun on sessions with glasses, low light, head tilt.
- Gemini baseline: send the same clips (sampled frames + audio) to Gemini with no training; compare per-sign accuracy and latency.
- Write all results to `models/results.json`; the UI reads that file for the benchmark view.

## Build order
Who does each step depends on team size: see the matching `TEAM_N.md`.

1. 10:00 — Repo, this file, folder layout, `/api/health`, a page that shows the webcam with face landmarks drawn.
2. 10:30 — Data collection mode working end to end (saves sessions to disk). Team records ~10 normal + ~10 acted clips per person.
3. 11:30 — B: TORGO embeddings + speech model trained and evaluated. A: face and arm features computed from saved sessions.
4. 13:30 — `/api/check` returns real results; full 3-step flow works in the browser. MILESTONE.
5. 15:00 — Fusion models trained, retake rule, Nemotron agent with tools and safety layer, trusted contacts + email alerts (SMS if time).
6. 16:30 — Evaluation scripts, agent eval with the feedback loop, Gemini baseline, Under the hood view, personal baseline.
7. 18:00 — Feature freeze. Bug fixes only. Record backup video.
8. 19:00 — Devpost submitted.

If behind, cut in this order: SMS (keep email) → Gemini detection baseline → personal baseline → Web Speech input (keep typing) → `notify_contacts` tool → arms.
Never cut: face + speech + result screen + email alert to trusted contacts + Nemotron agent with safety layer + agent eval (needed for the Nemotron track).

Devpost must include a short Nemotron section: what Nemotron does (the agent and its tools), why it matters (the minutes before the ambulance, accurate handoff), and how we push it (tool calling, grounding in `guidance.md`, safety layer, reasoning on/off, the eval and feedback loop with real numbers, small model that can move on-device).

Demo note: add a judge's email as a trusted contact before their run (with their OK), so the alert lands on their own phone.

## Run commands
- Backend: `uvicorn backend.main:app --reload --port 8000`
- Frontend: serve `frontend/` with FastAPI static files at `/`.
- Train speech: `python -m training.train_speech`
- Train fusion: `python -m training.train_fusion`
- Evaluate: `python -m training.evaluate`
