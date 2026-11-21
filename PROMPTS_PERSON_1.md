# Person 1 prompts (app + vision + alerts)

Paste these into Claude Code one at a time, in order. Don't start the next one until the current one runs and is pushed.
Your laptop is the demo laptop: all takes get recorded here.

## 10:00 Create the repo (you type this, not Claude)
```
cd stroke-check
git init
git add CLAUDE.md TEAM_2.md .gitignore .env.example
git commit -m "Project plan"
```
Create an empty GitHub repo, then:
```
git remote add origin <repo url>
git branch -M main
git push -u origin main
```
Add Person 2 as a collaborator and send them the repo link.

## 1. 10:05 Skeleton + stubs
```
Read CLAUDE.md and TEAM_2.md. I am Person 1. Do my 10:00–10:30 tasks:
- folder layout from CLAUDE.md
- backend/main.py: FastAPI with GET /api/health and frontend/ served as static files at /
- frontend page that shows the webcam with MediaPipe face landmarks drawn live
- stubs for face.py, arms.py, fusion.py, alerts.py, storage.py with the exact signatures in CLAUDE.md, returning fixed values
- stub routes for every endpoint in the API section, returning fixed responses in the documented shapes
Run it, tell me the URL to open, then commit and push.
```

## 2. 10:30 Data collection mode
```
Build data collection mode per CLAUDE.md: start screen with name field and label dropdown, the 3 steps with on-screen countdowns, record face landmarks with phase, pose + hand landmarks, and audio, then POST /api/collect, which saves to data/sessions/<person>/<take_id>/ in the shared format. Save one test take and show me the files on disk. Commit and push.
```

## 3. 11:15 Record (no prompt)
Both of you record on this laptop: ~10 normal + ~10 acted takes each, half in dim light. Zip 3 normal and 3 slurred takes and send them to Person 2.

## 4. 11:45 Face and arm features
```
Write backend/face.py and backend/arms.py per CLAUDE.md. Verify the landmark indices against the MediaPipe maps first. Print features and quality for one normal take and one acted take side by side so I can see the asymmetry. Commit and push.
```

## 5. 12:45 Full flow
```
Wire POST /api/check end to end: run face, arms and speech (Person 2's stub until the real one is pushed), use simple thresholds for now, return the response shape from CLAUDE.md with latency_ms, and build the result screen with one card per sign plus the retake message. Test with a real check in the browser. Commit and push.
```
13:30: pull Person 2's speech.py and models/speech.joblib, rerun a real check together. **Milestone.**

## 6. 13:30 Fusion
```
Write training/train_fusion.py: features for every session in data/sessions using face.py, arms.py and speech.py, one calibrated classifier per sign, leave-one-person-out evaluation printed per sign, models saved to models/. Update fusion.predict with the retake rule (quality fail or p between 0.4 and 0.6) and flag at p >= 0.6. Commit and push, including the model files.
```

## 7. 14:30 Alerts and contacts
```
Write backend/alerts.py and the contacts endpoints per CLAUDE.md: Gmail SMTP email from .env, consent required, send to all contacts in parallel, per-contact result, log to data/alerts.log, the test endpoint, and alerts.notify(user, text) for the agent. Build the trusted contacts screen. Send a test alert to my email and show me the log line. Commit and push.
```

## 8. 15:30 Emergency panel
```
Build the emergency panel per CLAUDE.md frontend steps 8 and 9, using /api/agent/start and /api/agent/message (stub responses until Person 2's agent is pushed): the fixed "Possible stroke signs. Call 911 now." banner first, the last-known-normal timer, speechSynthesis for each "say", an answer box, the handoff card filling in live, the Agent actions list, and the 20-second alert countdown with a big Cancel button that calls /api/alert if not cancelled. Commit and push.
```

## 9. 16:30 Arms into fusion, Under the hood, benchmark view
```
Add the arm features to fusion and retrain. Build the Under the hood toggle (raw features, per-model probabilities, latency) and a Benchmark view that reads models/results.json and shows only the numbers in it. Commit and push.
```

## 10. 17:15 Polish
```
Polish the UI: pixel font for headings, high contrast, clear big result text, works at 1366x768. Add "This is a warning tool, not a diagnosis" under every result. Don't change any logic. Commit and push.
```

## 11. 18:00 Feature freeze
```
Feature freeze. Run the test checklist: a normal take from someone not in training, each acted sign alone, dim light (should ask for a retake), the full warning flow with the alert email and Cancel. Fix only bugs you find. Commit and push.
```
Then record the 2-minute backup video.
