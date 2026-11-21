# Person 2 prompts (speech + Nemotron agent + pitch)

Paste these into Claude Code one at a time, in order. Don't start the next one until the current one runs and is pushed.

## 10:00 Get the repo (you type this, not Claude)
```
git clone <repo url from Person 1>
cd stroke-check
```
Copy your `.env` into the folder (NVIDIA_API_KEY, GEMINI_API_KEY). Never commit it.

## 1. 10:05 Stubs, Nemotron test, start TORGO
```
Read CLAUDE.md and TEAM_2.md. I am Person 2. Pull first, then:
- stubs for backend/speech.py, agent.py, agent_tools.py, safety.py with the exact signatures in CLAUDE.md, returning fixed values (agent stub returns a fixed say, actions, handoff, state)
- one test call to Nemotron 3.5 Bolt E4B through build.nvidia.com with NVIDIA_API_KEY and the openai package, including one simple tool call; copy the model id from the model card and print the reply
- start training/train_speech.py step 1 in the background: wav2vec2-base mean-pooled embeddings for TORGO, saved in chunks to data/torgo_embeddings/ so a crash doesn't lose progress; show me how to check progress
Commit and push the stubs.
```

## 2. 10:45 Speech classifier
```
Train the speech classifier on the cached embeddings, split by speaker so test speakers are never in training. Try logistic regression and XGBoost, print accuracy and F1 on held-out speakers, save the best to models/speech.joblib, and write the numbers to models/results.json under "speech". Commit and push, including models/speech.joblib.
```

## 3. 11:15 Record (no prompt)
Go to Person 1's laptop and record ~10 normal + ~10 acted takes. Get the zip of 3 normal and 3 slurred takes from Person 1.

## 4. 11:45 speech.py
```
Write backend/speech.py per CLAUDE.md: ffmpeg webm to 16 kHz mono wav, p_slurred from models/speech.joblib, faster-whisper base transcript and WER vs the sentence with jiwer, speaking rate, pause ratio, and quality (length, loudness, clipping). Test it on the takes Person 1 sent and print the features for normal vs slurred side by side. Commit and push.
```

## 5. 13:30 Safety layer
```
Write backend/safety.py and the fixed script per CLAUDE.md. safety.check must block "you're fine", "not a stroke", "wait and see", any medicine or dose advice, and any diagnosis, in English, Spanish and Vietnamese. Show it blocking 10 bad lines and passing 10 good ones. Commit and push.
```

## 6. 14:00 Nemotron agent
```
Write backend/guidance.md from the public American Stroke Association and CDC BE FAST pages (short sections, sources listed at the top), then backend/agent_tools.py and backend/agent.py per CLAUDE.md: Nemotron Bolt tool loop (max 6 steps), every tool listed, safety.check on every speak and notify_contacts, fixed script on timeout over 6 s or API error. Use alerts.notify from Person 1 for notify_contacts. Run one scenario from the terminal and print every tool call, what the family hears, and the final handoff. Commit and push.
```

## 7. 15:15 Agent eval + feedback loop
```
Write training/agent_scenarios.json with 30 scenarios per CLAUDE.md (include the hard ones) and training/eval_agent.py with the automatic checks. Run Bolt with reasoning off and on, Nemotron 3 Super, and Gemini. Print a table of pass rate per check and latency per model, list the failures, then fix the agent prompt or tools, rerun, and save every run with a prompt version label to models/results.json under "agent". Commit and push.
```
Repeat "Show me the failures from the last run, fix them, rerun and save" until the pass rate stops improving.

## 8. 16:30 Detection evaluation
```
Write training/evaluate.py: add held-out speaker results, leave-one-person-out results for face, arms and fusion (reuse Person 1's training/train_fusion.py functions), and robustness splits (glasses, low light) to models/results.json under "detection". It must run on Person 1's laptop where the takes are. Commit and push.
```

## 9. 17:00 README, Devpost, slides
```
Write README.md: what it does, how to run it, and a Credits section listing every library, model, dataset (TORGO), guidance source and AI tool we used. Then draft devpost.md with Inspiration, What it does, How we built it, Challenges, Accomplishments (only numbers from models/results.json), What we learned, What's next, and a "Why Nemotron" section: what the agent does, why it matters, how we push it (tools, grounding, safety layer, reasoning on/off, eval and feedback loop with our real numbers, small model that can run on-device). Commit and push.
```
Make 5 slides yourself: hook, demo, how it works, benchmark (detection + agent numbers), what's next.

## 10. 17:45 Gemini detection baseline (only if everything else works)
```
Write training/gemini_baseline.py per CLAUDE.md and add the results to models/results.json under "gemini_detection". Commit and push.
```

## 11. 18:00 Feature freeze
Bugs only. Rehearse the 2-minute demo with Person 1. Submit Devpost by 19:30 at the latest.
