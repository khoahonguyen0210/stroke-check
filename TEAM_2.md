# Team of 2

Read `CLAUDE.md` first. Ask which person you are working for (Person 1 or Person 2) if you were not told.
Only edit the files your person owns. Each person commits a stub of their modules in the first 30 minutes so the other can integrate.

## Who owns what
| | Person 1: app + vision | Person 2: speech + Nemotron agent |
| --- | --- | --- |
| Owns | `frontend/*`, `backend/main.py`, `backend/storage.py`, `backend/face.py`, `backend/arms.py`, `backend/fusion.py`, `backend/alerts.py`, `training/train_fusion.py` | `backend/speech.py`, `backend/agent.py`, `backend/agent_tools.py`, `backend/safety.py`, `backend/guidance.md`, `training/train_speech.py`, `training/evaluate.py`, `training/eval_agent.py`, `training/agent_scenarios.json`, `training/gemini_baseline.py`, `README.md`, Devpost, slides |
| Main job | The app works end to end; face and arm checks; final decision; alerts | Speech model on TORGO; Nemotron agent and its eval; pitch |

Out of scope for 2 people: personal baseline, SMS (email only), Web Speech input.

## Schedule
| Time | Person 1 | Person 2 |
| --- | --- | --- |
| 10:00–10:30 | Folder layout, `/api/health`, webcam page with face landmarks, stubs for `face/arms/fusion/alerts` | Stubs for `speech/agent/safety`; one test call to Nemotron; start TORGO embedding extraction in the background |
| 10:30–11:15 | Data collection mode (shared session format) | Speech classifier on cached embeddings, speaker-split evaluation |
| 11:15–11:45 | Both record: ~10 normal + ~10 acted takes each, two lighting setups | same |
| 11:45–13:30 | `face.py` + `arms.py` features and quality; `/api/check` with thresholds | `speech.py`: classifier probability, Whisper WER, rate, pauses, quality |
| 13:30 | **Integration: full flow shows a real result.** Pair up until it works | same |
| 13:30–15:30 | `fusion.py` trained (leave-one-person-out), retake rule, result cards; `alerts.py` (email, 20 s cancel, log) and contacts endpoints | `safety.py` + fixed script; `guidance.md`; `agent_tools.py` + `agent.py` tool loop on Nemotron Bolt |
| 15:30–17:00 | Emergency panel (answer box, handoff card, agent actions list), contacts screen; arms into fusion | `agent_scenarios.json` + `eval_agent.py`: Bolt off/on vs Super vs Gemini, fix-and-rerun loop; `evaluate.py` → `models/results.json` |
| 17:00–18:00 | Polish UI, Under the hood, benchmark view | README credits, Devpost with Nemotron section, slides; `gemini_baseline.py` only if time |
| 18:00 | **Feature freeze.** Bugs only; record backup video | same |
| 19:00 | Submit Devpost together | same |

## Prompts
**Person 1**
1. "Read CLAUDE.md and TEAM_2.md. I am Person 1. Do my 10:00 tasks, including stubs with the shared signatures."
2. "Build data collection mode. Save one take and show the files."
3. "Write face.py and arms.py. Print features for a normal take and an acted take side by side."
4. "Wire /api/check end to end using Person 2's speech stub until the real one lands."
5. "Train fusion with leave-one-person-out, add the retake rule, build the result cards."
6. "Write alerts.py and the contacts endpoints. Send me a test email."
7. "Build the emergency panel against the /api/agent endpoints in CLAUDE.md."

**Person 2**
1. "Read CLAUDE.md and TEAM_2.md. I am Person 2. Commit stubs, make one test call to Nemotron, then start TORGO embedding extraction with caching."
2. "Train the speech classifier split by speaker. Print accuracy and F1 on held-out speakers."
3. "Write speech.py per CLAUDE.md and test it on one recorded take."
4. "Write safety.py and the fixed script, then guidance.md, agent_tools.py and agent.py. Run one scenario and print every tool call and the handoff."
5. "Write agent_scenarios.json (30 scenarios) and eval_agent.py. Run Bolt off/on, Super and Gemini. Show failures, fix the prompt, rerun, save every run."
6. "Write evaluate.py and add the detection numbers to models/results.json."
