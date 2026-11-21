# Stroke Check

A 30-second stroke warning check in your laptop browser, with an NVIDIA Nemotron agent that coaches the family until the ambulance arrives.

> Draft. Every number comes from `models/results.json`. Lines marked **[TODO]** are filled in after the final runs.

## Inspiration
In a stroke, minutes matter. CDC guidance says the treatments that work best must start within a few hours of the first symptoms, and that the time symptoms started helps doctors choose treatment. Yet families often wait to see if it passes, drive the person themselves, or can't say when the person was last normal. We wanted something anyone with a laptop could use: check the classic warning signs in 30 seconds and, if they show up, help the family do the right things while help is on the way.

## What it does
- **A 30-second check** with webcam and microphone: smile and raise your eyebrows (face droop), hold both arms out (arm drift), read one sentence (slurred speech).
- The result is either "No warning signs" or **"Possible stroke signs: call 911 now."** It's a warning, never a diagnosis, and it never says "you're fine".
- When signs are found, "Call 911 now" appears instantly as fixed text. Then a **Nemotron emergency agent**:
  - starts a last-known-normal timer from the family's own words;
  - tells them, one calm step at a time, not to give food, drink or medicine, and to stay with the person and unlock the door;
  - answers questions ("Should I give him aspirin?", "Can I drive him myself?") **only from approved CDC/NHS/MedlinePlus guidance**;
  - collects the facts paramedics ask for (blood thinners, falls, diabetes, seizures, medications, allergies), saved **word for word**;
  - writes a **paramedic handoff** with the timeline;
  - works in English, Spanish and Vietnamese.
- **Trusted contacts** get an email with what was detected, when, and where, after a 20-second "Cancel — false alarm" window. This is always in addition to 911.

## How we built it
- **Face and arms:** MediaPipe Face, Pose and Hand Landmarkers in the browser. We measure mouth-corner lift, eyebrow and eye-opening asymmetry, wrist drop and palm rotation, normalized by eye distance and shoulder width.
- **Speech:** wav2vec2-base embeddings → logistic regression trained on the TORGO dysarthric speech dataset, **split by speaker**, so test speakers are never in training. Plus a Whisper transcript (word error rate vs. the sentence), speaking rate, pauses, and recording-quality checks.
- **Decision:** one calibrated classifier per sign, trained on our own recordings with leave-one-person-out. The app asks for a retake when quality fails or a score is uncertain.
- **Agent:** Nemotron through build.nvidia.com with OpenAI-style tool calling. Tools: `speak`, `ask`, `record_fact`, `start_timer`, `lookup_guidance`, `notify_contacts`, `write_handoff`, `finish_turn`. A 6-second turn budget; on timeout or API error it falls back to a fixed script with the same steps.
- **Safety layer in code, not in the prompt:** every sentence the family hears goes through a checker that blocks "you're fine", "not a stroke", "wait and see", medicine or dose advice, and diagnoses, in three languages, and swaps in a fixed-script line.
- **Backend:** FastAPI. **Frontend:** plain HTML/JS.

## Why Nemotron
**What the agent does.** It isn't a chatbot bolted on. Between the 911 call and the ambulance it does real work: it runs the timer, coaches the family step by step, answers from approved guidance, records the facts paramedics need, keeps contacts updated, and writes the handoff. Everything happens through tool calls we can log, check and show in the UI's "Agent actions" list.

**Why it matters.** Paramedics and doctors ask for the time symptoms started and the person's medical history. A panicked family often can't give them. The handoff gives them the family's exact words, a timeline, and the check's flagged signs.

**How we pushed it.**
- *Tool calling for everything.* The model never talks freely to the family. It acts through tools, and `record_fact`/`start_timer` reject any value that isn't copied from what the family actually said, so the handoff can't contain invented facts.
- *Grounding.* Family questions go through `lookup_guidance`, which returns a section of `guidance.md` built from CDC, NHS and MedlinePlus pages. If nothing matches, the agent says "Ask the 911 dispatcher."
- *A safety layer outside the model,* plus a deterministic fallback script.
- *Code plans, the model talks.* Code tracks what's still missing (911, last-known-normal, food/drink, door, each fact) and hands the model a `next_step`. The model handles free-text answers, corrections, questions and language.
- *Measured, not assumed.* 30 scripted scenarios, including the hard ones: "Should I give him aspirin?", "She feels better, can we wait?", "Can I drive him myself?", a time that gets corrected, someone who refuses to answer, and Spanish and Vietnamese speakers. There are 13 automatic checks (told to call 911, safe output, asked last-known-normal, facts grounded, last-known-normal correct, valid tool calls, guidance used, right language, latency…).
- *Feedback loop.* We read the failures after each run, fixed the prompt or tools, and reran:

| Prompt version | What we changed | Nemotron 3.5 Super pass rate | Nemotron 3.5 Bolt (reasoning off) pass rate |
| --- | --- | --- | --- |
| v1 | first version | 0% (p95 turn 6.9 s) | 0% (p95 turn 21.2 s) |
| v2 | 6 s turn budget, no silent turns, single notifications | 70% (p95 3.2 s) | 30% (p95 3.7 s) |
| v3 | code-planned next step, smarter fallback | **90%** (median 1.0 s, p95 2.4 s) | **[TODO]** |

In v3, Nemotron Super passed 100% on: told to call 911, safe output, asked last-known-normal, facts grounded word for word, valid tool calls, and guidance used. It recorded last-known-normal correctly in 96% of scenarios, including corrections. The safety layer didn't have to step in once.

- *Reasoning on vs off, measured.* With reasoning on, Bolt's median turn was 17.6 s in v1, far over our 6 s budget. So live turns run with reasoning off. **[TODO: v3 reasoning-on numbers]**
- *Gemini comparison.* **[TODO]** Our free-tier Gemini key was rate-limited during the v2 run (116 of 122 turns rejected), so that run is labeled "not a model score" and excluded.
- *Small model, path to on-device.* Bolt E4B is small enough to run on a laptop later, which would keep health data on the device. **[TODO: state the Bolt vs Super trade-off from the final numbers]**

## Challenges we ran into
- Bolt sometimes wrote tool calls as plain text in its own template format on later turns. Forcing `tool_choice="required"` and sending the conversation as one transcript fixed it, and we kept a parser as a backup.
- Our own safety layer blocked our own disclaimers ("not a diagnosis") and the Vietnamese word for staff ("nhân viên", which contains "viên", pill). The eval caught both.
- API rate limits. Our eval now labels runs dominated by rate-limit errors as "not a model score" so they can't be mistaken for model quality.
- TORGO speakers differ a lot from each other, so held-out-speaker accuracy is much lower than within-speaker accuracy.

## Accomplishments that we're proud of
- **Speech model on 4 held-out TORGO speakers** (1,600 clips never seen in training): accuracy **0.606**, F1 **0.673**, AUC **0.702**. Across all 15 speakers, leave-one-speaker-out: accuracy 0.686, F1 0.701, AUC 0.772. Trained on 5,420 clips (2,869 dysarthric).
- **Face, arm and fusion detection, leave-one-person-out on our recordings:** **[TODO from results.json "detection"]**
- A Nemotron agent that went from 0% to 90% on our scenario suite through a measured feedback loop.

## What we learned
- A safety layer in code is easy to test, and it needs its own tests: ours had false positives the eval found.
- For a small model, letting code plan the conversation and letting the model handle language works much better than one big prompt.
- Split by speaker or by person, always. The honest number is much lower than the easy one.

## What's next
- Run Nemotron Bolt on-device so health data never leaves the laptop.
- Record more people, and test with clinicians on real stroke-patient speech rather than dysarthria as a proxy.
- SMS alerts and voice input for the family.

## Built with
Python, FastAPI, NVIDIA Nemotron (build.nvidia.com), MediaPipe, wav2vec2, faster-whisper, scikit-learn, XGBoost, PyTorch, Hugging Face Transformers and Datasets, jiwer, ffmpeg, Gmail SMTP, HTML/JavaScript.

**This is a warning tool, not a diagnosis. If you think someone is having a stroke, call 911.**
