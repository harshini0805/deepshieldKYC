# Demo Speaker Script — Deepfake Detection Layer

> Scope: the **Deepfake Detection layer only** (Stage 2 of the pipeline).
> Stage directions are in [brackets]. Read the rest aloud. Target length ~3–5 min.

---

## 0. Opening — where this layer sits

"Now I'll demonstrate the core of our system: the **deepfake detection layer**. In
our pipeline this is the second gate. The liveness check has already confirmed
that there is a real, live person in front of the camera. The job of this layer
is to answer a harder question: even if the input is live, **is the identity
itself synthetic?** A deepfake can be played through a real camera, so liveness
alone is not enough. This is the layer that catches the truth attack."

[Point to the pipeline rail at the top: Liveness → Deepfake Detection → Face Verification → Decision.]

---

## 1. Capture and the 32-frame window

[Click **Start Camera**. Let the face box appear.]

"The frontend captures the webcam stream and sends frames to the backend over a
WebSocket, at around 21 frames per second. You can see the live face box and the
frame rate here."

"One important design choice: we do **not** decide from a single frame. The
detector collects a **32-frame window**. Deepfakes often look convincing in a
still image but break down over time — they show temporal flicker, frozen
micro-movements, or unnatural synchronization. Analyzing a short clip instead of
one frame is what lets us see those artifacts. You'll see this 32-frame buffer
fill up on the right as we run a verification."

---

## 2. The three streams

"Inside this layer we don't trust just one signal. We run **three independent
streams**, and then check whether they agree with each other."

[Point to the three bars in the Cross-Modal Detection Engine panel.]

"The first is the **visual stream**, handled by a micro-transformer. It focuses on
micro-expressions — tiny involuntary facial movements like muscle twitches and
nasolabial fold changes that are very hard for synthetic media to reproduce
naturally."

"The second is the **audio stream**, an audio CNN. It analyzes the voice as a
Mel-spectrogram — essentially a picture of the sound — to detect cloned speech by
its rhythm, tone, and acoustic artifacts."

"The third is the **metadata stream**, an LSTM. It looks at behavior and device
signals: frame-rate stability, capture timing, gaze shifts, and other telemetry
that reveals injection or replay attacks."

---

## 3. Cross-modal attention fusion

[Point to the fusion box with weights [0.4, 0.4, 0.2].]

"These three streams feed a **cross-modal attention fusion** layer. This is the key
idea of the whole system: an attacker might fake one modality very well — say, a
perfect face — but faking the face, the voice, and the behavior all at once, in
perfect synchronization, is far harder. The fusion layer combines the three
scores and looks for disagreement between them."

"The output is a single probability that the identity is fake. We smooth it across
the window using an exponential moving average so one noisy frame can't flip the
decision, and then we apply two thresholds: above **0.60** we call it a deepfake,
below **0.35** we call it real, and the gray zone in between is sent to human
review instead of being guessed."

---

## 4. Scenario A — a genuine user

[Click **Run REAL scenario**. Talk while it runs.]

"Here is a genuine user. Watch the three stream scores — they stay low, near zero,
because the face, voice, and behavior are all consistent and natural. The 32-frame
buffer fills, the fused probability stays well under our threshold..."

[Wait for the verdict.]

"...and the layer returns **REAL**. Notice the inference latency — around 80
milliseconds per window — so this happens in near real time; the user doesn't feel
it. Because this gate passed, the pipeline now moves on to face verification, and
the session is approved."

[Point to the gates lighting green across the rail.]

---

## 5. Scenario B — a deepfake attack

[Click **Reset Session**, then **Run DEEPFAKE scenario**. Talk while it runs.]

"Now the same flow, but this time the input is synthetic. Watch the streams climb —
the visual and audio scores rise as the model picks up artifacts, and you can see
the fused probability crossing our 0.60 threshold..."

[Wait for the verdict.]

"...and the verdict flips to **DEEPFAKE**. Two things happen immediately. First,
the pipeline **stops** — notice face verification is skipped; there's no point
matching a face we already know is synthetic. Second, the server emits a
**SESSION_REJECTED** event and closes the connection. The decision gate turns red:
rejected."

[Point to the event log showing SESSION_REJECTED / the red decision gate.]

"So this layer behaves like a circuit breaker. The moment it has enough evidence,
it halts the verification rather than letting a fabricated identity continue
downstream."

---

## 6. Why this design matters (one-line close)

"To summarize this layer: instead of trusting a single face image, we analyze a
short clip across three independent signals, fuse them, and require them to agree.
That combination — temporal analysis plus cross-modal consistency — is what makes
it much harder to fool than traditional, single-input KYC checks."

---

## Presenter notes (not read aloud)

- **The "Simulated Demo" badge is intentional.** This demo shows the real
  architecture and live UX, but the scores are generated for the walkthrough — the
  model isn't trained in this build. If anyone asks directly, say so plainly:
  *"This is an architecture and UX prototype; the detection scores here are
  simulated. Our trained results are reported separately."* It reads as confident,
  not evasive.
- **On the benchmark numbers (98.7% / 0.996 AUC / 82 ms):** those are targets from
  the source research (Rawat et al.) for the full federated tri-modal system, not
  measured output of this demo. If you mention them, frame them as *"the published
  benchmark our architecture targets,"* not *"our results."*
- **Fallback:** if the webcam fails, flip **Connect backend** on (with `server.py`
  running) — the same scenarios stream from the FastAPI server, and the page
  auto-falls-back to local simulation if the server isn't reachable.
- **Suggested order:** REAL first (clean green approve), then DEEPFAKE (red reject)
  for contrast. Reset between runs.
