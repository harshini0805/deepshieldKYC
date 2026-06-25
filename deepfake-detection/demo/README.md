# FED-MEMF Demo (Simulated)

A presentation demo of the FED-MEMF deepfake-defense pipeline. It shows the full
gated eKYC flow and the tri-modal detection engine running live, driven by your
webcam, with operator-triggered REAL / DEEPFAKE outcomes.

> **This is a simulation.** No model is loaded and nothing is actually classified.
> All scores are generated for demonstration. A "SIMULATED DEMO" badge is shown
> on screen at all times. Use it to communicate the architecture and UX — not as
> evidence of detection accuracy.

## What it shows

- **Full pipeline rail:** Liveness -> Deepfake Detection -> Face Verification -> Decision, with each gate lighting up pass/fail in sequence.
- **Cross-modal detection engine:** Stream 1 (visual / micro-expression), Stream 2 (audio / voice-clone), Stream 3 (metadata / behavioral), combined by an animated cross-modal attention fusion with weights [0.4, 0.4, 0.2].
- **Live capture:** real webcam feed with a drifting face box, ~21 FPS counter, capture latency.
- **Verdict panel:** REAL / DEEPFAKE / UNCERTAIN, confidence gauge, EMA-smoothed P(FAKE), inference latency (~74-89 ms), and the 32-frame buffer filling.
- **Event log** mirroring the real service (SESSION_REJECTED, ws close 1008, etc.).

## Two ways to run

### 1. Standalone (no install) — simplest for a demo
Just open the file in a browser:

```
demo/index.html   ->  double-click, or drag into Chrome/Edge
```

Click **Start Camera**, allow the webcam, then press **Run REAL scenario** or
**Run DEEPFAKE scenario**. Everything runs locally in the page.

> Note: some browsers only allow webcam access over `http(s)` or `localhost`,
> not `file://`. If the camera won't start from a double-click, use mode 2 below
> (or run `python -m http.server` in the `demo/` folder and open the localhost URL).

### 2. Connected to the mock backend — looks like the real server
```
pip install fastapi uvicorn
cd demo
python server.py
```
Open **http://localhost:8000**, then flip the **Connect backend** switch on.
Now the REAL/DEEPFAKE scenarios are streamed frame-by-frame from the FastAPI
WebSocket (`/demo/ws/{session_id}`) in the same JSON schema as the production
`router.py`. If the backend is unreachable, the dashboard automatically falls
back to the standalone simulation, so the demo never breaks on stage.

## Presenter tips

- Start the camera before the audience is watching so the permission prompt is done.
- Run **REAL** first (green approve through all gates), then **DEEPFAKE** (red
  reject at stage 2, face verification skipped, ws closes) for contrast.
- The **Reset Session** button clears the buffer and EMA between runs.
- Keep the connected backend running as a fallback; the toggle lets you switch
  live if the webcam misbehaves.

## Honesty note (for Q&A)

If asked, be straight: this is a UI/architecture prototype. The trained model,
audio and metadata streams, federation, and the deck's benchmark numbers
(98.7% / 0.996 AUC) are not implemented here — those are targets from the source
paper (Rawat et al.), not measured results of this build.

## Files

| File | Purpose |
|------|---------|
| `index.html` | Self-contained dashboard (HTML+CSS+JS, no build step). |
| `server.py`  | Mock FastAPI backend: serves the page + streams simulated verdicts. |
| `README.md`  | This file. |
