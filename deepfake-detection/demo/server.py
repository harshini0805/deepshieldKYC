"""
FED-MEMF Demo - Mock FastAPI backend
====================================
A SIMULATION backend for the demo dashboard. It serves index.html and exposes
a WebSocket that streams fake (simulated) per-frame verdicts in the SAME schema
as the real router.py, so the dashboard can run "wired to the server" for a more
convincing demo. No model is loaded and nothing is actually classified.

Run:
    pip install fastapi uvicorn
    python server.py
Then open http://localhost:8000  and flip the "Connect backend" switch.

Endpoints:
    GET  /                       -> the dashboard (index.html)
    GET  /demo/health            -> liveness check
    POST /demo/reset/{sid}       -> clear a session
    WS   /demo/ws/{sid}          -> send {"scenario":"real"|"fake"}; receive frames
"""

from __future__ import annotations

import asyncio
import random
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

HERE = Path(__file__).resolve().parent
INDEX = HERE / "index.html"

app = FastAPI(title="FED-MEMF Demo (Simulated)", version="0.1.0")

FAKE_THRESHOLD = 0.60
UNCERTAIN_LOW = 0.35
BUFFER_SIZE = 32


@app.get("/")
async def root():
    return FileResponse(INDEX)


@app.get("/demo/health")
async def health():
    return {"status": "ok", "mode": "SIMULATED", "backbone": "tf_efficientnet_b4_ns (mock)"}


@app.post("/demo/reset/{session_id}")
async def reset(session_id: str):
    return {"session_id": session_id, "status": "reset"}


def _targets(kind: str):
    if kind == "fake":
        return dict(s1=random.uniform(0.71, 0.86),
                    s2=random.uniform(0.63, 0.82),
                    s3=random.uniform(0.55, 0.74),
                    live=random.uniform(0.82, 0.93))
    return dict(s1=random.uniform(0.05, 0.16),
                s2=random.uniform(0.06, 0.20),
                s3=random.uniform(0.10, 0.24),
                live=random.uniform(0.94, 0.99))


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


async def _run_scenario(ws: WebSocket, session_id: str, kind: str):
    """Stream a full pipeline run, frame by frame, in the dashboard's schema."""
    tgt = _targets(kind)
    live = tgt["live"]

    # Stage 1: liveness
    await ws.send_json({"stage": "liveness", "state": "run", "label": "analyzing"})
    await asyncio.sleep(0.5)
    await ws.send_json({"stage": "liveness", "state": "pass", "label": f"PASS {live:.2f}"})

    # Stage 2: deepfake detection over a 32-frame window
    await ws.send_json({"stage": "deepfake", "state": "run", "label": "32-frame window"})
    ema = 0.5
    has_prior = False
    for n in range(1, BUFFER_SIZE + 1):
        p = n / BUFFER_SIZE
        s1 = _clamp(tgt["s1"] * p + random.uniform(-0.05, 0.05))
        s2 = _clamp(tgt["s2"] * p + random.uniform(-0.05, 0.05))
        s3 = _clamp(tgt["s3"] * p + random.uniform(-0.05, 0.05))
        fused = 0.4 * s1 + 0.4 * s2 + 0.2 * s3
        ema = 0.6 * fused + 0.4 * ema if has_prior else fused
        has_prior = True
        lat = random.uniform(74, 89)
        conf = min(abs(ema - 0.5) * 2, 1.0)
        verdict = ("DEEPFAKE" if ema >= FAKE_THRESHOLD
                   else "UNCERTAIN" if ema >= UNCERTAIN_LOW else "REAL")
        await ws.send_json({
            "session_id": session_id,
            "frame_index": n,
            "verdict": verdict,
            "smoothed_score": round(ema, 4),
            "confidence": round(conf, 4),
            "visual_score": round(s1, 4),
            "audio_score": round(s2, 4),
            "metadata_score": round(s3, 4),
            "inference_latency_ms": round(lat, 2),
            "buffer_fill": n,
            "liveness_score": round(live, 3),
            "timestamp_ms": time.time() * 1000,
        })
        await asyncio.sleep(0.048)

    # Stage resolution
    if verdict == "DEEPFAKE":
        await ws.send_json({"stage": "deepfake", "state": "fail", "label": f"DEEPFAKE {ema:.2f}"})
        await ws.send_json({"stage": "faceverify", "state": None, "label": "skipped"})
        await ws.send_json({"stage": "decision", "state": "fail", "label": "REJECTED"})
        await ws.send_json({
            "event": "SESSION_REJECTED", "reason": "DEEPFAKE_DETECTED",
            "session_id": session_id, "score": round(ema, 4),
        })
    elif verdict == "UNCERTAIN":
        await ws.send_json({"stage": "deepfake", "state": "run", "label": "UNCERTAIN"})
        await ws.send_json({"stage": "decision", "state": "run", "label": "REVIEW"})
        await ws.send_json({"done": True})
    else:
        await ws.send_json({"stage": "deepfake", "state": "pass", "label": f"REAL {ema:.2f}"})
        await ws.send_json({"stage": "faceverify", "state": "run", "label": "matching"})
        await asyncio.sleep(0.7)
        match = random.uniform(0.88, 0.97)
        await ws.send_json({"stage": "faceverify", "state": "pass", "label": f"MATCH {match:.2f}"})
        await ws.send_json({"stage": "decision", "state": "pass", "label": "APPROVED"})
        await ws.send_json({"done": True})


@app.websocket("/demo/ws/{session_id}")
async def demo_ws(websocket: WebSocket, session_id: str):
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive_json()
            kind = (msg or {}).get("scenario", "real")
            await _run_scenario(websocket, session_id, "fake" if kind == "fake" else "real")
    except WebSocketDisconnect:
        pass
    except Exception:
        # keep the demo robust; never crash the socket on a stray message
        try:
            await websocket.send_json({"done": True})
        except Exception:
            pass


if __name__ == "__main__":
    print("=" * 56)
    print("  FED-MEMF DEMO BACKEND  (SIMULATED - not a real model)")
    print("  Open http://localhost:8000  then toggle 'Connect backend'")
    print("=" * 56)
    uvicorn.run(app, host="0.0.0.0", port=8000)
