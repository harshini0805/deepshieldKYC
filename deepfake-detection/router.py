"""
FED-MEMF Deepfake Layer — FastAPI Router

Integrates the deepfake detection gate (Slide 13: [DEEPFAKE DETECTION])
into the existing FastAPI backend, which already handles liveness via WebSocket.

Pipeline position (Slide 13):
  [LIVENESS CHECK] → [DEEPFAKE DETECTION] ← this module → [FACE VERIFICATION]

Integration into your existing main.py:
    from deepfake_layer.router import deepfake_router
    app.include_router(deepfake_router)

The WebSocket at /ws/deepfake accepts the same JPEG frame format as
the existing liveness endpoint. It expects an optional JSON header
field carrying the liveness bbox and score from the upstream step.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import asdict
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from detector import DeepfakeDetector
from temporal import Verdict, WindowResult

# ─── Router ──────────────────────────────────────────────────────────────────

deepfake_router = APIRouter(prefix="/deepfake", tags=["deepfake-detection"])

# Single global detector instance (loaded once at startup)
_detector: Optional[DeepfakeDetector] = None


def get_detector() -> DeepfakeDetector:
    global _detector
    if _detector is None:
        raise RuntimeError(
            "DeepfakeDetector not initialized. "
            "Call init_detector() on app startup."
        )
    return _detector


def init_detector(weights_path: Optional[str] = None, force_cpu: bool = False) -> None:
    """
    Call this from your FastAPI lifespan or startup event.

    Example (main.py):
        from contextlib import asynccontextmanager
        from fastapi import FastAPI
        from deepfake_layer.router import deepfake_router, init_detector

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            init_detector(weights_path="weights/ff_efficientnet_b4.pth")
            yield

        app = FastAPI(lifespan=lifespan)
        app.include_router(deepfake_router)
    """
    global _detector
    _detector = DeepfakeDetector.build(
        weights_path=weights_path,
        force_cpu=force_cpu,
    )
    print("[DeepfakeRouter] Detector ready.")


# ─── Response Models ──────────────────────────────────────────────────────────

class DeepfakeFrameResult(BaseModel):
    """
    JSON message sent back over WebSocket for each frame.
    If no window has been completed yet, verdict = "PENDING".
    """
    session_id: str
    frame_index: int
    verdict: str                  # REAL | DEEPFAKE | UNCERTAIN | PENDING
    smoothed_score: float         # P(FAKE) after EMA smoothing [0, 1]
    confidence: float             # Distance from boundary [0, 1]
    visual_score: float
    audio_score: float            # Stub (0.5 until Audio CNN is integrated)
    metadata_score: float         # Stub (0.5 until Metadata LSTM is integrated)
    inference_latency_ms: float
    buffer_fill: int              # Frames currently in buffer
    timestamp_ms: float


class SessionStatus(BaseModel):
    session_id: str
    frames_seen: int
    current_score: float
    verdict: str
    buffer_fill: int


# ─── REST Endpoints ───────────────────────────────────────────────────────────

@deepfake_router.get("/health")
async def health_check():
    """Quick health check — verifies model is loaded."""
    try:
        detector = get_detector()
        return {
            "status": "ok",
            "device": str(detector.device),
            "backbone": detector.config.backbone.name,
        }
    except RuntimeError as e:
        return {"status": "not_ready", "error": str(e)}


@deepfake_router.get("/session/{session_id}", response_model=SessionStatus)
async def get_session_status(session_id: str):
    """Get the current verdict and score for an ongoing session."""
    detector = get_detector()
    return SessionStatus(
        session_id=session_id,
        frames_seen=detector.frames_seen,
        current_score=detector.current_score,
        verdict=detector.session_verdict.value,
        buffer_fill=detector.buffer.buffer_fill,
    )


@deepfake_router.post("/reset/{session_id}")
async def reset_session(session_id: str):
    """
    Reset the detector for a new verification session.
    Call this between user sessions to clear the frame buffer.
    """
    detector = get_detector()
    detector.reset_session()
    return {"session_id": session_id, "status": "reset"}


# ─── WebSocket Handler ────────────────────────────────────────────────────────

@deepfake_router.websocket("/ws/{session_id}")
async def deepfake_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time deepfake detection.

    Message format (client → server):
    Each message is either:
      Option A: Raw bytes (JPEG frame)
      Option B: JSON metadata frame with binary attachment:
        {
          "liveness_score": 0.94,
          "bbox": [x1, y1, x2, y2],  // optional, from liveness layer
          "capture_latency_ms": 47.2
        }
        Followed by the raw frame bytes in the next message.

    For simplicity TODAY: accept raw JPEG bytes and detect face internally.
    Add Option B when integrating with liveness layer to share the MediaPipe bbox.

    Message format (server → client):
    JSON conforming to DeepfakeFrameResult schema.
    """
    await websocket.accept()
    detector = get_detector()
    detector.reset_session()

    frame_counter = 0
    pending_metadata: dict = {}  # Stores JSON metadata until next binary message
    last_frame_time = time.time() * 1000

    try:
        while True:
            data = await websocket.receive()

            current_time = time.time() * 1000
            capture_latency_ms = current_time - last_frame_time
            last_frame_time = current_time

            # Handle JSON metadata message (bbox + liveness score from upstream)
            if "text" in data:
                try:
                    pending_metadata = json.loads(data["text"])
                except json.JSONDecodeError:
                    pass
                continue

            # Handle binary frame message
            if "bytes" not in data:
                continue

            raw_bytes: bytes = data["bytes"]
            frame_counter += 1

            # Extract metadata from pending header (if any)
            liveness_score = float(pending_metadata.get("liveness_score", 1.0))
            bbox = pending_metadata.get("bbox", None)
            if bbox:
                bbox = tuple(bbox)
            pending_metadata = {}   # Consume after use

            # Run through the detector
            # This is blocking — run in executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            window_result: Optional[WindowResult] = await loop.run_in_executor(
                None,
                lambda: detector.process_frame(
                    raw_bytes=raw_bytes,
                    liveness_score=liveness_score,
                    bbox=bbox,
                    capture_latency_ms=capture_latency_ms,
                ),
            )

            # Build response
            if window_result is not None:
                verdict = window_result.verdict
                smoothed = window_result.smoothed_score
                conf = window_result.confidence
                lat = window_result.inference_latency_ms
                v_score = window_result.visual_score
                a_score = window_result.audio_score
                m_score = window_result.metadata_score
            else:
                verdict = Verdict.PENDING
                smoothed = detector.current_score
                conf = 0.0
                lat = 0.0
                v_score = 0.0
                a_score = 0.5
                m_score = 0.5

            response = DeepfakeFrameResult(
                session_id=session_id,
                frame_index=frame_counter,
                verdict=verdict.value if hasattr(verdict, "value") else str(verdict),
                smoothed_score=round(smoothed, 4),
                confidence=round(conf, 4),
                visual_score=round(v_score, 4),
                audio_score=round(a_score, 4),
                metadata_score=round(m_score, 4),
                inference_latency_ms=round(lat, 2),
                buffer_fill=detector.buffer.buffer_fill,
                timestamp_ms=current_time,
            )

            await websocket.send_json(response.model_dump())

            # Hard stop: once DEEPFAKE is confirmed, notify and close
            if verdict == Verdict.DEEPFAKE and window_result is not None:
                await websocket.send_json({
                    "event": "SESSION_REJECTED",
                    "reason": "DEEPFAKE_DETECTED",
                    "session_id": session_id,
                    "score": round(smoothed, 4),
                    "confidence": round(conf, 4),
                })
                await websocket.close(code=1008)  # Policy violation
                break

    except WebSocketDisconnect:
        pass
    finally:
        # Don't reset here — keep session state for audit log retrieval
        pass