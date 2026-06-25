"""
FED-MEMF Deepfake Layer — Temporal Buffer & Aggregation

The deepfake detector doesn't run per-frame like liveness — it runs on 32-frame windows.
This matches Slide 15: "Analyzes 32-frame clips for involuntary neuromotor micro-expressions."

Buffer design:
  - Circular buffer of size 32 (face crops in RGB uint8)
  - When full (or half-full, for early-warning), triggers inference
  - Sliding window with overlap=8 frames to maintain temporal continuity
  - Per-session EMA smoothing of synthetic scores
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from config import TemporalConfig


class Verdict(str, Enum):
    REAL      = "REAL"
    DEEPFAKE  = "DEEPFAKE"
    UNCERTAIN = "UNCERTAIN"   # Score in gray zone → human review queue
    PENDING   = "PENDING"     # Not enough frames yet


@dataclass
class FrameRecord:
    """A single frame with its extracted face crop and metadata."""
    frame_index: int           # Global frame counter (monotonic)
    timestamp_ms: float        # Wall-clock timestamp
    face_crop: np.ndarray      # RGB uint8, variable size (pre-resize)
    bbox: Optional[tuple]      # (x1, y1, x2, y2) in pixels, or None
    liveness_score: float      # Score from the upstream liveness layer (0–1)
    # Frame-rate telemetry (for metadata LSTM stub; populated by WebSocket handler)
    capture_latency_ms: float = 0.0


@dataclass
class WindowResult:
    """Inference result for one 32-frame window."""
    window_id: int
    frame_indices: list[int]        # Which frames were in this window
    raw_synthetic_score: float      # Mean P(FAKE) over the window
    smoothed_score: float           # EMA-smoothed score
    verdict: Verdict
    confidence: float               # |smoothed_score - 0.5| * 2 → [0, 1]
    inference_latency_ms: float
    # Per-stream scores (for future cross-modal fusion logging)
    visual_score: float = 0.0
    audio_score: float = 0.5        # Neutral stub
    metadata_score: float = 0.5     # Neutral stub
    timestamp_ms: float = field(default_factory=lambda: time.time() * 1000)


class FrameBuffer:
    """
    Thread-safe circular buffer for face crops.

    Triggers inference when:
    1. Buffer reaches FULL capacity (buffer_size frames)
    2. Buffer reaches HALF capacity AND liveness has already passed
       (early deepfake warning before the full 32 frames)

    After inference, slides the window forward by (buffer_size - overlap_frames),
    retaining the overlap for temporal continuity.
    """

    def __init__(self, config: TemporalConfig):
        self.cfg = config
        self._buffer: deque[FrameRecord] = deque(maxlen=config.buffer_size)
        self._frame_counter: int = 0
        self._window_counter: int = 0

        # EMA state
        self._smoothed_score: float = 0.5   # Start neutral
        self._has_prior_window: bool = False

        # Session-level verdict (updated after each window)
        self._session_verdict: Verdict = Verdict.PENDING
        self._results_history: list[WindowResult] = []

    def push(self, record: FrameRecord) -> bool:
        """
        Add a frame to the buffer.
        Returns True if the buffer is ready for inference.
        """
        record.frame_index = self._frame_counter
        self._frame_counter += 1
        self._buffer.append(record)

        # Trigger: full buffer
        if len(self._buffer) == self.cfg.buffer_size:
            return True

        # Trigger: half-full for early warning (only after first frame)
        if (len(self._buffer) == self.cfg.min_frames_for_verdict
                and not self._has_prior_window):
            return True

        return False

    def get_crops_for_inference(self) -> tuple[list[np.ndarray], list[int]]:
        """
        Returns the face crops and their frame indices for the current window.
        Pops the oldest (non-overlap) frames after consumption.
        """
        records = list(self._buffer)
        crops = [r.face_crop for r in records]
        indices = [r.frame_index for r in records]
        return crops, indices

    def slide_window(self) -> None:
        """
        Slide the buffer forward after inference.
        Removes oldest (buffer_size - overlap) frames, retaining the tail.
        """
        n_to_remove = self.cfg.buffer_size - self.cfg.overlap_frames
        for _ in range(min(n_to_remove, len(self._buffer))):
            self._buffer.popleft()

    def update_score(
        self,
        raw_score: float,
        frame_indices: list[int],
        fake_threshold: float,
        uncertain_low: float,
        inference_latency_ms: float,
    ) -> WindowResult:
        """
        Compute EMA-smoothed score, determine verdict, record result.

        Args:
            raw_score:           Mean P(FAKE) from model on this window
            frame_indices:       Frame IDs in this window
            fake_threshold:      Above this → DEEPFAKE
            uncertain_low:       Above this but below fake_threshold → UNCERTAIN
            inference_latency_ms: Time taken for this window's inference

        Returns:
            WindowResult with verdict
        """
        alpha = self.cfg.ema_alpha

        if not self._has_prior_window:
            # First window: no prior EMA state
            smoothed = raw_score
            self._has_prior_window = True
        else:
            smoothed = alpha * raw_score + (1 - alpha) * self._smoothed_score

        self._smoothed_score = smoothed

        # Verdict logic
        if smoothed >= fake_threshold:
            verdict = Verdict.DEEPFAKE
        elif smoothed >= uncertain_low:
            verdict = Verdict.UNCERTAIN
        else:
            verdict = Verdict.REAL

        # Confidence = distance from decision boundary (0.5), normalized
        confidence = min(abs(smoothed - 0.5) * 2.0, 1.0)

        result = WindowResult(
            window_id=self._window_counter,
            frame_indices=frame_indices,
            raw_synthetic_score=float(raw_score),
            smoothed_score=float(smoothed),
            verdict=verdict,
            confidence=float(confidence),
            inference_latency_ms=inference_latency_ms,
            visual_score=float(raw_score),
        )
        self._window_counter += 1
        self._results_history.append(result)
        self._session_verdict = verdict
        return result

    @property
    def session_verdict(self) -> Verdict:
        return self._session_verdict

    @property
    def current_score(self) -> float:
        return self._smoothed_score

    @property
    def frame_count(self) -> int:
        return self._frame_counter

    @property
    def buffer_fill(self) -> int:
        return len(self._buffer)

    def reset(self) -> None:
        """Reset for a new verification session."""
        self._buffer.clear()
        self._frame_counter = 0
        self._window_counter = 0
        self._smoothed_score = 0.5
        self._has_prior_window = False
        self._session_verdict = Verdict.PENDING
        self._results_history = []

    def frame_rate_stats(self) -> dict:
        """Compute per-session frame-rate statistics (telemetry for metadata stub)."""
        records = list(self._buffer)
        if len(records) < 2:
            return {"fps": 0.0, "variance": 0.0, "latency_mean": 0.0}

        timestamps = [r.timestamp_ms for r in records]
        deltas = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps) - 1)]
        mean_delta = np.mean(deltas)
        fps = 1000.0 / mean_delta if mean_delta > 0 else 0.0
        variance = float(np.var(deltas))
        latencies = [r.capture_latency_ms for r in records]

        return {
            "fps": float(fps),
            "delta_ms_mean": float(mean_delta),
            "delta_ms_variance": variance,    # High variance → frame injection artifact
            "latency_mean_ms": float(np.mean(latencies)),
        }