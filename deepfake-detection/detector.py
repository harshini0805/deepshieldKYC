"""
FED-MEMF Deepfake Layer — Core Detector

DeepfakeDetector orchestrates:
  1. Face extraction (preprocessing.py)
  2. Model inference (models/backbone.py)
  3. Temporal aggregation (temporal.py)
  4. Stub fusion for audio and metadata streams

This is Stream 1 (visual) with neutral stubs for Streams 2 & 3.
Once Audio CNN and Metadata LSTM are built, replace the stub calls.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from config import DeepfakeConfig, get_production_config, get_cpu_fallback_config
from models.backbone import DeepfakeBackbone
from preprocessing import FaceExtractor, decode_websocket_frame
from temporal import FrameBuffer, FrameRecord, Verdict, WindowResult


class DeepfakeDetector:
    """
    Main entry point for the deepfake detection layer.

    Typical integration with the existing FastAPI/WebSocket pipeline:

        detector = DeepfakeDetector.build()

        # Per-frame call (inside WebSocket message handler):
        result = detector.process_frame(raw_bytes, liveness_score=0.92, bbox=bbox)

        if result is not None:
            # A 32-frame window has been analyzed → emit verdict
            await websocket.send_json(result.to_dict())
    """

    def __init__(self, config: DeepfakeConfig):
        self.config = config
        self.device = self._resolve_device(config.inference.device)

        # Build and configure model
        self.model = DeepfakeBackbone.build(
            name=config.backbone.name,
            pretrained=True,  # ImageNet weights as base
            weights_path=(
                Path(config.backbone.weights_path)
                if Path(config.backbone.weights_path).exists()
                else None
            ),
            feature_dim=256,
        ).to(self.device)

        self.model.eval()

        if config.inference.fp16 and self.device.type == "cuda":
            self.model = self.model.half()
            self._dtype = torch.float16
        else:
            self._dtype = torch.float32

        # Face extractor
        self.extractor = FaceExtractor(
            config=config.face_extract,
            input_size=config.backbone.input_size,
        )

        # Per-session frame buffer
        self.buffer = FrameBuffer(config.temporal)

        print(
            f"[DeepfakeDetector] Initialized | "
            f"backbone={config.backbone.name} | "
            f"device={self.device} | "
            f"fp16={config.inference.fp16}"
        )

    @classmethod
    def build(
        cls,
        weights_path: Optional[str] = None,
        force_cpu: bool = False,
    ) -> "DeepfakeDetector":
        """
        Convenience factory. Auto-selects GPU config if CUDA available.

        Args:
            weights_path: Path to FF++ pretrained .pth file. None = ImageNet weights only.
            force_cpu:    Force CPU mode (for testing without GPU).

        Example:
            # Production (GPU):
            detector = DeepfakeDetector.build(weights_path="weights/ff_efficientnet_b4.pth")

            # Testing (CPU, ImageNet weights):
            detector = DeepfakeDetector.build(force_cpu=True)
        """
        if force_cpu or not torch.cuda.is_available():
            cfg = get_cpu_fallback_config()
        else:
            cfg = get_production_config()

        if weights_path:
            cfg.backbone.weights_path = weights_path

        return cls(cfg)

    # ─── Primary Public API ───────────────────────────────────────────────────

    def process_frame(
        self,
        raw_bytes: bytes,
        liveness_score: float = 1.0,
        bbox: Optional[tuple] = None,
        capture_latency_ms: float = 0.0,
    ) -> Optional[WindowResult]:
        """
        Process one raw WebSocket frame. Returns a WindowResult when a
        32-frame window is complete; returns None otherwise.

        Args:
            raw_bytes:           JPEG/PNG bytes from React WebSocket
            liveness_score:      P(LIVE) from upstream liveness layer [0, 1]
            bbox:                Face bounding box (x1, y1, x2, y2) from liveness.
                                 If None, runs MediaPipe internally.
            capture_latency_ms:  Time since previous frame (for frame-rate telemetry)

        Returns:
            WindowResult if inference was triggered, else None
        """
        timestamp_ms = time.time() * 1000

        # Decode frame
        frame_bgr = decode_websocket_frame(raw_bytes)

        # Extract face crop
        if bbox is not None:
            face_crop = self.extractor.extract_from_bbox(frame_bgr, bbox)
        else:
            face_crop, bbox = self.extractor.extract_from_frame(frame_bgr)

        if face_crop is None:
            # No face detected — skip frame (don't push to buffer)
            return None

        # Build and push frame record
        record = FrameRecord(
            frame_index=-1,           # Will be assigned by buffer
            timestamp_ms=timestamp_ms,
            face_crop=face_crop,
            bbox=bbox,
            liveness_score=liveness_score,
            capture_latency_ms=capture_latency_ms,
        )

        ready = self.buffer.push(record)

        if not ready:
            return None

        return self._run_window_inference()

    def process_video_file(self, video_path: str) -> list[WindowResult]:
        """
        Offline analysis of a full video (for testing / batch evals).
        Returns list of WindowResults, one per 32-frame window.
        """
        import cv2
        cap = cv2.VideoCapture(video_path)
        results = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            face_crop, bbox = self.extractor.extract_from_frame(frame)
            if face_crop is None:
                continue

            record = FrameRecord(
                frame_index=-1,
                timestamp_ms=cap.get(cv2.CAP_PROP_POS_MSEC),
                face_crop=face_crop,
                bbox=bbox,
                liveness_score=1.0,
            )

            if self.buffer.push(record):
                result = self._run_window_inference()
                if result:
                    results.append(result)

        cap.release()
        return results

    def reset_session(self) -> None:
        """Call this when a new verification session starts (new user)."""
        self.buffer.reset()

    # ─── Inference Engine ─────────────────────────────────────────────────────

    def _run_window_inference(self) -> WindowResult:
        """
        Run inference on the current buffer window.

        Steps:
          1. Retrieve crops from buffer
          2. Batch preprocess (resize, normalize)
          3. GPU/CPU inference in sub-batches
          4. Mean pool per-frame P(FAKE) → window score
          5. Fuse with audio/metadata stubs
          6. Update EMA and compute verdict
          7. Slide buffer window
        """
        t_start = time.perf_counter()

        crops, frame_indices = self.buffer.get_crops_for_inference()

        # Preprocess: (N, 3, H, W) float tensor
        tensor_batch = self.extractor.preprocess_batch(crops, augment=False)
        tensor_batch = tensor_batch.to(self.device, dtype=self._dtype)

        # Sub-batch inference
        per_frame_scores = self._batched_inference(tensor_batch)

        # Mean P(FAKE) over the window (visual stream score)
        visual_score = float(per_frame_scores.mean().item())

        # Weighted fusion with stub streams
        fused_score = self._fuse_streams(
            visual_score=visual_score,
            audio_score=self.config.audio.neutral_score,
            metadata_score=self.config.metadata.neutral_score,
        )

        t_end = time.perf_counter()
        inference_ms = (t_end - t_start) * 1000

        result = self.buffer.update_score(
            raw_score=fused_score,
            frame_indices=frame_indices,
            fake_threshold=self.config.inference.fake_threshold,
            uncertain_low=self.config.inference.uncertain_low,
            inference_latency_ms=inference_ms,
        )
        result.visual_score = visual_score

        self.buffer.slide_window()

        return result

    def _batched_inference(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Run inference in chunks of batch_size to avoid OOM on large windows.
        Returns per-frame sigmoid scores (N,) on CPU.
        """
        batch_size = self.config.temporal.batch_size
        n = tensor.shape[0]
        all_scores = []

        with torch.no_grad():
            for start in range(0, n, batch_size):
                chunk = tensor[start : start + batch_size]
                logits, _ = self.model(chunk)        # (B,), (B, 256)
                scores = torch.sigmoid(logits)       # P(FAKE) ∈ [0, 1]
                all_scores.append(scores.float().cpu())

        return torch.cat(all_scores, dim=0)

    def _fuse_streams(
        self,
        visual_score: float,
        audio_score: float,
        metadata_score: float,
    ) -> float:
        """
        Weighted fusion of three stream scores.
        With stubs disabled, visual_weight=1.0 → fused = visual_score.

        When Audio CNN and Metadata LSTM are integrated:
          visual_weight:   0.4  (micro-expressions)
          audio_weight:    0.4  (voice clone artifacts)
          metadata_weight: 0.2  (behavioral telemetry)
        """
        w_v = self.config.fusion.visual_weight
        w_a = self.config.fusion.audio_weight
        w_m = self.config.fusion.metadata_weight

        total_weight = w_v + w_a + w_m
        if total_weight == 0:
            return visual_score

        fused = (
            w_v * visual_score
            + w_a * audio_score
            + w_m * metadata_score
        ) / total_weight

        return float(np.clip(fused, 0.0, 1.0))

    # ─── Utilities ────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)

    @property
    def session_verdict(self) -> Verdict:
        return self.buffer.session_verdict

    @property
    def current_score(self) -> float:
        return self.buffer.current_score

    @property
    def frames_seen(self) -> int:
        return self.buffer.frame_count