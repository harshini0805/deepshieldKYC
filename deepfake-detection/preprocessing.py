"""
FED-MEMF Deepfake Layer — Preprocessing

Responsibilities:
  1. Detect and crop face from raw webcam frames (reuses MediaPipe already in stack)
  2. Normalize to ImageNet stats (required for pretrained EfficientNet)
  3. Optional augmentation for training
  4. Batch collation for GPU inference

Key design decision: we reuse MediaPipe (already running in the liveness layer)
to avoid running two separate face detectors. In the integrated pipeline,
the bounding box from liveness can be passed directly to skip re-detection.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torchvision import transforms

from config import FaceExtractConfig


# ─── ImageNet normalization constants ────────────────────────────────────────
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


class FaceExtractor:
    """
    Extracts and preprocesses face crops from BGR frames (OpenCV format).

    Can operate in two modes:
    1. Standalone: runs MediaPipe face detection internally
    2. Passthrough: accepts pre-computed bounding boxes (from liveness layer)
    """

    def __init__(self, config: FaceExtractConfig, input_size: int = 380):
        self.cfg = config
        self.input_size = input_size

        # Lazy-import MediaPipe to avoid heavy import at module load
        self._face_detector = None

        # Preprocessing pipeline for inference (no augmentation)
        self.inference_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

        # Augmented pipeline for fine-tuning on local data
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                       saturation=0.1, hue=0.05)
            ], p=0.4),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))
            ], p=0.3),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    def _get_face_detector(self):
        """Lazy-init MediaPipe face detection."""
        if self._face_detector is None:
            import mediapipe as mp
            self._face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,  # Full-range model (works up to 5m)
                min_detection_confidence=self.cfg.min_detection_confidence,
            )
        return self._face_detector

    def extract_from_bbox(
        self,
        frame_bgr: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> np.ndarray | None:
        """
        Crop face using a pre-computed bounding box (e.g., from liveness layer).

        Args:
            frame_bgr: BGR frame from OpenCV / WebSocket frame
            bbox: (x_min, y_min, x_max, y_max) in pixel coordinates

        Returns:
            RGB face crop (np.ndarray, uint8) or None if bbox is invalid
        """
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Apply margin
        margin_x = int((x2 - x1) * self.cfg.margin)
        margin_y = int((y2 - y1) * self.cfg.margin)
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w, x2 + margin_x)
        y2 = min(h, y2 + margin_y)

        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return None

        face_crop = frame_bgr[y1:y2, x1:x2]
        return cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

    def extract_from_frame(
        self,
        frame_bgr: np.ndarray,
    ) -> tuple[np.ndarray | None, tuple | None]:
        """
        Run MediaPipe detection + crop in one call. Use when no prior bbox available.

        Returns:
            (rgb_crop, bbox) or (None, None) if no face found
        """
        detector = self._get_face_detector()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        if not results.detections:
            return None, None

        # Take the highest-confidence detection
        det = max(results.detections, key=lambda d: d.score[0])
        score = det.score[0]

        if score < self.cfg.min_detection_confidence:
            return None, None

        h, w = frame_bgr.shape[:2]
        bbox_rel = det.location_data.relative_bounding_box
        x1 = int(bbox_rel.xmin * w)
        y1 = int(bbox_rel.ymin * h)
        x2 = int((bbox_rel.xmin + bbox_rel.width) * w)
        y2 = int((bbox_rel.ymin + bbox_rel.height) * h)

        # Minimum face area filter
        face_area = (x2 - x1) * (y2 - y1)
        if face_area < (w * h * self.cfg.min_face_fraction):
            return None, None

        crop = self.extract_from_bbox(frame_bgr, (x1, y1, x2, y2))
        return crop, (x1, y1, x2, y2)

    def preprocess_batch(
        self,
        crops: list[np.ndarray],
        augment: bool = False,
    ) -> torch.Tensor:
        """
        Convert a list of RGB face crops to a normalized tensor batch.

        Args:
            crops: list of (H, W, 3) uint8 RGB arrays
            augment: use augmented transform (for fine-tuning only)

        Returns:
            (N, 3, H, W) float32 tensor on CPU
        """
        transform = self.train_transform if augment else self.inference_transform
        tensors = [transform(crop) for crop in crops]
        return torch.stack(tensors, dim=0)


def decode_websocket_frame(raw_bytes: bytes) -> np.ndarray:
    """
    Decode a JPEG frame received over WebSocket into a BGR numpy array.
    This matches the format sent by the React frontend (requestAnimationFrame → canvas.toBlob).
    """
    buf = np.frombuffer(raw_bytes, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode WebSocket frame as JPEG/PNG")
    return frame