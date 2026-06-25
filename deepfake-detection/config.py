"""
FED-MEMF Deepfake Detection Layer — Configuration
Pipeline position: USER INPUT → CAPTURE → LIVENESS CHECK → [DEEPFAKE DETECTION] → FACE VERIFY → DECISION
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class BackboneConfig:
    """Model backbone configuration."""

    # tf_efficientnet_b4_ns = Noisy Student EfficientNet-B4
    # Trained on 300M pseudo-labeled images — much stronger transfer learning
    # baseline than plain efficientnet_b4. Used by top DFDC solutions.
    # Downloads automatically from timm hub (HuggingFace) on first run.
    name: Literal[
        "tf_efficientnet_b4_ns",
        "efficientnet_b4",
        "xception",
        "vit_base_patch16_224",
    ] = "tf_efficientnet_b4_ns"

    # Path to fine-tuned weights produced by finetune.py.
    # If this file doesn't exist, timm pretrained weights are used.
    weights_path: Path = Path("weights/deepfake_finetuned.pth")

    # Input spatial resolution — tf_efficientnet_b4_ns expects 380×380
    input_size: int = 380

    num_classes: int = 2  # Binary: REAL(0) vs FAKE(1)

    # Feature dimension after GAP (used for cross-modal attention stub)
    feature_dim: int = 1792  # EfficientNet-B4 final conv channels


@dataclass
class TemporalConfig:
    """Frame buffer and temporal aggregation settings."""
    buffer_size: int = 32        # Analyze 32-frame clips (matching Slide 15)
    batch_size: int = 8          # Inference batch size; 32 / 8 = 4 forward passes
    overlap_frames: int = 8      # Overlap between consecutive windows (sliding)

    # Temporal smoothing: exponential moving average of per-window scores
    # score_t = alpha * raw_score + (1 - alpha) * score_{t-1}
    ema_alpha: float = 0.6

    # Minimum frames required before issuing any verdict
    min_frames_for_verdict: int = 16

    # Frame-rate stabilization (must match liveness layer ~21 FPS)
    expected_fps: float = 21.0


@dataclass
class FaceExtractConfig:
    """MediaPipe-based face extraction settings."""
    margin: float = 0.3
    min_detection_confidence: float = 0.7
    min_face_fraction: float = 0.03
    crop_size: int = 256


@dataclass
class InferenceConfig:
    """Runtime inference settings."""
    device: Literal["cuda", "cpu", "auto"] = "auto"
    fp16: bool = True

    # Score above this → DEEPFAKE verdict
    fake_threshold: float = 0.60

    # Score in [uncertain_low, fake_threshold) → UNCERTAIN → human review queue
    uncertain_low: float = 0.35

    latency_budget_ms: float = 45.0


@dataclass
class AudioStubConfig:
    """Placeholder for Stream 2: Audio CNN (not built yet)."""
    enabled: bool = False
    neutral_score: float = 0.5


@dataclass
class MetadataStubConfig:
    """Placeholder for Stream 3: Metadata LSTM (not built yet)."""
    enabled: bool = False
    fields: list = field(default_factory=lambda: [
        "keystroke_latency_ms",
        "frame_rate_variance",
        "gaze_shift_velocity",
    ])
    neutral_score: float = 0.5


@dataclass
class FusionConfig:
    """
    Cross-modal fusion weights.
    With audio/metadata stubs disabled, visual stream carries full weight.
    """
    visual_weight: float = 1.0
    audio_weight: float = 0.0
    metadata_weight: float = 0.0


@dataclass
class DeepfakeConfig:
    """Root configuration — pass this to DeepfakeDetector."""
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    face_extract: FaceExtractConfig = field(default_factory=FaceExtractConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    audio: AudioStubConfig = field(default_factory=AudioStubConfig)
    metadata: MetadataStubConfig = field(default_factory=MetadataStubConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)


# ─── Preset Configs ──────────────────────────────────────────────────────────

def get_production_config() -> DeepfakeConfig:
    """High-accuracy, GPU config (target: 45ms window latency)."""
    cfg = DeepfakeConfig()
    cfg.inference.device = "cuda"
    cfg.inference.fp16 = True
    return cfg


def get_cpu_fallback_config() -> DeepfakeConfig:
    """CPU-compatible config for testing or machines without GPU."""
    cfg = DeepfakeConfig()
    cfg.backbone.name = "tf_efficientnet_b4_ns"
    cfg.backbone.input_size = 224   # Smaller input → faster CPU inference
    cfg.temporal.buffer_size = 16
    cfg.temporal.batch_size = 4
    cfg.inference.device = "cpu"
    cfg.inference.fp16 = False
    cfg.inference.latency_budget_ms = 400.0
    return cfg