"""
FED-MEMF Stream 1: Visual Backbone
Wraps timm models with a consistent interface + feature extraction for cross-modal attention.

Architecture correspondence (from slides):
  Slide 6:  "μ-Transformer: extracts 7×7 patch embeddings for micro-expressions"
  Slide 15: "32-frame clips for involuntary neuromotor micro-expressions"

TODAY's implementation:
  EfficientNet-B4 pretrained on FaceForensics++ — processes full face crops.
  Outputs both a binary deepfake logit AND a feature vector for future
  cross-modal attention fusion.

NEXT STEP (true μ-Transformer):
  Replace backbone with DeiT-S or ViT-B/16, fine-tuned on CASME II + SAMM
  micro-expression datasets. Use attention map visualization to confirm the
  model attends to naso-labial folds and periorbital regions.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import timm


class _FeatureExtractHead(nn.Module):
    """
    Adds a parallel feature projection head to the backbone.
    The primary head outputs logits. This head outputs a d-dim feature vector
    for use in the cross-modal attention layer (when audio/metadata are added).
    """

    def __init__(self, in_dim: int, feature_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DeepfakeBackbone(nn.Module):
    """
    Wrapper around a timm backbone for binary deepfake classification.

    Forward pass returns:
        logits:   (B,) — raw pre-sigmoid score; positive = FAKE
        features: (B, feature_dim) — for cross-modal attention fusion

    Usage:
        model = DeepfakeBackbone.build("efficientnet_b4", feature_dim=256)
        logits, feats = model(face_crops)  # face_crops: (B, 3, H, W)
        fake_probs = torch.sigmoid(logits)
    """

    def __init__(self, backbone: nn.Module, in_features: int, feature_dim: int):
        super().__init__()
        self.backbone = backbone
        self.feature_head = _FeatureExtractHead(in_features, feature_dim)

    @classmethod
    def build(
        cls,
        name: str = "efficientnet_b4",
        pretrained: bool = True,
        weights_path: Optional[Path] = None,
        feature_dim: int = 256,
    ) -> "DeepfakeBackbone":
        """
        Factory method. Loads backbone from timm, replaces classifier with
        a binary head, optionally loads FaceForensics++ fine-tuned weights.
        """
        # Load backbone with ImageNet pretrained weights (or random init)
        backbone = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=1,  # Binary: single logit, apply sigmoid for P(FAKE)
        )

        # Grab the in_features *before* we swap the head
        in_features = backbone.get_classifier().in_features

        # Set the classifier to binary
        backbone.reset_classifier(num_classes=1)

        model = cls(backbone, in_features, feature_dim)

        # Load FaceForensics++ fine-tuned weights if provided
        if weights_path is not None:
            model._load_ff_plus_weights(weights_path)

        return model

    def _load_ff_plus_weights(self, path: Path) -> None:
        """
        Load weights from FaceForensics++ pretrained checkpoint.

        The FF++ checkpoint format (from the official repo) is a dict:
          { "model_state_dict": ..., "epoch": ..., "acc": ... }
        """
        if not path.exists():
            raise FileNotFoundError(
                f"FF++ weights not found at {path}.\n"
                "Download from: https://github.com/ondyari/FaceForensics\n"
                "Or use Hugging Face: kaggle datasets download -d deepfakedetection/dfdc-efficientnet\n"
                "Then set config.backbone.weights_path to the .pth file."
            )
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Handle DataParallel-wrapped checkpoints
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # Strict=False: ignores missing keys (e.g., our new feature_head)
        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
        if unexpected:
            print(f"[DeepfakeBackbone] Unexpected keys in checkpoint: {unexpected}")
        print(f"[DeepfakeBackbone] Loaded FF++ weights from {path}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) normalized face crops

        Returns:
            logits:   (B,) — raw score; sigmoid gives P(FAKE)
            features: (B, feature_dim) — for cross-modal fusion
        """
        # Forward through backbone up to the pooled features
        features_pool = self.backbone.forward_features(x)

        # Global average pooling if backbone returns spatial features
        if features_pool.dim() == 4:
            # CNN-style: (B, C, H, W)
            features_pool = features_pool.mean(dim=[2, 3])
        elif features_pool.dim() == 3:
            # ViT-style: (B, N_tokens, C) — use CLS token
            features_pool = features_pool[:, 0, :]

        # Classification logit (B,) via backbone's head
        logits = self.backbone.get_classifier()(features_pool).squeeze(-1)

        # Feature projection for cross-modal attention
        feat_out = self.feature_head(features_pool)

        return logits, feat_out


# ─── Xception variant ────────────────────────────────────────────────────────

class XceptionDetector(nn.Module):
    """
    Xception-based detector (Rossler et al., ICCV 2019 FaceForensics++ baseline).
    Used here as an alternative backbone for comparison.

    Xception is specifically designed for texture manipulation artifacts —
    it's strong on Deepfakes and Face2Face but weaker on NeuralTextures
    compared to EfficientNet. Keep both for ensemble.
    """

    def __init__(self, feature_dim: int = 256, pretrained: bool = True):
        super().__init__()
        self.net = timm.create_model("xception", pretrained=pretrained, num_classes=1)
        in_features = self.net.get_classifier().in_features
        self.net.reset_classifier(num_classes=1)
        self.feature_proj = _FeatureExtractHead(in_features, feature_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.net.forward_features(x)
        if feats.dim() == 4:
            feats = feats.mean(dim=[2, 3])
        logits = self.net.get_classifier()(feats).squeeze(-1)
        feat_out = self.feature_proj(feats)
        return logits, feat_out