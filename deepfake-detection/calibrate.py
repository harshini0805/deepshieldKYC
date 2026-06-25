# calibrate.py - FED-MEMF deepfake detector threshold calibration
# Run AFTER train.py produces weights/deepfake_finetuned.pth.
#
# Why this exists: config.py ships fake_threshold=0.60 / uncertain_low=0.35 as
# placeholders. With ImageNet weights every frame scores ~0.50, so those numbers
# are meaningless. After fine-tuning, the score distribution separates and the
# operating points should be set from a real ROC curve on the validation set,
# tuned to the eKYC cost model (a missed deepfake is far worse than a re-review).
#
# Run: python calibrate.py            # uses val split + saved weights
#      python calibrate.py --target-fpr 0.02

import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from train import get_data_dirs, val_tf, BACKBONE, INPUT_SIZE  # reuse exact pipeline

WEIGHTS = _HERE / "weights" / "deepfake_finetuned.pth"


def collect_scores(model, loader, device):
    """Return (labels, scores) with fake=1, matching train.py polarity."""
    model.eval()
    labels, scores = [], []
    with torch.no_grad():
        for images, lbl in loader:
            images = images.to(device)
            lbl = (1 - lbl).float()  # ImageFolder fake=0,real=1 -> invert -> fake=1
            s = torch.sigmoid(model(images).squeeze(-1)).cpu()
            scores.extend(s.tolist())
            labels.extend(lbl.tolist())
    return labels, scores


def roc_points(labels, scores):
    """Sweep thresholds; return list of (thr, tpr, fpr) sorted by threshold desc."""
    P = sum(labels)
    N = len(labels) - P
    pts = []
    for thr in [i / 100 for i in range(0, 101)]:
        tp = sum(1 for l, s in zip(labels, scores) if s >= thr and l == 1)
        fp = sum(1 for l, s in zip(labels, scores) if s >= thr and l == 0)
        tpr = tp / P if P else 0.0
        fpr = fp / N if N else 0.0
        pts.append((thr, tpr, fpr))
    return pts


def auc_from_points(pts):
    """Trapezoidal AUC over ascending FPR."""
    ordered = sorted(pts, key=lambda p: p[2])  # by fpr
    auc = 0.0
    for (_, t0, f0), (_, t1, f1) in zip(ordered, ordered[1:]):
        auc += (f1 - f0) * (t0 + t1) / 2
    return auc


def main():
    ap = argparse.ArgumentParser(description="Calibrate deepfake decision thresholds")
    ap.add_argument("--target-fpr", type=float, default=0.02,
                    help="Max acceptable false-positive rate (real flagged as fake). "
                         "fake_threshold is set to the lowest score meeting this.")
    ap.add_argument("--review-tpr", type=float, default=0.95,
                    help="Recall target for routing to UNCERTAIN/human review. "
                         "uncertain_low is set to catch this fraction of fakes.")
    args = ap.parse_args()

    if not WEIGHTS.exists():
        print(f"ERROR: {WEIGHTS} not found. Run `python train.py` first.")
        sys.exit(1)

    _, val_dir = get_data_dirs()
    if val_dir is None:
        print("ERROR: validation set not found (see train.py data resolution).")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(WEIGHTS, map_location=device)
    model = timm.create_model(BACKBONE, pretrained=False, num_classes=1).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded {WEIGHTS.name}  (train AUC={ckpt.get('val_auc', '?')})")

    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)
    loader = DataLoader(val_ds, 64, shuffle=False, num_workers=0)
    print(f"  Scoring {len(val_ds):,} validation images...")

    labels, scores = collect_scores(model, loader, device)
    pts = roc_points(labels, scores)
    auc = auc_from_points(pts)

    # fake_threshold: smallest threshold whose FPR <= target  -> high precision on FAKE
    cand = [p for p in pts if p[2] <= args.target_fpr]
    fake_thr = min((p[0] for p in cand), default=0.60)
    fake_tpr = next((t for thr, t, f in pts if thr == fake_thr), 0.0)

    # uncertain_low: threshold that still recalls review-tpr of fakes -> lower bound
    cand2 = [p for p in pts if p[1] >= args.review_tpr]
    unc_low = max((p[0] for p in cand2), default=0.35)

    print("\n" + "=" * 56)
    print(f"  Val AUC                 : {auc:.4f}")
    print(f"  fake_threshold  (FPR<={args.target_fpr:.0%}) : {fake_thr:.2f}  "
          f"(catches {fake_tpr:.0%} of fakes at this point)")
    print(f"  uncertain_low   (TPR>={args.review_tpr:.0%}) : {unc_low:.2f}")
    print("=" * 56)
    print("\n  Apply to config.py InferenceConfig:")
    print(f"    fake_threshold: float = {fake_thr:.2f}")
    print(f"    uncertain_low:  float = {unc_low:.2f}")

    out = _HERE / "weights" / "calibration.json"
    out.write_text(json.dumps({
        "val_auc": auc,
        "fake_threshold": fake_thr,
        "uncertain_low": unc_low,
        "target_fpr": args.target_fpr,
        "review_tpr": args.review_tpr,
        "roc": [{"thr": t, "tpr": tp, "fpr": fp} for t, tp, fp in pts],
    }, indent=2))
    print(f"\n  Full ROC + chosen points written to {out}")


if __name__ == "__main__":
    main()
