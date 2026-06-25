# evaluate.py - FED-MEMF Layer 1 (visual stream) honest evaluation
#
# Runs the fine-tuned EfficientNet visual detector on the HELD-OUT test/ split
# (never seen during train.py or calibrate.py) and reports the real numbers:
# accuracy, AUC, F1, precision/recall, FPR/FNR, and a confusion matrix.
#
# Why this exists: the deck/source paper advertise 98.7% acc / 0.996 AUC for the
# full federated tri-modal FED-MEMF trained on FF++/CAS(ME)2/FinVox2024. THIS is
# a single visual-stream EfficientNet-B4 on the 140k static-face set -- the paper's
# *baseline*, not the proposed method. Use the numbers this script prints, not the
# paper's, when describing what you actually built.
#
# Run after train.py:
#   python evaluate.py                 # at decision threshold 0.5
#   python evaluate.py --use-calibration   # at thresholds from weights/calibration.json
#   python evaluate.py --smoke             # quick run on a subset

import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import timm

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Reuse the EXACT eval transform + data resolution from train.py so the pipeline
# matches what produced the weights.
from train import get_data_dirs, find_train_val, val_tf, BACKBONE

WEIGHTS = _HERE / "weights" / "deepfake_finetuned.pth"
CALIB   = _HERE / "weights" / "calibration.json"


def find_test_dir():
    """Locate the held-out test/ split; fall back to valid/ if absent."""
    train_dir, val_dir = get_data_dirs()
    if val_dir is None:
        return None
    test_dir = val_dir.parent / "test"
    if test_dir.exists():
        return test_dir
    print(f"  NOTE: no test/ split found next to {val_dir.name}; using valid/ instead.")
    return val_dir


def collect(model, loader, device):
    """Return (labels, scores) with fake=1 (matches train.py label inversion)."""
    model.eval()
    labels, scores = [], []
    with torch.no_grad():
        for images, lbl in loader:
            images = images.to(device)
            lbl = (1 - lbl).float()  # ImageFolder: fake=0,real=1 -> invert -> fake=1
            s = torch.sigmoid(model(images).squeeze(-1)).cpu()
            scores.extend(s.tolist())
            labels.extend(lbl.tolist())
    return labels, scores


def roc_auc(labels, scores):
    paired = sorted(zip(scores, labels), reverse=True)
    n_pos = sum(labels); n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return 0.5
    tp = fp = auc = prev_fp = 0
    for _, lbl in paired:
        if lbl == 1:
            tp += 1
        else:
            fp += 1
            auc += tp * (fp - prev_fp)
            prev_fp = fp
    return auc / (n_pos * n_neg)


def metrics_at(labels, scores, thr):
    tp = fp = tn = fn = 0
    for l, s in zip(labels, scores):
        pred = 1 if s >= thr else 0
        if   pred == 1 and l == 1: tp += 1
        elif pred == 1 and l == 0: fp += 1
        elif pred == 0 and l == 0: tn += 1
        else:                      fn += 1
    acc  = (tp + tn) / max(1, tp + fp + tn + fn)
    prec = tp / max(1, tp + fp)
    rec  = tp / max(1, tp + fn)            # recall on FAKE = 1 - FNR
    f1   = 2 * prec * rec / max(1e-9, prec + rec)
    fpr  = fp / max(1, fp + tn)            # real wrongly flagged as fake
    fnr  = fn / max(1, fn + tp)            # fake that slipped through
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, fpr=fpr, fnr=fnr,
                tp=tp, fp=fp, tn=tn, fn=fn)


def main():
    ap = argparse.ArgumentParser(description="Honest evaluation of the Layer 1 visual detector")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Decision threshold for FAKE (default 0.5).")
    ap.add_argument("--use-calibration", action="store_true",
                    help="Use fake_threshold from weights/calibration.json if present.")
    ap.add_argument("--smoke", action="store_true", help="Evaluate on a small subset.")
    ap.add_argument("--limit", type=int, default=256, help="Images per class in smoke mode.")
    args = ap.parse_args()

    if not WEIGHTS.exists():
        print(f"ERROR: {WEIGHTS} not found. Run `python train.py` first.")
        sys.exit(1)

    thr = args.threshold
    if args.use_calibration and CALIB.exists():
        thr = json.loads(CALIB.read_text()).get("fake_threshold", thr)
        print(f"  Using calibrated fake_threshold = {thr:.2f}")

    test_dir = find_test_dir()
    if test_dir is None:
        print("ERROR: could not locate a test/ or valid/ split.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(WEIGHTS, map_location=device)
    # Mirror training: a bare timm model with a single logit head. The deployed
    # DeepfakeBackbone uses the same backbone+classifier, so logits are identical.
    model = timm.create_model(BACKBONE, pretrained=False, num_classes=1).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    print(f"  Loaded {WEIGHTS.name}  (reported train-time val AUC={ckpt.get('val_auc','?')})")

    ds = datasets.ImageFolder(test_dir, transform=val_tf)
    if args.smoke:
        import random
        random.seed(0)
        idx = random.sample(range(len(ds)), min(args.limit * 2, len(ds)))
        ds = torch.utils.data.Subset(ds, idx)
    loader = DataLoader(ds, 64, shuffle=False, num_workers=0)
    print(f"  Evaluating on {len(ds):,} images from: {test_dir}")

    labels, scores = collect(model, loader, device)
    auc = roc_auc(labels, scores)
    m = metrics_at(labels, scores, thr)

    print("\n" + "=" * 60)
    print(f"  LAYER 1 (visual EfficientNet) -- held-out results @ thr={thr:.2f}")
    print("=" * 60)
    print(f"  Accuracy : {m['acc']*100:6.2f}%")
    print(f"  AUC      : {auc:6.4f}")
    print(f"  F1       : {m['f1']*100:6.2f}%   Precision {m['prec']*100:5.2f}%   Recall {m['rec']*100:5.2f}%")
    print(f"  FPR      : {m['fpr']*100:6.2f}%  (real wrongly flagged as fake)")
    print(f"  FNR      : {m['fnr']*100:6.2f}%  (fake that slipped through)")
    print(f"\n  Confusion matrix (fake=positive):")
    print(f"                 pred FAKE   pred REAL")
    print(f"    true FAKE      {m['tp']:7d}    {m['fn']:7d}")
    print(f"    true REAL      {m['fp']:7d}    {m['tn']:7d}")

    print("\n  --- Honest framing vs. source paper (Rawat et al.) ---")
    print("  Paper FED-MEMF (full federated tri-modal): 98.7% acc / 0.996 AUC")
    print("  Paper Table 6 EfficientNet-B4 BASELINE   : 92.78% acc / 0.933 AUC")
    print("  ^ This script measures YOUR equivalent of that baseline. Report the")
    print("    numbers above as the visual stream's performance, not the paper's.")
    print("=" * 60)

    out = _HERE / "weights" / "eval_test.json"
    out.write_text(json.dumps({
        "threshold": thr, "n_images": len(ds), "auc": auc, **m,
        "split": str(test_dir),
    }, indent=2))
    print(f"\n  Written to {out}")


if __name__ == "__main__":
    main()
