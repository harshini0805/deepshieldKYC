# train.py - FED-MEMF deepfake detector fine-tuning
# Standalone: downloads data via kagglehub, trains, saves weights.
# Run: python train.py

import os
import sys
import time
import json
import argparse
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# -- Config --------------------------------------------------------------------
BACKBONE     = "tf_efficientnet_b4_ns"
INPUT_SIZE   = 380
BATCH_SIZE   = 32    # lower to 16 if CUDA OOM
EPOCHS       = 5
LR           = 2e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 0     # keep 0 on Windows

WEIGHTS_DIR  = _HERE / "weights"
OUTPUT_PATH  = WEIGHTS_DIR / "deepfake_finetuned.pth"

# -- Locate dataset ------------------------------------------------------------

def find_train_val(root: Path):
    """Walk root to find a folder containing both train/ and valid/ subfolders."""
    if (root / "train").exists() and (root / "valid").exists():
        return root / "train", root / "valid"
    for sub in sorted(root.rglob("train")):
        if (sub.parent / "valid").exists():
            return sub, sub.parent / "valid"
    return None, None


def get_data_dirs():
    # 1. Env var override
    env = os.environ.get("DEEPFAKE_DATA_ROOT")
    if env:
        t, v = find_train_val(Path(env))
        if t:
            return t, v

    # 2. kagglehub (auto-downloads if not cached, instant if already cached)
    try:
        import kagglehub
        print("  Locating dataset via kagglehub...")
        hub_path = Path(kagglehub.dataset_download("xhlulu/140k-real-and-fake-faces"))
        print(f"  kagglehub path: {hub_path}")
        t, v = find_train_val(hub_path)
        if t:
            return t, v
    except Exception as e:
        print(f"  kagglehub failed: {e}")

    # 3. Local data/ folder - search the whole tree so any layout works,
    #    including the kagglehub cache form
    #    (data/140k-real-and-fake-faces/versions/N/real_vs_fake/real-vs-fake/).
    t, v = find_train_val(_HERE / "data")
    if t:
        return t, v

    return None, None

# -- Imports -------------------------------------------------------------------
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.1)], p=0.4),
    transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 1.5))], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

val_tf = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

# -- Model ---------------------------------------------------------------------

def build_model(device, max_retries: int = 5):
    """
    Build the backbone. The pretrained weights are fetched from HuggingFace on
    first use; flaky connections can drop mid-download (IncompleteRead /
    ChunkedEncodingError). HF resumes from cache, so we just retry with backoff.
    """
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            model = timm.create_model(BACKBONE, pretrained=True, num_classes=1)
            return model.to(device)
        except Exception as e:  # network errors surface as various request exceptions
            last_err = e
            wait = min(2 ** attempt, 30)
            print(f"  [build_model] download attempt {attempt}/{max_retries} failed: "
                  f"{type(e).__name__}. Resuming in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(
        f"Failed to download backbone weights after {max_retries} attempts. "
        f"Last error: {last_err}\n"
        "  Fixes: (1) rerun (HF resumes from cache); "
        "(2) faster downloader: pip install hf_transfer then set "
        "HF_HUB_ENABLE_HF_TRANSFER=1; "
        "(3) pre-fetch once with prefetch_weights.py."
    )

# -- Training helpers ----------------------------------------------------------

def run_epoch(model, loader, optimizer, criterion, device, scaler, train):
    model.train() if train else model.eval()
    total_loss = correct = total = 0

    ctx = torch.no_grad() if not train else torch.enable_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(device)
            # ImageFolder sorts alphabetically: fake=0, real=1 -> invert so fake=1
            labels = (1 - labels).float().to(device)

            if train and scaler:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images).squeeze(-1)
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                logits = model(images).squeeze(-1)
                loss   = criterion(logits, labels)
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            total_loss += loss.item() * images.size(0)
            preds  = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total  += images.size(0)

    return total_loss / total, correct / total * 100


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


def val_with_auc(model, loader, criterion, device):
    model.eval()
    all_s, all_l = [], []
    total_loss = correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels_f = (1 - labels).float().to(device)
            logits = model(images).squeeze(-1)
            loss   = criterion(logits, labels_f)
            total_loss += loss.item() * images.size(0)
            scores = torch.sigmoid(logits)
            preds  = (scores > 0.5).float()
            correct += (preds == labels_f).sum().item()
            total   += images.size(0)
            all_s.extend(scores.cpu().tolist())
            all_l.extend(labels_f.cpu().tolist())
    return total_loss / total, correct / total * 100, roc_auc(all_l, all_s)

# -- Main ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="FED-MEMF deepfake detector fine-tuning")
    ap.add_argument("--smoke", action="store_true",
                    help="Quick sanity run: 1 epoch on a tiny random subset. "
                         "Verifies data path, model build, AMP, and checkpoint save "
                         "without the full multi-hour run. Safe on CPU.")
    ap.add_argument("--limit", type=int, default=128,
                    help="Images per split when --smoke is set (default 128).")
    args = ap.parse_args()

    epochs = 1 if args.smoke else EPOCHS

    print("=" * 60)
    print("  FED-MEMF Deepfake Detector  --  Fine-tuning"
          + ("  [SMOKE TEST]" if args.smoke else ""))
    print("=" * 60)

    print("\n  Locating training data...")
    train_dir, val_dir = get_data_dirs()

    if train_dir is None:
        print("\nERROR: Could not locate the dataset. Options:\n")
        print("  Option A (recommended) - kagglehub:")
        print("    pip install kagglehub")
        print("    python -c \"import kagglehub; kagglehub.dataset_download('xhlulu/140k-real-and-fake-faces')\"")
        print("    python train.py\n")
        print("  Option B - set path explicitly:")
        print("    In PowerShell before running train.py:")
        print("    $env:DEEPFAKE_DATA_ROOT = 'C:/path/to/real-vs-fake'")
        print("    python train.py\n")
        sys.exit(1)

    print(f"  train : {train_dir}")
    print(f"  val   : {val_dir}")

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"\n  Device  : {device}  (fp16 AMP: {use_amp})")
    print(f"  Backbone: {BACKBONE}")
    print(f"  Epochs  : {epochs}   Batch: {BATCH_SIZE}   LR: {LR}")

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds   = datasets.ImageFolder(val_dir,   transform=val_tf)

    # Fail loud and early if the folders resolved but contain no images.
    if len(train_ds) == 0 or len(val_ds) == 0:
        print("\nERROR: dataset folders resolved but are empty.")
        print(f"  train ({len(train_ds)} imgs): {train_dir}")
        print(f"  val   ({len(val_ds)} imgs): {val_dir}")
        print("  Expected real/ and fake/ subfolders of .jpg images under each.")
        sys.exit(1)
    if set(train_ds.classes) != {"real", "fake"}:
        print(f"\nWARNING: expected classes ['fake','real'], got {train_ds.classes}.")
        print("  Label inversion assumes fake sorts before real; verify polarity.")

    if args.smoke:
        import random
        random.seed(0)
        tr_idx = random.sample(range(len(train_ds)), min(args.limit, len(train_ds)))
        vl_idx = random.sample(range(len(val_ds)),   min(args.limit, len(val_ds)))
        train_ds = torch.utils.data.Subset(train_ds, tr_idx)
        val_ds   = torch.utils.data.Subset(val_ds, vl_idx)

    print(f"\n  Train: {len(train_ds):,} images  |  Val: {len(val_ds):,} images")
    classes = train_ds.dataset.classes if args.smoke else train_ds.classes
    print(f"  Classes: {classes}  (fake=1 after label inversion)")

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=use_amp)
    val_loader   = DataLoader(val_ds, BATCH_SIZE * 2, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=use_amp)

    print(f"\n  Loading {BACKBONE}...")
    model     = build_model(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=LR * 0.01)
    criterion = nn.BCEWithLogitsLoss()
    # torch.cuda.amp.GradScaler is deprecated in torch>=2.4; prefer torch.amp.
    if use_amp:
        try:
            scaler = torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    WEIGHTS_DIR.mkdir(exist_ok=True)
    best_auc = 0.0
    history  = []

    print("\n" + "-" * 60)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer,
                                    criterion, device, scaler, train=True)
        vl_loss, vl_acc, auc = val_with_auc(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{epochs}  "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.1f}%  "
              f"val_loss={vl_loss:.4f} val_acc={vl_acc:.1f}%  "
              f"AUC={auc:.4f}  ({elapsed:.0f}s)")

        history.append({"epoch": epoch,
                        "train": {"loss": tr_loss, "acc": tr_acc},
                        "val":   {"loss": vl_loss, "acc": vl_acc, "auc": auc}})

        if auc > best_auc:
            best_auc = auc
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_auc": auc, "val_acc": vl_acc,
                        "backbone": BACKBONE, "input_size": INPUT_SIZE},
                       OUTPUT_PATH)
            print(f"  -> saved best checkpoint  AUC={best_auc:.4f}  -> {OUTPUT_PATH}")

    (WEIGHTS_DIR / "train_history.json").write_text(json.dumps(history, indent=2))

    print("\n" + "=" * 60)
    print(f"  Done.  Best AUC: {best_auc:.4f}")
    print(f"  Weights: {OUTPUT_PATH}")
    print("\n  Restart the server to load fine-tuned weights:")
    print("    python main.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
