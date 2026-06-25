"""
run_test.py — Smoke test for the deepfake detection layer
Run with: python run_test.py
"""

# ─── Path setup: must come before any local imports ──────────────────────────
import os
import sys

# Ensure the directory containing THIS file is always on sys.path,
# regardless of where Python was launched from.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ─── Diagnostic: show Python what it sees ────────────────────────────────────
def check_file_structure():
    """Print the directory layout and flag any missing files."""
    required = [
        "config.py",
        "detector.py",
        "preprocessing.py",
        "temporal.py",
        "router.py",
        os.path.join("models", "backbone.py"),
        os.path.join("models", "__init__.py"),
    ]
    print(f"\n  Working directory : {_HERE}")
    print(f"  Python executable : {sys.executable}")
    print(f"  Python version    : {sys.version.split()[0]}")
    print(f"\n  File check:")

    all_ok = True
    for f in required:
        path = os.path.join(_HERE, f)
        exists = os.path.isfile(path)
        status = "✓" if exists else "✗ MISSING"
        print(f"    {status}  {f}")
        if not exists:
            all_ok = False

    if not all_ok:
        print("\n  One or more files are missing.")
        print("  Download ALL files from both file sets Claude provided,")
        print("  and ensure they are in the SAME folder as run_test.py.\n")
        sys.exit(1)

    print()  # blank line before tests
# ─────────────────────────────────────────────────────────────────────────────

import time
import numpy as np


def make_synthetic_face_frame(size: int = 480) -> bytes:
    import cv2
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[:] = (60, 60, 60)
    cx, cy = size // 2, size // 2
    cv2.ellipse(frame, (cx, cy), (80, 110), 0, 0, 360, (150, 110, 90), -1)
    cv2.circle(frame, (cx - 25, cy - 20), 8, (40, 30, 25), -1)
    cv2.circle(frame, (cx + 25, cy - 20), 8, (40, 30, 25), -1)
    cv2.ellipse(frame, (cx, cy + 30), (20, 8), 0, 0, 180, (100, 50, 50), -1)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


def test_model_loads():
    print("[1/5] Testing model load (ImageNet pretrained)...")
    from detector import DeepfakeDetector
    detector = DeepfakeDetector.build(force_cpu=True)
    print(f"      Device   : {detector.device}")
    print(f"      Backbone : {detector.config.backbone.name}")
    print("      ✓ Model loaded")
    return detector


def test_preprocessing(detector):
    print("\n[2/5] Testing face extraction...")
    from preprocessing import FaceExtractor, decode_websocket_frame

    extractor = FaceExtractor(
        config=detector.config.face_extract,
        input_size=detector.config.backbone.input_size,
    )

    raw = make_synthetic_face_frame()
    frame = decode_websocket_frame(raw)
    print(f"      Frame decoded : {frame.shape}  dtype={frame.dtype}")

    h, w = frame.shape[:2]
    bbox = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)
    crop = extractor.extract_from_bbox(frame, bbox)

    if crop is None:
        print("      ✗ Face extraction returned None")
        sys.exit(1)

    batch = extractor.preprocess_batch([crop])
    print(f"      Crop shape    : {crop.shape}")
    print(f"      Batch tensor  : {batch.shape}  (expected [1, 3, H, W])")
    print("      ✓ Preprocessing OK")
    return extractor


def test_single_frame(detector):
    print("\n[3/5] Testing single-frame inference...")
    import torch
    from preprocessing import FaceExtractor

    extractor = FaceExtractor(
        config=detector.config.face_extract,
        input_size=detector.config.backbone.input_size,
    )
    crop = np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8)
    batch = extractor.preprocess_batch([crop]).to(detector.device, dtype=detector._dtype)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits, feats = detector.model(batch)
    latency_ms = (time.perf_counter() - t0) * 1000

    score = float(torch.sigmoid(logits).item())
    print(f"      P(FAKE)      : {score:.4f}")
    print(f"      Feature dim  : {feats.shape[-1]}")
    print(f"      Latency      : {latency_ms:.1f}ms")
    print("      ✓ Single-frame inference OK")


def test_buffer_fill(detector):
    print("\n[4/5] Testing 32-frame buffer and temporal aggregation...")
    import torch
    from preprocessing import FaceExtractor
    from temporal import FrameRecord, Verdict

    detector.reset_session()
    extractor = FaceExtractor(
        config=detector.config.face_extract,
        input_size=detector.config.backbone.input_size,
    )

    buffer_size = detector.config.temporal.buffer_size
    min_frames  = detector.config.temporal.min_frames_for_verdict
    print(f"      Buffer size       : {buffer_size} frames")
    print(f"      Early-trigger at  : {min_frames} frames")

    result = None
    for i in range(buffer_size + 5):
        crop = np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8)
        record = FrameRecord(
            frame_index=-1,
            timestamp_ms=time.time() * 1000 + i * 47,
            face_crop=crop,
            bbox=(0, 0, 256, 256),
            liveness_score=0.95,
        )

        ready = detector.buffer.push(record)
        if ready:
            result = detector._run_window_inference()
            print(
                f"      Frame {i+1:02d}: window triggered → "
                f"verdict={result.verdict.value}  "
                f"score={result.smoothed_score:.4f}  "
                f"latency={result.inference_latency_ms:.1f}ms"
            )
            break

    if result is None:
        print("      ✗ Buffer never triggered")
        sys.exit(1)

    assert result.verdict in (Verdict.REAL, Verdict.DEEPFAKE, Verdict.UNCERTAIN)
    print("      ✓ Buffer + temporal aggregation OK")
    return result


def test_reset(detector):
    print("\n[5/5] Testing session reset...")
    from temporal import Verdict
    detector.reset_session()
    assert detector.session_verdict == Verdict.PENDING
    assert detector.buffer.frame_count == 0
    assert detector.buffer.buffer_fill == 0
    print("      ✓ Reset OK")


def run_all():
    print("=" * 55)
    print("  FED-MEMF Deepfake Layer — Smoke Test")
    print("=" * 55)

    check_file_structure()   # Diagnose missing files before trying imports

    t_start = time.time()

    detector = test_model_loads()
    test_preprocessing(detector)
    test_single_frame(detector)
    test_buffer_fill(detector)
    test_reset(detector)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 55}")
    print(f"  All 5 tests passed in {elapsed:.1f}s")
    print(f"  Device: {detector.device}")
    print(f"\n  Next step: python main.py")
    print(f"  Then open: http://localhost:8000/docs")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    run_all()