# Layer 1 — Visual Deepfake Detection Stream

This is **Stream 1** of the FED-MEMF deepfake-detection layer: a frame-level
EfficientNet-B4 classifier that scores each face crop as `P(FAKE)`, aggregated
over 32-frame windows with EMA smoothing and served over WebSocket.

## Honest scope

The source paper (Rawat et al., *Federated Micro-Expression Mining and Multi-Modal
Metadata Fusion*) describes a **federated, tri-modal** system (μ-Transformer +
audio CNN + metadata LSTM, fused by cross-modal attention, aggregated via FedAvg)
trained on FaceForensics++ / CAS(ME)² / a proprietary KYC-FinVox2024 set.

**What is built here is only the visual stream**, as an EfficientNet-B4 trained on
the 140k Real-and-Fake-Faces set. In the paper's own Table 6, EfficientNet-B4 is a
**baseline (92.78% acc / 0.933 AUC)** that the full method beats — so this layer is
that baseline, not the proposed FED-MEMF. Do **not** quote the paper's 98.7% / 0.996
as results for this code. Use the numbers `evaluate.py` prints on your held-out split.

The μ-Transformer was not built because it requires micro-expression *video* data
(CAS(ME)²); the 140k set is static face-swap images, which can only train a
frame-level CNN.

## Run order (on a CUDA machine — no GPU needed to read this)

```bash
# 0. dataset already at data/140k-real-and-fake-faces/versions/2/real_vs_fake/real-vs-fake/
#    (train/ valid/ test/, each with real/ and fake/)

python train.py --smoke      # ~2 min: verifies data path, model, AMP, checkpoint save
python train.py              # full 5-epoch fine-tune (~30 min GPU) -> weights/deepfake_finetuned.pth
python calibrate.py          # sets fake_threshold / uncertain_low from the val ROC
python evaluate.py --use-calibration   # honest metrics on the held-out test/ split
python main.py               # serves on :8000, auto-loads the fine-tuned weights
```

Outputs land in `weights/`: `deepfake_finetuned.pth`, `train_history.json`,
`calibration.json`, `eval_test.json`.

## Files

| File | Role |
|------|------|
| `train.py` | Fine-tune EfficientNet-B4 on the 140k set. `--smoke` for a fast sanity run. Saves best-AUC checkpoint. |
| `calibrate.py` | Pick `fake_threshold` (target FPR) and `uncertain_low` (target fake-recall) from the validation ROC. |
| `evaluate.py` | Held-out **test/** metrics: accuracy, AUC, F1, precision/recall, FPR/FNR, confusion matrix. The honest report. |
| `detector.py` | `DeepfakeDetector`: face extraction → batched inference → window scoring → EMA → verdict. |
| `models/backbone.py` | `DeepfakeBackbone` (timm wrapper). Returns `(logits, features)`; features are for future fusion. |
| `temporal.py` | 32-frame circular buffer, EMA smoothing, `Verdict` enum. |
| `router.py` | FastAPI router: `/health`, `/session/{id}`, `/reset/{id}`, WebSocket `/ws/{id}`. |
| `preprocessing.py` | MediaPipe face crop + ImageNet normalization. |
| `config.py` | All hyperparameters; thresholds get updated from `calibrate.py`. |

## After calibration

Paste the values `calibrate.py` prints into `config.py → InferenceConfig`:

```python
fake_threshold: float = <from calibration>
uncertain_low:  float = <from calibration>
```

## Known follow-ups (not Layer 1)

- `train.py` trains the bare timm backbone; the wrapper's `feature_head` stays
  randomly initialized. Harmless now (inference uses only `logits`), but the head
  must be trained before cross-modal fusion uses those features.
- CPU fallback config uses `input_size=224` while the model was trained at 380 —
  fine for a smoke test, lower accuracy. Keep 380 for real eval.
- Streams 2 (audio) and 3 (metadata), cross-modal attention, FedAvg federation,
  agentic orchestration, and Face Verification remain unbuilt.
