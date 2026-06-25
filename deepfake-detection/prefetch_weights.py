# prefetch_weights.py - download the pretrained backbone once, with resume.
# Useful on flaky connections: HuggingFace resumes partial downloads from cache,
# so re-running this picks up where it dropped. Once it succeeds, train.py /
# evaluate.py / main.py find the weights in cache and never hit the network.
#
#   python prefetch_weights.py
#
# Optional speed-up (resumable, parallel chunks):
#   pip install hf_transfer
#   PowerShell:  $env:HF_HUB_ENABLE_HF_TRANSFER = "1"; python prefetch_weights.py

import time
import timm

BACKBONE = "tf_efficientnet_b4_ns"   # -> mapped to tf_efficientnet_b4.ns_jft_in1k

def main(max_retries: int = 8):
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  Attempt {attempt}/{max_retries}: fetching {BACKBONE} weights...")
            timm.create_model(BACKBONE, pretrained=True, num_classes=1)
            print("  Done. Pretrained weights are cached; train.py will not re-download.")
            return
        except Exception as e:
            wait = min(2 ** attempt, 60)
            print(f"    failed ({type(e).__name__}: {e}); resuming in {wait}s")
            time.sleep(wait)
    raise SystemExit("Could not download weights. Check connection / proxy and retry.")

if __name__ == "__main__":
    main()
