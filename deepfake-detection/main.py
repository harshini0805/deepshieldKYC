"""
main.py — Standalone deepfake detection server
Run with: python main.py

This is the complete entry point for the deepfake layer running independently,
before it is integrated with the liveness layer.

Access:
  http://localhost:8000/deepfake/health
  ws://localhost:8000/deepfake/ws/{session_id}
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from router import deepfake_router, init_detector


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Check if FF++ / DFDC weights exist; fall back to ImageNet if not
    weights_candidates = [
        "weights/dfdc_resnext50.pth",
        "weights/ff_efficientnet_b4.pth",
        "weights/xception_c23_all.p",
    ]
    weights_path = next(
        (p for p in weights_candidates if Path(p).exists()),
        None,
    )

    if weights_path:
        print(f"[Startup] Using weights: {weights_path}")
    else:
        print("[Startup] No fine-tuned weights found — using ImageNet pretrained.")
        print("          Detection will work but quality is lower than FF++ weights.")
        print("          Run: python get_weights.py to download DFDC checkpoint.")

    init_detector(
        weights_path=weights_path,
        force_cpu=False,   # Change to True if no CUDA GPU available
    )

    yield  # Server runs here
    print("[Shutdown] Deepfake detector released.")


app = FastAPI(
    title="FED-MEMF Deepfake Detection Layer",
    description="Visual stream deepfake gate — Stream 1 of the FED-MEMF architecture",
    version="0.1.0",
    lifespan=lifespan,
)

# Allow React frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Tighten to your frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(deepfake_router)


@app.get("/")
async def root():
    return {
        "service": "FED-MEMF Deepfake Detection Layer",
        "version": "0.1.0",
        "endpoints": {
            "health": "/deepfake/health",
            "websocket": "ws://localhost:8000/deepfake/ws/{session_id}",
            "docs": "/docs",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )