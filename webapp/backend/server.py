"""
webapp/backend/server.py

FastAPI backend for Shazam Maison.
Exposes:
    POST /api/identify  — identify an audio file or recording
    GET  /api/config    — return UI config values
    GET  /api/health    — liveness check
    GET  /              — serve React build (production)
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import traceback
import warnings
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shazam")

warnings.filterwarnings("ignore", message=".*upsample_bicubic2d.*", category=UserWarning)

# ── resolve project root (webapp/backend → project root) ──────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# All relative paths in the pipeline (data/index, data/chroma, etc.) expect
# the working directory to be the project root — force it here.
os.chdir(ROOT)

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import src.config as config
from src.api.app import build_identification_response, get_ui_config

# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Shazam Maison API", version="1.0.0")

# CORS — allow Vite dev server (port 5173) and same origin in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── routes ────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/config")
def get_config() -> dict:
    return get_ui_config()


def _to_wav(src: str) -> str | None:
    """Convert any audio file to a clean 22050 Hz mono WAV using ffmpeg.
    Returns the path to the WAV file, or None if ffmpeg is unavailable."""
    dst = src + "_converted.wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src,
             "-ar", str(config.SAMPLE_RATE),
             "-ac", "1",
             "-sample_fmt", "s16",
             dst],
            capture_output=True,
            check=True,
        )
        return dst
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning("ffmpeg conversion failed: %s", exc)
        return None


@app.post("/api/identify")
async def identify(file: UploadFile = File(...)) -> dict:
    """
    Accept a WAV/MP3/OGG/WebM audio file, run the full identification pipeline
    and return ranked results with metadata, streaming links and recommendations.
    """
    # Save upload to a temp file
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Convert to clean WAV so librosa uses soundfile (fast, no warnings)
    wav_path = _to_wav(tmp_path)
    pipeline_path = wav_path if wav_path else tmp_path

    try:
        response = build_identification_response(
            pipeline_path,
            top=config.VECTOR_TOP_N_RESULTS,
            detailed=True,
        )
    except Exception as exc:
        logger.error("build_identification_response failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        os.unlink(tmp_path)
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)
    return response


# ── serve React build in production ───────────────────────────────────────────
FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        index = FRONTEND_DIST / "index.html"
        return FileResponse(str(index))


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
