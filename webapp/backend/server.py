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

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import src.config as config
from src.retrieval.query_pipeline import identify_track

# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Shazam Maison API", version="1.0.0")

# CORS — allow Vite dev server (port 5173) and same origin in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── helpers ───────────────────────────────────────────────────────────────────
METADATA_PATH = ROOT / "data" / "processed" / "metadata.parquet"
_metadata_cache: dict[str, dict] | None = None


def _clean(val, default=None):
    """Return None for NaN / empty values, otherwise the value."""
    if val is None:
        return default
    try:
        import math
        if isinstance(val, float) and math.isnan(val):
            return default
    except (TypeError, ValueError):
        pass
    return val if val != "" else default


def _get_metadata() -> dict[str, dict]:
    """Return {track_id: {...}} — lazy-loaded once."""
    global _metadata_cache
    if _metadata_cache is not None:
        return _metadata_cache
    if not METADATA_PATH.exists():
        _metadata_cache = {}
        return _metadata_cache
    cols = ["track_id", "title", "artist", "genre", "duration_s"]
    available = pd.read_parquet(METADATA_PATH, columns=["track_id"]).columns.tolist()
    read_cols = [c for c in cols if c in
                 pd.read_parquet(METADATA_PATH, columns=cols[:1]).columns.tolist()
                 or c == "track_id"]
    # Read what's available
    df = pd.read_parquet(METADATA_PATH)
    out: dict[str, dict] = {}
    for row in df.itertuples(index=False):
        tid = str(row.track_id)
        out[tid] = {
            "title":      _clean(getattr(row, "title",      None), tid),
            "artist":     _clean(getattr(row, "artist",     None), "Unknown"),
            "album":      _clean(getattr(row, "album",      None)),
            "genre":      _clean(getattr(row, "genre",      None)),
            "duration_s": _clean(getattr(row, "duration_s", None)),
            "cover_url":  _clean(getattr(row, "cover_url",  None)),
        }
    _metadata_cache = out
    return _metadata_cache


def _streaming_links(artist: str, title: str) -> dict[str, str]:
    """Build search-based streaming links (no API key needed)."""
    q = f"{artist} {title}".replace(" ", "+")
    return {
        "youtube":  f"https://www.youtube.com/results?search_query={q}",
        "spotify":  f"https://open.spotify.com/search/{q}",
        "deezer":   f"https://www.deezer.com/search/{q}",
        "apple":    f"https://music.apple.com/search?term={q}",
    }


def _recommendations(track_id: str, genre: str | None, top: int = 4) -> list[dict]:
    """Return tracks from the same genre (excluding the identified track)."""
    if not genre:
        return []
    meta = _get_metadata()
    recs = [
        {
            "track_id":  tid,
            "title":     v["title"],
            "artist":    v["artist"],
            "album":     v.get("album"),
            "cover_url": v.get("cover_url"),
            "streaming": _streaming_links(v["artist"], v["title"]),
        }
        for tid, v in meta.items()
        if v.get("genre") == genre and tid != track_id
    ]
    return recs[:top]


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/config")
def get_config() -> dict:
    return {
        "listen_duration": config.UI_LISTEN_DURATION,
        "confidence_ratio": config.UI_CONFIDENCE_RATIO,
        "embedding_method": config.EMBEDDING_METHOD,
    }


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
        raw_results = identify_track(pipeline_path, top_n=config.VECTOR_TOP_N_RESULTS)
    except Exception as exc:
        logger.error("identify_track failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        os.unlink(tmp_path)
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)

    try:
        meta = _get_metadata()
    except Exception as exc:
        logger.error("_get_metadata failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"metadata error: {exc}")

    if not raw_results:
        return {"results": [], "confident": False}

    results = []
    for rank, (track_id, score) in enumerate(raw_results, start=1):
        info      = meta.get(track_id, {})
        artist    = info.get("artist",    "Unknown")
        title     = info.get("title",     track_id)
        album     = info.get("album")
        genre     = info.get("genre")
        cover_url = info.get("cover_url")
        results.append({
            "rank":      rank,
            "track_id":  track_id,
            "cover_url": cover_url,
            "album":     album,
            "title":    title,
            "artist":   artist,
            "genre":    genre,
            "score":    round(score, 4),
            "streaming": _streaming_links(artist, title),
        })

    # Confidence: score[0] / score[1] ≥ UI_CONFIDENCE_RATIO
    confident = False
    if len(raw_results) >= 2 and raw_results[1][1] > 0:
        confident = (raw_results[0][1] / raw_results[1][1]) >= config.UI_CONFIDENCE_RATIO
    elif len(raw_results) == 1:
        confident = True

    best = results[0]
    recs = _recommendations(best["track_id"], best.get("genre"))

    return {
        "results":         results,
        "confident":       confident,
        "recommendations": recs,
    }


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
