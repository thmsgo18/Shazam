"""
src/retrieval/query_pipeline.py

Complete pipeline to identify a track from an audio excerpt.
Orchestrates: audio loading → embeddings → FAISS → fingerprint re-ranking.

Final ranking strategy (cascading):
  - Primary key   : fingerprint score (DESC) — source of truth
  - Secondary key : FAISS score (DESC)       — fallback if FP=0 for all
No mixing of both scores: FP alone decides when it has a signal.
"""

from __future__ import annotations

import os
import pickle
import sqlite3
from collections import OrderedDict
from pathlib import Path

# Enables CPU fallback for operations unsupported on MPS (Apple Silicon).
# No effect on other machines (CUDA, CPU).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import librosa
import torch
torch.set_num_threads(4)

from concurrent.futures import ThreadPoolExecutor

import src.config as config

ROOT = Path(__file__).resolve().parents[2]

from src.audio.loading import load_audio
from src.audio.preprocessing import iter_segments, preprocess_query
from src.features.embeddings_audio import embed_segment, muq_batch_embeddings
from src.retrieval.searcher import load_searcher, search_segments, aggregate_by_track
from src.features.fingerprint import extract_fingerprint, fingerprint_similarity

_FINGERPRINT_CACHE: OrderedDict[str, set] | None = None
_FINGERPRINT_DB_MTIME_NS: int | None = None


def _enforce_fingerprint_cache_limit(cache: OrderedDict[str, set]) -> None:
    """Keeps at most FINGERPRINT_CACHE_MAX recent fingerprints in memory."""
    max_size = max(0, int(getattr(config, "FINGERPRINT_CACHE_MAX", 256)))
    if max_size == 0:
        cache.clear()
        return
    while len(cache) > max_size:
        cache.popitem(last=False)


def load_fingerprint_cache(force_reload: bool = False) -> OrderedDict[str, set]:
    """Initializes/returns the in-memory fingerprint cache, populated on demand."""
    global _FINGERPRINT_CACHE, _FINGERPRINT_DB_MTIME_NS

    fp_db = ROOT / config.FINGERPRINTS_DB
    if not fp_db.exists():
        _FINGERPRINT_CACHE = OrderedDict()
        _FINGERPRINT_DB_MTIME_NS = None
        return _FINGERPRINT_CACHE

    current_mtime_ns = fp_db.stat().st_mtime_ns
    if not force_reload and _FINGERPRINT_CACHE is not None and _FINGERPRINT_DB_MTIME_NS == current_mtime_ns:
        return _FINGERPRINT_CACHE

    _FINGERPRINT_CACHE = OrderedDict()
    _FINGERPRINT_DB_MTIME_NS = current_mtime_ns
    return _FINGERPRINT_CACHE


def warmup_fingerprint_store() -> dict[str, int | bool]:
    """
    Prepares the fingerprint layer without loading 2+ GB of blobs into RAM.

    Initializes the empty memory cache and forces a very lightweight SQLite access
    to prevent the first identify call from also paying this opening cost.
    """
    cache = load_fingerprint_cache()
    fp_db = ROOT / config.FINGERPRINTS_DB
    if not fp_db.exists():
        return {"available": False, "cached": len(cache)}

    with sqlite3.connect(fp_db) as conn:
        row = conn.execute("SELECT COUNT(*) FROM fingerprints").fetchone()

    return {
        "available": True,
        "rows": int(row[0]) if row else 0,
        "cached": len(cache),
    }


def get_cached_fingerprint(track_id: str) -> set | None:
    """Returns a track's fingerprint from the memory cache or SQLite."""
    cache = load_fingerprint_cache()
    if track_id in cache:
        fingerprint = cache.pop(track_id)
        cache[track_id] = fingerprint
        return fingerprint

    fp_db = ROOT / config.FINGERPRINTS_DB
    if not fp_db.exists():
        return None
    with sqlite3.connect(fp_db) as conn:
        row = conn.execute(
            "SELECT hashes FROM fingerprints WHERE track_id = ?", (track_id,)
        ).fetchone()
    fingerprint = pickle.loads(row[0]) if row else None
    if fingerprint is not None:
        cache[track_id] = fingerprint
        _enforce_fingerprint_cache_limit(cache)
    return fingerprint


def identify_track(
    audio_path: str,
    method: str | None = None,
    top_n: int = config.VECTOR_TOP_N_RESULTS,
    detailed: bool = False,
) -> list[tuple]:
    """
    Identifies the track corresponding to an audio file.

    Two-stage pipeline:
    - Stage 1 (Embeddings + FAISS): filters the N closest candidates
      in the vector index. Fast but approximate — recall role.
    - Stage 2 (Fingerprinting): ranks candidates by exact match
      of spectral peaks + temporal alignment. Precise — precision role.

    Cascading final ranking:
      1. score_fp   DESC — fingerprint decides (source of truth)
      2. score_faiss DESC — fallback if FP=0 for all (audio too degraded)
    No mixing of both scores: FP always prevails when it has a signal.

    Args:
        audio_path: path to the audio file to identify.
        method:     embedding method — None uses config.EMBEDDING_METHOD.
        top_n:      number of final results to return.
        detailed:   if True, returns
                    (track_id, score_final, score_faiss, score_fp)
                    instead of (track_id, score_final).

    Returns:
        If detailed=False : [(track_id, score_final), ...]
        If detailed=True  : [(track_id, score_final, score_faiss, score_fp), ...]
    """
    if method is None:
        method = config.EMBEDDING_METHOD # If no method is given, use the config one.

    # We take the ideal Sample Rate value based on the method used:
    if method == "clap":
        targ_sr = config.CLAP_SAMPLE_RATE
    elif method == "muq":
        targ_sr = config.MUQ_SAMPLE_RATE
    else:
        targ_sr = config.SAMPLE_RATE

    # ---------- Stage 1 (Embeddings + FAISS): ----------

    index, segments = load_searcher(method) # Load the FAISS index and the .parquet for matching.

    waveform, sr = load_audio(path=audio_path, target_sr=targ_sr)       # Load audio.
    waveform = preprocess_query(waveform, sr)                           # Preprocessing: HP 80Hz + LUFS -14 + peak norm

    # Split the audio into segments
    segment_list = [seg for _, seg in iter_segments(waveform=waveform, sr=sr)]

    all_results = [] # FAISS results storage for each segment.

    if config.OPT_BATCH_EMBED and method == "muq":
        # Batch embedding: process in groups of MUQ_BATCH_SIZE segments
        batch_size = config.MUQ_BATCH_SIZE
        for i in range(0, len(segment_list), batch_size):
            batch = segment_list[i:i + batch_size]
            embeddings = muq_batch_embeddings(batch, sr, model_name=config.MUQ_MODEL_NAME)
            for embedding in embeddings:
                distances, indices = search_segments(index=index, query_embedding=embedding, k=config.VECTOR_TOP_K_SEGMENTS)
                all_results.append((distances, indices))
    else:
        # Segment-by-segment embedding (default behavior)
        for segment in segment_list:
            embedding = embed_segment(
                waveform=segment,
                sr=sr,
                method=method,
                clap_model_name=config.CLAP_MODEL_NAME,
                muq_model_name=config.MUQ_MODEL_NAME,
                mert_model_name=config.MERT_MODEL_NAME,
            )
            distances, indices = search_segments(index=index, query_embedding=embedding, k=config.VECTOR_TOP_K_SEGMENTS)
            all_results.append((distances, indices))

    global_scores = {} # Global dictionary: track_id → cumulative score over all segments.

    for distances, indices in all_results:                                  # For each segment of the queried audio.
        partial = aggregate_by_track(indices, distances, segments)          # Translate FAISS indices into track_id and aggregate
        for track_id, score in partial:                                     # For each track found by this segment
            global_scores[track_id] = global_scores.get(track_id, 0.0) + score # Add its score to the global total

    candidates = sorted(global_scores.items(), key=lambda x: x[1], reverse=True)[:config.VECTOR_TOP_N_TRACKS]

    # ---------- Stage 2 (Fingerprinting): ----------

    # Compute query fingerprint — always at SAMPLE_RATE for consistency with stored fingerprints
    if targ_sr != config.SAMPLE_RATE:
        waveform_fp = librosa.resample(waveform, orig_sr=targ_sr, target_sr=config.SAMPLE_RATE)
    else:
        waveform_fp = waveform
    query_fp = extract_fingerprint(waveform_fp, config.SAMPLE_RATE)

    def process_candidate(candidate):
        """Retrieves a candidate's fingerprint and returns its detailed scores."""
        track_id, score_faiss = candidate
        candidate_fp = get_cached_fingerprint(track_id)
        if candidate_fp is None or len(candidate_fp) == 0:
            # Missing or empty fingerprint: FP score to 0, FAISS will act as fallback.
            return (track_id, score_faiss, 0.0)
        score_fp = fingerprint_similarity(query_fp, candidate_fp)
        return (track_id, score_faiss, score_fp)

    if config.OPT_FINGERPRINT_PARALLEL:
        # Compute fingerprint similarities in parallel (CPU-bound)
        with ThreadPoolExecutor(max_workers=4) as executor:
            scored = list(executor.map(process_candidate, candidates))
    else:
        scored = [process_candidate(c) for c in candidates]

    # ── Cascading final ranking ──────────────────────────────────────────────
    # Primary key   : score_fp   (DESC) — fingerprint is the source of truth
    # Secondary key : score_faiss (DESC) — fallback if FP=0 for all candidates
    scored.sort(key=lambda x: (x[2], x[1]), reverse=True)

    top = scored[:top_n]
    use_faiss_fallback = not any(score_fp > 0 for _, _, score_fp in top)

    if detailed:
        return [
            (
                tid,
                score_faiss if use_faiss_fallback else score_fp,
                score_faiss,
                score_fp,
            )
            for tid, score_faiss, score_fp in top
        ]
    return [
        (tid, score_faiss if use_faiss_fallback else score_fp)
        for tid, score_faiss, score_fp in top
    ]