"""
src/retrieval/query_pipeline.py

Pipeline complet d'identification d'un morceau à partir d'un extrait audio.
Orchestre : chargement audio → embeddings → FAISS → re-ranking fingerprint.

Stratégie de classement final (cascade) :
  - Clé primaire   : score fingerprint (DESC) — source de vérité
  - Clé secondaire : score FAISS (DESC)       — fallback si FP=0 pour tous
Aucun mélange des deux scores : le FP décide seul quand il a un signal.
"""

from __future__ import annotations

import os
import pickle
import sqlite3
from pathlib import Path

# Active le fallback CPU pour les opérations non supportées sur MPS (Apple Silicon).
# Sans effet sur les autres machines (CUDA, CPU).
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


def identify_track(
    audio_path: str,
    method: str | None = None,
    top_n: int = config.VECTOR_TOP_N_RESULTS,
    detailed: bool = False,
) -> list[tuple]:
    """
    Identifie le morceau correspondant à un fichier audio.

    Pipeline en deux étapes :
    - Stage 1 (Embeddings + FAISS) : filtre les N candidats les plus proches
      dans l'index vectoriel. Rapide mais approximatif — rôle de recall.
    - Stage 2 (Fingerprinting) : classe les candidats par correspondance exacte
      des pics spectraux + alignement temporel. Précis — rôle de precision.

    Classement final en cascade :
      1. score_fp  DESC  — le fingerprint décide (source de vérité)
      2. score_faiss DESC — fallback si FP=0 pour tous (audio trop dégradé)
    Aucun mélange des deux scores : le FP prime toujours quand il a un signal.

    Args:
        audio_path: chemin vers le fichier audio à identifier.
        method:     méthode d'embedding — None utilise config.EMBEDDING_METHOD.
        top_n:      nombre de résultats finaux à retourner.
        detailed:   si True, retourne (track_id, score_fp, score_faiss, score_fp)
                    au lieu de (track_id, score_fp).

    Returns:
        Si detailed=False : [(track_id, score_fp), ...]
        Si detailed=True  : [(track_id, score_fp, score_faiss, score_fp), ...]
    """
    if method is None:
        method = config.EMBEDDING_METHOD # Si pas de méthode donnée on utilise celle du config.

    # On prend la valeur du Sample Rate idéale en fonction de la méthode utilisé :
    if method == "clap":
        targ_sr = config.CLAP_SAMPLE_RATE
    elif method == "muq":
        targ_sr = config.MUQ_SAMPLE_RATE
    else:
        targ_sr = config.SAMPLE_RATE

    # ---------- Stage 1 (Embeddings + FAISS) : ----------

    index, segments = load_searcher(method) # Charge l'index FAISS et le .parquet pour la correspondance.

    waveform, sr = load_audio(path=audio_path, target_sr=targ_sr)  # Chargement de l'audio.
    waveform = preprocess_query(waveform, sr)                       # Prétraitement : HP 80Hz + LUFS -14 + peak norm

    # Découpage de l'audio en segments
    segment_list = [seg for _, seg in iter_segments(waveform=waveform, sr=sr)]

    all_results = [] # Stockage des résultats FAISS pour chaque segment.

    if config.OPT_BATCH_EMBED and method == "muq":
        # Batch embedding : traitement par groupes de MUQ_BATCH_SIZE segments
        batch_size = config.MUQ_BATCH_SIZE
        for i in range(0, len(segment_list), batch_size):
            batch = segment_list[i:i + batch_size]
            embeddings = muq_batch_embeddings(batch, sr, model_name=config.MUQ_MODEL_NAME)
            for embedding in embeddings:
                distances, indices = search_segments(index=index, query_embedding=embedding, k=config.VECTOR_TOP_K_SEGMENTS)
                all_results.append((distances, indices))
    else:
        # Embedding segment par segment (comportement de base)
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

    global_scores = {} # Dictionnaire global : track_id → score cumulé sur tous les segments.

    for distances, indices in all_results:                          # Pour chaque segment de l'audio que l'on cherche.
        partial = aggregate_by_track(indices, distances, segments)  # Traduction les indices FAISS en track_id et agréger
        for track_id, score in partial:                             # Pour chaque morceau trouvé par ce segment
            global_scores[track_id] = global_scores.get(track_id, 0.0) + score # Ajouter son score au total global

    candidates = sorted(global_scores.items(), key=lambda x: x[1], reverse=True)[:config.VECTOR_TOP_N_TRACKS]


    # ---------- Stage 2 (Fingerprinting) : ----------

    # Calcul du fingerprint de la requête — toujours à SAMPLE_RATE pour cohérence avec les fingerprints stockés
    if targ_sr != config.SAMPLE_RATE:
        waveform_fp = librosa.resample(waveform, orig_sr=targ_sr, target_sr=config.SAMPLE_RATE)
    else:
        waveform_fp = waveform
    query_fp = extract_fingerprint(waveform_fp, config.SAMPLE_RATE)

    # Fingerprints : chargement depuis SQLite uniquement pour les candidats
    # (plus efficace que charger tout le fichier — on ne lit que ~20 lignes sur 3000)
    fp_db = ROOT / config.FINGERPRINTS_DB

    def _get_fp(track_id: str) -> set | None:
        if not fp_db.exists():
            return None
        with sqlite3.connect(fp_db) as conn:
            row = conn.execute(
                "SELECT hashes FROM fingerprints WHERE track_id = ?", (track_id,)
            ).fetchone()
        return pickle.loads(row[0]) if row else None

    def process_candidate(candidate):
        """Récupère le fingerprint d'un candidat et retourne ses scores détaillés."""
        track_id, score_faiss = candidate
        candidate_fp = _get_fp(track_id)
        if candidate_fp is None or len(candidate_fp) == 0:
            # Fingerprint manquant ou vide : score FP à 0, FAISS servira de fallback.
            return (track_id, score_faiss, 0.0)
        score_fp = fingerprint_similarity(query_fp, candidate_fp)
        return (track_id, score_faiss, score_fp)

    if config.OPT_FINGERPRINT_PARALLEL:
        # Calcul des similarités fingerprint en parallèle (CPU-bound)
        with ThreadPoolExecutor(max_workers=4) as executor:
            scored = list(executor.map(process_candidate, candidates))
    else:
        scored = [process_candidate(c) for c in candidates]

    # ── Classement final en cascade ──────────────────────────────────────────
    # Clé primaire   : score_fp  (DESC) — le fingerprint est la source de vérité
    # Clé secondaire : score_faiss (DESC) — fallback si FP=0 pour tous les candidats
    scored.sort(key=lambda x: (x[2], x[1]), reverse=True)

    top = scored[:top_n]

    if detailed:
        return [(tid, score_fp, score_faiss, score_fp) for tid, score_faiss, score_fp in top]
    return [(tid, score_fp) for tid, score_faiss, score_fp in top]
