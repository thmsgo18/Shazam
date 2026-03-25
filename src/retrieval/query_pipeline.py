"""
src/retrieval/query_pipeline.py

Pipeline complet d'identification d'un morceau à partir d'un extrait audio.
Orchestre : chargement audio → embeddings → FAISS → re-ranking fingerprint.
Responsable : Personne C
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

# Active le fallback CPU pour les opérations non supportées sur MPS (Apple Silicon).
# Sans effet sur les autres machines (CUDA, CPU).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import librosa
import torch
torch.set_num_threads(4)

from concurrent.futures import ThreadPoolExecutor

import src.config as config
from src.audio.loading import load_audio
from src.audio.preprocessing import iter_segments
from src.features.embeddings_audio import embed_segment, muq_batch_embeddings
from src.retrieval.searcher import load_searcher, search_segments, aggregate_by_track
from src.features.fingerprint import extract_fingerprint, fingerprint_similarity


def identify_track(
    audio_path: str,
    method: str | None = None,
    top_n: int = config.VECTOR_TOP_N_RESULTS,
) -> list[tuple[str, float]]:
    """
    Identifie le morceau correspondant à un fichier audio.

    Pipeline en deux étapes :
    - Stage 1 (Embeddings + FAISS) : cherche les N candidats les plus proches
      dans l'index vectoriel. Rapide, mais approximatif.
    - Stage 2 (Fingerprinting) : re-classe les candidats en comparant les
      fingerprints de l'audio requête avec ceux des candidats.
      Précis, mais plus lent — ne s'applique qu'aux N candidats du stage 1.

    Args:
        audio_path: chemin vers le fichier audio à identifier.
        method:     méthode d'embedding — None utilise config.EMBEDDING_METHOD.
        top_n:      nombre de résultats finaux à retourner.

    Returns:
        Liste triée [(track_id, score_final), ...] du meilleur au moins bon.

    Dépendances :
        - src.audio.loading.load_audio()
        - src.audio.preprocessing.iter_segments()
        - src.features.embeddings_audio.embed_segment()
        - src.retrieval.searcher.load_searcher()
        - src.retrieval.searcher.search_segments()
        - src.retrieval.searcher.aggregate_by_track()
        - src.features.fingerprint.extract_fingerprint()
        - src.features.fingerprint.fingerprint_similarity()
    """
    if method is None:
        method = config.EMBEDDING_METHOD # Si pas de méthode donnée on utilise celle du config.
    
    # On prend la valeur du Sample Rate idéale en fonction de la méthode utilisé :
    if method == "clap":
        targ_sr = config.CLAP_SAMPLE_RATE
    elif method == "muq":
        targ_sr = config.MUQ_SAMPLE_RATE
    else :
        targ_sr = config.SAMPLE_RATE
    
    # ---------- Stage 1 (Embeddings + FAISS) : ----------

    index, segments = load_searcher(method) # Charge l'index FAISS et le .parquet pour la correspondance.

    waveform, sr = load_audio(path= audio_path, target_sr= targ_sr) # Chargement de l'audio.

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
                muq_model_name=config.MUQ_MODEL_NAME
            )
            distances, indices = search_segments(index=index, query_embedding=embedding, k=config.VECTOR_TOP_K_SEGMENTS)
            all_results.append((distances, indices))
    
    global_scores = {} # Dictionnaire global : track_id → score cumulé sur tous les segments.

    for distances, indices in all_results:                          # Pour chaque segment de l'audio que l'on cherche.
        partial = aggregate_by_track(indices, distances, segments)  # Traduction les indices FAISS en track_id et agréger
        for track_id, score in partial:                             # Pour chaque morceau trouvé par ce segment
            global_scores[track_id]= global_scores.get(track_id, 0.0) + score # Ajouter son score au total global
    
    candidates = sorted(global_scores.items(), key= lambda x: x[1], reverse= True) [:config.VECTOR_TOP_N_TRACKS] # Garder les k meilleurs candidats triés par score décroissant.


    # ---------- Stage 2 (Fingerprinting) : ----------

    # Court-circuit : si le 1er candidat est très largement devant, inutile de faire le fingerprinting
    if config.OPT_SHORTCIRCUIT and len(candidates) >= 2:
        if candidates[1][1] > 0 and candidates[0][1] / candidates[1][1] >= config.OPT_SHORTCIRCUIT_RATIO:
            return [(track_id, score) for track_id, score in candidates[:top_n]]

    # Calcul du fingerprint de la requête — toujours à SAMPLE_RATE pour cohérence avec les fingerprints stockés
    if targ_sr != config.SAMPLE_RATE:
        waveform_fp = librosa.resample(waveform, orig_sr=targ_sr, target_sr=config.SAMPLE_RATE)
    else:
        waveform_fp = waveform
    query_fp = extract_fingerprint(waveform_fp, config.SAMPLE_RATE)

    # Charger les fingerprints pré-calculés si disponibles (évite de recharger les MP3)
    fp_path = Path(config.FINGERPRINTS_PATH)
    stored_fps = None
    if fp_path.exists():
        with open(fp_path, "rb") as f:
            stored_fps = pickle.load(f)

    def process_candidate(candidate):
        """Récupère le fingerprint d'un candidat (depuis le cache ou en rechargeant l'audio)."""
        track_id, score_faiss = candidate
        if stored_fps is not None and track_id in stored_fps:
            # Fingerprint pré-calculé disponible — pas besoin de recharger l'audio
            candidate_fp = stored_fps[track_id]
        else:
            # Fallback : recharger l'audio depuis le disque
            path = segments[segments["track_id"] == track_id].iloc[0]["path"]
            candidate_waveform, _ = load_audio(path, target_sr=config.SAMPLE_RATE)
            candidate_fp = extract_fingerprint(candidate_waveform, config.SAMPLE_RATE)
        score_fp = fingerprint_similarity(query_fp, candidate_fp)
        return (track_id, score_faiss * score_fp)

    if config.OPT_FINGERPRINT_PARALLEL:
        # Chargement et fingerprinting des candidats en parallèle
        with ThreadPoolExecutor(max_workers=4) as executor:
            final_scores = list(executor.map(process_candidate, candidates))
    else:
        # Traitement séquentiel (comportement de base)
        final_scores = [process_candidate(c) for c in candidates]

    return sorted(final_scores, key=lambda x: x[1], reverse=True)[:top_n] # Retourner les top_n meilleurs morceaux.
