"""
src/retrieval/query_pipeline.py

Pipeline complet d'identification d'un morceau à partir d'un extrait audio.
Orchestre : chargement audio → embeddings → FAISS → re-ranking fingerprint.
Responsable : Personne C
"""

from __future__ import annotations

import src.config as config
from src.audio.loading import load_audio
from src.audio.preprocessing import iter_segments
from src.features.embeddings_audio import embed_segment
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

    all_results = [] # Stockage des résultats FAISS pour chaque segment.
    for _, segment in iter_segments(waveform= waveform, sr= sr): # Découpage de l'audio.
        embedding = embed_segment(
            waveform= segment,
            sr= sr,
            method= method,
            clap_model_name=config.CLAP_MODEL_NAME,
            muq_model_name=config.MUQ_MODEL_NAME
            )
        
        # K voisins avec FAISS :
        distances, indices = search_segments(index= index, query_embedding= embedding, k= config.VECTOR_TOP_K_SEGMENTS)

        all_results.append((distances, indices)) # Ajout du résultat du segment dans la liste des scores.
    
    global_scores = {} # Dictionnaire global : track_id → score cumulé sur tous les segments.

    for distances, indices in all_results:                          # Pour chaque segment de l'audio que l'on cherche.
        partial = aggregate_by_track(indices, distances, segments)  # Traduction les indices FAISS en track_id et agréger
        for track_id, score in partial:                             # Pour chaque morceau trouvé par ce segment
            global_scores[track_id]= global_scores.get(track_id, 0.0) + score # Ajouter son score au total global
    
    candidates = sorted(global_scores.items(), key= lambda x: x[1], reverse= True) [:config.VECTOR_TOP_N_TRACKS] # Garder les k meilleurs candidats triés par score décroissant.


    # ---------- Stage 2 (Fingerprinting) : ----------


    query_fp = extract_fingerprint(waveform, sr) # Extraction du fingerprint de l'audio recherché.

    final_scores = [] # Scores finaux combiné (FAISS + fingerprinting)
    
    for track_id, score_faiss in candidates: # Boucle sur les candidats de FAISS.
        path = segments[segments["track_id"]== track_id].iloc[0]["path"]        # Récupération du chemin de l'audio candidat.

        candidate_waveform, candidate_sr = load_audio(path, target_sr=targ_sr)  # Chargement de l'audio candidat.
        candidate_fp = extract_fingerprint(candidate_waveform, candidate_sr)    # Extraction du fingerprinting de l'audio candidat.

        score_fp = fingerprint_similarity(query_fp, candidate_fp)               # Calcule du score de similarité avec la méthode de fingerprint.
        score_final = score_faiss * score_fp                                    # Calcule du score avec FAISS et fingerprint.
        final_scores.append((track_id, score_final))

    return sorted(final_scores, key=lambda x: x[1], reverse=True)[:top_n] # Retourner les top_n meilleurs morceaux.




