"""
src/retrieval/query_pipeline.py

Pipeline complet d'identification d'un morceau à partir d'un extrait audio.
Orchestre : chargement audio → embeddings → FAISS → re-ranking fingerprint.
Responsable : Personne C
"""

from __future__ import annotations


def identify_track(
    audio_path: str,
    method: str | None = None,
    top_n: int = 5,
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
    ...
