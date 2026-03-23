"""
src/retrieval/searcher.py

Chargement de l'index FAISS et fonctions de recherche.
Responsable : Personne C
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pandas as pd


def load_searcher(method: str) -> tuple[faiss.Index, pd.DataFrame]:
    """
    Charge l'index FAISS et le parquet de segments pour une méthode donnée.

    Args:
        method: méthode d'embedding — "mfcc", "clap" ou "muq".

    Returns:
        Tuple (index FAISS, DataFrame segments).

    Raises:
        FileNotFoundError: si l'index n'existe pas (build_index.py non lancé).
    """
    index_path = Path(f"data/index/index_{method}.faiss")
    seg_path = Path(f"data/features/segments_{method}.parquet")

    if not index_path.exists():
        raise FileNotFoundError(f"Index introuvable : {index_path}. Lance build_index.py d'abord.")

    index = faiss.read_index(str(index_path))
    segments = pd.read_parquet(str(seg_path))

    return (index, segments)


def search_segments(
    index: faiss.Index,
    query_embedding: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cherche les k segments les plus proches dans l'index FAISS.

    Args:
        index:           index FAISS chargé.
        query_embedding: vecteur de la requête, shape (D,).
        k:               nombre de voisins à retourner.

    Returns:
        Tuple (distances, indices) de shape (k,).
        Attention : FAISS retourne -1 dans indices si pas assez de voisins.
    """
    xq = query_embedding.astype("float32").reshape(1,-1)    # FAISS attend un tableau 2D (1, D) en float32
    faiss.normalize_L2(x= xq)                               # Même normalisation qu'à l'indexation
    distances, indices = index.search(x= xq, k= k)          # Recherche des k voisins
    return distances[0], indices[0]                         # [0] pour enlever le coté batch


def aggregate_by_track(
    indices: np.ndarray,
    distances: np.ndarray,
    segments: pd.DataFrame,
) -> list[tuple[str, float]]:
    """
    Agrège les résultats de recherche FAISS par track_id.

    Pour chaque indice retourné, récupère le track_id correspondant
    dans le DataFrame et additionne les scores.

    Args:
        indices:   indices FAISS de shape (k,).
        distances: distances FAISS de shape (k,) — scores cosine (plus élevé = plus proche).
        segments:  DataFrame segments avec colonne "track_id".

    Returns:
        Liste triée [(track_id, score_total), ...] du meilleur au moins bon.
    """
    scores = {}
    for idx, dist in zip(indices, distances):
        if idx == -1 :
            continue
        track_id = segments.iloc[idx]["track_id"]
        scores[track_id]= scores.get(track_id, 0.0) + float(dist)
        
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

