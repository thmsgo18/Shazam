"""
src/retrieval/searcher.py

Loading the FAISS index and search functions.
Owner: Person C
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pandas as pd

import src.config as config

ROOT = Path(__file__).resolve().parents[2]
_SEARCHER_CACHE: dict[str, tuple[faiss.Index, pd.DataFrame]] = {}


def load_searcher(method: str, force_reload: bool = False) -> tuple[faiss.Index, pd.DataFrame]:
    """
    Loads the FAISS index and the segments parquet for a given method.

    Args:
        method: embedding method — "mfcc", "clap", or "muq".
                The collection key (method + model) is resolved via config.get_collection_key().

    Returns:
        Tuple (FAISS index, segments DataFrame).

    Raises:
        FileNotFoundError: if the index does not exist or has not been rebuilt.
    """
    key        = config.get_collection_key(method)

    if not force_reload and key in _SEARCHER_CACHE:
        return _SEARCHER_CACHE[key]

    index_type = config.INDEX_TYPE
    index_dir  = ROOT / config.INDEX_DIR
    index_path = index_dir / f"index_{key}_{index_type}.faiss"

    if not index_path.exists():
        # Look for available keys to help the user
        available = [p.stem for p in index_dir.glob(f"index_*_{index_type}.faiss")]
        if available:
            raise FileNotFoundError(
                f"No index for '{method}' (key='{key}', type={index_type}) in {index_dir}/.\n"
                f"Available indexes: {', '.join(sorted(available))}.\n"
                f"Change EMBEDDING_METHOD / CLAP_MODEL_NAME in config.py or run "
                f"`python manage.py rebuild --what index`."
            )
        raise FileNotFoundError(
            f"No index found in {index_dir}/.\n"
            f"Run first: python manage.py ingest"
        )

    index = faiss.read_index(str(index_path))

    # The order of segments is saved in INDEX_DIR by build_index.py
    # (rebuilt from ChromaDB at each build — always synchronized with the FAISS index)
    seg_path = index_dir / f"segments_{key}.parquet"
    if not seg_path.exists():
        raise FileNotFoundError(
            f"Missing segment order file: {seg_path}\n"
            f"Run first: python manage.py rebuild --what index"
        )
    segments = pd.read_parquet(str(seg_path))

    _SEARCHER_CACHE[key] = (index, segments)
    return _SEARCHER_CACHE[key]


def clear_searcher_cache(method: str | None = None) -> None:
    """Clears the memory cache of indexes/searchers."""
    if method is None:
        _SEARCHER_CACHE.clear()
        return
    key = config.get_collection_key(method)
    _SEARCHER_CACHE.pop(key, None)


def search_segments(
    index: faiss.Index,
    query_embedding: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Searches for the k closest segments in the FAISS index.

    Args:
        index:           loaded FAISS index.
        query_embedding: query vector, shape (D,).
        k:               number of neighbors to return.

    Returns:
        Tuple (distances, indices) of shape (k,).
        Warning: FAISS returns -1 in indices if there are not enough neighbors.
    """
    xq = query_embedding.astype("float32").reshape(1,-1)    # FAISS expects a 2D array (1, D) in float32
    faiss.normalize_L2(x= xq)                               # Same normalization as during indexing
    distances, indices = index.search(x= xq, k= k)          # Search for the k neighbors
    return distances[0], indices[0]                         # [0] to remove the batch dimension


def aggregate_by_track(
    indices: np.ndarray,
    distances: np.ndarray,
    segments: pd.DataFrame,
) -> list[tuple[str, float]]:
    """
    Aggregates FAISS search results by track_id.

    For each returned index, fetches the corresponding track_id
    in the DataFrame and sums the scores.

    Args:
        indices:   FAISS indices of shape (k,).
        distances: FAISS distances of shape (k,) — cosine scores (higher = closer).
        segments:  segments DataFrame with a "track_id" column.

    Returns:
        Sorted list [(track_id, total_score), ...] from best to worst.
    """
    scores = {}
    for idx, dist in zip(indices, distances):                       # zip pairs the items together
        if idx == -1 :
            continue
        track_id = segments.iloc[idx]["track_id"]                   # Accesses a dataframe row given by idx
        scores[track_id]= scores.get(track_id, 0.0) + float(dist)   # Accumulates the score. High score = very close song.

    return sorted(scores.items(), key=lambda x: x[1], reverse=True) # Returns a list of tuples sorted in descending order.