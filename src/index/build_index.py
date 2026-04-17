"""
src/index/build_index.py

Construction and management of the FAISS index.
Also supports HNSW and IVF.
"""

from __future__ import annotations

from pathlib import Path

import faiss

ROOT = Path(__file__).resolve().parents[2]
import numpy as np
import pandas as pd


def build_index(embeddings: np.ndarray, index_type: str = "flat") -> faiss.Index:
    """
    Builds a FAISS index, according to the chosen type, from an embeddings matrix.

    Args:
        embeddings: matrix of shape (N, D) in float32.
                    Vectors must be L2 normalized.
        index_type: "flat", "hnsw" or "ivf"

    Returns:
        FAISS index ready for search (L2 normalized, cosine similarity).
    """
    
    xb = embeddings.astype("float32") # Conversion to float32 for FAISS

    # L2 normalization of vectors for cosine similarity
    faiss.normalize_L2(xb)

    # Create the FlatIP index
    d = xb.shape[1] # Embedding dim, allows changing method without dim issues.
    N = len(xb)

    index_type = index_type.lower()

    if index_type == "flat" : 
        index = faiss.IndexFlatIP(d)
    elif index_type == "hnsw" : 
        M = 32 # M: graph connectivity
        index = faiss.IndexHNSWFlat(d, M)
        index.hnsw.efConstruction = 40 # tuning for precision
    elif index_type == "ivf" :
        nbList = max(1, int(np.sqrt(N))) # Nb of inverted lists: square root of nb of vectors
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nbList, faiss.METRIC_INNER_PRODUCT)
        index.train(xb)
    else : 
        raise ValueError(f"Index type unknown: {index_type}. Possible choices: flat, hnsw, ivf") 

    # Add vectors
    index.add(xb)

    return index


def save_index(index: faiss.Index, path: Path) -> None:
    """
    Saves a FAISS index to disk.

    Args:
        index: FAISS index to save.
        path:  destination path (e.g.: data/index/index_mfcc.faiss).
    """
    
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    print(f"Index saved successfully at: {path}")


def load_index(path: Path) -> faiss.Index:
    """
    Loads a FAISS index from disk.

    Args:
        path: path to the .faiss file.

    Returns:
        Loaded FAISS index.
    """
    
    if not path.exists() : 
        raise FileNotFoundError(f"The index doesn't exist at: {path}")
    
    return faiss.read_index(str(path))


def _build_for_method(collection_key: str, index_type: str, chroma_client) -> None:
    """
    Builds and saves the FAISS index for a given ChromaDB collection.

    Args:
        collection_key: method+model key, e.g. "clap_larger_clap_music" or "mfcc".
                        Also the name of the ChromaDB collection.
    """
    import src.config as config

    index_dir  = ROOT / config.INDEX_DIR
    out_path   = index_dir / f"index_{collection_key}_{index_type}.faiss"
    order_path = index_dir / f"segments_{collection_key}.parquet"

    try:
        collection = chroma_client.get_collection(name=collection_key)
    except Exception:
        print(f"[build_index] Collection '{collection_key}' not found in ChromaDB — ignored.")
        return

    n = collection.count()
    if n == 0:
        print(f"[build_index] Collection '{collection_key}' is empty — ignored.")
        return

    print(f"\n[build_index] ── Collection: {collection_key} | {n} segments ──")

    # Pagination to avoid SQLite "too many SQL variables" error
    PAGE = 500
    data: dict = {"embeddings": [], "metadatas": []}
    offset = 0
    while True:
        page = collection.get(include=["embeddings", "metadatas"], limit=PAGE, offset=offset)
        if not page["ids"]:
            break
        data["embeddings"].extend(page["embeddings"])
        data["metadatas"].extend(page["metadatas"])
        if len(page["ids"]) < PAGE:
            break
        offset += PAGE

    embeddings = np.array(data["embeddings"], dtype=np.float32)

    # Save the order of segments (essential for searcher.py
    # to translate a FAISS index into track_id)
    df_order = pd.DataFrame(data["metadatas"])   # columns : track_id, start_s
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_order.to_parquet(order_path, index=False)
    print(f"[build_index] Segments order saved → {order_path}")

    print(f"[build_index] {len(embeddings)} vectors, dim={embeddings.shape[1]}")
    index = build_index(embeddings, index_type=index_type)
    save_index(index, out_path)

    # Final verification
    check_index = load_index(out_path)
    assert check_index.ntotal == len(embeddings), (
        f"Mismatch: {check_index.ntotal} vectors in index vs {len(embeddings)} expected"
    )
    print(f"[build_index] ✓ {check_index.ntotal} vectors indexed → {out_path}")


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import chromadb
    import src.config as config

    parser = argparse.ArgumentParser(description="Builds the FAISS index from ChromaDB.")
    parser.add_argument(
        "--method",
        default=None,
        help="Embedding method to index (mfcc, clap, muq). "
             "If not specified, all available collections are indexed.",
    )
    args = parser.parse_args()

    index_type    = getattr(config, "INDEX_TYPE", "flat")
    chroma_client = chromadb.PersistentClient(path=str(ROOT / config.CHROMA_DIR))

    if args.method:
        # The user passes "clap", "mfcc", "muq" → the collection key is resolved
        collection_keys = [config.get_collection_key(args.method)]
        print(f"[build_index] Collection key resolved: {collection_keys[0]}")
    else:
        # All collections present in ChromaDB
        collection_keys = [c.name for c in chroma_client.list_collections()]
        if not collection_keys:
            print("[build_index] No collections found in ChromaDB.")
            print("[build_index] Run first: python manage.py ingest")
            sys.exit(1)
        print(f"[build_index] Available collections: {collection_keys}")

    for key in collection_keys:
        _build_for_method(key, index_type, chroma_client)

    print("\n[build_index] Done.")
