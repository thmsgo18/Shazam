"""
src/index/build_index.py

Construction et gestion de l'index FAISS.
Supporte aussi HNSW et IVF.
Responsable : Personne A
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pandas as pd


def build_index(embeddings: np.ndarray, index_type: str = "flat") -> faiss.Index:
    """
    Construit un index FAISS, selon le type choisi, à partir d'une matrice d'embeddings.

    Args:
        embeddings: matrice de shape (N, D) en float32.
                    Les vecteurs doivent être normalisés en L2.
        index_type: "flat", "hnsw" ou "ivf"

    Returns:
        Index FAISS prêt à la recherche (normalisé L2, similarité cosinus).
    """
    
    xb = embeddings.astype("float32") # Conversion en float32 pour FAISS

    # Normalisation des vecteurs en L2 pour similarité cosine
    faiss.normalize_L2(xb)

    # On crée l'index FlatIP
    d = xb.shape[1] # Dim de l'embedding, permet de changer de méthode sans problèmes de dim.
    N = len(xb)

    index_type = index_type.lower()

    if index_type == "flat" : 
        index = faiss.IndexFlatIP(d)
    elif index_type == "hnsw" : 
        M = 32 # M : connectivité des graphes
        index = faiss.IndexHNSWFlat(d, M)
        index.hnsw.efConstruction = 40 # tuning pour la précision
    elif index_type == "ivf" :
        nbList = max(1, int(np.sqrt(N))) # Nb de listes inversées : racine carrée du nb de vecteurs
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nbList, faiss.METRIC_INNER_PRODUCT)
        index.train(xb)
    else : 
        raise ValueError(f"Index type unknwon : {index_type}. Possible choices : falt, hnsw, ivf") 

    # Ajout des vecteurs
    index.add(xb)

    return index


def save_index(index: faiss.Index, path: Path) -> None:
    """
    Sauvegarde un index FAISS sur disque.

    Args:
        index: index FAISS à sauvegarder.
        path:  chemin de destination (ex: data/index/index_mfcc.faiss).
    """
    
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    print(f"Index saved successfully in : {path}")


def load_index(path: Path) -> faiss.Index:
    """
    Charge un index FAISS depuis le disque.

    Args:
        path: chemin vers le fichier .faiss.

    Returns:
        Index FAISS chargé.
    """
    
    if not path.exists() : 
        raise FileNotFoundError(f"The index doesn't exist at : {path}")
    
    return faiss.read_index(str(path))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import chromadb
    import src.config as config

    method     = config.EMBEDDING_METHOD
    index_type = getattr(config, "INDEX_TYPE", "flat")
    out_path   = Path(f"{config.INDEX_DIR}/index_{method}_{index_type}.faiss")
    order_path = Path(f"{config.INDEX_DIR}/segments_{method}.parquet")

    # Charger les embeddings et métadonnées depuis ChromaDB
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    try:
        collection = chroma_client.get_collection(name=method)
    except Exception:
        print(f"[build_index] Collection '{method}' introuvable dans ChromaDB.")
        print(f"[build_index] Lance d'abord : python scripts/download_music.py")
        sys.exit(1)

    n = collection.count()
    if n == 0:
        print(f"[build_index] La collection '{method}' est vide — rien à indexer.")
        sys.exit(1)

    print(f"[build_index] Chargement de {n} segments depuis ChromaDB…")

    # Pagination pour éviter l'erreur SQLite "too many SQL variables"
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

    # Sauvegarder l'ordre des segments (indispensable pour que searcher.py
    # puisse traduire un indice FAISS en track_id)
    df_order = pd.DataFrame(data["metadatas"])   # colonnes : track_id, start_s
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_order.to_parquet(order_path, index=False)
    print(f"[build_index] Ordre des segments sauvegardé → {order_path}")

    print(f"[build_index] Méthode = {method} | Index type = {index_type}")
    print(f"[build_index] {len(embeddings)} vecteurs, dim={embeddings.shape[1]}")

    index = build_index(embeddings, index_type=index_type)
    save_index(index, out_path)
    print(f"[build_index] Index sauvegardé → {out_path}")

    # Vérification finale
    check_index = load_index(out_path)
    assert check_index.ntotal == len(embeddings), (
        f"Mismatch : {check_index.ntotal} vecteurs dans l'index vs {len(embeddings)} attendus"
    )
    print(f"[build_index] Vérification OK — {check_index.ntotal} vecteurs dans l'index.")