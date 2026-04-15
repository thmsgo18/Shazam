"""
src/index/build_index.py

Construction et gestion de l'index FAISS.
Supporte aussi HNSW et IVF.
Responsable : Personne A
"""

from __future__ import annotations

from pathlib import Path

import faiss

ROOT = Path(__file__).resolve().parents[2]
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


def _build_for_method(collection_key: str, index_type: str, chroma_client) -> None:
    """
    Construit et sauvegarde l'index FAISS pour une collection ChromaDB donnée.

    Args:
        collection_key: clé méthode+modèle, ex. "clap_larger_clap_music" ou "mfcc".
                        C'est aussi le nom de la collection ChromaDB.
    """
    import src.config as config

    index_dir  = ROOT / config.INDEX_DIR
    out_path   = index_dir / f"index_{collection_key}_{index_type}.faiss"
    order_path = index_dir / f"segments_{collection_key}.parquet"

    try:
        collection = chroma_client.get_collection(name=collection_key)
    except Exception:
        print(f"[build_index] Collection '{collection_key}' introuvable dans ChromaDB — ignorée.")
        return

    n = collection.count()
    if n == 0:
        print(f"[build_index] La collection '{collection_key}' est vide — ignorée.")
        return

    print(f"\n[build_index] ── Collection : {collection_key} | {n} segments ──")

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

    print(f"[build_index] {len(embeddings)} vecteurs, dim={embeddings.shape[1]}")
    index = build_index(embeddings, index_type=index_type)
    save_index(index, out_path)

    # Vérification finale
    check_index = load_index(out_path)
    assert check_index.ntotal == len(embeddings), (
        f"Mismatch : {check_index.ntotal} vecteurs dans l'index vs {len(embeddings)} attendus"
    )
    print(f"[build_index] ✓ {check_index.ntotal} vecteurs indexés → {out_path}")


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import chromadb
    import src.config as config

    parser = argparse.ArgumentParser(description="Construit l'index FAISS depuis ChromaDB.")
    parser.add_argument(
        "--method",
        default=None,
        help="Méthode d'embedding à indexer (mfcc, clap, muq). "
             "Si non spécifié, toutes les collections disponibles sont indexées.",
    )
    args = parser.parse_args()

    index_type    = getattr(config, "INDEX_TYPE", "flat")
    chroma_client = chromadb.PersistentClient(path=str(ROOT / config.CHROMA_DIR))

    if args.method:
        # L'utilisateur passe "clap", "mfcc", "muq" → on résout la clé collection
        collection_keys = [config.get_collection_key(args.method)]
        print(f"[build_index] Clé collection résolue : {collection_keys[0]}")
    else:
        # Toutes les collections présentes dans ChromaDB
        collection_keys = [c.name for c in chroma_client.list_collections()]
        if not collection_keys:
            print("[build_index] Aucune collection trouvée dans ChromaDB.")
            print("[build_index] Lance d'abord : python manage.py ingest")
            sys.exit(1)
        print(f"[build_index] Collections disponibles : {collection_keys}")

    for key in collection_keys:
        _build_for_method(key, index_type, chroma_client)

    print("\n[build_index] Terminé.")
