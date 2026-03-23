"""
src/index/build_index.py

Construction et gestion de l'index FAISS.
Responsable : Personne A
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pandas as pd


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Construit un index FAISS à partir d'une matrice d'embeddings.

    Args:
        embeddings: matrice de shape (N, D) en float32.
                    Les vecteurs doivent être normalisés en L2.

    Returns:
        Index FAISS prêt à la recherche.
    """
    
    xb = embeddings.astype("float32") # Conversion en float32 pour FAISS

    # Normalisation des vecteurs en L2 pour similarité cosine
    faiss.normalize_L2(xb)

    # On crée l'index FlatIP
    d = xb.shape[1] # Dim de l'embedding, permet de changer de méthode sans problèmes de dim.
    index = faiss.IndexFlatIP(d)

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
    import src.config as config

    method    = config.EMBEDDING_METHOD
    emb_path  = Path(f"{config.FEATURES_DIR}/embeddings_{method}.npy")
    seg_path  = Path(f"{config.FEATURES_DIR}/segments_{method}.parquet")
    out_path  = Path(f"{config.INDEX_DIR}/index_{method}.faiss")

    if not emb_path.exists() or not seg_path.exists() : 
        print(f"Error : the files for {method} are not found.")
        sys.exit(1)
    
    embeddings = np.load(emb_path)
    segments   = pd.read_parquet(seg_path)

    # Vérifie si ligne i de embeddings.npy = ligne i de segments.parquet
    assert len(embeddings) == len(segments), (
        f"Mismatch : {len(embeddings)} embeddings vs {len(segments)} segments"
    )

    print(f"[build_index] {len(embeddings)} vecteurs, dim={embeddings.shape[1]}")

    index = build_index(embeddings)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_index(index, out_path)

    print(f"[build_index] Index sauvegardé → {out_path}")

    # On vérifie que index.ntotal est égal au nb de ligne du .parquet
    check_index = load_index(out_path)
    print(f"Total number of vectors in the index : {check_index.ntotal}")
    assert check_index.ntotal == len(segments)
    print("Verification completed.")