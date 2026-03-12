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
    ...


def save_index(index: faiss.Index, path: Path) -> None:
    """
    Sauvegarde un index FAISS sur disque.

    Args:
        index: index FAISS à sauvegarder.
        path:  chemin de destination (ex: data/index/index_mfcc.faiss).
    """
    ...


def load_index(path: Path) -> faiss.Index:
    """
    Charge un index FAISS depuis le disque.

    Args:
        path: chemin vers le fichier .faiss.

    Returns:
        Index FAISS chargé.
    """
    ...


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import src.config as config

    method    = config.EMBEDDING_METHOD
    emb_path  = Path(f"data/features/embeddings_{method}.npy")
    seg_path  = Path(f"data/features/segments_{method}.parquet")
    out_path  = Path(f"data/index/index_{method}.faiss")

    embeddings = np.load(emb_path)
    segments   = pd.read_parquet(seg_path)

    assert len(embeddings) == len(segments), (
        f"Mismatch : {len(embeddings)} embeddings vs {len(segments)} segments"
    )

    print(f"[build_index] {len(embeddings)} vecteurs, dim={embeddings.shape[1]}")

    index = build_index(embeddings)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_index(index, out_path)

    print(f"[build_index] Index sauvegardé → {out_path}")
