"""
scripts/evaluate.py

Évaluation du pipeline d'identification audio.
Compare les méthodes d'embedding sur des requêtes de test avec et sans bruit.
Responsable : Personne D

Usage :
    python scripts/evaluate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def generate_test_queries(
    metadata_path: str,
    n_queries_per_track: int = 3,
) -> list[dict]:
    """
    Génère des requêtes de test à partir du dataset.

    Pour chaque morceau, charge l'audio, segmente et sélectionne
    n_queries_per_track segments aléatoirement.

    Args:
        metadata_path:       chemin vers data/processed/metadata.parquet.
        n_queries_per_track: nombre de segments à extraire par morceau.

    Returns:
        Liste de dicts avec les clés : "track_id", "waveform", "sr", "start_s".

    Dépendances :
        - src.audio.loading.load_audio()
        - src.audio.preprocessing.iter_segments()
    """
    ...


def add_noise(waveform: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """
    Ajoute du bruit gaussien à un signal audio.

    Simule un enregistrement au micro dans un environnement bruité.
    Plus snr_db est faible, plus le bruit est intense.

    Args:
        waveform: signal audio mono en float32.
        snr_db:   rapport signal/bruit en décibels (20 dB = bruit léger).

    Returns:
        Signal bruité en float32.
    """
    ...


def evaluate(
    method: str,
    n_queries_per_track: int = 3,
    with_noise: bool = False,
) -> dict:
    """
    Évalue le pipeline d'identification pour une méthode d'embedding donnée.

    Pour chaque requête générée, appelle identify_track() et vérifie si
    le bon morceau est retrouvé en Top-1 et Top-5.

    Args:
        method:              méthode d'embedding à évaluer ("mfcc", "clap").
        n_queries_per_track: nombre de requêtes par morceau.
        with_noise:          si True, ajoute du bruit aux requêtes (snr_db=20).

    Returns:
        Dict avec les clés :
          - "method"         : nom de la méthode
          - "n_queries"      : nombre total de requêtes
          - "top1_accuracy"  : fraction des requêtes dont le Top-1 est correct
          - "top5_accuracy"  : fraction des requêtes dont le Top-5 contient le bon morceau
          - "with_noise"     : booléen indiquant si le test était bruité

    Dépendances :
        - src.retrieval.query_pipeline.identify_track()
    """
    ...


if __name__ == "__main__":
    results = []

    for method in ["mfcc", "clap"]:
        for with_noise in [False, True]:
            r = evaluate(method, n_queries_per_track=5, with_noise=with_noise)
            results.append(r)

    df = pd.DataFrame(results)
    print("\n=== Résultats d'évaluation ===")
    print(df.to_string(index=False))
