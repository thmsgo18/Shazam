"""
src/features/fingerprint.py

Fingerprinting audio inspiré de l'algorithme Shazam.
Utilisé comme étape de re-ranking après la recherche FAISS.
Responsable : Personne B
"""

from __future__ import annotations

import numpy as np


def extract_fingerprint(waveform: np.ndarray, sr: int) -> set[tuple]:
    """
    Extrait le fingerprint audio d'un waveform sous forme de constellation map.

    Principe (algorithme Shazam simplifié) :
    1. Calculer le spectrogramme (magnitude STFT).
    2. Détecter les pics locaux dans l'espace temps-fréquence.
    3. Pour chaque paire de pics proches, créer un hash (freq1, freq2, delta_t).

    Args:
        waveform: signal audio mono en float32.
        sr:       taux d'échantillonnage du signal.

    Returns:
        Ensemble de tuples (freq1, freq2, delta_t) représentant le fingerprint.
    """
    ...


def fingerprint_similarity(fp1: set[tuple], fp2: set[tuple]) -> float:
    """
    Calcule la similarité entre deux fingerprints (similarité de Jaccard).

    Score = |fp1 ∩ fp2| / |fp1 ∪ fp2|

    Args:
        fp1: fingerprint du premier audio.
        fp2: fingerprint du second audio.

    Returns:
        Score entre 0.0 (aucune ressemblance) et 1.0 (identiques).
        Retourne 0.0 si l'un des deux ensembles est vide.
    """
    ...
