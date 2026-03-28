"""
src/features/fingerprint.py

Fingerprinting audio inspiré de l'algorithme Shazam.
Utilisé comme étape de re-ranking après la recherche FAISS.
Responsable : Personne B
"""

from __future__ import annotations

import numpy as np
import librosa
from scipy.ndimage import maximum_filter
import src.config as config


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
    # 1. Calcul du spectrogramme
    stft = librosa.stft(waveform, n_fft=config.FP_N_FFT, hop_length=config.FP_HOP_LENGTH)
    spectrogram = np.abs(stft)

    # 2. Détection des pics locaux
    local_max = (maximum_filter(spectrogram, size=config.FP_NEIGHBORHOOD) == spectrogram)

    threshold = np.percentile(spectrogram, config.FP_THRESHOLD_PERCENTILE)
    peaks_mask = local_max & (spectrogram > threshold)

    freq_bins, time_frames = np.where(peaks_mask)
    peaks = sorted(zip(time_frames, freq_bins))

    # 3. Génération des hashes combinatoires
    hashes = set()

    for i, (t1, f1) in enumerate(peaks):
        connections = 0
        target_reached = False
        out_of_zone = False
        j=1

        while j < len(peaks) - i and not target_reached and not out_of_zone:
            t2, f2 = peaks[i + j]
            delta_t = t2 - t1

            # si le pic est dans la zone visée
            if config.FP_MIN_DELTA_T <= delta_t <= config.FP_MAX_DELTA_T:
                hashes.add((int(f1), int(f2), int(delta_t)))
                connections += 1
                if connections >= config.FP_FAN_OUT:
                    target_reached = True

            elif delta_t > config.FP_MAX_DELTA_T:
                out_of_zone = True

            j += 1

    return hashes




def fingerprint_similarity(fp_query: set[tuple], fp_candidate: set[tuple]) -> float:
    """
    Calcule la similarité entre deux fingerprints (recall-based).

    Score = |fp_query ∩ fp_candidate| / |fp_query|

    On mesure quelle fraction des hashes de la requête (extrait court) se retrouve
    dans le candidat (morceau entier). Contrairement à Jaccard, ce score n'est pas
    pénalisé par le fait que le morceau en base est plus long que la requête.

    Args:
        fp_query:     fingerprint de l'extrait requête.
        fp_candidate: fingerprint du morceau en base.

    Returns:
        Score entre 0.0 (aucune ressemblance) et 1.0 (tous les hashes retrouvés).
        Retourne 0.0 si l'un des deux ensembles est vide.
    """
    if not fp_query or not fp_candidate:
        return 0.0

    return len(fp_query.intersection(fp_candidate)) / len(fp_query)



################# TESTS #################
'''
if __name__ == "__main__":
    
    sr = 22050
    dummy_waveform = np.random.randn(sr * 3).astype(np.float32) # 3 secondes
    
    fp = extract_fingerprint(dummy_waveform, sr)
    print(f"Nombre de hashes générés: {len(fp)}")
    
    # Test de similarité
    score = fingerprint_similarity(fp, fp)
    print(f"Similarité sur le même morceau (attendu 1.0) : {score}")
'''