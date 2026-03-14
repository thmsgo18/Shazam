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
    n_fft = 2048
    hop_length = 512
    stft = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(stft)

    # 2. Détection des pics locaux
    neighborhood = (20, 15) 
    local_max = (maximum_filter(spectrogram, size=neighborhood) == spectrogram)

    threshold = np.percentile(spectrogram, 80)
    peaks_mask = local_max & (spectrogram > threshold)

    freq_bins, time_frames = np.where(peaks_mask)
    peaks = sorted(zip(time_frames, freq_bins))

    # 3. Génération des hashes combinatoires
    hashes = set()
    fan_out = 5 # nb max de paires par pic
    min_delta_t = 3 # ignorer les paires trop proches (en frames)
    max_delta_t = 50 # borne max de la target zone (en frames)

    for i, (t1, f1) in enumerate(peaks):
        connections = 0
        target_reached = False
        out_of_zone = False
        j=1

        while j < len(peaks) - i and not target_reached and not out_of_zone:
            t2, f2 = peaks[i + j]
            delta_t = t2 - t1
            
            # si le pic est dans la zone visée
            if min_delta_t <= delta_t <= max_delta_t:
                hashes.add((int(f1), int(f2), int(delta_t)))
                connections += 1
                if connections >= fan_out:
                    target_reached = True
            
            elif delta_t > max_delta_t:
                out_of_zone = True

            j += 1

    return hashes




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
    # si l'un des deux fingerprints est vide, le score est forcément 0
    if not fp1 or not fp2:
        return 0.0
    
    # Calcul de l'intersection et de l'union
    intersection = fp1.intersection(fp2)
    union = fp1.union(fp2)

    if len(union) == 0:
        return 0.0

    return len(intersection) / len(union)



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