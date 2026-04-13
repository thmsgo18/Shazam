"""
src/features/fingerprint.py

Audio fingerprinting inspired by the Shazam algorithm.
Used as a re-ranking step after FAISS research.
"""

from __future__ import annotations

import numpy as np
import librosa
from scipy.ndimage import maximum_filter
import src.config as config


def extract_fingerprint(waveform: np.ndarray, sr: int) -> set[tuple]:
    """
    Extracts the audio fingerprint (v2) from a waveform in the form of a constellation map.

    Principle (Shazam algorithm with temporal anchor):
    1. Calculate the spectrogram (STFT magnitude).
    2. Detect local peaks in time-frequency space.
    3. For each pair of nearby peaks, create a hash (freq1, freq2, delta_t, t1_anchor).
       The anchor time t1 enables temporal alignment during matching (offset histogram),
       which filters out coincidental hash matches and greatly improves precision on noisy audio.

    Args:
        waveform: mono audio signal in float32.
        sr:       signal sampling rate.

    Returns:
        Set of 4-tuples (freq1, freq2, delta_t, t1_anchor) — format v2.
    """
    # 1. Calculation of the spectrogram
    stft = librosa.stft(waveform, n_fft=config.FP_N_FFT, hop_length=config.FP_HOP_LENGTH)
    spectrogram = np.abs(stft)

    # 2. Detection of local peaks
    local_max = (maximum_filter(spectrogram, size=config.FP_NEIGHBORHOOD) == spectrogram)

    threshold = np.percentile(spectrogram, config.FP_THRESHOLD_PERCENTILE)
    peaks_mask = local_max & (spectrogram > threshold)

    freq_bins, time_frames = np.where(peaks_mask)
    peaks = sorted(zip(time_frames, freq_bins))

    # 3. Generation of composite hashes (v2 : inclut l'ancre temporelle t1)
    hashes = set()

    for i, (t1, f1) in enumerate(peaks):
        connections = 0
        target_reached = False
        out_of_zone = False
        j = 1

        while j < len(peaks) - i and not target_reached and not out_of_zone:
            t2, f2 = peaks[i + j]
            delta_t = t2 - t1

            if config.FP_MIN_DELTA_T <= delta_t <= config.FP_MAX_DELTA_T:
                # t1 inclus → alignement temporel possible lors du matching
                hashes.add((int(f1), int(f2), int(delta_t), int(t1)))
                connections += 1
                if connections >= config.FP_FAN_OUT:
                    target_reached = True

            elif delta_t > config.FP_MAX_DELTA_T:
                out_of_zone = True

            j += 1

    return hashes




def fingerprint_similarity(fp_query: set[tuple], fp_candidate: set[tuple]) -> float:
    """
    Calculates the similarity between two fingerprints using temporal alignment.

    Format v2 (4-tuples : f1, f2, delta_t, t1_anchor) :
        Pour chaque hash (f1, f2, delta_t) commun entre la requête et le candidat,
        on calcule l'offset temporel : offset = t1_candidate - t1_query.
        Les vrais matches s'accumulent tous au même offset (la position du clip dans
        le morceau). Les faux positifs ont des offsets aléatoires → ne s'accumulent pas.
        Score = pic de l'histogramme des offsets / |fp_query|

    Format v1 (3-tuples, rétrocompatibilité) :
        Score = |fp_query ∩ fp_candidate| / |fp_query|  (simple recall)

    Args:
        fp_query:     fingerprint de l'extrait requête.
        fp_candidate: fingerprint du morceau candidat (base de données).

    Returns:
        Score entre 0.0 et 1.0. Retourne 0.0 si l'un des fingerprints est vide.
    """
    if not fp_query or not fp_candidate:
        return 0.0

    # Détection du format v1 / v2
    sample = next(iter(fp_query))
    if len(sample) == 3:
        # v1 — rétrocompatibilité : simple intersection
        return len(fp_query.intersection(fp_candidate)) / len(fp_query)

    # v2 — alignement temporel via histogramme d'offsets
    # Construit un index {(f1,f2,delta_t): [t1_db, ...]} pour le candidat
    candidate_index: dict[tuple, list[int]] = {}
    for (f1, f2, dt, t1_db) in fp_candidate:
        key = (f1, f2, dt)
        if key not in candidate_index:
            candidate_index[key] = []
        candidate_index[key].append(t1_db)

    # Histogramme des offsets pour les hashes communs
    offsets: dict[int, int] = {}
    for (f1, f2, dt, t1_query) in fp_query:
        key = (f1, f2, dt)
        if key in candidate_index:
            for t1_db in candidate_index[key]:
                offset = t1_db - t1_query
                offsets[offset] = offsets.get(offset, 0) + 1

    if not offsets:
        return 0.0

    # Score = nombre de matches au meilleur offset / taille de la requête
    return max(offsets.values()) / len(fp_query)



################# TESTS #################
'''
if __name__ == "__main__":
    
    sr = 22050
    dummy_waveform = np.random.randn(sr * 3).astype(np.float32) # 3 secondes
    
    fp = extract_fingerprint(dummy_waveform, sr)
    print(f"Number of hashes generated: {len(fp)}")
    
    # Test of similarity
    score = fingerprint_similarity(fp, fp)
    print(f"Similarity on the same clip (expected 1.0) : {score}")
'''