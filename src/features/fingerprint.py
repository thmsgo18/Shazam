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
                # t1 included → temporal alignment possible during matching
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
        For each common hash (f1, f2, delta_t) between the query and the candidate,
        we calculate the temporal offset: offset = t1_candidate - t1_query.
        True matches all accumulate at the same offset (the position of the clip in
        the track). False positives have random offsets → do not accumulate.
        Score = peak of the offsets histogram / |fp_query|

    Format v1 (3-tuples, retrocompatibility) :
        Score = |fp_query ∩ fp_candidate| / |fp_query|  (simple recall)

    Args:
        fp_query:     fingerprint of the query excerpt.
        fp_candidate: fingerprint of the candidate track (database).

    Returns:
        Score between 0.0 and 1.0. Returns 0.0 if one of the fingerprints is empty.
    """
    if not fp_query or not fp_candidate:
        return 0.0

    # Detection of v1 / v2 format
    sample = next(iter(fp_query))
    if len(sample) == 3:
        # v1 — retrocompatibility: simple intersection
        return len(fp_query.intersection(fp_candidate)) / len(fp_query)

    # v2 — temporal alignment via offsets histogram
    # Build an index {(f1,f2,delta_t): [t1_db, ...]} for the candidate
    candidate_index: dict[tuple, list[int]] = {}
    for (f1, f2, dt, t1_db) in fp_candidate:
        key = (f1, f2, dt)
        if key not in candidate_index:
            candidate_index[key] = []
        candidate_index[key].append(t1_db)

    # Histogram of offsets for common hashes
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