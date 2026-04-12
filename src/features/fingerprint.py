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
    Extracts the audio fingerprint from a waveform in the form of a constellation map. 

    Principle (simplified Shazam algorithm): 
    1. Calculate the spectrogram (STFT magnitude). 
    2. Detect local peaks in time-frequency space. 
    3. For each pair of nearby peaks, create a hash (freq1, freq2, delta_t).

    Args:
        waveform: mono audio signal in float32.
        sr:       signal sampling rate.

    Returns:
        Set of tuples (freq1, freq2, delta_t) representing the fingerprint.
    """
    # 1. Calculation of the spectrogram
    stft = librosa.stft(waveform, n_fft=config.FP_N_FFT, hop_length=config.FP_HOP_LENGTH)
    spectrogram = np.abs(stft)

    # 2. DDetection of local peaks
    local_max = (maximum_filter(spectrogram, size=config.FP_NEIGHBORHOOD) == spectrogram)

    threshold = np.percentile(spectrogram, config.FP_THRESHOLD_PERCENTILE)
    peaks_mask = local_max & (spectrogram > threshold)

    freq_bins, time_frames = np.where(peaks_mask)
    peaks = sorted(zip(time_frames, freq_bins))

    # 3. Generation of composite hashes
    hashes = set()

    for i, (t1, f1) in enumerate(peaks):
        connections = 0
        target_reached = False
        out_of_zone = False
        j=1

        while j < len(peaks) - i and not target_reached and not out_of_zone:
            t2, f2 = peaks[i + j]
            delta_t = t2 - t1

            # if the peak is in the target area
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
    Calculates the similarity between two fingerprints (recall-based). 

    Score = |fp_query ∩ fp_candidate| / |fp_query|

    We measure what fraction of the hashes of the query (short extract) are found 
    in the candidate (whole piece). Unlike Jaccard, this score is not 
    penalized by the fact that the base piece is longer than the query.

    Args:
        fp_query:     fingerprint of the query extract.
        fp_candidate: fingerprint of the candidate piece.

    Returns:
        Score between 0.0 (no similarity) and 1.0 (all hashes found).
        Returns 0.0 if either set is empty.
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
    print(f"Number of hashes generated: {len(fp)}")
    
    # Test of similarity
    score = fingerprint_similarity(fp, fp)
    print(f"Similarity on the same clip (expected 1.0) : {score}")
'''