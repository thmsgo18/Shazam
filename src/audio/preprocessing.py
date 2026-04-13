"""
src/audio/preprocessing.py

Ce module fournit :
  - iter_segments() : découpage d'un waveform en fenêtres chevauchantes
  - preprocess_query() : pipeline de prétraitement de la requête audio
      1. Filtre passe-haut 80 Hz  → coupe grondements micro / HVAC / vent
      2. Normalisation LUFS -14   → aligne le niveau sur la base de données
      3. Peak normalization        → sécurité anti-saturation
"""
from src.config import (
    SEGMENT_WIN_S,
    SEGMENT_HOP_S,
    SEGMENT_MIN_WIN
)

import numpy as np
from typing import Iterator, Tuple

def iter_segments(waveform: np.ndarray, sr: int, win_s: float = SEGMENT_WIN_S, hop_s: float = SEGMENT_HOP_S, min_win: float = SEGMENT_MIN_WIN)-> Iterator[Tuple[float, np.ndarray]]:
    """
    Iterate over fixed-size audio segments with optional overlap.
    This function splits an audio waveform into segments of a given window size, moving forward using a hop size. If the final remaining audio is long enough.

    Args:
        waveform (np.ndarray): Input audio waveform.
        sr (int): Sampling rate of the waveform.
        win_s (float, optional): Segment window size in seconds.
        hop_s (float, optional): Step size between consecutive segments in seconds.
        min_win (float, optional): Minimum fraction of window size required to keep the last segment.

    Yields:
            - Segment start time in seconds.
            - Audio segment waveform.

    Raises:
        ValueError:
            If window size, hop size, or sampling rate are invalid.
    """
    if win_s <= 0:
        raise ValueError("win_s <= 0")
    if hop_s <= 0:
        raise ValueError("hop_s <= 0")
    if sr <= 0:
        raise ValueError("sr <= 0")
    
    # Convert window and hop sizes from seconds to samples :
    size_win = int(win_s * sr)
    size_hop = int(hop_s * sr)

    size_waveform = len(waveform)                           # Total number of samples in waveform.
    start = 0
    while start + size_win <= size_waveform:                # Generate full-size segments while possible
        yield start / sr, waveform[start:start + size_win]  # Yield segment start time (seconds) and waveform slice.
        start += size_hop

    remaining = size_waveform - start                       # Handle remaining audio at the end of the waveform.

    if remaining >= min_win * size_win and remaining > 0:   # If remaining audio is long enough, pad it and return as final segment.
        segment = np.pad(waveform[start:], (0, size_win - remaining))
        yield start / sr, segment


def preprocess_query(waveform: np.ndarray, sr: int) -> np.ndarray:
    """
    Prétraitement de la requête audio avant embedding + fingerprinting.

    Étapes (dans l'ordre) :
      1. Filtre passe-haut Butterworth ordre 4 à 80 Hz
         → supprime grondements micro, HVAC, vent
         → la littérature (ISMIR 2025) montre que descendre à 80 Hz
           améliore la robustesse en environnement réverbérant
      2. Normalisation LUFS à -14 LUFS (standard streaming)
         → CLAP est entraîné sur des audios normalisés ; les écarts de
           volume dégradent directement la qualité des embeddings
      3. Peak normalization à 0.95
         → sécurité anti-saturation numérique

    Args:
        waveform: signal audio mono, float32, valeurs dans [-1, 1]
        sr:       fréquence d'échantillonnage en Hz

    Returns:
        waveform prétraité, même shape et dtype que l'entrée
    """
    from scipy import signal as scipy_signal
    import pyloudnorm as pyln
    import src.config as config

    audio = waveform.astype(np.float32)

    # 1. Filtre passe-haut 80 Hz
    sos = scipy_signal.butter(4, 80.0, btype="highpass", fs=sr, output="sos")
    audio = scipy_signal.sosfilt(sos, audio).astype(np.float32)

    # 2. Débruitage spectral (optionnel, contrôlé par OPT_QUERY_DENOISE)
    #    Mode non-stationnaire : adapté aux bruits variables (café, rue)
    #    n_fft=2048  : résolution temps-fréquence adaptée à la musique (vs 512 pour la voix)
    #    prop_decrease=0.75 : débruitage partiel → moins d'artefacts musicaux
    if config.OPT_QUERY_DENOISE:
        import noisereduce as nr
        audio = nr.reduce_noise(
            y=audio,
            sr=sr,
            stationary=False,
            prop_decrease=0.75,
            n_fft=2048,
            n_jobs=-1,
        ).astype(np.float32)

    # 3. Normalisation LUFS (-14 LUFS)
    #    Le warning "clipped samples" de pyloudnorm est inoffensif : la peak norm
    #    (étape 4) rattrape toute saturation introduite ici.
    import warnings as _warnings
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio.astype(np.float64))
    if not np.isinf(loudness) and not np.isnan(loudness):
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            audio = pyln.normalize.loudness(
                audio.astype(np.float64), loudness, -14.0
            ).astype(np.float32)

    # 4. Peak normalization (sécurité anti-saturation)
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.95

    return audio
