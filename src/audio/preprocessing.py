"""
src/audio/preprocessing.py

This module provides:

    - iter_segments(): slices a waveform into overlapping windows
    - preprocess_query(): preprocesses the audio query

    1. 80 Hz high-pass filter → cuts mic/HVAC/wind noise
    2. LUFS -14 normalization → aligns the level with the database
    3. Peak normalization → prevents clipping
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
    - Segment waveform audio. 

    Reasons: 
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
    Audio request preprocessing before embedding and fingerprinting.

    Steps (in order):

    1. Butterworth high-pass filter, order 4, at 80 Hz
        → removes mic rumble, HVAC noise, wind noise
        → literature (ISMIR 2025) shows that going down to 80 Hz
        improves robustness in reverberant environments

    2. LUFS normalization to -14 LUFS (standard streaming)
        → CLAP is trained on normalized audio; volume deviations directly degrade embedding quality

    3. Peak normalization to 0.95
        → digital clipping protection

    Args:
    waveform: mono audio signal, float32, values ​​in [-1, 1]
    sr: sampling frequency in Hz

    Returns:
    preprocessed waveform, same shape and dtype as the input
    """
    from scipy import signal as scipy_signal
    import pyloudnorm as pyln
    import src.config as config

    audio = waveform.astype(np.float32)

    # 1. 80 Hz High-Pass Filter
    sos = scipy_signal.butter(4, 80.0, btype="highpass", fs=sr, output="sos")
    audio = scipy_signal.sosfilt(sos, audio).astype(np.float32)

    # 2. Spectral noise reduction (optional, controlled by OPT_QUERY_DENOISE)
    #    Non-stationary mode: suitable for varying noise levels (cafe, street)
    #    n_fft=2048: time-frequency resolution suitable for music (vs. 512 for voice)
    #    prop_decrease=0.75: partial noise reduction → fewer musical artifacts
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

    # 3. LUFS normalization (-14 LUFS)
    #   The "clipped samples" warning from pyloudnorm is harmless: the peak norm
    #   (step 4) compensates for any saturation introduced here.
    import warnings as _warnings
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio.astype(np.float64))
    if not np.isinf(loudness) and not np.isnan(loudness):
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            audio = pyln.normalize.loudness(
                audio.astype(np.float64), loudness, -14.0
            ).astype(np.float32)

    # 4. Peak normalization (anti-saturation safety)
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.95

    return audio
