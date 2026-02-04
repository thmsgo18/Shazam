"""
src/audio/preprocessing.py

This module provides utilities to split an audio waveform into overlapping time segments.
It is used before feature extraction or embedding computation to process long audio files in smaller chunks.
"""
from src.config import (
    SEGMENT_WIN_S,
    SEGMENT_HOP_S,
    SEGMENT_MIN_WIN
)

import numpy as np
from typing import Iterator, Tuple

def iter_segments( waveform: np.ndarray, sr: int, win_s: float = SEGMENT_WIN_S, hop_s: float = SEGMENT_HOP_S, min_win: float = SEGMENT_MIN_WIN)-> Iterator[Tuple[float, np.ndarray]]:
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
