from src.config import (
    SEGMENT_WIN_S,
    SEGMENT_HOP_S,
    SEGMENT_MIN_WIN
)

import numpy as np
from typing import Iterator, Tuple

def iter_segments( waveform: np.ndarray, sr: int, win_s: float = SEGMENT_WIN_S, hop_s: float = SEGMENT_HOP_S, min_win: float = SEGMENT_MIN_WIN)-> Iterator[Tuple[float, np.ndarray]]:
    if win_s <= 0:
        raise ValueError("win_s <= 0")
    if hop_s <= 0:
        raise ValueError("hop_s <= 0")
    if sr <= 0:
        raise ValueError("sr <= 0")
    
    size_win = int(win_s * sr)
    size_hop = int(hop_s * sr)

    size_waveform = len(waveform)
    start = 0
    while start + size_win <= size_waveform:
        yield start / sr, waveform[start:start + size_win]
        start += size_hop

    remaining = size_waveform - start
    if remaining >= min_win * size_win and remaining > 0:
        segment = np.pad(waveform[start:], (0, size_win - remaining))
        yield start / sr, segment
