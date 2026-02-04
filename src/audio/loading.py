"""
src/audio/loading.py

This module provides helper functions to load audio files in a standardized way
across the project. It ensures consistent sampling rate and channel handling
for downstream audio processing and embedding extraction.
"""

import numpy as np
import librosa
from src.config import SAMPLE_RATE

def load_audio(path: str, target_sr: int = SAMPLE_RATE, mono: bool = True) -> tuple[list, int]:
    waveform, sr = librosa.load(path, target_sr, mono)
    return waveform, sr