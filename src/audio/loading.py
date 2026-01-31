import numpy as np
import librosa
from src.config import SAMPLE_RATE

def load_audio(path: str, target_sr: int = SAMPLE_RATE, mono: bool = True) -> tuple[list, int]:
    waveform, sr = librosa.load(path, target_sr, mono)
    return waveform, sr