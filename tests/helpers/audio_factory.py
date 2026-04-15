from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


def sine_wave(
    frequency: float = 440.0,
    sr: int = 22050,
    duration_s: float = 1.0,
    amplitude: float = 0.5,
) -> np.ndarray:
    timeline = np.linspace(0.0, duration_s, int(sr * duration_s), endpoint=False)
    waveform = amplitude * np.sin(2 * np.pi * frequency * timeline)
    return waveform.astype(np.float32)


def noise_wave(sr: int = 22050, duration_s: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(sr * duration_s)).astype(np.float32)


def mixed_wave(sr: int = 22050, duration_s: float = 1.0) -> np.ndarray:
    waveform = sine_wave(220.0, sr=sr, duration_s=duration_s, amplitude=0.35)
    waveform += sine_wave(440.0, sr=sr, duration_s=duration_s, amplitude=0.25)
    waveform += sine_wave(880.0, sr=sr, duration_s=duration_s, amplitude=0.15)
    return waveform.astype(np.float32)


def write_wav(path: str | Path, waveform: np.ndarray, sr: int = 22050) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.asarray(waveform, dtype=np.float32)
    if data.ndim == 2 and data.shape[0] <= 8:
        data = data.T

    sf.write(str(output_path), data, sr)
    return output_path


def make_temp_wav(waveform: np.ndarray, sr: int = 22050, suffix: str = ".wav") -> Path:
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    handle.close()
    return write_wav(handle.name, waveform, sr=sr)
