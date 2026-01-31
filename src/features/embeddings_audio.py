import numpy as np
import librosa
import torch
from transformers import ClapModel, ClapProcessor

def mfcc_stats_embedding(waveform: np.ndarray, sr: int, n_mfcc: int = 20, n_fft: int = 2048, hop_length: int = 512, normalize: bool = True, eps: float = 1e-10) -> np.ndarray:
    if waveform is None or len(waveform) == 0:
        return np.zeros((2 * n_mfcc,), dtype=np.float32)

    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

    mean = mfcc.mean(axis=1)
    std = mfcc.std(axis=1)

    embedding = np.concatenate([mean, std]).astype(np.float32)

    if normalize:
        norm = np.linalg.norm(embedding)
        embedding = embedding / max(norm, eps)

    return embedding


_CLAP_CACHE = {"model": None, "processor": None, "device": None, "model_name": None}

def _load_clap(model_name: str, device: str | None = None):
    if _CLAP_CACHE["model"] is not None and _CLAP_CACHE["model_name"] == model_name:
        return _CLAP_CACHE["model"], _CLAP_CACHE["processor"], _CLAP_CACHE["device"]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = ClapProcessor.from_pretrained(model_name)
    model = ClapModel.from_pretrained(model_name).to(device)
    model.eval()

    _CLAP_CACHE.update({"model": model, "processor": processor, "device": device, "model_name": model_name})
    return model, processor, device

def clap_embedding(waveform: np.ndarray, sr: int, model_name: str, normalize: bool = True, eps: float = 1e-10) -> np.ndarray:
    if waveform is None or len(waveform) == 0:
        return np.zeros((512,), dtype=np.float32)

    model, processor, device = _load_clap(model_name=model_name)

    inputs = processor(audios=waveform, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        audio_features = model.get_audio_features(**inputs)

    emb = audio_features[0].detach().cpu().numpy().astype(np.float32)

    if normalize:
        norm = np.linalg.norm(emb)
        emb = emb / max(norm, eps)

    return emb

def embed_segment(waveform: np.ndarray, sr: int, method: str = "mfcc", clap_model_name: str | None = None) -> np.ndarray:
    method = method.lower()
    if method == "mfcc":
        return mfcc_stats_embedding(waveform, sr)
    if method == "clap":
        if clap_model_name is None:
            raise ValueError("clap_model_name is required when method='clap'")
        return clap_embedding(waveform, sr, model_name=clap_model_name)
    raise ValueError(f"Unknown embedding method: {method}")
