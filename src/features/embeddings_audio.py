"""
src/features/embeddings_audio.py

This file allows you to group together everything related to embedding.
We have the available embedding functions
and the function that will be called each time, which will choose the embedding function to apply based on the selected function.
"""

import numpy as np
import librosa

def mfcc_stats_embedding(waveform: np.ndarray, sr: int, n_mfcc: int = 20, n_fft: int = 2048, hop_length: int = 512, normalize: bool = True, eps: float = 1e-10) -> np.ndarray:
    """
    Compute a fixed-size audio embedding using MFCC statistics.
    This function extracts MFCC features from an audio waveform, computes the mean and standard deviation over time, and concatenates them into a single embedding vector.

    Args:
        waveform (np.ndarray): Input audio waveform as a 1D NumPy array.
        sr (int): Frequency of the audio signal.
        n_mfcc (int, optional): Number of MFCC coefficients to extract. Default 20.
        n_fft (int, optional): FFT window size for MFCC computation. Default 2048.
        hop_length (int, optional): Number of samples between frames. Default 512.
        normalize (bool, optional): Whether to L2-normalize the final embedding vector. Default True.
        eps (float, optional): Small value to avoid division by zero during normalization. Default 1e-10.

    Returns:
        np.ndarray: A 1D embedding vector of shape (2 * n_mfcc,), containing MFCC means. Returns a zero vector if waveform is empty or None.
    """
    # If waveform is None or empty, return a zero vector.
    if waveform is None or len(waveform) == 0:
        return np.zeros((2 * n_mfcc,), dtype=np.float32)

    # Compute MFCC features from the waveform using librosa.
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

    mean = mfcc.mean(axis=1)
    std = mfcc.std(axis=1)

    # Concatenate mean and std into a single embedding vector.
    embedding = np.concatenate([mean, std]).astype(np.float32)

    # Optionally normalize the embedding vector using L2 normalization:
    if normalize:
        norm = np.linalg.norm(embedding)
        embedding = embedding / max(norm, eps)

    return embedding

# ******************** CLAP : ********************

_CLAP_CACHE = {"model": None, "processor": None, "device": None, "model_name": None}

def _load_clap(model_name: str, device: str | None = None):
    """
    Load and cache a CLAP model.
    This function loads a pretrained CLAP model, reusing a cached version if available. The model use GPU if available or CPU.

    Args:
        model_name (str): Name or path of the pretrained CLAP model.
        device (str | None, optional): Target device ("cuda" or "cpu"). If None, automatically selects CUDA or CPU.

    Returns (tuple):
        - model: Loaded CLAP model in evaluation mode.
        - processor: Associated CLAP processor.
        - device: Device where the model is loaded.
    """
    import torch
    from transformers import ClapModel, ClapProcessor

    # If a model is already cached we return directly to avoid reloading from disk or network.
    if _CLAP_CACHE["model"] is not None and _CLAP_CACHE["model_name"] == model_name:
        return _CLAP_CACHE["model"], _CLAP_CACHE["processor"], _CLAP_CACHE["device"]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = ClapProcessor.from_pretrained(model_name)       # Load the CLAP processor.
    model = ClapModel.from_pretrained(model_name, use_safetensors=True).to(device)    # Load the pretrained CLAP model.
    model.eval()                                                # Set the model to evaluation mode.

    # Store loaded objects in cache for faster reuse in future calls :
    _CLAP_CACHE.update({"model": model, "processor": processor, "device": device, "model_name": model_name})

    return model, processor, device

def clap_embedding(waveform: np.ndarray, sr: int, model_name: str, normalize: bool = True, eps: float = 1e-10) -> np.ndarray:
    """
    Compute an audio embedding using a pretrained CLAP model.
    This function generates an embedding from an audio waveform using a pretrained CLAP model, with optional L2 normalization.

    Args:
        waveform (np.ndarray): Input audio waveform as a 1D NumPy array.
        sr (int): Sampling rate of the audio signal.
        model_name (str): Name or path of the pretrained CLAP model.
        normalize (bool, optional): Whether to L2-normalize the embedding. Default True.
        eps (float, optional): Small value to avoid division by zero during normalization. Default 1e-10.

    Returns:
        np.ndarray: A 1D embedding vector (typically size 512). Zero vector if the waveform is empty or None.
    """
    import torch
    if waveform is None or len(waveform) == 0: # If waveform is None or empty.
        return np.zeros((512,), dtype=np.float32)

    model, processor, device = _load_clap(model_name=model_name) # Load or retrieve from cache the information of the CLAP.

    inputs = processor(audios=waveform, sampling_rate=sr, return_tensors="pt") # Convert raw waveform into model-ready tensors with the CLAP processor.
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad(): # Disable gradient computation for faster inference and lower memory usage
        audio_features = model.get_audio_features(**inputs) # Extract audio feature embedding from the CLAP model

    emb = audio_features[0].detach().cpu().numpy().astype(np.float32)

    if normalize: # Optionally normalize the embedding vector using L2 normalization
        norm = np.linalg.norm(emb)
        emb = emb / max(norm, eps)

    return emb

# ******************** MuQ : ********************

_MUQ_CACHE = {"model": None, "device": None, "model_name": None}

def _load_muq(model_name: str, device: str | None = None):
    """
    Load and cache a MuQ model (pretrained).
    - Loads once, then reuses from cache for all segments.
    - Uses GPU if available, otherwise CPU.
    """
    import torch
    from muq import MuQ

    # Reuse cached model if it matches the requested checkpoint
    if _MUQ_CACHE["model"] is not None and _MUQ_CACHE["model_name"] == model_name:
        return _MUQ_CACHE["model"], _MUQ_CACHE["device"]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MuQ.from_pretrained(model_name).to(device)
    model.eval()

    _MUQ_CACHE.update({"model": model, "device": device, "model_name": model_name})
    return model, device


def muq_embedding(waveform: np.ndarray, sr: int, model_name: str, target_sr: int = 24000, normalize: bool = True, eps: float = 1e-10) -> np.ndarray:
    """
    Compute an audio embedding using MuQ.

    Strategy:
    - Resample to target_sr (MuQ commonly expects 24kHz).
    - Run MuQ forward -> BaseModelOutput
    - Use mean pooling over time on last_hidden_state => fixed-size vector (H,)
    - Optional L2 normalization.
    """
    import torch

    # Handle empty input
    if waveform is None or len(waveform) == 0:
        # If possible, return correct dim based on model config; else fallback to 0-len safe vector
        try:
            model, _device = _load_muq(model_name=model_name)
            h = int(getattr(model.config, "hidden_size", 0)) or 0
            return np.zeros((h,), dtype=np.float32) if h > 0 else np.zeros((1,), dtype=np.float32)
        except Exception:
            return np.zeros((1,), dtype=np.float32)

    # Ensure float32
    y = np.asarray(waveform, dtype=np.float32)

    # Resample if needed
    if sr != target_sr:
        # librosa.resample expects float array
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Load model (cached)
    model, device = _load_muq(model_name=model_name)

    # MuQ forward signature: forward(self, x, attention_mask=None, output_hidden_states=True)
    # Build input tensor: (B=1, T)
    x = torch.from_numpy(y).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)  # BaseModelOutput
        hs = out.last_hidden_state  # (1, T', H)

        # Mean pooling over time dimension => (1, H)
        emb_t = hs.mean(dim=1)

    emb = emb_t[0].detach().cpu().numpy().astype(np.float32)

    if normalize:
        norm = float(np.linalg.norm(emb))
        emb = emb / max(norm, eps)

    return emb

from typing import List, Tuple

def muq_batch_embeddings(
    segments: List[np.ndarray],
    sr: int,
    model_name: str,
    target_sr: int = 24000,
    normalize: bool = True,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Compute MuQ embeddings for a batch of segments.

    Returns:
        emb: (B, H) float32
    """
    import time
    t_rs = time.time()
    import torch

    if segments is None or len(segments) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    # Resample each segment if needed, keep as float32
    ys = []
    lengths = []

    for seg in segments:
        if seg is None or len(seg) == 0:
            ys.append(np.zeros((1,), dtype=np.float32))
            lengths.append(1)
            continue

        y = np.asarray(seg, dtype=np.float32)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        ys.append(y)
        lengths.append(len(y))

    # Pad to same length
    max_len = max(lengths)
    B = len(ys)

    x = np.zeros((B, max_len), dtype=np.float32)
    attn = np.zeros((B, max_len), dtype=np.int64)

    for i, y in enumerate(ys):
        L = len(y)
        x[i, :L] = y
        attn[i, :L] = 1

    t_fwd = time.time()

    # Load model (cached) and run forward
    model, device = _load_muq(model_name=model_name)

    x_t = torch.from_numpy(x).to(device)
    attn_t = torch.from_numpy(attn).to(device)

    with torch.no_grad():
        out = model(x_t, attention_mask=attn_t)  # BaseModelOutput
        hs = out.last_hidden_state  # (B, T', H)

    # Pooling: masked mean if possible, else normal mean
    # (Sometimes T' != max_len; in that case we fallback to mean)
    if hs.shape[1] == attn_t.shape[1]:
        mask = attn_t.unsqueeze(-1).float()  # (B, T, 1)
        summed = (hs * mask).sum(dim=1)      # (B, H)
        denom = mask.sum(dim=1).clamp_min(eps)  # (B, 1)
        emb_t = summed / denom
    else:
        emb_t = hs.mean(dim=1)

    emb = emb_t.detach().cpu().numpy().astype(np.float32)  # (B, H)

    if normalize:
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norms, eps)
        
    t_end = time.time()
    print(f"[muq_batch] B={len(segments)} rs+pad={t_fwd-t_rs:.3f}s fwd={t_end-t_fwd:.3f}s")

    return emb

# ******************** Embedding general : ********************

def embed_segment(waveform: np.ndarray, sr: int, method: str = "mfcc", muq_model_name: str | None = None, clap_model_name: str | None = None) -> np.ndarray:
    """
    Compute an audio embedding using the selected embedding method.
    This function routes the audio waveform to the appropriate embedding function based on the selected method (MFCC or CLAP).

    Args:
        waveform (np.ndarray): Input audio waveform as a 1D NumPy array.
        sr (int): Sampling rate of the audio signal.
        method (str, optional): Embedding method to use ("mfcc" or "clap"). Default is "mfcc".
        clap_model_name (str | None, optional): Name or path of the pretrained CLAP model. Required if method is "clap".

    Returns:
        np.ndarray: A 1D embedding vector produced by the selected method.

    Raises:
        ValueError: If an unknown method is provided or if CLAP is selected without a model name.
    """
    method = method.lower()
    if method == "mfcc":
        return mfcc_stats_embedding(waveform, sr)
    if method == "muq":
        if muq_model_name is None:
            raise ValueError("muq_model_name is required when method='muq'")
        return muq_embedding(waveform, sr, model_name=muq_model_name)
    if method == "clap":
        if clap_model_name is None:
            raise ValueError("clap_model_name is required when method='clap'")
        return clap_embedding(waveform, sr, model_name=clap_model_name)
    raise ValueError(f"Unknown embedding method: {method}")
