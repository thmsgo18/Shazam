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

def embed_segment(waveform: np.ndarray, sr: int, method: str = "mfcc", clap_model_name: str | None = None) -> np.ndarray:
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
    if method == "clap":
        if clap_model_name is None:
            raise ValueError("clap_model_name is required when method='clap'")
        return clap_embedding(waveform, sr, model_name=clap_model_name)
    raise ValueError(f"Unknown embedding method: {method}")
