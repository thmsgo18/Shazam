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

def _load_clap(model_name: str, device: str | None = None, local_files_only: bool = False):
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
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"   # GPU Apple Silicon (M1/M2/M3)
        else:
            device = "cpu"

    # On Apple Silicon (MPS), some CLAP operations are not natively supported.
    # We automatically enable CPU fallback so it works on all machines.
    if device == "mps":
        import os
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    import src.config as config
    processor = ClapProcessor.from_pretrained(model_name, local_files_only=local_files_only)       # Load the CLAP processor.
    # Float16 uniquement sur CUDA — sur CPU/MPS beaucoup d'opérations ne supportent pas Half
    dtype = torch.float16 if (config.OPT_FLOAT16 and device == "cuda") else torch.float32
    model = ClapModel.from_pretrained(
        model_name,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    ).to(device)
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

    # CLAP checkpoints are trained for a fixed sampling rate (usually 48 kHz).
    # Resample the input on the fly so CLAP can process project audio loaded at another rate.
    waveform = np.asarray(waveform, dtype=np.float32)
    target_sr = int(getattr(processor.feature_extractor, "sampling_rate", sr))
    if sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    inputs = processor(audios=[waveform], sampling_rate=sr, return_tensors="pt") # Convert raw waveform into model-ready tensors with the CLAP processor.
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad(): # Disable gradient computation for faster inference and lower memory usage
        audio_features = model.get_audio_features(**inputs) # Extract audio feature embedding from the CLAP model

    emb = audio_features[0].detach().cpu().numpy().astype(np.float32)

    if normalize: # Optionally normalize the embedding vector using L2 normalization
        norm = np.linalg.norm(emb)
        emb = emb / max(norm, eps)

    return emb

def clap_batch_embeddings(
    segments: list[np.ndarray],
    sr: int,
    model_name: str,
    normalize: bool = True,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Compute CLAP embeddings for a batch of segments in a single GPU call.
    Similar to muq_batch_embeddings — avoids repeated GPU round trips.

    Returns:
        emb: (B, 512) float32
    """
    import torch

    if not segments:
        return np.zeros((0, 512), dtype=np.float32)

    model, processor, device = _load_clap(model_name=model_name)

    target_sr = int(getattr(processor.feature_extractor, "sampling_rate", sr))

    resampled = []
    for seg in segments:
        y = np.asarray(seg, dtype=np.float32)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        resampled.append(y)

    inputs = processor(audios=resampled, sampling_rate=target_sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        audio_features = model.get_audio_features(**inputs)  # (B, 512)

    emb = audio_features.detach().cpu().numpy().astype(np.float32)

    if normalize:
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norms, eps)

    return emb


# ******************** MuQ : ********************

_MUQ_CACHE = {"model": None, "device": None, "model_name": None}

def _load_muq(model_name: str, device: str | None = None, local_files_only: bool = False):
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
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"  # MPS does not support ComplexFloat (required by MuQ)

    import src.config as config
    # Float16 uniquement sur CUDA — sur CPU LayerNorm ne supporte pas Half
    dtype = torch.float16 if (config.OPT_FLOAT16 and device == "cuda") else torch.float32
    try:
        model = MuQ.from_pretrained(model_name, local_files_only=local_files_only).to(dtype).to(device)
    except TypeError:
        model = MuQ.from_pretrained(model_name).to(dtype).to(device)
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

    # Handle empty input (dim 1024 fixe pour MuQ-large)
    if waveform is None or len(waveform) == 0:
        return np.zeros((1024,), dtype=np.float32)

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

def muq_batch_embeddings(
    segments: list[np.ndarray],
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

    # Load model (cached) and run forward
    model, device = _load_muq(model_name=model_name)

    # Cast input to the same type as the model (float16 or float32)
    model_dtype = next(model.parameters()).dtype
    x_t = torch.from_numpy(x).to(device).to(model_dtype)
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

    return emb

# ******************** MERT : ********************

_MERT_CACHE = {"model": None, "processor": None, "device": None, "model_name": None}

def _load_mert(model_name: str, device: str | None = None, local_files_only: bool = False):
    """
    Load and cache a MERT model (Music undERstanding Transformer).
    MERT is trained exclusively on music → better robustness
    to timbral variations (microphone noise, reverberation) than CLAP.

    Args:
        model_name: e.g. "m-a-p/MERT-v1-95M" or "m-a-p/MERT-v1-330M"
        device: "cuda", "mps" or "cpu". Automatic detection if None.

    Returns:
        (model, processor, device)
    """
    import torch
    from transformers import Wav2Vec2FeatureExtractor, AutoModel

    if _MERT_CACHE["model"] is not None and _MERT_CACHE["model_name"] == model_name:
        return _MERT_CACHE["model"], _MERT_CACHE["processor"], _MERT_CACHE["device"]

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if device == "mps":
        import os
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    import src.config as config
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    dtype = torch.float16 if (config.OPT_FLOAT16 and device == "cuda") else torch.float32
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    ).to(device)
    model.eval()

    _MERT_CACHE.update({"model": model, "processor": processor, "device": device, "model_name": model_name})
    return model, processor, device


def mert_embedding(waveform: np.ndarray, sr: int, model_name: str, target_sr: int = 24000, normalize: bool = True, eps: float = 1e-10) -> np.ndarray:
    """
    Compute an audio embedding with MERT (mean pooling on last_hidden_state).

    Returns:
        np.ndarray: 1D vector of dimension 768 (MERT-v1-95M) or 1024 (MERT-v1-330M).
    """
    import torch

    if waveform is None or len(waveform) == 0:
        return np.zeros((768,), dtype=np.float32)

    y = np.asarray(waveform, dtype=np.float32)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    model, processor, device = _load_mert(model_name=model_name)

    inputs = processor(y, sampling_rate=target_sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=False)
        emb_t = out.last_hidden_state.mean(dim=1)   # (1, H)

    emb = emb_t[0].detach().cpu().numpy().astype(np.float32)

    if normalize:
        norm = float(np.linalg.norm(emb))
        emb = emb / max(norm, eps)

    return emb


def mert_batch_embeddings(
    segments: list[np.ndarray],
    sr: int,
    model_name: str,
    target_sr: int = 24000,
    normalize: bool = True,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Compute MERT embeddings for a batch of segments in a single GPU call.

    Returns:
        emb: (B, H) float32
    """
    import torch

    if not segments:
        return np.zeros((0, 768), dtype=np.float32)

    model, processor, device = _load_mert(model_name=model_name)

    resampled = []
    for seg in segments:
        y = np.asarray(seg, dtype=np.float32)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        resampled.append(y)

    inputs = processor(resampled, sampling_rate=target_sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=False)
        emb_t = out.last_hidden_state.mean(dim=1)   # (B, H)

    emb = emb_t.detach().cpu().numpy().astype(np.float32)

    if normalize:
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norms, eps)

    return emb


# ******************** Embedding general : ********************

def embed_segment(waveform: np.ndarray, sr: int, method: str = "mfcc", muq_model_name: str | None = None, clap_model_name: str | None = None, mert_model_name: str | None = None) -> np.ndarray:
    """
    Compute an audio embedding using the selected embedding method.
    This function routes the audio waveform to the appropriate embedding function based on the selected method (MFCC or CLAP).

    Args:
        waveform (np.ndarray): Input audio waveform as a 1D NumPy array.
        sr (int): Sampling rate of the audio signal.
        method (str, optional): Embedding method to use ("mfcc", "clap", "muq" or "mert"). Default is "mfcc".
        muq_model_name (str | None, optional): Name or path of the pretrained MuQ model. Required if method is "muq".
        clap_model_name (str | None, optional): Name or path of the pretrained CLAP model. Required if method is "clap".
        mert_model_name (str | None, optional): Name or path of the pretrained MERT model. Required if method is "mert".

    Returns:
        np.ndarray: A 1D embedding vector produced by the selected method.

    Raises:
        ValueError: If an unknown method is provided or if a required model name is missing.
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
    if method == "mert":
        if mert_model_name is None:
            raise ValueError("mert_model_name is required when method='mert'")
        return mert_embedding(waveform, sr, model_name=mert_model_name)
    raise ValueError(f"Unknown embedding method: {method}")
