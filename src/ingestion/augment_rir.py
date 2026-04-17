"""
src/ingestion/augment_rir.py

Augmentation of embeddings by convolution with Room Impulse Responses (RIR).

For each track, N degraded versions are added to ChromaDB under the same
track_id. The augmentation improves robustness to recording conditions
(reverberation, ambient acoustics).

Architecture:
  Stage 1 — N parallel YouTube download threads
  Stage 2 — Parallel RIR convolution on CPU (scipy releases the GIL)
  Stage 3 — GPU embedding (sequential) + ChromaDB saving

Public entry point: run_augment(...)
"""

from __future__ import annotations

import os
import queue
import shutil
import signal
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK",      "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("OMP_NUM_THREADS",                  "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS",             "1")

import warnings
warnings.filterwarnings("ignore", message=".*upsample_bicubic2d.*", category=UserWarning)

import chromadb
import librosa
import numpy as np
import pandas as pd
import torch
from scipy.signal import fftconvolve

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress,
    SpinnerColumn, TextColumn, TimeElapsedColumn,
)
from rich.table import Table
from rich import box

from src import config
from src.audio.preprocessing import iter_segments
from src.audio.loading import load_audio
from src.utils.youtube import download_audio_from_url
from src.utils.metadata import atomic_write_parquet

ROOT       = Path(__file__).resolve().parents[2]
FLOWERS_ID = "f01ab00f1fdc5a57fd2676f4d68631a8"

console = Console()

# Clean stop on Ctrl+C — accessible by threads
_stop_requested = threading.Event()

# ---------------------------------------------------------------------------
# RIR — synthetic generation
# ---------------------------------------------------------------------------

SYNTHETIC_RIR_PARAMS = [
    ("bathroom",     0.15, 0),
    ("small_room",   0.25, 1),
    ("office",       0.40, 2),
    ("living_room",  0.60, 3),
    ("classroom",    0.80, 4),
    ("large_hall",   1.20, 5),
    ("concert_hall", 1.60, 6),
    ("bedroom",      0.35, 7),
    ("corridor",     0.55, 8),
    ("warehouse",    0.90, 9),
]


def _make_rir(rt60: float, sr: int, seed: int) -> np.ndarray:
    """Generates a synthetic RIR (direct sound + reflections + diffuse tail)."""
    rng      = np.random.RandomState(seed)
    n_samples = int(rt60 * sr * 1.5)
    rir       = np.zeros(n_samples, dtype=np.float32)

    rir[max(1, int(0.001 * sr))] = 1.0

    for _ in range(rng.randint(12, 30)):
        delay = rng.randint(int(0.002 * sr), int(0.08 * sr))
        amp   = rng.uniform(0.1, 0.7) * np.exp(-3.0 * delay / (rt60 * sr))
        if delay < n_samples:
            rir[delay] += (1 if rng.random() > 0.5 else -1) * amp

    late  = int(0.05 * sr)
    t     = np.arange(n_samples - late, dtype=np.float32)
    decay = np.exp(-6.908 * t / (rt60 * sr))
    rir[late:] += rng.randn(n_samples - late).astype(np.float32) * decay * 0.15

    norm = np.linalg.norm(rir)
    return (rir / norm) if norm > 0 else rir


def _estimate_rt60(rir: np.ndarray, sr: int) -> float:
    """
    Estimates the RT60 of a RIR by energy decay (Schroeder method).
    Returns the time (in seconds) for a 60 dB drop from the peak.
    """
    energy = rir ** 2
    peak_idx = int(np.argmax(energy))
    tail = energy[peak_idx:]
    # Schroeder backward integral
    schroeder = np.cumsum(tail[::-1])[::-1]
    schroeder = np.maximum(schroeder, 1e-12)
    db = 10.0 * np.log10(schroeder / schroeder[0])
    # First index where it goes below -60 dB
    indices = np.where(db <= -60.0)[0]
    if len(indices) == 0:
        # The RIR is too short to measure -60 dB → extrapolate
        indices_20 = np.where(db <= -20.0)[0]
        if len(indices_20) == 0:
            return len(tail) / sr
        return (indices_20[0] / sr) * 3.0  # RT20 × 3 ≈ RT60
    return indices[0] / sr


def _select_diverse_mit_rirs(
    candidates: list[tuple[str, np.ndarray, float]],
    n: int,
) -> list[tuple[str, np.ndarray]]:
    """
    Selects N RIRs from candidates to maximize acoustic diversity.

    Strategy: sort by RT60 then sample uniformly — this covers
    the entire range from dry spaces (short RT60) to large halls
    (long RT60), with maximum diversity possible for N samples.

    Args:
        candidates: list of (name, waveform, rt60) already loaded.
        n:          number of RIRs to return.

    Returns:
        List of (name, waveform) sorted from driest to most reverberant.
    """
    sorted_c = sorted(candidates, key=lambda x: x[2])  # sort by RT60 ascending
    if n >= len(sorted_c):
        return [(name, rir) for name, rir, _ in sorted_c]
    # Uniform sampling on the sorted list
    indices = np.linspace(0, len(sorted_c) - 1, n, dtype=int)
    selected = [sorted_c[i] for i in indices]
    console.print(
        f"[green]{len(candidates)} MIT RIRs available → "
        f"{n} selected by RT60 diversity "
        f"({selected[0][2]:.2f}s … {selected[-1][2]:.2f}s)[/green]"
    )
    return [(name, rir) for name, rir, _ in selected]


def _load_rirs(rir_dir: Path, n: int, sr: int, source: str = "synthetic") -> list[tuple[str, np.ndarray]]:
    """
    Loads N RIRs according to the configured source.

    Args:
        rir_dir: folder containing MIT WAVs (ignored if source="synthetic").
        n:       number of RIRs to return.
        sr:      target sample rate.
        source:  "synthetic" → mathematical RIRs | "mit" → WAVs from rir_dir folder.
    """
    if source == "mit":
        wavs = sorted(rir_dir.glob("*.wav")) + sorted(rir_dir.glob("*.WAV"))
        if not wavs:
            console.print(
                f"[yellow]⚠ No WAV in {rir_dir} (source='mit') "
                f"— switching to synthetic RIRs.[/yellow]"
            )
        else:
            console.print(f"[cyan]{len(wavs)} MIT WAV file(s) found in {rir_dir}[/cyan]")
            candidates: list[tuple[str, np.ndarray, float]] = []
            for wav in wavs:
                try:
                    y, _ = librosa.load(str(wav), sr=sr, mono=True)
                    rt60  = _estimate_rt60(y, sr)
                    candidates.append((wav.stem, y, rt60))
                except Exception as exc:
                    console.print(f"[yellow]⚠ Unable to load {wav.name}: {exc}[/yellow]")
            if candidates:
                return _select_diverse_mit_rirs(candidates, n)
            console.print("[yellow]⚠ No valid MIT RIR — switching to synthetics.[/yellow]")

    # source="synthetic" or fallback
    console.print(f"[cyan]Synthetic RIRs generated ({n} environments)[/cyan]")
    return [
        (f"synth_{name}_rt{int(rt60 * 100)}ms", _make_rir(rt60, sr, seed))
        for name, rt60, seed in SYNTHETIC_RIR_PARAMS[:n]
    ]


def _apply_rir(waveform: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve the waveform with the RIR, normalize RMS level, truncate."""
    deg     = fftconvolve(waveform.astype(np.float32), rir.astype(np.float32))[:len(waveform)]
    rms_orig = np.sqrt(np.mean(waveform ** 2)) + 1e-9
    rms_deg  = np.sqrt(np.mean(deg ** 2)) + 1e-9
    deg      = deg * (rms_orig / rms_deg)
    mx       = np.max(np.abs(deg))
    if mx > 1.0:
        deg /= mx
    return deg.astype(np.float32)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _get_batch_size(method: str) -> int:
    return {"clap": config.CLAP_BATCH_SIZE,
            "muq":  config.MUQ_BATCH_SIZE,
            "mert": config.MERT_BATCH_SIZE}.get(method, 64)


def _get_load_sr(method: str) -> int:
    if method == "clap":
        return config.CLAP_SAMPLE_RATE
    if method == "muq":
        return config.MUQ_SAMPLE_RATE
    if method == "mert":
        return config.MERT_SAMPLE_RATE
    return config.SAMPLE_RATE


def _batch_embed(segments: list[np.ndarray], sr: int, method: str) -> np.ndarray:
    """Embeds a batch of segments — method-agnostic."""
    if method == "clap":
        from src.features.embeddings_audio import clap_batch_embeddings
        result = clap_batch_embeddings(segments, sr=sr, model_name=config.CLAP_MODEL_NAME)
    elif method == "mert":
        from src.features.embeddings_audio import mert_batch_embeddings
        result = mert_batch_embeddings(segments, sr=sr, model_name=config.MERT_MODEL_NAME)
    elif method == "muq":
        from src.features.embeddings_audio import muq_batch_embeddings
        result = muq_batch_embeddings(segments, sr=sr, model_name=config.MUQ_MODEL_NAME)
    else:
        from src.features.embeddings_audio import mfcc_stats_embedding
        return np.stack([mfcc_stats_embedding(s, sr) for s in segments])

    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Tracking RIRs in metadata.parquet
# ---------------------------------------------------------------------------

def _backfill_rir_done(
    meta_path: Path,
    collection,
    collection_key: str,
) -> dict[str, list[str]]:
    """
    Scans ChromaDB to rebuild RIR history in metadata.parquet.
    Used when tracking did not exist yet during augmentation.

    Returns {track_id: [rir_names]}.
    """
    console.print("[yellow]Scanning ChromaDB to rebuild RIR history...[/yellow]")
    PAGE   = 500
    offset = 0
    rir_map: dict[str, set[str]] = {}

    while True:
        page = collection.get(limit=PAGE, offset=offset)
        ids  = page["ids"]
        if not ids:
            break
        for id_ in ids:
            if "_rir_" not in id_:
                continue
            parts = id_.split("_rir_", 1)
            if len(parts) != 2:
                continue
            track_id = parts[0]
            rir_name = "_".join(parts[1].split("_")[:-1])
            rir_map.setdefault(track_id, set()).add(rir_name)
        if len(ids) < PAGE:
            break
        offset += PAGE

    if not rir_map:
        console.print("[dim]No RIR segment found in ChromaDB.[/dim]")
        return {}

    console.print(f"[green]✓ {len(rir_map)} track(s) with RIRs found in ChromaDB[/green]")

    df = pd.read_parquet(meta_path)
    if "rir_augmented" not in df.columns:
        df["rir_augmented"] = [{} for _ in range(len(df))]

    updated = 0
    for i, row in df.iterrows():
        tid = row["track_id"]
        if tid not in rir_map:
            continue
        current  = row["rir_augmented"]
        if not isinstance(current, dict) or current is None:
            current = {}
        existing = current.get(collection_key)
        done_set = set(existing) if existing is not None and hasattr(existing, "__iter__") else set()
        done_set.update(rir_map[tid])
        current[collection_key] = sorted(done_set)
        df.at[i, "rir_augmented"] = current
        updated += 1

    try:
        atomic_write_parquet(meta_path, df)
    except Exception as exc:
        console.print(f"[red]Error writing metadata: {exc}[/red]")

    console.print(f"[green]✓ metadata.parquet updated ({updated} tracks)[/green]\n")
    return {tid: list(rirs) for tid, rirs in rir_map.items()}


def _load_rir_done(df_meta: pd.DataFrame, collection_key: str) -> dict[str, list[str]]:
    """
    Reads the rir_augmented column and returns {track_id: [rir_names already done]}.
    """
    if "rir_augmented" not in df_meta.columns:
        return {}
    result = {}
    for row in df_meta.itertuples():
        val = getattr(row, "rir_augmented", None)
        if isinstance(val, dict):
            done = val.get(collection_key, [])
            if done is not None:
                done_list = list(done) if hasattr(done, "__iter__") else []
                if done_list:
                    result[row.track_id] = done_list
    return result


def _mark_rir_done(
    meta_path: Path,
    track_id: str,
    collection_key: str,
    rir_name: str,
    lock: threading.Lock,
) -> None:
    """Adds rir_name to rir_augmented[collection_key] for this track_id (atomic)."""
    with lock:
        df = pd.read_parquet(meta_path)
        if "rir_augmented" not in df.columns:
            df["rir_augmented"] = [{} for _ in range(len(df))]

        idx = df.index[df["track_id"] == track_id]
        if len(idx) == 0:
            return

        i       = idx[0]
        current = df.at[i, "rir_augmented"]
        if not isinstance(current, dict) or current is None:
            current = {}
        existing  = current.get(collection_key)
        done_list = list(existing) if existing is not None and hasattr(existing, "__iter__") else []
        if rir_name not in done_list:
            done_list.append(rir_name)
        current[collection_key] = done_list
        df.at[i, "rir_augmented"] = current
        atomic_write_parquet(meta_path, df)


# ---------------------------------------------------------------------------
# FAISS index reconstruction
# ---------------------------------------------------------------------------

def _rebuild_index(collection_key: str, chroma_client) -> None:
    from src.index.build_index import _build_for_method
    console.print("\n[yellow]Rebuilding FAISS index...[/yellow]")
    _build_for_method(collection_key, config.INDEX_TYPE, chroma_client)
    console.print("[green]✓ FAISS index rebuilt.[/green]")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_augment(
    method: str | None = None,
    tracks: str = "all",
    n_rir: int | None = None,
    rir_dir: str | None = None,
    source: str | None = None,
    workers: int | None = None,
    device: str | None = None,
    rebuild_index: bool = True,
) -> None:
    """
    Augments the ChromaDB database with RIR-degraded versions.

    Args:
        method:        embedding method (None = config.EMBEDDING_METHOD).
        tracks:        'flowers', 'all' or a specific track_id.
        n_rir:         number of RIRs to apply per track (None = config.RIR_N).
        rir_dir:       MIT WAV folder (None = config.RIR_MIT_DIR).
        source:        "synthetic" | "mit" (None = config.RIR_SOURCE).
        workers:       number of download threads (None = config.DOWNLOAD_WORKERS).
        device:        PyTorch device: 'cpu' / 'cuda' / 'mps' (None = auto).
        rebuild_index: if True, rebuilds FAISS index after augmentation.
    """
    torch.set_num_threads(4)
    _stop_requested.clear()

    if method is None:
        method = config.EMBEDDING_METHOD
    if workers is None:
        workers = config.DOWNLOAD_WORKERS
    if n_rir is None:
        n_rir = config.RIR_N
    if source is None:
        source = config.RIR_SOURCE
    if rir_dir is None:
        rir_dir = config.RIR_MIT_DIR

    load_sr    = _get_load_sr(method)
    batch_size = _get_batch_size(method)
    rir_path   = Path(rir_dir)
    rir_path.mkdir(parents=True, exist_ok=True)

    console.print(Panel(
        f"[bold]Method     :[/bold] [cyan]{method}[/cyan]\n"
        f"[bold]Tracks      :[/bold] [cyan]{tracks}[/cyan]\n"
        f"[bold]RIR Source  :[/bold] [cyan]{source}[/cyan]\n"
        f"[bold]N RIRs      :[/bold] [cyan]{n_rir}[/cyan]\n"
        f"[bold]DL Workers  :[/bold] [cyan]{workers}[/cyan]\n"
        + (f"[bold]RIR dir     :[/bold] [cyan]{rir_dir}[/cyan]" if source == "mit" else ""),
        title="[bold cyan]RIR Augmentation[/bold cyan]",
        expand=False,
    ))

    # Pre-loading model BEFORE any faiss import
    if method == "clap":
        from src.features.embeddings_audio import _load_clap
        console.print(f"[cyan]Loading model {config.CLAP_MODEL_NAME} ({device})...[/cyan]")
        _load_clap(config.CLAP_MODEL_NAME, device=device)
        console.print("[green]✓ CLAP model ready.[/green]\n")
    elif method == "muq":
        from src.features.embeddings_audio import _load_muq
        console.print(f"[cyan]Loading model {config.MUQ_MODEL_NAME} ({device})...[/cyan]")
        _load_muq(config.MUQ_MODEL_NAME)
        console.print("[green]✓ MuQ model ready.[/green]\n")
    elif method == "mert":
        from src.features.embeddings_audio import _load_mert
        console.print(f"[cyan]Loading model {config.MERT_MODEL_NAME} ({device})...[/cyan]")
        _load_mert(config.MERT_MODEL_NAME)
        console.print("[green]✓ MERT model ready.[/green]\n")

    collection_key = config.get_collection_key(method)
    chroma_client  = chromadb.PersistentClient(path=str(ROOT / config.CHROMA_DIR))
    try:
        collection = chroma_client.get_collection(name=collection_key)
    except Exception:
        console.print(f"[red]Collection '{collection_key}' not found. Run ingestion first.[/red]")
        return

    def _on_interrupt(sig, frame):
        console.print("\n[yellow]⚠ Interrupt detected — graceful shutdown in progress...[/yellow]")
        _stop_requested.set()

    signal.signal(signal.SIGINT,  _on_interrupt)
    signal.signal(signal.SIGTERM, _on_interrupt)

    console.print("[yellow]Loading RIRs...[/yellow]")
    rirs = _load_rirs(rir_path, n_rir, sr=22050, source=source)
    if not rirs:
        console.print("[red]No RIR available. Aborting.[/red]")
        return

    t_rir = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
    t_rir.add_column("RIR", width=38)
    t_rir.add_column("Duration", justify="right", width=12)
    for name, rir in rirs:
        t_rir.add_row(name, f"{len(rir) / 22050 * 1000:.0f} ms")
    console.print(t_rir)

    meta_path = ROOT / config.METADATA_PATH
    df_meta   = pd.read_parquet(meta_path)

    if tracks == "flowers":
        df_all = df_meta[df_meta["track_id"] == FLOWERS_ID]
    elif tracks == "all" or tracks is None:
        df_all = df_meta
    elif isinstance(tracks, list):
        df_all = df_meta[df_meta["track_id"].isin(tracks)]
    else:
        df_all = df_meta[df_meta["track_id"] == tracks]

    if df_all.empty:
        console.print(f"[red]No track found for '{tracks}'.[/red]")
        return

    # Automatic backfill if tracking is missing
    needs_backfill = (
        "rir_augmented" not in df_all.columns
        or df_all["rir_augmented"].apply(
            lambda x: not isinstance(x, dict) or collection_key not in (x or {})
        ).all()
    )
    if needs_backfill:
        _backfill_rir_done(meta_path, collection, collection_key)
        df_all = pd.read_parquet(meta_path)
        if tracks == "flowers":
            df_all = df_all[df_all["track_id"] == FLOWERS_ID]
        elif tracks != "all":
            df_all = df_all[df_all["track_id"] == tracks]

    rir_names_requested = {name for name, _ in rirs}
    rir_done_map        = _load_rir_done(df_all, collection_key)

    df_targets = df_all[
        df_all["track_id"].apply(
            lambda tid: not rir_names_requested.issubset(set(rir_done_map.get(tid, [])))
        )
    ]

    n_already_done = len(df_all) - len(df_targets)
    if n_already_done:
        console.print(
            f"[dim]{n_already_done} track(s) skipped[/dim] "
            f"[dim](all {len(rirs)} RIRs already applied)[/dim]"
        )

    if df_targets.empty:
        console.print("[green]✓ Everything is already augmented — nothing to do.[/green]")
        return

    total     = len(df_targets)
    meta_lock = threading.Lock()

    console.print(f"\n[bold]{total} track(s) to augment[/bold] × {len(rirs)} RIRs\n")

    # ── Stage 1: downloads ───────────────────────────────────────────────────
    work_q: queue.Queue = queue.Queue()
    dl_q:   queue.Queue = queue.Queue(maxsize=workers * 2)

    for row in df_targets.itertuples():
        work_q.put(row)

    active_dl = threading.Semaphore(0)

    def _dl_worker():
        active_dl.release()
        while not _stop_requested.is_set():
            try:
                row = work_q.get_nowait()
            except queue.Empty:
                break
            url = getattr(row, "url", None)
            if not url or not isinstance(url, str) or not url.startswith("http"):
                dl_q.put((row, None, None))
            else:
                tmpdir, path = download_audio_from_url(url, stop_event=_stop_requested)
                dl_q.put((row, tmpdir, path))
        active_dl.acquire()

    dl_threads = [threading.Thread(target=_dl_worker, daemon=True) for _ in range(workers)]
    for t in dl_threads:
        t.start()

    def _sentinel_watcher():
        for t in dl_threads:
            t.join()
        dl_q.put(None)

    threading.Thread(target=_sentinel_watcher, daemon=True).start()

    # ── Stage 2: RIR convolution ─────────────────────────────────────────────
    conv_q: queue.Queue = queue.Queue(maxsize=2)

    def _convolve_worker():
        while not _stop_requested.is_set():
            item = dl_q.get()
            if item is None:
                conv_q.put(None)
                break

            row, tmpdir, audio_path = item
            track_id = row.track_id

            if audio_path is None:
                conv_q.put((row, None, None, set()))
                continue

            done_for_track = set(rir_done_map.get(track_id, []))
            rirs_todo      = [(name, rir) for name, rir in rirs if name not in done_for_track]

            if not rirs_todo:
                shutil.rmtree(tmpdir, ignore_errors=True)
                conv_q.put((row, [], None, done_for_track))
                continue

            try:
                waveform, sr = load_audio(audio_path, target_sr=load_sr)
            except Exception:
                shutil.rmtree(tmpdir, ignore_errors=True)
                conv_q.put((row, None, None, set()))
                continue
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

            def _conv_one(args):
                rir_name, rir = args
                rir_rs = (
                    librosa.resample(rir, orig_sr=22050, target_sr=load_sr)
                    if load_sr != 22050 else rir
                )
                wf_deg = _apply_rir(waveform, rir_rs)
                segs   = list(iter_segments(
                    wf_deg, load_sr,
                    config.SEGMENT_WIN_S, config.SEGMENT_HOP_S, config.SEGMENT_MIN_WIN,
                ))
                return rir_name, segs

            n_conv = min(len(rirs_todo), 4)
            with ThreadPoolExecutor(max_workers=n_conv) as pool:
                degraded = list(pool.map(_conv_one, rirs_todo))

            conv_q.put((row, degraded, None, done_for_track))

    conv_thread = threading.Thread(target=_convolve_worker, daemon=True)
    conv_thread.start()

    # ── Stage 3: GPU embedding + ChromaDB persistence ───────────────────────
    n_added = n_skipped = n_failed = done = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        refresh_per_second=4,
    ) as progress:
        task_tracks = progress.add_task("[bold]Tracks[/bold]", total=total)
        task_detail = progress.add_task("", total=None, visible=False)

        while True:
            if _stop_requested.is_set():
                break

            item = conv_q.get()
            if item is None:
                break

            row, degraded, _, existing_ids = item
            track_id = row.track_id
            label    = f"{getattr(row,'artist','')[:18]} — {getattr(row,'title',track_id)[:32]}"

            progress.update(task_tracks, description=f"[bold cyan]{label}[/bold cyan]")

            if degraded is None:
                progress.update(task_detail, description="[red]✗ Failed[/red]", visible=True)
                n_failed += 1
                done     += 1
                progress.update(task_tracks, completed=done)
                continue

            if degraded == []:
                progress.update(task_detail, description="[dim]already augmented[/dim]", visible=True)
                n_skipped += 1
                done      += 1
                progress.update(task_tracks, completed=done)
                continue

            for rir_idx, (rir_name, segs) in enumerate(degraded, 1):
                if _stop_requested.is_set():
                    break
                if not segs:
                    continue

                progress.update(
                    task_detail,
                    description=(
                        f"[yellow][RIR {rir_idx}/{len(degraded)}][/yellow] "
                        f"[dim]{rir_name}[/dim] ({len(segs)} seg)"
                    ),
                    visible=True,
                )

                ids_batch, emb_batch, meta_batch = [], [], []
                for i in range(0, len(segs), batch_size):
                    batch     = segs[i: i + batch_size]
                    seg_waves = [seg for _, seg in batch]
                    embs      = _batch_embed(seg_waves, load_sr, method)
                    for j, (start_s, _) in enumerate(batch):
                        seg_id = f"{track_id}_rir_{rir_name}_{i + j}"
                        ids_batch.append(seg_id)
                        emb_batch.append(embs[j].tolist())
                        meta_batch.append({"track_id": track_id, "start_s": float(start_s)})

                if ids_batch:
                    collection.add(
                        ids=ids_batch,
                        embeddings=emb_batch,
                        metadatas=meta_batch,
                        documents=[""] * len(ids_batch),
                    )
                    n_added += len(ids_batch)
                    _mark_rir_done(meta_path, track_id, collection_key, rir_name, meta_lock)

            done += 1
            progress.update(task_tracks, completed=done)

        progress.update(task_detail, visible=False)

    conv_thread.join(timeout=5)
    for t in dl_threads:
        t.join(timeout=2)

    interrupted = _stop_requested.is_set()
    console.print(Panel(
        f"[bold]Segments added:[/bold] [green]{n_added}[/green]\n"
        f"[bold]Tracks skipped:[/bold] [dim]{n_skipped}[/dim]\n"
        f"[bold]Tracks failed:[/bold] [yellow]{n_failed}[/yellow]\n"
        f"[bold]Total ChromaDB   :[/bold] [cyan]{collection.count()}[/cyan] segments"
        + ("\n[yellow]⚠ Stopped early — rerun with the same command[/yellow]" if interrupted else ""),
        title="[bold green]Augmentation completed[/bold green]" if not interrupted
              else "[bold yellow]Augmentation interrupted[/bold yellow]",
        expand=False,
    ))

    if rebuild_index and n_added > 0:
        _rebuild_index(collection_key, chroma_client)
    elif n_added == 0:
        console.print("[dim]No segments added — FAISS index left unchanged.[/dim]")
