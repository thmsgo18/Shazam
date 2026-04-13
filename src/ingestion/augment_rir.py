"""
src/ingestion/augment_rir.py

Augmentation des embeddings par convolution de Room Impulse Responses (RIR).

Pour chaque track, N versions dégradées sont ajoutées dans ChromaDB sous le même
track_id. L'augmentation améliore la robustesse aux conditions d'enregistrement
(réverbération, acoustique ambiante).

Architecture :
  Étage 1 — N threads de téléchargement YouTube en parallèle
  Étage 2 — Convolution RIR en parallèle sur CPU (scipy libère le GIL)
  Étage 3 — Embedding GPU (séquentiel) + sauvegarde ChromaDB

Point d'entrée public : run_augment(...)
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

# Arrêt propre sur Ctrl+C — accessible par les threads
_stop_requested = threading.Event()

# ---------------------------------------------------------------------------
# RIR — génération synthétique
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
    """Génère une RIR synthétique (son direct + réflexions + queue diffuse)."""
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


def _load_rirs(rir_dir: Path, n: int, sr: int) -> list[tuple[str, np.ndarray]]:
    """Charge N RIRs depuis rir_dir (WAV) ou génère des RIRs synthétiques."""
    wavs = sorted(rir_dir.glob("*.wav")) + sorted(rir_dir.glob("*.WAV"))
    if wavs:
        console.print(f"[green]{len(wavs)} fichier(s) RIR trouvé(s) dans {rir_dir}[/green]")
        rirs = []
        for wav in wavs[:n]:
            try:
                y, _ = librosa.load(str(wav), sr=sr, mono=True)
                rirs.append((wav.stem, y))
            except Exception as exc:
                console.print(f"[yellow]⚠ Impossible de charger {wav.name} : {exc}[/yellow]")
        return rirs

    console.print(f"[cyan]Aucun WAV dans {rir_dir} — RIRs synthétiques générées[/cyan]")
    return [
        (f"synth_{name}_rt{int(rt60 * 100)}ms", _make_rir(rt60, sr, seed))
        for name, rt60, seed in SYNTHETIC_RIR_PARAMS[:n]
    ]


def _apply_rir(waveform: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve le waveform avec la RIR, normalise le niveau RMS, tronque."""
    deg     = fftconvolve(waveform.astype(np.float32), rir.astype(np.float32))[:len(waveform)]
    rms_orig = np.sqrt(np.mean(waveform ** 2)) + 1e-9
    rms_deg  = np.sqrt(np.mean(deg ** 2)) + 1e-9
    deg      = deg * (rms_orig / rms_deg)
    mx       = np.max(np.abs(deg))
    if mx > 1.0:
        deg /= mx
    return deg.astype(np.float32)


# ---------------------------------------------------------------------------
# Helpers d'embedding
# ---------------------------------------------------------------------------

def _get_batch_size(method: str) -> int:
    return {"clap": config.CLAP_BATCH_SIZE,
            "muq":  config.MUQ_BATCH_SIZE,
            "mert": config.MERT_BATCH_SIZE}.get(method, 64)


def _get_load_sr(method: str) -> int:
    if method == "clap":
        return config.CLAP_SAMPLE_RATE
    if method in ("muq", "mert"):
        return config.MERT_SAMPLE_RATE
    return config.SAMPLE_RATE


def _batch_embed(segments: list[np.ndarray], sr: int, method: str) -> np.ndarray:
    """Embedde un batch de segments — agnostique à la méthode."""
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
# Suivi des RIRs dans metadata.parquet
# ---------------------------------------------------------------------------

def _backfill_rir_done(
    meta_path: Path,
    collection,
    collection_key: str,
) -> dict[str, list[str]]:
    """
    Scanne ChromaDB pour reconstruire l'historique RIR dans metadata.parquet.
    Utilisé quand le tracking n'existait pas encore lors de l'augmentation.

    Retourne {track_id: [rir_names]}.
    """
    console.print("[yellow]Scan ChromaDB pour reconstruire l'historique RIR...[/yellow]")
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
        console.print("[dim]Aucun segment RIR trouvé dans ChromaDB.[/dim]")
        return {}

    console.print(f"[green]✓ {len(rir_map)} track(s) avec RIRs trouvés dans ChromaDB[/green]")

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
        console.print(f"[red]Erreur écriture metadata : {exc}[/red]")

    console.print(f"[green]✓ metadata.parquet mis à jour ({updated} tracks)[/green]\n")
    return {tid: list(rirs) for tid, rirs in rir_map.items()}


def _load_rir_done(df_meta: pd.DataFrame, collection_key: str) -> dict[str, list[str]]:
    """
    Lit la colonne rir_augmented et retourne {track_id: [rir_names déjà faits]}.
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
    """Ajoute rir_name dans rir_augmented[collection_key] pour ce track_id (atomique)."""
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
# Reconstruction index FAISS
# ---------------------------------------------------------------------------

def _rebuild_index(collection_key: str, chroma_client) -> None:
    from src.index.build_index import _build_for_method
    console.print("\n[yellow]Reconstruction de l'index FAISS...[/yellow]")
    _build_for_method(collection_key, config.INDEX_TYPE, chroma_client)
    console.print("[green]✓ Index FAISS reconstruit.[/green]")


# ---------------------------------------------------------------------------
# Point d'entrée public
# ---------------------------------------------------------------------------

def run_augment(
    method: str | None = None,
    tracks: str = "flowers",
    n_rir: int = 5,
    rir_dir: str = "data/rir",
    workers: int | None = None,
    device: str | None = None,
    rebuild_index: bool = True,
) -> None:
    """
    Augmente la base ChromaDB avec des versions dégradées par RIR.

    Args:
        method:        méthode d'embedding (None = config.EMBEDDING_METHOD).
        tracks:        'flowers', 'all' ou un track_id précis.
        n_rir:         nombre de RIRs à appliquer par track.
        rir_dir:       dossier contenant les fichiers RIR .wav.
        workers:       nombre de threads de téléchargement (None = config.DOWNLOAD_WORKERS).
        device:        device PyTorch : 'cpu' / 'cuda' / 'mps' (None = auto).
        rebuild_index: si True, reconstruit l'index FAISS après l'augmentation.
    """
    torch.set_num_threads(4)
    _stop_requested.clear()

    if method is None:
        method = config.EMBEDDING_METHOD
    if workers is None:
        workers = config.DOWNLOAD_WORKERS

    load_sr    = _get_load_sr(method)
    batch_size = _get_batch_size(method)
    rir_path   = Path(rir_dir)
    rir_path.mkdir(parents=True, exist_ok=True)

    console.print(Panel(
        f"[bold]Méthode     :[/bold] [cyan]{method}[/cyan]\n"
        f"[bold]Tracks      :[/bold] [cyan]{tracks}[/cyan]\n"
        f"[bold]N RIRs      :[/bold] [cyan]{n_rir}[/cyan]\n"
        f"[bold]Workers DL  :[/bold] [cyan]{workers}[/cyan]\n"
        f"[bold]RIR dir     :[/bold] [cyan]{rir_dir}[/cyan]",
        title="[bold cyan]Augmentation RIR[/bold cyan]",
        expand=False,
    ))

    # Pré-chargement modèle AVANT tout import faiss
    if method == "clap":
        from src.features.embeddings_audio import _load_clap
        console.print(f"[cyan]Chargement modèle {config.CLAP_MODEL_NAME} ({device})...[/cyan]")
        _load_clap(config.CLAP_MODEL_NAME, device=device)
        console.print("[green]✓ Modèle CLAP prêt.[/green]\n")
    elif method == "muq":
        from src.features.embeddings_audio import _load_muq
        console.print(f"[cyan]Chargement modèle {config.MUQ_MODEL_NAME} ({device})...[/cyan]")
        _load_muq(config.MUQ_MODEL_NAME)
        console.print("[green]✓ Modèle MuQ prêt.[/green]\n")
    elif method == "mert":
        from src.features.embeddings_audio import _load_mert
        console.print(f"[cyan]Chargement modèle {config.MERT_MODEL_NAME} ({device})...[/cyan]")
        _load_mert(config.MERT_MODEL_NAME)
        console.print("[green]✓ Modèle MERT prêt.[/green]\n")

    collection_key = config.get_collection_key(method)
    chroma_client  = chromadb.PersistentClient(path=str(ROOT / config.CHROMA_DIR))
    try:
        collection = chroma_client.get_collection(name=collection_key)
    except Exception:
        console.print(f"[red]Collection '{collection_key}' introuvable. Lance d'abord l'ingestion.[/red]")
        return

    def _on_interrupt(sig, frame):
        console.print("\n[yellow]⚠ Interruption détectée — arrêt propre en cours...[/yellow]")
        _stop_requested.set()

    signal.signal(signal.SIGINT,  _on_interrupt)
    signal.signal(signal.SIGTERM, _on_interrupt)

    console.print("[yellow]Chargement des RIRs...[/yellow]")
    rirs = _load_rirs(rir_path, n_rir, sr=22050)
    if not rirs:
        console.print("[red]Aucune RIR disponible. Abandon.[/red]")
        return

    t_rir = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
    t_rir.add_column("RIR", width=38)
    t_rir.add_column("Durée", justify="right", width=12)
    for name, rir in rirs:
        t_rir.add_row(name, f"{len(rir) / 22050 * 1000:.0f} ms")
    console.print(t_rir)

    meta_path = ROOT / config.METADATA_PATH
    df_meta   = pd.read_parquet(meta_path)

    if tracks == "flowers":
        df_all = df_meta[df_meta["track_id"] == FLOWERS_ID]
    elif tracks == "all":
        df_all = df_meta
    else:
        df_all = df_meta[df_meta["track_id"] == tracks]

    if df_all.empty:
        console.print(f"[red]Aucun track trouvé pour '{tracks}'.[/red]")
        return

    # Backfill automatique si tracking absent
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
            f"[dim]{n_already_done} track(s) ignorés[/dim] "
            f"[dim](toutes les {len(rirs)} RIRs déjà appliquées)[/dim]"
        )

    if df_targets.empty:
        console.print("[green]✓ Tout est déjà augmenté — rien à faire.[/green]")
        return

    total     = len(df_targets)
    meta_lock = threading.Lock()

    console.print(f"\n[bold]{total} track(s) à augmenter[/bold] × {len(rirs)} RIRs\n")

    # ── Étage 1 : téléchargements ─────────────────────────────────────────────
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

    # ── Étage 2 : convolution RIR ─────────────────────────────────────────────
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

    # ── Étage 3 : embedding GPU + sauvegarde ChromaDB ────────────────────────
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
                progress.update(task_detail, description="[red]✗ Échec[/red]", visible=True)
                n_failed += 1
                done     += 1
                progress.update(task_tracks, completed=done)
                continue

            if degraded == []:
                progress.update(task_detail, description="[dim]déjà augmenté[/dim]", visible=True)
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
        f"[bold]Segments ajoutés :[/bold] [green]{n_added}[/green]\n"
        f"[bold]Tracks ignorés   :[/bold] [dim]{n_skipped}[/dim]\n"
        f"[bold]Tracks échoués   :[/bold] [yellow]{n_failed}[/yellow]\n"
        f"[bold]Total ChromaDB   :[/bold] [cyan]{collection.count()}[/cyan] segments"
        + ("\n[yellow]⚠ Arrêt anticipé — reprends avec la même commande[/yellow]" if interrupted else ""),
        title="[bold green]Augmentation terminée[/bold green]" if not interrupted
              else "[bold yellow]Augmentation interrompue[/bold yellow]",
        expand=False,
    ))

    if rebuild_index and n_added > 0:
        _rebuild_index(collection_key, chroma_client)
    elif n_added == 0:
        console.print("[dim]Aucun segment ajouté — index FAISS non modifié.[/dim]")
