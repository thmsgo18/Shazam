"""
src/ingestion/ingest.py


Main ingestion pipeline: Spotify CSV → YouTube → Embeddings + Fingerprints → FAISS.


Reads one or more Kaggle Spotify CSV files, performs YouTube matching, downloads audio
into RAM, computes embeddings and fingerprints, and saves to ChromaDB / SQLite.
No audio file is stored on disk.


Public entry point: run_ingest(csv_paths=None)
"""


from __future__ import annotations


import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait as fut_wait, FIRST_COMPLETED
from datetime import datetime
from pathlib import Path


os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


import warnings
warnings.filterwarnings("ignore", message=".*upsample_bicubic2d.*", category=UserWarning)


import chromadb
import librosa
import numpy as np
import pandas as pd
import torch


from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress,
    SpinnerColumn, TextColumn, TimeElapsedColumn,
)


from src import config
from src.audio.preprocessing import iter_segments
from src.features.embeddings_audio import embed_segment, muq_batch_embeddings, clap_batch_embeddings
from src.features.fingerprint import extract_fingerprint
from src.utils.youtube import download_audio_search
from src.utils.metadata import atomic_write_parquet
from src.utils.fingerprints_db import (
    fp_init, fp_load_ids, fp_save, fp_delete, fp_migrate_from_pkl,
)


ROOT          = Path(__file__).resolve().parents[2]
FEATURES_DIR  = ROOT / "data" / "features"
PROCESSED_DIR = ROOT / "data" / "processed"


POSSIBLE_TITLE_COLS  = ["track_name", "name", "title", "song", "track"]
POSSIBLE_ARTIST_COLS = ["artist_name", "artists", "artist", "performer", "track_artist"]


KAGGLE_DATASET = "anxods/spotify-top-50-playlist-songs-anxods"
KAGGLE_DIR     = ROOT / "data" / "kaggle" / "data"


console = Console()



# ===========================================================================
# CSV — reading Spotify metadata
# ===========================================================================


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Returns the first column in the DataFrame that matches one of the candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    return None



def get_csv_files(csv_path: str | Path) -> list[Path]:
    """Returns the list of CSV files to process (single file or all CSVs in a directory)."""
    p = Path(csv_path)
    if p.is_dir():
        csvs = sorted(p.glob("*.csv"))
        if not csvs:
            console.print(f"[red]No CSV file found in {p}[/red]")
            sys.exit(1)
        return csvs
    if not p.exists():
        console.print(f"[red]File not found: {p}[/red]")
        sys.exit(1)
    return [p]



def normalize_title(title: str) -> str:
    """
    Normalizes a title for deduplication.

    Removes version markers (Taylor's Version, Radio Edit, Live…)
    without touching feats or remixes.
    """
    t = title.strip()
    t = re.sub(r"\s*\([^)]*version\b[^)]*\)", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*\(from the vault\)", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*-\s*live\b.*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*\((radio edit|single version|album version)\)", "", t, flags=re.IGNORECASE)
    return t.strip()



def load_tracks_from_csv(csv_path: Path) -> list[dict]:
    """Reads a Kaggle CSV and returns all unique tracks (artist + title)."""
    df = pd.read_csv(csv_path)
    title_col  = find_column(df, POSSIBLE_TITLE_COLS)
    artist_col = find_column(df, POSSIBLE_ARTIST_COLS)


    if title_col is None or artist_col is None:
        console.print(f"[red]Title/artist columns not found in {csv_path}[/red]")
        return []


    df = df[[title_col, artist_col]].dropna()
    df["_title_norm"] = df[title_col].apply(normalize_title)
    df = df.drop_duplicates(subset=["_title_norm", artist_col])
    df = df.drop(columns=["_title_norm"])


    tracks = []
    for row in df.itertuples(index=False):
        title  = str(getattr(row, title_col)).strip()
        artist = str(getattr(row, artist_col)).strip()
        if artist.startswith("["):
            artist = re.sub(r"[\[\]'\"]", "", artist).split(",")[0].strip()
        tracks.append({"title": title, "artist": artist, "source": csv_path.name})


    return tracks



def download_kaggle_csvs() -> None:
    """Automatically downloads Kaggle CSVs if not already present."""
    if KAGGLE_DIR.exists() and list(KAGGLE_DIR.glob("*.csv")):
        console.print(f"[green]Kaggle CSVs already present in {KAGGLE_DIR}[/green]")
        return


    console.print("[bold]Downloading Kaggle CSVs...[/bold]")
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)


    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", tmpdir, "--unzip"],
            timeout=120,
        )
        if result.returncode != 0:
            console.print("[red]Kaggle download failed. Check your kaggle.json API key.[/red]")
            sys.exit(1)


        for csv_file in Path(tmpdir).rglob("*.csv"):
            dest = KAGGLE_DIR / csv_file.name
            dest.write_bytes(csv_file.read_bytes())
            console.print(f"  [green]✓[/green] {csv_file.name}")


    console.print(f"[green]CSVs downloaded to {KAGGLE_DIR}[/green]\n")



# ===========================================================================
# Method utilities
# ===========================================================================


def get_method_key(method: str) -> str:
    """
    Returns the full key identifying both the method AND the model.

    Examples:
        "mfcc" → "mfcc"
        "clap" → "clap:laion/clap-htsat-unfused"
        "muq"  → "muq:OpenMuQ/MuQ-large-msd-iter"
    """
    if method == "clap":
        return f"clap:{config.CLAP_MODEL_NAME}"
    if method == "muq":
        return f"muq:{config.MUQ_MODEL_NAME}"
    return method



def load_already_processed(method: str) -> set[tuple[str, str]]:
    """
    Returns the set of (artist, title) already processed for the given method.
    Source of truth: embedded_methods column in metadata.parquet.
    """
    meta_path = PROCESSED_DIR / "metadata.parquet"
    if not meta_path.exists():
        return set()


    df_meta = pd.read_parquet(meta_path)
    if "embedded_methods" not in df_meta.columns:
        return set()
    if "artist" not in df_meta.columns or "title" not in df_meta.columns:
        return set()


    method_key = get_method_key(method)
    return {
        (str(r.artist).lower(), str(r.title).lower())
        for r in df_meta.itertuples()
        if (hasattr(r.embedded_methods, "__iter__")
            and not isinstance(r.embedded_methods, str)
            and method_key in r.embedded_methods)
    }



# ===========================================================================
# RAM pipeline — immediate save after each track
# ===========================================================================


def _save_track(
    track_id: str,
    method: str,
    track_embeddings: list,
    track_segments: list[dict],
    new_fp_hashes: set | None,
    metadata_row: dict,
    collection,
    fp_db: Path,
    meta_path: Path,
) -> None:
    """
    Saves a track to ChromaDB + SQLite + metadata.parquet.

    In case of a previous partial crash (track_id already present), old
    segments are deleted and rewritten cleanly — no duplicates ever.
    """
    new_emb = np.vstack(track_embeddings).astype(np.float32)


    # Delete old segments if partial crash occurred
    existing = collection.get(where={"track_id": {"$eq": track_id}})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])


    collection.add(
        embeddings=new_emb.tolist(),
        ids=[f"{track_id}_{i}" for i in range(len(new_emb))],
        metadatas=[
            {"track_id": track_id, "start_s": float(s["start_s"])}
            for s in track_segments
        ],
    )


    if new_fp_hashes is not None:
        fp_save(fp_db, track_id, new_fp_hashes)


    if meta_path.exists():
        df_meta = pd.read_parquet(meta_path)
        if track_id in set(df_meta["track_id"]):
            idx = df_meta.index[df_meta["track_id"] == track_id][0]
            current = df_meta.at[idx, "embedded_methods"]
            current = list(current) if hasattr(current, "__iter__") and not isinstance(current, str) else []
            method_key = get_method_key(method)
            if method_key not in current:
                df_meta.at[idx, "embedded_methods"] = current + [method_key]
        else:
            df_meta = pd.concat([df_meta, pd.DataFrame([metadata_row])], ignore_index=True)
    else:
        df_meta = pd.DataFrame([metadata_row])


    atomic_write_parquet(meta_path, df_meta)



def process_in_ram(tracks: list[dict], csv_sources: list[str]) -> None:
    """
    Main pipeline: downloads audio into RAM, computes embeddings + fingerprints,
    and saves to ChromaDB / fingerprints.db / metadata.parquet.

    Saves immediately after each track — clean resumption on interruption.
    """
    torch.set_num_threads(4)


    method = config.EMBEDDING_METHOD


    batch_size = {"clap": config.CLAP_BATCH_SIZE,
                  "muq":  config.MUQ_BATCH_SIZE,
                  "mert": config.MERT_BATCH_SIZE}.get(method, 1)
    load_sr    = {"clap": config.CLAP_SAMPLE_RATE,
                  "muq":  config.MUQ_SAMPLE_RATE,
                  "mert": config.MERT_SAMPLE_RATE}.get(method, config.SAMPLE_RATE)
    win_s      = config.SEGMENT_WIN_S
    hop_s      = config.SEGMENT_HOP_S
    min_win    = config.SEGMENT_MIN_WIN


    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)


    fp_db     = FEATURES_DIR / "fingerprints.db"
    meta_path = PROCESSED_DIR / "metadata.parquet"


    # Automatic one-shot migration: fingerprints.pkl → fingerprints.db
    fp_pkl = FEATURES_DIR / "fingerprints.pkl"
    if fp_pkl.exists() and not fp_db.exists():
        console.print("[yellow]Migrating fingerprints.pkl → fingerprints.db…[/yellow]")
        n = fp_migrate_from_pkl(fp_pkl, fp_db)
        console.print(f"[green]✓ {n} fingerprints migrated.[/green]")


    fp_init(fp_db)
    existing_fp_ids: set[str] = fp_load_ids(fp_db)


    collection_key = config.get_collection_key(method)
    chroma_client  = chromadb.PersistentClient(path=str(ROOT / config.CHROMA_DIR))
    collection     = chroma_client.get_or_create_collection(
        name=collection_key, metadata={"hnsw:space": "cosine"},
    )


    if torch.cuda.is_available():
        device       = "cuda"
        device_label = "[bold green]GPU (CUDA)[/bold green]"
    elif torch.backends.mps.is_available():
        device       = "mps"
        device_label = "[bold green]GPU (MPS — Apple Silicon)[/bold green]"
    else:
        device       = "cpu"
        device_label = "[yellow]CPU[/yellow]"


    console.print(Panel(
        f"[bold]Sources  :[/bold] {', '.join(csv_sources)}\n"
        f"[bold]Method   :[/bold] [cyan]{method}[/cyan]\n"
        f"[bold]Device   :[/bold] {device_label}\n"
        f"[bold]Tracks   :[/bold] {len(tracks)}\n"
        f"[bold]Mode     :[/bold] [green]RAM (no audio file stored on disk)[/green]",
        title="[bold cyan]Download + Build Pipeline[/bold cyan]",
        expand=False,
    ))


    # Model pre-loading
    if method == "clap":
        from src.features.embeddings_audio import _load_clap
        console.print(f"[cyan]Loading model {config.CLAP_MODEL_NAME} on {device}...[/cyan]")
        _load_clap(config.CLAP_MODEL_NAME, device=device)
        console.print("[green]✓ CLAP model ready.[/green]\n")
    elif method == "muq":
        from src.features.embeddings_audio import _load_muq
        console.print(f"[cyan]Loading model {config.MUQ_MODEL_NAME} on {device}...[/cyan]")
        _load_muq(config.MUQ_MODEL_NAME, device=device)
        console.print("[green]✓ MuQ model ready.[/green]\n")
    elif method == "mert":
        from src.features.embeddings_audio import _load_mert
        console.print(f"[cyan]Loading model {config.MERT_MODEL_NAME} on {device}...[/cyan]")
        _load_mert(config.MERT_MODEL_NAME, device=device)
        console.print("[green]✓ MERT model ready.[/green]\n")


    saved_count = 0
    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    ]


    with Progress(*progress_columns, console=console) as progress:
        dataset_task = (
            progress.add_task("[cyan]Tracks", total=len(tracks))
            if config.PROGRESS_DATASET else None
        )
        dl_status_task = progress.add_task("[yellow]⬇ Waiting...", total=None)
        track_task = (
            progress.add_task("", total=1, visible=False)
            if config.PROGRESS_TRACK else None
        )


        def _download_task(track: dict) -> tuple:
            tmpdir, audio_path, url, dl_error = download_audio_search(
                track["artist"], track["title"]
            )
            return tmpdir, audio_path, url, track, dl_error


        lookahead  = config.DOWNLOAD_WORKERS + 2
        track_iter = iter(tracks)
        dl_pool    = ThreadPoolExecutor(max_workers=config.DOWNLOAD_WORKERS)


        try:
            pending: set = set()
            for t in track_iter:
                pending.add(dl_pool.submit(_download_task, t))
                if len(pending) >= lookahead:
                    break


            while pending:
                finished, pending = fut_wait(pending, return_when=FIRST_COMPLETED)


                slots = lookahead - len(pending)
                for t in track_iter:
                    pending.add(dl_pool.submit(_download_task, t))
                    slots -= 1
                    if slots <= 0:
                        break


                for future in finished:
                    tmpdir, audio_path, youtube_url, track, dl_error = future.result()
                    artist = track["artist"]
                    title  = track["title"]
                    label  = f"{artist} — {title}"


                    if audio_path is None:
                        ts = datetime.now().strftime("%H:%M:%S")
                        console.print(f"[red]  ✗ [{ts}] {label} — {dl_error}[/red]")
                        if dataset_task is not None:
                            progress.advance(dataset_task)
                        continue


                    progress.update(dl_status_task, description=f"[yellow]⬇ {label[:55]}")


                    try:
                        waveform, sr = librosa.load(audio_path, sr=load_sr, mono=True)
                    except Exception:
                        ts = datetime.now().strftime("%H:%M:%S")
                        console.print(f"[red]  ✗ [{ts}] Audio read failed: {label}[/red]")
                        if dataset_task is not None:
                            progress.advance(dataset_task)
                        continue
                    finally:
                        shutil.rmtree(tmpdir, ignore_errors=True)


                    track_id = hashlib.md5(
                        f"{artist.lower()}_{title.lower()}".encode()
                    ).hexdigest()


                    if sr != config.SAMPLE_RATE:
                        _wf_std  = librosa.resample(waveform, orig_sr=sr, target_sr=config.SAMPLE_RATE)
                        duration_s = len(_wf_std) / config.SAMPLE_RATE
                    else:
                        duration_s = len(waveform) / sr


                    metadata_row = {
                        "track_id":         track_id,
                        "title":            title,
                        "artist":           artist,
                        "duration":         duration_s,
                        "source":           track["source"],
                        "url":              youtube_url,
                        "embedded_methods": [get_method_key(method)],
                        "album":            None,
                        "release_date":     None,
                        "genre":            None,
                        "cover_url":        None,
                    }


                    new_fp_hashes: set | None = None
                    if method not in ("muq", "clap") and track_id not in existing_fp_ids:
                        waveform_fp = (
                            librosa.resample(waveform, orig_sr=sr, target_sr=config.SAMPLE_RATE)
                            if sr != config.SAMPLE_RATE else waveform
                        )
                        new_fp_hashes = extract_fingerprint(waveform_fp, config.SAMPLE_RATE)
                        existing_fp_ids.add(track_id)


                    segs = list(iter_segments(waveform, sr, win_s, hop_s, min_win))


                    if track_task is not None:
                        progress.update(
                            track_task,
                            description=f"[green]{label[:50]}",
                            total=len(segs),
                            completed=0,
                            visible=True,
                        )


                    track_embeddings: list = []
                    track_segments:   list = []


                    if method in ("muq", "clap", "mert"):
                        for i in range(0, len(segs), batch_size):
                            batch = segs[i : i + batch_size]
                            if method == "muq":
                                embs = muq_batch_embeddings(
                                    [seg for _, seg in batch], sr=sr,
                                    model_name=config.MUQ_MODEL_NAME,
                                )
                            elif method == "mert":
                                from src.features.embeddings_audio import mert_batch_embeddings
                                embs = mert_batch_embeddings(
                                    [seg for _, seg in batch], sr=sr,
                                    model_name=config.MERT_MODEL_NAME,
                                )
                            else:
                                embs = clap_batch_embeddings(
                                    [seg for _, seg in batch], sr=sr,
                                    model_name=config.CLAP_MODEL_NAME,
                                )
                            for j, (start_s, _) in enumerate(batch):
                                track_embeddings.append(embs[j])
                                track_segments.append({"track_id": track_id, "start_s": float(start_s)})
                            if track_task is not None:
                                progress.advance(track_task, advance=len(batch))
                    else:
                        for start_s, seg in segs:
                            emb = embed_segment(
                                seg, sr, method=method,
                                clap_model_name=config.CLAP_MODEL_NAME,
                                muq_model_name=config.MUQ_MODEL_NAME,
                            )
                            track_embeddings.append(emb)
                            track_segments.append({"track_id": track_id, "start_s": float(start_s)})
                            if track_task is not None:
                                progress.advance(track_task)


                    if method in ("muq", "clap", "mert") and track_id not in existing_fp_ids:
                        waveform_fp = (
                            librosa.resample(waveform, orig_sr=sr, target_sr=config.SAMPLE_RATE)
                            if sr != config.SAMPLE_RATE else waveform
                        )
                        new_fp_hashes = extract_fingerprint(waveform_fp, config.SAMPLE_RATE)
                        existing_fp_ids.add(track_id)


                    if track_task is not None:
                        progress.update(track_task, visible=False)


                    if track_embeddings:
                        _save_track(
                            track_id, method,
                            track_embeddings, track_segments,
                            new_fp_hashes, metadata_row,
                            collection, fp_db, meta_path,
                        )
                        saved_count += 1


                    if dataset_task is not None:
                        progress.advance(dataset_task)


        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted — cancelling ongoing downloads...[/yellow]")
            for f in pending:
                f.cancel()
            dl_pool.shutdown(wait=False, cancel_futures=True)
            console.print(f"[green]{saved_count} track(s) already saved.[/green]")
            sys.exit(0)
        finally:
            dl_pool.shutdown(wait=False)


    if saved_count == 0:
        console.print("[red]No track processed — pipeline aborted.[/red]")
        sys.exit(1)


    total_meta = len(pd.read_parquet(meta_path)) if meta_path.exists() else 0
    total_seg  = collection.count()
    sample     = collection.get(limit=1, include=["embeddings"])
    emb_list   = sample.get("embeddings")
    emb_dim    = len(emb_list[0]) if emb_list is not None and len(emb_list) > 0 else "?"


    console.print(Panel(
        f"[bold]New tracks       :[/bold] {saved_count}\n"
        f"[bold]Total in DB      :[/bold] {total_meta}\n"
        f"[bold]Total segments   :[/bold] {total_seg}\n"
        f"[bold]Embedding dim    :[/bold] {emb_dim}\n"
        f"[bold]Fingerprints     :[/bold] {len(existing_fp_ids)} tracks",
        title="[bold green]Embeddings + Fingerprints — OK[/bold green]",
        expand=False,
    ))



# ===========================================================================
# Public entry point
# ===========================================================================


def run_ingest(csv_paths: tuple[str, ...] | None = None) -> None:
    """
    Reads Kaggle CSVs, downloads audio into RAM and builds the database.

    Args:
        csv_paths: tuple of CSV file paths or directories. If None or empty,
                   automatically uses all available Kaggle CSVs.
    """
    if not csv_paths:
        download_kaggle_csvs()
        csv_files = get_csv_files(str(KAGGLE_DIR))
    else:
        csv_files = []
        for p in csv_paths:
            csv_files.extend(get_csv_files(p))


    console.print(f"[bold]{len(csv_files)} CSV file(s) detected:[/bold]")
    for f in csv_files:
        console.print(f"  • {f}")


    method = config.EMBEDDING_METHOD
    already_processed = load_already_processed(method)
    if already_processed:
        console.print(f"\n[yellow]{len(already_processed)} track(s) already processed with '{method}' → skipped[/yellow]")


    all_tracks = []
    seen       = set()
    skipped    = 0
    for csv_file in csv_files:
        for t in load_tracks_from_csv(csv_file):
            key = (t["artist"].lower(), t["title"].lower())
            if key in already_processed:
                skipped += 1
                continue
            if key not in seen:
                seen.add(key)
                all_tracks.append(t)


    if skipped > 0:
        console.print(f"[yellow]{skipped} track(s) skipped (already in database)[/yellow]")


    csv_sources = [f.name for f in csv_files]


    if not all_tracks:
        console.print("[green]All tracks are already in the database.[/green]")
    else:
        console.print(f"\n[bold]{len(all_tracks)} new track(s) to process.[/bold]\n")
        process_in_ram(all_tracks, csv_sources)


    # Build FAISS index — always at the end
    from src.index.build_index import _build_for_method
    chroma_client = chromadb.PersistentClient(path=str(ROOT / config.CHROMA_DIR))
    collection_key = config.get_collection_key(method)
    console.print("\n[bold]▶ Building FAISS index[/bold]")
    _build_for_method(collection_key, config.INDEX_TYPE, chroma_client)