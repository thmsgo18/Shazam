"""
scripts/download_music.py

But : lire un ou plusieurs CSV Kaggle Spotify, faire le matching YouTube,
      télécharger l'audio en RAM, calculer embeddings + fingerprints,
      et construire l'index FAISS. Aucun MP3 n'est stocké sur disque.

Prérequis :
    pip install pandas yt-dlp rich
    brew install ffmpeg

Téléchargement du CSV Kaggle :
    pip install kaggle
    kaggle datasets download -d anxods/spotify-top-50-playlist-songs-anxods
    unzip spotify-top-50-playlist-songs-anxods.zip -d data/kaggle/

Usage :
    # Un seul CSV
    python scripts/download_music.py --csv data/kaggle/data/spotify-streaming-top-50-world.csv --n 50

    # Dossier entier (tous les CSV fusionnés)
    python scripts/download_music.py --csv data/kaggle/data/ --n 50
"""

from __future__ import annotations

import hashlib
import os
import pickle
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import click
import librosa
import numpy as np
import pandas as pd
import torch

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress,
    SpinnerColumn, TextColumn, TimeElapsedColumn
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.audio.preprocessing import iter_segments
from src.features.embeddings_audio import embed_segment, muq_batch_embeddings
from src.features.fingerprint import extract_fingerprint

RAW_DIR = Path("data/raw")
FEATURES_DIR = Path("data/features")
PROCESSED_DIR = Path("data/processed")

POSSIBLE_TITLE_COLS  = ["track_name", "name", "title", "song", "track"]
POSSIBLE_ARTIST_COLS = ["artist_name", "artists", "artist", "performer", "track_artist"]

console = Console()


# ===========================================================================
# CSV — lecture des métadonnées Spotify
# ===========================================================================

def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def get_csv_files(csv_path: str) -> list[Path]:
    """Retourne la liste des CSV à traiter (fichier unique ou tous les CSV d'un dossier)."""
    p = Path(csv_path)
    if p.is_dir():
        csvs = sorted(p.glob("*.csv"))
        if not csvs:
            console.print(f"[red]Aucun fichier CSV trouvé dans {p}[/red]")
            sys.exit(1)
        return csvs
    if not p.exists():
        console.print(f"[red]Fichier introuvable : {p}[/red]")
        sys.exit(1)
    return [p]


def load_tracks_from_csv(csv_path: Path) -> list[dict]:
    """Lit un CSV Kaggle et retourne tous les titres uniques."""
    df = pd.read_csv(csv_path)
    title_col  = find_column(df, POSSIBLE_TITLE_COLS)
    artist_col = find_column(df, POSSIBLE_ARTIST_COLS)

    if title_col is None or artist_col is None:
        console.print(f"[red]Colonnes titre/artiste introuvables dans {csv_path}[/red]")
        return []

    df = df[[title_col, artist_col]].dropna().drop_duplicates(subset=[title_col, artist_col])

    tracks = []
    for row in df.itertuples(index=False):
        title  = str(getattr(row, title_col)).strip()
        artist = str(getattr(row, artist_col)).strip()
        if artist.startswith("["):
            artist = re.sub(r"[\[\]'\"]", "", artist).split(",")[0].strip()
        tracks.append({"title": title, "artist": artist, "source": csv_path.name})

    return tracks


# ===========================================================================
# Téléchargement
# ===========================================================================

def download_to_disk(artist: str, title: str, dest_dir: Path) -> bool:
    """Télécharge le MP3 dans dest_dir (ancien comportement --store-audio)."""
    safe_name = "".join(
        c if c.isalnum() or c in "-_ " else "_"
        for c in f"{artist}_{title}"
    ).strip()
    filename  = f"{safe_name[:60]}.mp3"
    dest_path = dest_dir / filename

    if dest_path.exists():
        return True

    query = f"{artist} {title} official audio"
    cmd = [
        "yt-dlp", f"ytsearch1:{query}",
        "--extract-audio", "--audio-format", "mp3",
        "--audio-quality", "5",
        "--output", str(dest_dir / f"{safe_name[:60]}.%(ext)s"),
        "--quiet", "--no-warnings", "--socket-timeout", "30",
    ]
    try:
        result = subprocess.run(cmd, timeout=120)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def download_to_ram(artist: str, title: str, target_sr: int) -> tuple[np.ndarray, int] | tuple[None, None]:
    """
    Télécharge l'audio dans un dossier temporaire, le charge en RAM, puis supprime le fichier.
    Aucun MP3 n'est conservé sur disque après l'appel.
    """
    query = f"{artist} {title} official audio"
    cmd = [
        "yt-dlp", f"ytsearch1:{query}",
        "--extract-audio", "--audio-format", "mp3",
        "--audio-quality", "5",
        "--output", "%(id)s.%(ext)s",
        "--quiet", "--no-warnings", "--socket-timeout", "30",
    ]
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(cmd, timeout=120, cwd=tmpdir)
            if result.returncode != 0:
                return None, None

            files = list(Path(tmpdir).glob("*.mp3"))
            if not files:
                return None, None

            waveform, sr = librosa.load(str(files[0]), sr=target_sr, mono=True)
            return waveform, sr
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return None, None


# ===========================================================================
# Pipeline RAM — traitement sans stockage MP3
# ===========================================================================

def process_in_ram(tracks: list[dict], csv_sources: list[str]) -> None:
    """
    Pour chaque morceau : télécharge en RAM, calcule embeddings + fingerprint,
    sauvegarde embeddings.npy / segments.parquet / fingerprints.pkl / metadata.parquet.
    Aucun MP3 n'est écrit dans data/raw/.
    """
    torch.set_num_threads(4)

    method     = config.EMBEDDING_METHOD
    batch_size = config.MUQ_BATCH_SIZE

    if method == "clap":
        load_sr = config.CLAP_SAMPLE_RATE
    elif method == "muq":
        load_sr = config.MUQ_SAMPLE_RATE
    else:
        load_sr = config.SAMPLE_RATE

    win_s   = config.SEGMENT_WIN_S
    hop_s   = config.SEGMENT_HOP_S
    min_win = config.SEGMENT_MIN_WIN

    # Sauvegarder la source pour l'affichage dans build_segment_embeddings.py
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "source.txt").write_text(", ".join(csv_sources))

    console.print(Panel(
        f"[bold]Sources  :[/bold] {', '.join(csv_sources)}\n"
        f"[bold]Méthode  :[/bold] [cyan]{method}[/cyan]\n"
        f"[bold]Tracks   :[/bold] {len(tracks)}\n"
        f"[bold]Mode     :[/bold] [green]RAM (aucun MP3 stocké)[/green]",
        title="[bold cyan]Download + Build Pipeline[/bold cyan]",
        expand=False
    ))

    segments_rows   = []
    embeddings_list = []
    fingerprints    = {}
    metadata_rows   = []
    segment_id      = 0

    batch_segments  = []
    batch_meta      = []

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

        for track in tracks:
            artist = track["artist"]
            title  = track["title"]
            label  = f"{artist} — {title}"

            # Téléchargement en RAM
            waveform, sr = download_to_ram(artist, title, load_sr)

            if waveform is None:
                console.print(f"[red]  ✗ Échec : {label}[/red]")
                if dataset_task is not None:
                    progress.advance(dataset_task)
                continue

            # track_id stable basé sur le contenu audio
            track_id = hashlib.md5(waveform.tobytes()[:8192]).hexdigest()

            metadata_rows.append({
                "track_id": track_id,
                "title":    title,
                "artist":   artist,
                "source":   track["source"],
                "duration": len(waveform) / sr,
            })

            # Fingerprint (toujours à SAMPLE_RATE)
            if sr != config.SAMPLE_RATE:
                waveform_fp = librosa.resample(waveform, orig_sr=sr, target_sr=config.SAMPLE_RATE)
            else:
                waveform_fp = waveform
            fingerprints[track_id] = extract_fingerprint(waveform_fp, config.SAMPLE_RATE)

            # Segmentation
            segs = list(iter_segments(waveform, sr, win_s, hop_s, min_win))

            track_task = (
                progress.add_task(f"[green]{label[:50]}", total=len(segs))
                if config.PROGRESS_TRACK else None
            )

            for start_s, seg in segs:
                if method == "muq":
                    batch_segments.append(seg)
                    batch_meta.append((track_id, float(start_s)))

                    if len(batch_segments) >= batch_size:
                        embs = muq_batch_embeddings(
                            batch_segments, sr=sr, model_name=config.MUQ_MODEL_NAME
                        )
                        for i in range(embs.shape[0]):
                            embeddings_list.append(embs[i])
                            t_id, st = batch_meta[i]
                            segments_rows.append({
                                "segment_id": segment_id,
                                "track_id":   t_id,
                                "start_s":    st
                            })
                            segment_id += 1
                        batch_segments.clear()
                        batch_meta.clear()
                else:
                    emb = embed_segment(
                        seg, sr, method=method,
                        clap_model_name=config.CLAP_MODEL_NAME,
                        muq_model_name=config.MUQ_MODEL_NAME
                    )
                    embeddings_list.append(emb)
                    segments_rows.append({
                        "segment_id": segment_id,
                        "track_id":   track_id,
                        "start_s":    float(start_s)
                    })
                    segment_id += 1

                if track_task is not None:
                    progress.advance(track_task)

            if track_task is not None:
                progress.remove_task(track_task)
            if dataset_task is not None:
                progress.advance(dataset_task)

            time.sleep(0.5)  # pause pour éviter le rate-limiting YouTube

    # Flush dernier batch MuQ incomplet
    if method == "muq" and len(batch_segments) > 0:
        embs = muq_batch_embeddings(batch_segments, sr=load_sr, model_name=config.MUQ_MODEL_NAME)
        for i in range(embs.shape[0]):
            embeddings_list.append(embs[i])
            t_id, st = batch_meta[i]
            segments_rows.append({"segment_id": segment_id, "track_id": t_id, "start_s": st})
            segment_id += 1

    if not embeddings_list:
        console.print("[red]Aucun morceau traité — pipeline annulé.[/red]")
        sys.exit(1)

    # Fusion avec les données existantes
    emb_path = FEATURES_DIR / f"embeddings_{method}.npy"
    seg_path = FEATURES_DIR / f"segments_{method}.parquet"
    fp_path  = FEATURES_DIR / "fingerprints.pkl"
    meta_path = PROCESSED_DIR / "metadata.parquet"

    new_emb = np.vstack(embeddings_list).astype(np.float32)

    if emb_path.exists():
        existing_emb = np.load(emb_path)
        # Décaler les segment_id pour éviter les collisions
        offset = len(existing_emb)
        for row in segments_rows:
            row["segment_id"] += offset
        emb_mat = np.vstack([existing_emb, new_emb]).astype(np.float32)
    else:
        emb_mat = new_emb

    np.save(emb_path, emb_mat)

    df_new_seg = pd.DataFrame(segments_rows)
    if seg_path.exists():
        df_existing_seg = pd.read_parquet(seg_path)
        df_segments = pd.concat([df_existing_seg, df_new_seg], ignore_index=True)
    else:
        df_segments = df_new_seg
    df_segments.to_parquet(seg_path, index=False)

    existing_fp = {}
    if fp_path.exists():
        with open(fp_path, "rb") as f:
            existing_fp = pickle.load(f)
    existing_fp.update(fingerprints)
    with open(fp_path, "wb") as f:
        pickle.dump(existing_fp, f)

    df_new_meta = pd.DataFrame(metadata_rows)
    if meta_path.exists():
        df_existing_meta = pd.read_parquet(meta_path)
        df_meta = pd.concat([df_existing_meta, df_new_meta], ignore_index=True)
    else:
        df_meta = df_new_meta
    df_meta.to_parquet(meta_path, index=False)

    console.print(Panel(
        f"[bold]Nouveaux tracks  :[/bold] {len(metadata_rows)}\n"
        f"[bold]Total en base    :[/bold] {len(df_meta)}\n"
        f"[bold]Total segments   :[/bold] {len(df_segments)}\n"
        f"[bold]Embedding dim    :[/bold] {emb_mat.shape[1]}\n"
        f"[bold]Fingerprints     :[/bold] {len(existing_fp)} tracks",
        title="[bold green]Embeddings + Fingerprints — OK[/bold green]",
        expand=False
    ))

    # Construction de l'index FAISS
    run_step("Construction de l'index FAISS", [sys.executable, "src/index/build_index.py"])


# ===========================================================================
# Pipeline ancien — avec stockage MP3
# ===========================================================================

def run_step(label: str, cmd: list[str]) -> None:
    console.print(f"\n[bold]▶ {label}[/bold]")
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1])
    if result.returncode != 0:
        console.print(f"[red]Échec : {label} (code {result.returncode})[/red]")
        sys.exit(result.returncode)


# ===========================================================================
# CLI
# ===========================================================================

KAGGLE_DATASET = "anxods/spotify-top-50-playlist-songs-anxods"
KAGGLE_DIR     = Path("data/kaggle/data")


def download_kaggle_csvs() -> None:
    """Télécharge automatiquement les CSV Kaggle si pas déjà présents."""
    if KAGGLE_DIR.exists() and list(KAGGLE_DIR.glob("*.csv")):
        console.print(f"[green]CSV Kaggle déjà présents dans {KAGGLE_DIR}[/green]")
        return

    console.print(f"[bold]Téléchargement des CSV Kaggle...[/bold]")
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", tmpdir, "--unzip"],
            timeout=120
        )
        if result.returncode != 0:
            console.print("[red]Échec du téléchargement Kaggle. Vérifie ta clé API kaggle.json.[/red]")
            sys.exit(1)

        # Copier uniquement les CSV dans KAGGLE_DIR
        for csv_file in Path(tmpdir).rglob("*.csv"):
            dest = KAGGLE_DIR / csv_file.name
            dest.write_bytes(csv_file.read_bytes())
            console.print(f"  [green]✓[/green] {csv_file.name}")

    console.print(f"[green]CSV téléchargés dans {KAGGLE_DIR}[/green]\n")


def load_already_processed(method: str) -> set[tuple[str, str]]:
    """
    Retourne l'ensemble des (artist, title) déjà traités POUR LA MÉTHODE DONNÉE.
    Vérifie segments_{method}.parquet et non metadata.parquet — un morceau peut
    être dans metadata sans avoir ses embeddings pour toutes les méthodes.
    """
    seg_path  = FEATURES_DIR / f"segments_{method}.parquet"
    meta_path = PROCESSED_DIR / "metadata.parquet"

    if not seg_path.exists() or not meta_path.exists():
        return set()

    # track_ids déjà indexés pour cette méthode
    seg_ids = set(pd.read_parquet(seg_path)["track_id"].unique())

    # Correspondance track_id → (artist, title) via metadata
    df_meta = pd.read_parquet(meta_path)
    if "artist" not in df_meta.columns or "title" not in df_meta.columns:
        return set()

    return {
        (str(r.artist).lower(), str(r.title).lower())
        for r in df_meta.itertuples()
        if r.track_id in seg_ids
    }


@click.command()
@click.option("--csv", "csv_paths", multiple=True,
              help="Fichier CSV ou dossier. Répétable. Si absent, utilise tous les CSV Kaggle.")
@click.option("--store-audio", is_flag=True, default=False,
              help="Sauvegarder les MP3 dans data/raw/ (ancien comportement).")
def main(csv_paths: tuple[str], store_audio: bool) -> None:
    """
    Lit un ou plusieurs CSV Kaggle Spotify, télécharge l'audio et construit la base.
    Si --csv est absent, utilise automatiquement tous les CSV Kaggle disponibles.
    Les morceaux déjà traités pour la méthode active sont automatiquement ignorés.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Collecter les fichiers CSV
    if not csv_paths:
        download_kaggle_csvs()
        csv_files = get_csv_files(str(KAGGLE_DIR))
    else:
        csv_files = []
        for p in csv_paths:
            csv_files.extend(get_csv_files(p))

    console.print(f"[bold]{len(csv_files)} fichier(s) CSV détecté(s) :[/bold]")
    for f in csv_files:
        console.print(f"  • {f}")

    # Charger les morceaux déjà traités POUR CETTE MÉTHODE pour les ignorer
    method = config.EMBEDDING_METHOD
    already_processed = load_already_processed(method)
    if already_processed:
        console.print(f"\n[yellow]{len(already_processed)} morceau(x) déjà traité(s) avec '{method}' → ignorés[/yellow]")

    # Charger tous les tracks en dédupliquant et en filtrant les déjà traités
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
        console.print(f"[yellow]{skipped} morceau(x) ignoré(s) (déjà dans la base)[/yellow]")

    if not all_tracks:
        console.print("[green]Tous les morceaux sont déjà dans la base. Rien à faire.[/green]")
        sys.exit(0)

    console.print(f"\n[bold]{len(all_tracks)} nouveaux morceaux à traiter.[/bold]\n")

    csv_sources = [f.name for f in csv_files]

    if store_audio:
        success = 0
        for i, track in enumerate(all_tracks, 1):
            console.print(f"[{i}/{len(all_tracks)}] {track['artist']} — {track['title']}")
            if download_to_disk(track["artist"], track["title"], RAW_DIR):
                success += 1
            time.sleep(1.0)

        console.print(f"\n{success}/{len(all_tracks)} morceaux téléchargés dans {RAW_DIR}/")
        if success == 0:
            sys.exit(1)

        run_step("1/3 — Métadonnées",  [sys.executable, "src/data_utils/build_metadata.py"])
        run_step("2/3 — Embeddings",   [sys.executable, "scripts/build_segment_embeddings.py"])
        run_step("3/3 — Index FAISS",  [sys.executable, "src/index/build_index.py"])
    else:
        process_in_ram(all_tracks, csv_sources)


if __name__ == "__main__":
    main()
