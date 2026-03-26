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
    python scripts/download_music.py --csv data/kaggle/data/spotify-streaming-top-50-world.csv

    # Dossier entier (tous les CSV fusionnés)
    python scripts/download_music.py --csv data/kaggle/data/

    # Sans --csv : utilise automatiquement tous les CSV Kaggle disponibles
    python scripts/download_music.py
"""

from __future__ import annotations

import hashlib
import os
import pickle
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
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

def _download_mp3(
    artist: str, title: str, retries: int = 3
) -> tuple[str, str, str, None] | tuple[None, None, None, str]:
    """
    Télécharge uniquement le MP3 via yt-dlp (subprocess seul, pas de numpy/librosa).
    Retourne (tmpdir, mp3_path, youtube_url, None) en cas de succès,
    ou (None, None, None, reason) en cas d'échec.
    """
    queries = [
        f"{artist} {title} official audio",
        f"{artist} {title}",
    ]
    base_cmd = [
        "yt-dlp",
        "--extract-audio", "--audio-format", "mp3",
        "--audio-quality", "5",
        "--output", "%(id)s.%(ext)s",
        "--quiet", "--no-warnings",
        "--socket-timeout", "30",
        "--extractor-args", "youtube:player_client=android",
    ]
    last_reason = "introuvable sur YouTube"
    for query in queries:
        cmd = [base_cmd[0], f"ytsearch1:{query}"] + base_cmd[1:]
        for attempt in range(retries):
            tmpdir = tempfile.mkdtemp()
            proc = None
            try:
                # start_new_session=True : yt-dlp + ffmpeg dans le même groupe de processus
                # → on peut tout tuer d'un coup en cas de timeout
                proc = subprocess.Popen(
                    cmd, cwd=tmpdir,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                )
                try:
                    _, stderr_bytes = proc.communicate(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    proc.wait()
                    shutil.rmtree(tmpdir, ignore_errors=True)
                    last_reason = "timeout"
                    if attempt < retries - 1:
                        time.sleep(3)
                        continue
                    break

                if proc.returncode != 0:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                    stderr = stderr_bytes.decode(errors="ignore")
                    if "Sign in" in stderr or "age" in stderr:
                        last_reason = "restriction d'âge YouTube"
                        break
                    if "unavailable" in stderr or "not available" in stderr:
                        last_reason = "vidéo indisponible dans cette région"
                        break
                    last_reason = "erreur yt-dlp"
                    if attempt < retries - 1:
                        time.sleep(3)
                        continue
                    break

                files = list(Path(tmpdir).glob("*.mp3"))
                if not files:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                    last_reason = "introuvable sur YouTube"
                    break

                youtube_url = f"https://www.youtube.com/watch?v={files[0].stem}"
                return tmpdir, str(files[0]), youtube_url, None
            except Exception as e:
                if proc is not None:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except Exception:
                        pass
                shutil.rmtree(tmpdir, ignore_errors=True)
                last_reason = str(e)
                break

    return None, None, None, last_reason


# ===========================================================================
# Pipeline RAM — traitement sans stockage MP3
# ===========================================================================

def _save_track(
    track_id: str,
    method: str,
    track_embeddings: list,
    track_segments: list[dict],
    new_fp: dict,
    metadata_row: dict,
    existing_fp: dict,
    emb_path: Path,
    seg_path: Path,
    fp_path: Path,
    meta_path: Path,
) -> None:
    """
    Sauvegarde un track sur disque immédiatement (append).
    Si le track_id existe déjà (crash partiel précédent), ses anciennes données
    sont supprimées et réécrites proprement — jamais de doublon.
    """
    new_emb = np.vstack(track_embeddings).astype(np.float32)

    # Charger embeddings existants (avec gestion corruption)
    if emb_path.exists():
        try:
            existing_emb = np.load(emb_path)
        except Exception:
            console.print("[yellow]⚠ embeddings.npy corrompu — réinitialisé.[/yellow]")
            existing_emb = np.empty((0, new_emb.shape[1]), dtype=np.float32)
    else:
        existing_emb = np.empty((0, new_emb.shape[1]), dtype=np.float32)

    # Charger segments existants
    if seg_path.exists():
        df_seg = pd.read_parquet(seg_path)
    else:
        df_seg = pd.DataFrame(columns=["segment_id", "track_id", "start_s"])

    # Écrasement si le track existait déjà partiellement (crash précédent)
    if track_id in df_seg["track_id"].values:
        old_seg_ids = df_seg[df_seg["track_id"] == track_id]["segment_id"].values
        df_seg = df_seg[df_seg["track_id"] != track_id].reset_index(drop=True)
        keep = np.ones(len(existing_emb), dtype=bool)
        valid_ids = old_seg_ids[old_seg_ids < len(existing_emb)]  # sécurité bounds
        keep[valid_ids] = False
        existing_emb = existing_emb[keep]
        df_seg["segment_id"] = range(len(df_seg))

    offset = len(existing_emb)

    # Assigner les nouveaux segment_ids
    df_new_seg = pd.DataFrame([
        {"segment_id": offset + i, "track_id": s["track_id"], "start_s": s["start_s"]}
        for i, s in enumerate(track_segments)
    ])

    # Écrire segments EN PREMIER — source de vérité pour l'overwrite au prochain run
    df_seg = pd.concat([df_seg, df_new_seg], ignore_index=True)
    df_seg.to_parquet(seg_path, index=False)

    # Puis embeddings
    emb_mat = np.vstack([existing_emb, new_emb]).astype(np.float32)
    np.save(emb_path, emb_mat)

    # Fingerprints
    existing_fp.update(new_fp)
    with open(fp_path, "wb") as f:
        pickle.dump(existing_fp, f)

    # Metadata — mise à jour embedded_methods si le track existe déjà
    if meta_path.exists():
        df_meta = pd.read_parquet(meta_path)
        if track_id in set(df_meta["track_id"]):
            idx = df_meta.index[df_meta["track_id"] == track_id][0]
            current = df_meta.at[idx, "embedded_methods"] or []
            if method not in current:
                df_meta.at[idx, "embedded_methods"] = list(current) + [method]
        else:
            df_meta = pd.concat([df_meta, pd.DataFrame([metadata_row])], ignore_index=True)
    else:
        df_meta = pd.DataFrame([metadata_row])
    df_meta.to_parquet(meta_path, index=False)


def process_in_ram(tracks: list[dict], csv_sources: list[str]) -> None:
    """
    Pour chaque morceau : télécharge en RAM, calcule embeddings + fingerprint,
    sauvegarde embeddings.npy / segments.parquet / fingerprints.pkl / metadata.parquet.
    La sauvegarde est faite après chaque track — reprise possible en cas d'interruption.
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

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    emb_path  = FEATURES_DIR / f"embeddings_{method}.npy"
    seg_path  = FEATURES_DIR / f"segments_{method}.parquet"
    fp_path   = FEATURES_DIR / "fingerprints.pkl"
    meta_path = PROCESSED_DIR / "metadata.parquet"

    # Charger les fingerprints existants pour éviter de les recalculer
    existing_fp: dict = {}
    if fp_path.exists():
        try:
            with open(fp_path, "rb") as f:
                existing_fp = pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            console.print("[yellow]⚠ fingerprints.pkl corrompu — réinitialisé.[/yellow]")
            existing_fp = {}

    console.print(Panel(
        f"[bold]Sources  :[/bold] {', '.join(csv_sources)}\n"
        f"[bold]Méthode  :[/bold] [cyan]{method}[/cyan]\n"
        f"[bold]Tracks   :[/bold] {len(tracks)}\n"
        f"[bold]Mode     :[/bold] [green]RAM (aucun MP3 stocké)[/green]",
        title="[bold cyan]Download + Build Pipeline[/bold cyan]",
        expand=False
    ))

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

        # Tâche Rich pour afficher les téléchargements en cours
        dl_status_task = progress.add_task("[yellow]⬇ En attente...", total=None)

        def _download_task(track: dict) -> tuple:
            """
            Télécharge uniquement le MP3 via subprocess — PAS de librosa/numpy ici.
            librosa.load sera appelé dans le thread principal pour éviter les deadlocks.
            """
            tmpdir, mp3_path, url, dl_error = _download_mp3(track["artist"], track["title"])
            return tmpdir, mp3_path, url, track, dl_error

        # Sliding window : on ne soumet que DOWNLOAD_WORKERS+2 futures à la fois
        # pour ne pas bombarder YouTube avec toutes les requêtes d'un coup.
        from concurrent.futures import wait as fut_wait, FIRST_COMPLETED

        lookahead  = config.DOWNLOAD_WORKERS + 2
        track_iter = iter(tracks)

        dl_pool = ThreadPoolExecutor(max_workers=config.DOWNLOAD_WORKERS)
        try:
            # Queue initiale
            pending: set = set()
            for t in track_iter:
                pending.add(dl_pool.submit(_download_task, t))
                if len(pending) >= lookahead:
                    break

            while pending:
                finished, pending = fut_wait(pending, return_when=FIRST_COMPLETED)

                # Remplir à nouveau jusqu'au lookahead
                slots = lookahead - len(pending)
                for t in track_iter:
                    pending.add(dl_pool.submit(_download_task, t))
                    slots -= 1
                    if slots <= 0:
                        break

                for future in finished:
                    tmpdir, mp3_path, youtube_url, track, dl_error = future.result()
                    artist = track["artist"]
                    title  = track["title"]
                    label  = f"{artist} — {title}"

                    if mp3_path is None:
                        ts = datetime.now().strftime("%H:%M:%S")
                        console.print(f"[red]  ✗ [{ts}] {label} — {dl_error}[/red]")
                        if dataset_task is not None:
                            progress.advance(dataset_task)
                        continue

                    # Mise à jour affichage track en cours
                    progress.update(dl_status_task, description=f"[yellow]⬇ {label[:55]}")

                    # librosa.load dans le thread principal — évite les deadlocks numpy
                    try:
                        waveform, sr = librosa.load(mp3_path, sr=load_sr, mono=True)
                    except Exception:
                        ts = datetime.now().strftime("%H:%M:%S")
                        console.print(f"[red]  ✗ [{ts}] Échec lecture audio : {label}[/red]")
                        if dataset_task is not None:
                            progress.advance(dataset_task)
                        continue
                    finally:
                        shutil.rmtree(tmpdir, ignore_errors=True)

                    # track_id stable basé sur (artist, title)
                    track_id = hashlib.md5(
                        f"{artist.lower()}_{title.lower()}".encode()
                    ).hexdigest()

                    metadata_row = {
                        "track_id":         track_id,
                        "title":            title,
                        "artist":           artist,
                        "duration":         len(waveform) / sr,
                        "source":           track["source"],
                        "url":              youtube_url,
                        "embedded_methods": [method],
                        "album":            None,
                        "release_date":     None,
                        "genre":            None,
                        "cover_url":        None,
                    }

                    # Fingerprint — seulement si pas déjà calculé
                    new_fp: dict = {}
                    if track_id not in existing_fp:
                        waveform_fp = (
                            librosa.resample(waveform, orig_sr=sr, target_sr=config.SAMPLE_RATE)
                            if sr != config.SAMPLE_RATE else waveform
                        )
                        new_fp[track_id] = extract_fingerprint(waveform_fp, config.SAMPLE_RATE)

                    # Segmentation + embedding
                    segs = list(iter_segments(waveform, sr, win_s, hop_s, min_win))

                    track_task = (
                        progress.add_task(f"[green]{label[:50]}", total=len(segs))
                        if config.PROGRESS_TRACK else None
                    )

                    track_embeddings: list = []
                    track_segments:   list = []
                    local_id = 0

                    if method == "muq":
                        for i in range(0, len(segs), batch_size):
                            batch = segs[i : i + batch_size]
                            embs  = muq_batch_embeddings(
                                [seg for _, seg in batch], sr=sr,
                                model_name=config.MUQ_MODEL_NAME,
                            )
                            for j, (start_s, _) in enumerate(batch):
                                track_embeddings.append(embs[j])
                                track_segments.append({
                                    "segment_id": local_id,
                                    "track_id":   track_id,
                                    "start_s":    float(start_s),
                                })
                                local_id += 1
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
                            track_segments.append({
                                "segment_id": local_id,
                                "track_id":   track_id,
                                "start_s":    float(start_s),
                            })
                            local_id += 1
                            if track_task is not None:
                                progress.advance(track_task)

                    if track_task is not None:
                        progress.remove_task(track_task)

                    # Sauvegarde immédiate sur disque
                    if track_embeddings:
                        _save_track(
                            track_id, method,
                            track_embeddings, track_segments,
                            new_fp, metadata_row, existing_fp,
                            emb_path, seg_path, fp_path, meta_path,
                        )
                        saved_count += 1

                    if dataset_task is not None:
                        progress.advance(dataset_task)

        except KeyboardInterrupt:
            console.print("\n[yellow]Interruption — annulation des téléchargements en cours...[/yellow]")
            for f in pending:
                f.cancel()
            dl_pool.shutdown(wait=False, cancel_futures=True)
            console.print(f"[green]{saved_count} track(s) déjà sauvegardé(s).[/green]")
            sys.exit(0)
        finally:
            dl_pool.shutdown(wait=False)

    if saved_count == 0:
        console.print("[red]Aucun morceau traité — pipeline annulé.[/red]")
        sys.exit(1)

    # Résumé final
    total_meta = len(pd.read_parquet(meta_path)) if meta_path.exists() else 0
    total_seg  = len(pd.read_parquet(seg_path))  if seg_path.exists()  else 0
    emb_dim    = np.load(emb_path).shape[1]      if emb_path.exists()  else 0

    console.print(Panel(
        f"[bold]Nouveaux tracks  :[/bold] {saved_count}\n"
        f"[bold]Total en base    :[/bold] {total_meta}\n"
        f"[bold]Total segments   :[/bold] {total_seg}\n"
        f"[bold]Embedding dim    :[/bold] {emb_dim}\n"
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
    Source de vérité : colonne embedded_methods dans metadata.parquet.
    Un morceau est ignoré si et seulement si method est dans sa liste embedded_methods.
    """
    meta_path = PROCESSED_DIR / "metadata.parquet"

    if not meta_path.exists():
        return set()

    df_meta = pd.read_parquet(meta_path)

    if "embedded_methods" not in df_meta.columns:
        return set()
    if "artist" not in df_meta.columns or "title" not in df_meta.columns:
        return set()

    return {
        (str(r.artist).lower(), str(r.title).lower())
        for r in df_meta.itertuples()
        if hasattr(r.embedded_methods, '__iter__') and not isinstance(r.embedded_methods, str) and method in r.embedded_methods
    }


@click.command()
@click.option("--csv", "csv_paths", multiple=True,
              help="Fichier CSV ou dossier. Répétable. Si absent, utilise tous les CSV Kaggle.")
def main(csv_paths: tuple[str]) -> None:
    """
    Lit un ou plusieurs CSV Kaggle Spotify, télécharge l'audio en RAM et construit la base.
    Si --csv est absent, utilise automatiquement tous les CSV Kaggle disponibles.
    Les morceaux déjà traités pour la méthode active sont automatiquement ignorés.
    """
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
    process_in_ram(all_tracks, csv_sources)


if __name__ == "__main__":
    main()
