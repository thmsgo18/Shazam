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
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Évite les deadlocks librosa/numpy/OpenBLAS quand un ThreadPoolExecutor tourne en parallèle
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import chromadb
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


# ===========================================================================
# Fingerprints — stockage SQLite
# ===========================================================================

def _fp_init(db_path: Path) -> None:
    """Crée la table fingerprints si elle n'existe pas encore."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fingerprints (
                track_id TEXT PRIMARY KEY,
                hashes   BLOB    NOT NULL,
                n_hashes INTEGER NOT NULL
            )
        """)


def _fp_load_ids(db_path: Path) -> set[str]:
    """Retourne l'ensemble des track_ids qui ont déjà un fingerprint."""
    if not db_path.exists():
        return set()
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT track_id FROM fingerprints").fetchall()
    return {r[0] for r in rows}


def _fp_save(db_path: Path, track_id: str, hashes: set) -> None:
    """Insère ou remplace le fingerprint d'un track (atomique par SQLite)."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO fingerprints VALUES (?, ?, ?)",
            (track_id, pickle.dumps(hashes), len(hashes)),
        )


def _fp_delete(db_path: Path, track_ids: set[str]) -> int:
    """Supprime les fingerprints d'un ensemble de tracks. Retourne le nombre supprimé."""
    if not db_path.exists() or not track_ids:
        return 0
    with sqlite3.connect(db_path) as conn:
        placeholders = ",".join("?" * len(track_ids))
        cur = conn.execute(
            f"DELETE FROM fingerprints WHERE track_id IN ({placeholders})",
            list(track_ids),
        )
    return cur.rowcount


def _fp_migrate_from_pkl(pkl_path: Path, db_path: Path) -> int:
    """
    Migration one-shot : importe fingerprints.pkl dans fingerprints.db.
    Appelée automatiquement au démarrage si le .pkl existe et le .db non.
    Retourne le nombre de fingerprints migrés.
    """
    try:
        with open(pkl_path, "rb") as f:
            old = pickle.load(f)
    except Exception:
        return 0
    _fp_init(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO fingerprints VALUES (?, ?, ?)",
            [(tid, pickle.dumps(fp), len(fp)) for tid, fp in old.items()],
        )
    return len(old)


# ===========================================================================
# Écriture atomique — metadata.parquet
# ===========================================================================

def _atomic_write_pickle(path: Path, obj: object) -> None:
    """Écrit un fichier pickle de manière atomique (temp file + rename).
    Garantit qu'en cas de crash pendant l'écriture, l'ancien fichier reste intact.
    """
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            pickle.dump(obj, f)
        os.replace(tmp_path, path)  # atomique sur tous les OS
    except Exception:
        os.unlink(tmp_path)
        raise


def _atomic_write_parquet(path: Path, df) -> None:
    """Écrit un fichier parquet de manière atomique (temp file + rename)."""
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(tmp_fd)
    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise

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

def _kill_proc(proc: subprocess.Popen) -> None:
    """Tue un subprocess de façon propre sur Unix (groupe de processus) et Windows."""
    if sys.platform == "win32":
        proc.kill()
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            proc.kill()


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
                # Sur Unix : start_new_session met yt-dlp + ffmpeg dans le même groupe
                # → on peut tout tuer d'un coup en cas de timeout.
                # Sur Windows : start_new_session n'existe pas, on utilise CREATE_NEW_PROCESS_GROUP.
                _is_windows = sys.platform == "win32"
                popen_kwargs: dict = {"cwd": tmpdir, "stderr": subprocess.PIPE}
                if _is_windows:
                    popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
                else:
                    popen_kwargs["start_new_session"] = True
                proc = subprocess.Popen(cmd, **popen_kwargs)
                try:
                    _, stderr_bytes = proc.communicate(timeout=45)
                except subprocess.TimeoutExpired:
                    _kill_proc(proc)
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
                        _kill_proc(proc)
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
    new_fp_hashes: set | None,   # set de hashes, ou None si déjà calculé
    metadata_row: dict,
    collection,                  # ChromaDB collection
    fp_db: Path,                 # SQLite fingerprints.db
    meta_path: Path,
) -> None:
    """
    Sauvegarde un track immédiatement dans ChromaDB + SQLite + metadata.
    Si le track_id existe déjà (crash partiel précédent), ses anciens segments
    sont supprimés et réécrits proprement — jamais de doublon ni de décalage.
    """
    new_emb = np.vstack(track_embeddings).astype(np.float32)

    # Supprimer les anciens segments si crash partiel précédent
    existing = collection.get(where={"track_id": {"$eq": track_id}})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    # Ajouter les nouveaux segments avec des IDs stables (track_id + index local)
    collection.add(
        embeddings=new_emb.tolist(),
        ids=[f"{track_id}_{i}" for i in range(len(new_emb))],
        metadatas=[
            {"track_id": track_id, "start_s": float(s["start_s"])}
            for s in track_segments
        ],
    )

    # Fingerprint — SQLite (atomique nativement)
    if new_fp_hashes is not None:
        _fp_save(fp_db, track_id, new_fp_hashes)

    # Metadata — écriture atomique
    if meta_path.exists():
        df_meta = pd.read_parquet(meta_path)
        if track_id in set(df_meta["track_id"]):
            idx = df_meta.index[df_meta["track_id"] == track_id][0]
            current = df_meta.at[idx, "embedded_methods"]
            current = list(current) if hasattr(current, "__iter__") and not isinstance(current, str) else []
            if method not in current:
                df_meta.at[idx, "embedded_methods"] = current + [method]
        else:
            df_meta = pd.concat([df_meta, pd.DataFrame([metadata_row])], ignore_index=True)
    else:
        df_meta = pd.DataFrame([metadata_row])
    _atomic_write_parquet(meta_path, df_meta)


def process_in_ram(tracks: list[dict], csv_sources: list[str]) -> None:
    """
    Pour chaque morceau : télécharge en RAM, calcule embeddings + fingerprint,
    sauvegarde dans ChromaDB (embeddings + segments) / fingerprints.pkl / metadata.parquet.
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

    fp_db     = FEATURES_DIR / "fingerprints.db"
    meta_path = PROCESSED_DIR / "metadata.parquet"

    # Migration automatique fingerprints.pkl → fingerprints.db (one-shot)
    fp_pkl = FEATURES_DIR / "fingerprints.pkl"
    if fp_pkl.exists() and not fp_db.exists():
        console.print("[yellow]Migration fingerprints.pkl → fingerprints.db…[/yellow]")
        n = _fp_migrate_from_pkl(fp_pkl, fp_db)
        console.print(f"[green]✓ {n} fingerprints migrés.[/green]")

    # Initialiser la DB (crée la table si première utilisation)
    _fp_init(fp_db)

    # Charger les track_ids déjà fingerprinted (set léger, pas les données)
    existing_fp_ids: set[str] = _fp_load_ids(fp_db)

    # Initialiser ChromaDB — collection par méthode d'embedding
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    collection = chroma_client.get_or_create_collection(
        name=method,
        metadata={"hnsw:space": "cosine"},
    )

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

                    # Fingerprint — seulement si pas déjà dans la DB
                    new_fp_hashes: set | None = None
                    if track_id not in existing_fp_ids:
                        waveform_fp = (
                            librosa.resample(waveform, orig_sr=sr, target_sr=config.SAMPLE_RATE)
                            if sr != config.SAMPLE_RATE else waveform
                        )
                        new_fp_hashes = extract_fingerprint(waveform_fp, config.SAMPLE_RATE)
                        existing_fp_ids.add(track_id)

                    # Segmentation + embedding
                    segs = list(iter_segments(waveform, sr, win_s, hop_s, min_win))

                    track_task = (
                        progress.add_task(f"[green]{label[:50]}", total=len(segs))
                        if config.PROGRESS_TRACK else None
                    )

                    track_embeddings: list = []
                    track_segments:   list = []

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
                                    "track_id": track_id,
                                    "start_s":  float(start_s),
                                })
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
                                "track_id": track_id,
                                "start_s":  float(start_s),
                            })
                            if track_task is not None:
                                progress.advance(track_task)

                    if track_task is not None:
                        progress.remove_task(track_task)

                    # Sauvegarde immédiate dans ChromaDB + SQLite + metadata
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
    total_seg  = collection.count()
    sample     = collection.get(limit=1, include=["embeddings"])
    emb_dim    = len(sample["embeddings"][0]) if sample.get("embeddings") else "?"

    console.print(Panel(
        f"[bold]Nouveaux tracks  :[/bold] {saved_count}\n"
        f"[bold]Total en base    :[/bold] {total_meta}\n"
        f"[bold]Total segments   :[/bold] {total_seg}\n"
        f"[bold]Embedding dim    :[/bold] {emb_dim}\n"
        f"[bold]Fingerprints     :[/bold] {len(existing_fp_ids)} tracks",
        title="[bold green]Embeddings + Fingerprints — OK[/bold green]",
        expand=False
    ))



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

    csv_sources = [f.name for f in csv_files]

    if not all_tracks:
        console.print("[green]Tous les morceaux sont déjà dans la base.[/green]")
    else:
        console.print(f"\n[bold]{len(all_tracks)} nouveaux morceaux à traiter.[/bold]\n")
        process_in_ram(all_tracks, csv_sources)

    # Construction de l'index FAISS — toujours à la fin, même si rien de nouveau
    run_step("Construction de l'index FAISS", [sys.executable, "src/index/build_index.py"])


if __name__ == "__main__":
    main()
