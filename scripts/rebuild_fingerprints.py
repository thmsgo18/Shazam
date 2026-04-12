#!/usr/bin/env python3
"""
scripts/rebuild_fingerprints.py

Reconstruit les fingerprints de la base après un changement de format.

Les fingerprints stockés en v1 (hashes 3-tuples sans ancre temporelle) ne profitent
pas de l'alignement temporel. Ce script les détecte et les recompute en v2
(hashes 4-tuples avec ancre temporelle).

Fonctionnement :
    - Lit les track_ids depuis SQLite (sans charger les blobs — rapide).
    - Échantillonne 1 fingerprint pour détecter le format courant (v1/v2).
    - Re-télécharge l'audio via yt-dlp, calcule le fingerprint v2 et met à jour SQLite.
    - Conversion MP3 → WAV via ffmpeg + soundfile (sans deadlock sur macOS).
    - Les workers téléchargent en parallèle (--workers, défaut : 3).

Usage :
    python scripts/rebuild_fingerprints.py                # rebuild les fingerprints v1
    python scripts/rebuild_fingerprints.py --all          # rebuild tous (même v2)
    python scripts/rebuild_fingerprints.py --limit 20     # test sur N tracks
    python scripts/rebuild_fingerprints.py --dry-run      # aperçu sans modification
    python scripts/rebuild_fingerprints.py --workers 5    # plus de workers
"""
from __future__ import annotations

# OMP/OPENBLAS AVANT tout import numpy/librosa pour éviter les deadlocks macOS
import os
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pickle
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click
import numpy as np
import pandas as pd
import soundfile as sf
from rich.console import Console
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress,
    SpinnerColumn, TextColumn, TimeElapsedColumn,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.features.fingerprint import extract_fingerprint

ROOT     = Path(__file__).resolve().parents[1]
FP_DB    = ROOT / config.FINGERPRINTS_DB
METADATA = ROOT / config.METADATA_PATH
console  = Console()

_db_lock = threading.Lock()   # SQLite n'est pas thread-safe en écriture


# ── SQLite — lecture légère (sans désérialiser les blobs) ───────────────────

def _fp_track_ids(db_path: Path) -> set[str]:
    """Retourne les track_ids présents dans la base (lecture rapide, pas de blobs)."""
    if not db_path.exists():
        return set()
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT track_id FROM fingerprints WHERE n_hashes > 0"
        ).fetchall()
    return {r[0] for r in rows}


def _fp_detect_format(db_path: Path) -> str:
    """
    Détecte le format des fingerprints en désérialisant UN SEUL blob.
    Retourne 'v1' (3-tuples), 'v2' (4-tuples) ou 'unknown'.
    Évite de charger plusieurs GB de données en mémoire.
    """
    if not db_path.exists():
        return "unknown"
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT hashes FROM fingerprints WHERE n_hashes > 0 LIMIT 1"
        ).fetchone()
    if row is None:
        return "unknown"
    try:
        fp = pickle.loads(row[0])
        sample = next(iter(fp))
        return "v2" if len(sample) == 4 else "v1"
    except Exception:
        return "unknown"


def _fp_save(db_path: Path, track_id: str, hashes: set) -> None:
    with _db_lock:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO fingerprints VALUES (?, ?, ?)",
                (track_id, pickle.dumps(hashes), len(hashes)),
            )


# ── Téléchargement audio ────────────────────────────────────────────────────

def _kill_proc(proc: subprocess.Popen) -> None:
    if sys.platform == "win32":
        proc.kill()
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            proc.kill()


def _download_audio(artist: str, title: str, retries: int = 2):
    """Télécharge l'audio via yt-dlp → (tmpdir, mp3_path) ou (None, None)."""
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
        "--socket-timeout", "20",
        "--extractor-args", "youtube:player_client=android",
    ]
    for query in queries:
        cmd = [base_cmd[0], f"ytsearch1:{query}"] + base_cmd[1:]
        for attempt in range(retries):
            tmpdir = tempfile.mkdtemp()
            proc = None
            try:
                popen_kw: dict = {"cwd": tmpdir, "stderr": subprocess.PIPE}
                if sys.platform == "win32":
                    popen_kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
                else:
                    popen_kw["start_new_session"] = True
                proc = subprocess.Popen(cmd, **popen_kw)
                try:
                    proc.communicate(timeout=40)
                except subprocess.TimeoutExpired:
                    _kill_proc(proc)
                    proc.wait()
                    shutil.rmtree(tmpdir, ignore_errors=True)
                    if attempt < retries - 1:
                        time.sleep(2)
                    continue
                if proc.returncode != 0:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                    break
                files = list(Path(tmpdir).glob("*.mp3"))
                if not files:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                    break
                return tmpdir, str(files[0])
            except Exception:
                if proc:
                    try:
                        _kill_proc(proc)
                    except Exception:
                        pass
                shutil.rmtree(tmpdir, ignore_errors=True)
                break
    return None, None


def _load_audio_safe(mp3_path: str, sr: int) -> np.ndarray | None:
    """
    Charge un MP3 sans deadlock :
      ffmpeg convertit MP3 → WAV mono (subprocess propre, pas de threads internes)
      soundfile lit le WAV (thread-safe, pas d'OpenBLAS)
    """
    wav_path = mp3_path + "_tmp.wav"
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path,
             "-ar", str(sr), "-ac", "1", "-sample_fmt", "s16", wav_path],
            capture_output=True, timeout=60,
        )
        if result.returncode != 0:
            return None
        data, _ = sf.read(wav_path, dtype="float32")
        return data
    except Exception:
        return None
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


# ── Worker ──────────────────────────────────────────────────────────────────

def _process_track(track: dict) -> dict:
    """Exécuté dans un thread : download + fingerprint + save."""
    tid    = track["track_id"]
    title  = track["title"]
    artist = track["artist"]

    tmpdir, mp3_path = _download_audio(artist, title)
    if mp3_path is None:
        return {"track_id": tid, "status": "failed",
                "message": f"YouTube introuvable : {artist} — {title}"}
    try:
        waveform = _load_audio_safe(mp3_path, config.SAMPLE_RATE)
        if waveform is None:
            return {"track_id": tid, "status": "failed",
                    "message": f"Erreur ffmpeg/soundfile : {artist} — {title}"}
        new_fp = extract_fingerprint(waveform, config.SAMPLE_RATE)
        _fp_save(FP_DB, tid, new_fp)
        return {"track_id": tid, "status": "ok",
                "message": f"{artist} — {title}  ({len(new_fp)} hashes v2)"}
    except Exception as exc:
        return {"track_id": tid, "status": "failed",
                "message": f"Erreur {artist} — {title} : {exc}"}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── CLI ─────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--all",     "force_all", is_flag=True, default=False,
              help="Rebuild tous les fingerprints, même ceux déjà en v2.")
@click.option("--limit",   type=int, default=None,
              help="Limite le nombre de tracks à traiter (test).")
@click.option("--workers", type=int, default=config.DOWNLOAD_WORKERS,
              help=f"Nombre de workers parallèles (défaut : {config.DOWNLOAD_WORKERS}).")
@click.option("--dry-run", is_flag=True, default=False,
              help="Affiche ce qui serait fait sans modifier la base.")
def main(force_all: bool, limit: int | None, workers: int, dry_run: bool) -> None:
    """Reconstruit les fingerprints v1 (sans ancre temporelle) en v2."""

    if not METADATA.exists():
        console.print("[red]metadata.parquet introuvable. Lance d'abord download_music.py.[/red]")
        sys.exit(1)

    # ── Détection rapide du format (1 seul blob lu) ──────────────────────────
    console.print("🔍  Détection du format des fingerprints…")
    current_fmt = _fp_detect_format(FP_DB)
    console.print(f"    Format détecté : [bold]{current_fmt}[/bold]")

    if current_fmt == "v2" and not force_all:
        console.print("[green]✅  Tous les fingerprints sont déjà en v2. Rien à faire.[/green]")
        console.print("    Utilise [bold]--all[/bold] pour forcer le rebuild complet.")
        sys.exit(0)

    # ── Chargement léger : track_ids + métadonnées ───────────────────────────
    console.print("📂  Chargement des track_ids depuis SQLite…")
    fp_ids = _fp_track_ids(FP_DB)
    console.print(f"    {len(fp_ids)} fingerprints en base")

    meta_df = pd.read_parquet(METADATA)
    if "track_id" not in meta_df.columns:
        console.print("[red]Colonne track_id absente de metadata.parquet.[/red]")
        sys.exit(1)

    # ── Sélection des tracks à traiter ───────────────────────────────────────
    to_rebuild: list[dict] = []
    for _, row in meta_df.iterrows():
        tid = str(row["track_id"])
        if tid not in fp_ids:
            continue  # pas de fingerprint → laissé à download_music.py
        to_rebuild.append({
            "track_id": tid,
            "title":    str(row.get("title",  tid)),
            "artist":   str(row.get("artist", "Unknown")),
        })

    if limit:
        to_rebuild = to_rebuild[:limit]

    eta_min = max(1, len(to_rebuild) * 15 // max(workers, 1) // 60)

    # ── Résumé ────────────────────────────────────────────────────────────────
    console.print(f"\n[bold]Tracks à rebuilder :[/bold] {len(to_rebuild)}")
    console.print(f"  Workers parallèles  : {workers}")
    console.print(f"  Durée estimée       : ~{eta_min} min")

    if dry_run:
        console.print("\n[yellow]-- Mode dry-run : aucune modification --[/yellow]")
        for t in to_rebuild[:20]:
            console.print(f"  · {t['artist']} — {t['title']}")
        if len(to_rebuild) > 20:
            console.print(f"  … et {len(to_rebuild) - 20} autres")
        return

    if not to_rebuild:
        console.print("[green]✅  Rien à rebuilder.[/green]")
        return

    console.print()

    # ── Rebuild en parallèle ─────────────────────────────────────────────────
    ok = failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Rebuild…", total=len(to_rebuild))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_track, t): t for t in to_rebuild}
            try:
                for future in as_completed(futures):
                    result = future.result()
                    if result["status"] == "ok":
                        ok += 1
                    else:
                        failed += 1
                        console.print(f"  [yellow]⚠[/yellow]  {result['message']}")
                    progress.advance(task)
            except KeyboardInterrupt:
                console.print("\n[yellow]Interruption — annulation des workers…[/yellow]")
                executor.shutdown(wait=False, cancel_futures=True)

    # ── Bilan ─────────────────────────────────────────────────────────────────
    console.print(f"\n[bold green]✅  {ok} fingerprints reconstruits en v2[/bold green]")
    if failed:
        console.print(f"[yellow]⚠   {failed} tracks ignorés (YouTube introuvable ou erreur)[/yellow]")
    console.print("ℹ   FAISS n'est pas affecté — pas besoin de reconstruire l'index.")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
