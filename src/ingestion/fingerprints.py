"""
src/ingestion/fingerprints.py

Reconstruction des fingerprints audio depuis YouTube.

Détecte les fingerprints v1 (hashes 3-tuples, sans alignement temporel) et les
reconstruit en v2 (hashes 4-tuples avec ancre temporelle t1), qui bénéficient de
l'alignement par histogramme d'offsets dans fingerprint_similarity().

Point d'entrée : run_rebuild_fingerprints(...)
"""

from __future__ import annotations

# OMP/OPENBLAS AVANT tout import numpy/librosa pour éviter les deadlocks macOS
import os
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress,
    SpinnerColumn, TextColumn, TimeElapsedColumn,
)

from src import config
from src.features.fingerprint import extract_fingerprint
from src.utils.youtube import download_audio_search, load_audio_safe
from src.utils.fingerprints_db import (
    fp_load_ids,
    fp_detect_format,
    fp_save,
)

console = Console()

ROOT  = Path(__file__).resolve().parents[2]
FP_DB = ROOT / config.FINGERPRINTS_DB


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _process_track(track: dict) -> dict:
    """Télécharge + recalcule le fingerprint d'un seul track. Appelé en thread."""
    tid    = track["track_id"]
    title  = track["title"]
    artist = track["artist"]

    tmpdir, mp3_path, _, _ = download_audio_search(artist, title)
    if mp3_path is None:
        return {"track_id": tid, "status": "failed",
                "message": f"YouTube introuvable : {artist} — {title}"}
    try:
        waveform = load_audio_safe(mp3_path, config.SAMPLE_RATE)
        if waveform is None:
            return {"track_id": tid, "status": "failed",
                    "message": f"Erreur ffmpeg/soundfile : {artist} — {title}"}
        new_fp = extract_fingerprint(waveform, config.SAMPLE_RATE)
        fp_save(FP_DB, tid, new_fp, thread_safe=True)
        return {"track_id": tid, "status": "ok",
                "message": f"{artist} — {title}  ({len(new_fp)} hashes v2)"}
    except Exception as exc:
        return {"track_id": tid, "status": "failed",
                "message": f"Erreur {artist} — {title} : {exc}"}
    finally:
        import shutil
        shutil.rmtree(tmpdir or "", ignore_errors=True)


# ---------------------------------------------------------------------------
# Point d'entrée public
# ---------------------------------------------------------------------------

def run_rebuild_fingerprints(
    force_all: bool = False,
    limit: int | None = None,
    workers: int = config.DOWNLOAD_WORKERS,
    dry_run: bool = False,
) -> None:
    """
    Reconstruit les fingerprints v1 (sans ancre temporelle) en v2.

    Args:
        force_all: si True, rebuild tous les fingerprints même déjà v2.
        limit:     nombre maximum de tracks à traiter (None = tous).
        workers:   nombre de threads parallèles.
        dry_run:   affiche ce qui serait fait sans modifier la base.
    """
    import pandas as pd

    meta_path = ROOT / config.METADATA_PATH
    if not meta_path.exists():
        console.print("[red]metadata.parquet introuvable. Lance d'abord l'ingestion.[/red]")
        sys.exit(1)

    # ── Détection rapide du format (1 seul blob lu) ──────────────────────────
    console.print("🔍  Détection du format des fingerprints…")
    current_fmt = fp_detect_format(FP_DB)
    console.print(f"    Format détecté : [bold]{current_fmt}[/bold]")

    if current_fmt == "v2" and not force_all:
        console.print("[green]✅  Tous les fingerprints sont déjà en v2. Rien à faire.[/green]")
        console.print("    Utilise [bold]--all[/bold] pour forcer le rebuild complet.")
        return

    # ── Chargement léger : track_ids présents en base ───────────────────────
    console.print("📂  Chargement des track_ids depuis SQLite…")
    fp_ids = fp_load_ids(FP_DB)
    console.print(f"    {len(fp_ids)} fingerprints en base")

    meta_df = pd.read_parquet(meta_path)
    if "track_id" not in meta_df.columns:
        console.print("[red]Colonne track_id absente de metadata.parquet.[/red]")
        sys.exit(1)

    # ── Sélection des tracks à traiter ───────────────────────────────────────
    to_rebuild: list[dict] = []
    for _, row in meta_df.iterrows():
        tid = str(row["track_id"])
        if tid not in fp_ids:
            continue  # pas de fingerprint → laissé à l'ingestion
        to_rebuild.append({
            "track_id": tid,
            "title":    str(row.get("title",  tid)),
            "artist":   str(row.get("artist", "Unknown")),
        })

    if limit:
        to_rebuild = to_rebuild[:limit]

    eta_min = max(1, len(to_rebuild) * 15 // max(workers, 1) // 60)

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
