"""
src/ingestion/fingerprints.py


Audio fingerprint reconstruction from YouTube.


Detects v1 fingerprints (3-tuple hashes, without temporal alignment) and
rebuilds them as v2 (4-tuple hashes with temporal anchor t1), which benefit from
offset-histogram alignment in fingerprint_similarity().


Entry point: run_rebuild_fingerprints(...)
"""


from __future__ import annotations


# OMP/OPENBLAS BEFORE any numpy/librosa import to avoid macOS deadlocks
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
    """Downloads and recomputes the fingerprint of a single track. Called in a thread."""
    tid    = track["track_id"]
    title  = track["title"]
    artist = track["artist"]


    tmpdir, mp3_path, _, _ = download_audio_search(artist, title)
    if mp3_path is None:
        return {"track_id": tid, "status": "failed",
                "message": f"YouTube not found: {artist} — {title}"}
    try:
        waveform = load_audio_safe(mp3_path, config.SAMPLE_RATE)
        if waveform is None:
            return {"track_id": tid, "status": "failed",
                    "message": f"ffmpeg/soundfile error: {artist} — {title}"}
        new_fp = extract_fingerprint(waveform, config.SAMPLE_RATE)
        fp_save(FP_DB, tid, new_fp, thread_safe=True)
        return {"track_id": tid, "status": "ok",
                "message": f"{artist} — {title}  ({len(new_fp)} v2 hashes)"}
    except Exception as exc:
        return {"track_id": tid, "status": "failed",
                "message": f"Error {artist} — {title}: {exc}"}
    finally:
        import shutil
        shutil.rmtree(tmpdir or "", ignore_errors=True)



# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_rebuild_fingerprints(
    force_all: bool = False,
    limit: int | None = None,
    workers: int = config.DOWNLOAD_WORKERS,
    dry_run: bool = False,
) -> None:
    """
    Rebuilds v1 fingerprints (without temporal anchor) to v2.


    Args:
        force_all: if True, rebuilds all fingerprints even if already v2.
        limit:     maximum number of tracks to process (None = all).
        workers:   number of parallel threads.
        dry_run:   prints what would be done without modifying the database.
    """
    import pandas as pd


    meta_path = ROOT / config.METADATA_PATH
    if not meta_path.exists():
        console.print("[red]metadata.parquet not found. Run ingestion first.[/red]")
        sys.exit(1)


    # ── Quick format detection (reads only 1 blob) ───────────────────────────
    console.print("🔍  Detecting fingerprint format…")
    current_fmt = fp_detect_format(FP_DB)
    console.print(f"    Detected format: [bold]{current_fmt}[/bold]")


    if current_fmt == "v2" and not force_all:
        console.print("[green]✅  All fingerprints are already v2. Nothing to do.[/green]")
        console.print("    Use [bold]--all[/bold] to force a full rebuild.")
        return


    # ── Lightweight load: track_ids present in the database ─────────────────
    console.print("📂  Loading track_ids from SQLite…")
    fp_ids = fp_load_ids(FP_DB)
    console.print(f"    {len(fp_ids)} fingerprints in database")


    meta_df = pd.read_parquet(meta_path)
    if "track_id" not in meta_df.columns:
        console.print("[red]Column track_id missing from metadata.parquet.[/red]")
        sys.exit(1)


    # ── Select tracks to process ─────────────────────────────────────────────
    to_rebuild: list[dict] = []
    for _, row in meta_df.iterrows():
        tid = str(row["track_id"])
        if tid not in fp_ids:
            continue  # no fingerprint → left to ingestion
        to_rebuild.append({
            "track_id": tid,
            "title":    str(row.get("title",  tid)),
            "artist":   str(row.get("artist", "Unknown")),
        })


    if limit:
        to_rebuild = to_rebuild[:limit]


    eta_min = max(1, len(to_rebuild) * 15 // max(workers, 1) // 60)


    console.print(f"\n[bold]Tracks to rebuild:[/bold] {len(to_rebuild)}")
    console.print(f"  Parallel workers    : {workers}")
    console.print(f"  Estimated duration  : ~{eta_min} min")


    if dry_run:
        console.print("\n[yellow]-- Dry-run mode: no changes will be made --[/yellow]")
        for t in to_rebuild[:20]:
            console.print(f"  · {t['artist']} — {t['title']}")
        if len(to_rebuild) > 20:
            console.print(f"  … and {len(to_rebuild) - 20} more")
        return


    if not to_rebuild:
        console.print("[green]✅  Nothing to rebuild.[/green]")
        return


    console.print()


    # ── Parallel rebuild ──────────────────────────────────────────────────────
    ok = failed = 0


    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Rebuilding…", total=len(to_rebuild))


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
                console.print("\n[yellow]Interrupted — cancelling workers…[/yellow]")
                executor.shutdown(wait=False, cancel_futures=True)


    # ── Summary ───────────────────────────────────────────────────────────────
    console.print(f"\n[bold green]✅  {ok} fingerprints rebuilt to v2[/bold green]")
    if failed:
        console.print(f"[yellow]⚠   {failed} tracks skipped (YouTube not found or error)[/yellow]")
    console.print("ℹ   FAISS is unaffected — no need to rebuild the index.")