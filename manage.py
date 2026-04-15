#!/usr/bin/env python3
"""
manage.py — single entry point for the Shazam Maison project.

Usage :
    python manage.py <command> [options]

Available commands:
    ── Build ──────────────────────────────────────────────────────────────────
    build                  Full pipeline: ingest → augment → enrich
    ingest                 Download and ingest tracks from Kaggle CSV files
    augment                Augment embeddings with RIRs
    enrich                 Enrich metadata.parquet via Deezer + MusicBrainz

    ── Maintenance ────────────────────────────────────────────────────────────
    check                  Verify data consistency
    rebuild                Recompute fingerprints and/or rebuild the FAISS index
    clean                  Delete data (track, RIR, or everything)

    ── Usage ──────────────────────────────────────────────────────────────────
    config                 Show the active configuration (src/config.py)
    identify               Identify an audio file
    download-test          Download a test audio clip from YouTube

    ── Evaluation ─────────────────────────────────────────────────────────────
    eval                   Run the report-oriented base evaluation suite
    eval base              Explicit alias for the base evaluation suite
    eval studio-mic        Compare studio and microphone queries
    eval duration          Analyze the effect of clip duration
    eval stage12           Compare Stage 1 and final ranking
    eval rir               Compare with and without RIR (separate analysis)
    eval mic-conditions    Analyze microphone distance + concurrent speech

    ── Web Interface ──────────────────────────────────────────────────────────
    webapp                 Start the FastAPI backend + React/Vite frontend
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*upsample_bicubic2d.*", category=UserWarning)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import click


# ═══════════════════════════════════════════════════════════════════════════════
# Groupe principal
# ═══════════════════════════════════════════════════════════════════════════════

@click.group()
def cli():
    """Shazam Maison — music recognition system (Master IAD S2)."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

@cli.command()
@click.option("--csv", "csv_paths", multiple=True,
              help="Path(s) to Kaggle CSV files (repeatable). Default: data/kaggle/")
@click.option("--skip-rir", is_flag=True, default=False,
              help="Skip the RIR augmentation step after ingestion")
@click.option("--skip-enrich", is_flag=True, default=False,
              help="Skip metadata enrichment")
def build(csv_paths: tuple[str, ...], skip_rir: bool, skip_enrich: bool) -> None:
    """Full pipeline: ingest -> augment -> enrich (recommended entry point).

    \b
    All parameters (method, RIR, workers...) are read from src/config.py.

    \b
    Examples:
      python manage.py build --csv data/kaggle/data/spotify-streaming-top-50-world.csv
      python manage.py build --csv data/kaggle/data/spotify-streaming-top-50-world.csv --skip-rir
      python manage.py build  # all CSVs in data/kaggle/
    """
    from src.ingestion.ingest import run_ingest

    click.echo("\n[build] ── Step 1/3: Ingestion ───────────────────────────────")
    run_ingest(list(csv_paths) if csv_paths else None)

    if not skip_rir:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK",      "1")
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
        from src.ingestion.augment_rir import run_augment
        click.echo("\n[build] ── Step 2/3: RIR augmentation ───────────────────────")
        run_augment()
    else:
        click.echo("\n[build] ── Step 2/3: RIR augmentation (skipped) ─────────────")

    if not skip_enrich:
        from src.maintenance.enrich import run_enrich
        click.echo("\n[build] ── Step 3/3: Metadata enrichment ────────────────────")
        run_enrich()
    else:
        click.echo("\n[build] ── Step 3/3: Enrichment (skipped) ───────────────────")

    click.echo("\n[build] Done.")


@cli.command()
@click.option("--csv", "csv_paths", multiple=True,
              help="Path(s) to Kaggle CSV files (repeatable). Default: data/kaggle/")
def ingest(csv_paths: tuple[str, ...]) -> None:
    """Download and ingest tracks from Kaggle CSV files.

    \b
    Automatic resume: already-processed tracks are skipped.
    The embedding method is read from src/config.py (EMBEDDING_METHOD).
    """
    from src.ingestion.ingest import run_ingest
    run_ingest(list(csv_paths) if csv_paths else None)


@cli.command()
@click.option("--tracks", default="all", show_default=True,
              help="Tracks to augment: all, flowers, or a specific track_id")
def augment(tracks: str) -> None:
    """Augment embeddings with RIRs (Room Impulse Responses).

    \b
    All parameters are read from src/config.py:
      RIR_SOURCE  - synthetic (generated) or mit (real WAV files)
      RIR_N       - number of RIRs per track
      RIR_MIT_DIR - MIT WAV directory (if RIR_SOURCE=mit)

    \b
    Examples:
      python manage.py augment
      python manage.py augment --tracks flowers
      python manage.py augment --tracks f01ab00f1fdc5a57fd2676f4d68631a8
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK",      "1")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    from src.ingestion.augment_rir import run_augment
    run_augment(tracks=tracks)


@cli.command()
@click.option("--force", is_flag=True, default=False,
              help="Update all tracks, including already-enriched ones")
@click.option("--only-missing", is_flag=True, default=False,
              help="Process only tracks with at least one missing field")
def enrich(force: bool, only_missing: bool) -> None:
    """Enrich metadata.parquet via Deezer + MusicBrainz.

    \b
    Enriched fields: album, genre, release_date, cover_url.
    With no option: enrich only tracks that have not been processed yet.
    """
    from src.maintenance.enrich import run_enrich
    run_enrich(force=force, only_missing=only_missing)


# ═══════════════════════════════════════════════════════════════════════════════
# MAINTENANCE
# ═══════════════════════════════════════════════════════════════════════════════

@cli.command()
@click.option("--details", is_flag=True, default=False,
              help="Show detailed warnings by category (codes C1-C7, Q1-Q4, FP)")
@click.option("--metadata", is_flag=True, default=False,
              help="List tracks with missing or partial metadata")
@click.option("--purge", is_flag=True, default=False,
              help="Delete problematic tracks from every store")
@click.option("--purge-missing-fp", is_flag=True, default=False,
              help="Purge only tracks without fingerprints")
@click.option("--yes", "-y", is_flag=True, default=False,
              help="Skip confirmation before purging")
def check(details: bool, metadata: bool, purge: bool, purge_missing_fp: bool, yes: bool) -> None:
    """Check data consistency (ChromaDB / FAISS / fingerprints / metadata).

    \b
    Check codes (--details):
      C1  Unexpected embedding dimension
      C2  NaN or Inf values in embeddings
      C3  ChromaDB <-> metadata desynchronization
      C5  FAISS index out of sync
      C6  Orphan segments
      C7  Incomplete embedding (< 80% of expected segments)
      Q3  Empty fingerprint
      FP  Track without fingerprint
    """
    from src.maintenance.check import run_check
    run_check(
        method=None,
        details=details,
        metadata=metadata,
        purge=purge,
        purge_missing_fp=purge_missing_fp,
        yes=yes,
    )


@cli.command()
@click.option("--what", type=click.Choice(["index", "fps", "all"]), default="all",
              show_default=True,
              help="index = FAISS only · fps = fingerprints only · all = both")
@click.option("--force", is_flag=True, default=False,
              help="Recompute even tracks already present in SQLite (fingerprints only)")
def rebuild(what: str, force: bool) -> None:
    """Recompute fingerprints and/or rebuild the FAISS index.

    \b
    Examples:
      python manage.py rebuild               # rebuild everything
      python manage.py rebuild --what index  # FAISS only (after check --purge)
      python manage.py rebuild --what fps --force  # fingerprints, full recompute
    """
    if what in ("fps", "all"):
        from src.ingestion.fingerprints import run_rebuild_fingerprints
        click.echo("[rebuild] Recomputing fingerprints...")
        run_rebuild_fingerprints(force_all=force, limit=None, workers=4, dry_run=False)
        click.echo("[rebuild] Fingerprints done.")

    if what in ("index", "all"):
        import chromadb
        from src import config
        from src.index.build_index import _build_for_method
        click.echo("[rebuild] Rebuilding the FAISS index...")
        index_type    = getattr(config, "INDEX_TYPE", "flat")
        chroma_client = chromadb.PersistentClient(path=str(ROOT / config.CHROMA_DIR))
        keys = [c.name for c in chroma_client.list_collections()]
        if not keys:
            click.echo("[rebuild] No ChromaDB collection found.")
            click.echo("[rebuild] Run `python manage.py ingest` first.")
            sys.exit(1)
        for key in keys:
            _build_for_method(key, index_type, chroma_client)
        click.echo("[rebuild] FAISS index done.")


@cli.command()
@click.option("--track", "track_id", default=None, metavar="TRACK_ID",
              help="Delete a specific track from every store")
@click.option("--rir", is_flag=True, default=False,
              help="Delete RIR segments for the active method (EMBEDDING_METHOD in config.py)")
@click.option("--all", "all_data", is_flag=True, default=False,
              help="Full reset - delete everything (ChromaDB, FAISS, SQLite, metadata)")
@click.option("--yes", "-y", is_flag=True, default=False,
              help="Skip confirmation")
def clean(track_id: str | None, rir: bool, all_data: bool, yes: bool) -> None:
    """Delete data: one track, RIR segments, or the whole database.

    \b
    Examples:
      python manage.py clean --track f01ab00f1fdc5a57fd2676f4d68631a8
      python manage.py clean --rir
      python manage.py clean --all --yes
    """
    if not track_id and not rir and not all_data:
        click.echo("Specify what you want to delete:")
        click.echo("  --track TRACK_ID   Delete one specific track")
        click.echo("  --rir              Delete RIR segments for the active method")
        click.echo("  --all              Delete everything (full reset)")
        sys.exit(0)

    if track_id:
        from src.maintenance.clean import run_clean_track
        run_clean_track(track_id=track_id, yes=yes)

    if rir:
        from src.maintenance.delete_rir import run_delete_rir
        from src import config
        run_delete_rir(method=config.EMBEDDING_METHOD, dry_run=False, yes=yes)

    if all_data:
        from src.maintenance.clean import run_clean
        run_clean(yes=yes)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILISATION
# ═══════════════════════════════════════════════════════════════════════════════

@cli.command("config")
def show_config() -> None:
    """Show the active configuration (src/config.py)."""
    from src import config

    click.echo()
    click.echo("── Embedding ───────────────────────────────────────────────────")
    click.echo(f"  EMBEDDING_METHOD         : {config.EMBEDDING_METHOD}")
    if config.EMBEDDING_METHOD == "clap":
        click.echo(f"  CLAP_MODEL_NAME          : {config.CLAP_MODEL_NAME}")
        click.echo(f"  CLAP_SAMPLE_RATE         : {config.CLAP_SAMPLE_RATE} Hz")
    elif config.EMBEDDING_METHOD == "muq":
        click.echo(f"  MUQ_MODEL_NAME           : {config.MUQ_MODEL_NAME}")
        click.echo(f"  MUQ_SAMPLE_RATE          : {config.MUQ_SAMPLE_RATE} Hz")
    elif config.EMBEDDING_METHOD == "mert":
        click.echo(f"  MERT_MODEL_NAME          : {config.MERT_MODEL_NAME}")
        click.echo(f"  MERT_SAMPLE_RATE         : {config.MERT_SAMPLE_RATE} Hz")
    else:
        click.echo(f"  SAMPLE_RATE              : {config.SAMPLE_RATE} Hz")

    click.echo()
    click.echo("── Index & Vector Search ──────────────────────────────────────")
    click.echo(f"  INDEX_TYPE               : {config.INDEX_TYPE}")
    click.echo(f"  VECTOR_TOP_K_SEGMENTS    : {config.VECTOR_TOP_K_SEGMENTS}  (FAISS neighbors per query segment)")
    click.echo(f"  VECTOR_TOP_N_TRACKS      : {config.VECTOR_TOP_N_TRACKS}   (Stage 1 candidates passed to Stage 2)")
    click.echo(f"  VECTOR_TOP_N_RESULTS     : {config.VECTOR_TOP_N_RESULTS}  (final results returned)")

    click.echo()
    click.echo("── Segmentation ────────────────────────────────────────────────")
    click.echo(f"  SEGMENT_WIN_S            : {config.SEGMENT_WIN_S} s")
    click.echo(f"  SEGMENT_HOP_S            : {config.SEGMENT_HOP_S} s")

    click.echo()
    click.echo("── Augmentation RIR ────────────────────────────────────────────")
    click.echo(f"  RIR_SOURCE               : {config.RIR_SOURCE}")
    click.echo(f"  RIR_N                    : {config.RIR_N}")
    if config.RIR_SOURCE == "mit":
        click.echo(f"  RIR_MIT_DIR              : {config.RIR_MIT_DIR}")

    click.echo()
    click.echo("── Web Interface ──────────────────────────────────────────────")
    click.echo(f"  UI_LISTEN_DURATION       : {config.UI_LISTEN_DURATION} s")

    click.echo()
    click.echo("── Optimizations ──────────────────────────────────────────────")
    click.echo(f"  OPT_FLOAT16              : {config.OPT_FLOAT16}")
    click.echo(f"  OPT_BATCH_EMBED          : {config.OPT_BATCH_EMBED}")
    click.echo(f"  OPT_QUERY_DENOISE        : {config.OPT_QUERY_DENOISE}")
    click.echo(f"  OPT_FINGERPRINT_PARALLEL : {config.OPT_FINGERPRINT_PARALLEL}")
    click.echo()


@cli.command()
@click.argument("audio", type=click.Path(exists=True))
@click.option("--top", default=5, show_default=True,
              help="Number of results to display")
@click.option("--detailed", is_flag=True, default=False,
              help="Show FAISS and fingerprint scores separately")
@click.option("--target", "target_track_id", default=None, metavar="TRACK_ID",
              help="Expected track_id - enable evaluation mode and show the target rank")
def identify(audio: str, top: int, detailed: bool, target_track_id: str | None) -> None:
    """Identify the track corresponding to an audio file.

    \b
    Without --target: return the top N results.
    With --target: evaluation mode - also show the rank of the expected track.

    \b
    Examples:
      python manage.py identify data/raw/my_audio.mp3
      python manage.py identify data/raw/my_audio.mp3 --top 10 --detailed
      python manage.py identify data/raw/my_audio.mp3 --target f01ab00f
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK",      "1")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    if target_track_id:
        from src.evaluation.find_track import run_find_track
        run_find_track(audio=audio, target_track_id=target_track_id, top=top, method=None)
    else:
        from src.api.app import run_identify_cli
        run_identify_cli(audio, method=None, top=top, detailed=detailed)


_POSITIONS = ["start", "first-quarter", "middle", "third-quarter", "end"]


@cli.command("download-test")
@click.argument("query", nargs=-1, required=True)
@click.option("--duration", type=click.Choice(["5", "10", "15", "30"]), default=None,
              help="Clip duration in seconds. Omit for the full track.")
@click.option("--position", type=click.Choice(_POSITIONS), default="start", show_default=True,
              help="Position in the track: start / first-quarter / middle / third-quarter / end")
def download_test(query: tuple[str, ...], duration: str | None, position: str) -> None:
    """Download a test audio clip from YouTube into data/raw/.

    \b
    The clip is automatically added to the test manifest
    (data/raw/manifest.json) used by all `eval` analyses.

    \b
    Examples:
      python manage.py download-test "Miley Cyrus Flowers" --duration 30 --position middle
      python manage.py download-test "Daft Punk Get Lucky" --duration 15 --position middle
      python manage.py download-test "The Weeknd Blinding Lights" --duration 5 --position middle
    """
    import json
    import subprocess
    import tempfile

    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    search_query = " ".join(query)
    duration_s   = int(duration) if duration else None

    click.echo(f"Search query: {search_query}")

    resolve = subprocess.run(
        ["yt-dlp", "--get-id", "--no-playlist", f"ytsearch1:{search_query}"],
        capture_output=True, text=True,
    )
    if resolve.returncode != 0 or not resolve.stdout.strip():
        click.echo("Error: no video found on YouTube.", err=True)
        sys.exit(1)

    video_id  = resolve.stdout.strip()
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    click.echo(f"Video URL: {video_url}")

    if duration_s is None:
        result = subprocess.run([
            "yt-dlp", video_url,
            "--extract-audio", "--audio-format", "mp3", "--audio-quality", "5",
            "--output", str(raw_dir / "%(title)s.%(ext)s"),
            "--socket-timeout", "30",
        ])
        if result.returncode != 0:
            click.echo(f"yt-dlp error (code {result.returncode})", err=True)
            sys.exit(result.returncode)
    else:
        click.echo("Fetching video metadata...")
        meta = subprocess.run(
            ["yt-dlp", "--dump-json", "--no-playlist", video_url],
            capture_output=True, text=True,
        )
        total_duration: float | None = None
        if meta.returncode == 0:
            try:
                total_duration = float(json.loads(meta.stdout).get("duration", 0)) or None
            except (json.JSONDecodeError, ValueError):
                pass

        if total_duration is None:
            click.echo("Could not fetch track duration — starting at 0s.", err=True)
            start_s = 0.0
        else:
            if position == "start":
                start_s = 0.0
            elif position == "first-quarter":
                start_s = total_duration * 0.25
            elif position == "middle":
                start_s = max(0.0, total_duration / 2 - duration_s / 2)
            elif position == "third-quarter":
                start_s = total_duration * 0.75
            else:
                start_s = max(0.0, total_duration - duration_s)
            max_start = max(0.0, total_duration - duration_s)
            start_s   = min(start_s, max_start)
            click.echo(
                f"Total duration: {total_duration:.0f}s  |  "
                f"Clip: {duration_s}s at {position} (starting at {start_s:.1f}s)"
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dl = subprocess.run([
                "yt-dlp", video_url,
                "--extract-audio", "--audio-format", "mp3", "--audio-quality", "5",
                "--output", str(tmp_path / "%(title)s.%(ext)s"),
                "--socket-timeout", "30",
            ])
            if dl.returncode != 0:
                click.echo(f"yt-dlp error (code {dl.returncode})", err=True)
                sys.exit(dl.returncode)

            src_files = list(tmp_path.glob("*.mp3"))
            if not src_files:
                click.echo("Error: MP3 file not found after download.", err=True)
                sys.exit(1)

            src      = src_files[0]
            out_path = raw_dir / f"{src.stem}__{position}_{duration_s}s.mp3"
            cut      = subprocess.run([
                "ffmpeg", "-y", "-ss", str(start_s), "-i", str(src),
                "-t", str(duration_s), "-acodec", "copy", str(out_path),
            ], capture_output=True)
            if cut.returncode != 0:
                click.echo("ffmpeg error while trimming the clip.", err=True)
                click.echo(cut.stderr.decode(errors="ignore"), err=True)
                sys.exit(cut.returncode)

    files = sorted(raw_dir.glob("*.mp3"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        return
    downloaded_file = files[0]
    click.echo(f"\n  Downloaded file: {downloaded_file.name}")

    # Mise à jour du manifest (ground truth pour eval)
    try:
        import pandas as pd
        from src.evaluation.evaluate import find_track_id_by_query

        track_id = find_track_id_by_query(search_query)
        if track_id:
            meta_path = ROOT / "data" / "processed" / "metadata.parquet"
            artist, title = "", ""
            if meta_path.exists():
                df  = pd.read_parquet(meta_path, columns=["track_id", "artist", "title"])
                row = df[df["track_id"] == track_id]
                if not row.empty:
                    artist = row.iloc[0]["artist"]
                    title  = row.iloc[0]["title"]

            manifest_path = raw_dir / "manifest.json"
            existing: list = []
            if manifest_path.exists():
                with open(manifest_path, encoding="utf-8") as mf:
                    existing = json.load(mf)

            entry = {
                "filename":   downloaded_file.name,
                "track_id":   track_id,
                "artist":     artist,
                "title":      title,
                "position":   position,
                "duration_s": int(duration) if duration else None,
            }
            existing = [e for e in existing if e.get("filename") != entry["filename"]]
            existing.append(entry)

            with open(manifest_path, "w", encoding="utf-8") as mf:
                json.dump(existing, mf, ensure_ascii=False, indent=2)

            click.echo(f"  Manifest updated: {artist} — {title} ({track_id[:8]}...)")
        else:
            click.echo(
                "  Track not found in the database — manifest not updated.\n"
                "  Ingest this track first: python manage.py ingest"
            )
    except Exception as e:
        click.echo(f"  Manifest not updated: {e}", err=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ÉVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

@cli.group("eval", invoke_without_command=True)
@click.option("--n-tracks", default=0, show_default=True,
              help="Limit evaluation to N tracks from the manifest (0 = all)")
@click.option("--no-plot", is_flag=True, default=False,
              help="Skip plot generation")
@click.pass_context
def eval_group(ctx: click.Context, n_tracks: int, no_plot: bool) -> None:
    """Report-oriented evaluation analyses.

    Without a subcommand, run the base evaluation suite:
      - base-eval-rows (shared single-pass evaluation)
      - studio-mic
      - duration
      - stage12
      - mic-conditions
    """
    if ctx.invoked_subcommand is None:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK",      "1")
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
        from src import config
        from src.evaluation.report_analyses import run_base_eval_suite
        run_base_eval_suite(methods=[config.EMBEDDING_METHOD], n_tracks=n_tracks, plot=not no_plot)


@eval_group.command("base")
@click.option("--n-tracks", default=0, show_default=True,
              help="Limit evaluation to N tracks from the manifest (0 = all)")
@click.option("--no-plot", is_flag=True, default=False,
              help="Skip plot generation")
def eval_base(n_tracks: int, no_plot: bool) -> None:
    """Run the shared base evaluation suite (excluding RIR)."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK",      "1")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    from src import config
    from src.evaluation.report_analyses import run_base_eval_suite
    run_base_eval_suite(methods=[config.EMBEDDING_METHOD], n_tracks=n_tracks, plot=not no_plot)


@eval_group.command("studio-mic")
@click.option("--n-tracks", default=0, show_default=True,
              help="Limit evaluation to N tracks from the manifest (0 = all)")
@click.option("--no-plot", is_flag=True, default=False,
              help="Skip plot generation")
def eval_studio_mic(n_tracks: int, no_plot: bool) -> None:
    """Compare studio and microphone queries on the same tracks."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK",      "1")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    from src import config
    from src.evaluation.report_analyses import run_studio_mic_analysis
    run_studio_mic_analysis(methods=[config.EMBEDDING_METHOD], n_tracks=n_tracks, plot=not no_plot)


@eval_group.command("duration")
@click.option("--n-tracks", default=0, show_default=True,
              help="Limit evaluation to N tracks from the manifest (0 = all)")
@click.option("--no-plot", is_flag=True, default=False,
              help="Skip plot generation")
def eval_duration(n_tracks: int, no_plot: bool) -> None:
    """Analyze the effect of studio clip duration (5s / 15s / 30s)."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK",      "1")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    from src import config
    from src.evaluation.report_analyses import run_duration_analysis
    run_duration_analysis(methods=[config.EMBEDDING_METHOD], n_tracks=n_tracks, plot=not no_plot)


@eval_group.command("stage12")
@click.option("--n-tracks", default=0, show_default=True,
              help="Limit evaluation to N tracks from the manifest (0 = all)")
@click.option("--no-plot", is_flag=True, default=False,
              help="Skip plot generation")
def eval_stage12(n_tracks: int, no_plot: bool) -> None:
    """Compare Stage 1 (FAISS) with the final ranking after reranking."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK",      "1")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    from src import config
    from src.evaluation.report_analyses import run_stage12_analysis
    run_stage12_analysis(methods=[config.EMBEDDING_METHOD], n_tracks=n_tracks, plot=not no_plot)


@eval_group.command("rir")
@click.option("--n-tracks", default=0, show_default=True,
              help="Limit evaluation to N tracks from the manifest (0 = all)")
@click.option("--no-plot", is_flag=True, default=False,
              help="Skip plot generation")
def eval_rir(n_tracks: int, no_plot: bool) -> None:
    """Compare the system with and without RIR vectors in the index."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK",      "1")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    from src import config
    from src.evaluation.report_analyses import run_rir_analysis
    run_rir_analysis(methods=[config.EMBEDDING_METHOD], n_tracks=n_tracks, plot=not no_plot)


@eval_group.command("mic-conditions")
@click.option("--n-tracks", default=0, show_default=True,
              help="Limit evaluation to N tracks from the manifest (0 = all)")
@click.option("--no-plot", is_flag=True, default=False,
              help="Skip plot generation")
def eval_mic_conditions(n_tracks: int, no_plot: bool) -> None:
    """Analyze the effect of microphone conditions (distance, concurrent speech)."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK",      "1")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    from src import config
    from src.evaluation.report_analyses import run_mic_conditions_analysis
    run_mic_conditions_analysis(methods=[config.EMBEDDING_METHOD], n_tracks=n_tracks, plot=not no_plot)


# ═══════════════════════════════════════════════════════════════════════════════
# WEB
# ═══════════════════════════════════════════════════════════════════════════════

@cli.command()
@click.option("--prod", is_flag=True, default=False,
              help="Production mode: build the frontend and serve everything through FastAPI")
@click.option("--port", default=8000, show_default=True,
              help="FastAPI backend port")
@click.option("--reload/--no-reload", default=True, show_default=True,
              help="Enable or disable FastAPI hot reload in dev mode")
def webapp(prod: bool, port: int, reload: bool) -> None:
    """Start the FastAPI backend and the React/Vite frontend.

    \b
    Dev  : backend :8000 (reload) + Vite frontend :5173 (hot reload)
    Prod : static build + backend only on the selected port
    """
    import os
    import signal
    import subprocess
    import time

    frontend_dir = ROOT / "webapp" / "frontend"
    frontend_port = 5173

    try:
        subprocess.run(["npm", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        click.echo("`npm` not found. Install Node.js: https://nodejs.org/")
        sys.exit(1)

    if not (frontend_dir / "node_modules").exists():
        click.echo("Installing frontend dependencies (`npm install`) ...")
        result = subprocess.run(["npm", "install"], cwd=frontend_dir)
        if result.returncode != 0:
            click.echo("`npm install` failed.")
            sys.exit(1)
        click.echo("Dependencies installed.\n")

    processes: list[subprocess.Popen] = []

    def _port_owner(port_to_check: int) -> str | None:
        """Return a description of the process listening on a port, if any."""
        result = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port_to_check}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if len(lines) <= 1:
            return None
        return lines[1]

    def _ensure_port_available(port_to_check: int, label: str) -> None:
        owner = _port_owner(port_to_check)
        if owner is None:
            return
        click.echo(f"Port {port_to_check} is already in use ({label}).")
        click.echo(owner)
        click.echo("Stop the listed process, then rerun `python manage.py webapp`.")
        sys.exit(1)

    def _signal_process_tree(process: subprocess.Popen, sig: int) -> None:
        """Envoie un signal au groupe de processus si possible."""
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        except Exception:
            try:
                process.send_signal(sig)
            except Exception:
                pass

    def _cleanup(*_):
        click.echo("\nStopping servers...")
        for p in processes:
            try:
                _signal_process_tree(p, signal.SIGTERM)
            except Exception:
                pass
        time.sleep(0.5)
        for p in processes:
            try:
                if p.poll() is None:
                    _signal_process_tree(p, signal.SIGKILL)
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT,  _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    if prod:
        _ensure_port_available(port, "backend FastAPI")
        click.echo("Building the React frontend...")
        result = subprocess.run(["npm", "run", "build"], cwd=frontend_dir)
        if result.returncode != 0:
            click.echo("Frontend build failed.")
            sys.exit(1)
        click.echo(f"Starting on http://localhost:{port} ...")
        backend = subprocess.Popen(
            [sys.executable, "-m", "uvicorn",
             "webapp.backend.server:app",
             "--host", "0.0.0.0", "--port", str(port)],
            cwd=ROOT,
            start_new_session=True,
        )
        processes.append(backend)
        click.echo(f"Interface available at http://localhost:{port}")
    else:
        _ensure_port_available(port, "backend FastAPI")
        _ensure_port_available(frontend_port, "frontend Vite")
        click.echo(f"Starting FastAPI backend on http://localhost:{port} ...")
        backend_cmd = [
            sys.executable, "-m", "uvicorn",
            "webapp.backend.server:app",
            "--host", "0.0.0.0", "--port", str(port),
        ]
        if reload:
            backend_cmd.append("--reload")

        backend = subprocess.Popen(
            backend_cmd,
            cwd=ROOT,
            start_new_session=True,
        )
        processes.append(backend)
        time.sleep(1)
        if backend.poll() is not None:
            click.echo("The backend failed to start. The frontend will not be launched.")
            sys.exit(1)
        click.echo("Starting Vite frontend (hot reload)...")
        frontend = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=frontend_dir,
            start_new_session=True,
        )
        processes.append(frontend)
        click.echo(f"Interface available at http://localhost:{frontend_port}")
        click.echo(f"Backend API         → http://localhost:{port}")

    click.echo("Press Ctrl+C to stop\n")
    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        _cleanup()


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cli()
