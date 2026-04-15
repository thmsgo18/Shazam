"""
src/api/app.py

Canonical entry point for project audio identification.

This module serves as:

- a CLI interface (`python src/api/app.py ...`)
- a reusable layer for `manage.py identify`
- a reusable layer for the FastAPI web backend
"""

from __future__ import annotations

import math
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*upsample_bicubic2d.*", category=UserWarning)

try:
    import click
except ModuleNotFoundError:  # pragma: no cover - dépendance CLI optionnelle
    click = None

ROOT          = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.config as config

METADATA_PATH = ROOT / "data" / "processed" / "metadata.parquet"
_METADATA_CACHE: dict[str, dict] | None = None
_METADATA_MTIME_NS: int | None = None


def ensure_project_root_context() -> None:
    """Enforces a consistent execution context for all entry points."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)


def _clean(val, default=None):
    """Returns default for NaN / empty strings, otherwise the value."""
    if val is None:
        return default
    if isinstance(val, float) and math.isnan(val):
        return default
    return val if val != "" else default


def _load_metadata(force_reload: bool = False) -> dict[str, dict]:
    """Returns {track_id: {...}} from metadata.parquet."""
    import pandas as pd

    global _METADATA_CACHE, _METADATA_MTIME_NS

    if not METADATA_PATH.exists():
        _METADATA_CACHE = {}
        _METADATA_MTIME_NS = None
        return {}

    current_mtime_ns = METADATA_PATH.stat().st_mtime_ns
    if not force_reload and _METADATA_CACHE is not None and _METADATA_MTIME_NS == current_mtime_ns:
        return _METADATA_CACHE

    df = pd.read_parquet(METADATA_PATH)
    metadata: dict[str, dict] = {}
    for row in df.itertuples(index=False):
        track_id = str(row.track_id)
        metadata[track_id] = {
            "title":      _clean(getattr(row, "title", None), track_id),
            "artist":     _clean(getattr(row, "artist", None), "Unknown"),
            "album":      _clean(getattr(row, "album", None)),
            "genre":      _clean(getattr(row, "genre", None)),
            "duration_s": _clean(getattr(row, "duration_s", None)),
            "cover_url":  _clean(getattr(row, "cover_url", None)),
        }
    _METADATA_CACHE = metadata
    _METADATA_MTIME_NS = current_mtime_ns
    return metadata


def warmup_runtime(
    method: str | None = None,
    include_model: bool = False,
    local_files_only: bool = True,
) -> dict:
    """
    Preload the resource-intensive components used by the web API into memory.

    Objectives:

    - Avoid the overhead of the first /api/identify call
    - Load the FAISS index and segments
    - Load the metadata
    - Optionally initialize the configured embedding model
    """
    ensure_project_root_context()

    chosen_method = (method or config.EMBEDDING_METHOD).lower()
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    metadata = _load_metadata()
    timings["metadata_s"] = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    from src.retrieval.searcher import load_searcher
    index, segments = load_searcher(chosen_method)
    timings["searcher_s"] = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    from src.retrieval.query_pipeline import warmup_fingerprint_store
    fingerprint_info = warmup_fingerprint_store()
    timings["fingerprints_s"] = round(time.perf_counter() - t0, 3)

    model_warmed = False
    if include_model:
        t0 = time.perf_counter()
        if chosen_method == "clap":
            from src.features.embeddings_audio import _load_clap
            _load_clap(config.CLAP_MODEL_NAME, local_files_only=local_files_only)
        elif chosen_method == "muq":
            from src.features.embeddings_audio import _load_muq
            _load_muq(config.MUQ_MODEL_NAME, local_files_only=local_files_only)
        elif chosen_method == "mert":
            from src.features.embeddings_audio import _load_mert
            _load_mert(config.MERT_MODEL_NAME, local_files_only=local_files_only)
        timings["model_s"] = round(time.perf_counter() - t0, 3)
        model_warmed = True

    return {
        "method": chosen_method,
        "tracks": len(metadata),
        "vectors": int(index.ntotal),
        "segments": int(len(segments)),
        "fingerprints": fingerprint_info,
        "model_warmed": model_warmed,
        "local_files_only": local_files_only,
        "timings": timings,
    }


def _streaming_links(artist: str, title: str) -> dict[str, str]:
    """Builds search links to streaming platforms."""
    query = f"{artist} {title}".replace(" ", "+")
    return {
        "youtube": f"https://www.youtube.com/results?search_query={query}",
        "spotify": f"https://open.spotify.com/search/{query}",
        "deezer":  f"https://www.deezer.com/search/{query}",
        "apple":   f"https://music.apple.com/search?term={query}",
    }


def _recommendations(track_id: str, genre: str | None, metadata: dict[str, dict], top: int = 4) -> list[dict]:
    """Returns simple recommendations based on the genre."""
    if not genre:
        return []

    recommendations = []
    for candidate_id, info in metadata.items():
        if candidate_id == track_id or info.get("genre") != genre:
            continue
        recommendations.append({
            "track_id":  candidate_id,
            "title":     info["title"],
            "artist":    info["artist"],
            "album":     info.get("album"),
            "cover_url": info.get("cover_url"),
            "streaming": _streaming_links(info["artist"], info["title"]),
        })
        if len(recommendations) >= top:
            break
    return recommendations


def get_ui_config() -> dict:
    """Expose the UI configuration read from src/config.py."""
    return {
        "listen_duration": config.UI_LISTEN_DURATION,
        "confidence_ratio": config.UI_CONFIDENCE_RATIO,
        "embedding_method": config.EMBEDDING_METHOD,
    }


def identify_audio(
    audio_file: str,
    method: str | None = None,
    top: int | None = None,
    detailed: bool = False,
    metadata: dict[str, dict] | None = None,
) -> list[dict]:
    """Returns the identification results enriched with metadata."""
    ensure_project_root_context()

    top_n = top if top is not None else config.VECTOR_TOP_N_RESULTS
    from src.retrieval.query_pipeline import identify_track

    raw_results = identify_track(audio_file, method=method, top_n=top_n, detailed=detailed)
    if not raw_results:
        return []

    metadata = metadata if metadata is not None else _load_metadata()
    results = []
    for rank, row in enumerate(raw_results, start=1):
        track_id = row[0]
        score = row[1]
        score_faiss = row[2] if detailed else None
        score_fp = row[3] if detailed else score
        info = metadata.get(track_id, {})
        title = info.get("title", track_id)
        artist = info.get("artist", "Unknown")
        results.append({
            "rank":       rank,
            "track_id":   track_id,
            "title":      title,
            "artist":     artist,
            "album":      info.get("album"),
            "genre":      info.get("genre"),
            "duration_s": info.get("duration_s"),
            "cover_url":  info.get("cover_url"),
            "score":      round(score, 4),
            "score_faiss": round(score_faiss, 4) if score_faiss is not None else None,
            "score_fp":   round(score_fp, 4) if score_fp is not None else None,
            "streaming":  _streaming_links(artist, title),
        })
    return results


def build_identification_response(
    audio_file: str,
    method: str | None = None,
    top: int | None = None,
    detailed: bool = False,
) -> dict:
    """Builds the canonical response used by the web interface."""
    metadata = _load_metadata()
    results = identify_audio(audio_file, method=method, top=top, detailed=detailed, metadata=metadata)
    if not results:
        return {"results": [], "confident": False, "recommendations": []}

    confident = False
    if len(results) >= 2 and results[1]["score"] > 0:
        confident = (results[0]["score"] / results[1]["score"]) >= config.UI_CONFIDENCE_RATIO
    elif len(results) == 1:
        confident = True

    best = results[0]
    recommendations = _recommendations(best["track_id"], best.get("genre"), metadata)
    return {
        "results": results,
        "confident": confident,
        "recommendations": recommendations,
    }


def render_identification_table(results: list[dict], detailed: bool = False) -> None:
    """Render identification results in a Rich table."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=3)
    table.add_column("Artist", width=24)
    table.add_column("Title", width=34)
    table.add_column("Final score", justify="right")
    if detailed:
        table.add_column("FAISS score", justify="right")
        table.add_column("FP score", justify="right")

    for row in results:
        artist = row["artist"][:24]
        title = row["title"][:34]
        if detailed:
            table.add_row(
                str(row["rank"]),
                artist,
                title,
                f"{row['score']:.4f}",
                f"{row['score_faiss']:.4f}",
                f"{row['score_fp']:.4f}",
            )
        else:
            table.add_row(str(row["rank"]), artist, title, f"{row['score']:.4f}")

    console.print(table)


def run_identify_cli(audio_file: str, method: str | None, top: int, detailed: bool) -> None:
    """Run canonical identification and print it for the CLI."""
    from rich.console import Console

    console = Console()
    console.print(f"\n[bold cyan]Identifying:[/bold cyan] {audio_file}")
    if method:
        console.print(f"[dim]Method: {method}[/dim]\n")

    results = identify_audio(audio_file, method=method, top=top, detailed=detailed)
    if not results:
        console.print("[red]No result found.[/red]")
        return

    render_identification_table(results, detailed=detailed)


if click is not None:
    @click.command()
    @click.argument("audio_file", type=click.Path(exists=True))
    @click.option("--method",   default=None,  help="mfcc / clap / muq (default: config.py)")
    @click.option("--top",      default=5,     show_default=True, help="Number of results to display")
    @click.option("--detailed", is_flag=True,  default=False, help="Show FAISS and fingerprint scores separately")
    def identify(audio_file: str, method: str | None, top: int, detailed: bool) -> None:
        """
        Identify the track corresponding to AUDIO_FILE.

        Display the most likely tracks with their scores.
        With --detailed: also show FAISS and fingerprint scores separately.
        """
        run_identify_cli(audio_file, method=method, top=top, detailed=detailed)
else:
    def identify(*_args, **_kwargs) -> None:
        raise RuntimeError("click is required to run src/api/app.py as a CLI entrypoint.")


if __name__ == "__main__":
    identify()
