"""
src/evaluation/rir_impact.py

Measure the impact of RIR on FAISS score WITHOUT deleting the database.

Strategy:
  - Load only original vectors (without _rir_) from ChromaDB
  - Build a temporary FAISS index in memory (no file)
  - Compare the target track position WITH and WITHOUT RIR in the index

Public entry point: run_rir_impact(audio, target_track_id, top, method)
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import chromadb
import faiss
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn, TextColumn
from rich.table import Table
from rich import box

import src.config as config
from src.audio.loading import load_audio
from src.audio.preprocessing import iter_segments, preprocess_query
from src.features.embeddings_audio import embed_segment

ROOT    = Path(__file__).resolve().parents[2]
console = Console()
_NO_RIR_CACHE: dict[str, tuple[faiss.Index, pd.DataFrame]] = {}

FLOWERS_ID    = "f01ab00f1fdc5a57fd2676f4d68631a8"
AUDIO_DEFAULT = str(ROOT / "data" / "raw" / "93-Rue-Belliard.mp3")
PAGE          = 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_no_rir_index(collection_key: str):
    """
    Load only NON-RIR vectors from ChromaDB and build
    a FAISS index in memory + DataFrame segments.
    """
    client     = chromadb.PersistentClient(path=str(ROOT / config.CHROMA_DIR))
    collection = client.get_collection(name=collection_key)
    total      = collection.count()

    embeddings_list: list = []
    metadatas_list:  list = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("{task.fields[current]}"),
        console=console,
        transient=False,
    ) as prog:
        task   = prog.add_task("No-RIR index", total=total, current="fetching ChromaDB vectors")
        offset = 0
        while True:
            page = collection.get(limit=PAGE, offset=offset, include=["embeddings", "metadatas"])
            if not page["ids"]:
                break
            for id_, emb, meta in zip(page["ids"], page["embeddings"], page["metadatas"]):
                if "_rir_" not in id_:
                    embeddings_list.append(emb)
                    metadatas_list.append(meta)
            prog.advance(task, len(page["ids"]))
            prog.update(task, current=f"kept {len(embeddings_list):,} original vectors")
            if len(page["ids"]) < PAGE:
                break
            offset += PAGE

    console.print(f"  {len(embeddings_list):,} original vectors loaded (out of {total:,} total)")

    console.print("  Converting to numpy array…")
    xb = np.array(embeddings_list, dtype=np.float32)
    del embeddings_list

    console.print(f"  Array shape: {xb.shape} — L2 normalization…")
    norms = np.linalg.norm(xb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    xb   /= norms

    console.print("  Building FAISS index…")
    faiss.omp_set_num_threads(1)
    index = faiss.IndexFlatIP(xb.shape[1])
    index.add(xb)
    del xb
    del norms
    gc.collect()
    console.print(f"  [green]✓ Index ready ({index.ntotal:,} vectors)[/green]\n")

    segments = pd.DataFrame(metadatas_list)
    return index, segments


def _no_rir_index_paths(collection_key: str) -> tuple[Path, Path]:
    index_dir = ROOT / config.INDEX_DIR
    index_dir.mkdir(parents=True, exist_ok=True)
    return (
        index_dir / f"index_{collection_key}_no_rir_flat.faiss",
        index_dir / f"segments_{collection_key}_no_rir.parquet",
    )


def load_no_rir_index_cached(collection_key: str, force_rebuild: bool = False) -> tuple[faiss.Index, pd.DataFrame]:
    """
    Load the WITHOUT RIR index from disk cache if available.
    Otherwise build it from ChromaDB, then save it.
    """
    if not force_rebuild and collection_key in _NO_RIR_CACHE:
        return _NO_RIR_CACHE[collection_key]

    index_path, seg_path = _no_rir_index_paths(collection_key)
    if not force_rebuild and index_path.exists() and seg_path.exists():
        console.print("[yellow]Loading WITHOUT RIR index from disk…[/yellow]")
        index = faiss.read_index(str(index_path))
        segments = pd.read_parquet(seg_path)
        console.print(f"  [green]✓ WITHOUT RIR index loaded ({index.ntotal:,} vectors)[/green]\n")
        _NO_RIR_CACHE[collection_key] = (index, segments)
        return _NO_RIR_CACHE[collection_key]

    console.print("[yellow]Building WITHOUT RIR index…[/yellow]")
    index, segments = _load_no_rir_index(collection_key)
    console.print("[yellow]Saving WITHOUT RIR index…[/yellow]")
    faiss.write_index(index, str(index_path))
    segments.to_parquet(seg_path, index=False)
    console.print(f"  [green]✓ WITHOUT RIR disk cache written[/green]\n")
    _NO_RIR_CACHE[collection_key] = (index, segments)
    return _NO_RIR_CACHE[collection_key]


def _search(index, segments: pd.DataFrame, query_emb: np.ndarray, k: int) -> dict[str, float]:
    """FAISS search + aggregation by track_id."""
    q = query_emb.reshape(1, -1).astype(np.float32)
    q /= np.linalg.norm(q, keepdims=True).clip(min=1e-10)
    dists, idxs = index.search(q, k)

    scores: dict[str, float] = {}
    for idx, dist in zip(idxs[0], dists[0]):
        if idx < 0 or idx >= len(segments):
            continue
        tid = segments.iloc[idx]["track_id"]
        scores[tid] = scores.get(tid, 0.0) + float(dist)
    return scores


def _rank_label(rank: int | None) -> str:
    if rank is None:
        return "[red]NF[/red]"
    if rank == 1:
        return f"[bold green]#{rank} ✅[/bold green]"
    if rank <= 3:
        return f"[green]#{rank}[/green]"
    if rank <= 10:
        return f"[yellow]#{rank}[/yellow]"
    return f"[red]#{rank}[/red]"


def _load_model(method: str, verbose: bool = True) -> None:
    """Load the model BEFORE faiss (Apple Silicon)."""
    if method == "clap":
        from src.features.embeddings_audio import _CLAP_CACHE, _load_clap
        already_loaded = (
            _CLAP_CACHE.get("model") is not None
            and _CLAP_CACHE.get("model_name") == config.CLAP_MODEL_NAME
        )
        if verbose and not already_loaded:
            console.print(f"[cyan]Loading {config.CLAP_MODEL_NAME}…[/cyan]")
        _load_clap(config.CLAP_MODEL_NAME)
        if verbose and not already_loaded:
            console.print("[green]✓ Model ready.[/green]\n")
    elif method == "muq":
        from src.features.embeddings_audio import _load_muq
        _load_muq(config.MUQ_MODEL_NAME)
    elif method == "mert":
        from src.features.embeddings_audio import _load_mert
        _load_mert(config.MERT_MODEL_NAME)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_rir_impact(
    audio:           str = AUDIO_DEFAULT,
    target_track_id: str = FLOWERS_ID,
    top:             int = 20,
    method:          str | None = None,
) -> None:
    """
    Compare the target track position WITH and WITHOUT RIR vectors in the index.

    Args:
        audio:           path to the audio file to test.
        target_track_id: track_id to track (default: Flowers).
        top:             number of results in the tables.
        method:          embedding method (default: config.EMBEDDING_METHOD).
    """
    if not Path(audio).exists():
        console.print(f"[red]File not found: {audio}[/red]")
        sys.exit(1)

    if method is None:
        method = config.EMBEDDING_METHOD
    collection_key = config.get_collection_key(method)

    targ_sr = {
        "clap": config.CLAP_SAMPLE_RATE,
        "muq":  config.MUQ_SAMPLE_RATE,
        "mert": config.MERT_SAMPLE_RATE,
    }.get(method, config.SAMPLE_RATE)

    console.print(Panel(
        f"[bold]Method  :[/bold] [cyan]{method}[/cyan]\n"
        f"[bold]Audio    :[/bold] [cyan]{audio}[/cyan]\n"
        f"[bold]Goal     :[/bold] compare FAISS with vs without RIR vectors",
        title="[bold cyan]RIR Impact Test[/bold cyan]",
        expand=False,
    ))

    # Load model BEFORE faiss (Apple Silicon)
    _load_model(method, verbose=False)

    # Embeddings of the query
    console.print("[yellow]Preparing audio query…[/yellow]")
    waveform, sr = load_audio(audio, target_sr=targ_sr)
    waveform     = preprocess_query(waveform, sr)
    seg_list     = [seg for _, seg in iter_segments(waveform=waveform, sr=sr)]
    console.print(f"  {len(seg_list)} segments to embed\n")

    query_embeddings = []
    for seg in seg_list:
        emb = embed_segment(
            seg, sr, method=method,
            clap_model_name=config.CLAP_MODEL_NAME,
            muq_model_name=config.MUQ_MODEL_NAME,
            mert_model_name=config.MERT_MODEL_NAME,
        )
        query_embeddings.append(emb)

    # Index WITHOUT RIR (in memory)
    index_no_rir, segments_no_rir = load_no_rir_index_cached(collection_key)
    console.print(f"  Index without RIR: [white]{index_no_rir.ntotal:,}[/white] vectors\n")

    # Index WITH RIR (existing file)
    console.print("[yellow]Loading index WITH RIR…[/yellow]")
    from src.retrieval.searcher import load_searcher
    index_rir, segments_rir = load_searcher(method)
    console.print(f"  Index with RIR: [white]{index_rir.ntotal:,}[/white] vectors\n")

    # Search across both indexes
    k = config.VECTOR_TOP_K_SEGMENTS

    scores_no_rir: dict[str, float] = {}
    scores_rir:    dict[str, float] = {}

    for emb in query_embeddings:
        for tid, s in _search(index_no_rir, segments_no_rir, emb, k).items():
            scores_no_rir[tid] = scores_no_rir.get(tid, 0.0) + s
        for tid, s in _search(index_rir, segments_rir, emb, k).items():
            scores_rir[tid] = scores_rir.get(tid, 0.0) + s

    ranked_no_rir = sorted(scores_no_rir.items(), key=lambda x: x[1], reverse=True)
    ranked_rir    = sorted(scores_rir.items(),    key=lambda x: x[1], reverse=True)

    rank_no_rir  = next((i+1 for i, (t, _) in enumerate(ranked_no_rir) if t == target_track_id), None)
    rank_rir     = next((i+1 for i, (t, _) in enumerate(ranked_rir)    if t == target_track_id), None)
    score_no_rir = scores_no_rir.get(target_track_id, 0.0)
    score_rir    = scores_rir.get(target_track_id, 0.0)

    # Result
    console.rule("[bold]Stage 1 Result — FAISS")

    t = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE_HEAD)
    t.add_column("Condition",        width=20)
    t.add_column("Target Position",  width=18)
    t.add_column("FAISS Score",     justify="right", width=12)
    t.add_column("Index Vectors",  justify="right", width=15)

    t.add_row("Without RIR", _rank_label(rank_no_rir), f"{score_no_rir:.4f}", f"{index_no_rir.ntotal:,}")
    t.add_row("With RIR", _rank_label(rank_rir),    f"{score_rir:.4f}",    f"{index_rir.ntotal:,}")
    console.print(t)

    if rank_rir is not None and rank_no_rir is not None:
        delta = rank_no_rir - rank_rir
        if delta > 0:
            console.print(f"\n[green]✓ RIR improves by {delta} positions ({rank_no_rir} → {rank_rir})[/green]")
        elif delta < 0:
            console.print(f"\n[red]✗ RIR degrades by {abs(delta)} positions ({rank_no_rir} → {rank_rir})[/red]")
        else:
            console.print(f"\n[yellow]= RIR has no impact on position ({rank_rir})[/yellow]")
    console.print()

    # ── Structured data (useful for run_rir_evaluate) ──────────────────
    run_rir_impact._last_result = {
        "with_rir":    {"rank": rank_rir,    "faiss_score": round(score_rir,    4), "n_vectors": index_rir.ntotal},
        "without_rir": {"rank": rank_no_rir, "faiss_score": round(score_no_rir, 4), "n_vectors": index_no_rir.ntotal},
    }

    # Top-N of both indices for comparison
    meta_path = ROOT / config.METADATA_PATH
    meta_df   = pd.read_parquet(meta_path, columns=["track_id", "title", "artist"])
    meta      = {r.track_id: f"{r.artist[:18]} — {r.title[:25]}" for r in meta_df.itertuples()}

    for label_str, ranked in [("Without RIR", ranked_no_rir), ("With RIR", ranked_rir)]:
        console.rule(f"[dim]Top {top} — {label_str}")
        t2 = Table(show_header=True, header_style="bold", box=box.SIMPLE)
        t2.add_column("#",     width=4, style="dim")
        t2.add_column("Track", width=50)
        t2.add_column("Score", justify="right", width=10)

        displayed    = 0
        target_shown = False
        for rank, (tid, score) in enumerate(ranked, 1):
            is_target = tid == target_track_id
            if displayed >= top and not (is_target and not target_shown):
                if target_shown:
                    break
                continue
            style = "[bold green]" if is_target else ""
            t2.add_row(
                f"{style}#{rank}[/bold green]" if is_target else f"#{rank}",
                f"{style}{meta.get(tid, tid)}{'  ← 🎯 TARGET' if is_target else ''}",
                f"{score:.4f}",
            )
            if is_target:
                target_shown = True
            displayed += 1
        console.print(t2)


# ---------------------------------------------------------------------------
# Silent API for run_rir_evaluate (no display)
# ---------------------------------------------------------------------------

def rir_impact_scores(
    audio_path: str,
    track_id:   str,
    method:     str | None = None,
    prebuilt_no_rir: tuple | None = None,
) -> dict:
    """
    Compare Stage 1 FAISS scores for an audio file: with vs without RIR.
    Produces no display — returns structured data.

    Args:
        audio_path:       path to the audio file (temporary WAV accepted).
        track_id:         track_id expected as correct answer.
        method:           embedding method (default: config.EMBEDDING_METHOD).
        prebuilt_no_rir:  (index, segments) already built — multi-track optimization.

    Returns:
        {
          "with_rir":    {"rank": int|None, "faiss_score": float, "n_vectors": int},
          "without_rir": {"rank": int|None, "faiss_score": float, "n_vectors": int},
        }
    """
    if method is None:
        method = config.EMBEDDING_METHOD
    collection_key = config.get_collection_key(method)

    targ_sr = {
        "clap": config.CLAP_SAMPLE_RATE,
        "muq":  config.MUQ_SAMPLE_RATE,
        "mert": config.MERT_SAMPLE_RATE,
    }.get(method, config.SAMPLE_RATE)

    # Load the model BEFORE faiss (Apple Silicon)
    _load_model(method, verbose=False)

    # Embedding the query segments
    waveform, sr = load_audio(audio_path, target_sr=targ_sr)
    waveform     = preprocess_query(waveform, sr)
    seg_list     = [seg for _, seg in iter_segments(waveform=waveform, sr=sr)]

    query_embeddings = []
    for seg in seg_list:
        emb = embed_segment(
            seg, sr, method=method,
            clap_model_name=config.CLAP_MODEL_NAME,
            muq_model_name=config.MUQ_MODEL_NAME,
            mert_model_name=config.MERT_MODEL_NAME,
        )
        query_embeddings.append(emb)

    # WITHOUT RIR Index
    if prebuilt_no_rir is not None:
        index_no_rir, segments_no_rir = prebuilt_no_rir
    else:
        index_no_rir, segments_no_rir = load_no_rir_index_cached(collection_key)

    # WITH RIR Index (existing file on disk)
    from src.retrieval.searcher import load_searcher
    index_rir, segments_rir = load_searcher(method)

    k = config.VECTOR_TOP_K_SEGMENTS
    scores_no_rir: dict[str, float] = {}
    scores_rir:    dict[str, float] = {}

    for emb in query_embeddings:
        for tid, s in _search(index_no_rir, segments_no_rir, emb, k).items():
            scores_no_rir[tid] = scores_no_rir.get(tid, 0.0) + s
        for tid, s in _search(index_rir, segments_rir, emb, k).items():
            scores_rir[tid] = scores_rir.get(tid, 0.0) + s

    ranked_no_rir = sorted(scores_no_rir.items(), key=lambda x: x[1], reverse=True)
    ranked_rir    = sorted(scores_rir.items(),    key=lambda x: x[1], reverse=True)

    rank_no_rir = next(
        (i + 1 for i, (t, _) in enumerate(ranked_no_rir) if t == track_id), None
    )
    rank_rir = next(
        (i + 1 for i, (t, _) in enumerate(ranked_rir) if t == track_id), None
    )

    return {
        "with_rir": {
            "rank":        rank_rir,
            "faiss_score": round(scores_rir.get(track_id, 0.0),    4),
            "n_vectors":   index_rir.ntotal,
        },
        "without_rir": {
            "rank":        rank_no_rir,
            "faiss_score": round(scores_no_rir.get(track_id, 0.0), 4),
            "n_vectors":   index_no_rir.ntotal,
        },
    }
