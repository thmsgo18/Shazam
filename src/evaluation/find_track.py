"""
src/evaluation/find_track.py

Complete pipeline on an audio file, displaying the position
of a target track at each step: Stage 1 (FAISS) and Stage 2 (Fingerprint).

Public entry point: run_find_track(audio, target_track_id, top, method)
"""

from __future__ import annotations

import os
import pickle
import sqlite3
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import warnings
warnings.filterwarnings("ignore", message=".*upsample_bicubic2d.*", category=UserWarning)

import librosa
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

import src.config as config
from src.audio.loading import load_audio
from src.audio.preprocessing import iter_segments, preprocess_query
from src.features.embeddings_audio import embed_segment

ROOT    = Path(__file__).resolve().parents[2]
console = Console()

FLOWERS_ID    = "f01ab00f1fdc5a57fd2676f4d68631a8"
AUDIO_DEFAULT = str(ROOT / "data" / "raw" / "93-Rue-Belliard.mp3")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_metadata() -> dict[str, dict]:
    path = ROOT / config.METADATA_PATH
    if not path.exists():
        return {}
    df = pd.read_parquet(path, columns=["track_id", "title", "artist"])
    return {row.track_id: {"title": row.title, "artist": row.artist} for row in df.itertuples()}


def _get_fp(track_id: str) -> set | None:
    fp_db = ROOT / config.FINGERPRINTS_DB
    if not fp_db.exists():
        return None
    with sqlite3.connect(fp_db) as conn:
        row = conn.execute(
            "SELECT hashes FROM fingerprints WHERE track_id = ?", (track_id,)
        ).fetchone()
    return pickle.loads(row[0]) if row else None


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


def _get_target_sr(method: str) -> int:
    return {
        "clap": config.CLAP_SAMPLE_RATE,
        "muq":  config.MUQ_SAMPLE_RATE,
        "mert": config.MERT_SAMPLE_RATE,
    }.get(method, config.SAMPLE_RATE)


def _load_model(method: str) -> None:
    """Load the PyTorch model BEFORE any import of files (Apple Silicon)."""
    if method == "clap":
        from src.features.embeddings_audio import _load_clap
        console.print(f"[cyan]Loading the {config.CLAP_MODEL_NAME} model...[/cyan]")
        _load_clap(config.CLAP_MODEL_NAME)
        console.print("[green]✓ Model ready.[/green]\n")
    elif method == "muq":
        from src.features.embeddings_audio import _load_muq
        console.print(f"[cyan]Loading the {config.MUQ_MODEL_NAME} model...[/cyan]")
        _load_muq(config.MUQ_MODEL_NAME)
        console.print("[green]✓ Model ready.[/green]\n")
    elif method == "mert":
        from src.features.embeddings_audio import _load_mert
        console.print(f"[cyan]Loading the {config.MERT_MODEL_NAME} model...[/cyan]")
        _load_mert(config.MERT_MODEL_NAME)
        console.print("[green]✓ Model ready.[/green]\n")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_find_track(
    audio:           str = AUDIO_DEFAULT,
    target_track_id: str = FLOWERS_ID,
    top:             int = 20,
    method:          str | None = None,
) -> None:
    """
    Analyzes an audio file and displays the position of the target track at each step.

    Args:
        audio:           path to the audio file to analyze.
        target_track_id: track_id to follow in the rankings (default: Flowers).
        top:             number of results to display in the tables.
        method:          embedding method (default: config.EMBEDDING_METHOD).
    """
    if not Path(audio).exists():
        console.print(f"[red]File not found: {audio}[/red]")
        sys.exit(1)

    if method is None:
        method = config.EMBEDDING_METHOD

    targ_sr  = _get_target_sr(method)
    metadata = _load_metadata()

    def label(track_id: str) -> str:
        info   = metadata.get(track_id, {})
        artist = info.get("artist", track_id)[:20]
        title  = info.get("title",  "—")[:28]
        return f"{artist} — {title}"

    console.print(f"\n[bold cyan]File:[/bold cyan] {audio}")
    console.print(f"[bold cyan]Method:[/bold cyan] {method}")
    console.print(f"[bold cyan]Target:[/bold cyan] {label(target_track_id)}\n")

    # Front-loading model (Apple Silicon)
    _load_model(method)

    # Lazy import of searcher — AFTER the model to avoid Accelerate conflict
    from src.retrieval.searcher import load_searcher, search_segments, aggregate_by_track

    # ── Stage 1: FAISS ──────────────────────────────────────────────────────
    console.print("[yellow]Stage 1 — Loading index + embeddings...[/yellow]")
    index, segments = load_searcher(method)

    waveform, sr = load_audio(audio, target_sr=targ_sr)
    waveform = preprocess_query(waveform, sr)
    seg_list = [seg for _, seg in iter_segments(waveform=waveform, sr=sr)]
    console.print(f"  {len(seg_list)} segments | index : {index.ntotal} vectors\n")

    global_scores: dict[str, float] = {}
    for seg in seg_list:
        emb = embed_segment(
            seg, sr, method=method,
            clap_model_name=config.CLAP_MODEL_NAME,
            muq_model_name=config.MUQ_MODEL_NAME,
            mert_model_name=config.MERT_MODEL_NAME,
        )
        dists, idxs = search_segments(index=index, query_embedding=emb, k=config.VECTOR_TOP_K_SEGMENTS)
        for tid, score in aggregate_by_track(idxs, dists, segments):
            global_scores[tid] = global_scores.get(tid, 0.0) + score

    ranked_s1 = sorted(global_scores.items(), key=lambda x: x[1], reverse=True)

    rank_s1  = next((i + 1 for i, (tid, _) in enumerate(ranked_s1) if tid == target_track_id), None)
    score_s1 = global_scores.get(target_track_id, 0.0)

    console.rule("[bold]Stage 1 — FAISS (embedding)")
    console.print(f"  Target position: {_rank_label(rank_s1)}  |  FAISS score  = [cyan]{score_s1:.4f}[/cyan]")
    if rank_s1 and rank_s1 > 1:
        top1_tid, top1_score = ranked_s1[0]
        console.print(f"  Top-1          : {label(top1_tid)}  score = [cyan]{top1_score:.4f}[/cyan]")
        console.print(f"  Gap target/Top-1: [red]{top1_score / score_s1:.1f}×[/red] higher\n")
    else:
        console.print()

    t1 = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
    t1.add_column("#",            width=4,  style="dim")
    t1.add_column("Artist — Title",        width=52)
    t1.add_column("FAISS Score",  justify="right", width=12)

    target_shown = rank_s1 is None or rank_s1 <= top
    for rank, (tid, score) in enumerate(ranked_s1, 1):
        is_target = tid == target_track_id
        if rank <= top or is_target:
            style  = "bold green" if is_target else ("dim" if rank > top else "")
            marker = " ← 🎯 TARGET" if is_target else ""
            t1.add_row(str(rank), label(tid) + marker, f"{score:.4f}", style=style)
        if rank > top and target_shown:
            break

    console.print(t1)

    # ── Stage 2: Fingerprinting ──────────────────────────────────────────────
    from src.features.fingerprint import extract_fingerprint, fingerprint_similarity

    console.rule("[bold]Stage 2 — Fingerprint (re-ranking)")

    candidates = ranked_s1[:config.VECTOR_TOP_N_TRACKS]
    target_in_candidates = any(tid == target_track_id for tid, _ in candidates)

    if not target_in_candidates:
        console.print(
            f"  [red]⚠ The target is not in the {config.VECTOR_TOP_N_TRACKS} candidates of Stage 2.[/red]"
        )
        console.print(f"  [dim](FAISS rank = #{rank_s1} > cutoff = {config.VECTOR_TOP_N_TRACKS})[/dim]\n")
    else:
        console.print(f"  [green]✓ The target is in the {config.VECTOR_TOP_N_TRACKS} candidates of Stage 2.[/green]\n")

    if targ_sr != config.SAMPLE_RATE:
        wf_fp = librosa.resample(waveform, orig_sr=targ_sr, target_sr=config.SAMPLE_RATE)
    else:
        wf_fp = waveform
    query_fp = extract_fingerprint(wf_fp, config.SAMPLE_RATE)
    console.print(f"  {len(query_fp)} hashes extracted from the query\n")

    final: list[tuple[str, float, float, float]] = []
    for tid, score_faiss in candidates:
        fp = _get_fp(tid)
        if fp is None or len(fp) == 0:
            score_fp = 0.0
        else:
            score_fp = fingerprint_similarity(query_fp, fp)
        score_final = score_faiss * (1.0 + score_fp)
        final.append((tid, score_final, score_faiss, score_fp))

    final.sort(key=lambda x: x[1], reverse=True)

    rank_s2    = next((i + 1 for i, (tid, *_) in enumerate(final) if tid == target_track_id), None)
    target_s2  = next(((sf, sf_faiss, fp) for tid, sf, sf_faiss, fp in final if tid == target_track_id), None)

    console.print(f"  Target position: {_rank_label(rank_s2)}", end="")
    if target_s2:
        console.print(
            f"  |  final score = [cyan]{target_s2[0]:.4f}[/cyan]  "
            f"(faiss={target_s2[1]:.4f}  fp={target_s2[2]:.4f})"
        )
    elif not target_in_candidates:
        console.print("  [dim](not in candidates)[/dim]")
    else:
        console.print()

    if rank_s2 and rank_s2 > 1 and final:
        top1_tid, top1_sf, *_ = final[0]
        console.print(f"  Final Top-1: {label(top1_tid)}  score = [cyan]{top1_sf:.4f}[/cyan]\n")
    else:
        console.print()

    t2 = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
    t2.add_column("#",           width=4,  style="dim")
    t2.add_column("Artist — Title",       width=48)
    t2.add_column("Score final", justify="right", width=12)
    t2.add_column("Score FAISS", justify="right", width=12)
    t2.add_column("Score FP",    justify="right", width=10)

    for rank, (tid, sf, sf_faiss, fp_score) in enumerate(final[:top], 1):
        is_target = tid == target_track_id
        style  = "bold green" if is_target else ""
        marker = " ← 🎯" if is_target else ""
        t2.add_row(
            str(rank), label(tid) + marker,
            f"{sf:.4f}", f"{sf_faiss:.4f}", f"{fp_score:.4f}",
            style=style,
        )

    console.print(t2)

    # ── Summary ────────────────────────────────────────────────────────────────
    console.rule("[bold]Summary")
    console.print(f"  Stage 1 (FAISS)      : {_rank_label(rank_s1)}  score={score_s1:.4f}")
    if target_in_candidates and target_s2:
        console.print(
            f"  Stage 2 (Fingerprint): {_rank_label(rank_s2)}  "
            f"score={target_s2[0]:.4f}  fp={target_s2[2]:.4f}"
        )
    else:
        console.print(
            f"  Stage 2 (Fingerprint): [red]NF[/red]  "
            f"(cut off at Stage 1, cutoff={config.VECTOR_TOP_N_TRACKS})"
        )
    console.print()
