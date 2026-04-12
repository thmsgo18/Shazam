"""
scripts/find_flowers.py

Pipeline complet sur un fichier audio, avec affichage de la position
de Flowers (Miley Cyrus) à chaque étape : Stage 1 (FAISS) et Stage 2 (Fingerprint).

Usage :
    python scripts/find_flowers.py
    python scripts/find_flowers.py --audio data/raw/93-Rue-Belliard.mp3
    python scripts/find_flowers.py --audio data/raw/93-Rue-Belliard.mp3 --top 30
"""

from __future__ import annotations

import os
import pickle
import sqlite3
import sys
import warnings
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
warnings.filterwarnings("ignore", message=".*upsample_bicubic2d.*", category=UserWarning)

sys.path.insert(0, ".")

import click
import librosa
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

import src.config as config
from src.audio.loading import load_audio
from src.audio.preprocessing import iter_segments
from src.features.embeddings_audio import embed_segment
from src.features.fingerprint import extract_fingerprint, fingerprint_similarity
# NE PAS importer searcher ici : searcher.py importe faiss au niveau module,
# ce qui initialise Apple Accelerate et bloque ensuite PyTorch/MPS.
# L'import est fait dans main() APRÈS le chargement du modèle CLAP.

console = Console()

FLOWERS_ID  = "f01ab00f1fdc5a57fd2676f4d68631a8"
AUDIO_DEFAULT = "data/raw/93-Rue-Belliard.mp3"


def _load_metadata() -> dict[str, dict]:
    path = Path(config.METADATA_PATH)
    if not path.exists():
        return {}
    df = pd.read_parquet(path, columns=["track_id", "title", "artist"])
    return {row.track_id: {"title": row.title, "artist": row.artist} for row in df.itertuples()}


def _get_fp(track_id: str) -> set | None:
    fp_db = Path(config.FINGERPRINTS_DB)
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


@click.command()
@click.option("--audio",  default=AUDIO_DEFAULT, show_default=True, help="Fichier audio à tester")
@click.option("--top",    default=20, show_default=True, help="Nb de résultats affichés dans les tableaux")
@click.option("--method", default=None, help="mfcc / clap / muq (défaut : config.py)")
def main(audio: str, top: int, method: str | None) -> None:
    """Trouve la position de Flowers dans le pipeline, étape par étape."""

    if not Path(audio).exists():
        console.print(f"[red]Fichier introuvable : {audio}[/red]")
        sys.exit(1)

    if method is None:
        method = config.EMBEDDING_METHOD

    if method == "clap":
        targ_sr = config.CLAP_SAMPLE_RATE
    elif method == "muq":
        targ_sr = config.MUQ_SAMPLE_RATE
    else:
        targ_sr = config.SAMPLE_RATE

    metadata = _load_metadata()

    def label(track_id: str) -> str:
        info = metadata.get(track_id, {})
        artist = info.get("artist", track_id)[:20]
        title  = info.get("title",  "—")[:28]
        return f"{artist} — {title}"

    console.print(f"\n[bold cyan]Fichier :[/bold cyan] {audio}")
    console.print(f"[bold cyan]Méthode :[/bold cyan] {method}\n")

    # ── Pré-chargement du modèle AVANT tout import de faiss ─────────────────
    # Sur Mac Apple Silicon, importer faiss initialise Accelerate et bloque MPS.
    # Il faut charger le modèle PyTorch EN PREMIER, puis importer faiss.
    if method == "clap":
        from src.features.embeddings_audio import _load_clap
        console.print(f"[cyan]Chargement du modèle {config.CLAP_MODEL_NAME}...[/cyan]")
        _load_clap(config.CLAP_MODEL_NAME)
        console.print("[green]✓ Modèle prêt.[/green]\n")
    elif method == "muq":
        from src.features.embeddings_audio import _load_muq
        console.print(f"[cyan]Chargement du modèle {config.MUQ_MODEL_NAME}...[/cyan]")
        _load_muq(config.MUQ_MODEL_NAME)
        console.print("[green]✓ Modèle prêt.[/green]\n")

    # Import lazy de searcher — APRÈS le modèle pour éviter le conflit Accelerate
    from src.retrieval.searcher import load_searcher, search_segments, aggregate_by_track

    # ── Stage 1 : FAISS ───────────────────────────────────────────────────────
    console.print("[yellow]Stage 1 — Chargement index + embeddings...[/yellow]")
    index, segments = load_searcher(method)

    waveform, sr = load_audio(audio, target_sr=targ_sr)
    seg_list = [seg for _, seg in iter_segments(waveform=waveform, sr=sr)]
    console.print(f"  {len(seg_list)} segments | index : {index.ntotal} vecteurs\n")

    global_scores: dict[str, float] = {}
    for seg in seg_list:
        emb = embed_segment(seg, sr, method=method,
                            clap_model_name=config.CLAP_MODEL_NAME,
                            muq_model_name=config.MUQ_MODEL_NAME)
        dists, idxs = search_segments(index=index, query_embedding=emb, k=config.VECTOR_TOP_K_SEGMENTS)
        for tid, score in aggregate_by_track(idxs, dists, segments):
            global_scores[tid] = global_scores.get(tid, 0.0) + score

    ranked_s1 = sorted(global_scores.items(), key=lambda x: x[1], reverse=True)

    # Trouver la position de Flowers en Stage 1 (sur TOUS les tracks, pas que top-20)
    rank_s1 = next((i + 1 for i, (tid, _) in enumerate(ranked_s1) if tid == FLOWERS_ID), None)
    score_s1 = global_scores.get(FLOWERS_ID, 0.0)

    # ── Affichage Stage 1 ─────────────────────────────────────────────────────
    console.rule("[bold]Stage 1 — FAISS (embedding)")
    console.print(f"  Position de Flowers : {_rank_label(rank_s1)}  |  score FAISS = [cyan]{score_s1:.4f}[/cyan]")
    if rank_s1 and rank_s1 > 1:
        top1_tid, top1_score = ranked_s1[0]
        console.print(f"  Top-1               : {label(top1_tid)}  score = [cyan]{top1_score:.4f}[/cyan]")
        console.print(f"  Écart Flowers/Top-1 : [red]{top1_score / score_s1:.1f}×[/red] plus haut\n")
    else:
        console.print()

    t1 = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
    t1.add_column("#",       width=4,  style="dim")
    t1.add_column("Artiste — Titre",   width=52)
    t1.add_column("Score FAISS",       justify="right", width=12)

    displayed = 0
    flowers_shown = rank_s1 is None or rank_s1 <= top
    for rank, (tid, score) in enumerate(ranked_s1, 1):
        is_flowers = tid == FLOWERS_ID
        if rank <= top or is_flowers:
            style = "bold green" if is_flowers else ("dim" if rank > top else "")
            marker = " ← 🌸 FLOWERS" if is_flowers else ""
            t1.add_row(str(rank), label(tid) + marker, f"{score:.4f}", style=style)
            displayed += 1
        if rank > top and flowers_shown:
            break

    console.print(t1)

    # ── Stage 2 : Fingerprinting ──────────────────────────────────────────────
    console.rule("[bold]Stage 2 — Fingerprint (re-ranking)")

    # Candidats envoyés en Stage 2 (top-N tracks du Stage 1)
    candidates = ranked_s1[:config.VECTOR_TOP_N_TRACKS]
    flowers_in_candidates = any(tid == FLOWERS_ID for tid, _ in candidates)

    if not flowers_in_candidates:
        console.print(f"  [red]⚠ Flowers n'est pas dans les {config.VECTOR_TOP_N_TRACKS} candidats envoyés en Stage 2.[/red]")
        console.print(f"  [dim](rang FAISS = #{rank_s1} > cutoff = {config.VECTOR_TOP_N_TRACKS})[/dim]\n")
    else:
        console.print(f"  [green]✓ Flowers est dans les {config.VECTOR_TOP_N_TRACKS} candidats envoyés en Stage 2.[/green]\n")

    # Fingerprint de la requête
    if targ_sr != config.SAMPLE_RATE:
        wf_fp = librosa.resample(waveform, orig_sr=targ_sr, target_sr=config.SAMPLE_RATE)
    else:
        wf_fp = waveform
    query_fp = extract_fingerprint(wf_fp, config.SAMPLE_RATE)
    console.print(f"  {len(query_fp)} hashes extraits de la requête\n")

    # Re-ranking
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

    rank_s2 = next((i + 1 for i, (tid, *_) in enumerate(final) if tid == FLOWERS_ID), None)
    flowers_s2 = next(((sf, sf_faiss, fp) for tid, sf, sf_faiss, fp in final if tid == FLOWERS_ID), None)

    console.print(f"  Position de Flowers : {_rank_label(rank_s2)}", end="")
    if flowers_s2:
        console.print(f"  |  score final = [cyan]{flowers_s2[0]:.4f}[/cyan]  "
                      f"(faiss={flowers_s2[1]:.4f}  fp={flowers_s2[2]:.4f})")
    elif not flowers_in_candidates:
        console.print("  [dim](absent des candidats)[/dim]")
    else:
        console.print()

    if rank_s2 and rank_s2 > 1 and final:
        top1_tid, top1_sf, *_ = final[0]
        console.print(f"  Top-1 final         : {label(top1_tid)}  score = [cyan]{top1_sf:.4f}[/cyan]\n")
    else:
        console.print()

    t2 = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
    t2.add_column("#",            width=4,  style="dim")
    t2.add_column("Artiste — Titre",        width=48)
    t2.add_column("Score final",  justify="right", width=12)
    t2.add_column("Score FAISS",  justify="right", width=12)
    t2.add_column("Score FP",     justify="right", width=10)

    for rank, (tid, sf, sf_faiss, fp_score) in enumerate(final[:top], 1):
        is_flowers = tid == FLOWERS_ID
        style  = "bold green" if is_flowers else ""
        marker = " ← 🌸" if is_flowers else ""
        t2.add_row(str(rank), label(tid) + marker,
                   f"{sf:.4f}", f"{sf_faiss:.4f}", f"{fp_score:.4f}", style=style)

    console.print(t2)

    # ── Résumé ────────────────────────────────────────────────────────────────
    console.rule("[bold]Résumé")
    console.print(f"  Stage 1 (FAISS)      : {_rank_label(rank_s1)}  score={score_s1:.4f}")
    if flowers_in_candidates:
        console.print(f"  Stage 2 (Fingerprint): {_rank_label(rank_s2)}  score={flowers_s2[0]:.4f}  fp={flowers_s2[2]:.4f}")
    else:
        console.print(f"  Stage 2 (Fingerprint): [red]NF[/red]  (coupé au Stage 1, cutoff={config.VECTOR_TOP_N_TRACKS})")
    console.print()


if __name__ == "__main__":
    main()
