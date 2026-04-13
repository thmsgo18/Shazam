"""
src/evaluation/rir_impact.py

Mesure l'impact des RIR sur le score FAISS SANS supprimer la base.

Stratégie :
  - Charge uniquement les vecteurs originaux (sans _rir_) depuis ChromaDB
  - Construit un index FAISS temporaire en mémoire (pas de fichier)
  - Compare la position du track cible AVEC et SANS RIR dans l'index

Point d'entrée public : run_rir_impact(audio, target_track_id, top, method)
"""

from __future__ import annotations

import sys
from pathlib import Path

import chromadb
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn
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
PAGE          = 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_no_rir_index(collection_key: str):
    """
    Charge uniquement les vecteurs NON-RIR depuis ChromaDB et construit
    un index FAISS en mémoire + DataFrame segments.
    """
    import faiss  # import lazy — APRÈS le chargement du modèle PyTorch

    client     = chromadb.PersistentClient(path=str(ROOT / config.CHROMA_DIR))
    collection = client.get_collection(name=collection_key)
    total      = collection.count()

    embeddings_list: list = []
    metadatas_list:  list = []

    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as prog:
        task   = prog.add_task("Chargement vecteurs originaux…", total=total)
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
            if len(page["ids"]) < PAGE:
                break
            offset += PAGE

    console.print(f"  {len(embeddings_list):,} vecteurs originaux chargés (sur {total:,} total)")

    console.print("  Conversion en array numpy…")
    xb = np.array(embeddings_list, dtype=np.float32)
    del embeddings_list

    console.print(f"  Array shape : {xb.shape} — normalisation L2…")
    norms = np.linalg.norm(xb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    xb   /= norms

    console.print("  Construction index FAISS…")
    faiss.omp_set_num_threads(1)
    index = faiss.IndexFlatIP(xb.shape[1])
    index.add(xb)
    console.print(f"  [green]✓ Index prêt ({index.ntotal:,} vecteurs)[/green]\n")

    segments = pd.DataFrame(metadatas_list)
    return index, segments


def _search(index, segments: pd.DataFrame, query_emb: np.ndarray, k: int) -> dict[str, float]:
    """Recherche FAISS + agrégation par track_id."""
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


def _load_model(method: str) -> None:
    """Charge le modèle AVANT faiss (Apple Silicon)."""
    if method == "clap":
        from src.features.embeddings_audio import _load_clap
        console.print(f"[cyan]Chargement {config.CLAP_MODEL_NAME}…[/cyan]")
        _load_clap(config.CLAP_MODEL_NAME)
        console.print("[green]✓ Modèle prêt.[/green]\n")
    elif method == "muq":
        from src.features.embeddings_audio import _load_muq
        _load_muq(config.MUQ_MODEL_NAME)
    elif method == "mert":
        from src.features.embeddings_audio import _load_mert
        _load_mert(config.MERT_MODEL_NAME)


# ---------------------------------------------------------------------------
# Point d'entrée public
# ---------------------------------------------------------------------------

def run_rir_impact(
    audio:           str = AUDIO_DEFAULT,
    target_track_id: str = FLOWERS_ID,
    top:             int = 20,
    method:          str | None = None,
) -> None:
    """
    Compare la position du track cible AVEC et SANS vecteurs RIR dans l'index.

    Args:
        audio:           chemin vers le fichier audio à tester.
        target_track_id: track_id à suivre (défaut : Flowers).
        top:             nombre de résultats dans les tableaux.
        method:          méthode d'embedding (défaut : config.EMBEDDING_METHOD).
    """
    if not Path(audio).exists():
        console.print(f"[red]Fichier introuvable : {audio}[/red]")
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
        f"[bold]Méthode  :[/bold] [cyan]{method}[/cyan]\n"
        f"[bold]Audio    :[/bold] [cyan]{audio}[/cyan]\n"
        f"[bold]But      :[/bold] comparer FAISS avec vs sans vecteurs RIR",
        title="[bold cyan]Test impact RIR[/bold cyan]",
        expand=False,
    ))

    # Chargement modèle AVANT faiss (Apple Silicon)
    _load_model(method)

    # Embeddings de la requête
    console.print("[yellow]Préparation de la requête audio…[/yellow]")
    waveform, sr = load_audio(audio, target_sr=targ_sr)
    waveform     = preprocess_query(waveform, sr)
    seg_list     = [seg for _, seg in iter_segments(waveform=waveform, sr=sr)]
    console.print(f"  {len(seg_list)} segments à embedder\n")

    query_embeddings = []
    for seg in seg_list:
        emb = embed_segment(
            seg, sr, method=method,
            clap_model_name=config.CLAP_MODEL_NAME,
            muq_model_name=config.MUQ_MODEL_NAME,
            mert_model_name=config.MERT_MODEL_NAME,
        )
        query_embeddings.append(emb)

    # Index SANS RIR (en mémoire)
    console.print("[yellow]Construction de l'index SANS RIR…[/yellow]")
    index_no_rir, segments_no_rir = _load_no_rir_index(collection_key)
    console.print(f"  Index sans RIR : [white]{index_no_rir.ntotal:,}[/white] vecteurs\n")

    # Index AVEC RIR (fichier existant)
    console.print("[yellow]Chargement de l'index AVEC RIR…[/yellow]")
    from src.retrieval.searcher import load_searcher
    index_rir, segments_rir = load_searcher(method)
    console.print(f"  Index avec RIR : [white]{index_rir.ntotal:,}[/white] vecteurs\n")

    # Recherche sur les deux index
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

    # Résultat
    console.rule("[bold]Résultat Stage 1 — FAISS")

    t = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE_HEAD)
    t.add_column("Condition",        width=20)
    t.add_column("Position cible",  width=18)
    t.add_column("Score FAISS",     justify="right", width=12)
    t.add_column("Vecteurs index",  justify="right", width=15)

    t.add_row("Sans RIR", _rank_label(rank_no_rir), f"{score_no_rir:.4f}", f"{index_no_rir.ntotal:,}")
    t.add_row("Avec RIR", _rank_label(rank_rir),    f"{score_rir:.4f}",    f"{index_rir.ntotal:,}")
    console.print(t)

    if rank_rir is not None and rank_no_rir is not None:
        delta = rank_no_rir - rank_rir
        if delta > 0:
            console.print(f"\n[green]✓ Les RIR améliorent de {delta} positions ({rank_no_rir} → {rank_rir})[/green]")
        elif delta < 0:
            console.print(f"\n[red]✗ Les RIR dégradent de {abs(delta)} positions ({rank_no_rir} → {rank_rir})[/red]")
        else:
            console.print(f"\n[yellow]= Les RIR n'ont pas d'impact sur la position ({rank_rir})[/yellow]")
    console.print()

    # ── Données structurées (utiles pour run_rir_evaluate) ──────────────────
    run_rir_impact._last_result = {
        "with_rir":    {"rank": rank_rir,    "faiss_score": round(score_rir,    4), "n_vectors": index_rir.ntotal},
        "without_rir": {"rank": rank_no_rir, "faiss_score": round(score_no_rir, 4), "n_vectors": index_no_rir.ntotal},
    }

    # Top-N des deux index pour comparaison
    meta_path = ROOT / config.METADATA_PATH
    meta_df   = pd.read_parquet(meta_path, columns=["track_id", "title", "artist"])
    meta      = {r.track_id: f"{r.artist[:18]} — {r.title[:25]}" for r in meta_df.itertuples()}

    for label_str, ranked in [("Sans RIR", ranked_no_rir), ("Avec RIR", ranked_rir)]:
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
                f"{style}{meta.get(tid, tid)}{'  ← 🎯 CIBLE' if is_target else ''}",
                f"{score:.4f}",
            )
            if is_target:
                target_shown = True
            displayed += 1
        console.print(t2)


# ---------------------------------------------------------------------------
# API silencieuse pour run_rir_evaluate (pas d'affichage)
# ---------------------------------------------------------------------------

def rir_impact_scores(
    audio_path: str,
    track_id:   str,
    method:     str | None = None,
    prebuilt_no_rir: tuple | None = None,
) -> dict:
    """
    Compare Stage 1 FAISS scores pour un fichier audio : avec vs sans RIR.
    Ne produit aucun affichage — retourne des données structurées.

    Args:
        audio_path:       chemin vers le fichier audio (WAV temporaire accepté).
        track_id:         track_id attendu comme bonne réponse.
        method:           méthode d'embedding (défaut : config.EMBEDDING_METHOD).
        prebuilt_no_rir:  (index, segments) déjà construit — optimisation multi-tracks.

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

    # Chargement du modèle AVANT faiss (Apple Silicon)
    _load_model(method)

    # Embedding des segments de la requête
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

    # Index SANS RIR
    if prebuilt_no_rir is not None:
        index_no_rir, segments_no_rir = prebuilt_no_rir
    else:
        index_no_rir, segments_no_rir = _load_no_rir_index(collection_key)

    # Index AVEC RIR (fichier existant sur disque)
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
