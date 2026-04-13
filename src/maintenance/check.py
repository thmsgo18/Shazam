"""
src/maintenance/check.py

Vérification de la cohérence des données (ChromaDB, FAISS, fingerprints, metadata).

Checks critiques :
  [C1] Dimension embeddings cohérente
  [C2] NaN / Inf dans les embeddings
  [C3] ChromaDB ↔ FAISS order parquet (même nombre de segments)
  [C5] FAISS index ↔ ChromaDB (même nombre de vecteurs)
  [C6] Segments orphelins (dans ChromaDB mais absents de metadata.parquet)
  [C7] Tracks avec embedding incomplet (< 80% segments attendus)

Checks qualité :
  [Q1] Duration aberrante (≤ 0s ou > 10min)
  [Q2] start_s des segments > duration du track
  [Q3] Fingerprint complètement vide
  [Q4] Fingerprint outlier vers le bas (IQR par tranche de durée)
  [FP] Tracks sans fingerprint

Point d'entrée public : run_check(method, details, metadata, purge, purge_missing_fp, yes)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src import config
from src.utils.fingerprints_db import fp_delete, fp_load_stats
from src.utils.metadata import atomic_write_parquet

ROOT          = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
ITUNES_FIELDS = ["album", "genre", "release_date", "cover_url"]

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_dur(seconds: float) -> str:
    """Formate des secondes en Xm Ys."""
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def _pct(num: int, den: int) -> str:
    if den == 0:
        return "—"
    return f"{num / den:.0%}"


def _fp_load_stats(db_path: Path) -> dict[str, int]:
    return fp_load_stats(db_path)


def _fp_count(db_path: Path) -> int:
    import sqlite3
    if not db_path.exists():
        return 0
    with sqlite3.connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM fingerprints").fetchone()[0]


def _chroma_get_all(collection, include: list[str]) -> dict:
    """
    Récupère tous les documents d'une collection ChromaDB par pages de 500.
    Évite l'erreur SQLite 'too many SQL variables' sur les grandes collections.
    """
    PAGE   = 500
    offset = 0
    result: dict = {k: [] for k in (["ids"] + include)}
    while True:
        page = collection.get(include=include, limit=PAGE, offset=offset)
        if not page["ids"]:
            break
        result["ids"].extend(page["ids"])
        for key in include:
            if page.get(key) is not None:
                result[key].extend(page[key])
        if len(page["ids"]) < PAGE:
            break
        offset += PAGE
    return result


def _get_chroma_collection(method: str) -> tuple[chromadb.Collection | None, str]:
    """Retourne (collection, "") si OK, ou (None, message_erreur) si introuvable."""
    try:
        client     = chromadb.PersistentClient(path=str(ROOT / config.CHROMA_DIR))
        collection = client.get_collection(name=method)
        return collection, ""
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Warning dataclass
# ---------------------------------------------------------------------------

@dataclass
class Warning:
    level:    str                          # "CRITIQUE" | "QUALITE"
    code:     str                          # "C1", "Q1", "FP", etc.
    label:    str                          # Description courte
    method:   str
    track_id: str | None = None
    artist:   str | None = None
    title:    str | None = None
    metrics:  dict = field(default_factory=dict)
    action:   str  = ""


# ---------------------------------------------------------------------------
# check_method — checks complets (pour --details et --purge)
# ---------------------------------------------------------------------------

def check_method(method: str) -> list[Warning]:
    """Retourne la liste des avertissements détectés pour une méthode d'embedding."""
    warns: list[Warning] = []

    fp_db      = ROOT / config.FINGERPRINTS_DB
    meta_path  = PROCESSED_DIR / "metadata.parquet"
    index_path = ROOT / config.INDEX_DIR / f"index_{method}_{config.INDEX_TYPE}.faiss"
    order_path = ROOT / config.INDEX_DIR / f"segments_{method}.parquet"

    # --- ChromaDB : collection obligatoire ---
    collection, err = _get_chroma_collection(method)
    if collection is None:
        warns.append(Warning(
            level="CRITIQUE", code="MANQUANT", label="Collection ChromaDB manquante",
            method=method,
            metrics={"Méthode": method, "Erreur": err},
            action=f"Lancer download_music.py (EMBEDDING_METHOD={method} dans config.py)",
        ))
        return warns

    n_segments = collection.count()
    if n_segments == 0:
        warns.append(Warning(
            level="CRITIQUE", code="VIDE", label="Collection ChromaDB vide",
            method=method,
            action="Lancer download_music.py pour générer les embeddings",
        ))
        return warns

    # Charger tous les embeddings + métadonnées (une seule fois pour tous les checks)
    all_data = _chroma_get_all(collection, include=["embeddings", "metadatas"])
    emb      = np.array(all_data["embeddings"], dtype=np.float32)

    # --- [C1] Dimension des embeddings ---
    if emb.ndim != 2:
        warns.append(Warning(
            level="CRITIQUE", code="C1", label="Embeddings malformés (pas 2D)",
            method=method,
            metrics={"Shape observée": str(emb.shape)},
            action="Re-générer les embeddings",
        ))
        return warns

    expected_dims = {
        "mfcc": 2 * config.N_MFCC,
        "clap": 512,
        "muq":  768,
    }
    if method in expected_dims and emb.shape[1] != expected_dims[method]:
        warns.append(Warning(
            level="CRITIQUE", code="C1", label="Dimension embedding inattendue",
            method=method,
            metrics={
                "Dimension observée": str(emb.shape[1]),
                "Dimension attendue": str(expected_dims[method]),
                **({"Config N_MFCC actuel": str(config.N_MFCC)} if method == "mfcc" else {}),
            },
            action="Vérifier config.py — si N_MFCC a changé, supprimer la collection et relancer",
        ))

    # --- [C2] NaN / Inf ---
    if not np.isfinite(emb).all():
        n_bad = (~np.isfinite(emb)).any(axis=1).sum()
        warns.append(Warning(
            level="CRITIQUE", code="C2", label="NaN ou Inf dans les embeddings",
            method=method,
            metrics={"Vecteurs corrompus": f"{n_bad} / {len(emb)}"},
            action="Identifier les tracks concernés et les re-télécharger",
        ))

    # --- [C3] ChromaDB ↔ order parquet (si l'index a été construit) ---
    if order_path.exists():
        df_order = pd.read_parquet(order_path)
        if len(df_order) != n_segments:
            warns.append(Warning(
                level="CRITIQUE", code="C3",
                label="Order parquet désynchronisé avec ChromaDB",
                method=method,
                metrics={
                    "Segments dans ChromaDB":      str(n_segments),
                    "Segments dans order parquet": str(len(df_order)),
                    "Différence":                  str(abs(n_segments - len(df_order))),
                },
                action="Relancer build_index.py pour reconstruire l'index et l'order parquet",
            ))

    # --- [C5] FAISS index ↔ ChromaDB ---
    if index_path.exists():
        try:
            import faiss
            index = faiss.read_index(str(index_path))
            if index.ntotal != n_segments:
                warns.append(Warning(
                    level="CRITIQUE", code="C5", label="FAISS index désynchronisé",
                    method=method,
                    metrics={
                        "Vecteurs dans FAISS":    str(index.ntotal),
                        "Vecteurs dans ChromaDB": str(n_segments),
                        "Différence":             str(abs(index.ntotal - n_segments)),
                    },
                    action="Relancer build_index.py",
                ))
        except Exception as e:
            warns.append(Warning(
                level="CRITIQUE", code="C5", label="FAISS index illisible",
                method=method,
                metrics={"Erreur": str(e)},
                action="Supprimer l'index et relancer build_index.py",
            ))
    else:
        warns.append(Warning(
            level="CRITIQUE", code="C5", label="FAISS index manquant",
            method=method,
            action="Relancer build_index.py",
        ))

    # --- Métadonnées ---
    if not meta_path.exists():
        warns.append(Warning(
            level="CRITIQUE", code="META", label="metadata.parquet manquant",
            method=method,
            action="Lancer download_music.py",
        ))
        return warns

    try:
        df_meta = pd.read_parquet(meta_path)
    except Exception as e:
        warns.append(Warning(
            level="CRITIQUE", code="META", label="metadata.parquet illisible",
            method=method,
            metrics={"Erreur": str(e)},
            action="Vérifier data/processed/metadata.parquet",
        ))
        return warns

    # Construire le mapping track_id → nombre de segments depuis ChromaDB
    chroma_track_seg_counts: dict[str, int] = {}
    for m in all_data["metadatas"]:
        tid = m["track_id"]
        chroma_track_seg_counts[tid] = chroma_track_seg_counts.get(tid, 0) + 1

    chroma_track_ids = set(chroma_track_seg_counts.keys())
    meta_track_ids   = set(df_meta["track_id"].unique())

    # --- [C6] Segments orphelins (dans ChromaDB mais absents de metadata) ---
    orphans = chroma_track_ids - meta_track_ids
    for oid in list(orphans)[:5]:
        warns.append(Warning(
            level="CRITIQUE", code="C6", label="Segments orphelins (sans metadata)",
            method=method,
            track_id=oid,
            metrics={"track_id": oid[:16] + "..."},
            action="Supprimer les segments de ce track et relancer download_music.py",
        ))
    if len(orphans) > 5:
        warns.append(Warning(
            level="CRITIQUE", code="C6",
            label=f"… et {len(orphans) - 5} autre(s) track(s) orphelins",
            method=method,
        ))

    # --- [C7] Embedding incomplet (< 80% des segments attendus) ---
    if "duration" in df_meta.columns:
        win_s = config.SEGMENT_WIN_S
        hop_s = config.SEGMENT_HOP_S
        for tid, actual in chroma_track_seg_counts.items():
            row = df_meta[df_meta["track_id"] == tid]
            if row.empty or "duration" not in row.columns:
                continue
            duration = float(row["duration"].values[0])
            expected = int(max(0, (duration - win_s) / hop_s)) + 1
            if expected > 0 and actual / expected < 0.8:
                warns.append(Warning(
                    level="CRITIQUE", code="C7", label="Embedding incomplet",
                    method=method,
                    track_id=tid,
                    artist=str(row["artist"].values[0]) if "artist" in row else None,
                    title=str(row["title"].values[0])   if "title"  in row else None,
                    metrics={
                        "Segments réels / attendus": f"{actual} / {expected}  ({actual/expected:.0%})",
                        "Durée du track":            _fmt_dur(duration),
                    },
                    action="Relancer download_music.py (le track sera re-traité automatiquement)",
                ))

    # --- [Q1] Durée aberrante ---
    if "duration" in df_meta.columns:
        bad_dur = df_meta[(df_meta["duration"] <= 0) | (df_meta["duration"] > 600)]
        for row in bad_dur.itertuples():
            n_segs = chroma_track_seg_counts.get(row.track_id, 0)
            warns.append(Warning(
                level="QUALITE", code="Q1", label="Durée aberrante",
                method=method,
                track_id=row.track_id,
                artist=str(getattr(row, "artist", "?")),
                title=str(getattr(row, "title", "?")),
                metrics={
                    "Durée":            f"{row.duration:.0f}s  ({_fmt_dur(row.duration)})",
                    "Seuil normal":     "entre 1s et 10min (600s)",
                    "Segments générés": str(n_segs),
                },
                action=(
                    "Supprimer le track (--purge) et le re-télécharger"
                    if row.duration > 600 else
                    "Durée nulle ou négative — fichier audio probablement corrompu"
                ),
            ))

    # --- [Q2] start_s > durée du track ---
    # Tolérance d'une fenêtre de segment (SEGMENT_WIN_S) pour les écarts yt-dlp / librosa
    Q2_TOLERANCE_S = config.SEGMENT_WIN_S
    if "duration" in df_meta.columns:
        for m_meta in all_data["metadatas"]:
            tid     = m_meta["track_id"]
            start_s = m_meta["start_s"]
            row     = df_meta[df_meta["track_id"] == tid]
            if row.empty:
                continue
            duration = float(row["duration"].values[0])
            if start_s > duration + Q2_TOLERANCE_S:
                warns.append(Warning(
                    level="QUALITE", code="Q2", label="Segment hors durée du track",
                    method=method,
                    track_id=tid,
                    artist=str(row["artist"].values[0]) if "artist" in row else None,
                    title=str(row["title"].values[0])   if "title"  in row else None,
                    metrics={
                        "start_s":        f"{start_s:.1f}s",
                        "Durée du track": f"{duration:.1f}s",
                        "Écart":          f"{start_s - duration:.1f}s  (tolérance : {Q2_TOLERANCE_S}s)",
                    },
                    action="Supprimer le track (--purge) et relancer download_music.py",
                ))
                break  # un seul warning par track

    # --- Tracks marqués traités mais sans segments dans ChromaDB ---
    if "embedded_methods" in df_meta.columns:
        should_have = set(
            df_meta[df_meta["embedded_methods"].apply(
                lambda x: hasattr(x, "__iter__") and not isinstance(x, str) and method in x
            )]["track_id"]
        )
        missing_segs = should_have - chroma_track_ids
        for tid in list(missing_segs)[:5]:
            row = df_meta[df_meta["track_id"] == tid]
            warns.append(Warning(
                level="CRITIQUE", code="C6b",
                label="Track marqué traité mais sans segments",
                method=method,
                track_id=tid,
                artist=str(row["artist"].values[0]) if not row.empty and "artist" in row else None,
                title=str(row["title"].values[0])   if not row.empty and "title"  in row else None,
                metrics={"track_id": tid[:16] + "..."},
                action="Supprimer via --purge et relancer download_music.py",
            ))

    # --- Fingerprints ---
    fp_stats = _fp_load_stats(fp_db)  # {track_id: n_hashes}

    # [Q3] Fingerprint vide (0 hash)
    for tid in [t for t, n in fp_stats.items() if n == 0][:5]:
        row = df_meta[df_meta["track_id"] == tid]
        warns.append(Warning(
            level="QUALITE", code="Q3", label="Fingerprint vide (0 hash)",
            method=method,
            track_id=tid,
            artist=str(row["artist"].values[0]) if not row.empty and "artist" in row else None,
            title=str(row["title"].values[0])   if not row.empty and "title"  in row else None,
            metrics={"Hashes": "0"},
            action="Re-télécharger si la qualité audio est suspecte (--purge)",
        ))

    # [FP] Tracks sans fingerprint
    missing_fp = chroma_track_ids - set(fp_stats.keys())
    if missing_fp:
        n_total = len(chroma_track_ids)
        warns.append(Warning(
            level="QUALITE", code="FP", label="Fingerprints manquants",
            method=method,
            metrics={
                "Tracks sans fingerprint": (
                    f"{len(missing_fp)} / {n_total}  ({len(missing_fp)/n_total:.0%})"
                ),
            },
            action=(
                "Stage 2 (re-ranking Shazam) inopérant pour ces tracks.\n"
                "Pour recalculer : utiliser --purge-missing-fp puis relancer download_music.py"
            ),
        ))

    # [Q4] Fingerprint outlier (IQR par tranche de durée)
    if "duration" in df_meta.columns:
        rows = []
        for tid, n_hashes in fp_stats.items():
            row = df_meta[df_meta["track_id"] == tid]
            if row.empty or float(row["duration"].values[0]) == 0:
                continue
            rows.append({
                "track_id": tid,
                "artist":   str(row["artist"].values[0]) if "artist" in row else "",
                "title":    str(row["title"].values[0])  if "title"  in row else "",
                "duration": float(row["duration"].values[0]),
                "n_hashes": n_hashes,
            })

        if len(rows) >= 4:
            df_fp = pd.DataFrame(rows)
            df_fp["duration_bin"] = pd.cut(
                df_fp["duration"],
                bins=range(0, int(df_fp["duration"].max()) + 31, 30),
                labels=False,
            )
            for _, group in df_fp.groupby("duration_bin"):
                if len(group) < 4:
                    continue
                q1     = group["n_hashes"].quantile(0.25)
                q3     = group["n_hashes"].quantile(0.75)
                iqr    = q3 - q1
                median = group["n_hashes"].median()
                lower  = q1 - 2.5 * iqr
                for r in group[group["n_hashes"] < lower].itertuples():
                    warns.append(Warning(
                        level="QUALITE", code="Q4",
                        label="Fingerprint anormalement pauvre",
                        method=method,
                        track_id=r.track_id,
                        artist=r.artist,
                        title=r.title,
                        metrics={
                            "Hashes":            str(r.n_hashes),
                            "Médiane du groupe": f"{median:.0f}",
                            "Durée":             _fmt_dur(r.duration),
                        },
                        action="Re-télécharger si la qualité audio est suspecte (--purge)",
                    ))

    return warns


# ---------------------------------------------------------------------------
# Résumé par méthode (léger, sans check_method complet)
# ---------------------------------------------------------------------------

@dataclass
class MethodSummary:
    method:       str
    n_segments:   int
    n_tracks:     int
    n_incomplete: int    # embeddings < 80%
    n_fp:         int    # fingerprints présents
    n_fp_total:   int    # tracks avec embeddings (dénominateur fp)
    n_fp_missing: int    # tracks sans fingerprint
    n_fp_empty:   int    # fingerprints vides (0 hash)
    n_fp_poor:    int    # fingerprints outliers (Q4)
    n_crit:       int    # nb warnings critiques
    n_qual:       int    # nb warnings qualité
    index_ok:     bool


def _method_summary(method: str, df_meta: pd.DataFrame | None) -> MethodSummary:
    """Collecte les stats résumées pour une méthode (rapide)."""
    fp_db      = ROOT / config.FINGERPRINTS_DB
    index_path = ROOT / config.INDEX_DIR / f"index_{method}_{config.INDEX_TYPE}.faiss"

    collection, _ = _get_chroma_collection(method)
    if collection is None:
        return MethodSummary(
            method=method, n_segments=0, n_tracks=0, n_incomplete=0,
            n_fp=0, n_fp_total=0, n_fp_missing=0, n_fp_empty=0, n_fp_poor=0,
            n_crit=1, n_qual=0, index_ok=False,
        )

    n_segments = collection.count()

    # Compter tracks et embeddings incomplets
    n_tracks     = 0
    n_incomplete = 0
    all_data     = None
    if n_segments > 0 and df_meta is not None and "duration" in df_meta.columns:
        all_data = _chroma_get_all(collection, include=["metadatas"])
        track_seg_counts: dict[str, int] = {}
        for m in all_data["metadatas"]:
            tid = m["track_id"]
            track_seg_counts[tid] = track_seg_counts.get(tid, 0) + 1
        n_tracks = len(track_seg_counts)

        win_s = config.SEGMENT_WIN_S
        hop_s = config.SEGMENT_HOP_S
        for tid, actual in track_seg_counts.items():
            row = df_meta[df_meta["track_id"] == tid]
            if row.empty:
                continue
            duration = float(row["duration"].values[0])
            expected = int(max(0, (duration - win_s) / hop_s)) + 1
            if expected > 0 and actual / expected < 0.8:
                n_incomplete += 1
    elif n_segments > 0:
        all_data = _chroma_get_all(collection, include=["metadatas"])
        n_tracks = len(set(m["track_id"] for m in all_data["metadatas"]))

    # Fingerprints
    fp_stats   = _fp_load_stats(fp_db)
    chroma_ids: set[str] = set()
    if n_segments > 0 and all_data is not None:
        chroma_ids = set(m["track_id"] for m in all_data["metadatas"])
    elif n_segments > 0:
        all_data2  = _chroma_get_all(collection, include=["metadatas"])
        chroma_ids = set(m["track_id"] for m in all_data2["metadatas"])

    n_fp_total   = len(chroma_ids)
    n_fp_missing = len(chroma_ids - set(fp_stats.keys()))
    n_fp         = n_fp_total - n_fp_missing
    n_fp_empty   = sum(1 for tid in fp_stats if fp_stats[tid] == 0 and tid in chroma_ids)

    # Q4 — fingerprints pauvres (compte rapide, pas de détail)
    n_fp_poor = 0
    if df_meta is not None and "duration" in df_meta.columns:
        rows = []
        for tid, n_hashes in fp_stats.items():
            if tid not in chroma_ids:
                continue
            row = df_meta[df_meta["track_id"] == tid]
            if row.empty or float(row["duration"].values[0]) == 0:
                continue
            rows.append({"duration": float(row["duration"].values[0]), "n_hashes": n_hashes})
        if len(rows) >= 4:
            df_fp = pd.DataFrame(rows)
            df_fp["bin"] = pd.cut(
                df_fp["duration"],
                bins=range(0, int(df_fp["duration"].max()) + 31, 30),
                labels=False,
            )
            for _, grp in df_fp.groupby("bin"):
                if len(grp) < 4:
                    continue
                q1    = grp["n_hashes"].quantile(0.25)
                q3    = grp["n_hashes"].quantile(0.75)
                lower = q1 - 2.5 * (q3 - q1)
                n_fp_poor += int((grp["n_hashes"] < lower).sum())

    # Vérifier l'index FAISS
    index_ok = False
    if index_path.exists():
        try:
            import faiss
            idx      = faiss.read_index(str(index_path))
            index_ok = (idx.ntotal == n_segments)
        except Exception:
            index_ok = False

    # Compter warnings critiques/qualité (rapide : juste les flags déjà calculés)
    n_crit = int(n_incomplete > 0) + int(not index_ok and n_segments > 0)
    n_qual = int(n_fp_missing > 0) + int(n_fp_empty > 0) + int(n_fp_poor > 0)

    return MethodSummary(
        method=method,
        n_segments=n_segments,
        n_tracks=n_tracks,
        n_incomplete=n_incomplete,
        n_fp=n_fp,
        n_fp_total=n_fp_total,
        n_fp_missing=n_fp_missing,
        n_fp_empty=n_fp_empty,
        n_fp_poor=n_fp_poor,
        n_crit=n_crit,
        n_qual=n_qual,
        index_ok=index_ok,
    )


# ---------------------------------------------------------------------------
# Vue résumé (défaut)
# ---------------------------------------------------------------------------

def _render_summary(methods: list[str]) -> None:
    """Affiche le résumé global en deux blocs : audio/embeddings + métadonnées."""
    meta_path = PROCESSED_DIR / "metadata.parquet"

    df_meta: pd.DataFrame | None = None
    n_total = 0
    if meta_path.exists():
        try:
            df_meta = pd.read_parquet(meta_path)
            n_total = len(df_meta)
        except Exception:
            pass

    # ── BLOC 1 : Audio & Embeddings ─────────────────────────────────────────
    console.print()
    console.rule("[bold cyan]Audio & Embeddings[/bold cyan]")
    console.print()

    if n_total:
        console.print(f"  [dim]Tracks dans metadata.parquet :[/dim]  [bold]{n_total}[/bold]")
        console.print()

    if not methods:
        console.print("  [yellow]Aucune collection ChromaDB détectée.[/yellow]")
    else:
        for method in sorted(methods):
            s = _method_summary(method, df_meta)

            if s.n_crit > 0:
                status_color = "red"
                status_icon  = "✗"
            elif s.n_qual > 0:
                status_color = "yellow"
                status_icon  = "⚠"
            else:
                status_color = "green"
                status_icon  = "✓"

            console.print(
                f"  [{status_color}]{status_icon}[/{status_color}]  "
                f"[bold]{method.upper()}[/bold]"
            )

            tbl = Table(box=None, show_header=False, padding=(0, 2))
            tbl.add_column("label", style="dim", no_wrap=True)
            tbl.add_column("value")

            tbl.add_row("Segments",    str(s.n_segments))
            tbl.add_row(
                "Tracks couverts",
                f"{s.n_tracks} / {n_total}  ({_pct(s.n_tracks, n_total)})"
                if n_total else str(s.n_tracks),
            )

            if s.n_incomplete == 0:
                tbl.add_row("Embeddings",  "[green]complets[/green]")
            else:
                tbl.add_row(
                    "Embeddings",
                    f"[red]{s.n_incomplete} incomplet(s)[/red]  "
                    f"[dim](< 80 % segments attendus)[/dim]",
                )

            fp_label  = f"{s.n_fp} / {s.n_fp_total}  ({_pct(s.n_fp, s.n_fp_total)})"
            fp_issues = []
            if s.n_fp_missing > 0:
                fp_issues.append(f"[red]{s.n_fp_missing} manquant(s)[/red]")
            if s.n_fp_empty > 0:
                fp_issues.append(f"[yellow]{s.n_fp_empty} vide(s)[/yellow]")
            if s.n_fp_poor > 0:
                fp_issues.append(f"[yellow]{s.n_fp_poor} pauvre(s)[/yellow]")
            fp_str = fp_label
            if fp_issues:
                fp_str += "  " + "  ".join(fp_issues)
            tbl.add_row("Fingerprints", fp_str)

            if s.n_segments > 0:
                tbl.add_row(
                    "Index FAISS",
                    "[green]OK[/green]" if s.index_ok else "[red]manquant ou désynchronisé[/red]",
                )

            console.print(tbl)

            if s.n_crit > 0 or s.n_qual > 0:
                console.print(
                    "    [dim]→ [bold]--details[/bold] pour voir les problèmes détaillés[/dim]"
                )
            console.print()

    # ── BLOC 2 : Métadonnées ─────────────────────────────────────────────────
    console.rule("[bold cyan]Complétude des métadonnées[/bold cyan]")
    console.print()

    if df_meta is None or n_total == 0:
        console.print("  [yellow]metadata.parquet introuvable ou vide.[/yellow]")
        console.print()
        return

    for col in ITUNES_FIELDS:
        if col not in df_meta.columns:
            df_meta[col] = None

    tbl2 = Table(box=None, show_header=False, padding=(0, 2))
    tbl2.add_column("champ",  style="dim", no_wrap=True)
    tbl2.add_column("compte")
    tbl2.add_column("barre")

    BAR_WIDTH = 20
    for col in ITUNES_FIELDS:
        n_filled = int(df_meta[col].notna().sum())
        ratio    = n_filled / n_total if n_total else 0
        n_bar    = round(ratio * BAR_WIDTH)
        bar      = "[green]" + "█" * n_bar + "[/green]" + "[dim]" + "░" * (BAR_WIDTH - n_bar) + "[/dim]"
        pct_str  = f"{ratio:.0%}"
        color    = "green" if ratio >= 0.95 else ("yellow" if ratio >= 0.80 else "red")
        tbl2.add_row(
            col,
            f"[{color}]{n_filled} / {n_total}[/{color}]",
            f"{bar}  {pct_str}",
        )

    console.print(tbl2)
    console.print()

    mask_all_none = df_meta[ITUNES_FIELDS].isnull().all(axis=1)
    mask_any_none = df_meta[ITUNES_FIELDS].isnull().any(axis=1) & ~mask_all_none
    mask_complete = ~df_meta[ITUNES_FIELDS].isnull().any(axis=1)

    n_complete  = int(mask_complete.sum())
    n_partial   = int(mask_any_none.sum())
    n_not_found = int(mask_all_none.sum())

    tbl3 = Table(box=None, show_header=False, padding=(0, 2))
    tbl3.add_column("label", style="dim")
    tbl3.add_column("value")

    tbl3.add_row(
        "Complets (tous champs)",
        f"[green]{n_complete} / {n_total}[/green]  ({_pct(n_complete, n_total)})",
    )
    tbl3.add_row(
        "Partiels (≥ 1 champ vide)",
        (f"[yellow]{n_partial} / {n_total}[/yellow]  ({_pct(n_partial, n_total)})"
         if n_partial else f"[green]0[/green]"),
    )
    tbl3.add_row(
        "Non trouvés (aucun champ)",
        (f"[red]{n_not_found} / {n_total}[/red]  ({_pct(n_not_found, n_total)})"
         if n_not_found else f"[green]0[/green]"),
    )

    console.print(tbl3)
    console.print()

    if n_partial > 0 or n_not_found > 0:
        console.print(
            "  [dim]→ [bold]--metadata[/bold] pour voir les tracks avec métadonnées manquantes[/dim]"
        )
        console.print()


# ---------------------------------------------------------------------------
# Vue --metadata
# ---------------------------------------------------------------------------

def _render_metadata_report() -> None:
    """Affiche les tracks avec métadonnées manquantes ou partielles."""
    meta_path = PROCESSED_DIR / "metadata.parquet"

    if not meta_path.exists():
        console.print("[red]metadata.parquet introuvable.[/red]")
        return

    try:
        df = pd.read_parquet(meta_path)
    except Exception as e:
        console.print(f"[red]Impossible de lire metadata.parquet : {e}[/red]")
        return

    for col in ITUNES_FIELDS:
        if col not in df.columns:
            df[col] = None

    n_total = len(df)

    # ── Tracks non trouvés (tous les champs None) ────────────────────────────
    console.print()
    df_none = df[df[ITUNES_FIELDS].isnull().all(axis=1)].copy()
    console.rule(
        f"[bold red]Non trouvés sur Deezer / MusicBrainz[/bold red]  "
        f"[dim]{len(df_none)} / {n_total}[/dim]"
    )
    console.print()

    if df_none.empty:
        console.print("  [green]Aucun — tous les tracks ont au moins un champ renseigné.[/green]")
    else:
        tbl = Table(show_header=True, header_style="bold red")
        tbl.add_column("#",       style="dim",  width=4)
        tbl.add_column("Artiste", style="bold")
        tbl.add_column("Titre")

        for i, row in enumerate(df_none.itertuples(), start=1):
            tbl.add_row(
                str(i),
                str(getattr(row, "artist", "—")),
                str(getattr(row, "title",  "—")),
            )
        console.print(tbl)

    console.print()

    # ── Tracks partiels (≥ 1 champ vide, mais pas tous) ──────────────────────
    mask_partial = df[ITUNES_FIELDS].isnull().any(axis=1) & ~df[ITUNES_FIELDS].isnull().all(axis=1)
    df_partial   = df[mask_partial].copy()

    console.rule(
        f"[bold yellow]Métadonnées partielles[/bold yellow]  "
        f"[dim]{len(df_partial)} / {n_total}[/dim]"
    )
    console.print()

    if df_partial.empty:
        console.print("  [green]Aucun — tous les tracks ont leurs champs complets.[/green]")
    else:
        tbl2 = Table(show_header=True, header_style="bold yellow")
        tbl2.add_column("#",            style="dim",    width=4)
        tbl2.add_column("Artiste",      style="bold")
        tbl2.add_column("Titre")
        tbl2.add_column("album",        justify="center")
        tbl2.add_column("genre",        justify="center")
        tbl2.add_column("release_date", justify="center")
        tbl2.add_column("cover_url",    justify="center")

        CHECK = "[green]✓[/green]"
        CROSS = "[red]✗[/red]"

        for i, row in enumerate(df_partial.itertuples(), start=1):
            tbl2.add_row(
                str(i),
                str(getattr(row, "artist", "—")),
                str(getattr(row, "title",  "—")),
                CHECK if pd.notna(getattr(row, "album",        None)) else CROSS,
                CHECK if pd.notna(getattr(row, "genre",        None)) else CROSS,
                CHECK if pd.notna(getattr(row, "release_date", None)) else CROSS,
                CHECK if pd.notna(getattr(row, "cover_url",    None)) else CROSS,
            )
        console.print(tbl2)

    console.print()

    if not df_none.empty or not df_partial.empty:
        console.print(
            "  [dim]→ Relancer [bold]python manage.py enrich[/bold] "
            "pour tenter d'enrichir les champs manquants.[/dim]"
        )
        console.print()


# ---------------------------------------------------------------------------
# Rendu d'un warning (pour --details)
# ---------------------------------------------------------------------------

def _render_warning(w: Warning) -> None:
    """Affiche un warning formaté avec rich.Panel."""
    LEVEL_COLOR = {"CRITIQUE": "red", "QUALITE": "yellow"}
    LEVEL_ICON  = {"CRITIQUE": "✗", "QUALITE": "⚠"}

    color = LEVEL_COLOR.get(w.level, "white")
    icon  = LEVEL_ICON.get(w.level, "•")

    lines: list[str] = []

    if w.artist and w.title:
        lines.append(f"  [bold]{w.artist}[/bold] — [bold]{w.title}[/bold]")
    elif w.title:
        lines.append(f"  [bold]{w.title}[/bold]")
    elif w.track_id:
        lines.append(f"  track_id : {w.track_id[:16]}...")

    if w.metrics:
        if lines:
            lines.append("")
        for k, v in w.metrics.items():
            lines.append(f"  [dim]{k:<30}[/dim] {v}")

    if w.action:
        lines.append("")
        first = True
        for line in w.action.split("\n"):
            prefix = f"  [bold {color}]→[/bold {color}] " if first else "    "
            lines.append(f"{prefix}{line}")
            first = False

    title   = f"[bold {color}]{icon} {w.level} · {w.code} · {w.label}[/bold {color}]  [dim cyan]({w.method})[/dim cyan]"
    content = "\n".join(lines) if lines else "  (pas de détails)"

    console.print(Panel(
        content,
        title=title,
        title_align="left",
        border_style=color,
        padding=(0, 1),
    ))
    console.print("")


# ---------------------------------------------------------------------------
# Vue --details
# ---------------------------------------------------------------------------

def _render_details(methods: list[str]) -> list[Warning]:
    """Affiche les avertissements détaillés par méthode."""
    all_warns: list[Warning] = []

    for m in sorted(methods):
        collection, _ = _get_chroma_collection(m)
        n_seg    = collection.count() if collection else "—"
        n_tracks = "—"
        if collection and n_seg != "—" and n_seg > 0:
            meta_data = _chroma_get_all(collection, include=["metadatas"])
            n_tracks  = len(set(md["track_id"] for md in meta_data["metadatas"]))

        n_fp = _fp_count(ROOT / config.FINGERPRINTS_DB)

        method_warns = check_method(m)
        n_crit = sum(1 for w in method_warns if w.level == "CRITIQUE")
        n_qual = sum(1 for w in method_warns if w.level == "QUALITE")

        if not method_warns:
            status = "[green]✓ tout OK[/green]"
        elif n_crit:
            status = (
                f"[red]{n_crit} critique(s)[/red]"
                + (f"  [yellow]{n_qual} qualité[/yellow]" if n_qual else "")
            )
        else:
            status = f"[yellow]{n_qual} alerte(s) qualité[/yellow]"

        console.rule(
            f"[bold cyan]{m.upper()}[/bold cyan]  "
            f"[dim]{n_seg} segs · {n_tracks} tracks · {n_fp} fps[/dim]  "
            + status
        )
        console.print("")

        if method_warns:
            for w in method_warns:
                _render_warning(w)
        else:
            console.print("  [green]Aucun problème détecté.[/green]\n")

        all_warns.extend(method_warns)

    # Résumé final
    console.rule()
    n_crit = sum(1 for w in all_warns if w.level == "CRITIQUE")
    n_qual = sum(1 for w in all_warns if w.level == "QUALITE")

    if not all_warns:
        console.print("[bold green]✓ Toutes les données sont cohérentes.[/bold green]")
    elif n_crit:
        console.print(
            f"[bold red]{n_crit} problème(s) critique(s)[/bold red]"
            + (f"  [yellow]{n_qual} alerte(s) qualité[/yellow]" if n_qual else "")
        )
    else:
        console.print(
            f"[yellow]{n_qual} alerte(s) qualité[/yellow] — données utilisables, "
            "mais certains tracks peuvent donner de mauvais résultats."
        )

    console.print()
    return all_warns


# ---------------------------------------------------------------------------
# Purge
# ---------------------------------------------------------------------------

def purge_tracks(method: str, track_ids: set[str]) -> dict:
    """
    Supprime les données d'un ensemble de tracks pour une méthode donnée.

    Comportement chirurgical :
    - Segments supprimés de ChromaDB uniquement pour la méthode purgée
    - La méthode est retirée de embedded_methods dans metadata.parquet
    - Si embedded_methods devient vide → la ligne entière est supprimée + fingerprint supprimé
    - Si d'autres méthodes restent → la ligne est conservée (les autres méthodes sont intactes)
    - FAISS index + order parquet de la méthode supprimés (à reconstruire)

    Retourne un dict de stats :
        segments_removed, tracks_updated, tracks_removed, fingerprints_removed.
    """
    fp_db      = ROOT / config.FINGERPRINTS_DB
    meta_path  = PROCESSED_DIR / "metadata.parquet"
    index_path = ROOT / config.INDEX_DIR / f"index_{method}_{config.INDEX_TYPE}.faiss"
    order_path = ROOT / config.INDEX_DIR / f"segments_{method}.parquet"

    stats = {
        "segments_removed":     0,
        "tracks_updated":       0,
        "tracks_removed":       0,
        "fingerprints_removed": 0,
    }

    # --- ChromaDB ---
    collection, _ = _get_chroma_collection(method)
    if collection is not None:
        for tid in track_ids:
            result = collection.get(where={"track_id": {"$eq": tid}})
            if result["ids"]:
                collection.delete(ids=result["ids"])
                stats["segments_removed"] += len(result["ids"])

    # --- FAISS index + order parquet (obsolètes) ---
    for p in [index_path, order_path]:
        if p.exists():
            p.unlink()

    # --- Metadata : retrait chirurgical de la méthode ---
    fully_removed: set[str] = set()

    if meta_path.exists():
        df_meta  = pd.read_parquet(meta_path)
        affected = df_meta["track_id"].isin(track_ids)

        if "embedded_methods" in df_meta.columns:
            def _remove_method(methods):
                if methods is None:
                    return None
                if isinstance(methods, (list, set, np.ndarray)):
                    updated = [m for m in methods if m != method]
                    return updated if updated else None
                return methods

            df_meta.loc[affected, "embedded_methods"] = (
                df_meta.loc[affected, "embedded_methods"].apply(_remove_method)
            )

            mask_empty = affected & df_meta["embedded_methods"].isna()
            fully_removed = set(df_meta.loc[mask_empty, "track_id"])

            stats["tracks_removed"] = int(mask_empty.sum())
            stats["tracks_updated"] = int(affected.sum()) - stats["tracks_removed"]

            df_meta = df_meta[~mask_empty]
        else:
            fully_removed           = track_ids
            stats["tracks_removed"] = int(affected.sum())
            df_meta                 = df_meta[~affected]

        atomic_write_parquet(meta_path, df_meta)

    # Tracks absents de metadata (orphelins C6) → aussi complètement supprimés
    if meta_path.exists():
        remaining_ids = set(pd.read_parquet(meta_path)["track_id"].unique())
    else:
        remaining_ids = set()
    fully_removed |= (track_ids - remaining_ids)

    # --- Fingerprints (SQLite) ---
    if fully_removed:
        stats["fingerprints_removed"] = fp_delete(fp_db, fully_removed)

    return stats


def _run_purge(by_method: dict[str, set[str]], yes: bool) -> None:
    """Affiche le récap, demande confirmation, et purge."""
    console.print("")
    console.rule("[bold red]Récapitulatif de la purge[/bold red]")
    console.print("")

    meta_path = PROCESSED_DIR / "metadata.parquet"
    df_meta   = pd.read_parquet(meta_path) if meta_path.exists() else pd.DataFrame()

    for m, tids in sorted(by_method.items()):
        for tid in sorted(tids):
            row = df_meta[df_meta["track_id"] == tid] if not df_meta.empty else pd.DataFrame()

            will_fully_remove = True
            if not row.empty and "embedded_methods" in df_meta.columns:
                methods = row["embedded_methods"].values[0]
                if isinstance(methods, (list, set, np.ndarray)):
                    remaining = [x for x in methods if x != m]
                    will_fully_remove = len(remaining) == 0

            if not row.empty:
                artist = str(row["artist"].values[0])
                title  = str(row["title"].values[0])
                label  = f"{artist} — {title}"
            else:
                label  = f"track_id {tid[:12]}..."

            if will_fully_remove:
                console.print(
                    f"  [red]✗[/red]  [dim cyan]({m})[/dim cyan]  {label}"
                    f"  [dim red]→ supprimé entièrement[/dim red]"
                )
            else:
                console.print(
                    f"  [yellow]↺[/yellow]  [dim cyan]({m})[/dim cyan]  {label}"
                    f"  [dim yellow]→ méthode retirée, autres méthodes conservées[/dim yellow]"
                )

    total_tracks = sum(len(v) for v in by_method.values())
    console.print("")
    console.print(
        f"[bold]{total_tracks} track(s)[/bold] vont être purgés pour la méthode concernée."
    )
    console.print("[dim]• Segments ChromaDB supprimés pour la méthode purgée uniquement.[/dim]")
    console.print("[dim]• Si le track n'a plus aucune méthode active → ligne metadata + fingerprint supprimés.[/dim]")
    console.print("[dim]• L'index FAISS de la méthode sera supprimé — relancer build_index.py après.[/dim]")

    if not yes:
        console.print("")
        confirm = console.input(
            "[bold yellow]Continuer ?[/bold yellow] [dim](o = confirmer / Entrée = annuler) [/dim]"
        ).strip().lower()
        if confirm not in ("o", "oui", "y", "yes"):
            console.print("[dim]Annulé.[/dim]")
            return

    console.print("")
    total_segs = 0
    for m, tids in sorted(by_method.items()):
        console.print(f"  Purge méthode [cyan]{m}[/cyan]…", end=" ")
        stats = purge_tracks(m, tids)
        total_segs += stats["segments_removed"]
        parts = [f"{stats['segments_removed']} segments supprimés"]
        if stats["tracks_updated"] > 0:
            parts.append(f"{stats['tracks_updated']} track(s) mis à jour (méthode retirée)")
        if stats["tracks_removed"] > 0:
            parts.append(f"{stats['tracks_removed']} track(s) supprimé(s) entièrement")
        if stats["fingerprints_removed"] > 0:
            parts.append(f"{stats['fingerprints_removed']} fingerprint(s) supprimé(s)")
        console.print(f"[green]✓[/green]  " + "  •  ".join(parts))

    console.print("")
    console.print(
        f"[bold green]Purge terminée.[/bold green]  "
        f"{total_tracks} track(s) traités, {total_segs} segments supprimés."
    )
    console.print("\n[dim]Pour re-télécharger et reconstruire l'index :[/dim]")
    console.print("  [bold cyan]python manage.py ingest[/bold cyan]")
    console.print("  [bold cyan]python src/index/build_index.py[/bold cyan]")


def _run_purge_missing_fp(methods: list[str], yes: bool) -> None:
    """Purge les tracks qui ont des embeddings mais pas de fingerprint."""
    fp_db  = ROOT / config.FINGERPRINTS_DB
    fp_ids = _fp_load_stats(fp_db).keys()

    by_method: dict[str, set[str]] = {}
    for m in methods:
        collection, _ = _get_chroma_collection(m)
        if collection is None:
            continue
        data       = _chroma_get_all(collection, include=["metadatas"])
        chroma_ids = set(md["track_id"] for md in data["metadatas"])
        missing    = chroma_ids - set(fp_ids)
        if missing:
            by_method[m] = missing

    if not by_method:
        console.print("[green]Aucun track sans fingerprint — rien à purger.[/green]")
        return

    total = sum(len(v) for v in by_method.values())
    console.print("")
    console.rule("[bold yellow]Purge des tracks sans fingerprint[/bold yellow]")
    console.print(f"\n  [bold]{total} track(s)[/bold] sans fingerprint vont être supprimés.")
    console.print("  [dim]Ils seront re-téléchargés et re-fingerprinted au prochain run.[/dim]\n")

    if not yes:
        confirm = console.input(
            "[bold yellow]Continuer ?[/bold yellow] [dim](o = confirmer / Entrée = annuler) [/dim]"
        ).strip().lower()
        if confirm not in ("o", "oui", "y", "yes"):
            console.print("[dim]Annulé.[/dim]")
            return

    _run_purge(by_method, yes=True)


# ---------------------------------------------------------------------------
# Point d'entrée public
# ---------------------------------------------------------------------------

def run_check(
    method:           str | None = None,
    details:          bool = False,
    metadata:         bool = False,
    purge:            bool = False,
    purge_missing_fp: bool = False,
    yes:              bool = False,
) -> None:
    """
    Vérifie la cohérence des données du projet Shazam Maison.

    Args:
        method:           méthode à vérifier (défaut : toutes détectées).
        details:          affiche le détail de chaque problème.
        metadata:         affiche les tracks avec métadonnées manquantes.
        purge:            supprime les tracks problématiques.
        purge_missing_fp: purge uniquement les tracks sans fingerprint.
        yes:              ne pas demander de confirmation avant de purger.
    """
    # Détecter les méthodes disponibles depuis ChromaDB
    if method:
        methods = [method]
    else:
        try:
            client  = chromadb.PersistentClient(path=str(ROOT / config.CHROMA_DIR))
            methods = [c.name for c in client.list_collections()]
        except Exception:
            methods = []

    # ── Mode --metadata ────────────────────────────────────────────────────
    if metadata:
        _render_metadata_report()
        return

    # ── Mode --details (+ éventuellement --purge) ──────────────────────────
    if details or purge or purge_missing_fp:
        if not methods:
            console.print(
                f"[yellow]Aucune collection ChromaDB trouvée dans {config.CHROMA_DIR}[/yellow]"
            )
            sys.exit(0)

        all_warns = _render_details(methods)

        if purge_missing_fp:
            _run_purge_missing_fp(methods, yes)
            return

        if purge:
            by_method: dict[str, set[str]] = {}
            for w in all_warns:
                if w.track_id:
                    by_method.setdefault(w.method, set()).add(w.track_id)

            if not by_method:
                console.print(
                    "\n[yellow]Aucun track individuel à purger "
                    "(les alertes sont globales — voir les actions suggérées).[/yellow]"
                )
                return

            _run_purge(by_method, yes)
        elif not details:
            console.print(
                "\n[dim]Astuce : relance avec [bold]--purge[/bold] pour supprimer "
                "automatiquement les tracks problématiques.[/dim]"
            )
        return

    # ── Vue résumé (défaut) ────────────────────────────────────────────────
    _render_summary(methods)
