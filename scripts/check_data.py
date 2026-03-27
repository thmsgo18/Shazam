"""
scripts/check_data.py

Vérifie la cohérence des données générées par download_music.py.

Checks critiques :
  [C1] Dimension embeddings cohérente (2*N_MFCC / 512 / 768 selon méthode)
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

Usage :
    python scripts/check_data.py
    python scripts/check_data.py --method clap
    python scripts/check_data.py --purge
    python scripts/check_data.py --purge --yes
"""
from __future__ import annotations

import pickle
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel

from src import config

FEATURES_DIR = Path("data/features")
PROCESSED_DIR = Path("data/processed")


# ---------------------------------------------------------------------------
# Helpers SQLite fingerprints
# ---------------------------------------------------------------------------

def _fp_load_all(db_path: Path) -> dict[str, set]:
    """Charge tous les fingerprints {track_id: set_of_hashes}."""
    if not db_path.exists():
        return {}
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT track_id, hashes FROM fingerprints").fetchall()
    return {r[0]: pickle.loads(r[1]) for r in rows}


def _fp_load_stats(db_path: Path) -> dict[str, int]:
    """Charge {track_id: n_hashes} sans désérialiser les hashes (rapide)."""
    if not db_path.exists():
        return {}
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT track_id, n_hashes FROM fingerprints").fetchall()
    return {r[0]: r[1] for r in rows}


def _fp_count(db_path: Path) -> int:
    if not db_path.exists():
        return 0
    with sqlite3.connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM fingerprints").fetchone()[0]


def _chroma_get_all(collection, include: list[str]) -> dict:
    """
    Récupère tous les documents d'une collection ChromaDB par pages de 500.
    Évite l'erreur SQLite 'too many SQL variables' sur les grandes collections.
    """
    PAGE = 500
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


def _fp_delete(db_path: Path, track_ids: set[str]) -> int:
    """Supprime les fingerprints pour les track_ids donnés. Retourne le nombre supprimé."""
    if not db_path.exists() or not track_ids:
        return 0
    with sqlite3.connect(db_path) as conn:
        placeholders = ",".join("?" * len(track_ids))
        cursor = conn.execute(
            f"DELETE FROM fingerprints WHERE track_id IN ({placeholders})",
            list(track_ids),
        )
        return cursor.rowcount

console = Console()


def _fmt_dur(seconds: float) -> str:
    """Formate des secondes en Xm Ys."""
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def _get_chroma_collection(method: str) -> tuple[chromadb.Collection | None, str]:
    """
    Retourne (collection, "") si OK, ou (None, message_erreur) si introuvable.
    """
    try:
        client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        collection = client.get_collection(name=method)
        return collection, ""
    except Exception as e:
        return None, str(e)


@dataclass
class Warning:
    level: str                              # "CRITIQUE" | "QUALITE"
    code: str                               # "C1", "Q1", "FP", etc.
    label: str                              # Description courte
    method: str
    track_id: str | None = None
    artist: str | None = None
    title: str | None = None
    metrics: dict = field(default_factory=dict)
    action: str = ""


def check_method(method: str) -> list[Warning]:
    """Retourne la liste des avertissements détectés pour une méthode d'embedding."""
    warns: list[Warning] = []

    fp_db      = Path(config.FINGERPRINTS_DB)
    meta_path  = PROCESSED_DIR / "metadata.parquet"
    index_path = Path(config.INDEX_DIR) / f"index_{method}_{config.INDEX_TYPE}.faiss"
    order_path = Path(config.INDEX_DIR) / f"segments_{method}.parquet"

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
    if "duration" in df_meta.columns:
        for m_meta in all_data["metadatas"]:
            tid     = m_meta["track_id"]
            start_s = m_meta["start_s"]
            row     = df_meta[df_meta["track_id"] == tid]
            if row.empty:
                continue
            duration = float(row["duration"].values[0])
            if start_s > duration:
                warns.append(Warning(
                    level="QUALITE", code="Q2", label="Segment hors durée du track",
                    method=method,
                    track_id=tid,
                    artist=str(row["artist"].values[0]) if "artist" in row else None,
                    title=str(row["title"].values[0])   if "title"  in row else None,
                    metrics={
                        "start_s":        f"{start_s:.1f}s",
                        "Durée du track": f"{duration:.1f}s",
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
                lower  = q1 - 1.5 * iqr
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


def purge_tracks(method: str, track_ids: set[str]) -> dict:
    """
    Supprime les données d'un ensemble de tracks pour une méthode donnée.

    - Segments supprimés de ChromaDB (delete par track_id)
    - Track supprimé de metadata.parquet (ligne entière)
    - Fingerprint supprimé de fingerprints.db (SQLite)
    - FAISS index + order parquet supprimés (devenus obsolètes → à reconstruire)

    Retourne un dict de stats : segments_removed, tracks_cleared, fingerprints_removed.
    """
    fp_db      = Path(config.FINGERPRINTS_DB)
    meta_path  = PROCESSED_DIR / "metadata.parquet"
    index_path = Path(config.INDEX_DIR) / f"index_{method}_{config.INDEX_TYPE}.faiss"
    order_path = Path(config.INDEX_DIR) / f"segments_{method}.parquet"

    stats = {"segments_removed": 0, "tracks_cleared": 0, "fingerprints_removed": 0}

    # --- ChromaDB ---
    collection, err = _get_chroma_collection(method)
    if collection is not None:
        for tid in track_ids:
            result = collection.get(where={"track_id": {"$eq": tid}})
            if result["ids"]:
                collection.delete(ids=result["ids"])
                stats["segments_removed"] += len(result["ids"])

    # --- FAISS index + order parquet (obsolètes après suppression) ---
    for p in [index_path, order_path]:
        if p.exists():
            p.unlink()

    # --- Metadata ---
    if meta_path.exists():
        df_meta  = pd.read_parquet(meta_path)
        affected = df_meta["track_id"].isin(track_ids)
        stats["tracks_cleared"] = int(affected.sum())
        if stats["tracks_cleared"] > 0:
            df_meta = df_meta[~affected]
            df_meta.to_parquet(meta_path, index=False)

    # --- Fingerprints (SQLite) ---
    stats["fingerprints_removed"] = _fp_delete(fp_db, track_ids)

    return stats


def _render_warning(w: Warning) -> None:
    """Affiche un warning formaté avec rich.Panel."""
    LEVEL_COLOR = {"CRITIQUE": "red", "QUALITE": "yellow"}
    LEVEL_ICON  = {"CRITIQUE": "✗", "QUALITE": "⚠"}

    color = LEVEL_COLOR.get(w.level, "white")
    icon  = LEVEL_ICON.get(w.level, "•")

    lines: list[str] = []

    # Artiste — Titre
    if w.artist and w.title:
        lines.append(f"  [bold]{w.artist}[/bold] — [bold]{w.title}[/bold]")
    elif w.title:
        lines.append(f"  [bold]{w.title}[/bold]")
    elif w.track_id:
        lines.append(f"  track_id : {w.track_id[:16]}...")

    # Métriques
    if w.metrics:
        if lines:
            lines.append("")
        for k, v in w.metrics.items():
            lines.append(f"  [dim]{k:<30}[/dim] {v}")

    # Action
    if w.action:
        lines.append("")
        first = True
        for line in w.action.split("\n"):
            prefix = f"  [bold {color}]→[/bold {color}] " if first else "    "
            lines.append(f"{prefix}{line}")
            first = False

    title   = f"[bold {color}]{icon} {w.level} · {w.code} · {w.label}[/bold {color}]"
    content = "\n".join(lines) if lines else "  (pas de détails)"

    console.print(Panel(
        content,
        title=title,
        title_align="left",
        border_style=color,
        padding=(0, 1),
    ))
    console.print("")


def main() -> None:
    import click

    @click.command()
    @click.option("--method", default=None,
                  help="Méthode à vérifier (défaut : toutes les méthodes détectées)")
    @click.option("--purge", is_flag=True, default=False,
                  help="Supprimer les tracks problématiques et les re-mettre en file")
    @click.option("--purge-missing-fp", is_flag=True, default=False,
                  help="Purger uniquement les tracks sans fingerprint")
    @click.option("--yes", "-y", is_flag=True, default=False,
                  help="Ne pas demander de confirmation avant de purger")
    def _main(
        method: str | None,
        purge: bool,
        purge_missing_fp: bool,
        yes: bool,
    ) -> None:
        # Détecter les méthodes disponibles depuis ChromaDB
        if method:
            methods = [method]
        else:
            try:
                client  = chromadb.PersistentClient(path=config.CHROMA_DIR)
                methods = [c.name for c in client.list_collections()]
            except Exception:
                methods = []

            if not methods:
                console.print(
                    "[yellow]Aucune collection ChromaDB trouvée dans "
                    f"{config.CHROMA_DIR}[/yellow]"
                )
                sys.exit(0)

        all_warns: list[Warning] = []

        for m in sorted(methods):
            # Compter les stats pour le header
            collection, _ = _get_chroma_collection(m)
            n_seg    = collection.count() if collection else "—"
            n_tracks = "—"
            if collection and n_seg != "—" and n_seg > 0:
                meta_data = _chroma_get_all(collection, include=["metadatas"])
                n_tracks  = len(set(md["track_id"] for md in meta_data["metadatas"]))

            n_fp = _fp_count(Path(config.FINGERPRINTS_DB))

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

        # --- Résumé final ---
        console.rule()
        n_crit = sum(1 for w in all_warns if w.level == "CRITIQUE")
        n_qual = sum(1 for w in all_warns if w.level == "QUALITE")

        if not all_warns:
            console.print("[bold green]✓ Toutes les données sont cohérentes.[/bold green]")
            if not purge and not purge_missing_fp:
                return

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

        # --- Mode PURGE MISSING FP ---
        if purge_missing_fp:
            _run_purge_missing_fp(methods, yes)
            return

        if not purge:
            console.print(
                "\n[dim]Astuce : relance avec [bold]--purge[/bold] pour supprimer "
                "automatiquement les tracks problématiques.[/dim]"
            )
            return

        # --- Mode PURGE ---
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

    _main()


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
            if not row.empty:
                artist = str(row["artist"].values[0])
                title  = str(row["title"].values[0])
                console.print(f"  [red]✗[/red]  [{m}]  {artist} — {title}")
            else:
                console.print(f"  [red]✗[/red]  [{m}]  track_id {tid[:12]}...")

    total_tracks = sum(len(v) for v in by_method.values())
    console.print("")
    console.print(
        f"[bold]{total_tracks} track(s)[/bold] vont être supprimés de ChromaDB, "
        "fingerprints et metadata, puis remis en file de téléchargement."
    )
    console.print(
        "[dim]L'index FAISS sera également supprimé — relancer build_index.py après.[/dim]"
    )

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
        console.print(
            f"[green]✓[/green]  "
            f"{stats['segments_removed']} segments supprimés, "
            f"{stats['tracks_cleared']} tracks délistés, "
            f"{stats['fingerprints_removed']} fingerprints supprimés"
        )

    console.print("")
    console.print(
        f"[bold green]Purge terminée.[/bold green]  "
        f"{total_tracks} track(s), {total_segs} segments supprimés."
    )
    console.print("\n[dim]Pour re-télécharger et reconstruire l'index :[/dim]")
    console.print("  [bold cyan]python scripts/download_music.py[/bold cyan]")
    console.print("  [bold cyan]python src/index/build_index.py[/bold cyan]")


def _run_purge_missing_fp(methods: list[str], yes: bool) -> None:
    """Purge les tracks qui ont des embeddings mais pas de fingerprint."""
    fp_db = Path(config.FINGERPRINTS_DB)
    fp_ids = _fp_load_stats(fp_db).keys()  # set-like view des track_id connus

    by_method: dict[str, set[str]] = {}
    for m in methods:
        collection, _ = _get_chroma_collection(m)
        if collection is None:
            continue
        data = _chroma_get_all(collection, include=["metadatas"])
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


if __name__ == "__main__":
    main()
