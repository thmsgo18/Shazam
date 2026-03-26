"""
scripts/check_data.py

Vérifie la cohérence des données générées par download_music.py.

Checks critiques :
  [C1] Dimension embeddings cohérente (2*N_MFCC / 512 / 768 selon méthode)
  [C2] NaN / Inf dans embeddings.npy
  [C3] embeddings.npy ↔ segments.parquet (même nombre de lignes)
  [C4] segment_ids dupliqués
  [C5] FAISS index ↔ embeddings.npy (même nombre de vecteurs)
  [C6] segments.parquet ↔ metadata.parquet (pas de track_id orphelin)
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
"""
from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel

from src import config

FEATURES_DIR = Path("data/features")
PROCESSED_DIR = Path("data/processed")

console = Console()


def _fmt_dur(seconds: float) -> str:
    """Formate des secondes en Xm Ys."""
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


@dataclass
class Warning:
    level: str                              # "CRITIQUE" | "QUALITE"
    code: str                               # "C1", "Q1", "FP", etc.
    label: str                              # Description courte
    method: str
    track_id: str | None = None
    artist: str | None = None
    title: str | None = None
    metrics: dict = field(default_factory=dict)  # {label: valeur} à afficher
    explanation: str = ""                   # Cause probable
    action: str = ""                        # Quoi faire pour corriger


def check_method(method: str) -> list[Warning]:
    """Retourne la liste des avertissements détectés pour une méthode d'embedding."""
    warns: list[Warning] = []

    emb_path  = FEATURES_DIR / f"embeddings_{method}.npy"
    seg_path  = FEATURES_DIR / f"segments_{method}.parquet"
    fp_path   = FEATURES_DIR / "fingerprints.pkl"
    meta_path = PROCESSED_DIR / "metadata.parquet"

    # --- Fichiers manquants ---
    if not emb_path.exists():
        warns.append(Warning(
            level="CRITIQUE", code="MANQUANT", label="Fichier embeddings manquant",
            method=method,
            metrics={"Fichier": str(emb_path)},
            explanation="Le fichier n'a pas été créé — aucun track traité pour cette méthode.",
            action=f"Lancer download_music.py avec EMBEDDING_METHOD={method}",
        ))
        return warns
    if not seg_path.exists():
        warns.append(Warning(
            level="CRITIQUE", code="MANQUANT", label="Fichier segments manquant",
            method=method,
            metrics={"Fichier": str(seg_path)},
            explanation="Le fichier n'a pas été créé.",
            action="Lancer download_music.py",
        ))
        return warns

    # --- Chargement embeddings + segments ---
    try:
        emb = np.load(emb_path)
    except Exception as e:
        warns.append(Warning(
            level="CRITIQUE", code="CORROMPU", label="embeddings.npy illisible",
            method=method,
            metrics={"Erreur": str(e)},
            explanation="Le fichier numpy est corrompu (crash pendant l'écriture).",
            action="Supprimer le fichier et relancer download_music.py",
        ))
        return warns

    try:
        df_seg = pd.read_parquet(seg_path)
    except Exception as e:
        warns.append(Warning(
            level="CRITIQUE", code="CORROMPU", label="segments.parquet illisible",
            method=method,
            metrics={"Erreur": str(e)},
            explanation="Le fichier parquet est corrompu.",
            action="Supprimer le fichier et relancer download_music.py",
        ))
        return warns

    # --- [C1] Dimension des embeddings ---
    if emb.ndim != 2:
        warns.append(Warning(
            level="CRITIQUE", code="C1", label="embeddings.npy malformé (pas 2D)",
            method=method,
            metrics={"Shape observée": str(emb.shape), "Attendu": "2D"},
            explanation="Le tableau n'est pas 2D — probablement un bug dans embed_segment.",
            action="Re-générer les embeddings",
        ))
        return warns

    expected_dims = {
        "mfcc": 2 * config.N_MFCC,  # mean + std → 2 × N_MFCC
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
            explanation=(
                "Les embeddings ont été générés avec une config différente de l'actuelle "
                f"(N_MFCC a probablement changé)." if method == "mfcc"
                else f"Dimension inattendue pour {method}."
            ),
            action="Vérifier config.py — si N_MFCC a changé, supprimer les embeddings et relancer",
        ))

    # --- [C2] NaN / Inf ---
    if not np.isfinite(emb).all():
        n_bad = (~np.isfinite(emb)).any(axis=1).sum()
        warns.append(Warning(
            level="CRITIQUE", code="C2", label="NaN ou Inf dans les embeddings",
            method=method,
            metrics={
                "Vecteurs corrompus": f"{n_bad} / {len(emb)}",
            },
            explanation="Des valeurs NaN/Inf font retourner de faux résultats à FAISS.",
            action="Identifier les tracks concernés et les re-télécharger",
        ))

    # --- [C3] Cohérence embeddings ↔ segments ---
    if len(emb) != len(df_seg):
        warns.append(Warning(
            level="CRITIQUE", code="C3", label="Désynchronisation embeddings ↔ segments",
            method=method,
            metrics={
                "Embeddings (npy)":    str(len(emb)),
                "Segments (parquet)":  str(len(df_seg)),
                "Différence":          str(abs(len(emb) - len(df_seg))),
            },
            explanation="Tailles différentes entre .npy et .parquet — probablement un crash pendant _save_track.",
            action="Supprimer les deux fichiers et relancer download_music.py",
        ))

    # --- [C4] segment_ids dupliqués ---
    if df_seg["segment_id"].duplicated().any():
        n_dup = df_seg["segment_id"].duplicated().sum()
        dup_tracks = df_seg[df_seg["segment_id"].duplicated(keep=False)]["track_id"].unique()
        warns.append(Warning(
            level="CRITIQUE", code="C4", label="segment_ids dupliqués",
            method=method,
            metrics={
                "Doublons":          str(n_dup),
                "Tracks concernés":  str(len(dup_tracks)),
            },
            explanation="Des segment_ids identiques → FAISS retournera le même résultat plusieurs fois.",
            action="Supprimer les fichiers et relancer download_music.py",
        ))

    # --- [C5] FAISS index ↔ embeddings ---
    index_path = Path(config.INDEX_DIR) / f"index_{method}_{config.INDEX_TYPE}.faiss"
    if index_path.exists():
        try:
            import faiss
            index = faiss.read_index(str(index_path))
            if index.ntotal != len(emb):
                warns.append(Warning(
                    level="CRITIQUE", code="C5", label="FAISS index désynchronisé",
                    method=method,
                    metrics={
                        "Vecteurs dans FAISS":         str(index.ntotal),
                        "Vecteurs dans embeddings.npy": str(len(emb)),
                        "Différence":                  str(abs(index.ntotal - len(emb))),
                    },
                    explanation="L'index FAISS n'a pas été reconstruit après le dernier ajout de tracks.",
                    action="Relancer build_index.py",
                ))
        except Exception as e:
            warns.append(Warning(
                level="CRITIQUE", code="C5", label="FAISS index illisible",
                method=method,
                metrics={"Erreur": str(e)},
                explanation="Le fichier FAISS est corrompu.",
                action="Supprimer l'index et relancer build_index.py",
            ))

    # --- Chargement metadata ---
    if not meta_path.exists():
        warns.append(Warning(
            level="CRITIQUE", code="META", label="metadata.parquet manquant",
            method=method,
            explanation="Aucune métadonnée disponible.",
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
            explanation="Le fichier metadata est corrompu.",
            action="Vérifier data/processed/metadata.parquet",
        ))
        return warns

    seg_track_ids  = set(df_seg["track_id"].unique())
    meta_track_ids = set(df_meta["track_id"].unique())

    # --- [C6] Segments orphelins ---
    orphans = seg_track_ids - meta_track_ids
    for oid in list(orphans)[:5]:
        warns.append(Warning(
            level="CRITIQUE", code="C6", label="Segments orphelins (sans metadata)",
            method=method,
            track_id=oid,
            metrics={"track_id": oid[:16] + "..."},
            explanation="Ce track a des segments mais est absent de metadata.parquet — crash partiel pendant _save_track.",
            action="Supprimer les segments de ce track et relancer download_music.py",
        ))
    if len(orphans) > 5:
        warns.append(Warning(
            level="CRITIQUE", code="C6",
            label=f"… et {len(orphans) - 5} autre(s) track(s) orphelins",
            method=method,
        ))

    # --- [C7] Embedding incomplet ---
    if "duration" in df_meta.columns:
        win_s = config.SEGMENT_WIN_S
        hop_s = config.SEGMENT_HOP_S
        seg_counts = df_seg.groupby("track_id").size().rename("actual").reset_index()
        df_check = seg_counts.merge(
            df_meta[["track_id", "duration", "artist", "title"]], on="track_id", how="left"
        )
        df_check["expected"] = (
            ((df_check["duration"] - win_s) / hop_s).clip(lower=0).astype(int) + 1
        )
        df_check["ratio"] = df_check["actual"] / df_check["expected"]
        for row in df_check[df_check["ratio"] < 0.8].sort_values("ratio").itertuples():
            warns.append(Warning(
                level="CRITIQUE", code="C7", label="Embedding incomplet",
                method=method,
                track_id=row.track_id,
                artist=str(row.artist),
                title=str(row.title),
                metrics={
                    "Segments réels / attendus": f"{row.actual} / {row.expected}  ({row.ratio:.0%})",
                    "Durée du track":            _fmt_dur(row.duration),
                },
                explanation="Moins de 80% des segments attendus ont été générés — crash pendant le processing.",
                action="Relancer download_music.py (le track sera re-traité automatiquement)",
            ))

    # --- [Q1] Durée aberrante ---
    if "duration" in df_meta.columns:
        bad_dur = df_meta[(df_meta["duration"] <= 0) | (df_meta["duration"] > 600)]
        for row in bad_dur.itertuples():
            n_segs = df_seg[df_seg["track_id"] == row.track_id].shape[0]
            warns.append(Warning(
                level="QUALITE", code="Q1", label="Durée aberrante",
                method=method,
                track_id=row.track_id,
                artist=str(getattr(row, "artist", "?")),
                title=str(getattr(row, "title", "?")),
                metrics={
                    "Durée":             f"{row.duration:.0f}s  ({_fmt_dur(row.duration)})",
                    "Seuil normal":      "entre 1s et 10min (600s)",
                    "Segments générés":  str(n_segs),
                },
                explanation=(
                    "yt-dlp a probablement téléchargé une compilation ou une version live longue "
                    "au lieu du single." if row.duration > 600
                    else "Durée nulle ou négative — fichier audio probablement corrompu."
                ),
                action="Supprimer le track et le re-télécharger en affinant la requête yt-dlp",
            ))

    # --- [Q2] start_s > duration ---
    if "duration" in df_meta.columns:
        df_seg_dur = df_seg.merge(
            df_meta[["track_id", "duration", "artist", "title"]], on="track_id", how="left"
        )
        bad_starts = df_seg_dur[df_seg_dur["start_s"] > df_seg_dur["duration"]]
        for tid in bad_starts["track_id"].unique()[:5]:
            sub      = bad_starts[bad_starts["track_id"] == tid]
            row_meta = df_meta[df_meta["track_id"] == tid].iloc[0]
            warns.append(Warning(
                level="QUALITE", code="Q2", label="Segment hors durée du track",
                method=method,
                track_id=tid,
                artist=str(row_meta.get("artist", "?")),
                title=str(row_meta.get("title", "?")),
                metrics={
                    "start_s max observé":   f"{sub['start_s'].max():.1f}s",
                    "Durée du track":        f"{row_meta['duration']:.1f}s",
                    "Segments incohérents":  str(len(sub)),
                },
                explanation="Des segments démarrent après la fin du track — incohérence dans la segmentation.",
                action="Supprimer les segments de ce track et relancer download_music.py",
            ))

    # --- Tracks marqués traités mais sans segments ---
    if "embedded_methods" in df_meta.columns:
        should_have = set(
            df_meta[df_meta["embedded_methods"].apply(
                lambda m: hasattr(m, '__iter__') and not isinstance(m, str) and method in m
            )]["track_id"]
        )
        for tid in list(should_have - seg_track_ids)[:5]:
            row_meta = df_meta[df_meta["track_id"] == tid]
            warns.append(Warning(
                level="CRITIQUE", code="C6b", label="Track marqué traité mais sans segments",
                method=method,
                track_id=tid,
                artist=str(row_meta["artist"].values[0]) if "artist" in row_meta else None,
                title=str(row_meta["title"].values[0])  if "title"  in row_meta else None,
                metrics={"track_id": tid[:16] + "..."},
                explanation=f"embedded_methods contient '{method}' mais aucun segment n'existe — crash partiel pendant _save_track.",
                action="Supprimer la méthode de embedded_methods et relancer download_music.py",
            ))

    # --- Fingerprints ---
    if not fp_path.exists():
        warns.append(Warning(
            level="QUALITE", code="FP", label="fingerprints.pkl manquant",
            method=method,
            explanation="Aucun fingerprint calculé — Stage 2 entièrement inopérant.",
            action="Lancer download_music.py pour générer les fingerprints",
        ))
        return warns

    try:
        with open(fp_path, "rb") as f:
            fingerprints = pickle.load(f)
    except Exception as e:
        warns.append(Warning(
            level="CRITIQUE", code="FP", label="fingerprints.pkl corrompu",
            method=method,
            metrics={"Erreur": str(e)},
            explanation="Le fichier pickle est illisible.",
            action="Supprimer fingerprints.pkl et relancer download_music.py",
        ))
        return warns

    # [Q3] Fingerprint vide
    for tid in [t for t, fp in fingerprints.items() if len(fp) == 0][:5]:
        row_meta = df_meta[df_meta["track_id"] == tid]
        warns.append(Warning(
            level="QUALITE", code="Q3", label="Fingerprint vide (0 hash)",
            method=method,
            track_id=tid,
            artist=str(row_meta["artist"].values[0]) if not row_meta.empty and "artist" in row_meta else None,
            title=str(row_meta["title"].values[0])  if not row_meta.empty and "title"  in row_meta else None,
            metrics={"Hashes": "0"},
            explanation="extract_fingerprint n'a trouvé aucun pic spectral — signal trop silencieux ou corrompu.",
            action="Stage 2 inopérant pour ce track. Re-télécharger si la qualité audio est suspecte.",
        ))

    # [FP] Tracks sans fingerprint
    missing_fp = seg_track_ids - set(fingerprints.keys())
    if missing_fp:
        n_total = len(seg_track_ids)
        warns.append(Warning(
            level="QUALITE", code="FP", label="Fingerprints manquants",
            method=method,
            metrics={
                "Tracks sans fingerprint": f"{len(missing_fp)} / {n_total}  ({len(missing_fp)/n_total:.0%})",
            },
            explanation=(
                "fingerprints.pkl a été réinitialisé plusieurs fois après corruption (EOFError). "
                "Les tracks traités avant le dernier reset ont perdu leur fingerprint."
            ),
            action=(
                "Stage 2 (re-ranking Shazam) inopérant pour ces tracks.\n"
                "Pour recalculer : lancer --purge pour remettre ces tracks en file,\n"
                "puis relancer download_music.py (re-téléchargement + re-fingerprint complet)."
            ),
        ))

    # [Q4] Fingerprint outlier (IQR par tranche de durée)
    if "duration" in df_meta.columns:
        rows = []
        for tid, fp in fingerprints.items():
            row = df_meta[df_meta["track_id"] == tid]
            if row.empty or float(row["duration"].values[0]) == 0:
                continue
            rows.append({
                "track_id": tid,
                "artist":   str(row["artist"].values[0]) if "artist" in row else "",
                "title":    str(row["title"].values[0])  if "title"  in row else "",
                "duration": float(row["duration"].values[0]),
                "n_hashes": len(fp),
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
                        level="QUALITE", code="Q4", label="Fingerprint anormalement pauvre",
                        method=method,
                        track_id=r.track_id,
                        artist=r.artist,
                        title=r.title,
                        metrics={
                            "Hashes":              str(r.n_hashes),
                            "Médiane du groupe":   f"{median:.0f}",
                            "Durée":               _fmt_dur(r.duration),
                        },
                        explanation=(
                            "Beaucoup moins de hashes que les autres tracks de durée similaire — "
                            "signal audio de mauvaise qualité, silence excessif, ou encoding tronqué."
                        ),
                        action="Écouter le fichier audio pour vérifier, re-télécharger si nécessaire",
                    ))

    return warns


def purge_tracks(method: str, track_ids: set[str]) -> dict:
    """
    Supprime les données d'un ensemble de tracks pour une méthode donnée.

    Pour chaque track supprimé :
      - Les segments sont retirés de segments_{method}.parquet
      - Les lignes correspondantes sont retirées de embeddings_{method}.npy
        et les segment_ids des tracks restants sont renumérotés (0..N-1)
      - La méthode est retirée de embedded_methods dans metadata.parquet
        → le track sera re-traité lors du prochain download_music.py
      - Le fingerprint est supprimé de fingerprints.pkl

    Retourne un dict de stats : segments_removed, tracks_cleared, fingerprints_removed.
    """
    emb_path  = FEATURES_DIR / f"embeddings_{method}.npy"
    seg_path  = FEATURES_DIR / f"segments_{method}.parquet"
    fp_path   = FEATURES_DIR / "fingerprints.pkl"
    meta_path = PROCESSED_DIR / "metadata.parquet"

    stats = {"segments_removed": 0, "tracks_cleared": 0, "fingerprints_removed": 0}

    if not seg_path.exists() or not emb_path.exists():
        return stats

    # --- Segments + Embeddings ---
    df_seg = pd.read_parquet(seg_path)
    emb    = np.load(emb_path)

    mask_remove = df_seg["track_id"].isin(track_ids)
    n_remove    = int(mask_remove.sum())

    if n_remove > 0:
        stats["segments_removed"] = n_remove

        # Récupérer les indices (= segment_ids) des lignes à supprimer dans emb
        seg_ids_to_remove = set(df_seg.loc[mask_remove, "segment_id"].values)

        # Garder uniquement les lignes des tracks conservés
        df_seg_new = df_seg[~mask_remove].copy().reset_index(drop=True)

        # Reconstruire embeddings.npy : garder les lignes dont l'index n'est pas supprimé
        keep_mask = np.ones(len(emb), dtype=bool)
        for sid in seg_ids_to_remove:
            if sid < len(emb):
                keep_mask[sid] = False
        emb_new = emb[keep_mask]

        # Renuméroter les segment_ids (doivent rester == index de ligne dans le .npy)
        df_seg_new["segment_id"] = np.arange(len(df_seg_new))

        # Écrire (ordre : segments en premier, comme dans _save_track)
        df_seg_new.to_parquet(seg_path, index=False)
        np.save(emb_path, emb_new)

    # --- Metadata : supprimer entièrement les lignes des tracks purgés ---
    # (pas juste effacer embedded_methods — la ligne entière doit disparaître
    #  pour que download_music.py re-télécharge le track from scratch)
    if meta_path.exists():
        df_meta  = pd.read_parquet(meta_path)
        affected = df_meta["track_id"].isin(track_ids)
        stats["tracks_cleared"] = int(affected.sum())

        if stats["tracks_cleared"] > 0:
            df_meta = df_meta[~affected]
            df_meta.to_parquet(meta_path, index=False)

    # --- Fingerprints ---
    if fp_path.exists():
        try:
            with open(fp_path, "rb") as f:
                fingerprints = pickle.load(f)
            removed = [tid for tid in track_ids if tid in fingerprints]
            stats["fingerprints_removed"] = len(removed)
            if removed:
                for tid in removed:
                    del fingerprints[tid]
                with open(fp_path, "wb") as f:
                    pickle.dump(fingerprints, f)
        except Exception:
            pass

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

    title = f"[bold {color}]{icon} {w.level} · {w.code} · {w.label}[/bold {color}]"
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
                  help="Supprimer les données des tracks flaggés et les re-mettre en file de téléchargement")
    @click.option("--yes", "-y", is_flag=True, default=False,
                  help="Ne pas demander de confirmation avant de purger")
    def _main(method: str | None, purge: bool, yes: bool) -> None:
        if method:
            methods = [method]
        else:
            methods = [
                p.stem.replace("embeddings_", "")
                for p in FEATURES_DIR.glob("embeddings_*.npy")
            ]
            if not methods:
                console.print("[yellow]Aucun embedding trouvé dans data/features/[/yellow]")
                sys.exit(0)

        all_warns: list[Warning] = []

        for m in sorted(methods):
            emb_path  = FEATURES_DIR / f"embeddings_{m}.npy"
            seg_path  = FEATURES_DIR / f"segments_{m}.parquet"
            fp_path   = FEATURES_DIR / "fingerprints.pkl"
            meta_path = PROCESSED_DIR / "metadata.parquet"

            n_emb    = len(np.load(emb_path))            if emb_path.exists()  else "—"
            n_seg    = len(pd.read_parquet(seg_path))    if seg_path.exists()  else "—"
            n_fp     = len(pickle.load(open(fp_path, "rb"))) if fp_path.exists() else "—"
            n_tracks = "—"
            if meta_path.exists():
                df_meta = pd.read_parquet(meta_path)
                if "embedded_methods" in df_meta.columns:
                    n_tracks = df_meta["embedded_methods"].apply(
                        lambda x: hasattr(x, '__iter__') and not isinstance(x, str) and m in x
                    ).sum()

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
                f"[dim]{n_emb} emb · {n_seg} segs · {n_tracks} tracks · {n_fp} fps[/dim]  "
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

        if not purge:
            console.print(
                "\n[dim]Astuce : relance avec [bold]--purge[/bold] pour supprimer automatiquement "
                "les tracks problématiques et les re-mettre en file de téléchargement.[/dim]"
            )
            return

        # --- Mode PURGE ---
        # Collecter les track_ids purgeable par méthode (ceux qui ont un track_id dans le warning)
        # Les warnings globaux (FP manquants, etc.) sans track_id ne sont pas purgeable unitairement.
        by_method: dict[str, set[str]] = {}
        for w in all_warns:
            if w.track_id:
                by_method.setdefault(w.method, set()).add(w.track_id)

        if not by_method:
            console.print(
                "\n[yellow]Aucun track individuel à purger "
                "(les alertes sont globales, voir les actions suggérées ci-dessus).[/yellow]"
            )
            return

        # Afficher le récapitulatif de ce qui sera supprimé
        console.print("")
        console.rule("[bold red]Récapitulatif de la purge[/bold red]")
        console.print("")

        for m, tids in sorted(by_method.items()):
            meta_path = PROCESSED_DIR / "metadata.parquet"
            if meta_path.exists():
                df_meta = pd.read_parquet(meta_path)
                for tid in sorted(tids):
                    row = df_meta[df_meta["track_id"] == tid]
                    if not row.empty:
                        artist = str(row["artist"].values[0])
                        title  = str(row["title"].values[0])
                        console.print(f"  [red]✗[/red]  [{m}]  {artist} — {title}")
                    else:
                        console.print(f"  [red]✗[/red]  [{m}]  track_id {tid[:12]}... (absent de metadata)")

        total_tracks = sum(len(v) for v in by_method.values())
        console.print("")
        console.print(
            f"[bold]{total_tracks} track(s)[/bold] vont être supprimés des embeddings, "
            "segments et fingerprints, puis remis en file de téléchargement."
        )

        # Confirmation
        if not yes:
            console.print("")
            confirm = console.input(
                "[bold yellow]Continuer ? [/bold yellow][dim](o = confirmer / Entrée = annuler) [/dim]"
            ).strip().lower()
            if confirm not in ("o", "oui", "y", "yes"):
                console.print("[dim]Annulé.[/dim]")
                return

        # Purge
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
        console.print(
            "\n[dim]Pour re-télécharger et re-traiter ces tracks :[/dim]"
        )
        console.print(
            "  [bold cyan]python scripts/download_music.py[/bold cyan]"
        )

    _main()


if __name__ == "__main__":
    main()
