"""
src/maintenance/check.py

Data consistency check (ChromaDB, FAISS, fingerprints, metadata).

Critical checks:
  [C1] Consistent embedding dimension
  [C2] NaN / Inf in embeddings
  [C3] ChromaDB ↔ FAISS order parquet (same number of segments)
  [C5] FAISS index ↔ ChromaDB (same number of vectors)
  [C6] Orphan segments (in ChromaDB but missing from metadata.parquet)
  [C7] Tracks with incomplete embedding (< 80% expected segments)

Quality checks:
  [Q1] Abnormal duration (≤ 0s or > 10min)
  [Q2] start_s of segments > track duration
  [Q3] Completely empty fingerprint
  [Q4] Fingerprint outlier downward (IQR by duration bucket)
  [FP] Tracks without fingerprint

Public entry point: run_check(method, details, metadata, purge, purge_missing_fp, yes)
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
    """Formats seconds as Xm Ys."""
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
    Fetches all documents from a ChromaDB collection in pages of 500.
    Avoids the SQLite 'too many SQL variables' error on large collections.
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
    """Returns (collection, "") if OK, or (None, error_message) if not found."""
    try:
        client     = chromadb.PersistentClient(path=str(ROOT / config.CHROMA_DIR))
        collection = client.get_collection(name=method)
        return collection, ""
    except Exception as e:
        return None, str(e)


def _collection_family(collection_key: str) -> str:
    """Infers the embedding family from a ChromaDB collection key."""
    for family in ("mfcc", "clap", "muq", "mert"):
        if collection_key == family or collection_key.startswith(f"{family}_"):
            return family
    return collection_key


def _expected_embedding_dim(collection_key: str) -> int | None:
    """Returns the expected embedding dimension for a given collection key."""
    family = _collection_family(collection_key)
    if family == "mfcc":
        return 2 * config.N_MFCC
    if family == "clap":
        return 512
    if family == "muq":
        return 1024
    if family == "mert":
        return 1024 if "330m" in collection_key.lower() else 768
    return None


def _embedded_method_matches_collection(method_key, collection_key: str) -> bool:
    """True if an embedded_methods entry matches the ChromaDB collection."""
    if not isinstance(method_key, str):
        return False
    if method_key == "mfcc":
        return collection_key == "mfcc"
    if ":" not in method_key:
        return method_key == collection_key

    family, model_name = method_key.split(":", 1)
    if family not in {"clap", "muq", "mert"}:
        return False
    model_slug = model_name.split("/")[-1].replace("-", "_")
    return collection_key == f"{family}_{model_slug}"


# ---------------------------------------------------------------------------
# Warning dataclass
# ---------------------------------------------------------------------------

@dataclass
class Warning:
    level:    str                          # "CRITICAL" | "QUALITY"
    code:     str                          # "C1", "Q1", "FP", etc.
    label:    str                          # Short description
    method:   str
    track_id: str | None = None
    artist:   str | None = None
    title:    str | None = None
    metrics:  dict = field(default_factory=dict)
    action:   str  = ""


# ---------------------------------------------------------------------------
# check_method — full checks (for --details and --purge)
# ---------------------------------------------------------------------------

def check_method(method: str) -> list[Warning]:
    """Returns the list of warnings detected for a given embedding method."""
    warns: list[Warning] = []

    fp_db      = ROOT / config.FINGERPRINTS_DB
    meta_path  = PROCESSED_DIR / "metadata.parquet"
    index_path = ROOT / config.INDEX_DIR / f"index_{method}_{config.INDEX_TYPE}.faiss"
    order_path = ROOT / config.INDEX_DIR / f"segments_{method}.parquet"

    # --- ChromaDB: collection is mandatory ---
    collection, err = _get_chroma_collection(method)
    if collection is None:
        warns.append(Warning(
            level="CRITICAL", code="MISSING", label="Missing ChromaDB collection",
            method=method,
            metrics={"Method": method, "Error": err},
            action="Run `python manage.py ingest` to (re)create the collection.",
        ))
        return warns

    n_segments = collection.count()
    if n_segments == 0:
        warns.append(Warning(
            level="CRITICAL", code="EMPTY", label="Empty ChromaDB collection",
            method=method,
            action="Run `python manage.py ingest` to generate the embeddings.",
        ))
        return warns
    
    # Load embeddings + metadata. Embeddings are only needed for C1/C2 checks,
    # so the list-of-lists and the numpy array are freed as soon as those checks pass.
    all_data = _chroma_get_all(collection, include=["embeddings", "metadatas"])
    emb      = np.array(all_data["embeddings"], dtype=np.float32)
    del all_data["embeddings"]  # free list-of-lists immediately after conversion

   # --- [C1] Embedding dimension ---
    if emb.ndim != 2:
        warns.append(Warning(
            level="CRITICAL", code="C1", label="Malformed embeddings (not 2D)",
            method=method,
            metrics={"Observed shape": str(emb.shape)},
            action="Re-generate embeddings",
        ))
        return warns

    expected_dim = _expected_embedding_dim(method)
    if expected_dim is not None and emb.shape[1] != expected_dim:
        warns.append(Warning(
            level="CRITICAL", code="C1", label="Unexpected embedding dimension",
            method=method,
            metrics={
                "Observed dimension": str(emb.shape[1]),
                "Expected dimension": str(expected_dim),
                **({("Current config N_MFCC"): str(config.N_MFCC)} if _collection_family(method) == "mfcc" else {}),
            },
            action="Check config.py — if N_MFCC changed, delete the collection and re-run",
        ))

    # --- [C2] NaN / Inf ---
    if not np.isfinite(emb).all():
        n_bad = (~np.isfinite(emb)).any(axis=1).sum()
        warns.append(Warning(
            level="CRITICAL", code="C2", label="NaN or Inf in embeddings",
            method=method,
            metrics={"Corrupted vectors": f"{n_bad} / {len(emb)}"},
            action="Identify affected tracks and re-download them",
        ))

    del emb  # no longer needed after C1/C2

    # --- [C3] ChromaDB ↔ order parquet (if index was built) ---
    if order_path.exists():
        df_order = pd.read_parquet(order_path)
        if len(df_order) != n_segments:
            warns.append(Warning(
                level="CRITICAL", code="C3",
                label="Order parquet out of sync with ChromaDB",
                method=method,
                metrics={
                    "Segments in ChromaDB":      str(n_segments),
                    "Segments in order parquet": str(len(df_order)),
                    "Difference":                str(abs(n_segments - len(df_order))),
                },
                action="Run `python manage.py rebuild --what index` to rebuild the index.",
            ))

    # --- [C5] FAISS index ↔ ChromaDB ---
    if index_path.exists():
        try:
            import faiss
            index = faiss.read_index(str(index_path))
            if index.ntotal != n_segments:
                warns.append(Warning(
                    level="CRITICAL", code="C5", label="FAISS index out of sync",
                    method=method,
                    metrics={
                        "Vectors in FAISS":    str(index.ntotal),
                        "Vectors in ChromaDB": str(n_segments),
                        "Difference":          str(abs(index.ntotal - n_segments)),
                    },
                    action="Run `python manage.py rebuild --what index`.",
                ))
        except Exception as e:
            warns.append(Warning(
                level="CRITICAL", code="C5", label="FAISS index unreadable",
                method=method,
                metrics={"Error": str(e)},
                action="Run `python manage.py rebuild --what index`.",
            ))
    else:
        warns.append(Warning(
            level="CRITICAL", code="C5", label="FAISS index missing",
            method=method,
            action="Run `python manage.py rebuild --what index`.",
        ))

    # --- Metadata ---
    if not meta_path.exists():
        warns.append(Warning(
            level="CRITICAL", code="META", label="metadata.parquet missing",
            method=method,
            action="Run `python manage.py ingest`.",
        ))
        return warns

    try:
        df_meta = pd.read_parquet(meta_path)
    except Exception as e:
        warns.append(Warning(
            level="CRITICAL", code="META", label="metadata.parquet unreadable",
            method=method,
            metrics={"Error": str(e)},
            action="Check data/processed/metadata.parquet",
        ))
        return warns

    # Build O(1) index: avoids repeated O(N) full-scan inside every check loop
    df_meta_idx  = df_meta.drop_duplicates("track_id").set_index("track_id")
    has_duration = "duration" in df_meta.columns
    has_artist   = "artist"   in df_meta.columns
    has_title    = "title"    in df_meta.columns

    # Build mapping track_id → segment count from ChromaDB
    chroma_track_seg_counts: dict[str, int] = {}
    for m in all_data["metadatas"]:
        tid = m["track_id"]
        chroma_track_seg_counts[tid] = chroma_track_seg_counts.get(tid, 0) + 1

    chroma_track_ids = set(chroma_track_seg_counts.keys())
    meta_track_ids   = set(df_meta["track_id"].unique())

    # --- [C6] Orphan segments (in ChromaDB but missing from metadata) ---
    orphans = chroma_track_ids - meta_track_ids
    for oid in list(orphans)[:5]:
        warns.append(Warning(
            level="CRITICAL", code="C6", label="Orphan segments (no metadata)",
            method=method,
            track_id=oid,
            metrics={"track_id": oid[:16] + "..."},
            action="Delete this track's segments then run `python manage.py ingest`.",
        ))
    if len(orphans) > 5:
        warns.append(Warning(
            level="CRITICAL", code="C6",
            label=f"… and {len(orphans) - 5} more orphan track(s)",
            method=method,
        ))

    # --- [C7] Incomplete embedding (< 80% of expected segments) ---
    if has_duration:
        win_s = config.SEGMENT_WIN_S
        hop_s = config.SEGMENT_HOP_S
        for tid, actual in chroma_track_seg_counts.items():
            if tid not in df_meta_idx.index:
                continue
            row      = df_meta_idx.loc[tid]
            duration = float(row["duration"])
            expected = int(max(0, (duration - win_s) / hop_s)) + 1
            if expected > 0 and actual / expected < 0.8:
                warns.append(Warning(
                    level="CRITICAL", code="C7", label="Incomplete embedding",
                    method=method,
                    track_id=tid,
                    artist=str(row["artist"]) if has_artist else None,
                    title=str(row["title"])   if has_title  else None,
                    metrics={
                        "Actual / expected segments": f"{actual} / {expected}  ({actual/expected:.0%})",
                        "Track duration":             _fmt_dur(duration),
                    },
                    action="Run `python manage.py ingest` (track will be re-processed automatically).",
                ))

    # --- [Q1] Abnormal duration ---
    if "duration" in df_meta.columns:
        bad_dur = df_meta[(df_meta["duration"] <= 0) | (df_meta["duration"] > 600)]
        for row in bad_dur.itertuples():
            n_segs = chroma_track_seg_counts.get(row.track_id, 0)
            warns.append(Warning(
                level="QUALITY", code="Q1", label="Abnormal duration",
                method=method,
                track_id=row.track_id,
                artist=str(getattr(row, "artist", "?")),
                title=str(getattr(row, "title", "?")),
                metrics={
                    "Duration":          f"{row.duration:.0f}s  ({_fmt_dur(row.duration)})",
                    "Expected range":    "between 1s and 10min (600s)",
                    "Generated segments": str(n_segs),
                },
                action=(
                    "Delete the track (--purge) and re-download it"
                    if row.duration > 600 else
                    "Zero or negative duration — audio file is likely corrupted"
                ),
            ))

    # --- [Q2] start_s > track duration ---
    # Tolerance of one segment window (SEGMENT_WIN_S) for yt-dlp / librosa discrepancies
    Q2_TOLERANCE_S = config.SEGMENT_WIN_S
    if has_duration:
        for m_meta in all_data["metadatas"]:
            tid     = m_meta["track_id"]
            start_s = m_meta["start_s"]
            if tid not in df_meta_idx.index:
                continue
            row      = df_meta_idx.loc[tid]
            duration = float(row["duration"])
            if start_s > duration + Q2_TOLERANCE_S:
                warns.append(Warning(
                    level="QUALITY", code="Q2", label="Segment beyond track duration",
                    method=method,
                    track_id=tid,
                    artist=str(row["artist"]) if has_artist else None,
                    title=str(row["title"])   if has_title  else None,
                    metrics={
                        "start_s":        f"{start_s:.1f}s",
                        "Track duration": f"{duration:.1f}s",
                        "Gap":            f"{start_s - duration:.1f}s  (tolerance: {Q2_TOLERANCE_S}s)",
                    },
                    action="Delete the track (`--purge`) then run `python manage.py ingest`.",
                ))
                break  # one warning per track only

    # --- Tracks marked as processed but missing segments in ChromaDB ---
    if "embedded_methods" in df_meta.columns:
        should_have = set(
            df_meta[df_meta["embedded_methods"].apply(
                lambda x: hasattr(x, "__iter__") and not isinstance(x, str) and method in x
            )]["track_id"]
        )
        missing_segs = should_have - chroma_track_ids
        for tid in list(missing_segs)[:5]:
            has_tid = tid in df_meta_idx.index
            warns.append(Warning(
                level="CRITICAL", code="C6b",
                label="Track marked as processed but has no segments",
                method=method,
                track_id=tid,
                artist=str(df_meta_idx.loc[tid, "artist"]) if has_tid and has_artist else None,
                title=str(df_meta_idx.loc[tid, "title"])   if has_tid and has_title  else None,
                metrics={"track_id": tid[:16] + "..."},
                action="Delete via `--purge` then run `python manage.py ingest`.",
            ))

    # --- Fingerprints ---
    fp_stats = _fp_load_stats(fp_db)  # {track_id: n_hashes}

    # [Q3] Empty fingerprint (0 hashes)
    for tid in [t for t, n in fp_stats.items() if n == 0][:5]:
        has_tid = tid in df_meta_idx.index
        warns.append(Warning(
            level="QUALITY", code="Q3", label="Empty fingerprint (0 hashes)",
            method=method,
            track_id=tid,
            artist=str(df_meta_idx.loc[tid, "artist"]) if has_tid and has_artist else None,
            title=str(df_meta_idx.loc[tid, "title"])   if has_tid and has_title  else None,
            metrics={"Hashes": "0"},
            action="Re-download if audio quality is suspect (--purge)",
        ))

    # [FP] Tracks without fingerprint
    missing_fp = chroma_track_ids - set(fp_stats.keys())
    if missing_fp:
        n_total = len(chroma_track_ids)
        warns.append(Warning(
            level="QUALITY", code="FP", label="Missing fingerprints",
            method=method,
            metrics={
                "Tracks without fingerprint": (
                    f"{len(missing_fp)} / {n_total}  ({len(missing_fp)/n_total:.0%})"
                ),
            },
            action=(
                "Stage 2 (Shazam re-ranking) inoperative for these tracks.\n"
                "To recompute: use --purge-missing-fp then run `python manage.py ingest`."
            ),
        ))

    # [Q4] Poor fingerprint outlier (IQR by duration bucket)
    if has_duration:
        rows = []
        for tid, n_hashes in fp_stats.items():
            if tid not in df_meta_idx.index:
                continue
            row      = df_meta_idx.loc[tid]
            duration = float(row["duration"])
            if duration == 0:
                continue
            rows.append({
                "track_id": tid,
                "artist":   str(row["artist"]) if has_artist else "",
                "title":    str(row["title"])  if has_title  else "",
                "duration": duration,
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
                        level="QUALITY", code="Q4",
                        label="Abnormally sparse fingerprint",
                        method=method,
                        track_id=r.track_id,
                        artist=r.artist,
                        title=r.title,
                        metrics={
                            "Hashes":         str(r.n_hashes),
                            "Group median":   f"{median:.0f}",
                            "Track duration": _fmt_dur(r.duration),
                        },
                        action="Re-download if audio quality is suspect (--purge)",
                    ))

    return warns


# ---------------------------------------------------------------------------
# Per-method summary (lightweight, without full check_method)
# ---------------------------------------------------------------------------

@dataclass
class MethodSummary:
    method:       str
    n_segments:   int
    n_tracks:     int
    n_incomplete: int    # embeddings < 80%
    n_fp:         int    # fingerprints present
    n_fp_total:   int    # tracks with embeddings (fp denominator)
    n_fp_missing: int    # tracks without fingerprint
    n_fp_empty:   int    # empty fingerprints (0 hashes)
    n_fp_poor:    int    # fingerprint outliers (Q4)
    n_crit:       int    # number of critical warnings
    n_qual:       int    # number of quality warnings
    index_ok:     bool


def _method_summary(method: str, df_meta: pd.DataFrame | None) -> MethodSummary:
    """Collects summary stats for a method (fast)."""
    import pyarrow.parquet as pq

    fp_db        = ROOT / config.FINGERPRINTS_DB
    index_path   = ROOT / config.INDEX_DIR / f"index_{method}_{config.INDEX_TYPE}.faiss"
    order_path   = ROOT / config.INDEX_DIR / f"segments_{method}.parquet"
    df_meta_idx  = df_meta.drop_duplicates("track_id").set_index("track_id") if df_meta is not None else None
    has_duration = df_meta is not None and "duration" in df_meta.columns

    collection, _ = _get_chroma_collection(method)
    if collection is None:
        return MethodSummary(
            method=method, n_segments=0, n_tracks=0, n_incomplete=0,
            n_fp=0, n_fp_total=0, n_fp_missing=0, n_fp_empty=0, n_fp_poor=0,
            n_crit=1, n_qual=0, index_ok=False,
        )

    n_segments = collection.count()

    # Count tracks and incomplete embeddings.
    # Fast path: read the segments parquet (columnar, one I/O) instead of
    # paginating ChromaDB with LIMIT/OFFSET which is O(N²) in SQLite.
    n_tracks          = 0
    n_incomplete      = 0
    chroma_ids: set[str] = set()

    if n_segments > 0:
        if order_path.exists():
            try:
                import pyarrow.compute as pc
                # value_counts runs in C++ — avoids materialising 1M Python strings
                vc     = pc.value_counts(
                    pq.read_table(order_path, columns=["track_id"]).column("track_id")
                )
                track_seg_counts: dict[str, int] = dict(zip(
                    vc.field("values").to_pylist(),
                    vc.field("counts").to_pylist(),
                ))
                n_tracks   = len(track_seg_counts)
                chroma_ids = set(track_seg_counts.keys())
            except Exception:
                track_seg_counts = {}
        else:
            # Fallback: no parquet yet (index not built) — use ChromaDB
            all_data = _chroma_get_all(collection, include=["metadatas"])
            track_seg_counts = {}
            for m in all_data["metadatas"]:
                tid = m["track_id"]
                track_seg_counts[tid] = track_seg_counts.get(tid, 0) + 1
            n_tracks   = len(track_seg_counts)
            chroma_ids = set(track_seg_counts.keys())

        if has_duration and track_seg_counts:
            win_s = config.SEGMENT_WIN_S
            hop_s = config.SEGMENT_HOP_S
            for tid, actual in track_seg_counts.items():
                if df_meta_idx is None or tid not in df_meta_idx.index:
                    continue
                duration = float(df_meta_idx.loc[tid, "duration"])
                expected = int(max(0, (duration - win_s) / hop_s)) + 1
                if expected > 0 and actual / expected < 0.8:
                    n_incomplete += 1

    # Fingerprints
    fp_stats     = _fp_load_stats(fp_db)
    n_fp_total   = len(chroma_ids)
    n_fp_missing = len(chroma_ids - set(fp_stats.keys()))
    n_fp         = n_fp_total - n_fp_missing
    n_fp_empty   = sum(1 for tid in fp_stats if fp_stats[tid] == 0 and tid in chroma_ids)

    # Q4 — poor fingerprints (fast count, no detail)
    n_fp_poor = 0
    if has_duration:
        rows = []
        for tid, n_hashes in fp_stats.items():
            if tid not in chroma_ids:
                continue
            if df_meta_idx is None or tid not in df_meta_idx.index:
                continue
            duration = float(df_meta_idx.loc[tid, "duration"])
            if duration == 0:
                continue
            rows.append({"duration": duration, "n_hashes": n_hashes})
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

    # FAISS index check: read only the parquet footer (num_rows) instead of
    # loading the full index — avoids a 1-4 GB RAM spike per method.
    index_ok = False
    if index_path.exists():
        try:
            if order_path.exists():
                n_in_parquet = pq.read_metadata(order_path).num_rows
                index_ok = (n_in_parquet == n_segments)
            else:
                import faiss
                idx      = faiss.read_index(str(index_path))
                index_ok = (idx.ntotal == n_segments)
        except Exception:
            index_ok = False

    # Count critical/quality warnings (fast: use already-computed flags)
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
# Summary view (default)
# ---------------------------------------------------------------------------

def _render_summary(methods: list[str]) -> None:
    """Displays the global summary in two blocks: audio/embeddings + metadata."""
    meta_path = PROCESSED_DIR / "metadata.parquet"

    df_meta: pd.DataFrame | None = None
    n_total = 0
    if meta_path.exists():
        try:
            df_meta = pd.read_parquet(meta_path)
            n_total = len(df_meta)
        except Exception:
            pass

    # ── BLOCK 1: Audio & Embeddings ──────────────────────────────────────────
    console.print()
    console.rule("[bold cyan]Audio & Embeddings[/bold cyan]")
    console.print()

    if n_total:
        console.print(f"  [dim]Tracks in metadata.parquet:[/dim]  [bold]{n_total}[/bold]")
        console.print()

    if not methods:
        console.print("  [yellow]No ChromaDB collection detected.[/yellow]")
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
                "Covered tracks",
                f"{s.n_tracks} / {n_total}  ({_pct(s.n_tracks, n_total)})"
                if n_total else str(s.n_tracks),
            )

            if s.n_incomplete == 0:
                tbl.add_row("Embeddings",  "[green]complete[/green]")
            else:
                tbl.add_row(
                    "Embeddings",
                    f"[red]{s.n_incomplete} incomplete[/red]  "
                    f"[dim](< 80% expected segments)[/dim]",
                )

            fp_label  = f"{s.n_fp} / {s.n_fp_total}  ({_pct(s.n_fp, s.n_fp_total)})"
            fp_issues = []
            if s.n_fp_missing > 0:
                fp_issues.append(f"[red]{s.n_fp_missing} missing[/red]")
            if s.n_fp_empty > 0:
                fp_issues.append(f"[yellow]{s.n_fp_empty} empty[/yellow]")
            if s.n_fp_poor > 0:
                fp_issues.append(f"[yellow]{s.n_fp_poor} poor[/yellow]")
            fp_str = fp_label
            if fp_issues:
                fp_str += "  " + "  ".join(fp_issues)
            tbl.add_row("Fingerprints", fp_str)

            if s.n_segments > 0:
                tbl.add_row(
                    "FAISS index",
                    "[green]OK[/green]" if s.index_ok else "[red]missing or out of sync[/red]",
                )

            console.print(tbl)

            if s.n_crit > 0 or s.n_qual > 0:
                console.print(
                    "    [dim]→ [bold]--details[/bold] to see detailed issues[/dim]"
                )
            console.print()

    # ── BLOCK 2: Metadata ────────────────────────────────────────────────────
    console.rule("[bold cyan]Metadata completeness[/bold cyan]")
    console.print()

    if df_meta is None or n_total == 0:
        console.print("  [yellow]metadata.parquet not found or empty.[/yellow]")
        console.print()
        return

    for col in ITUNES_FIELDS:
        if col not in df_meta.columns:
            df_meta[col] = None

    tbl2 = Table(box=None, show_header=False, padding=(0, 2))
    tbl2.add_column("field",  style="dim", no_wrap=True)
    tbl2.add_column("count")
    tbl2.add_column("bar")

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
        "Complete (all fields)",
        f"[green]{n_complete} / {n_total}[/green]  ({_pct(n_complete, n_total)})",
    )
    tbl3.add_row(
        "Partial (≥ 1 empty field)",
        (f"[yellow]{n_partial} / {n_total}[/yellow]  ({_pct(n_partial, n_total)})"
         if n_partial else f"[green]0[/green]"),
    )
    tbl3.add_row(
        "Not found (no field)",
        (f"[red]{n_not_found} / {n_total}[/red]  ({_pct(n_not_found, n_total)})"
         if n_not_found else f"[green]0[/green]"),
    )

    console.print(tbl3)
    console.print()

    if n_partial > 0 or n_not_found > 0:
        console.print(
            "  [dim]→ [bold]--metadata[/bold] to see tracks with missing metadata[/dim]"
        )
        console.print()


# ---------------------------------------------------------------------------
# --metadata view
# ---------------------------------------------------------------------------

def _render_metadata_report() -> None:
    """Displays tracks with missing or partial metadata."""
    meta_path = PROCESSED_DIR / "metadata.parquet"

    if not meta_path.exists():
        console.print("[red]metadata.parquet not found.[/red]")
        return

    try:
        df = pd.read_parquet(meta_path)
    except Exception as e:
        console.print(f"[red]Unable to read metadata.parquet: {e}[/red]")
        return

    for col in ITUNES_FIELDS:
        if col not in df.columns:
            df[col] = None

    n_total = len(df)

    # ── Tracks not found (all fields None) ──────────────────────────────────
    console.print()
    df_none = df[df[ITUNES_FIELDS].isnull().all(axis=1)].copy()
    console.rule(
        f"[bold red]Not found on Deezer / MusicBrainz[/bold red]  "
        f"[dim]{len(df_none)} / {n_total}[/dim]"
    )
    console.print()

    if df_none.empty:
        console.print("  [green]None — all tracks have at least one populated field.[/green]")
    else:
        tbl = Table(show_header=True, header_style="bold red")
        tbl.add_column("#",       style="dim",  width=4)
        tbl.add_column("Artist",  style="bold")
        tbl.add_column("Title")

        for i, row in enumerate(df_none.itertuples(), start=1):
            tbl.add_row(
                str(i),
                str(getattr(row, "artist", "—")),
                str(getattr(row, "title",  "—")),
            )
        console.print(tbl)

    console.print()

    # ── Partial tracks (≥ 1 empty field, but not all) ────────────────────────
    mask_partial = df[ITUNES_FIELDS].isnull().any(axis=1) & ~df[ITUNES_FIELDS].isnull().all(axis=1)
    df_partial   = df[mask_partial].copy()

    console.rule(
        f"[bold yellow]Partial Metadata[/bold yellow]  "
        f"[dim]{len(df_partial)} / {n_total}[/dim]"
    )
    console.print()

    if df_partial.empty:
        console.print("  [green]None — all tracks have their fields complete.[/green]")
    else:
        tbl2 = Table(show_header=True, header_style="bold yellow")
        tbl2.add_column("#",            style="dim",    width=4)
        tbl2.add_column("Artist",      style="bold")
        tbl2.add_column("Title")
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
            "  [dim]→ Relaunch [bold]python manage.py enrich[/bold] "
            "to try to enrich the missing fields.[/dim]"
        )
        console.print()


# ---------------------------------------------------------------------------
# Warning rendering (for --details)
# ---------------------------------------------------------------------------

def _render_warning(w: Warning) -> None:
    """Displays a warning formatted with rich.Panel."""
    LEVEL_COLOR = {"CRITICAL": "red", "QUALITY": "yellow"}
    LEVEL_ICON  = {"CRITICAL": "✗", "QUALITY": "⚠"}

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
    content = "\n".join(lines) if lines else "  (no details)"

    console.print(Panel(
        content,
        title=title,
        title_align="left",
        border_style=color,
        padding=(0, 1),
    ))
    console.print("")


# ---------------------------------------------------------------------------
# --details view
# ---------------------------------------------------------------------------

def _render_details(methods: list[str]) -> list[Warning]:
    """Displays the detailed warnings by method."""
    all_warns: list[Warning] = []

    for m in sorted(methods):
        collection, _ = _get_chroma_collection(m)
        n_seg    = collection.count() if collection else "—"
        n_tracks = "—"  # computed inside check_method; pre-fetching here would duplicate the query

        n_fp = _fp_count(ROOT / config.FINGERPRINTS_DB)

        method_warns = check_method(m)
        n_crit = sum(1 for w in method_warns if w.level == "CRITICAL")
        n_qual = sum(1 for w in method_warns if w.level == "QUALITY")

        if not method_warns:
            status = "[green]✓ all OK[/green]"
        elif n_crit:
            status = (
                f"[red]{n_crit} critical issue(s)[/red]"
                + (f"  [yellow]{n_qual} quality issue(s)[/yellow]" if n_qual else "")
            )
        else:
            status = f"[yellow]{n_qual} quality issue(s)[/yellow]"

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
            console.print("  [green]No issues detected.[/green]\n")

        all_warns.extend(method_warns)

    # Final summary
    console.rule()
    n_crit = sum(1 for w in all_warns if w.level == "CRITICAL")
    n_qual = sum(1 for w in all_warns if w.level == "QUALITY")

    if not all_warns:
        console.print("[bold green]✓ All data is consistent.[/bold green]")
    elif n_crit:
        console.print(
            f"[bold red]{n_crit} critical issue(s)[/bold red]"
            + (f"  [yellow]{n_qual} quality issue(s)[/yellow]" if n_qual else "")
        )
    else:
        console.print(
            f"[yellow]{n_qual} quality issue(s)[/yellow] — usable data, "
            "but some tracks may yield poor results."
        )

    console.print()
    return all_warns


# ---------------------------------------------------------------------------
# Purge
# ---------------------------------------------------------------------------

def purge_tracks(method: str, track_ids: set[str]) -> dict:
    """
    Removes data from a set of tracks for a given method.

    Surgical behavior:
    - Segments removed from ChromaDB only for the purged method
    - The method is removed from embedded_methods in metadata.parquet
    - If embedded_methods becomes empty → the entire row is removed + fingerprint removed
    - If other methods remain → the row is kept (other methods are intact)
    - FAISS index + order parquet of the method removed (to be rebuilt)
    
    Returns a statistics dictionary:
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

    # --- FAISS index + order parquet (obsolete) ---
    for p in [index_path, order_path]:
        if p.exists():
            p.unlink()

    # --- Metadata : surgical removal of the method ---
    fully_removed: set[str] = set()
    remaining_ids: set[str] = set()

    if meta_path.exists():
        df_meta  = pd.read_parquet(meta_path)
        affected = df_meta["track_id"].isin(track_ids)

        if "embedded_methods" in df_meta.columns:
            def _remove_method(methods):
                if methods is None:
                    return None
                if isinstance(methods, (list, set, np.ndarray)):
                    updated = [
                        m for m in methods
                        if not _embedded_method_matches_collection(m, method)
                    ]
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
        # Reuse the already-updated in-memory DataFrame instead of re-reading from disk
        remaining_ids = set(df_meta["track_id"].unique())

    fully_removed |= (track_ids - remaining_ids)

    # --- Fingerprints (SQLite) ---
    if fully_removed:
        stats["fingerprints_removed"] = fp_delete(fp_db, fully_removed)

    return stats


def _run_purge(by_method: dict[str, set[str]], yes: bool) -> None:
    """Display the recap, ask for confirmation, and purge."""
    console.print("")
    console.rule("[bold red]Recapitulative of the purge[/bold red]")
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
                    f"  [dim red]→ completely removed[/dim red]"
                )
            else:
                console.print(
                    f"  [yellow]↺[/yellow]  [dim cyan]({m})[/dim cyan]  {label}"
                    f"  [dim yellow]→ method removed, other methods preserved[/dim yellow]"
                )

    total_tracks = sum(len(v) for v in by_method.values())
    console.print("")
    console.print(
        f"[bold]{total_tracks} track(s)[/bold] will be purged for the selected method."
    )
    console.print("[dim]• ChromaDB segments removed for the purged method only.[/dim]")
    console.print("[dim]• If the track has no more active methods → metadata and fingerprint rows will be deleted.[/dim]")
    console.print("[dim]• The FAISS index for the method will be deleted — run `python manage.py rebuild --what index` after.[/dim]")

    if not yes:
        console.print("")
        confirm = console.input(
            "[bold yellow]Continue ?[/bold yellow] [dim](y = confirm / Enter = cancel) [/dim]"
        ).strip().lower()
        if confirm not in ("y", "yes"):
            console.print("[dim]Canceled.[/dim]")
            return

    console.print("")
    total_segs = 0
    for m, tids in sorted(by_method.items()):
        console.print(f"  Purge method [cyan]{m}[/cyan]…", end=" ")
        stats = purge_tracks(m, tids)
        total_segs += stats["segments_removed"]
        parts = [f"{stats['segments_removed']} segments removed"]
        if stats["tracks_updated"] > 0:
            parts.append(f"{stats['tracks_updated']} track(s) updated (method removed)")
        if stats["tracks_removed"] > 0:
            parts.append(f"{stats['tracks_removed']} track(s) completely removed")
        if stats["fingerprints_removed"] > 0:
            parts.append(f"{stats['fingerprints_removed']} fingerprint(s) removed")
        console.print(f"[green]✓[/green]  " + "  •  ".join(parts))

    console.print("")
    console.print(
        f"[bold green]Purge completed.[/bold green]  "
        f"{total_tracks} track(s) processed, {total_segs} segments removed."
    )
    console.print("\n[dim]To re-download and rebuild the index :[/dim]")
    console.print("  [bold cyan]python manage.py ingest[/bold cyan]")
    console.print("  [bold cyan]python manage.py rebuild --what index[/bold cyan]")


def _run_purge_missing_fp(methods: list[str], yes: bool) -> None:
    """Purge tracks that have embeddings but no fingerprint."""
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
        console.print("[green]No track without a fingerprint — nothing to purge.[/green]")
        return

    total = sum(len(v) for v in by_method.values())
    console.print("")
    console.rule("[bold yellow]Purge tracks without fingerprinting[/bold yellow]")
    console.print(f"\n  [bold]{total} track(s)[/bold] without fingerprints will be deleted.")
    console.print("  [dim]They will be re-downloaded and re-fingerprinted at the next run.[/dim]\n")

    if not yes:
        confirm = console.input(
            "[bold yellow]Continue ?[/bold yellow] [dim](y = confirm / Enter = cancel) [/dim]"
        ).strip().lower()
        if confirm not in ("y", "yes"):
            console.print("[dim]Canceled.[/dim]")
            return

    _run_purge(by_method, yes=True)


# ---------------------------------------------------------------------------
# Public entry point
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
    Checks the consistency of the Shazam Home project data.

    Args:
        method:           method to check (default: all detected).
        details:          displays the details of each problem.
        metadata:         displays tracks with missing metadata.
        purge:            removes problematic tracks.
        purge_missing_fp: purges only tracks without fingerprints.
        yes:              does not ask for confirmation before purging.
    """
    # Detect the methods available from ChromaDB
    if method:
        methods = [method]
    else:
        try:
            client  = chromadb.PersistentClient(path=str(ROOT / config.CHROMA_DIR))
            methods = [c.name for c in client.list_collections()]
        except Exception:
            methods = []

    # ── --metadata mode ────────────────────────────────────────────────────
    if metadata:
        _render_metadata_report()
        return

    # ── --details mode (+ eventually --purge) ──────────────────────────
    if details or purge or purge_missing_fp:
        if not methods:
            console.print(
                f"[yellow]No ChromaDB collection found in {config.CHROMA_DIR}[/yellow]"
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
                    "\n[yellow]No individual tracks to purge "
                    "(Alerts are global — see suggested actions).[/yellow]"
                )
                return

            _run_purge(by_method, yes)
        elif not details:
            console.print(
                "\n[dim]Tip: run with [bold]--purge[/bold] to automatically delete "
                "problematic tracks.[/dim]"
            )
        return

    # ── Summary view (default) ────────────────────────────────────────────────
    _render_summary(methods)
