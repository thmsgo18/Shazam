"""
src/index/build_index.py

Construction and management of the FAISS index.
Supports flat, HNSW, IVF, and IVF+SQ8 (ivfsq).

─── Memory & time strategy ───────────────────────────────────────────────────

Problem 1 — FAISS AlignedTable reallocation:
  Every index.add(batch) triggers alloc(new) + copy(all) + free(old).
  At the end with ntotal ≈ 1.1 M: peak ≈ 2 × 2.28 GB = 4.56 GB.
  Fix: single index.add() on an empty index → alloc once, copy once, free nothing.

Problem 2 — ChromaDB SQLite O(N²) scan:
  collection.get(limit=500, offset=k) scans from row 0 each time.
  For 1.1 M segments / PAGE 500 = 2 200 pages → ~1.28 billion row scans → 43 min.
  collection.get(limit=N) generates WHERE id IN (v1…vN) → SQLite 999-var limit crash.
  Fix: permanent embedding cache (embeddings_{key}.bin) on disk.

Permanent cache strategy:
  On the FIRST build: ChromaDB fill is unavoidable (O(N²), ~43 min).
    Normalized embeddings are written page-by-page to data/index/embeddings_{key}.bin.
    File is kept permanently (raw float32, size = n × dim × 4 bytes).
  On SUBSEQUENT rebuilds (n unchanged): cache is valid — ChromaDB is skipped entirely.
    index.add(cache_mmap) is the only I/O: ~30 seconds instead of 43 minutes.
  When n changes (new segments added / tracks deleted): cache is invalid,
    full rebuild runs automatically, cache is overwritten.

Disk cost: n × dim × 4 bytes alongside the FAISS index (~2.3 GB for 1.1 M × 512-dim).
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]

# Max segments per ChromaDB page.
# Must stay < 999 (SQLite variable limit for IN-clause when fetching by ID).
_PAGE = 500


# ---------------------------------------------------------------------------
# Public helpers (kept for external callers / backward compatibility)
# ---------------------------------------------------------------------------

def build_index(embeddings: np.ndarray, index_type: str = "flat") -> faiss.Index:
    """
    Builds a FAISS index from an in-memory embedding matrix.

    Kept for backward compatibility and small-scale use.
    For large collections prefer _build_for_method() which uses a permanent cache.

    Args:
        embeddings: matrix of shape (N, D) in float32.
        index_type: "flat", "hnsw", "ivf", or "ivfsq".

    Returns:
        Trained and populated FAISS index.
    """
    xb = embeddings if embeddings.dtype == np.float32 else embeddings.astype("float32")
    faiss.normalize_L2(xb)
    d, N = xb.shape[1], len(xb)
    index = _make_index(index_type, d, N)
    if not index.is_trained:
        index.train(xb)
    index.add(xb)
    return index


def save_index(index: faiss.Index, path: Path) -> None:
    """Saves a FAISS index to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    print(f"Index saved successfully at: {path}")


def load_index(path: Path) -> faiss.Index:
    """Loads a FAISS index from disk."""
    if not path.exists():
        raise FileNotFoundError(f"The index doesn't exist at: {path}")
    return faiss.read_index(str(path))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_index(index_type: str, dim: int, n: int) -> faiss.Index:
    """
    Creates an untrained FAISS index of the requested type.

    RAM once fully populated:
      flat  — exact cosine, N×D×4 bytes  (~2.28 GB for 1.1 M×512).
      hnsw  — approximate, graph-based.  Similar to flat + graph edges.
      ivf   — approximate, inverted file. Same storage as flat.
      ivfsq — IVF + 8-bit scalar quantisation. N×D×1 byte ≈ 4× less (~570 MB).
               Accuracy loss is negligible when fingerprint re-ranking is used.
    """
    index_type = index_type.lower()
    nlist      = max(1, int(n ** 0.5))

    if index_type == "flat":
        return faiss.IndexFlatIP(dim)

    if index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efConstruction = 40
        return index

    if index_type == "ivf":
        quantizer = faiss.IndexFlatIP(dim)
        return faiss.IndexIVFFlat(
            quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT
        )

    if index_type == "ivfsq":
        quantizer = faiss.IndexFlatIP(dim)
        return faiss.IndexIVFScalarQuantizer(
            quantizer, dim, nlist,
            faiss.ScalarQuantizer.QT_8bit,
            faiss.METRIC_INNER_PRODUCT,
        )

    raise ValueError(
        f"Index type unknown: '{index_type}'. "
        f"Options: flat, hnsw, ivf, ivfsq"
    )


# ---------------------------------------------------------------------------
# Main build routine
# ---------------------------------------------------------------------------

def _build_for_method(collection_key: str, index_type: str, chroma_client) -> None:
    """
    Builds and saves the FAISS index for a given ChromaDB collection.

    See module docstring for full memory & time strategy.

    Cache hit path (fast, ~30 s):
      embeddings_{key}.bin exists + size == n × dim × 4
      segments_{key}.parquet exists + row count == n
      → skip ChromaDB entirely, index.add(cache_mmap) only.

    Cache miss path (slow, ~43 min for 1.1 M segments):
      Full ChromaDB pagination fill + stream parquet + write cache.
      After this run, subsequent rebuilds will use the cache.
    """
    import src.config as config

    index_type  = index_type.lower()
    index_dir   = ROOT / config.INDEX_DIR
    out_path    = index_dir / f"index_{collection_key}_{index_type}.faiss"
    order_path  = index_dir / f"segments_{collection_key}.parquet"
    cache_path  = index_dir / f"embeddings_{collection_key}.bin"

    try:
        collection = chroma_client.get_collection(name=collection_key)
    except Exception:
        print(f"[build_index] Collection '{collection_key}' not found in ChromaDB — ignored.")
        return

    n = collection.count()
    if n == 0:
        print(f"[build_index] Collection '{collection_key}' is empty — ignored.")
        return

    print(f"\n[build_index] ── Collection: {collection_key} | {n:,} segments ──")

    # Peek at embedding dim (1 vector, almost free)
    peek = collection.get(limit=1, offset=0, include=["embeddings"])
    dim  = len(peek["embeddings"][0])
    del peek
    print(f"[build_index] dim={dim}, index_type={index_type}")

    # ── Cache validity check ──────────────────────────────────────────────
    expected_bytes = n * dim * 4
    parquet_ok     = (
        order_path.exists()
        and pq.read_metadata(str(order_path)).num_rows == n
    )
    cache_ok = (
        cache_path.exists()
        and cache_path.stat().st_size == expected_bytes
        and parquet_ok
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_ok:
        # ── FAST PATH: load from cache (skip ChromaDB entirely) ──────────
        print(
            f"[build_index] Cache hit — {cache_path.name} "
            f"({expected_bytes / 1e9:.2f} GB, skipping ChromaDB fill)"
        )
        emb_mmap   = np.memmap(str(cache_path), dtype="float32", mode="r", shape=(n, dim))
        fill_done  = True

    else:
        # ── SLOW PATH: fill cache from ChromaDB (first build or n changed) ─
        if cache_path.exists():
            print(
                f"[build_index] Cache stale (size mismatch or parquet mismatch) — "
                f"rebuilding from ChromaDB…"
            )
        else:
            print(
                f"[build_index] No cache — filling from ChromaDB "
                f"(first build, ~{expected_bytes / 1e9:.2f} GB disk needed)…"
            )

        try:
            emb_mmap = np.memmap(
                str(cache_path), dtype="float32", mode="w+", shape=(n, dim)
            )
        except OSError as exc:
            raise RuntimeError(
                f"Cannot create embedding cache ({exc}). "
                f"Need {expected_bytes / 1e9:.1f} GB free in {index_dir}."
            ) from exc

        try:
            from rich.progress import (
                BarColumn, MofNCompleteColumn, Progress,
                SpinnerColumn, TaskProgressColumn, TimeElapsedColumn,
            )
            _rich = True
        except ImportError:
            _rich = False

        pq_writer:    pq.ParquetWriter | None = None
        pq_col_order: list[str] | None        = None
        fill_idx = 0
        offset   = 0

        def _fill(prog=None, task=None) -> None:
            nonlocal pq_writer, pq_col_order, fill_idx, offset
            while True:
                page = collection.get(
                    limit=_PAGE, offset=offset,
                    include=["embeddings", "metadatas"],
                )
                if not page["ids"]:
                    break

                # Stream metadata → parquet (one row-group per page).
                # Sort columns alphabetically: ChromaDB dict key order is not
                # guaranteed across pages and would cause schema mismatch.
                table = pa.Table.from_pandas(pd.DataFrame(page["metadatas"]))
                if pq_writer is None:
                    pq_col_order = sorted(table.schema.names)
                    table        = table.select(pq_col_order)
                    pq_writer    = pq.ParquetWriter(str(order_path), table.schema)
                else:
                    table = table.select(pq_col_order)
                pq_writer.write_table(table)

                # Normalize and write to cache (1 MB in RAM, freed each iteration)
                xb      = np.array(page["embeddings"], dtype="float32")
                faiss.normalize_L2(xb)
                batch_n = len(page["ids"])
                emb_mmap[fill_idx : fill_idx + batch_n] = xb
                fill_idx += batch_n

                if prog is not None:
                    prog.advance(task, batch_n)
                if batch_n < _PAGE:
                    break
                offset += _PAGE

        try:
            if _rich:
                with Progress(
                    SpinnerColumn(),
                    "[progress.description]{task.description}",
                    BarColumn(),
                    TaskProgressColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                ) as prog:
                    task = prog.add_task(
                        f"Filling cache {collection_key}…", total=n
                    )
                    _fill(prog, task)
            else:
                _fill()
        finally:
            if pq_writer is not None:
                pq_writer.close()

        emb_mmap.flush()   # ensure all pages are written to disk
        print(f"[build_index] Cache saved    → {cache_path}")
        print(f"[build_index] Segments saved → {order_path}")
        fill_done = True

    # ── IVF pre-training on first N vectors from cache ───────────────────
    index = _make_index(index_type, dim, n)

    if index_type in ("ivf", "ivfsq"):
        nlist      = getattr(index, "nlist", 1)
        train_size = min(n, max(nlist * 64, 10_000))
        print(f"[build_index] IVF training on {train_size:,} cached vectors…")
        xt = np.array(emb_mmap[:train_size])   # read training slice into RAM
        index.train(xt)
        del xt
        print("[build_index] Training complete.")

    # ── Single FAISS add() from the cache ────────────────────────────────
    # Empty index → AlignedTable allocates exactly n × dim × 4 bytes once.
    # No prior data to copy → no doubling. OS pages the cache in/out under
    # memory pressure, so effective peak ≈ FAISS index size + small working set.
    gb = expected_bytes / 1e9
    print(f"[build_index] FAISS add — single allocation ({gb:.2f} GB)…")
    index.add(emb_mmap)
    del emb_mmap   # release memory mapping; file stays on disk for next rebuild

    save_index(index, out_path)
    assert index.ntotal == n, (
        f"[build_index] Mismatch: {index.ntotal} vectors in index vs {n} expected"
    )
    print(f"[build_index] ✓ {index.ntotal:,} vectors indexed → {out_path}")


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import chromadb
    import src.config as config

    parser = argparse.ArgumentParser(description="Builds the FAISS index from ChromaDB.")
    parser.add_argument(
        "--method",
        default=None,
        help=(
            "Embedding method to index (mfcc, clap, muq, mert). "
            "If not specified, all available collections are indexed."
        ),
    )
    args = parser.parse_args()

    index_type    = getattr(config, "INDEX_TYPE", "flat")
    chroma_client = chromadb.PersistentClient(path=str(ROOT / config.CHROMA_DIR))

    if args.method:
        collection_keys = [config.get_collection_key(args.method)]
        print(f"[build_index] Collection key resolved: {collection_keys[0]}")
    else:
        collection_keys = [c.name for c in chroma_client.list_collections()]
        if not collection_keys:
            print("[build_index] No collections found in ChromaDB.")
            print("[build_index] Run first: python manage.py ingest")
            sys.exit(1)
        print(f"[build_index] Available collections: {collection_keys}")

    for key in collection_keys:
        _build_for_method(key, index_type, chroma_client)

    print("\n[build_index] Done.")
