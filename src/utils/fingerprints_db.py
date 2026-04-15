"""
src/utils/fingerprints_db.py

Management of the SQLite database for audio fingerprints.
Shared between ingestion, fingerprint reconstruction, and the checker.

Schema:
    fingerprints(track_id TEXT PRIMARY KEY, hashes BLOB NOT NULL, n_hashes INTEGER NOT NULL)
"""

from __future__ import annotations

import contextlib
import pickle
import sqlite3
import threading
import time
from pathlib import Path

# Global lock for concurrent writes (ThreadPoolExecutor)
_db_lock = threading.Lock()

# SQLite timeout (seconds) — wait time if the DB is locked by another connection
_SQLITE_TIMEOUT = 30


@contextlib.contextmanager
def _connect(db_path: Path, timeout: float = _SQLITE_TIMEOUT):
    """
    Context manager that opens an SQLite connection, commits or rollbacks,
    and explicitly closes the connection on exit.

    Ensures no lock is left active after the block.
    """
    conn = sqlite3.connect(str(db_path), timeout=timeout)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def fp_init(db_path: Path) -> None:
    """Creates the fingerprints table if it doesn't exist yet, enables WAL mode."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        # WAL = Write-Ahead Logging: better concurrency management,
        # readers don't block writers and vice versa.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fingerprints (
                track_id TEXT PRIMARY KEY,
                hashes   BLOB    NOT NULL,
                n_hashes INTEGER NOT NULL
            )
        """)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def fp_load_ids(db_path: Path) -> set[str]:
    """Returns the set of track_ids that already have a fingerprint (n_hashes > 0)."""
    db_path = Path(db_path)
    if not db_path.exists():
        return set()
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT track_id FROM fingerprints WHERE n_hashes > 0"
        ).fetchall()
    return {r[0] for r in rows}


def fp_load_all(db_path: Path) -> dict[str, set]:
    """Loads all fingerprints → {track_id: set_of_hashes}. Can be slow on a large database."""
    db_path = Path(db_path)
    if not db_path.exists():
        return {}
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT track_id, hashes FROM fingerprints").fetchall()
    return {r[0]: pickle.loads(r[1]) for r in rows}


def fp_load_stats(db_path: Path) -> dict[str, int]:
    """Loads {track_id: n_hashes} without deserializing the blobs (fast read)."""
    db_path = Path(db_path)
    if not db_path.exists():
        return {}
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT track_id, n_hashes FROM fingerprints"
        ).fetchall()
    return {r[0]: r[1] for r in rows}


def fp_detect_format(db_path: Path) -> str:
    """
    Detects the fingerprint format by deserializing a single blob.

    Returns:
        'v1' — 3-tuple hashes (without temporal anchor)
        'v2' — 4-tuple hashes (with temporal anchor t1)
        'unknown' — empty database or error
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return "unknown"
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT hashes FROM fingerprints WHERE n_hashes > 0 LIMIT 1"
        ).fetchone()
    if row is None:
        return "unknown"
    try:
        fp = pickle.loads(row[0])
        sample = next(iter(fp))
        return "v2" if len(sample) == 4 else "v1"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def fp_save(db_path: Path, track_id: str, hashes: set, thread_safe: bool = False) -> None:
    """
    Inserts or replaces the fingerprint of a track.

    Args:
        db_path:      path to fingerprints.db.
        track_id:     track identifier.
        hashes:       set of hashes (v1 3-tuples or v2 4-tuples).
        thread_safe:  if True, uses the global lock (ThreadPoolExecutor).
    """
    db_path = Path(db_path)

    def _write(retries: int = 5, delay: float = 1.0):
        """Writes to the DB with automatic retry if database is locked."""
        for attempt in range(retries):
            try:
                with _connect(db_path) as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO fingerprints VALUES (?, ?, ?)",
                        (track_id, pickle.dumps(hashes), len(hashes)),
                    )
                return  # success
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < retries - 1:
                    time.sleep(delay * (attempt + 1))  # progressive backoff
                else:
                    raise

    if thread_safe:
        with _db_lock:
            _write()
    else:
        _write()


def fp_delete(db_path: Path, track_ids: set[str]) -> int:
    """
    Deletes the fingerprints of a set of tracks.

    Returns the number of deleted rows.
    """
    db_path = Path(db_path)
    if not db_path.exists() or not track_ids:
        return 0
    with _connect(db_path) as conn:
        placeholders = ",".join("?" * len(track_ids))
        cur = conn.execute(
            f"DELETE FROM fingerprints WHERE track_id IN ({placeholders})",
            list(track_ids),
        )
    return cur.rowcount


# ---------------------------------------------------------------------------
# Migration from old pickle format
# ---------------------------------------------------------------------------

def fp_migrate_from_pkl(pkl_path: Path, db_path: Path) -> int:
    """
    One-shot migration: imports fingerprints.pkl into fingerprints.db.
    Called automatically at startup if .pkl exists and .db does not.

    Returns the number of migrated fingerprints.
    """
    pkl_path = Path(pkl_path)
    db_path  = Path(db_path)
    try:
        with open(pkl_path, "rb") as f:
            old = pickle.load(f)
    except Exception:
        return 0
    fp_init(db_path)
    with _connect(db_path) as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO fingerprints VALUES (?, ?, ?)",
            [(tid, pickle.dumps(fp), len(fp)) for tid, fp in old.items()],
        )
    return len(old)