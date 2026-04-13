"""
src/utils/fingerprints_db.py

Gestion de la base SQLite des fingerprints audio.
Partagé entre download_music.py, rebuild_fingerprints.py et check_data.py.

Schéma :
    fingerprints(track_id TEXT PRIMARY KEY, hashes BLOB NOT NULL, n_hashes INTEGER NOT NULL)
"""

from __future__ import annotations

import contextlib
import pickle
import sqlite3
import threading
import time
from pathlib import Path

# Verrou global pour les écritures concurrentes (ThreadPoolExecutor)
_db_lock = threading.Lock()

# Timeout SQLite (secondes) — délai d'attente si la DB est verrouillée par une autre connexion
_SQLITE_TIMEOUT = 30


@contextlib.contextmanager
def _connect(db_path: Path, timeout: float = _SQLITE_TIMEOUT):
    """
    Context manager qui ouvre une connexion SQLite, commit ou rollback,
    et ferme explicitement la connexion à la sortie.

    Garantit qu'aucun verrou n'est laissé actif après le bloc.
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
# Initialisation
# ---------------------------------------------------------------------------

def fp_init(db_path: Path) -> None:
    """Crée la table fingerprints si elle n'existe pas encore, active le mode WAL."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        # WAL = Write-Ahead Logging : meilleure gestion de la concurrence,
        # les lecteurs ne bloquent pas les écrivains et vice-versa.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fingerprints (
                track_id TEXT PRIMARY KEY,
                hashes   BLOB    NOT NULL,
                n_hashes INTEGER NOT NULL
            )
        """)


# ---------------------------------------------------------------------------
# Lecture
# ---------------------------------------------------------------------------

def fp_load_ids(db_path: Path) -> set[str]:
    """Retourne l'ensemble des track_ids qui ont déjà un fingerprint (n_hashes > 0)."""
    db_path = Path(db_path)
    if not db_path.exists():
        return set()
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT track_id FROM fingerprints WHERE n_hashes > 0"
        ).fetchall()
    return {r[0] for r in rows}


def fp_load_all(db_path: Path) -> dict[str, set]:
    """Charge tous les fingerprints → {track_id: set_of_hashes}. Peut être lent sur grande base."""
    db_path = Path(db_path)
    if not db_path.exists():
        return {}
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT track_id, hashes FROM fingerprints").fetchall()
    return {r[0]: pickle.loads(r[1]) for r in rows}


def fp_load_stats(db_path: Path) -> dict[str, int]:
    """Charge {track_id: n_hashes} sans désérialiser les blobs (lecture rapide)."""
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
    Détecte le format des fingerprints en désérialisant un seul blob.

    Retourne :
        'v1' — hashes 3-tuples (sans ancre temporelle)
        'v2' — hashes 4-tuples (avec ancre temporelle t1)
        'unknown' — base vide ou erreur
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
# Écriture
# ---------------------------------------------------------------------------

def fp_save(db_path: Path, track_id: str, hashes: set, thread_safe: bool = False) -> None:
    """
    Insère ou remplace le fingerprint d'un track.

    Args:
        db_path:      chemin vers fingerprints.db.
        track_id:     identifiant du track.
        hashes:       set de hashes (3-tuples v1 ou 4-tuples v2).
        thread_safe:  si True, utilise le verrou global (ThreadPoolExecutor).
    """
    db_path = Path(db_path)

    def _write(retries: int = 5, delay: float = 1.0):
        """Écrit dans la DB avec retry automatique si database is locked."""
        for attempt in range(retries):
            try:
                with _connect(db_path) as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO fingerprints VALUES (?, ?, ?)",
                        (track_id, pickle.dumps(hashes), len(hashes)),
                    )
                return  # succès
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < retries - 1:
                    time.sleep(delay * (attempt + 1))  # backoff progressif
                else:
                    raise

    if thread_safe:
        with _db_lock:
            _write()
    else:
        _write()


def fp_delete(db_path: Path, track_ids: set[str]) -> int:
    """
    Supprime les fingerprints d'un ensemble de tracks.

    Retourne le nombre de lignes supprimées.
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
# Migration depuis l'ancien format pickle
# ---------------------------------------------------------------------------

def fp_migrate_from_pkl(pkl_path: Path, db_path: Path) -> int:
    """
    Migration one-shot : importe fingerprints.pkl dans fingerprints.db.
    Appelée automatiquement au démarrage si le .pkl existe et le .db non.

    Retourne le nombre de fingerprints migrés.
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
