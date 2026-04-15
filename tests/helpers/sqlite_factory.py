from __future__ import annotations

from pathlib import Path

from src.utils.fingerprints_db import fp_init, fp_save


def write_fingerprint_db(path: str | Path, fingerprints: dict[str, set[tuple]]) -> Path:
    db_path = Path(path)
    fp_init(db_path)
    for track_id, hashes in fingerprints.items():
        fp_save(db_path, track_id, hashes)
    return db_path
