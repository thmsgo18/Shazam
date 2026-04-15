"""
src/utils/metadata.py


Atomic writing of data files (parquet, pickle).
Ensures that a crash during writing does not corrupt the existing file.
"""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path


def atomic_write_parquet(path: Path, df) -> None:
    """
    Atomically writes a pandas DataFrame to parquet.


    Uses a temporary file + os.replace (atomic on all OS).
    In case of a crash during writing, the old file remains intact.


    Args:
        path: destination path of the .parquet file.
        df:   pandas DataFrame to write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(tmp_fd)
    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

def atomic_write_pickle(path: Path, obj: object) -> None:
    """
    Atomically writes a Python object to pickle.


    Args:
        path: destination path of the .pkl file.
        obj:  Python object to serialize.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            pickle.dump(obj, f)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise