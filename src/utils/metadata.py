"""
src/utils/metadata.py

Écriture atomique de fichiers de données (parquet, pickle).
Garantit qu'un crash pendant l'écriture ne corrompt pas le fichier existant.
"""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path


def atomic_write_parquet(path: Path, df) -> None:
    """
    Écrit un DataFrame pandas en parquet de manière atomique.

    Utilise un fichier temporaire + os.replace (atomique sur tous les OS).
    En cas de crash pendant l'écriture, l'ancien fichier reste intact.

    Args:
        path: chemin de destination du fichier .parquet.
        df:   DataFrame pandas à écrire.
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
    Écrit un objet Python en pickle de manière atomique.

    Args:
        path: chemin de destination du fichier .pkl.
        obj:  objet Python à sérialiser.
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
