from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src.maintenance import clean
from src.utils.fingerprints_db import fp_init, fp_save


class FakeCollection:
    def __init__(self, name: str, ids: list[str]) -> None:
        self.name = name
        self.ids = ids
        self.deleted_ids: list[list[str]] = []

    def get(self, where=None, include=None):
        return {"ids": list(self.ids)}

    def delete(self, ids):
        self.deleted_ids.append(list(ids))
        self.ids = []


class FakeClient:
    def __init__(self, collections):
        self._collections = collections

    def list_collections(self):
        return self._collections


class CleanTests(unittest.TestCase):
    def test_get_size_handles_file_and_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            file_path = root / "file.bin"
            file_path.write_bytes(b"a" * 10)
            folder = root / "folder"
            folder.mkdir()
            (folder / "nested.bin").write_bytes(b"b" * 20)

            self.assertIn("o", clean._get_size(file_path))
            self.assertIn("o", clean._get_size(folder))

    def test_run_clean_deletes_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            file_path = root / "data.txt"
            dir_path = root / "data_dir"
            file_path.write_text("demo", encoding="utf-8")
            dir_path.mkdir()
            (dir_path / "nested.txt").write_text("demo", encoding="utf-8")

            targets = [
                (dir_path, "Dir", "folder"),
                (file_path, "File", "file"),
            ]
            with patch.object(clean, "TARGETS", targets), patch.object(clean, "ROOT", root):
                clean.run_clean(yes=True)

            self.assertFalse(file_path.exists())
            self.assertFalse(dir_path.exists())

    def test_run_clean_track_removes_segments_fingerprint_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            chroma_dir = root / "chroma"
            chroma_dir.mkdir()
            metadata_path = root / "metadata.parquet"
            fp_db_path = root / "fingerprints.db"

            pd.DataFrame([{"track_id": "track_1", "title": "Song", "artist": "Artist"}]).to_parquet(metadata_path, index=False)
            fp_init(fp_db_path)
            fp_save(fp_db_path, "track_1", {(1, 2, 3, 4)})

            collection = FakeCollection("demo", ["seg_1", "seg_2"])
            fake_client = FakeClient([collection])

            with patch.object(clean, "ROOT", root), \
                 patch.object(clean.config, "METADATA_PATH", metadata_path.name), \
                 patch.object(clean.config, "FINGERPRINTS_DB", fp_db_path.name), \
                 patch.object(clean.config, "CHROMA_DIR", chroma_dir.name), \
                 patch("chromadb.PersistentClient", return_value=fake_client):
                clean.run_clean_track("track_1", yes=True)

            df = pd.read_parquet(metadata_path)
            with sqlite3.connect(fp_db_path) as conn:
                count = conn.execute("SELECT COUNT(*) FROM fingerprints").fetchone()[0]

        self.assertEqual(collection.deleted_ids, [["seg_1", "seg_2"]])
        self.assertTrue(df.empty)
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
