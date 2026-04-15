from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.maintenance import delete_rir


class FakeCollection:
    def __init__(self, pages: list[list[str]]) -> None:
        self.pages = pages
        self.deleted_batches: list[list[str]] = []

    def get(self, limit=1000, offset=0, include=None):
        page_index = offset // limit
        ids = self.pages[page_index] if page_index < len(self.pages) else []
        return {"ids": ids}

    def delete(self, ids):
        self.deleted_batches.append(list(ids))


class DeleteRirTests(unittest.TestCase):
    def test_scan_rir_ids_filters_only_rir_vectors(self) -> None:
        collection = FakeCollection([["track_1_rir_room_a_0", "track_1_0", "track_2_rir_room_b_1"]])
        ids = delete_rir._scan_rir_ids(collection)
        self.assertEqual(ids, ["track_1_rir_room_a_0", "track_2_rir_room_b_1"])

    def test_delete_from_chroma_supports_dry_run_and_batches(self) -> None:
        collection = FakeCollection([])
        ids = [f"id_{i}" for i in range(delete_rir.BATCH_DELETE + 1)]
        self.assertEqual(delete_rir._delete_from_chroma(collection, ids, dry_run=True), len(ids))
        deleted = delete_rir._delete_from_chroma(collection, ids, dry_run=False)
        self.assertEqual(deleted, len(ids))
        self.assertEqual(len(collection.deleted_batches), 2)

    def test_clear_metadata_removes_collection_key_and_empty_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_path = Path(tmpdir) / "metadata.parquet"
            pd.DataFrame(
                [
                    {"track_id": "track_1", "rir_augmented": {"clap_demo": ["room_a"]}},
                    {"track_id": "track_2", "rir_augmented": None},
                ]
            ).to_parquet(meta_path, index=False)

            updated = delete_rir._clear_metadata(meta_path, "clap_demo", dry_run=False)
            df = pd.read_parquet(meta_path)
            self.assertEqual(updated, 1)
            self.assertNotIn("rir_augmented", df.columns)


if __name__ == "__main__":
    unittest.main()
