from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.index import build_index as build_index_module


class FakeCollection:
    def __init__(self, embeddings: list[list[float]], metadatas: list[dict]):
        self._embeddings = embeddings
        self._metadatas = metadatas

    def count(self) -> int:
        return len(self._embeddings)

    def get(self, include: list[str], limit: int, offset: int) -> dict:
        end = offset + limit
        embeddings = self._embeddings[offset:end]
        metadatas = self._metadatas[offset:end]
        return {
            "ids": [str(index) for index in range(offset, offset + len(embeddings))],
            "embeddings": embeddings,
            "metadatas": metadatas,
        }


class FakeChromaClient:
    def __init__(self, collection: FakeCollection):
        self.collection = collection

    def get_collection(self, name: str) -> FakeCollection:
        return self.collection


class BuildIndexTests(unittest.TestCase):
    def test_build_index_creates_searchable_flat_index(self) -> None:
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        index = build_index_module.build_index(embeddings, index_type="flat")
        distances, indices = index.search(np.array([[1.0, 0.0]], dtype=np.float32), 1)

        self.assertEqual(index.ntotal, 2)
        self.assertEqual(int(indices[0][0]), 0)
        self.assertGreaterEqual(float(distances[0][0]), 0.99)

    def test_build_index_rejects_unknown_type(self) -> None:
        with self.assertRaises(ValueError):
            build_index_module.build_index(np.array([[1.0, 0.0]], dtype=np.float32), index_type="mystery")

    def test_save_and_load_index_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "index.faiss"
            index = build_index_module.build_index(np.array([[1.0, 0.0]], dtype=np.float32), index_type="flat")
            build_index_module.save_index(index, path)
            loaded = build_index_module.load_index(path)

        self.assertEqual(loaded.ntotal, 1)

    def test_build_for_method_writes_index_and_segments_order(self) -> None:
        embeddings = [[1.0, 0.0], [0.0, 1.0]]
        metadatas = [{"track_id": "track_1", "start_s": 0.0}, {"track_id": "track_2", "start_s": 5.0}]
        client = FakeChromaClient(FakeCollection(embeddings, metadatas))

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch.object(build_index_module, "ROOT", root), \
                 patch("src.config.INDEX_DIR", "index"):
                build_index_module._build_for_method("mfcc", "flat", client)

            segments_path = root / "index" / "segments_mfcc.parquet"
            index_path = root / "index" / "index_mfcc_flat.faiss"

            self.assertTrue(segments_path.exists())
            self.assertTrue(index_path.exists())
            df = pd.read_parquet(segments_path)

        self.assertEqual(df["track_id"].tolist(), ["track_1", "track_2"])


if __name__ == "__main__":
    unittest.main()
