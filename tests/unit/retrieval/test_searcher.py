from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np
import pandas as pd

from src.retrieval import searcher


class SearcherTests(unittest.TestCase):
    def setUp(self) -> None:
        searcher.clear_searcher_cache()

    def tearDown(self) -> None:
        searcher.clear_searcher_cache()

    def test_aggregate_by_track_sums_scores_and_skips_missing_indices(self) -> None:
        segments = pd.DataFrame(
            [
                {"track_id": "track_a"},
                {"track_id": "track_b"},
                {"track_id": "track_a"},
            ]
        )
        indices = np.array([0, 1, -1, 2])
        distances = np.array([0.4, 0.2, 0.9, 0.6], dtype=np.float32)

        aggregated = searcher.aggregate_by_track(indices, distances, segments)

        self.assertEqual([track_id for track_id, _ in aggregated], ["track_a", "track_b"])
        self.assertAlmostEqual(aggregated[0][1], 1.0, places=6)
        self.assertAlmostEqual(aggregated[1][1], 0.2, places=6)

    def test_load_searcher_reads_files_and_uses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            index_dir = root / "index"
            index_dir.mkdir()

            key = "mfcc"
            index_path = index_dir / f"index_{key}_flat.faiss"
            segments_path = index_dir / f"segments_{key}.parquet"

            index = faiss.IndexFlatIP(2)
            index.add(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
            faiss.write_index(index, str(index_path))
            pd.DataFrame([{"track_id": "a"}, {"track_id": "b"}]).to_parquet(segments_path, index=False)

            with patch.object(searcher, "ROOT", root), \
                 patch.object(searcher.config, "INDEX_DIR", "index"), \
                 patch.object(searcher.config, "INDEX_TYPE", "flat"), \
                 patch.object(searcher.config, "get_collection_key", return_value=key):
                first_index, first_segments = searcher.load_searcher("mfcc")
                second_index, second_segments = searcher.load_searcher("mfcc")

        self.assertIs(first_index, second_index)
        self.assertIs(first_segments, second_segments)
        self.assertEqual(first_index.ntotal, 2)
        self.assertEqual(len(first_segments), 2)

    def test_load_searcher_raises_when_index_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "index").mkdir()

            with patch.object(searcher, "ROOT", root), \
                 patch.object(searcher.config, "INDEX_DIR", "index"), \
                 patch.object(searcher.config, "INDEX_TYPE", "flat"), \
                 patch.object(searcher.config, "get_collection_key", return_value="mfcc"):
                with self.assertRaises(FileNotFoundError) as ctx:
                    searcher.load_searcher("mfcc", force_reload=True)

        self.assertIn("Run first", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
