from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np
import pandas as pd

from src.evaluation import rir_impact


class RirImpactTests(unittest.TestCase):
    def test_no_rir_index_paths_and_rank_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch.object(rir_impact, "ROOT", root), patch.object(rir_impact.config, "INDEX_DIR", "index"):
                index_path, seg_path = rir_impact._no_rir_index_paths("mfcc")
        self.assertTrue(str(index_path).endswith("index_mfcc_no_rir_flat.faiss"))
        self.assertTrue(str(seg_path).endswith("segments_mfcc_no_rir.parquet"))
        self.assertIn("NF", rir_impact._rank_label(None))

    def test_search_aggregates_track_scores(self) -> None:
        index = faiss.IndexFlatIP(2)
        xb = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        faiss.normalize_L2(xb)
        index.add(xb)
        segments = pd.DataFrame([{"track_id": "a"}, {"track_id": "a"}, {"track_id": "b"}])
        scores = rir_impact._search(index, segments, np.array([1.0, 0.0], dtype=np.float32), k=3)
        self.assertGreater(scores["a"], scores["b"])

    def test_load_no_rir_index_cached_reads_disk_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            index_dir = root / "index"
            index_dir.mkdir()
            index_path = index_dir / "index_mfcc_no_rir_flat.faiss"
            seg_path = index_dir / "segments_mfcc_no_rir.parquet"
            index = faiss.IndexFlatIP(2)
            index.add(np.array([[1.0, 0.0]], dtype=np.float32))
            faiss.write_index(index, str(index_path))
            pd.DataFrame([{"track_id": "track_1"}]).to_parquet(seg_path, index=False)

            with patch.object(rir_impact, "ROOT", root), patch.object(rir_impact.config, "INDEX_DIR", "index"):
                loaded_index, loaded_segments = rir_impact.load_no_rir_index_cached("mfcc")

        self.assertEqual(loaded_index.ntotal, 1)
        self.assertEqual(loaded_segments["track_id"].tolist(), ["track_1"])

    def test_rir_impact_scores_returns_rank_comparison(self) -> None:
        waveform = np.ones(22050, dtype=np.float32)
        index_no_rir = faiss.IndexFlatIP(2)
        index_with_rir = faiss.IndexFlatIP(2)
        xb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        faiss.normalize_L2(xb)
        index_no_rir.add(xb)
        index_with_rir.add(xb)
        segments = pd.DataFrame([{"track_id": "track_1"}, {"track_id": "track_2"}])

        with patch("src.evaluation.rir_impact._load_model"), \
             patch("src.evaluation.rir_impact.load_audio", return_value=(waveform, 22050)), \
             patch("src.evaluation.rir_impact.preprocess_query", return_value=waveform), \
             patch("src.evaluation.rir_impact.iter_segments", return_value=[(0.0, waveform[:1024])]), \
             patch("src.evaluation.rir_impact.embed_segment", return_value=np.array([1.0, 0.0], dtype=np.float32)), \
             patch("src.retrieval.searcher.load_searcher", return_value=(index_with_rir, segments)):
            result = rir_impact.rir_impact_scores(
                audio_path="query.wav",
                track_id="track_1",
                method="mfcc",
                prebuilt_no_rir=(index_no_rir, segments),
            )

        self.assertEqual(result["with_rir"]["rank"], 1)
        self.assertEqual(result["without_rir"]["rank"], 1)


if __name__ == "__main__":
    unittest.main()
