from __future__ import annotations

import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from src.retrieval import query_pipeline
from tests.helpers.sqlite_factory import write_fingerprint_db


class QueryPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        query_pipeline._FINGERPRINT_CACHE = None
        query_pipeline._FINGERPRINT_DB_MTIME_NS = None

    def tearDown(self) -> None:
        query_pipeline._FINGERPRINT_CACHE = None
        query_pipeline._FINGERPRINT_DB_MTIME_NS = None

    def test_enforce_fingerprint_cache_limit_trims_lru(self) -> None:
        cache = OrderedDict([("a", {1}), ("b", {2}), ("c", {3})])

        with patch.object(query_pipeline.config, "FINGERPRINT_CACHE_MAX", 2):
            query_pipeline._enforce_fingerprint_cache_limit(cache)

        self.assertEqual(list(cache.keys()), ["b", "c"])

    def test_load_and_get_cached_fingerprint_use_sqlite_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            db_path = write_fingerprint_db(root / "fingerprints.db", {"track_1": {(1, 2, 3, 4)}})

            with patch.object(query_pipeline, "ROOT", root), \
                 patch.object(query_pipeline.config, "FINGERPRINTS_DB", db_path.name):
                cache = query_pipeline.load_fingerprint_cache(force_reload=True)
                fingerprint = query_pipeline.get_cached_fingerprint("track_1")
                cached_again = query_pipeline.get_cached_fingerprint("track_1")

        self.assertEqual(cache, OrderedDict([("track_1", {(1, 2, 3, 4)})]))
        self.assertEqual(fingerprint, {(1, 2, 3, 4)})
        self.assertEqual(cached_again, {(1, 2, 3, 4)})

    def test_identify_track_uses_fingerprint_reranking_when_available(self) -> None:
        fake_index = MagicMock()
        fake_segments = object()
        waveform = np.ones(22050, dtype=np.float32)

        def fake_get_cached_fingerprint(track_id: str) -> set[tuple]:
            return {("track", track_id)}

        def fake_similarity(_query_fp: set[tuple], candidate_fp: set[tuple]) -> float:
            return 0.9 if ("track", "track_b") in candidate_fp else 0.2

        with patch.object(query_pipeline.config, "OPT_BATCH_EMBED", False), \
             patch.object(query_pipeline.config, "OPT_FINGERPRINT_PARALLEL", False), \
             patch.object(query_pipeline.config, "VECTOR_TOP_N_TRACKS", 5), \
             patch.object(query_pipeline.config, "VECTOR_TOP_K_SEGMENTS", 3), \
             patch("src.retrieval.query_pipeline.load_searcher", return_value=(fake_index, fake_segments)), \
             patch("src.retrieval.query_pipeline.load_audio", return_value=(waveform, 22050)), \
             patch("src.retrieval.query_pipeline.preprocess_query", return_value=waveform), \
             patch("src.retrieval.query_pipeline.iter_segments", return_value=[(0.0, waveform[:1024])]), \
             patch("src.retrieval.query_pipeline.embed_segment", return_value=np.array([1.0, 0.0], dtype=np.float32)), \
             patch("src.retrieval.query_pipeline.search_segments", return_value=(np.array([0.8, 0.6]), np.array([0, 1]))), \
             patch("src.retrieval.query_pipeline.aggregate_by_track", return_value=[("track_a", 0.8), ("track_b", 0.6)]), \
             patch("src.retrieval.query_pipeline.extract_fingerprint", return_value={("query", 1)}), \
             patch("src.retrieval.query_pipeline.get_cached_fingerprint", side_effect=fake_get_cached_fingerprint), \
             patch("src.retrieval.query_pipeline.fingerprint_similarity", side_effect=fake_similarity):
            results = query_pipeline.identify_track("audio.wav", method="mfcc", top_n=2, detailed=True)

        self.assertEqual(results[0], ("track_b", 0.9, 0.6, 0.9))
        self.assertEqual(results[1], ("track_a", 0.2, 0.8, 0.2))

    def test_identify_track_falls_back_to_faiss_when_all_fp_scores_are_zero(self) -> None:
        fake_index = MagicMock()
        fake_segments = object()
        waveform = np.ones(22050, dtype=np.float32)

        with patch.object(query_pipeline.config, "OPT_BATCH_EMBED", False), \
             patch.object(query_pipeline.config, "OPT_FINGERPRINT_PARALLEL", False), \
             patch.object(query_pipeline.config, "VECTOR_TOP_N_TRACKS", 5), \
             patch.object(query_pipeline.config, "VECTOR_TOP_K_SEGMENTS", 3), \
             patch("src.retrieval.query_pipeline.load_searcher", return_value=(fake_index, fake_segments)), \
             patch("src.retrieval.query_pipeline.load_audio", return_value=(waveform, 22050)), \
             patch("src.retrieval.query_pipeline.preprocess_query", return_value=waveform), \
             patch("src.retrieval.query_pipeline.iter_segments", return_value=[(0.0, waveform[:1024])]), \
             patch("src.retrieval.query_pipeline.embed_segment", return_value=np.array([1.0, 0.0], dtype=np.float32)), \
             patch("src.retrieval.query_pipeline.search_segments", return_value=(np.array([0.8, 0.6]), np.array([0, 1]))), \
             patch("src.retrieval.query_pipeline.aggregate_by_track", return_value=[("track_a", 0.8), ("track_b", 0.6)]), \
             patch("src.retrieval.query_pipeline.extract_fingerprint", return_value={("query", 1)}), \
             patch("src.retrieval.query_pipeline.get_cached_fingerprint", return_value={("candidate", 1)}), \
             patch("src.retrieval.query_pipeline.fingerprint_similarity", return_value=0.0):
            results = query_pipeline.identify_track("audio.wav", method="mfcc", top_n=2, detailed=False)

        self.assertEqual(results, [("track_a", 0.8), ("track_b", 0.6)])


if __name__ == "__main__":
    unittest.main()
