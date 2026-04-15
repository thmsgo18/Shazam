from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.ingestion import augment_rir


class FakeCollection:
    def __init__(self, pages: list[list[str]]) -> None:
        self.pages = pages

    def get(self, limit=500, offset=0):
        index = offset // limit
        ids = self.pages[index] if index < len(self.pages) else []
        return {"ids": ids}


class AugmentRirTests(unittest.TestCase):
    def test_make_rir_is_deterministic_and_normalized(self) -> None:
        rir1 = augment_rir._make_rir(0.4, 22050, seed=1)
        rir2 = augment_rir._make_rir(0.4, 22050, seed=1)

        np.testing.assert_allclose(rir1, rir2)
        self.assertAlmostEqual(float(np.linalg.norm(rir1)), 1.0, places=4)

    def test_estimate_rt60_returns_positive_value(self) -> None:
        rir = np.exp(-np.linspace(0, 4, 22050)).astype(np.float32)
        rt60 = augment_rir._estimate_rt60(rir, 22050)
        self.assertGreater(rt60, 0.0)

    def test_select_diverse_mit_rirs_preserves_order_on_rt60(self) -> None:
        candidates = [
            ("a", np.array([1.0]), 0.1),
            ("b", np.array([1.0]), 0.5),
            ("c", np.array([1.0]), 1.0),
        ]
        selected = augment_rir._select_diverse_mit_rirs(candidates, 2)
        self.assertEqual([name for name, _ in selected], ["a", "c"])

    def test_load_rirs_returns_synthetic_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            loaded = augment_rir._load_rirs(Path(tmpdir), n=3, sr=22050, source="synthetic")
        self.assertEqual(len(loaded), 3)
        self.assertTrue(all(name.startswith("synth_") for name, _ in loaded))

    def test_apply_rir_preserves_length_and_dtype(self) -> None:
        waveform = np.ones(1000, dtype=np.float32) * 0.2
        rir = np.array([1.0, 0.5, 0.25], dtype=np.float32)

        degraded = augment_rir._apply_rir(waveform, rir)

        self.assertEqual(degraded.dtype, np.float32)
        self.assertEqual(len(degraded), len(waveform))

    def test_batch_embed_dispatches_to_selected_backend(self) -> None:
        segments = [np.ones(32, dtype=np.float32)]
        with patch("src.features.embeddings_audio.clap_batch_embeddings", return_value=np.array([[1.0]])):
            np.testing.assert_array_equal(augment_rir._batch_embed(segments, 22050, "clap"), np.array([[1.0]]))
        with patch("src.features.embeddings_audio.mert_batch_embeddings", return_value=np.array([[2.0]])):
            np.testing.assert_array_equal(augment_rir._batch_embed(segments, 22050, "mert"), np.array([[2.0]]))
        with patch("src.features.embeddings_audio.muq_batch_embeddings", return_value=np.array([[3.0]])):
            np.testing.assert_array_equal(augment_rir._batch_embed(segments, 22050, "muq"), np.array([[3.0]]))
        with patch("src.features.embeddings_audio.mfcc_stats_embedding", return_value=np.array([4.0])):
            np.testing.assert_array_equal(augment_rir._batch_embed(segments, 22050, "mfcc"), np.array([[4.0]]))

    def test_load_rir_done_reads_existing_metadata_column(self) -> None:
        df = pd.DataFrame(
            [
                {"track_id": "track_1", "rir_augmented": {"clap_demo": ["rir_a", "rir_b"]}},
                {"track_id": "track_2", "rir_augmented": None},
            ]
        )
        done = augment_rir._load_rir_done(df, "clap_demo")
        self.assertEqual(done, {"track_1": ["rir_a", "rir_b"]})

    def test_mark_rir_done_updates_metadata_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_path = Path(tmpdir) / "metadata.parquet"
            pd.DataFrame([{"track_id": "track_1"}]).to_parquet(meta_path, index=False)

            augment_rir._mark_rir_done(meta_path, "track_1", "clap_demo", "room_a", threading.Lock())
            augment_rir._mark_rir_done(meta_path, "track_1", "clap_demo", "room_b", threading.Lock())

            df = pd.read_parquet(meta_path)

        self.assertEqual(list(df.iloc[0]["rir_augmented"]["clap_demo"]), ["room_a", "room_b"])

    def test_backfill_rir_done_rebuilds_history_from_chroma_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_path = Path(tmpdir) / "metadata.parquet"
            pd.DataFrame(
                [{"track_id": "track_1"}, {"track_id": "track_2"}]
            ).to_parquet(meta_path, index=False)
            collection = FakeCollection([["track_1_rir_room_a_0", "track_1_rir_room_b_1", "track_2_0"]])

            rir_map = augment_rir._backfill_rir_done(meta_path, collection, "clap_demo")
            df = pd.read_parquet(meta_path)

        self.assertEqual(set(rir_map["track_1"]), {"room_a", "room_b"})
        self.assertEqual(set(df.iloc[0]["rir_augmented"]["clap_demo"]), {"room_a", "room_b"})

    def test_rebuild_index_delegates_to_build_for_method(self) -> None:
        chroma_client = object()
        with patch("src.index.build_index._build_for_method") as build_for_method:
            augment_rir._rebuild_index("clap_demo", chroma_client)
        build_for_method.assert_called_once_with("clap_demo", augment_rir.config.INDEX_TYPE, chroma_client)


if __name__ == "__main__":
    unittest.main()
