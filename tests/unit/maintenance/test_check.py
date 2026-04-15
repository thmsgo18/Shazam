from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np
import pandas as pd

from src.maintenance import check
from src.utils.fingerprints_db import fp_init, fp_save


class FakeCollection:
    def __init__(self, metadatas: list[dict], ids: list[str] | None = None) -> None:
        self._metadatas = metadatas
        self._ids = ids or [f"id_{i}" for i in range(len(metadatas))]
        self.deleted_ids: list[list[str]] = []

    def count(self) -> int:
        return len(self._metadatas)

    def get(self, include=None, limit=500, offset=0, where=None):
        if where and "track_id" in where:
            target = where["track_id"]["$eq"]
            ids = [id_ for id_, meta in zip(self._ids, self._metadatas) if meta["track_id"] == target]
            return {"ids": ids}
        end = offset + limit
        return {
            "ids": self._ids[offset:end],
            "metadatas": self._metadatas[offset:end] if include and "metadatas" in include else None,
            "embeddings": [[1.0, 0.0] for _ in self._metadatas[offset:end]] if include and "embeddings" in include else None,
        }

    def delete(self, ids):
        self.deleted_ids.append(list(ids))


class CheckTests(unittest.TestCase):
    def test_basic_helpers(self) -> None:
        self.assertEqual(check._fmt_dur(65), "1m 05s")
        self.assertEqual(check._pct(1, 4), "25%")
        self.assertEqual(check._pct(1, 0), "—")
        self.assertEqual(check._collection_family("clap_demo"), "clap")
        self.assertEqual(check._expected_embedding_dim("mfcc"), 2 * check.config.N_MFCC)
        self.assertTrue(check._embedded_method_matches_collection("clap:laion/demo-model", "clap_demo_model"))

    def test_chroma_get_all_paginates(self) -> None:
        collection = FakeCollection(
            [{"track_id": "a"}, {"track_id": "b"}, {"track_id": "c"}],
            ids=["1", "2", "3"],
        )
        result = check._chroma_get_all(collection, include=["metadatas"])
        self.assertEqual(result["ids"], ["1", "2", "3"])
        self.assertEqual(len(result["metadatas"]), 3)

    def test_method_summary_reports_counts(self) -> None:
        metadatas = [{"track_id": "track_1"}, {"track_id": "track_1"}, {"track_id": "track_2"}]
        collection = FakeCollection(metadatas)
        df_meta = pd.DataFrame(
            [
                {"track_id": "track_1", "duration": 10.0},
                {"track_id": "track_2", "duration": 20.0},
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fp_db = root / "fingerprints.db"
            index_dir = root / "index"
            index_dir.mkdir()
            fp_init(fp_db)
            fp_save(fp_db, "track_1", {(1, 2, 3, 4)})
            index_path = index_dir / f"index_mfcc_{check.config.INDEX_TYPE}.faiss"
            index = faiss.IndexFlatIP(40)
            index.add(np.ones((3, 40), dtype=np.float32))
            faiss.write_index(index, str(index_path))

            with patch.object(check, "ROOT", root), \
                 patch.object(check.config, "FINGERPRINTS_DB", fp_db.relative_to(root).as_posix()), \
                 patch.object(check.config, "INDEX_DIR", index_dir.relative_to(root).as_posix()), \
                 patch("src.maintenance.check._get_chroma_collection", return_value=(collection, "")):
                summary = check._method_summary("mfcc", df_meta)

        self.assertEqual(summary.n_segments, 3)
        self.assertEqual(summary.n_tracks, 2)
        self.assertEqual(summary.n_fp, 1)
        self.assertTrue(summary.index_ok)

    def test_purge_tracks_removes_segments_metadata_and_fingerprints(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            processed_dir = root / "data" / "processed"
            processed_dir.mkdir(parents=True)
            index_dir = root / "data" / "index"
            index_dir.mkdir(parents=True)
            fp_db = root / "data" / "features" / "fingerprints.db"
            fp_db.parent.mkdir(parents=True)
            meta_path = processed_dir / "metadata.parquet"
            pd.DataFrame(
                [
                    {"track_id": "track_1", "embedded_methods": ["mfcc"]},
                    {"track_id": "track_2", "embedded_methods": ["mfcc", "clap:demo"]},
                ]
            ).to_parquet(meta_path, index=False)
            fp_init(fp_db)
            fp_save(fp_db, "track_1", {(1, 2, 3, 4)})
            fp_save(fp_db, "track_2", {(5, 6, 7, 8)})
            (index_dir / f"index_mfcc_{check.config.INDEX_TYPE}.faiss").write_text("", encoding="utf-8")
            (index_dir / "segments_mfcc.parquet").write_text("", encoding="utf-8")

            collection = FakeCollection([{"track_id": "track_1"}, {"track_id": "track_2"}], ids=["seg_1", "seg_2"])
            with patch.object(check, "ROOT", root), \
                 patch.object(check, "PROCESSED_DIR", processed_dir), \
                 patch.object(check.config, "FINGERPRINTS_DB", "data/features/fingerprints.db"), \
                 patch.object(check.config, "INDEX_DIR", "data/index"), \
                 patch("src.maintenance.check._get_chroma_collection", return_value=(collection, "")):
                stats = check.purge_tracks("mfcc", {"track_1"})

            df = pd.read_parquet(meta_path)

        self.assertEqual(stats["segments_removed"], 1)
        self.assertEqual(stats["tracks_removed"], 1)
        self.assertEqual(stats["fingerprints_removed"], 1)
        self.assertEqual(df["track_id"].tolist(), ["track_2"])

    def test_run_purge_missing_fp_builds_purge_plan(self) -> None:
        collection = FakeCollection([{"track_id": "track_1"}, {"track_id": "track_2"}])
        with patch("src.maintenance.check._fp_load_stats", return_value={"track_1": 10}), \
             patch("src.maintenance.check._get_chroma_collection", return_value=(collection, "")), \
             patch("src.maintenance.check._chroma_get_all", return_value={"metadatas": [{"track_id": "track_1"}, {"track_id": "track_2"}]}), \
             patch("src.maintenance.check._run_purge") as run_purge:
            check._run_purge_missing_fp(["mfcc"], yes=True)

        run_purge.assert_called_once_with({"mfcc": {"track_2"}}, yes=True)


if __name__ == "__main__":
    unittest.main()
