from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.ingestion import ingest
from src.utils.fingerprints_db import fp_init
from src.utils.fingerprints_db import fp_load_ids


class FakeCollection:
    def __init__(self) -> None:
        self.ids = ["old_segment"]
        self.deleted_ids: list[list[str]] = []
        self.add_calls: list[dict] = []

    def get(self, where=None):
        return {"ids": list(self.ids)}

    def delete(self, ids):
        self.deleted_ids.append(list(ids))
        self.ids = []

    def add(self, embeddings, ids, metadatas):
        self.add_calls.append(
            {"embeddings": embeddings, "ids": ids, "metadatas": metadatas}
        )
        self.ids = list(ids)


class IngestTests(unittest.TestCase):
    def test_find_column_returns_first_match(self) -> None:
        df = pd.DataFrame(columns=["name", "artist"])
        self.assertEqual(ingest.find_column(df, ["title", "name"]), "name")
        self.assertIsNone(ingest.find_column(df, ["missing"]))

    def test_get_csv_files_supports_file_and_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            file_path = root / "one.csv"
            other_path = root / "two.csv"
            file_path.write_text("a,b\n1,2\n", encoding="utf-8")
            other_path.write_text("a,b\n3,4\n", encoding="utf-8")

            self.assertEqual(ingest.get_csv_files(file_path), [file_path])
            self.assertEqual(ingest.get_csv_files(root), [file_path, other_path])

    def test_get_csv_files_exits_for_missing_or_empty_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            empty_dir = root / "empty"
            empty_dir.mkdir()

            with self.assertRaises(SystemExit):
                ingest.get_csv_files(root / "missing.csv")
            with self.assertRaises(SystemExit):
                ingest.get_csv_files(empty_dir)

    def test_normalize_title_removes_version_markers(self) -> None:
        self.assertEqual(ingest.normalize_title("Song (Taylor's Version)"), "Song")
        self.assertEqual(ingest.normalize_title("Song - Live at Wembley"), "Song")
        self.assertEqual(ingest.normalize_title("Song (Radio Edit)"), "Song")

    def test_load_tracks_from_csv_deduplicates_and_normalizes_artist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "tracks.csv"
            pd.DataFrame(
                [
                    {"track_name": "Song (Radio Edit)", "artists": "Artist One"},
                    {"track_name": "Song", "artists": "Artist One"},
                    {"track_name": "Other", "artists": "['Artist Two', 'Feat']"},
                ]
            ).to_csv(csv_path, index=False)

            tracks = ingest.load_tracks_from_csv(csv_path)

        self.assertEqual(
            tracks,
            [
                {"title": "Song (Radio Edit)", "artist": "Artist One", "source": "tracks.csv"},
                {"title": "Other", "artist": "Artist Two", "source": "tracks.csv"},
            ],
        )

    def test_get_method_key_uses_model_names(self) -> None:
        with patch.object(ingest.config, "CLAP_MODEL_NAME", "laion/demo"), \
             patch.object(ingest.config, "MUQ_MODEL_NAME", "OpenMuQ/demo"):
            self.assertEqual(ingest.get_method_key("mfcc"), "mfcc")
            self.assertEqual(ingest.get_method_key("clap"), "clap:laion/demo")
            self.assertEqual(ingest.get_method_key("muq"), "muq:OpenMuQ/demo")

    def test_load_already_processed_reads_embedded_methods(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            processed_dir = Path(tmpdir) / "processed"
            processed_dir.mkdir()
            meta_path = processed_dir / "metadata.parquet"
            pd.DataFrame(
                [
                    {"artist": "Artist", "title": "Song", "embedded_methods": ["mfcc", "clap:demo/model"]},
                    {"artist": "Other", "title": "Track", "embedded_methods": ["mfcc"]},
                ]
            ).to_parquet(meta_path, index=False)

            with patch.object(ingest, "PROCESSED_DIR", processed_dir), \
                 patch.object(ingest.config, "CLAP_MODEL_NAME", "demo/model"):
                processed = ingest.load_already_processed("clap")

        self.assertEqual(processed, {("artist", "song")})

    def test_save_track_rewrites_collection_saves_fp_and_updates_metadata(self) -> None:
        collection = FakeCollection()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fp_db = root / "fingerprints.db"
            meta_path = root / "metadata.parquet"
            fp_init(fp_db)
            pd.DataFrame(
                [
                    {
                        "track_id": "track_1",
                        "title": "Song",
                        "artist": "Artist",
                        "embedded_methods": ["mfcc"],
                    }
                ]
            ).to_parquet(meta_path, index=False)

            metadata_row = {
                "track_id": "track_1",
                "title": "Song",
                "artist": "Artist",
                "embedded_methods": ["clap:laion/demo"],
            }
            with patch.object(ingest.config, "CLAP_MODEL_NAME", "laion/demo"):
                ingest._save_track(
                    track_id="track_1",
                    method="clap",
                    track_embeddings=[[1.0, 0.0], [0.0, 1.0]],
                    track_segments=[{"start_s": 0.0}, {"start_s": 5.0}],
                    new_fp_hashes={(1, 2, 3, 4)},
                    metadata_row=metadata_row,
                    collection=collection,
                    fp_db=fp_db,
                    meta_path=meta_path,
                )

            df = pd.read_parquet(meta_path)
            self.assertEqual(collection.deleted_ids, [["old_segment"]])
            self.assertEqual(collection.add_calls[0]["ids"], ["track_1_0", "track_1_1"])
            self.assertEqual(fp_load_ids(fp_db), {"track_1"})
            self.assertIn("mfcc", df.iloc[0]["embedded_methods"])
            self.assertIn("clap:laion/demo", df.iloc[0]["embedded_methods"])


if __name__ == "__main__":
    unittest.main()
