from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.ingestion import fingerprints


class FingerprintsRebuildTests(unittest.TestCase):
    def test_process_track_returns_success_when_everything_works(self) -> None:
        with patch("src.ingestion.fingerprints.download_audio_search", return_value=("/tmp/demo", "/tmp/demo/audio.mp3", "url", None)), \
             patch("src.ingestion.fingerprints.load_audio_safe", return_value=[0.1, 0.2]), \
             patch("src.ingestion.fingerprints.extract_fingerprint", return_value={(1, 2, 3, 4)}), \
             patch("src.ingestion.fingerprints.fp_save") as fp_save, \
             patch("shutil.rmtree") as rmtree:
            result = fingerprints._process_track({"track_id": "track_1", "title": "Song", "artist": "Artist"})

        self.assertEqual(result["status"], "ok")
        fp_save.assert_called_once()
        rmtree.assert_called_once()

    def test_process_track_reports_download_or_audio_failure(self) -> None:
        with patch("src.ingestion.fingerprints.download_audio_search", return_value=(None, None, None, "not found")):
            result = fingerprints._process_track({"track_id": "track_1", "title": "Song", "artist": "Artist"})
        self.assertEqual(result["status"], "failed")

        with patch("src.ingestion.fingerprints.download_audio_search", return_value=("/tmp/demo", "/tmp/demo/audio.mp3", "url", None)), \
             patch("src.ingestion.fingerprints.load_audio_safe", return_value=None), \
             patch("shutil.rmtree"):
            result = fingerprints._process_track({"track_id": "track_1", "title": "Song", "artist": "Artist"})
        self.assertEqual(result["status"], "failed")

    def test_run_rebuild_fingerprints_requires_metadata_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch.object(fingerprints, "ROOT", root), \
                 patch.object(fingerprints.config, "METADATA_PATH", "metadata.parquet"):
                with self.assertRaises(SystemExit):
                    fingerprints.run_rebuild_fingerprints()

    def test_run_rebuild_fingerprints_returns_early_when_already_v2(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            meta_path = root / "metadata.parquet"
            pd.DataFrame([{"track_id": "track_1", "title": "Song", "artist": "Artist"}]).to_parquet(meta_path, index=False)
            with patch.object(fingerprints, "ROOT", root), \
                 patch.object(fingerprints.config, "METADATA_PATH", meta_path.name), \
                 patch("src.ingestion.fingerprints.fp_detect_format", return_value="v2"), \
                 patch("src.ingestion.fingerprints.fp_load_ids", return_value={"track_1"}), \
                 patch("src.ingestion.fingerprints._process_track") as process_track:
                fingerprints.run_rebuild_fingerprints(force_all=False)
        process_track.assert_not_called()

    def test_run_rebuild_fingerprints_dry_run_does_not_process_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            meta_path = root / "metadata.parquet"
            pd.DataFrame([{"track_id": "track_1", "title": "Song", "artist": "Artist"}]).to_parquet(meta_path, index=False)
            with patch.object(fingerprints, "ROOT", root), \
                 patch.object(fingerprints.config, "METADATA_PATH", meta_path.name), \
                 patch("src.ingestion.fingerprints.fp_detect_format", return_value="v1"), \
                 patch("src.ingestion.fingerprints.fp_load_ids", return_value={"track_1"}), \
                 patch("src.ingestion.fingerprints._process_track") as process_track:
                fingerprints.run_rebuild_fingerprints(dry_run=True)
        process_track.assert_not_called()


if __name__ == "__main__":
    unittest.main()
