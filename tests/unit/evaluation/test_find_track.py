from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.evaluation import find_track
from tests.helpers.metadata_factory import write_metadata
from tests.helpers.sqlite_factory import write_fingerprint_db


class FindTrackTests(unittest.TestCase):
    def test_rank_label_formats_expected_ranges(self) -> None:
        self.assertIn("NF", find_track._rank_label(None))
        self.assertIn("#1", find_track._rank_label(1))
        self.assertIn("#5", find_track._rank_label(5))

    def test_get_target_sr_uses_method_specific_rates(self) -> None:
        self.assertEqual(find_track._get_target_sr("clap"), find_track.config.CLAP_SAMPLE_RATE)
        self.assertEqual(find_track._get_target_sr("muq"), find_track.config.MUQ_SAMPLE_RATE)
        self.assertEqual(find_track._get_target_sr("mert"), find_track.config.MERT_SAMPLE_RATE)
        self.assertEqual(find_track._get_target_sr("mfcc"), find_track.config.SAMPLE_RATE)

    def test_load_metadata_and_get_fp_read_local_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metadata_path = write_metadata(root / "metadata.parquet")
            db_path = write_fingerprint_db(root / "fingerprints.db", {"track_1": {(1, 2, 3, 4)}})

            with patch.object(find_track, "ROOT", root), \
                 patch.object(find_track.config, "METADATA_PATH", metadata_path.name), \
                 patch.object(find_track.config, "FINGERPRINTS_DB", db_path.name):
                metadata = find_track._load_metadata()
                fingerprint = find_track._get_fp("track_1")

        self.assertEqual(metadata["track_1"]["title"], "Song One")
        self.assertEqual(fingerprint, {(1, 2, 3, 4)})


if __name__ == "__main__":
    unittest.main()
