from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

from src.utils.fingerprints_db import (
    fp_delete,
    fp_detect_format,
    fp_init,
    fp_load_all,
    fp_load_ids,
    fp_load_stats,
    fp_migrate_from_pkl,
    fp_save,
)


class FingerprintsDbTests(unittest.TestCase):
    def test_fp_save_and_load_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "fingerprints.db"
            fp_init(db_path)
            fp_save(db_path, "track_1", {(1, 2, 3, 4)})
            fp_save(db_path, "track_2", set())

            ids = fp_load_ids(db_path)
            stats = fp_load_stats(db_path)
            all_hashes = fp_load_all(db_path)

        self.assertEqual(ids, {"track_1"})
        self.assertEqual(stats["track_1"], 1)
        self.assertEqual(stats["track_2"], 0)
        self.assertEqual(all_hashes["track_1"], {(1, 2, 3, 4)})
        self.assertEqual(all_hashes["track_2"], set())

    def test_fp_delete_removes_requested_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "fingerprints.db"
            fp_init(db_path)
            fp_save(db_path, "track_1", {(1, 2, 3, 4)})
            fp_save(db_path, "track_2", {(2, 3, 4, 5)})

            deleted = fp_delete(db_path, {"track_1"})
            remaining = fp_load_ids(db_path)

        self.assertEqual(deleted, 1)
        self.assertEqual(remaining, {"track_2"})

    def test_fp_detect_format_handles_v2_and_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "fingerprints.db"
            fp_init(db_path)
            self.assertEqual(fp_detect_format(db_path), "unknown")

            fp_save(db_path, "track_1", {(1, 2, 3, 4)})
            self.assertEqual(fp_detect_format(db_path), "v2")

    def test_fp_migrate_from_pkl_imports_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = Path(tmpdir) / "fingerprints.pkl"
            db_path = Path(tmpdir) / "fingerprints.db"
            with open(pkl_path, "wb") as handle:
                pickle.dump({"track_1": {(1, 2, 3, 4)}, "track_2": {(5, 6, 7, 8)}}, handle)

            migrated = fp_migrate_from_pkl(pkl_path, db_path)

            self.assertEqual(migrated, 2)
            self.assertEqual(fp_load_ids(db_path), {"track_1", "track_2"})


if __name__ == "__main__":
    unittest.main()
