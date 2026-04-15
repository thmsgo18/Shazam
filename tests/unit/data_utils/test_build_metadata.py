from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.data_utils.build_metadata import generate_track_id


class BuildMetadataTests(unittest.TestCase):
    def test_generate_track_id_is_stable_for_same_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.bin"
            path.write_bytes(b"abc" * 5000)

            first = generate_track_id(path)
            second = generate_track_id(path)

        self.assertEqual(first, second)

    def test_generate_track_id_changes_with_file_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            first_path = Path(tmpdir) / "a.bin"
            second_path = Path(tmpdir) / "b.bin"
            first_path.write_bytes(b"a" * 9000)
            second_path.write_bytes(b"b" * 9000)

            first = generate_track_id(first_path)
            second = generate_track_id(second_path)

        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
