from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.utils.metadata import atomic_write_parquet, atomic_write_pickle


class MetadataUtilsTests(unittest.TestCase):
    def test_atomic_write_parquet_replaces_file_contents(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metadata.parquet"
            atomic_write_parquet(path, pd.DataFrame([{"track_id": "a"}]))
            atomic_write_parquet(path, pd.DataFrame([{"track_id": "b"}]))

            df = pd.read_parquet(path)

        self.assertEqual(df.to_dict(orient="records"), [{"track_id": "b"}])

    def test_atomic_write_pickle_replaces_file_contents(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.pkl"
            atomic_write_pickle(path, {"value": 1})
            atomic_write_pickle(path, {"value": 2})

            with open(path, "rb") as handle:
                payload = pickle.load(handle)

        self.assertEqual(payload, {"value": 2})


if __name__ == "__main__":
    unittest.main()
