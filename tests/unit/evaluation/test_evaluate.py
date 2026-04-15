from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.evaluation import evaluate


class EvaluateTests(unittest.TestCase):
    def test_manifest_signature_is_stable(self) -> None:
        manifest = [{"filename": "a.mp3", "track_id": "1", "position": "middle", "duration_s": 15}]
        methods = ["mfcc"]
        conditions = ["clean", "reverb"]

        first = evaluate._manifest_signature(manifest, methods, conditions)
        second = evaluate._manifest_signature(manifest, methods, conditions)

        self.assertEqual(first, second)

    def test_load_jsonl_cache_ignores_invalid_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.jsonl"
            path.write_text('{"_cache_key":"one","value":1}\nnot-json\n{"_cache_key":"two","value":2}\n', encoding="utf-8")

            cache = evaluate._load_jsonl_cache(path)

        self.assertEqual(cache["one"]["value"], 1)
        self.assertEqual(cache["two"]["value"], 2)

    def test_apply_condition_handles_clean_and_invalid_condition(self) -> None:
        waveform = np.ones(32, dtype=np.float32)
        np.testing.assert_array_equal(evaluate._apply_condition(waveform, 22050, "clean"), waveform)
        with self.assertRaises(ValueError):
            evaluate._apply_condition(waveform, 22050, "invalid")

    def test_load_manifest_keeps_only_existing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "data" / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / "exists.mp3").write_text("", encoding="utf-8")
            manifest_path = raw_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    [
                        {"filename": "exists.mp3", "track_id": "track_1"},
                        {"filename": "missing.mp3", "track_id": "track_2"},
                    ]
                ),
                encoding="utf-8",
            )

            with patch.object(evaluate, "ROOT", root):
                manifest = evaluate.load_manifest(manifest_path)

        self.assertEqual(manifest, [{"filename": "exists.mp3", "track_id": "track_1"}])

    def test_filter_rir_manifest_entries_keeps_only_report_queries(self) -> None:
        entries = [
            {"filename": "reference_clips/a.mp3"},
            {"filename": "mic_recordings/b.mp3"},
            {"filename": "other/c.mp3"},
        ]

        filtered = evaluate._filter_rir_manifest_entries(entries)

        self.assertEqual(filtered, entries[:2])

    def test_find_track_id_by_query_uses_metadata_jaccard_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = Path(tmpdir) / "metadata.parquet"
            pd.DataFrame(
                [
                    {"track_id": "track_1", "artist": "Daft Punk", "title": "Get Lucky"},
                    {"track_id": "track_2", "artist": "Phoenix", "title": "Lisztomania"},
                ]
            ).to_parquet(metadata_path, index=False)

            with patch.object(evaluate, "METADATA_PATH", metadata_path):
                match = evaluate.find_track_id_by_query("Daft Punk Lucky")

        self.assertEqual(match, "track_1")


if __name__ == "__main__":
    unittest.main()
