from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.api import app as api_app
from tests.helpers.metadata_factory import metadata_rows, write_metadata


class ApiAppTests(unittest.TestCase):
    def setUp(self) -> None:
        api_app._METADATA_CACHE = None
        api_app._METADATA_MTIME_NS = None

    def tearDown(self) -> None:
        api_app._METADATA_CACHE = None
        api_app._METADATA_MTIME_NS = None

    def test_clean_normalizes_none_nan_and_empty_string(self) -> None:
        self.assertEqual(api_app._clean(None, "fallback"), "fallback")
        self.assertEqual(api_app._clean(math.nan, "fallback"), "fallback")
        self.assertEqual(api_app._clean("", "fallback"), "fallback")
        self.assertEqual(api_app._clean("value", "fallback"), "value")

    def test_load_metadata_reads_parquet_and_caches_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = write_metadata(Path(tmpdir) / "metadata.parquet", metadata_rows())

            with patch.object(api_app, "METADATA_PATH", metadata_path):
                first = api_app._load_metadata(force_reload=True)
                second = api_app._load_metadata()

        self.assertIs(first, second)
        self.assertEqual(first["track_1"]["title"], "Song One")
        self.assertEqual(first["track_3"]["artist"], "Artist Three")

    def test_streaming_links_build_platform_urls(self) -> None:
        links = api_app._streaming_links("Daft Punk", "Get Lucky")

        self.assertIn("Daft+Punk+Get+Lucky", links["youtube"])
        self.assertIn("spotify", links["spotify"])

    def test_recommendations_filter_same_track_and_genre(self) -> None:
        metadata = {
            row["track_id"]: row
            for row in metadata_rows()
        }

        recommendations = api_app._recommendations("track_1", "Pop", metadata, top=4)

        self.assertEqual([row["track_id"] for row in recommendations], ["track_2"])

    def test_identify_audio_enriches_results_with_metadata(self) -> None:
        metadata = {
            "track_1": {
                "title": "Song One",
                "artist": "Artist One",
                "album": "Album One",
                "genre": "Pop",
                "duration_s": 180.0,
                "cover_url": "https://example.com/cover1.jpg",
            }
        }

        with patch("src.retrieval.query_pipeline.identify_track", return_value=[("track_1", 0.91, 0.5, 0.91)]):
            results = api_app.identify_audio("audio.wav", method="mfcc", top=1, detailed=True, metadata=metadata)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Song One")
        self.assertEqual(results[0]["score"], 0.91)
        self.assertEqual(results[0]["score_faiss"], 0.5)
        self.assertEqual(results[0]["score_fp"], 0.91)

    def test_build_identification_response_marks_confident_result(self) -> None:
        metadata = {
            "track_1": {
                "title": "Song One",
                "artist": "Artist One",
                "album": "Album One",
                "genre": "Pop",
                "duration_s": 180.0,
                "cover_url": "https://example.com/cover1.jpg",
            },
            "track_2": {
                "title": "Song Two",
                "artist": "Artist Two",
                "album": "Album Two",
                "genre": "Pop",
                "duration_s": 200.0,
                "cover_url": "https://example.com/cover2.jpg",
            },
        }
        identify_results = [
            {"rank": 1, "track_id": "track_1", "title": "Song One", "artist": "Artist One", "genre": "Pop", "score": 0.9},
            {"rank": 2, "track_id": "track_2", "title": "Song Two", "artist": "Artist Two", "genre": "Pop", "score": 0.2},
        ]

        with patch.object(api_app.config, "UI_CONFIDENCE_RATIO", 2.5), \
             patch("src.api.app._load_metadata", return_value=metadata), \
             patch("src.api.app.identify_audio", return_value=identify_results):
            response = api_app.build_identification_response("audio.wav", method="mfcc", top=2, detailed=True)

        self.assertTrue(response["confident"])
        self.assertEqual(response["results"], identify_results)
        self.assertEqual(response["recommendations"][0]["track_id"], "track_2")


if __name__ == "__main__":
    unittest.main()
