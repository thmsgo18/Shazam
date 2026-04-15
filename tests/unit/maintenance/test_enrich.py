from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.maintenance import enrich


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self.payload


class EnrichTests(unittest.TestCase):
    def test_clean_helpers(self) -> None:
        self.assertEqual(enrich._clean_artist("Artist One, Guest"), "Artist One")
        self.assertEqual(enrich._clean_artist("¥Artist$"), "Artist")
        self.assertEqual(enrich._clean_title("Song (Remastered) feat. Guest"), "Song")

    def test_deezer_search_returns_album_metadata(self) -> None:
        responses = [
            FakeResponse({"data": [{"album": {"id": 1, "title": "Album", "cover_big": "cover"}, "artist": {"id": 2}}]}),
            FakeResponse({"genres": {"data": [{"name": "Pop"}]}, "release_date": "2024-01-01"}),
        ]
        with patch.object(enrich.SESSION, "get", side_effect=responses):
            payload = enrich._deezer_search("Artist", "Song")

        self.assertEqual(payload["album"], "Album")
        self.assertEqual(payload["genre"], "Pop")
        self.assertEqual(payload["release_date"], "2024-01-01")

    def test_musicbrainz_search_returns_release_information(self) -> None:
        response = FakeResponse(
            {
                "recordings": [
                    {
                        "releases": [{"date": "2020-01-01", "title": "Album"}],
                        "tags": [{"name": "rock", "count": 10}],
                    }
                ]
            }
        )
        with patch.object(enrich.SESSION, "get", return_value=response):
            payload = enrich._musicbrainz_search("Artist", "Song")

        self.assertEqual(payload["album"], "Album")
        self.assertEqual(payload["genre"], "Rock")

    def test_fetch_metadata_falls_back_to_musicbrainz(self) -> None:
        with patch("src.maintenance.enrich._deezer_search", return_value=None), \
             patch("src.maintenance.enrich._musicbrainz_search", return_value={"album": "Fallback", "genre": None, "release_date": None, "cover_url": None}), \
             patch("time.sleep"):
            payload = enrich._fetch_metadata("Artist", "Song")

        self.assertEqual(payload["album"], "Fallback")

    def test_run_enrich_updates_missing_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = Path(tmpdir) / "metadata.parquet"
            pd.DataFrame(
                [{"track_id": "track_1", "artist": "Artist", "title": "Song", "album": None, "genre": None, "release_date": None, "cover_url": None}]
            ).to_parquet(metadata_path, index=False)

            new_metadata = {
                "album": "Album",
                "genre": "Pop",
                "release_date": "2024-01-01",
                "cover_url": "https://example.com/cover.jpg",
            }
            with patch.object(enrich, "METADATA_PATH", metadata_path), \
                 patch("src.maintenance.enrich._fetch_metadata", return_value=new_metadata):
                enrich.run_enrich()

            df = pd.read_parquet(metadata_path)

        self.assertEqual(df.iloc[0]["album"], "Album")
        self.assertEqual(df.iloc[0]["genre"], "Pop")


if __name__ == "__main__":
    unittest.main()
