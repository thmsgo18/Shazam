from __future__ import annotations

from pathlib import Path

import pandas as pd


def metadata_rows() -> list[dict]:
    return [
        {
            "track_id": "track_1",
            "artist": "Artist One",
            "title": "Song One",
            "album": "Album One",
            "genre": "Pop",
            "duration_s": 180.0,
            "cover_url": "https://example.com/cover1.jpg",
        },
        {
            "track_id": "track_2",
            "artist": "Artist Two",
            "title": "Song Two",
            "album": "Album Two",
            "genre": "Pop",
            "duration_s": 200.0,
            "cover_url": "https://example.com/cover2.jpg",
        },
        {
            "track_id": "track_3",
            "artist": "Artist Three",
            "title": "Song Three",
            "album": None,
            "genre": "Jazz",
            "duration_s": 210.0,
            "cover_url": None,
        },
    ]


def metadata_frame(rows: list[dict] | None = None) -> pd.DataFrame:
    return pd.DataFrame(rows or metadata_rows())


def write_metadata(path: str | Path, rows: list[dict] | None = None) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_frame(rows).to_parquet(output_path, index=False)
    return output_path
