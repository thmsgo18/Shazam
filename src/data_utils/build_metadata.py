"""
src/data_utils/build_metadata.py

This script scans the raw audio dataset directory, extracts metadata for each audio file, and saves the results into a parquet file.
For each supported audio file, the script:
    - Generates a unique track ID based on the file path.
    - Computes the audio duration using librosa.
    - Stores metadata into a structured pandas DataFrame.

The final metadata file is saved to: data/processed/metadata.parquet
"""
from pathlib import Path
import hashlib
import librosa
import pandas as pd

RAW_DIR = Path("data/raw")


def generate_track_id(path: Path) -> str:
    """Stable identifier based on the file content (first 8 KB)."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read(8192)).hexdigest()


if __name__ == "__main__":
    rows = []

    for audio_file in RAW_DIR.rglob("*"): # Loop over all audio files in the folder data/raw/.
        # Skip files that are not supported :
        if audio_file.suffix.lower() not in {".wav", ".mp3", ".flac"}:
            continue

        track_id = generate_track_id(audio_file)
        duration = librosa.get_duration(path=audio_file)

        # Append metadata row to list :
        rows.append({
            "track_id": track_id,
            "path": str(audio_file),
            "duration": duration
        })

    df = pd.DataFrame(rows) # Convert collected rows into a pandas DataFrame.

    Path("data/processed").mkdir(parents=True, exist_ok=True)   # Test if the output directory exists.
    df.to_parquet("data/processed/metadata.parquet", index=False)       # Save metadata to parquet file.