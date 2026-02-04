"""
src/data_utils/build_metadata.py

This script scans the raw audio dataset directory, extracts metadata for each audio file, and saves the results into a CSV file.
For each supported audio file, the script:
    - Generates a unique track ID based on the file path.
    - Computes the audio duration using librosa.
    - Stores metadata into a structured pandas DataFrame.

The final metadata file is saved to: data/processed/metadata.csv
"""
from pathlib import Path
import hashlib
import librosa
import pandas as pd

RAW_DIR = Path("data/raw")

rows=[]

def generate_track_id(path: str) -> str:
    return hashlib.md5(path.encode()).hexdigest()


for audio_file in RAW_DIR.rglob("*"): # Loop over all audio files in the folder data/raw/.
    # Skip files that are not supported :
    if audio_file.suffix.lower() not in {".wav", ".mp3", ".flac"}: 
        continue

    path = audio_file.relative_to(Path("."))
    track_id = generate_track_id(str(path))
    duration = librosa.get_duration(path= audio_file)

    # Append metadata row to list :
    rows.append({
        "track_id": track_id,
        "path": str(path),
        "duration": duration
    })

df = pd.DataFrame(rows) # Convert collected rows into a pandas DataFrame.

Path("data/processed").mkdir(parents=True, exist_ok=True)   # Test if the output directory exists.
df.to_csv("data/processed/metadata.csv", index=False)       # Save metadata to CSV file.