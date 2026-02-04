from pathlib import Path
from src import config
import pandas as pd
from pathlib import Path

FEATURES_DIR = Path("data/features")
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

sr = config.SAMPLE_RATE
win_s = config.SEGMENT_WIN_S
hop_s = config.SEGMENT_HOP_S
min_win = config.SEGMENT_MIN_WIN
method = config.EMBEDDING_METHOD

META_PATH = Path("data/processed/metadata.csv")
df_meta = pd.read_csv(META_PATH)

required_cols = {"track_id", "path"}

print("Tracks:", len(df_meta))
print(df_meta.head())
