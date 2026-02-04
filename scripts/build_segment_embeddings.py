from pathlib import Path
import pandas as pd
import numpy as np

from src import config
from src.audio.loading import load_audio
from src.audio.preprocessing import iter_segments
from src.features.embeddings_audio import embed_segment

FEATURES_DIR = Path("data/features")
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

sr = config.SAMPLE_RATE
win_s = config.SEGMENT_WIN_S
hop_s = config.SEGMENT_HOP_S
min_win = config.SEGMENT_MIN_WIN
method = config.EMBEDDING_METHOD
clap_model_name = getattr(config, "CLAP_MODEL_NAME", None)

META_PATH = Path("data/processed/metadata.parquet")

if not META_PATH.exists():
    raise FileNotFoundError(f"Metadata file not found: {META_PATH}")

df_meta = pd.read_parquet(META_PATH)

print("Tracks:", len(df_meta))
print(df_meta.head())

segments_rows = []
embeddings_list = []
segment_id = 0

for row in df_meta.itertuples(index=False):
    track_id = row.track_id
    path = row.path
    print(f"Processing track {track_id}: {path}")

    waveform, sr = load_audio(path, target_sr=sr)

    for start_s, seg in iter_segments(waveform, sr, win_s, hop_s, min_win):
        print(f"Segment start={start_s:.2f}s shape={seg.shape}")
        emb = embed_segment(seg, sr, method=method, clap_model_name=clap_model_name)
        print("Embedding shape:", emb.shape, "norm:", np.linalg.norm(emb))
        embeddings_list.append(emb)
        segments_rows.append({'segment_id':segment_id, 'track_id':track_id, 'path':str(path), 'start_s':float(start_s)})
        segment_id += 1

        print("segments_rows:", len(segments_rows))
        print("embeddings_list:", len(embeddings_list))

emb_mat = np.vstack(embeddings_list).astype(np.float32)
print(emb_mat.shape)

np.save(FEATURES_DIR / "embeddings.npy", emb_mat)

df_segments = pd.DataFrame(segments_rows)
df_segments.to_parquet(FEATURES_DIR / "segments.parquet", index=False)

