from pathlib import Path
import pandas as pd
import numpy as np
import time
import torch

from src import config
from src.audio.loading import load_audio
from src.audio.preprocessing import iter_segments
from src.features.embeddings_audio import embed_segment
from src.features.embeddings_audio import muq_batch_embeddings

if __name__ == "__main__":
    t0 = time.time()
    torch.set_num_threads(4)

    FEATURES_DIR = Path("data/features")
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    default_sr = config.SAMPLE_RATE
    win_s = config.SEGMENT_WIN_S
    hop_s = config.SEGMENT_HOP_S
    min_win = config.SEGMENT_MIN_WIN
    method = config.EMBEDDING_METHOD
    clap_model_name = config.CLAP_MODEL_NAME
    muq_model_name = config.MUQ_MODEL_NAME
    batch_size = config.MUQ_BATCH_SIZE

    if method == "clap":
        load_sr = int(getattr(config, "CLAP_SAMPLE_RATE", 48000))
    elif method == "muq":
        load_sr = int(getattr(config, "MUQ_SAMPLE_RATE", 24000))
    else:
        load_sr = int(default_sr)

    META_PATH = Path("data/processed/metadata.parquet")

    if not META_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {META_PATH}")

    df_meta = pd.read_parquet(META_PATH)

    print("Tracks:", len(df_meta))
    print(df_meta.head())

    segments_rows = []
    embeddings_list = []
    segment_id = 0
    sr = load_sr  # valeur par défaut au cas où le dataset est vide

    batch_segments = []
    batch_meta = []  # tuples (track_id, path, start_s)

    for row in df_meta.itertuples(index=False):
        track_id = row.track_id
        path = row.path
        print(f"Processing track {track_id}: {path}")

        waveform, sr = load_audio(path, target_sr=load_sr)

        for start_s, seg in iter_segments(waveform, sr, win_s, hop_s, min_win):

            if method == "muq":
                batch_segments.append(seg)
                batch_meta.append((track_id, str(path), float(start_s)))

                # Quand batch plein -> 1 forward MuQ pour tout le batch
                if len(batch_segments) >= batch_size:
                    embs = muq_batch_embeddings(
                        batch_segments, sr=sr, model_name=muq_model_name
                    )  # (B, 1024)

                    # On "déplie" embeddings + metadata
                    for i in range(embs.shape[0]):
                        embeddings_list.append(embs[i])
                        t_id, pth, st = batch_meta[i]
                        segments_rows.append({
                            "segment_id": segment_id,
                            "track_id": t_id,
                            "path": pth,
                            "start_s": st
                        })
                        segment_id += 1

                    batch_segments.clear()
                    batch_meta.clear()

                    if segment_id % 100 == 0:
                        dt = time.time() - t0
                        print(f"{segment_id} segments | {dt:.1f}s | {segment_id/dt:.2f} seg/s")

            else:
                # MFCC / CLAP (ou autre) : 1 segment -> 1 embedding
                emb = embed_segment(
                    seg, sr, method=method,
                    muq_model_name=muq_model_name,
                    clap_model_name=clap_model_name
                )
                embeddings_list.append(emb)
                segments_rows.append({
                    "segment_id": segment_id,
                    "track_id": track_id,
                    "path": str(path),
                    "start_s": float(start_s)
                })
                segment_id += 1

                if segment_id % 100 == 0:
                    dt = time.time() - t0
                    print(f"{segment_id} segments | {dt:.1f}s | {segment_id/dt:.2f} seg/s")

    # Flush last incomplete batch (MuQ)
    if method == "muq" and len(batch_segments) > 0:
        embs = muq_batch_embeddings(batch_segments, sr=sr, model_name=muq_model_name)

        for i in range(embs.shape[0]):
            embeddings_list.append(embs[i])
            t_id, pth, st = batch_meta[i]
            segments_rows.append({
                "segment_id": segment_id,
                "track_id": t_id,
                "path": pth,
                "start_s": st
            })
            segment_id += 1

        batch_segments.clear()
        batch_meta.clear()

    emb_mat = np.vstack(embeddings_list).astype(np.float32)

    print("Total segments:", len(embeddings_list))
    print("Embedding dim:", emb_mat.shape[1])

    assert emb_mat.shape[0] == len(segments_rows), (
        f"Mismatch: {emb_mat.shape[0]} embeddings vs {len(segments_rows)} segment rows"
    )

    np.save(FEATURES_DIR / f"embeddings_{method}.npy", emb_mat)

    df_segments = pd.DataFrame(segments_rows)
    df_segments.to_parquet(FEATURES_DIR / f"segments_{method}.parquet", index=False)
