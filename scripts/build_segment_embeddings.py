from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import time
import torch
import sys

import librosa
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress, BarColumn, TextColumn,
    TimeElapsedColumn, MofNCompleteColumn, SpinnerColumn
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.audio.loading import load_audio
from src.audio.preprocessing import iter_segments
from src.features.embeddings_audio import embed_segment
from src.features.embeddings_audio import muq_batch_embeddings
from src.features.fingerprint import extract_fingerprint

if __name__ == "__main__":
    t0 = time.time()
    torch.set_num_threads(4)

    console = Console()

    FEATURES_DIR = Path("data/features")
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    default_sr = config.SAMPLE_RATE
    win_s      = config.SEGMENT_WIN_S
    hop_s      = config.SEGMENT_HOP_S
    min_win    = config.SEGMENT_MIN_WIN
    method     = config.EMBEDDING_METHOD
    clap_model_name = config.CLAP_MODEL_NAME
    muq_model_name  = config.MUQ_MODEL_NAME
    batch_size      = config.MUQ_BATCH_SIZE

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

    # Lire la source CSV si disponible (écrite par download_music.py)
    source_path = Path("data/processed/source.txt")
    source_label = source_path.read_text().strip() if source_path.exists() else str(META_PATH)

    # Panneau d'information affiché au démarrage
    console.print(Panel(
        f"[bold]Source   :[/bold] {source_label}\n"
        f"[bold]Méthode  :[/bold] [cyan]{method}[/cyan]\n"
        f"[bold]Tracks   :[/bold] {len(df_meta)}\n"
        f"[bold]Segments :[/bold] fenêtre {win_s}s / pas {hop_s}s",
        title="[bold cyan]Build Segment Embeddings[/bold cyan]",
        expand=False
    ))

    segments_rows  = []
    embeddings_list = []
    fingerprints   = {}
    segment_id     = 0
    sr             = load_sr

    batch_segments = []
    batch_meta     = []

    # Construction des colonnes des barres de progression
    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    ]

    with Progress(*progress_columns, console=console) as progress:

        # Barre globale dataset
        dataset_task = (
            progress.add_task("[cyan]Dataset", total=len(df_meta))
            if config.PROGRESS_DATASET else None
        )

        for row in df_meta.itertuples(index=False):
            track_id = row.track_id
            path     = row.path
            filename = Path(path).name

            waveform, sr = load_audio(path, target_sr=load_sr)

            # Calcul du fingerprint (toujours à SAMPLE_RATE)
            if sr != config.SAMPLE_RATE:
                waveform_fp = librosa.resample(waveform, orig_sr=sr, target_sr=config.SAMPLE_RATE)
            else:
                waveform_fp = waveform
            fingerprints[track_id] = extract_fingerprint(waveform_fp, config.SAMPLE_RATE)

            # Pré-calculer les segments pour connaître le total exact
            segs = list(iter_segments(waveform, sr, win_s, hop_s, min_win))

            # Barre par morceau
            track_task = (
                progress.add_task(f"[green]{filename}", total=len(segs))
                if config.PROGRESS_TRACK else None
            )

            for start_s, seg in segs:

                if method == "muq":
                    batch_segments.append(seg)
                    batch_meta.append((track_id, str(path), float(start_s)))

                    if len(batch_segments) >= batch_size:
                        embs = muq_batch_embeddings(
                            batch_segments, sr=sr, model_name=muq_model_name
                        )
                        for i in range(embs.shape[0]):
                            embeddings_list.append(embs[i])
                            t_id, pth, st = batch_meta[i]
                            segments_rows.append({
                                "segment_id": segment_id,
                                "track_id":   t_id,
                                "path":       pth,
                                "start_s":    st
                            })
                            segment_id += 1
                        batch_segments.clear()
                        batch_meta.clear()

                else:
                    emb = embed_segment(
                        seg, sr, method=method,
                        muq_model_name=muq_model_name,
                        clap_model_name=clap_model_name
                    )
                    embeddings_list.append(emb)
                    segments_rows.append({
                        "segment_id": segment_id,
                        "track_id":   track_id,
                        "path":       str(path),
                        "start_s":    float(start_s)
                    })
                    segment_id += 1

                if track_task is not None:
                    progress.advance(track_task)

            # Fin du morceau : retirer la barre de track
            if track_task is not None:
                progress.remove_task(track_task)

            if dataset_task is not None:
                progress.advance(dataset_task)

    # Flush dernier batch incomplet (MuQ)
    if method == "muq" and len(batch_segments) > 0:
        embs = muq_batch_embeddings(batch_segments, sr=sr, model_name=muq_model_name)
        for i in range(embs.shape[0]):
            embeddings_list.append(embs[i])
            t_id, pth, st = batch_meta[i]
            segments_rows.append({
                "segment_id": segment_id,
                "track_id":   t_id,
                "path":       pth,
                "start_s":    st
            })
            segment_id += 1
        batch_segments.clear()
        batch_meta.clear()

    emb_mat = np.vstack(embeddings_list).astype(np.float32)

    assert emb_mat.shape[0] == len(segments_rows), (
        f"Mismatch: {emb_mat.shape[0]} embeddings vs {len(segments_rows)} segment rows"
    )

    np.save(FEATURES_DIR / f"embeddings_{method}.npy", emb_mat)
    df_segments = pd.DataFrame(segments_rows)
    df_segments.to_parquet(FEATURES_DIR / f"segments_{method}.parquet", index=False)

    fp_path = Path(config.FINGERPRINTS_PATH)
    with open(fp_path, "wb") as f:
        pickle.dump(fingerprints, f)

    dt = time.time() - t0
    console.print(Panel(
        f"[bold]Total segments  :[/bold] {len(embeddings_list)}\n"
        f"[bold]Embedding dim   :[/bold] {emb_mat.shape[1]}\n"
        f"[bold]Fingerprints    :[/bold] {len(fingerprints)} tracks → {fp_path}\n"
        f"[bold]Temps total     :[/bold] {dt:.1f}s",
        title="[bold green]Terminé[/bold green]",
        expand=False
    ))
