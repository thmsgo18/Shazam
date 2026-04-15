"""
src/config.py

This file allows you to group together all the main configurations of the application.
These different environment variables are divided into 7 categories :
    - Paths
    - Audio
    - Segmentation
    - Features
    - Vector search
    - Embedding
    - Optimisations
"""


# Paths
RAW_DIR              = "data/raw"
PROCESSED_DIR        = "data/processed"
METADATA_PATH        = "data/processed/metadata.parquet"
FEATURES_DIR         = "data/features"
INDEX_DIR            = "data/index"
CHROMA_DIR           = "data/chroma"
FINGERPRINTS_DB      = "data/features/fingerprints.db"

# Audio
SAMPLE_RATE = 22050

# Segmentation
SEGMENT_WIN_S = 5.0
SEGMENT_HOP_S = 3.0
SEGMENT_MIN_WIN = 0.8

# Features — MFCC
N_MFCC      = 20
MFCC_N_FFT       = 2048
MFCC_HOP_LENGTH  = 512

# Features — Fingerprinting
FP_N_FFT              = 2048  # STFT window size
FP_HOP_LENGTH         = 512   # Step between STFT windows
FP_NEIGHBORHOOD       = (20, 15)  # Neighborhood size for peak detection (freq, time)
FP_THRESHOLD_PERCENTILE = 80  # Minimum threshold to keep only significant peaks
FP_FAN_OUT            = 5     # Max number of pairs per peak
FP_MIN_DELTA_T        = 3     # Minimum temporal distance between two peaks (in frames)
FP_MAX_DELTA_T        = 50    # Maximum temporal distance between two peaks (in frames)

# Vector search
VECTOR_TOP_K_SEGMENTS = 200   # Number of candidate segments retrieved from FAISS per query segment
VECTOR_TOP_N_TRACKS   = 50    # Number of unique tracks that pass to Stage 2 (fingerprinting)
VECTOR_TOP_N_RESULTS  = 10    # Number of final results returned to the interface

INDEX_TYPE = "flat" # Options are: "flat", "hnsw", "ivf"

# Embedding
EMBEDDING_METHOD = "clap"                                   # "mfcc" or "clap" or "muq" or "mert"
CLAP_MODEL_NAME  = "laion/clap-htsat-unfused"        # "laion/clap-htsat-unfused" or "laion/larger_clap_music"
CLAP_SAMPLE_RATE = 48000

MUQ_SAMPLE_RATE  = 24000
MUQ_MODEL_NAME   = "OpenMuQ/MuQ-large-msd-iter"
MUQ_BATCH_SIZE   = 8

CLAP_BATCH_SIZE  = 32   # Adjust according to GPU: too large = counterproductive on MPS

# Features — MERT
MERT_MODEL_NAME  = "m-a-p/MERT-v1-95M"   # Or "m-a-p/MERT-v1-330M" (larger, slower)
MERT_SAMPLE_RATE = 24000
MERT_BATCH_SIZE  = 8

# Parallel downloads (python manage.py ingest)
# Increase if fast connection, decrease if frequent YouTube bans
DOWNLOAD_WORKERS = 5

# RIR Augmentation
# RIR_SOURCE: "synthetic" → mathematical RIRs generated on the fly
#             "mit"       → real WAV RIRs in RIR_MIT_DIR, selected by RT60 diversity
RIR_SOURCE  = "synthetic"   # "synthetic" | "mit"
RIR_N       = 5             # number of RIRs per track
RIR_MIT_DIR = "data/rir"    # folder containing MIT WAV files (if RIR_SOURCE = "mit")

# Optimizations
# Set to False to revert to default behavior without optimizations
OPT_FLOAT16              = True   # Load CLAP/MuQ in half-precision (float16) → less RAM, faster
OPT_BATCH_EMBED          = True   # Embed all segments in a single batch in identify_track
OPT_FINGERPRINT_PARALLEL = True   # Load and fingerprint candidates in parallel (Stage 2)
OPT_QUERY_DENOISE        = False  # Spectral denoise noisereduce (non-stationary) on the audio query
FINGERPRINT_CACHE_MAX    = 256    # Max size of LRU fingerprints cache in RAM

# Display
PROGRESS_DATASET  = True   # Global progress bar across all tracks
PROGRESS_TRACK    = True   # Progress bar per track (segments)

# Web interface (webapp/backend/server.py)
UI_LISTEN_DURATION  = 15          # Microphone recording duration in seconds
UI_CONFIDENCE_RATIO = 2.5         # score[0]/score[1] ratio for a certain result


def get_collection_key(method: str = None) -> str:
    """
    Returns the unique method+model key used to name:
      - the ChromaDB collection  (e.g., "clap_larger_clap_music")
      - the FAISS index          (e.g., "index_clap_larger_clap_music_flat.faiss")
      - the segments parquet     (e.g., "segments_clap_larger_clap_music.parquet")

    The key is filesystem-safe (no '/', ':', '-' → replaced by '_').
    """
    if method is None:
        method = EMBEDDING_METHOD
    if method == "clap":
        model_slug = CLAP_MODEL_NAME.split("/")[-1].replace("-", "_")
        return f"clap_{model_slug}"
    if method == "muq":
        model_slug = MUQ_MODEL_NAME.split("/")[-1].replace("-", "_")
        return f"muq_{model_slug}"
    if method == "mert":
        model_slug = MERT_MODEL_NAME.split("/")[-1].replace("-", "_")
        return f"mert_{model_slug}"
    return method  # mfcc: no external model
