"""
src/config.py

This file allows you to group together all the main configurations of the application.
These different environment variables are divided into 6 categories :
    - Paths
    - Audio
    - Segmentation
    - Features
    - Vector search
    - Embedding
"""


# Paths
RAW_DIR          = "data/raw"
PROCESSED_DIR    = "data/processed"
METADATA_PATH    = "data/processed/metadata.parquet"
FEATURES_DIR     = "data/features"
INDEX_DIR        = "data/index"

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
FP_N_FFT              = 2048  # Taille de la fenêtre STFT
FP_HOP_LENGTH         = 512   # Pas entre les fenêtres STFT
FP_NEIGHBORHOOD       = (20, 15)  # Taille du voisinage pour la détection des pics (fréq, temps)
FP_THRESHOLD_PERCENTILE = 80  # Seuil minimum pour ne garder que les pics significatifs
FP_FAN_OUT            = 5     # Nombre max de paires par pic
FP_MIN_DELTA_T        = 3     # Distance temporelle minimale entre deux pics (en frames)
FP_MAX_DELTA_T        = 50    # Distance temporelle maximale entre deux pics (en frames)

# Vector search
VECTOR_TOP_K_SEGMENTS = 200
VECTOR_TOP_N_TRACKS   = 20
VECTOR_TOP_N_RESULTS  = 5

# Embedding
EMBEDDING_METHOD = "mfcc"   # "mfcc" ou "clap" ou "muq"
CLAP_MODEL_NAME  = "laion/clap-htsat-unfused"
CLAP_SAMPLE_RATE = 48000

MUQ_SAMPLE_RATE  = 24000
MUQ_MODEL_NAME   = "OpenMuQ/MuQ-large-msd-iter"
MUQ_BATCH_SIZE   = 8
