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

INDEX_TYPE = "flat" # Options are : "flat", "hnsw", "ivf"

# Embedding
EMBEDDING_METHOD = "mfcc"   # "mfcc" ou "clap" ou "muq"
CLAP_MODEL_NAME  = "laion/clap-htsat-unfused"
CLAP_SAMPLE_RATE = 48000

MUQ_SAMPLE_RATE  = 24000
MUQ_MODEL_NAME   = "OpenMuQ/MuQ-large-msd-iter"
MUQ_BATCH_SIZE   = 8

# Téléchargements parallèles (download_music.py)
# Augmenter si connexion rapide, réduire si bans YouTube fréquents
DOWNLOAD_WORKERS = 3

# Optimisations
# Mettre à False pour revenir au comportement de base sans optimisations
OPT_FLOAT16              = True   # Charger CLAP/MuQ en demi-précision (float16) → moins de RAM, plus rapide
OPT_BATCH_EMBED          = True   # Embedder tous les segments en un seul batch dans identify_track
OPT_FINGERPRINT_PARALLEL = True   # Charger et fingerprinter les candidats en parallèle (Stage 2)
OPT_SHORTCIRCUIT         = True   # Court-circuiter le Stage 2 si le 1er candidat FAISS est évident
OPT_SHORTCIRCUIT_RATIO   = 10.0  # Ratio score[0]/score[1] au-delà duquel on court-circuite

# Affichage
PROGRESS_DATASET  = True   # Barre de progression globale sur l'ensemble des tracks
PROGRESS_TRACK    = True   # Barre de progression par morceau (segments)
