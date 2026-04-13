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
VECTOR_TOP_K_SEGMENTS = 200   # Nb de segments candidats récupérés depuis FAISS par segment requête
VECTOR_TOP_N_TRACKS   = 50    # Nb de tracks uniques qui passent en Stage 2 (fingerprinting)
VECTOR_TOP_N_RESULTS  = 10    # Nb de résultats finaux retournés à l'interface

INDEX_TYPE = "flat" # Options are : "flat", "hnsw", "ivf"

# Embedding
EMBEDDING_METHOD = "clap"                           # "mfcc" ou "clap" ou "muq" ou "mert"
CLAP_MODEL_NAME  = "laion/clap-htsat-unfused"        # "laion/clap-htsat-unfused" ou "laion/larger_clap_music"
CLAP_SAMPLE_RATE = 48000

MUQ_SAMPLE_RATE  = 24000
MUQ_MODEL_NAME   = "OpenMuQ/MuQ-large-msd-iter"
MUQ_BATCH_SIZE   = 8

CLAP_BATCH_SIZE  = 32   # À ajuster selon le GPU : trop grand = contre-productif sur MPS

# Features — MERT
MERT_MODEL_NAME  = "m-a-p/MERT-v1-95M"   # Ou "m-a-p/MERT-v1-330M" (plus grand, plus lent)
MERT_SAMPLE_RATE = 24000
MERT_BATCH_SIZE  = 8

# Téléchargements parallèles (download_music.py)
# Augmenter si connexion rapide, réduire si bans YouTube fréquents
DOWNLOAD_WORKERS = 5

# Optimisations
# Mettre à False pour revenir au comportement de base sans optimisations
OPT_FLOAT16              = True   # Charger CLAP/MuQ en demi-précision (float16) → moins de RAM, plus rapide
OPT_BATCH_EMBED          = True   # Embedder tous les segments en un seul batch dans identify_track
OPT_FINGERPRINT_PARALLEL = True   # Charger et fingerprinter les candidats en parallèle (Stage 2)
OPT_QUERY_DENOISE        = False  # Débruitage spectral noisereduce (non-stationnaire) sur la requête audio

# Affichage
PROGRESS_DATASET  = True   # Barre de progression globale sur l'ensemble des tracks
PROGRESS_TRACK    = True   # Barre de progression par morceau (segments)

# Interface web (webapp/backend/server.py)
UI_LISTEN_DURATION  = 15          # Durée d'enregistrement micro en secondes
UI_CONFIDENCE_RATIO = 2.5         # Ratio score[0]/score[1] pour un résultat certain


def get_collection_key(method: str = None) -> str:
    """
    Retourne la clé unique méthode+modèle utilisée pour nommer :
      - la collection ChromaDB  (ex. "clap_larger_clap_music")
      - l'index FAISS           (ex. "index_clap_larger_clap_music_flat.faiss")
      - le parquet segments     (ex. "segments_clap_larger_clap_music.parquet")

    La clé est filesystem-safe (pas de '/', ':', '-' → remplacés par '_').
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
    return method  # mfcc : pas de modèle externe
