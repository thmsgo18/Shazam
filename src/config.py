"""
src/config.py

This file allows you to group together all the main configurations of the application.
These different environment variables are divided into 5 categories :
    - Audio
    - Segmentation
    - Features
    - Vector search
    - Embedding
"""


# Audio
SAMPLE_RATE = 22050

# Segmentation
SEGMENT_WIN_S = 5.0
SEGMENT_HOP_S = 3.0
SEGMENT_MIN_WIN = 0.8

# Features
N_MFCC = 20

# Vector search
VECTOR_TOP_K_SEGMENTS = 200
VECTOR_TOP_N_TRACKS = 20

# Embedding
EMBEDDING_METHOD = "mfcc"   # "mfcc" ou "clap" ou "muq"
CLAP_MODEL_NAME = "laion/clap-htsat-unfused"
CLAP_SAMPLE_RATE = 48000

MUQ_SAMPLE_RATE = 24000
MUQ_MODEL_NAME = "OpenMuQ/MuQ-large-msd-iter"
MUQ_BATCH_SIZE = 8
