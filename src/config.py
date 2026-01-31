# Audio
SAMPLE_RATE = 22050

# Segmentation
SEGMENT_WIN_S = 5.0
SEGMENT_HOP_S = 1.0
SEGMENT_MIN_WIN = 0.8

# Features
N_MFCC = 20

# Vector search
VECTOR_TOP_K_SEGMENTS = 200
VECTOR_TOP_N_TRACKS = 20

# Embedding
EMBEDDING_METHOD = "clap"   # "mfcc" ou "clap"
CLAP_MODEL_NAME = "laion/clap-htsat-unfused"