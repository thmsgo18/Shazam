# Shazam

<div align="right">
  <img src="https://img.shields.io/badge/EN-current-222?style=flat-square" alt="English (current)">
  &nbsp;
  <a href="./README.fr.md"><img src="https://img.shields.io/badge/FR-version-0055A4?style=flat-square&labelColor=EF4135" alt="Lire en Français"></a>
</div>

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-vector_search-0064A5)
![ChromaDB](https://img.shields.io/badge/ChromaDB-embeddings-FF6B35)
![SQLite](https://img.shields.io/badge/SQLite-fingerprints-003B57?logo=sqlite&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React_18-61DAFB?logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-646CFF?logo=vite&logoColor=white)
![CLAP](https://img.shields.io/badge/CLAP-laion-green)
![MuQ](https://img.shields.io/badge/MuQ-OpenMuQ-blueviolet)
![License](https://img.shields.io/badge/License-Master_IAD_S2-lightgrey)

A music recognition system inspired by Shazam, developed as part of a Big Data project (Master IAD, S2). Given an audio clip captured from a microphone or uploaded as a file, the system identifies the matching track from a database of several hundred titles and displays streaming links.

The approach combines the power of deep learning embeddings (FAISS vector search) with spectral fingerprinting inspired by the original Shazam patent, forming a hybrid two-stage pipeline that remains robust against noise, reverberation, and short audio clips.

---

## Table of Contents

- [How It Works](#how-it-works)
  - [Fingerprinting in Detail](#fingerprinting-in-detail)
- [Project Architecture](#project-architecture)
- [Embedding Methods](#embedding-methods)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data](#data)
- [Quick Start](#quick-start)
- [Essential Commands](#essential-commands)
- [Configuration](#configuration)
- [Web Interface](#web-interface)
- [RIR Augmentation](#rir-augmentation)
- [Evaluation](#evaluation)
- [Data Storage](#data-storage)
- [Important Notes](#important-notes)
- [Technologies](#technologies)
- [Future Improvements](#future-improvements)
- [References](#references)
- [Team](#team)

---

## How It Works

The identification pipeline runs in two successive stages.

```
Query audio
      │
      ▼
┌─────────────────────────────────────────┐
│  Stage 1 — Vector Search                │
│                                         │
│  Split into 5s windows                  │
│  → Embedding (MFCC / CLAP / MuQ / MERT) │
│  → FAISS search (cosine)                │
│  → Top candidate tracks                 │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Stage 2 — Shazam Fingerprinting        │
│                                         │
│  Spectral constellation of the query    │
│  ↔ SQLite fingerprints of candidates    │
│  → Temporal alignment (histogram)       │
│  → Temporal coherence score             │
└────────────────┬────────────────────────┘
                 │
                 ▼
      Final ranking
      (FP score first, FAISS as tiebreaker)
```

**Stage 1** transforms each 5-second window into an embedding vector via a pre-trained model, then queries the FAISS index to find the nearest segments in vector space. Results are aggregated by track to produce the `VECTOR_TOP_N_TRACKS` (default: 50) best candidates.

**Stage 2** applies Shazam fingerprinting: extraction of a constellation of spectral peaks, comparison with fingerprints stored in SQLite, and temporal alignment via an offset histogram. This score is more discriminative than cosine similarity and more robust to audio degradation.

The **final score** sorts first by fingerprint score (ground truth), then by FAISS score as a tiebreaker. If fingerprinting fails for all candidates (heavily degraded audio), the ranking falls back to FAISS scores alone.

### Fingerprinting in Detail

The fingerprinting is an implementation of the original Shazam patent (Wang, 2003). It operates in three steps:

```
Audio signal (22 050 Hz)
        │
        ▼
┌───────────────────────────────┐
│  Spectrogram (STFT)           │
│                               │
│  Frequency                    │
│    ^   *       *              │
│    │      *  *    *    *      │  ← spectral peaks
│    │   *           *          │    (constellation map)
│    └──────────────────> Time  │
└──────────────┬────────────────┘
               │  local peak detection
               ▼
┌───────────────────────────────┐
│  Hash generation              │
│                               │
│  Anchor (f1, t1)              │
│      └─── target (f2, t2)     │
│                               │
│  hash = (f1, f2, Δt, t1)      │  ← 4-tuple per pair
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│  Temporal alignment           │
│                               │
│  For each shared hash         │
│  between query and candidate: │
│  offset = t1_base − t1_query  │
│                               │
│  Offset histogram →           │
│  peak = temporal coherence    │
└───────────────────────────────┘
```

Each hash encodes a peak pair `(anchor_freq, target_freq, delta_time)` along with the anchor's temporal position `t1`. A 3-minute track typically generates several thousand hashes, stored in SQLite as pickled BLOBs.

Temporal alignment is the key to robustness: two different tracks may share a few hashes by chance, but only the correct track will show a clear peak in the offset histogram — all its matching hashes align at the same temporal offset.

---

## Project Architecture

```
Project/
├── manage.py                        # Single entry point — all commands
├── src/
│   ├── config.py                    # Centralized parameters (READ FIRST)
│   ├── ingestion/
│   │   ├── ingest.py                # Download → embeddings → index pipeline
│   │   ├── augment_rir.py           # Room Impulse Response augmentation
│   │   └── fingerprints.py          # Fingerprint rebuilding
│   ├── features/
│   │   └── embeddings_audio.py      # MFCC / CLAP / MuQ / MERT extraction
│   ├── index/
│   │   └── build_index.py           # FAISS index construction
│   ├── retrieval/
│   │   ├── query_pipeline.py        # Stage 1 + Stage 2 orchestration
│   │   └── searcher.py              # FAISS search + aggregation
│   ├── evaluation/
│   │   ├── evaluate.py              # Top-1 / Top-5 / MRR / latency metrics
│   │   ├── rir_impact.py            # RIR impact analysis on a single file
│   │   └── plots.py                 # PNG chart generation
│   └── maintenance/
│       ├── check.py                 # Data integrity verification
│       ├── enrich.py                # Metadata enrichment (Deezer / MusicBrainz)
│       └── clean.py                 # Clean track deletion
├── webapp/
│   ├── backend/server.py            # FastAPI (3 routes)
│   └── frontend/                    # React 18 + Vite (SPA)
│       └── src/
│           ├── App.jsx              # Global state, view routing
│           └── components/          # ListenButton, DropZone, ResultView…
├── data/                            # Persistent data (git-ignored)
│   ├── chroma/                      # Embedding vectors (ChromaDB)
│   ├── features/fingerprints.db     # Shazam fingerprints (SQLite, WAL mode)
│   ├── index/                       # FAISS index + Parquet segments
│   ├── processed/metadata.parquet   # Enriched track metadata
│   ├── raw/                         # Test audio files
│   └── rir/                         # Room Impulse Response WAV files (optional)
├── results/
│   ├── EXPERIMENTS.md               # Experiment log (versioned)
│   ├── eval/                        # Evaluation JSON files (git-ignored)
│   ├── benchmark/                   # Benchmark JSON files (git-ignored)
│   └── plots/                       # PNG charts for the report (git-ignored)
└── research_paper/                  # Reference papers (PDF)
```

---

## Embedding Methods

Four methods are available and can coexist in the same database. The active method is defined by `EMBEDDING_METHOD` in `src/config.py`.

| Method | Model | Dim. | Sample Rate | Compatibility | Notes |
|--------|-------|------|-------------|---------------|-------|
| `mfcc` | — (librosa) | 40 | 22 050 Hz | CPU | Fast, no model dependency |
| `clap` | `laion/clap-htsat-unfused` | 512 | 48 000 Hz | CUDA, MPS, CPU | General-purpose, good quality/speed tradeoff |
| `clap` | `laion/larger_clap_music` | 512 | 48 000 Hz | CUDA, MPS, CPU | Music-specialized, higher precision |
| `muq` | `OpenMuQ/MuQ-large-msd-iter` | 1 024 | 24 000 Hz | CUDA only | Best precision, requires NVIDIA GPU |
| `mert` | `m-a-p/MERT-v1-95M` | 768 | 24 000 Hz | CUDA, MPS | Music representation model |

Each track stores already-computed methods in the `embedded_methods` field of `metadata.parquet`. Re-running `ingest` after a method change only computes what is missing — existing tracks are not re-downloaded.

---

## Requirements

- **Python 3.10** (some dependencies are not compatible with 3.11+)
- **Node.js 18+** (for the React frontend)
- **ffmpeg** installed and available in PATH
- **yt-dlp** (installed via `requirements.txt`)
- GPU recommended for CLAP / MuQ / MERT (CPU works for MFCC and CLAP)

---

## Installation

```bash
# 1. Clone the repository
git clone <url>
cd Projet

# 2. Create and activate the virtual environment
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install frontend dependencies
cd webapp/frontend && npm install && cd ../..
```

---

## Data

Music data comes from Spotify charts via Kaggle. The pipeline automatically downloads audio from YouTube into RAM (no MP3 stored on disk) using the artist and track names from the CSV.

**Expected CSV columns:**

| Column | Description |
|--------|-------------|
| `track_name` | Track title |
| `artist_names` | Artist name |

Kaggle `spotify-streaming-top-50-*.csv` files are directly compatible. Place them in `data/kaggle/data/`.

**Dataset used:** [Spotify Streaming Top 50 — Kaggle](https://www.kaggle.com/datasets/anxods/spotify-top-50-playlist-songs-anxods) — daily Top 50 charts worldwide and by country.

**Automatic deduplication:** a track appearing in multiple CSVs (global hits in the France, US, and World charts) is processed only once, identified by its `track_id` (MD5 hash of `artist_title`).

---

## Quick Start

```bash
source venv/bin/activate

# One command runs the full pipeline: ingest → augment → enrich
python manage.py build --csv data/kaggle/data/spotify-streaming-top-50-world.csv

# Launch the web interface
python manage.py webapp
# → http://localhost:5173
```

---

## Essential Commands

```bash
# ── Construction ───────────────────────────────────────────────────────────

# Full pipeline in one command: ingest → augment → enrich
python manage.py build --csv data/kaggle/data/spotify-streaming-top-50-world.csv

# Or step by step
python manage.py ingest --csv data/kaggle/data/spotify-streaming-top-50-world.csv
python manage.py augment
python manage.py enrich

# ── Identification ─────────────────────────────────────────────────────────

# Identify a track (simple output)
python manage.py identify data/raw/my_audio.mp3

# With score details and top 10
python manage.py identify data/raw/my_audio.mp3 --top 10 --detailed

# Download a test clip (saved to data/raw/ + registered in manifest)
python manage.py download-test "Daft Punk Get Lucky" --duration 30 --position middle

# ── Web interface ──────────────────────────────────────────────────────────

python manage.py webapp                    # dev  → http://localhost:5173
python manage.py webapp --prod             # prod → http://localhost:8000

# ── Maintenance ────────────────────────────────────────────────────────────

# Check active configuration
python manage.py config

# Check data integrity
python manage.py check
python manage.py check --details           # detailed warnings by category
python manage.py check --purge --yes       # delete corrupted tracks

# Rebuild after a purge
python manage.py rebuild --what index
```

> The exhaustive reference for all commands and options is in **[COMMANDS.md](./COMMANDS.md)**.

---

## Configuration

All parameters are centralized in `src/config.py`. No constant should be duplicated elsewhere.

```python
# ── Active embedding method ────────────────────────────────────────────────
EMBEDDING_METHOD = "clap"       # "mfcc" | "clap" | "muq" | "mert"

# ── Identification pipeline ────────────────────────────────────────────────
VECTOR_TOP_N_TRACKS = 50        # candidate tracks aggregated by Stage 1 and passed to Stage 2
SEGMENT_WIN_S       = 5.0       # audio window duration (seconds)
SEGMENT_HOP_S       = 3.0       # step between windows (seconds)

# ── RIR augmentation ───────────────────────────────────────────────────────
RIR_SOURCE  = "synthetic"       # "synthetic" | "mit"
RIR_N       = 5                 # number of RIRs applied per track
RIR_MIT_DIR = "data/rir"        # MIT WAV directory (if RIR_SOURCE = "mit")

# ── Web interface ──────────────────────────────────────────────────────────
UI_LISTEN_DURATION  = 15        # microphone recording duration (seconds)
UI_CONFIDENCE_RATIO = 2.5       # score[0]/score[1] ratio → "confident" badge

# ── Download ───────────────────────────────────────────────────────────────
DOWNLOAD_WORKERS = 5            # parallel yt-dlp workers
```

---

## Web Interface

The web interface allows track recognition directly from the browser, with no additional client-side installation required.

### API Routes (FastAPI)

| Method | Route | Body / Response |
|--------|-------|-----------------|
| `POST` | `/api/identify` | `multipart/form-data` → JSON (results + recommendations) |
| `GET` | `/api/config` | JSON (`listen_duration`, `embedding_method`, `confidence_ratio`) |
| `GET` | `/api/health` | `{"status": "ok"}` |

### Features

- **Microphone recording** — animated countdown, configurable duration via `UI_LISTEN_DURATION`
- **File upload** — drag-and-drop or file picker (MP3, WAV, FLAC, OGG, WebM…)
- **Result** — album art, title, artist, confidence score, direct streaming links (YouTube, Spotify, Deezer, Apple Music)
- **Recommendations** — 4 similar tracks with detail modal and links
- **Debug mode** — `</>` button: top 10 candidates with separate FP and FAISS scores
- **Dark / light theme** and **bilingual interface** EN / FR

### Launch Modes

| Mode | Frontend | Backend | Access |
|------|----------|---------|--------|
| Development | Vite hot-reload `:5173` | uvicorn `--reload` `:8000` | http://localhost:5173 |
| Production | Static build in `dist/` | uvicorn `:8000` | http://localhost:8000 |

---

## RIR Augmentation

A Room Impulse Response (RIR) is the acoustic response of a space to an impulse sound — it captures how a room colors audio (reflections, reverberation, absorption). Convolving a track with a RIR produces a version that sounds as if it were recorded in that space.

The principle: for each track in the database, N reverberant versions are generated via FFT convolution (`scipy.signal.fftconvolve`), their embeddings are computed, and they are added to ChromaDB and the FAISS index. When a query arrives from a reverberant environment, it naturally finds its nearest neighbors among these augmented versions.

**The operation is idempotent:** RIRs already applied to a track are recorded in the `rir_augmented` field of `metadata.parquet` and skipped on subsequent calls. Only missing RIRs are computed.

### Synthetic RIR Generation

Each synthetic RIR is built from three components:

1. **Direct sound** — impulse at 1 ms
2. **Early reflections** — 12 to 30 random reflections with exponential decay based on RT60
3. **Diffuse tail** — decaying Gaussian noise after 50 ms, simulating late reverberation

The 10 predefined environments cover a wide RT60 range:

| Environment | RT60 |
|-------------|------|
| `bathroom` | 0.15 s |
| `small_room` | 0.25 s |
| `bedroom` | 0.35 s |
| `office` | 0.40 s |
| `corridor` | 0.55 s |
| `living_room` | 0.60 s |
| `classroom` | 0.80 s |
| `warehouse` | 0.90 s |
| `large_hall` | 1.20 s |
| `concert_hall` | 1.60 s |

### Available Sources

| Source | Description | Advantages |
|--------|-------------|------------|
| `synthetic` | Generated mathematical RIRs (10 environments, RT60: 0.15 s – 1.60 s) | No download required, reproducible, fast |
| `mit` | Real RIRs measured in actual spaces (MIT Acoustical Survey) | More realistic, requires WAV files in `data/rir/` |

With `source = "mit"`, the system loads all available WAV files, estimates each one's RT60 via Schroeder integration (energy decay of −60 dB from peak), then selects the N most diverse ones by uniform sampling over the sorted RT60 curve — guaranteeing maximum coverage of the acoustic range.

---

## Evaluation

The project includes a comprehensive evaluation suite to measure and compare pipeline performance.

### Metrics

| Metric | Description |
|--------|-------------|
| **Top-1** | Correct track is in the first position |
| **Top-5** | Correct track is within the top 5 results |
| **MRR** | Mean Reciprocal Rank — measures ranking quality |
| **Latency** | Identification time in seconds |

### Degradation Conditions

| Condition | Description |
|-----------|-------------|
| `clean` | No degradation |
| `snr_20` | White noise at 20 dB SNR |
| `snr_10` | White noise at 10 dB SNR (heavy degradation) |
| `reverb` | Simulated reverberation |
| `combo` | 10 dB SNR + reverberation combined |

### Evaluation Workflow

```bash
# 1. Download test clips (30s, middle position recommended)
python manage.py download-test "Miley Cyrus Flowers"        --duration 30 --position middle
python manage.py download-test "Travis Scott PARASAIL"      --duration 30 --position middle
python manage.py download-test "The Weeknd Blinding Lights" --duration 30 --position middle

# 2. Full pipeline evaluation — Top-1, Top-5, MRR, latency
python manage.py eval multi

# 3. RIR impact evaluation (Stage 1 with vs without augmentation)
python manage.py eval rir --n-tracks 0

# 4. Generate all 7 PNG charts for the report
python manage.py eval plots \
  --eval     results/eval/eval_*.json \
  --rir-eval results/eval/rir_eval_*.json
```

### Generated Charts

| File | Chart | Source |
|------|-------|--------|
| `rir_paired_bar_*.png` | G1 — Accuracy with vs without RIR by condition | `eval rir` |
| `rir_delta_*.png` | G2 — Δ gain from RIR augmentation | `eval rir` |
| `rir_faiss_scores_*.png` | G4 — FAISS score per track with/without RIR | `eval rir` |
| `method_accuracy.png` | G6 — Top-1 accuracy per method and condition | `eval multi` |
| `stage_comparison.png` | G9 — Stage 1 (FAISS only) vs Stage 2 (+ fingerprint) | `eval multi` |
| `duration_impact.png` | G11 — Accuracy as a function of clip duration | `eval multi` |
| `heatmap_accuracy.png` | G12 — Methods × conditions heatmap | `eval multi` |

---

## Data Storage

### ChromaDB — `data/chroma/`

Embedding vector database. One collection per method (e.g. `clap_clap_htsat_unfused`). Each document corresponds to a 5-second audio segment, identified by `{track_id}_{segment_index}`, annotated with `track_id` and `start_s`.

### SQLite — `data/features/fingerprints.db`

Shazam spectral fingerprints. One row per track (`INSERT OR REPLACE`, idempotent). WAL mode enabled for concurrent access resistance. Each entry contains spectral constellation hashes serialized as a pickled BLOB.

### FAISS — `data/index/`

Vector search index per method and type:
- `index_{method}_{type}.faiss` — indexed vectors
- `segments_{method}.parquet` — FAISS position → (`track_id`, `start_s`) mapping

Three index types are available, configurable via `INDEX_TYPE` in `config.py` or `--index-type` at build time:

| Type | Algorithm | Precision | Speed | Parameters |
|------|-----------|-----------|-------|------------|
| `flat` | Bruteforce — exact inner product (`IndexFlatIP`) | Exact | Slow | None |
| `hnsw` | Hierarchical Navigable Small World (`IndexHNSWFlat`) | ~99% | Fast | M=32, efConstruction=40 |
| `ivf` | Inverted file lists (`IndexIVFFlat`) | Approximate | Fast | nlist=√N |

`flat` is recommended for databases under 100,000 vectors — the project size does not justify approximation. `hnsw` becomes relevant beyond that. The similarity metric used is **inner product**, embeddings being normalized upstream.

### Parquet — `data/processed/metadata.parquet`

Central track table. Main columns:

| Column | Description |
|--------|-------------|
| `track_id` | MD5 hash of `artist_title` |
| `title`, `artist` | Track identity |
| `album`, `genre`, `release_date` | Enriched metadata |
| `cover_url` | Album art URL |
| `embedded_methods` | List of computed methods |
| `rir_augmented` | Dict `{collection: [rir_names]}` |

Atomic writes (temp file + rename) to survive crashes during ingestion.

---

## Important Notes

**Automatic crash recovery** — `ingest` saves after each track. A sudden interruption loses at most the track in progress. Rerunning the same command resumes exactly where it left off via the `embedded_methods` field.

**Audio in RAM** — audio downloaded via `ingest` is never written to disk. It flows directly into memory via ffmpeg pipe → librosa. Only embeddings, fingerprints, and indexes are persisted.

**Multi-method** — multiple methods can coexist in the database. Change `EMBEDDING_METHOD` in `config.py` and rerun `ingest`: tracks already processed for this method are skipped, others are completed.

**MuQ on Apple Silicon** — MuQ does not support Float16 on CPU or MPS (ComplexFloat operations not supported). It runs exclusively on CUDA. On Mac without an NVIDIA GPU, use `clap` or `mfcc`.

**SQLite and concurrent access** — if the project resides in an iCloud-synced folder, background uploads can lock `fingerprints.db` and cause `database is locked` errors. The `data/` directory is excluded from iCloud via `xattr`. WAL mode and a retry mechanism are enabled to handle residual contention.

**After `check --purge`** — the FAISS index is removed during the purge. Running `python manage.py rebuild --what index` is mandatory before any identification.

---

## Technologies

| Layer | Tools and models |
|-------|-----------------|
| **Audio embeddings** | `laion/clap-htsat-unfused`, `laion/larger_clap_music`, `OpenMuQ/MuQ-large-msd-iter`, `m-a-p/MERT-v1-95M`, librosa (MFCC) |
| **Vector search** | FAISS (Flat / HNSW / IVF), ChromaDB |
| **Fingerprinting** | Shazam spectral constellation — librosa, scipy (FFT, correlation) |
| **Deep learning** | PyTorch (CUDA / MPS / CPU), Transformers (HuggingFace) |
| **Audio** | librosa, soundfile, torchaudio, yt-dlp, ffmpeg |
| **Backend** | FastAPI, uvicorn, pandas, SQLite (WAL), Apache Parquet |
| **Frontend** | React 18, Vite, JavaScript ES6+ |
| **Evaluation** | matplotlib, numpy, scipy |

---

## Future Improvements

The following directions were identified but not implemented within the scope of this project.

**Identification pipeline**
- Cross-normalization of FAISS and fingerprint scores to make relative weight independent of database size
- Adaptive segment windowing: shorter windows for tracks with high temporal variation, longer for stable ones
- In-memory FAISS index caching to eliminate load time across successive identifications
- Real-time audio streaming support (WebSocket) rather than fixed-duration recording

**Embeddings**
- Fine-tuning CLAP or MuQ on a corpus of degraded tracks (noise, reverb) to improve robustness in challenging conditions
- Late fusion of multiple embedding methods (score ensemble) to combine their respective strengths
- Vector quantization (PQ — Product Quantization) to reduce the FAISS index memory footprint

**Augmentation**
- Additional degradation conditions during augmentation: MP3 compression artifacts, tempo variations, pitch shifts
- Use of real ambient noise recordings (cafés, public transport) instead of Gaussian white noise

**Infrastructure**
- Replace ChromaDB with a dedicated vector server (Qdrant, Weaviate, Milvus) to scale to millions of tracks
- Incremental FAISS indexing without full reconstruction after each ingestion
- REST ingestion API to populate the database without direct server access

**Web interface**
- Client-side identification history (localStorage)
- Result sharing via short link
- Synchronized lyrics display (via external API)

---

## References

Reference papers that guided technical decisions are available in the `research_paper/` folder.

| Author(s) | Title | Relevance |
|-----------|-------|-----------|
| Wang, A. (2003) | *An Industrial Strength Audio Search Algorithm* | Foundation of Shazam fingerprinting — spectral constellation and temporal alignment via offset histogram |
| Wu et al. (2023) | *Large-Scale Contrastive Language-Audio Pretraining* (CLAP) | Multimodal audio-text embedding model used in Stage 1 |
| Zhu et al. (2024) | *MuQ: Self-Supervised Music Representation* | Self-supervised music representation model, best precision on musical corpus |
| Castellon et al. | *Music2Latent2: Audio Embedding via Diffusion* | Alternative embedding approach via latent diffusion model |
| Défossez et al. | *A Fast Audio Similarity Retrieval Method* | Large-scale audio similarity retrieval approach |
| Microsoft Research | *Audio Search and Retrieval* | Industrial techniques for large-scale audio search |
| — | *Fast Music Identification* | Comparison of fingerprinting approaches for rapid identification |
| — | *Audio Fingerprinting* | Survey of spectral fingerprinting methods |
| — | *Predicting Song Title from Audio* | Music recognition approaches via supervised learning |

---

## Team

Project completed as part of the Master in Artificial Intelligence and Data Science (IAD), Semester 2 — Big Data track.

| Student | GitHub |
|---------|--------|
| AIT MOKHTAR Clara | [@claraait123](https://github.com/claraait123) |
| AYDIN Maria | [@Mmajora53](https://github.com/Mmajora53) |
| GOURMELEN Thomas | [@thmsgo18](https://github.com/thmsgo18) |
| TAN Vincent | [@20centan](https://github.com/20centan) |
