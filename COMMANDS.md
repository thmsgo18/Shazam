# Command Reference — Shazam Maison

All operations go through `python manage.py <command> [options]`.

> **Prerequisite:** activate the virtual environment before any command.
> ```bash
> source venv/bin/activate
> ```

---

## Table of Contents

| Group | Commands |
|-------|----------|
| [Construction](#construction) | `build` · `ingest` · `augment` · `enrich` |
| [Maintenance](#maintenance) | `check` · `rebuild` · `clean` |
| [Usage](#usage) | `config` · `identify` · `download-test` |
| [Evaluation](#evaluation) | `eval benchmark` · `eval multi` · `eval rir` · `eval plots` |
| [Web interface](#web-interface) | `webapp` |
| [Typical workflows](#typical-workflows) | — |

---

## Construction

### `build` — Full pipeline (recommended entry point)

Runs the complete pipeline in one command: ingest → augment → enrich. All parameters (embedding method, RIR settings, workers…) are read from `src/config.py`.

```bash
python manage.py build [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--csv PATH` | all CSVs in `data/kaggle/` | Kaggle CSV path (repeatable) |
| `--skip-rir` | `False` | Skip the RIR augmentation step |
| `--skip-enrich` | `False` | Skip the metadata enrichment step |

```bash
# Full pipeline from one CSV
python manage.py build --csv data/kaggle/data/spotify-streaming-top-50-world.csv

# Multiple CSVs
python manage.py build \
  --csv data/kaggle/data/spotify-streaming-top-50-france.csv \
  --csv data/kaggle/data/spotify-streaming-top-50-usa.csv

# All CSVs in data/kaggle/, ingest + enrich only (no RIR)
python manage.py build --skip-rir

# Ingest only
python manage.py build --skip-rir --skip-enrich
```

---

### `ingest` — Populate the database

Downloads audio to RAM via yt-dlp, computes embeddings + fingerprints, stores them in ChromaDB + SQLite, then rebuilds the FAISS index. **No MP3 is written to disk.** Resumes automatically from where it left off.

```bash
python manage.py ingest [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--csv PATH` | all CSVs in `data/kaggle/` | Kaggle CSV path (repeatable) |

```bash
# Single CSV
python manage.py ingest --csv data/kaggle/data/spotify-streaming-top-50-world.csv

# Multiple CSVs
python manage.py ingest \
  --csv data/kaggle/data/spotify-streaming-top-50-france.csv \
  --csv data/kaggle/data/spotify-streaming-top-50-usa.csv

# All available Kaggle CSVs (no argument)
python manage.py ingest
```

> Tracks already processed for the active embedding method are skipped (`embedded_methods` field in `metadata.parquet`). Restarting after a crash resumes exactly where it left off.

---

### `augment` — RIR augmentation

Applies Room Impulse Responses to all tracks to generate reverberant versions in ChromaDB. Improves Stage 1 robustness against queries captured in reverberant environments. All parameters come from `src/config.py`.

```bash
python manage.py augment
```

No options — configure via `src/config.py`:

```python
RIR_SOURCE  = "synthetic"   # "synthetic" (generated) | "mit" (real WAV files)
RIR_N       = 5             # number of RIRs applied per track
RIR_MIT_DIR = "data/rir"    # MIT WAV directory (if RIR_SOURCE = "mit")
```

> **Idempotent:** RIRs already applied to a track are recorded in `metadata.parquet` and skipped on subsequent calls.

---

### `enrich` — Enrich metadata

Fills in `metadata.parquet` with `album`, `genre`, `release_date`, and `cover_url` via Deezer (with MusicBrainz as fallback). Does not touch embeddings or fingerprints.

```bash
python manage.py enrich [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--force` | `False` | Re-enrich all tracks, including already-complete ones |
| `--only-missing` | `False` | Process only tracks with at least one empty field |

```bash
# Enrich tracks not yet processed (default)
python manage.py enrich

# Force re-enrichment of all tracks
python manage.py enrich --force

# Only tracks with missing fields
python manage.py enrich --only-missing
```

---

## Maintenance

### `check` — Verify data integrity

Checks consistency across ChromaDB, FAISS, SQLite, and `metadata.parquet`. Optionally deletes problematic tracks.

```bash
python manage.py check [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--details` | `False` | Show per-category warning detail (codes C1–C7, Q1–Q4, FP) |
| `--metadata` | `False` | List tracks with missing or incomplete metadata |
| `--purge` | `False` | Delete all problematic tracks from every store |
| `--purge-missing-fp` | `False` | Purge only tracks with no fingerprint |
| `--yes` / `-y` | `False` | Skip confirmation prompt |

```bash
# Summary view
python manage.py check

# Detailed warnings by category
python manage.py check --details

# Tracks with incomplete metadata (cover, genre…)
python manage.py check --metadata

# Interactive purge
python manage.py check --purge

# Silent purge (scripts / CI)
python manage.py check --purge --yes

# Purge tracks without fingerprint only
python manage.py check --purge-missing-fp --yes
```

**Check codes (`--details`):**

| Code | Level | Description |
|------|-------|-------------|
| `C1` | Critical | Unexpected embedding dimension |
| `C2` | Critical | NaN or Inf in embeddings |
| `C3` | Critical | ChromaDB ↔ metadata desync |
| `C5` | Critical | FAISS index out of sync with ChromaDB |
| `C6` | Critical | Orphan segments (ChromaDB without metadata) |
| `C7` | Critical | Incomplete embedding (< 80% of expected segments) |
| `Q3` | Quality | Empty fingerprint (0 hashes) |
| `FP` | Quality | Track without fingerprint — Stage 2 inoperative for this track |

> After `--purge`, the FAISS index is deleted. Run `python manage.py rebuild --what index` before any identification.

---

### `rebuild` — Rebuild fingerprints and/or FAISS index

Recomputes Shazam fingerprints and/or rebuilds the FAISS index from ChromaDB. Use after a purge, a crash, or a parameter change.

```bash
python manage.py rebuild [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--what` | `all` | `index` = FAISS only · `fps` = fingerprints only · `all` = both |
| `--force` | `False` | Recompute even tracks already in SQLite (fingerprints only) |

```bash
# Rebuild everything (fingerprints + index)
python manage.py rebuild

# FAISS index only (after check --purge)
python manage.py rebuild --what index

# Fingerprints only
python manage.py rebuild --what fps

# Force-recompute all fingerprints from scratch
python manage.py rebuild --what fps --force
```

---

### `clean` — Delete data

Removes a single track, all RIR segments, or the entire database. Always asks for confirmation unless `--yes` is passed.

```bash
python manage.py clean [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--track TRACK_ID` | — | Delete one track from all stores (ChromaDB, SQLite, metadata) |
| `--rir` | `False` | Delete all RIR segments for the active embedding method |
| `--all` | `False` | Full reset — delete everything (ChromaDB, FAISS, SQLite, metadata) |
| `--yes` / `-y` | `False` | Skip confirmation prompt |

```bash
# Delete a specific track
python manage.py clean --track f01ab00f1fdc5a57fd2676f4d68631a8

# Delete a specific track without confirmation
python manage.py clean --track f01ab00f1fdc5a57fd2676f4d68631a8 --yes

# Delete all RIR segments (active method)
python manage.py clean --rir

# Full reset (irreversible)
python manage.py clean --all --yes
```

> After `clean --track`, run `python manage.py rebuild --what index` to update the FAISS index.

---

## Usage

### `config` — Show active configuration

Displays all values from `src/config.py` in the terminal. Useful to verify the active settings before running an ingestion or evaluation.

```bash
python manage.py config
```

No options.

---

### `identify` — Identify an audio file

Runs the full two-stage pipeline (FAISS → fingerprinting) and returns the most likely matches. This is the main CLI command — equivalent to using the web interface.

```bash
python manage.py identify AUDIO [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `AUDIO` | — | Audio file to identify (MP3, WAV, FLAC, OGG…) |
| `--top` | `5` | Number of results to display |
| `--detailed` | `False` | Show FAISS and fingerprint scores separately |
| `--target TRACK_ID` | — | Expected track ID — enables evaluation mode (shows rank of the target) |

```bash
# Simple identification
python manage.py identify data/raw/my_audio.mp3

# Top 10 with score breakdown
python manage.py identify data/raw/my_audio.mp3 --top 10 --detailed

# Evaluation mode: verify the correct track is found
python manage.py identify data/raw/my_audio.mp3 --target f01ab00f1fdc5a57fd2676f4d68631a8
```

**Score interpretation:**
- **FP score** (fingerprint): temporal coherence between query hashes and database hashes. A value > 0 indicates a temporal alignment was found — the higher, the more certain.
- **FAISS score**: cosine similarity in embedding space. Used as a tiebreaker when all FP scores are 0 (heavily degraded audio).

---

### `download-test` — Download a test clip

Downloads a track from YouTube into `data/raw/` and registers it in the test manifest (`data/raw/manifest.json`). The manifest is used as ground truth by `eval benchmark`, `eval multi`, and `eval rir`.

```bash
python manage.py download-test QUERY [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `QUERY` | — | YouTube search query (e.g. `"Daft Punk Get Lucky"`) |
| `--duration` | full track | Clip length in seconds (`5` / `10` / `15` / `30`) |
| `--position` | `start` | Position in the track: `start` · `first-quarter` · `middle` · `third-quarter` · `end` |

```bash
# Full track
python manage.py download-test "Miley Cyrus Flowers"

# 30s clip from the middle (recommended for evaluation)
python manage.py download-test "Miley Cyrus Flowers" --duration 30 --position middle

# Short clip (challenging case)
python manage.py download-test "Daft Punk Get Lucky" --duration 5 --position middle

# Different positions
python manage.py download-test "The Weeknd Blinding Lights" --duration 15 --position first-quarter
python manage.py download-test "The Weeknd Blinding Lights" --duration 15 --position end
```

Output file naming: `Video Title__position_Xs.mp3`
Example: `Miley Cyrus - Flowers (Official Video)__middle_30s.mp3`

> `--position middle` gives the best results — the chorus is acoustically more distinctive than the intro or outro.

---

## Evaluation

### `eval benchmark` — Single-track robustness benchmark

Evaluates pipeline robustness on a single audio file across multiple degradation conditions. The target track is auto-detected from `data/raw/manifest.json`.

```bash
python manage.py eval benchmark [AUDIO] [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `AUDIO` | — | Audio file to benchmark (must be in `manifest.json`) |
| `--label` | timestamp | Run name for traceability |
| `--full` | `False` | Full test suite (~10 cases) instead of the default 4 fast cases |
| `--compare JSON` | — | Compare existing result JSONs without re-running (repeatable) |

```bash
# Quick benchmark (4 cases: clean + SNR20 + light reverb + combo)
python manage.py eval benchmark data/raw/flowers__middle_30s.mp3

# With a label for traceability
python manage.py eval benchmark data/raw/flowers__middle_30s.mp3 --label "clap-v2"

# Full suite (~10 degradation cases)
python manage.py eval benchmark data/raw/flowers__middle_30s.mp3 --full

# Compare two previous runs (no re-execution)
python manage.py eval benchmark \
  --compare results/benchmark/run-clap-v1.json \
  --compare results/benchmark/run-clap-v2.json
```

**Default test cases (fast mode):**

| # | Degradation | Parameters |
|---|-------------|------------|
| 1 | Clean audio | — |
| 2 | White noise | SNR = 20 dB |
| 3 | Light reverb | RT60 ≈ 0.4 s |
| 4 | Combo | SNR 20 dB + reverb |

**Additional cases in `--full` mode:** heavier noise (SNR 10 dB), high-pass filter (simulates phone), Opus compression (64 kbps), short clip (5 s), and more.

---

### `eval multi` — Multi-track evaluation

Evaluates the full pipeline over multiple test tracks and degradation conditions. Computes Top-1, Top-5, MRR, and latency per method × condition. Produces JSON results and PNG charts.

```bash
python manage.py eval multi [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--n-tracks` | `0` (all) | Limit evaluation to N tracks from the manifest |
| `--no-plot` | `False` | Skip PNG chart generation |

Methods and conditions are read from `src/config.py`.

```bash
# Full evaluation (all tracks in manifest)
python manage.py eval multi

# Limit to 5 tracks (quick check)
python manage.py eval multi --n-tracks 5

# Without charts
python manage.py eval multi --no-plot
```

**Output:**
- `results/eval/eval_TIMESTAMP.json` — full metrics
- `results/plots/method_accuracy.png` — G6: accuracy per method × condition
- `results/plots/stage_comparison.png` — G9: Stage 1 vs Stage 2
- `results/plots/duration_impact.png` — G11: accuracy vs clip duration
- `results/plots/heatmap_accuracy.png` — G12: methods × conditions heatmap

**Prerequisite:** test clips registered in the manifest via `download-test`.

---

### `eval rir` — RIR impact evaluation

Evaluates the impact of RIR augmentation on Stage 1 accuracy (with vs. without augmented vectors). Builds a temporary index in memory — **does not modify the database**.

```bash
python manage.py eval rir [AUDIO] [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `AUDIO` | — | Single audio file for single-track analysis |
| `--target TRACK_ID` | auto-detected | Expected track ID (reads manifest if omitted) |
| `--n-tracks` | `0` (all) | Multi-track mode: evaluate N tracks from the manifest |
| `--no-plot` | `False` | Skip PNG chart generation |

```bash
# Single-track RIR analysis
python manage.py eval rir data/raw/flowers__middle_30s.mp3

# Multi-track evaluation (all manifest tracks)
python manage.py eval rir --n-tracks 0

# Limit to 5 tracks
python manage.py eval rir --n-tracks 5
```

**Output:**
- `results/eval/rir_eval_TIMESTAMP.json`
- `results/plots/rir_paired_bar_*.png` — G1: accuracy with vs without RIR
- `results/plots/rir_delta_*.png` — G2: Δ gain from RIR augmentation
- `results/plots/rir_faiss_scores_*.png` — G4: FAISS score per track with/without RIR

---

### `eval plots` — Generate charts from existing results

Reads JSON files produced by `eval multi` and/or `eval rir` and generates PNG charts for the report. Does not re-run any evaluation.

```bash
python manage.py eval plots [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--eval JSON` | — | Evaluation JSON file(s) from `eval multi` (repeatable) |
| `--rir-eval JSON` | — | Evaluation JSON file(s) from `eval rir` (repeatable) |
| `--out-dir PATH` | `results/plots/` | Output directory |

```bash
# Pipeline charts only (G6, G9, G11, G12)
python manage.py eval plots --eval results/eval/eval_*.json

# RIR charts only (G1, G2, G4)
python manage.py eval plots --rir-eval results/eval/rir_eval_*.json

# All 7 charts
python manage.py eval plots \
  --eval     results/eval/eval_*.json \
  --rir-eval results/eval/rir_eval_*.json

# Custom output directory
python manage.py eval plots --eval results/eval/eval_*.json --out-dir /tmp/charts
```

**Charts produced:**

| File | Chart | Source |
|------|-------|--------|
| `rir_paired_bar_*.png` | G1 — Accuracy with vs without RIR | `--rir-eval` |
| `rir_delta_*.png` | G2 — Δ gain from RIR augmentation | `--rir-eval` |
| `rir_faiss_scores_*.png` | G4 — FAISS score per track with/without RIR | `--rir-eval` |
| `method_accuracy.png` | G6 — Top-1 accuracy per method × condition | `--eval` |
| `stage_comparison.png` | G9 — Stage 1 (FAISS only) vs Stage 2 (+ fingerprint) | `--eval` |
| `duration_impact.png` | G11 — Accuracy as a function of clip duration | `--eval` |
| `heatmap_accuracy.png` | G12 — Methods × conditions heatmap | `--eval` |

---

## Web Interface

### `webapp` — Launch the web interface

Starts the FastAPI backend and the React frontend simultaneously.

```bash
python manage.py webapp [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--prod` | `False` | Production mode (static build, single port) |
| `--port` | `8000` | FastAPI backend port |

```bash
# Development mode (Vite hot-reload)
python manage.py webapp
# → http://localhost:5173

# Production mode
python manage.py webapp --prod
# → http://localhost:8000

# Custom port
python manage.py webapp --port 8080
python manage.py webapp --prod --port 9000
```

| Mode | Frontend | Backend | URL |
|------|----------|---------|-----|
| Dev | Vite hot-reload `:5173` | uvicorn `--reload` `:8000` | http://localhost:5173 |
| Prod | Static build in `dist/` | uvicorn `:8000` | http://localhost:8000 |

---

## Typical Workflows

### First-time setup

```bash
source venv/bin/activate

# One command does everything: ingest → augment → enrich
python manage.py build --csv data/kaggle/data/spotify-streaming-top-50-world.csv

# Launch the web interface
python manage.py webapp
# → http://localhost:5173
```

### Add new tracks from a CSV

```bash
python manage.py ingest --csv data/kaggle/data/new_chart.csv
# The FAISS index is rebuilt automatically at the end
```

### Change the embedding method

```bash
# 1. Edit EMBEDDING_METHOD in src/config.py
#    e.g. "mfcc" → "clap"

# 2. Re-run ingest (tracks already processed for clap are skipped)
python manage.py ingest

# 3. The FAISS index is rebuilt automatically
```

### After a purge

```bash
python manage.py check --purge --yes
python manage.py rebuild --what index   # mandatory — the index was deleted
```

### Delete a specific track

```bash
# Find the track_id first
python manage.py identify data/raw/the_track.mp3

# Delete it
python manage.py clean --track <track_id> --yes

# Rebuild the index
python manage.py rebuild --what index
```

### Generate report charts

```bash
# 1. Download test clips
python manage.py download-test "Miley Cyrus Flowers"        --duration 30 --position middle
python manage.py download-test "Travis Scott PARASAIL"      --duration 30 --position middle
python manage.py download-test "The Weeknd Blinding Lights" --duration 30 --position middle

# 2. Full pipeline evaluation (Top-1, Top-5, MRR, latency)
python manage.py eval multi

# 3. RIR impact evaluation
python manage.py eval rir --n-tracks 0

# 4. Generate all 7 PNG charts
python manage.py eval plots \
  --eval     results/eval/eval_*.json \
  --rir-eval results/eval/rir_eval_*.json
```

### Quick health check

```bash
python manage.py config                 # check active settings
python manage.py check                  # data integrity summary
python manage.py check --details        # per-category warning detail
python manage.py check --metadata       # tracks with missing metadata
```
