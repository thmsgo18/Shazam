# Command Reference — Shazam Maison

All operations go through `python manage.py <command> [options]`.

Quick help:
```bash
python manage.py --help
python manage.py <command> --help
```

> **Prerequisite:** activate the virtual environment before any command.
> ```bash
> source venv/bin/activate
> ```

---

## Table of Contents

| Group | Commands |
|-------|----------|
| [Construction](#construction) | `build` · `ingest` · `download-kaggle-csvs` · `augment` · `enrich` |
| [Maintenance](#maintenance) | `check` · `rebuild` · `clean` |
| [Tests](#tests) | `test` |
| [Usage](#usage) | `config` · `identify` · `download-test` |
| [Evaluation](#evaluation) | `eval` · `eval base` · `eval studio-mic` · `eval duration` · `eval stage12` · `eval rir` · `eval mic-conditions` |
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
| `--csv PATH` | all CSVs in `data/kaggle/data/` | Kaggle CSV path (repeatable) |
| `--skip-rir` | `False` | Skip the RIR augmentation step |
| `--skip-enrich` | `False` | Skip the metadata enrichment step |

```bash
# Full pipeline from one CSV
python manage.py build --csv data/kaggle/data/spotify-streaming-top-50-world.csv

# Multiple CSVs
python manage.py build \
  --csv data/kaggle/data/spotify-streaming-top-50-france.csv \
  --csv data/kaggle/data/spotify-streaming-top-50-usa.csv

# All CSVs in data/kaggle/data/, ingest + enrich only (no RIR)
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
| `--csv PATH` | all CSVs in `data/kaggle/data/` | Kaggle CSV path (repeatable) |

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

### `download-kaggle-csvs` — Download the Kaggle CSV files only

Downloads the Kaggle CSV files used by the project into the configured directory `data/kaggle/data/`. This is useful when you want the CSVs in place before running `build` or `ingest`.

```bash
python manage.py download-kaggle-csvs
```

Expected files:

```text
data/kaggle/data/spotify-streaming-top-50-world.csv
data/kaggle/data/spotify-streaming-top-50-france.csv
data/kaggle/data/spotify-streaming-top-50-usa.csv
```

---

### `augment` — RIR augmentation

Applies Room Impulse Responses to all tracks to generate reverberant versions in ChromaDB. Improves Stage 1 robustness against queries captured in reverberant environments. All parameters come from `src/config.py`.

```bash
python manage.py augment [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--tracks TEXT` | `all` | Tracks to augment: `all`, `flowers`, or one specific `track_id` |

Other parameters are configured via `src/config.py`:

```python
RIR_SOURCE  = "synthetic"   # "synthetic" (generated) | "mit" (real WAV files)
RIR_N       = 5             # number of RIRs applied per track
RIR_MIT_DIR = "data/rir"    # MIT WAV directory (if RIR_SOURCE = "mit")
```

```bash
# Augment the whole database
python manage.py augment

# Restrict to Flowers only
python manage.py augment --tracks flowers

# Restrict to one specific track
python manage.py augment --tracks f01ab00f1fdc5a57fd2676f4d68631a8
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

## Tests

### `test` — Run the automated test suite

Runs the Python/backend `unittest` suite located under `tests/`. This is the recommended entry point for local validation.

```bash
python manage.py test [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--path PATH` | `tests` | Restrict discovery to one test directory |
| `--pattern GLOB` | `test*.py` | Restrict discovery to matching files |
| `--unit` | `False` | Run only `tests/unit` |
| `--integration` | `False` | Run only `tests/integration` |
| `--failfast` | `False` | Stop at the first failing test |
| `--buffer` | `False` | Hide stdout/stderr for passing tests |
| `--quiet` | `False` | Lower verbosity |

```bash
# Full suite
python manage.py test

# Cleaner output
python manage.py test --buffer

# Unit tests only
python manage.py test --unit

# Integration tests only, stop on first failure
python manage.py test --integration --failfast

# Narrow to one folder
python manage.py test --path tests/unit/utils
```

> The current suite covers the Python/backend project. Frontend React tests are not part of this command yet.

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

Downloads a track from YouTube into `data/raw/` and registers it in the test manifest (`data/raw/manifest.json`). The manifest is used as ground truth by all `eval` analyses.

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

The evaluation CLI is now organized around **report-oriented analyses**. Each command reads the full test manifest, computes a focused analysis, and produces:

- a JSON payload in `results/eval/`
- a Markdown summary in `results/eval/`
- CSV summaries in `results/eval/`
- PNG figures in `results/plots/` when relevant

By default, all evaluations use the active embedding method from `src/config.py`.

---

### `eval` — Run the base report suite

Runs the shared base evaluation once, then derives the base report analyses from that common table:

- `base-eval-rows`
- `studio-mic`
- `duration`
- `stage12`
- `mic-conditions`

```bash
python manage.py eval [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--n-tracks` | `0` (all) | Limit evaluation to N tracks from the manifest |
| `--no-plot` | `False` | Skip chart generation |

```bash
# Full base evaluation suite on all manifest tracks
python manage.py eval

# Quick run on 3 tracks
python manage.py eval --n-tracks 3

# Metrics only, no PNG figures
python manage.py eval --n-tracks 5 --no-plot
```

**Main outputs:**
- `results/eval/base_eval_rows.json`
- `results/eval/base_eval_rows.md`
- `results/eval/eval_base_summary.json`
- `results/eval/eval_base_summary.md`
- `results/eval/eval_topk_summary_by_category.csv`
- `results/plots/pipeline_resilience_overview.png`
- per-analysis JSON / Markdown

---

### `eval base` — Explicit alias for the base suite

Runs the same shared base suite as `python manage.py eval`.

```bash
python manage.py eval base [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--n-tracks` | `0` (all) | Limit evaluation to N tracks from the manifest |
| `--no-plot` | `False` | Skip chart generation |

```bash
python manage.py eval base
python manage.py eval base --n-tracks 3
```

---

### `eval studio-mic` — Studio vs microphone comparison

Compares clean reference excerpts against real microphone captures for the same tracks.

```bash
python manage.py eval studio-mic [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--n-tracks` | `0` (all) | Limit evaluation to N tracks from the manifest |
| `--no-plot` | `False` | Skip chart generation |

```bash
python manage.py eval studio-mic
python manage.py eval studio-mic --n-tracks 4
```

**Typical figures:**
- `scatter_studio_micro_stage1_vs_final_rank.png`

---

### `eval duration` — Effect of query duration

Analyzes the effect of excerpt duration on retrieval quality, using the studio clips present in the manifest (typically `5s`, `15s`, `30s`).

```bash
python manage.py eval duration [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--n-tracks` | `0` (all) | Limit evaluation to N tracks from the manifest |
| `--no-plot` | `False` | Skip chart generation |

```bash
python manage.py eval duration
python manage.py eval duration --n-tracks 5
```

**Typical figures:**
- `scatter_duration_vs_final_rank.png`

---

### `eval stage12` — Stage 1 vs final pipeline

Compares Stage 1 (FAISS only) with the final ranking after Stage 2 re-ranking.

```bash
python manage.py eval stage12 [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--n-tracks` | `0` (all) | Limit evaluation to N tracks from the manifest |
| `--no-plot` | `False` | Skip chart generation |

```bash
python manage.py eval stage12
python manage.py eval stage12 --n-tracks 5
```

**Typical figures:**
- `scatter_stage1_vs_stage2_rank.png`

---

### `eval rir` — With vs without RIR augmentation

Measures the contribution of the RIR-augmented vectors by comparing retrieval performance with and without them. This analysis is intentionally separate from the shared base suite, because it uses a different comparison path and different index loading logic. It now uses the original queries from `data/raw/manifest.json` as-is, so the comparison is aligned with the main pipeline overview.

```bash
python manage.py eval rir [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--n-tracks` | `0` (all) | Limit evaluation to N tracks from the manifest |
| `--no-plot` | `False` | Skip chart generation |

```bash
python manage.py eval rir
python manage.py eval rir --n-tracks 5
```

**Typical figure:**
- `results/plots/rir_pipeline_overview.png`

**Main outputs:**
- `results/eval/rir_eval.json`
- `results/eval/rir_analysis.json`
- `results/eval/rir_analysis.md`
- `results/eval/rir_topk_summary_by_category.csv`
- `results/eval/rir_topk_summary_by_condition.csv`
- `results/plots/rir_pipeline_overview.png`
- resume cache in `results/eval/cache/`

`eval rir` now keeps only the dedicated overview chart in `results/plots/`:
- `rir_pipeline_overview.png`

---

### `eval mic-conditions` — Microphone condition analysis

Analyzes the impact of microphone recording conditions such as distance (`close`, `normal`, `far`) and concurrent speech (`clean`, `speech`).

```bash
python manage.py eval mic-conditions [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--n-tracks` | `0` (all) | Limit evaluation to N tracks from the manifest |
| `--no-plot` | `False` | Skip chart generation |

```bash
python manage.py eval mic-conditions
python manage.py eval mic-conditions --n-tracks 5
```

**Typical figures:**
- `scatter_studio_vs_micro_rank.png`
- `scatter_micro_clean_vs_speech_rank.png`

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
| `--reload / --no-reload` | `reload` | Enable or disable FastAPI hot reload in dev mode |

```bash
# Development mode (Vite hot-reload)
python manage.py webapp
# → http://localhost:5173

# Production mode
python manage.py webapp --prod
# → http://localhost:8000

# Custom port in development
# Requires updating webapp/frontend/vite.config.js
# because the Vite proxy targets localhost:8000 by default
python manage.py webapp --port 8080

# Custom port in production
python manage.py webapp --prod --port 9000
```

| Mode | Frontend | Backend | URL |
|------|----------|---------|-----|
| Dev | Vite hot-reload `:5173` | uvicorn `--reload` `:8000` | http://localhost:5173 |
| Prod | Static build in `dist/` | uvicorn `:8000` | http://localhost:8000 |

> In dev mode, the frontend proxy is hard-coded to `http://localhost:8000` in `webapp/frontend/vite.config.js`. If you change the backend port, update that proxy target too.

---

## Typical Workflows

### First-time setup

```bash
source venv/bin/activate

# Download the Kaggle CSVs into data/kaggle/data/
python manage.py download-kaggle-csvs

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

### Run the report evaluations

```bash
# 1. Shared base suite (single evaluation pass reused by multiple analyses)
python manage.py eval

# 2. Explicit alias if you want to be clear in scripts
python manage.py eval base

# 3. Focus on one analysis only
python manage.py eval studio-mic
python manage.py eval duration
python manage.py eval stage12
python manage.py eval mic-conditions

# 4. Run the separate RIR comparison
python manage.py eval rir

# 5. Quick run on a subset of tracks
python manage.py eval --n-tracks 3
```

### Quick health check

```bash
python manage.py config                 # check active settings
python manage.py check                  # data integrity summary
python manage.py check --details        # per-category warning detail
python manage.py check --metadata       # tracks with missing metadata
```
