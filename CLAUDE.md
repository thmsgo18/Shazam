# CLAUDE.md — Shazam Maison

Système de reconnaissance musicale inspiré de Shazam, développé dans le cadre d'un projet de Big Data (Master IAD S2). Pipeline hybride en deux étapes : recherche vectorielle (FAISS) + fingerprinting spectral (Shazam-style).

---

## Architecture globale

```
Project/
├── src/                        # Code source Python (logique métier)
│   ├── config.py               # Paramètres centralisés (LIRE EN PREMIER)
│   ├── audio/                  # Chargement et segmentation audio
│   ├── features/               # Extraction (embeddings + fingerprints)
│   ├── index/                  # Construction des index FAISS
│   ├── retrieval/              # Pipeline d'identification (Stage 1 + 2)
│   └── api/app.py              # Interface CLI (Click)
├── webapp/
│   ├── backend/server.py       # API FastAPI (3 routes)
│   └── frontend/               # React 18 + Vite (SPA)
├── scripts/                    # Scripts utilitaires (ingestion, évaluation, etc.)
├── data/                       # Données persistantes (git-ignorées)
│   ├── chroma/                 # Collections ChromaDB (embeddings)
│   ├── features/fingerprints.db # SQLite (fingerprints par track)
│   ├── index/                  # Index FAISS (.faiss + segments .parquet)
│   └── processed/metadata.parquet # Métadonnées enrichies des tracks
└── requirements.txt            # Dépendances Python
```

---

## Pipeline d'identification (cœur du système)

### Stage 1 — Recherche vectorielle (FAISS)
1. Chargement audio → resampling au sample rate de la méthode
2. Segmentation en fenêtres de 5 s (hop 3 s, 80% overlap)
3. Embedding de chaque segment (MFCC / CLAP / MuQ)
4. Recherche FAISS → top 200 segments candidats par segment requête
5. Agrégation par track → top 20 tracks (somme des scores)

### Stage 2 — Re-ranking par fingerprinting
1. Extraction du fingerprint de la requête (constellation Shazam, 22050 Hz fixe)
2. Chargement des fingerprints des 20 candidats depuis SQLite
3. Alignement temporel (histogramme d'offsets) → score de cohérence temporelle
4. Score final : `score_faiss × (1 + score_fp × FP_SCORE_WEIGHT)`
5. Re-classement → top 10 résultats retournés

**Court-circuit** : si `score[0]/score[1] ≥ OPT_SHORTCIRCUIT_RATIO (10.0)`, Stage 2 ignoré.

---

## Méthodes d'embedding

| Méthode | Dim | SR | Précision | GPU | Temps CPU |
|---------|-----|----|-----------|-----|-----------|
| `mfcc` | 40 | 22050 Hz | Bonne | Non | ~8 s |
| `clap` | 512 | 48000 Hz | Bonne | CUDA/MPS | ~20 s |
| `muq` | 1024 | 24000 Hz | Excellente | CUDA uniquement | ~2.5 min |

La méthode active est définie par `EMBEDDING_METHOD` dans `src/config.py`.

---

## Configuration centrale : `src/config.py`

**Tous les paramètres sont ici.** Ne pas dupliquer de constantes ailleurs.

Paramètres clés à connaître :
- `EMBEDDING_METHOD` — méthode active (`"mfcc"` / `"clap"` / `"muq"`)
- `FP_SCORE_WEIGHT` — poids du fingerprint dans le score final (défaut : 10.0)
- `VECTOR_TOP_N_TRACKS` — nb de candidats envoyés en Stage 2 (défaut : 20)
- `UI_LISTEN_DURATION` — durée d'enregistrement micro en secondes (défaut : 15)
- `OPT_SHORTCIRCUIT` — court-circuit Stage 2 si match évident (défaut : True)
- `DOWNLOAD_WORKERS` — parallélisme téléchargements YouTube (défaut : 3)

---

## Stockage des données

### ChromaDB (`data/chroma/`)
- Une collection par méthode : `"mfcc"`, `"clap"`, `"muq"`
- Chaque document = un segment audio : ID `{track_id}_{segment_index}`
- Métadonnées : `track_id`, `start_s`

### SQLite (`data/features/fingerprints.db`)
- Table `fingerprints` : `track_id` (PK), `hashes` (BLOB pickled), `n_hashes`
- Une ligne par track, INSERT OR REPLACE (idempotent)

### FAISS (`data/index/`)
- `index_{method}_{type}.faiss` — index de recherche
- `segments_{method}.parquet` — mapping index ↔ (track_id, start_s)

### Metadata (`data/processed/metadata.parquet`)
- Colonnes : `track_id`, `title`, `artist`, `duration_s`, `album`, `genre`, `release_date`, `cover_url`, `embedded_methods`
- `embedded_methods` : liste des méthodes déjà calculées pour ce track (permet de reprendre sans recalculer)
- Écriture atomique : fichier temp + rename pour la sûreté en cas de crash

---

## Commandes fréquentes

```bash
# Activer l'environnement virtuel
source venv/bin/activate

# Ingestion de données (CSV Kaggle → tout le pipeline)
python scripts/download_music.py --csv data/kaggle/data/spotify-streaming-top-50-world.csv

# Enrichir les métadonnées (Deezer / MusicBrainz)
python scripts/enrich_metadata.py

# Reconstruire l'index FAISS (après ajout/suppression de tracks)
python src/index/build_index.py

# Lancer l'interface web (dev)
python scripts/start_webapp.py

# Lancer l'interface web (production)
python scripts/start_webapp.py --prod

# Identifier une piste en CLI
python src/api/app.py data/raw/song.mp3 --method clap --top 5

# Vérifier l'intégrité des données
python scripts/check_data.py --details
python scripts/check_data.py --purge --yes   # Nettoyer les tracks corrompus

# Reconstruire les fingerprints
python scripts/rebuild_fingerprints.py
```

---

## Interface web

### Backend (`webapp/backend/server.py`)
FastAPI, 3 routes :
- `GET /api/health` — liveness check
- `GET /api/config` — paramètres UI (listen_duration, embedding_method, etc.)
- `POST /api/identify` — fichier audio → résultats d'identification

Le backend charge les métadonnées en cache lazy et génère les liens de streaming (YouTube, Spotify, Deezer, Apple Music) par recherche sans API key.

### Frontend (`webapp/frontend/`)
React 18 + Vite. La logique d'état est centralisée dans `App.jsx`.
Composants clés :
- `ListenButton.jsx` — MediaRecorder avec countdown animé
- `DropZone.jsx` — drag-drop de fichier
- `ResultView.jsx` — affichage résultat (2 colonnes : cover + recommandations)
- `i18n.js` — traductions FR/EN (toutes les chaînes UI sont ici)

Proxy Vite vers `localhost:8000` configuré dans `vite.config.js`.

---

## Scripts utilitaires

| Script | Rôle |
|--------|------|
| `scripts/download_music.py` | Pipeline principal d'ingestion (CSV → DB) |
| `scripts/enrich_metadata.py` | Enrichissement via Deezer/MusicBrainz |
| `scripts/check_data.py` | Vérification intégrité + nettoyage |
| `scripts/rebuild_fingerprints.py` | Recalcul des fingerprints uniquement |
| `scripts/evaluate.py` | Métriques Top-1 / Top-5 sur jeu de test |
| `scripts/download_test_audio.py` | Télécharge des clips de test via YouTube |
| `scripts/start_webapp.py` | Lance backend + frontend (dev ou prod) |

---

## Points d'attention

### Multi-méthode
Le système gère plusieurs méthodes en parallèle. Chaque track peut avoir des embeddings dans plusieurs collections ChromaDB. Le champ `embedded_methods` dans `metadata.parquet` suit ce qui a déjà été calculé. Pour changer de méthode, modifier `EMBEDDING_METHOD` dans `config.py` et relancer `download_music.py` (les tracks déjà traités par cette méthode sont sautés).

### Sécurité crash
L'ingestion (`download_music.py`) sauvegarde après chaque track. Un crash ne perd qu'au plus le track en cours. La relance reprend automatiquement là où ça s'est arrêté grâce au champ `embedded_methods`.

### Audio en RAM
L'audio téléchargé depuis YouTube n'est **jamais écrit sur disque**. Il passe directement en RAM via ffmpeg pipe. Seuls les embeddings, fingerprints et index sont persistés.

### MuQ sur CPU / MPS
MuQ ne supporte pas Float16 sur CPU ni sur MPS (opérations ComplexFloat non supportées). Il tourne exclusivement sur CUDA. Sur Mac sans GPU NVIDIA, utiliser `clap` ou `mfcc`.

### Fingerprinting — base 22050 Hz
Le fingerprinting utilise toujours 22050 Hz, indépendamment de la méthode d'embedding. C'est intentionnel pour assurer la cohérence entre tous les tracks de la base.

---

## Technologies principales

- **Python** : librosa, soundfile, torchaudio, transformers, torch, faiss-cpu, chromadb, pandas, sqlite3, scipy, yt-dlp, fastapi, uvicorn, click, rich
- **Frontend** : React 18, Vite, JavaScript ES6+
- **Modèles** : `laion/clap-htsat-unfused` (CLAP), `OpenMuQ/MuQ-large-msd-iter` (MuQ)
