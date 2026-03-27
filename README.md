# Shazam Maison — Reconnaissance Audio par Embeddings + Fingerprinting

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Méthodes](https://img.shields.io/badge/Embeddings-MFCC%20%7C%20CLAP%20%7C%20MuQ-green)
![Index](https://img.shields.io/badge/Index-ChromaDB%20%2B%20FAISS-orange)
![Stockage](https://img.shields.io/badge/Fingerprints-SQLite-lightgrey)

Système de reconnaissance musicale inspiré de Shazam. À partir d'un extrait audio, retrouve le morceau correspondant dans une base de données vectorielle.

---

## Cheat Sheet

```bash
# 1. Alimenter la base (télécharge + calcule embeddings + fingerprints + construit l'index)
python scripts/download_music.py --csv data/kaggle/data/spotify-streaming-top-50-world.csv

# 2. Vérifier l'intégrité des données
python scripts/check_data.py

# 3. Supprimer les tracks problématiques
python scripts/check_data.py --purge

# 4. Reconstruire l'index FAISS (si nécessaire)
python src/index/build_index.py

# 5. Identifier un morceau
python src/api/app.py mon_audio.mp3
python src/api/app.py mon_audio.mp3 --method mfcc --top 5
```

---

## Architecture de stockage

Le projet utilise trois stores complémentaires, chacun avec un rôle précis :

### ChromaDB — Embeddings (`data/chroma/`)

Base de données vectorielle persistante. Stocke les embeddings de chaque segment audio avec leurs métadonnées (`track_id`, `start_s`). Une **collection par méthode d'embedding** (`mfcc`, `clap`, `muq`).

- Chaque segment a un ID stable : `{track_id}_{i}`
- Permet de supprimer / réécrire proprement les segments d'un track sans décalage d'indices
- Source de vérité pour les embeddings

### FAISS — Index de recherche rapide (`data/index/`)

Index vectoriel en mémoire pour la recherche par similarité (Stage 1). Reconstruit depuis ChromaDB via `build_index.py`.

- `index_{method}_{type}.faiss` — l'index de recherche
- `segments_{method}.parquet` — table de correspondance indice FAISS → `track_id` (générée en même temps que l'index)

### SQLite — Fingerprints (`data/features/fingerprints.db`)

Base de données légère pour les empreintes audio (Stage 2 — re-ranking Shazam). Une ligne par track, mise à jour atomiquement via `INSERT OR REPLACE`.

- Remplace l'ancien `fingerprints.pkl` (non-atomique, tout-ou-rien)
- Permet de supprimer ou recalculer le fingerprint d'un seul track sans réécrire tout le fichier

---

## Pipeline

```
CSV Kaggle Spotify
      │
      ▼
download_music.py
      │  yt-dlp → audio en RAM (aucun MP3 stocké)
      │  embed_segment()      → ChromaDB  (data/chroma/)
      │  extract_fingerprint() → SQLite    (fingerprints.db)
      │                          metadata.parquet
      ▼
build_index.py
      │  ChromaDB → FAISS index + segments_{method}.parquet
      ▼
identify_track(audio)
      │
      ├── Stage 1 : segmenter → embedder → FAISS → Top 20 candidats
      │
      └── Stage 2 : fingerprint requête ↔ fingerprints SQLite → re-ranking
                                │
                                ▼
                        Top N résultats
```

---

## Installation

### Prérequis système

- Python 3.10
- ffmpeg : `brew install ffmpeg` (macOS) / `apt install ffmpeg` (Linux) / [ffmpeg.org](https://ffmpeg.org/download.html) (Windows)
- Clé API Kaggle : créer un compte sur [kaggle.com](https://kaggle.com), télécharger `kaggle.json` et le placer dans `~/.kaggle/kaggle.json`

### Environnement Python

```bash
# Cloner le projet
git clone https://github.com/thmsgo18/Shazam
cd Shazam

# Créer et activer le venv
python3 -m venv venv
source venv/bin/activate   # macOS / Linux
# .\venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Ajouter une librairie

```bash
pip install ma-librairie
pip freeze > requirements.txt
```

---

## Configuration — `src/config.py`

Tous les paramètres du projet sont centralisés ici. **Ne modifier que ce fichier** pour changer le comportement du système.

### Chemins

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `CHROMA_DIR` | `"data/chroma"` | Dossier ChromaDB (embeddings) |
| `FINGERPRINTS_DB` | `"data/features/fingerprints.db"` | Base SQLite des fingerprints |
| `INDEX_DIR` | `"data/index"` | Index FAISS + order parquet |
| `PROCESSED_DIR` | `"data/processed"` | metadata.parquet |

### Audio

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `SAMPLE_RATE` | `22050` | Fréquence d'échantillonnage de base (Hz) |
| `CLAP_SAMPLE_RATE` | `48000` | Fréquence requise par le modèle CLAP |
| `MUQ_SAMPLE_RATE` | `24000` | Fréquence requise par le modèle MuQ |

### Segmentation

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `SEGMENT_WIN_S` | `5.0` | Durée de chaque segment (secondes) |
| `SEGMENT_HOP_S` | `3.0` | Pas entre deux segments (secondes) |
| `SEGMENT_MIN_WIN` | `0.8` | Fraction minimale pour garder le dernier segment |

### Embedding

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `EMBEDDING_METHOD` | `"mfcc"` | Méthode active : `"mfcc"`, `"clap"` ou `"muq"` |
| `CLAP_MODEL_NAME` | `"laion/clap-htsat-unfused"` | Modèle CLAP (HuggingFace) |
| `MUQ_MODEL_NAME` | `"OpenMuQ/MuQ-large-msd-iter"` | Modèle MuQ (HuggingFace) |
| `MUQ_BATCH_SIZE` | `8` | Nombre de segments traités en batch par MuQ |

### Recherche vectorielle

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `VECTOR_TOP_K_SEGMENTS` | `200` | Nombre de segments récupérés par FAISS |
| `VECTOR_TOP_N_TRACKS` | `20` | Nombre de candidats pour le re-ranking |
| `VECTOR_TOP_N_RESULTS` | `5` | Nombre de résultats finaux retournés |
| `INDEX_TYPE` | `"flat"` | Type d'index : `"flat"`, `"hnsw"` ou `"ivf"` |

### Optimisations

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `OPT_FLOAT16` | `True` | Charger CLAP/MuQ en demi-précision (réduit la RAM, CUDA uniquement) |
| `OPT_BATCH_EMBED` | `True` | Embedder les segments par batch (plus rapide avec MuQ) |
| `OPT_SHORTCIRCUIT` | `True` | Sauter le Stage 2 si le 1er candidat FAISS est largement devant |
| `OPT_SHORTCIRCUIT_RATIO` | `10.0` | Ratio score[0]/score[1] au-delà duquel on court-circuite |
| `OPT_FINGERPRINT_PARALLEL` | `True` | Charger les fingerprints candidats en parallèle (Stage 2) |

### Affichage

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `PROGRESS_DATASET` | `True` | Barre de progression globale sur l'ensemble des tracks |
| `PROGRESS_TRACK` | `True` | Barre de progression par morceau (segments) |

---

## Alimenter la base — `scripts/download_music.py`

Télécharge l'audio en RAM, calcule embeddings + fingerprints, stocke dans ChromaDB + SQLite, construit l'index FAISS. **Aucun MP3 n'est stocké sur disque.**

Les morceaux déjà traités pour la méthode active sont automatiquement ignorés au démarrage (basé sur `embedded_methods` dans `metadata.parquet`).

```bash
# Un seul CSV
python scripts/download_music.py --csv data/kaggle/data/spotify-streaming-top-50-world.csv

# Plusieurs CSV spécifiques
python scripts/download_music.py \
  --csv data/kaggle/data/spotify-streaming-top-50-france.csv \
  --csv data/kaggle/data/spotify-streaming-top-50-usa.csv

# Tous les CSV d'un dossier
python scripts/download_music.py --csv data/kaggle/data/

# Sans --csv : utilise automatiquement tous les CSV Kaggle disponibles
python scripts/download_music.py
```

> **Changer de méthode :** modifier `EMBEDDING_METHOD` dans `config.py` et relancer.
> Les morceaux déjà traités en MFCC ne seront pas re-traités pour MFCC, mais seront traités pour CLAP — le script détecte la méthode indépendamment.

### Reprise automatique après crash

La sauvegarde est faite **après chaque track** dans les trois stores. En cas de crash ou d'interruption (Ctrl+C) :

- Les tracks déjà traités sont conservés
- Au relancement, ils sont automatiquement skippés
- Le track en cours de traitement au moment du crash sera simplement re-traité

### Ordre des étapes internes

```
Pour chaque morceau :
  1. Recherche YouTube via yt-dlp
  2. Téléchargement audio en RAM (dossier temporaire auto-supprimé)
  3. Calcul du fingerprint → SQLite (fingerprints.db)
  4. Segmentation + calcul des embeddings (méthode config.EMBEDDING_METHOD)
  5. Sauvegarde dans ChromaDB (embeddings + métadonnées segment)
  6. Mise à jour de metadata.parquet (écriture atomique)
→ Construction de l'index FAISS depuis ChromaDB
```

---

## Vérifier et nettoyer les données — `scripts/check_data.py`

Vérifie la cohérence des données générées par `download_music.py` et supprime les tracks problématiques.

```bash
# Vérifier toutes les méthodes disponibles
python scripts/check_data.py

# Vérifier une méthode spécifique
python scripts/check_data.py --method mfcc

# Supprimer les tracks problématiques (avec confirmation)
python scripts/check_data.py --purge

# Supprimer sans demander confirmation
python scripts/check_data.py --purge --yes

# Supprimer uniquement les tracks sans fingerprint (pour les recalculer)
python scripts/check_data.py --purge-missing-fp

# Combiner méthode + purge
python scripts/check_data.py --method mfcc --purge
```

### Checks effectués

| Code | Type | Description |
|------|------|-------------|
| C1 | Critique | Dimension des embeddings inattendue |
| C2 | Critique | NaN ou Inf dans les embeddings (résultats FAISS corrompus) |
| C3 | Critique | Désynchronisation ChromaDB ↔ order parquet |
| C5 | Critique | FAISS index désynchronisé avec ChromaDB (relancer `build_index.py`) |
| C6 | Critique | Segments orphelins (dans ChromaDB mais absents de metadata) |
| C6b | Critique | Track marqué comme traité dans metadata mais sans segments dans ChromaDB |
| C7 | Critique | Embedding incomplet (< 80% des segments attendus) |
| Q1 | Qualité | Durée aberrante (≤ 0s ou > 10min) |
| Q2 | Qualité | Segment dont le `start_s` dépasse la durée du track |
| Q3 | Qualité | Fingerprint vide (0 hash) |
| Q4 | Qualité | Fingerprint anormalement pauvre (outlier IQR par tranche de durée) |
| FP | Qualité | Tracks sans fingerprint (Stage 2 inopérant pour ces tracks) |

### Que fait `--purge` ?

Pour chaque track flaggé par un warning :

1. Ses segments sont supprimés de **ChromaDB**
2. Son fingerprint est supprimé de **SQLite** (`fingerprints.db`)
3. Sa ligne est supprimée de **`metadata.parquet`** → le track sera re-téléchargé au prochain `download_music.py`
4. L'**index FAISS** + l'order parquet sont supprimés (devenus obsolètes → relancer `build_index.py`)

### Que fait `--purge-missing-fp` ?

Purge uniquement les tracks qui ont des embeddings dans ChromaDB mais pas de fingerprint dans SQLite. Utile pour recalculer les fingerprints manquants sans tout re-télécharger.

---

## Reconstruire l'index — `src/index/build_index.py`

Reconstruit l'index FAISS depuis ChromaDB. Appelé automatiquement à la fin de `download_music.py`, mais à relancer manuellement après un `--purge`.

```bash
# Utilise la méthode et le type d'index définis dans config.py
python src/index/build_index.py
```

Ce script :
1. Charge tous les embeddings depuis la collection ChromaDB de la méthode active
2. Sauvegarde l'ordre des segments dans `data/index/segments_{method}.parquet`
3. Construit l'index FAISS et le sauvegarde dans `data/index/index_{method}_{type}.faiss`

---

## Identifier un morceau — `src/api/app.py`

```bash
# Avec la méthode par défaut (config.EMBEDDING_METHOD)
python src/api/app.py mon_audio.mp3

# Avec une méthode spécifique
python src/api/app.py mon_audio.mp3 --method clap

# Changer le nombre de résultats
python src/api/app.py mon_audio.mp3 --top 10
```

Ou depuis Python :

```python
from src.retrieval.query_pipeline import identify_track

results = identify_track("mon_audio.mp3")                    # méthode par défaut
results = identify_track("mon_audio.mp3", method="clap")     # méthode spécifique
results = identify_track("mon_audio.mp3", top_n=10)

# results = [(track_id, score), ...]
for rank, (track_id, score) in enumerate(results, 1):
    print(f"{rank}. {track_id} — {score:.4f}")
```

---

## Utilisation du GPU

| Méthode | GPU utilisé | Détail |
|---------|-------------|--------|
| MFCC | Non | 100% CPU (numpy/librosa) |
| CLAP | Oui si disponible | CUDA → MPS (Apple Silicon) → CPU |
| MuQ | Oui si disponible | CUDA → CPU (MPS exclu : opérations ComplexFloat non supportées) |

> Le float16 (`OPT_FLOAT16`) n'est activé que sur CUDA. Sur CPU et MPS, float32 est utilisé.

---

## Compatibilité

Le projet fonctionne sur **macOS, Linux et Windows**.

- Le fallback CPU MPS (Apple Silicon) est activé automatiquement
- Les variables `OMP_NUM_THREADS=1` et `OPENBLAS_NUM_THREADS=1` sont positionnées au démarrage pour éviter les deadlocks librosa/numpy sur macOS
- La gestion des processus yt-dlp utilise les API natives de chaque OS (`SIGTERM` + groupes de processus sur Unix, `CREATE_NEW_PROCESS_GROUP` sur Windows)

---

## Tester les trois méthodes

```bash
# MFCC (rapide, CPU uniquement)
python src/api/app.py mon_audio.mp3 --method mfcc

# CLAP (GPU si dispo, modèle ~1.5 Go)
python src/api/app.py mon_audio.mp3 --method clap

# MuQ (GPU si dispo, modèle ~1 Go, meilleure précision)
python src/api/app.py mon_audio.mp3 --method muq
```

### Résultats attendus

| Méthode | Temps (CPU) | Précision |
|---------|-------------|-----------|
| MFCC | ~8s | Bon |
| CLAP | ~20s | Bon |
| MuQ | ~2min30 | Excellent |

Le bon morceau doit toujours apparaître **en 1ère position** avec un score très largement supérieur au 2ème.

---

## Structure du projet

```
Shazam/
├── README.md
├── requirements.txt
│
├── data/
│   ├── kaggle/data/          # CSV Kaggle Spotify (téléchargés automatiquement)
│   ├── chroma/               # ChromaDB — embeddings (une collection par méthode)
│   ├── processed/            # metadata.parquet
│   ├── features/             # fingerprints.db (SQLite)
│   └── index/                # index_{method}_{type}.faiss
│                             # segments_{method}.parquet (ordre FAISS ↔ track_id)
│
├── src/
│   ├── config.py             # ← Tous les paramètres ici
│   ├── audio/
│   │   ├── loading.py        # Chargement audio (librosa)
│   │   └── preprocessing.py  # Découpage en segments
│   ├── features/
│   │   ├── embeddings_audio.py  # MFCC, CLAP, MuQ
│   │   └── fingerprint.py       # Constellation map (Shazam)
│   ├── index/
│   │   └── build_index.py    # Construction index FAISS depuis ChromaDB
│   ├── retrieval/
│   │   ├── searcher.py       # Recherche dans FAISS
│   │   └── query_pipeline.py # Pipeline complet (Stage 1 + Stage 2)
│   └── api/
│       └── app.py            # CLI Click
│
└── scripts/
    ├── download_music.py     # Téléchargement + build pipeline
    ├── check_data.py         # Vérification + purge des données
    └── evaluate.py           # Évaluation Top-1 / Top-5
```

---

## Dépannage

### `Collection 'mfcc' introuvable dans ChromaDB`
```
Lance d'abord : python scripts/download_music.py
```

### `FAISS index manquant`
L'index n'a pas encore été construit ou a été supprimé après un `--purge`.
```bash
python src/index/build_index.py
```

### `NotImplementedError: MPS device` (Mac Apple Silicon)
Normalement géré automatiquement. Si le problème persiste :
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python ...
```

### Le processus freeze et ne répond plus (même à Ctrl+C)
Deadlock librosa/numpy. Tuer le processus depuis un autre terminal :
```bash
ps aux | grep download_music   # trouver le PID
kill -9 <PID>
```
Les données déjà sauvegardées sont conservées. Relancer normalement — les tracks déjà traités seront skippés.

### Manque de RAM
MuQ nécessite ~2 Go libres, CLAP ~1.5 Go. Fermer les autres applications et relancer.

---

## Git — Récupérer les modifications de l'équipe

```bash
git stash          # Mettre de côté tes modifications locales
git pull           # Récupérer les dernières modifications
git stash pop      # Réappliquer tes modifications
```
