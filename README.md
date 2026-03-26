# Shazam Maison — Reconnaissance Audio par Embeddings + Fingerprinting

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Méthodes](https://img.shields.io/badge/Embeddings-MFCC%20%7C%20CLAP%20%7C%20MuQ-green)
![Index](https://img.shields.io/badge/Index-FAISS-orange)

Système de reconnaissance musicale inspiré de Shazam. À partir d'un extrait audio, retrouve le morceau correspondant dans une base de données vectorielle.

---

## Cheat Sheet

```bash
# 1. Alimenter la base (télécharge + calcule embeddings + fingerprints)
python scripts/download_music.py

# 2. Vérifier l'intégrité des données
python scripts/check_data.py

# 3. Supprimer les tracks problématiques et les re-télécharger
python scripts/check_data.py --purge

# 4. Reconstruire l'index FAISS
python src/index/build_index.py

# 5. Identifier un morceau
python -c "
from src.retrieval.query_pipeline import identify_track
results = identify_track('mon_audio.mp3', method='mfcc')
for rank, (track_id, score) in enumerate(results, 1):
    print(f'{rank}. {track_id} — {score:.4f}')
"
```

---

## Pipeline

```
CSV Kaggle Spotify
      │
      ▼
download_music.py
      │  yt-dlp → audio en RAM
      │  embed_segment() → embeddings.npy
      │  extract_fingerprint() → fingerprints.pkl
      │  segments.parquet + metadata.parquet
      ▼
build_index.py → index_{method}_{type}.faiss
      │
      ▼
identify_track(audio)
      │
      ├── Stage 1 : segmenter → embedder → FAISS → Top 20 candidats
      │
      └── Stage 2 : fingerprint requête ↔ fingerprints candidats → re-ranking
                                │
                                ▼
                        Top N résultats
```

---

## Installation

### Prérequis système

- Python 3.10
- ffmpeg : `brew install ffmpeg` (macOS) / `apt install ffmpeg` (Linux)
- Clé API Kaggle : créer un compte sur [kaggle.com](https://kaggle.com), télécharger `kaggle.json` et le placer dans `~/.kaggle/kaggle.json`

### Environnement Python

```bash
# Cloner le projet
git clone https://github.com/thmsgo18/Shazam
cd Shazam

# Créer et activer le venv
python3 -m venv venv
source venv/bin/activate  # macOS / Linux
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
| `MUQ_BATCH_SIZE` | `8` | Nombre de segments traités en parallèle par MuQ |

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
| `OPT_FLOAT16` | `True` | Charger CLAP/MuQ en demi-précision (réduit la RAM) |
| `OPT_BATCH_EMBED` | `True` | Embedder les segments par batch (plus rapide) |
| `OPT_SHORTCIRCUIT` | `True` | Sauter le Stage 2 si le 1er candidat est évident |
| `OPT_PARALLEL_FP` | `True` | Calculer les fingerprints en parallèle (Stage 2) |

### Affichage

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `PROGRESS_DATASET` | `True` | Barre de progression globale sur l'ensemble des tracks |
| `PROGRESS_TRACK` | `True` | Barre de progression par morceau (segments) |

---

## Alimenter la base — `scripts/download_music.py`

Télécharge l'audio en RAM, calcule embeddings + fingerprints, construit l'index FAISS. **Aucun MP3 n'est stocké sur disque.**

Les morceaux déjà traités pour la méthode active sont automatiquement ignorés.

```bash
# Sans --csv : utilise automatiquement tous les CSV Kaggle disponibles
python scripts/download_music.py

# Un seul CSV
python scripts/download_music.py --csv data/kaggle/data/spotify-streaming-top-50-world.csv

# Plusieurs CSV spécifiques
python scripts/download_music.py \
  --csv data/kaggle/data/spotify-streaming-top-50-france.csv \
  --csv data/kaggle/data/spotify-streaming-top-50-usa.csv

# Tous les CSV d'un dossier
python scripts/download_music.py --csv data/kaggle/data/
```

> **Changer de méthode :** modifier `EMBEDDING_METHOD` dans `config.py` et relancer.
> Les morceaux déjà traités en MFCC ne seront pas re-traités pour MFCC, mais seront traités pour CLAP — le script détecte la méthode indépendamment.

### Ordre des étapes internes

```
Pour chaque morceau :
  1. Recherche YouTube via yt-dlp
  2. Téléchargement audio en RAM (dossier temporaire auto-supprimé)
  3. Calcul des embeddings (méthode config.EMBEDDING_METHOD)
  4. Calcul du fingerprint (Shazam-like)
  5. Fusion avec la base existante
  6. Suppression de l'audio
→ Construction de l'index FAISS
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

# Combiner méthode + purge
python scripts/check_data.py --method mfcc --purge
```

### Checks effectués

| Code | Type | Description |
|------|------|-------------|
| C1 | Critique | Dimension des embeddings inattendue |
| C2 | Critique | NaN ou Inf dans les embeddings (résultats FAISS corrompus) |
| C3 | Critique | Désynchronisation embeddings.npy ↔ segments.parquet |
| C4 | Critique | segment_ids dupliqués |
| C5 | Critique | FAISS index désynchronisé (relancer build_index.py) |
| C6 | Critique | Segments sans entrée dans metadata (orphelins) |
| C7 | Critique | Embedding incomplet (< 80% des segments attendus) |
| Q1 | Qualité | Durée aberrante (≤ 0s ou > 10min) |
| Q2 | Qualité | Segment dont le start_s dépasse la durée du track |
| Q3 | Qualité | Fingerprint vide (0 hash) |
| Q4 | Qualité | Fingerprint anormalement pauvre (outlier IQR) |
| FP | Qualité | Tracks sans fingerprint (Stage 2 inopérant) |

### Que fait `--purge` ?

Pour chaque track flaggé par un warning :
1. Ses segments sont retirés de `segments_{method}.parquet`
2. Ses embeddings sont retirés de `embeddings_{method}.npy` (renumérotation automatique)
3. La méthode est retirée de `embedded_methods` dans `metadata.parquet` → le track sera **re-téléchargé** au prochain `download_music.py`
4. Son fingerprint est supprimé de `fingerprints.pkl`

---

## Identifier un morceau — `src/retrieval/query_pipeline.py`

```python
from src.retrieval.query_pipeline import identify_track

# Avec la méthode par défaut (config.EMBEDDING_METHOD)
results = identify_track("mon_audio.mp3")

# Avec une méthode spécifique
results = identify_track("mon_audio.mp3", method="clap")

# Changer le nombre de résultats
results = identify_track("mon_audio.mp3", top_n=10)

# results = [(track_id, score), ...]
for rank, (track_id, score) in enumerate(results, 1):
    print(f"{rank}. {track_id} — {score:.4f}")
```

---

## Reconstruire l'index — `src/index/build_index.py`

À relancer si les embeddings ont changé sans passer par `download_music.py`.

```bash
# Utilise la méthode et le type d'index définis dans config.py
python src/index/build_index.py
```

---

## Tester les trois méthodes

> **Mac Apple Silicon :** le fallback CPU est activé automatiquement. Aucune variable d'environnement nécessaire.

```bash
# MFCC (~8 secondes)
python -c "
from src.retrieval.query_pipeline import identify_track
results = identify_track('data/raw/mon_audio.mp3', method='mfcc')
for rank, (track_id, score) in enumerate(results, 1):
    print(f'{rank}. {track_id} — {score:.4f}')
"

# CLAP (~20 secondes)
python -c "
from src.retrieval.query_pipeline import identify_track
results = identify_track('data/raw/mon_audio.mp3', method='clap')
for rank, (track_id, score) in enumerate(results, 1):
    print(f'{rank}. {track_id} — {score:.4f}')
"

# MuQ (~2-3 minutes sur CPU)
python -c "
from src.retrieval.query_pipeline import identify_track
results = identify_track('data/raw/mon_audio.mp3', method='muq')
for rank, (track_id, score) in enumerate(results, 1):
    print(f'{rank}. {track_id} — {score:.4f}')
"
```

### Résultats attendus

| Méthode | Temps (CPU) | Score 1er / 2ème | Précision |
|---------|-------------|------------------|-----------|
| MFCC | ~8s | x36 | ✅ Bon |
| CLAP | ~20s | x27 | ✅ Bon |
| MuQ | ~2min30 | x70 | ✅ Excellent |

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
│   ├── processed/            # metadata.parquet
│   ├── features/             # embeddings_{method}.npy
│   │                         # segments_{method}.parquet
│   │                         # fingerprints.pkl
│   └── index/                # index_{method}_{type}.faiss
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
│   │   └── build_index.py    # Construction index FAISS
│   ├── retrieval/
│   │   ├── searcher.py       # Recherche dans FAISS
│   │   └── query_pipeline.py # Pipeline complet (Stage 1 + Stage 2)
│   └── api/
│       └── app.py            # CLI Click
│
└── scripts/
    ├── download_music.py        # Téléchargement + build pipeline
    ├── build_segment_embeddings.py  # Calcul embeddings sur data/raw/
    └── evaluate.py              # Évaluation Top-1 / Top-5
```

---

## Dépannage

### `FileNotFoundError: Pas d'index 'clap'`
```
Change EMBEDDING_METHOD dans config.py ou relance download_music.py.
Méthodes disponibles : mfcc
```
→ Relancer `download_music.py` avec `EMBEDDING_METHOD = "clap"` dans `config.py`.

### `NotImplementedError: MPS device` (Mac Apple Silicon)
→ Normalement géré automatiquement. Si le problème persiste :
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python ...
```

### `AssertionError: Mismatch embeddings vs segments`
→ Les fichiers `.npy` et `.parquet` sont désynchronisés. Relancer `download_music.py` pour reconstruire.

### Le processus est bloqué (0% CPU)
→ Manque de RAM. Fermer les autres applications et relancer. MuQ nécessite ~2Go libres, CLAP ~1.5Go.

---

## Git — Récupérer les modifications de l'équipe

```bash
git stash          # Mettre de côté tes modifications locales
git pull           # Récupérer les dernières modifications
git stash pop      # Réappliquer tes modifications
```
