# Shazam

<div align="right">
  <a href="./README.md"><img src="https://img.shields.io/badge/EN-version-1a73e8?style=flat-square&labelColor=1a1a2e" alt="Read in English"></a>
  &nbsp;
  <img src="https://img.shields.io/badge/FR-current-0055A4?style=flat-square&labelColor=EF4135" alt="Français (actuel)">
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
![Licence](https://img.shields.io/badge/Licence-Master_IAD_S2-lightgrey)

Système de reconnaissance musicale inspiré de Shazam, développé dans le cadre d'un projet Big Data (Master IAD, S2). À partir d'un extrait audio capturé au micro ou déposé en fichier, le système retrouve le morceau correspondant dans une base de plusieurs centaines de titres et affiche les liens de streaming.

L'approche combine la puissance des embeddings de deep learning (recherche vectorielle FAISS) avec le fingerprinting spectral inspiré du brevet Shazam original, formant un pipeline hybride en deux étapes qui reste robuste face au bruit, à la réverbération et aux extraits courts.

---

## Table des matières

- [Shazam](#shazam)
  - [Table des matières](#table-des-matières)
  - [Comment ça marche](#comment-ça-marche)
    - [Fonctionnement du fingerprinting](#fonctionnement-du-fingerprinting)
  - [Architecture du projet](#architecture-du-projet)
  - [Méthodes d'embedding](#méthodes-dembedding)
  - [Prérequis](#prérequis)
  - [Installation](#installation)
  - [Données](#données)
  - [Démarrage rapide](#démarrage-rapide)
  - [Commandes essentielles](#commandes-essentielles)
  - [Configuration](#configuration)
  - [Interface web](#interface-web)
    - [Routes API (FastAPI)](#routes-api-fastapi)
    - [Fonctionnalités](#fonctionnalités)
    - [Modes de lancement](#modes-de-lancement)
  - [Augmentation RIR](#augmentation-rir)
    - [Génération des RIRs synthétiques](#génération-des-rirs-synthétiques)
    - [Sources disponibles](#sources-disponibles)
  - [Évaluation](#évaluation)
    - [Métriques calculées](#métriques-calculées)
    - [Conditions de dégradation](#conditions-de-dégradation)
    - [Workflow d'évaluation](#workflow-dévaluation)
    - [Graphiques produits](#graphiques-produits)
  - [Stockage des données](#stockage-des-données)
    - [ChromaDB — `data/chroma/`](#chromadb--datachroma)
    - [SQLite — `data/features/fingerprints.db`](#sqlite--datafeaturesfingerprintsdb)
    - [FAISS — `data/index/`](#faiss--dataindex)
    - [Parquet — `data/processed/metadata.parquet`](#parquet--dataprocessedmetadataparquet)
  - [Points d'attention](#points-dattention)
  - [Technologies](#technologies)
  - [Améliorations possibles](#améliorations-possibles)
  - [Références](#références)
  - [Équipe](#équipe)

---

## Comment ça marche

Le pipeline d'identification se déroule en deux étapes successives.

```
Audio requête
      │
      ▼
┌─────────────────────────────────────────┐
│  Stage 1 — Recherche vectorielle        │
│                                         │
│  Découpage en fenêtres de 5s            │
│  → Embedding (MFCC / CLAP / MuQ / MERT) │
│  → Recherche FAISS (cosinus)            │
│  → Top 20 tracks candidats              │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Stage 2 — Fingerprinting Shazam        │
│                                         │
│  Constellation spectrale de la requête  │
│  ↔ Fingerprints SQLite des 20 candidats │
│  → Alignement temporel (histogramme)    │
│  → Score de cohérence temporelle        │
└────────────────┬────────────────────────┘
                 │
                 ▼
      Classement final
      (FP en priorité, FAISS en départage)
```

**Stage 1** transforme chaque fenêtre de 5 secondes en vecteur d'embedding via un modèle pré-entraîné, puis interroge l'index FAISS pour trouver les segments les plus proches dans l'espace vectoriel. Les résultats sont agrégés par track pour obtenir les `VECTOR_TOP_N_TRACKS` (défaut : 20) meilleurs candidats.

**Stage 2** applique le fingerprinting de Shazam : extraction d'une constellation de pics dans le spectrogramme, comparaison avec les fingerprints stockés en SQLite, et alignement temporel par histogramme d'offsets. Ce score est plus discriminant que la similarité cosinus et résiste mieux aux dégradations audio.

Le **score final** trie d'abord par score fingerprint (source de vérité), puis par score FAISS en cas d'égalité. Si le fingerprinting échoue sur toute la liste (audio très dégradé), le classement retombe sur les scores FAISS seuls.

### Fonctionnement du fingerprinting

Le fingerprinting est une implémentation du brevet Shazam original (Wang, 2003). Il opère en quatre étapes :

```
Signal audio (22 050 Hz)
        │
        ▼
┌───────────────────────────────┐
│  Spectrogramme (STFT)         │
│                               │
│  Fréquence                    │
│    ^   *       *              │
│    │      *  *    *    *      │  ← pics spectraux
│    │   *           *          │    (constellation map)
│    └──────────────────> Temps │
└──────────────┬────────────────┘
               │  détection de pics locaux
               ▼
┌───────────────────────────────┐
│  Génération des hashes        │
│                               │
│  Ancre (f1, t1)               │
│      └─── cible (f2, t2)      │
│                               │
│  hash = (f1, f2, Δt, t1)      │  ← 4-tuple par paire
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│  Alignement temporel          │
│                               │
│  Pour chaque hash commun      │
│  entre requête et candidat :  │
│  offset = t1_base − t1_query  │
│                               │
│  Histogramme des offsets →    │
│  pic = cohérence temporelle   │
└───────────────────────────────┘
```

Chaque hash encode une paire de pics `(fréquence_ancre, fréquence_cible, delta_temps)` avec la position temporelle de l'ancre `t1`. Un morceau de 3 minutes génère typiquement plusieurs milliers de hashes, stockés en SQLite sous forme de BLOB pickled.

L'alignement temporel est la clé de la robustesse : deux morceaux différents peuvent partager quelques hashes par hasard, mais seul le bon morceau présentera un pic franc dans l'histogramme des offsets — toutes ses correspondances s'alignent au même décalage temporel.

---

## Architecture du projet

```
Projet/
├── manage.py                        # Point d'entrée unique — toutes les commandes
├── src/
│   ├── config.py                    # Paramètres centralisés (LIRE EN PREMIER)
│   ├── ingestion/
│   │   ├── ingest.py                # Pipeline téléchargement → embeddings → index
│   │   ├── augment_rir.py           # Augmentation par Room Impulse Response
│   │   └── fingerprints.py          # Recalcul des fingerprints
│   ├── features/
│   │   └── embeddings_audio.py      # Extraction MFCC / CLAP / MuQ / MERT
│   ├── index/
│   │   └── build_index.py           # Construction des index FAISS
│   ├── retrieval/
│   │   ├── query_pipeline.py        # Orchestration Stage 1 + Stage 2
│   │   └── searcher.py              # Recherche FAISS + agrégation
│   ├── evaluation/
│   │   ├── evaluate.py              # Métriques Top-1 / Top-5 / MRR / latence
│   │   ├── rir_impact.py            # Analyse impact RIR sur un fichier
│   │   └── plots.py                 # Génération des graphiques PNG
│   └── maintenance/
│       ├── check.py                 # Vérification intégrité des données
│       ├── enrich.py                # Enrichissement métadonnées (Deezer / MusicBrainz)
│       └── clean.py                 # Suppression propre d'un track
├── webapp/
│   ├── backend/server.py            # API FastAPI (3 routes)
│   └── frontend/                    # React 18 + Vite (SPA)
│       └── src/
│           ├── App.jsx              # État global, routing entre vues
│           └── components/          # ListenButton, DropZone, ResultView…
├── data/                            # Données persistantes (git-ignorées)
│   ├── chroma/                      # Vecteurs d'embeddings (ChromaDB)
│   ├── features/fingerprints.db     # Fingerprints Shazam (SQLite, mode WAL)
│   ├── index/                       # Index FAISS + segments Parquet
│   ├── processed/metadata.parquet   # Métadonnées enrichies des tracks
│   ├── raw/                         # Fichiers audio de test
│   └── rir/                         # WAV de Room Impulse Response (optionnel)
├── results/
│   ├── EXPERIMENTS.md               # Journal des expériences (versionné)
│   ├── eval/                        # JSON d'évaluation (git-ignorés)
│   ├── benchmark/                   # JSON de benchmark (git-ignorés)
│   └── plots/                       # Graphiques PNG pour le rapport (git-ignorés)
└── research_paper/                  # Articles de référence (PDF)
```

---

## Méthodes d'embedding

Quatre méthodes sont disponibles et peuvent coexister dans la même base. La méthode active est définie par `EMBEDDING_METHOD` dans `src/config.py`.

| Méthode | Modèle | Dim. | Sample Rate | Compatibilité | Remarques |
|---------|--------|------|-------------|---------------|-----------|
| `mfcc` | — (librosa) | 40 | 22 050 Hz | CPU | Rapide, aucune dépendance modèle |
| `clap` | `laion/clap-htsat-unfused` | 512 | 48 000 Hz | CUDA, MPS, CPU | Modèle généraliste, bon compromis qualité / vitesse |
| `clap` | `laion/larger_clap_music` | 512 | 48 000 Hz | CUDA, MPS, CPU | Modèle spécialisé musique, meilleure précision |
| `muq` | `OpenMuQ/MuQ-large-msd-iter` | 1 024 | 24 000 Hz | CUDA uniquement | Meilleure précision, GPU NVIDIA requis |
| `mert` | `m-a-p/MERT-v1-95M` | 768 | 24 000 Hz | CUDA, MPS | Modèle de représentation musicale |

Chaque track mémorise les méthodes déjà calculées dans le champ `embedded_methods` de `metadata.parquet`. Relancer `ingest` après un changement de méthode ne recalcule que ce qui manque — les tracks existants ne sont pas re-téléchargés.

---

## Prérequis

- **Python 3.10** (les dépendances ne sont pas toutes compatibles 3.11+)
- **Node.js 18+** (pour le frontend React)
- **ffmpeg** installé et accessible dans le PATH
- **yt-dlp** (installé via `requirements.txt`)
- GPU recommandé pour CLAP / MuQ / MERT (CPU fonctionne pour MFCC et CLAP)

---

## Installation

```bash
# 1. Cloner le dépôt
git clone <url>
cd Projet

# 2. Créer et activer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# 3. Installer les dépendances Python
pip install -r requirements.txt

# 4. Installer les dépendances frontend
cd webapp/frontend && npm install && cd ../..
```

---

## Données

Les données musicales proviennent des classements Spotify via Kaggle. Le pipeline télécharge automatiquement l'audio depuis YouTube en RAM (aucun MP3 stocké sur disque) à partir des titres et artistes présents dans le CSV.

**Structure attendue du CSV :**

| Colonne | Description |
|---------|-------------|
| `track_name` | Titre du morceau |
| `artist_names` | Nom de l'artiste |

Les CSV Kaggle `spotify-streaming-top-50-*.csv` sont directement compatibles. Placer les fichiers dans `data/kaggle/data/`.

**Dataset utilisé :** [Spotify Streaming Top 50 — Kaggle](https://www.kaggle.com/datasets/anxods/spotify-top-50-playlist-songs-anxods) — classements quotidiens Top 50 mondial et par pays.

**Déduplication automatique :** un même morceau présent dans plusieurs CSV (hits mondiaux présents dans le top France, top US et top Monde) n'est traité qu'une fois, identifié par son `track_id` (hash MD5 de `artiste_titre`).

---

## Démarrage rapide

```bash
source venv/bin/activate

# 1. Alimenter la base de données
python manage.py ingest --csv data/kaggle/data/spotify-streaming-top-50-world.csv

# 2. Enrichir la base avec des variantes acoustiques
python manage.py augment

# 3. Enrichir les métadonnées (pochettes, genres, dates de sortie)
python manage.py enrich

# 4. Lancer l'interface web
python manage.py start-webapp
# → http://localhost:5173
```

---

## Commandes essentielles

```bash
# ── Ingestion ──────────────────────────────────────────────────────────────

# Alimenter la base (télécharge audio en RAM, calcule embeddings + fingerprints, indexe)
python manage.py ingest --csv data/kaggle/data/spotify-streaming-top-50-world.csv

# Enrichir la base avec des variantes acoustiques de chaque morceau
python manage.py augment

# Enrichir les métadonnées (Deezer + MusicBrainz)
python manage.py enrich

# ── Identification ─────────────────────────────────────────────────────────

# Identifier un morceau (résultat simple)
python manage.py identify data/raw/mon_audio.mp3

# Avec le détail des scores et le top 10
python manage.py identify data/raw/mon_audio.mp3 --top 10 --detailed

# Télécharger un extrait de test
python manage.py download-audio "Daft Punk Get Lucky" --duration 30 --position middle

# ── Interface web ──────────────────────────────────────────────────────────

python manage.py start-webapp              # dev  → http://localhost:5173
python manage.py start-webapp --prod       # prod → http://localhost:8000

# ── Maintenance ────────────────────────────────────────────────────────────

# Vérifier l'intégrité des données
python manage.py check
python manage.py check --details           # détail des warnings par catégorie
python manage.py check --purge --yes       # supprimer les tracks corrompus

# Reconstruire l'index FAISS manuellement (après une purge)
python manage.py build-index
```

> La référence exhaustive de toutes les commandes et options est dans **[COMMANDS.md](./COMMANDS.md)**.

---

## Configuration

Tous les paramètres sont centralisés dans `src/config.py`. Aucune constante ne doit être dupliquée ailleurs.

```python
# ── Méthode d'embedding active ─────────────────────────────────────────────
EMBEDDING_METHOD = "clap"       # "mfcc" | "clap" | "muq" | "mert"

# ── Pipeline d'identification ──────────────────────────────────────────────
VECTOR_TOP_N_TRACKS = 50        # tracks candidats agrégés par Stage 1 et soumis au Stage 2
SEGMENT_WIN_S       = 5.0       # durée d'une fenêtre audio (secondes)
SEGMENT_HOP_S       = 3.0       # pas entre deux fenêtres (secondes)

# ── Augmentation RIR ───────────────────────────────────────────────────────
RIR_SOURCE  = "synthetic"       # "synthetic" | "mit"
RIR_N       = 5                 # nombre de RIRs appliquées par track
RIR_MIT_DIR = "data/rir"        # dossier des WAV MIT (si RIR_SOURCE = "mit")

# ── Interface web ──────────────────────────────────────────────────────────
UI_LISTEN_DURATION  = 15        # durée d'enregistrement micro (secondes)
UI_CONFIDENCE_RATIO = 2.5       # ratio score[0]/score[1] → badge "certain"

# ── Téléchargement ─────────────────────────────────────────────────────────
DOWNLOAD_WORKERS = 5            # workers parallèles yt-dlp
```

---

## Interface web

L'interface web permet de reconnaître un morceau directement depuis le navigateur, sans installation supplémentaire côté client.

### Routes API (FastAPI)

| Méthode | Route | Corps / Réponse |
|---------|-------|-----------------|
| `POST` | `/api/identify` | `multipart/form-data` → JSON (résultats + recommandations) |
| `GET` | `/api/config` | JSON (`listen_duration`, `embedding_method`, `confidence_ratio`) |
| `GET` | `/api/health` | `{"status": "ok"}` |

### Fonctionnalités

- **Enregistrement micro** — countdown animé, durée configurable via `UI_LISTEN_DURATION`
- **Dépôt de fichier** — glisser-déposer ou sélection (MP3, WAV, FLAC, OGG, WebM…)
- **Résultat** — pochette d'album, titre, artiste, score de confiance, liens de streaming directs (YouTube, Spotify, Deezer, Apple Music)
- **Recommandations** — 4 morceaux similaires avec modal de détail et liens
- **Mode debug** — bouton `</>` : top 10 candidats avec scores FP et FAISS séparés
- **Thème sombre / clair** et **interface bilingue** FR / EN

### Modes de lancement

| Mode | Frontend | Backend | Accès |
|------|----------|---------|-------|
| Développement | Vite hot-reload `:5173` | uvicorn `--reload` `:8000` | http://localhost:5173 |
| Production | Build statique dans `dist/` | uvicorn `:8000` | http://localhost:8000 |

---

## Augmentation RIR

Une Room Impulse Response (RIR) est la réponse acoustique d'un espace à un son impulsionnel — elle capture comment une salle colore le son (réflexions, réverbération, absorption). Convoluer un morceau avec une RIR produit une version qui sonne comme si elle avait été enregistrée dans cet espace.

Le principe : pour chaque track de la base, on génère N versions réverbérées par convolution FFT (`scipy.signal.fftconvolve`), on calcule leurs embeddings, et on les ajoute à ChromaDB et à l'index FAISS. Quand une requête arrive depuis un environnement réverbérant, elle trouve naturellement ses plus proches voisins parmi ces versions augmentées.

**L'opération est idempotente :** les RIRs déjà appliquées à un track sont mémorisées dans le champ `rir_augmented` de `metadata.parquet` et ignorées lors d'un second appel. Seules les RIRs manquantes sont calculées.

### Génération des RIRs synthétiques

Chaque RIR synthétique est construite en trois composantes :

1. **Son direct** — pic à 1 ms
2. **Réflexions précoces** — 12 à 30 réflexions aléatoires avec atténuation exponentielle en fonction du RT60
3. **Queue diffuse** — bruit gaussien décroissant après 50 ms, simulant la réverbération tardive

Les 10 environnements prédéfinis couvrent une large plage de RT60 :

| Environnement | RT60 |
|---------------|------|
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

### Sources disponibles

| Source | Description | Avantages |
|--------|-------------|-----------|
| `synthetic` | RIRs mathématiques générées (10 environnements, RT60 : 0.15 s – 1.60 s) | Aucun téléchargement, reproductible, rapide |
| `mit` | Vraies RIRs mesurées en conditions réelles (MIT Acoustical Survey) | Plus réalistes, nécessite les WAV dans `data/rir/` |

Avec `source = "mit"`, le système charge tous les WAV disponibles, estime le RT60 de chacun par intégrale de Schroeder (décroissance d'énergie de −60 dB depuis le pic), puis sélectionne les N les plus diversifiés par échantillonnage uniforme sur la courbe RT60 triée — garantissant une couverture maximale de la plage acoustique.

---

## Évaluation

Le projet inclut une suite d'évaluation complète pour mesurer et comparer les performances du pipeline.

### Métriques calculées

| Métrique | Description |
|----------|-------------|
| **Top-1** | Le bon morceau est en première position |
| **Top-5** | Le bon morceau est dans les 5 premiers résultats |
| **MRR** | Mean Reciprocal Rank — mesure la qualité du classement |
| **Latence** | Temps d'identification en secondes |

### Conditions de dégradation

| Condition | Description |
|-----------|-------------|
| `clean` | Audio sans dégradation |
| `snr_20` | Bruit blanc à 20 dB SNR |
| `snr_10` | Bruit blanc à 10 dB SNR (dégradation forte) |
| `reverb` | Réverbération simulée |
| `combo` | SNR 10 dB + réverbération combinés |

### Workflow d'évaluation

```bash
# 1. Télécharger des extraits de test (30s, position milieu recommandée)
python manage.py download-audio "Miley Cyrus Flowers"        --duration 30 --position middle
python manage.py download-audio "Travis Scott PARASAIL"      --duration 30 --position middle
python manage.py download-audio "The Weeknd Blinding Lights" --duration 30 --position middle

# 2. Évaluation pipeline complet — Top-1, Top-5, MRR, latence
python manage.py evaluate --methods mfcc --methods clap

# 3. Évaluation impact RIR (Stage 1 avec vs sans augmentation)
python manage.py rir-evaluate --methods clap

# 4. Générer les 7 graphiques PNG pour le rapport
python manage.py plots \
  --eval     results/eval/eval_*.json \
  --rir-eval results/eval/rir_eval_*.json
```

### Graphiques produits

| Fichier | Graphique | Source |
|---------|-----------|--------|
| `rir_paired_bar_*.png` | G1 — Accuracy avec vs sans RIR par condition | `rir-evaluate` |
| `rir_delta_*.png` | G2 — Gain Δ apporté par l'augmentation RIR | `rir-evaluate` |
| `rir_faiss_scores_*.png` | G4 — Score FAISS par morceau avec/sans RIR | `rir-evaluate` |
| `method_accuracy.png` | G6 — Accuracy Top-1 par méthode et condition | `evaluate` |
| `stage_comparison.png` | G9 — Stage 1 (FAISS seul) vs Stage 2 (+ fingerprint) | `evaluate` |
| `duration_impact.png` | G11 — Accuracy en fonction de la durée d'extrait | `evaluate` |
| `heatmap_accuracy.png` | G12 — Heatmap méthodes × conditions | `evaluate` |

---

## Stockage des données

### ChromaDB — `data/chroma/`

Base de vecteurs d'embeddings. Une collection par méthode (ex : `clap_clap_htsat_unfused`). Chaque document correspond à un segment audio de 5 secondes, identifié par `{track_id}_{segment_index}`, et annoté de `track_id` et `start_s`.

### SQLite — `data/features/fingerprints.db`

Fingerprints spectraux Shazam. Une ligne par track (`INSERT OR REPLACE`, idempotent). Mode WAL activé pour résister aux accès concurrents. Chaque entrée contient les hashes de la constellation spectrale sérialisés en BLOB.

### FAISS — `data/index/`

Index de recherche vectorielle par méthode et type :
- `index_{method}_{type}.faiss` — vecteurs indexés
- `segments_{method}.parquet` — mapping position FAISS → (`track_id`, `start_s`)

Trois types d'index sont disponibles, configurables via `INDEX_TYPE` dans `config.py` ou avec `--index-type` à la construction :

| Type | Algorithme | Précision | Vitesse | Paramètres |
|------|-----------|-----------|---------|------------|
| `flat` | Bruteforce — produit scalaire exact (`IndexFlatIP`) | Exacte | Lente | Aucun |
| `hnsw` | Hierarchical Navigable Small World (`IndexHNSWFlat`) | ~99 % | Rapide | M=32, efConstruction=40 |
| `ivf` | Listes inversées (`IndexIVFFlat`) | Approchée | Rapide | nlist=√N |

`flat` est recommandé pour des bases de moins de 100 000 vecteurs — la taille du projet ne justifie pas l'approximation. `hnsw` devient pertinent au-delà. La similarité utilisée est le **produit scalaire** (inner product), les embeddings étant normalisés en amont.

### Parquet — `data/processed/metadata.parquet`

Table centrale des tracks. Colonnes principales :

| Colonne | Description |
|---------|-------------|
| `track_id` | Hash MD5 `artiste_titre` |
| `title`, `artist` | Identité du morceau |
| `album`, `genre`, `release_date` | Métadonnées enrichies |
| `cover_url` | URL de la pochette |
| `embedded_methods` | Liste des méthodes calculées |
| `rir_augmented` | Dict `{collection: [rir_names]}` |

Écriture atomique (fichier temporaire + rename) pour résister aux crashs en cours d'ingestion.

---

## Points d'attention

**Reprise automatique sur crash** — `ingest` sauvegarde après chaque track. Un arrêt brutal ne perd qu'au plus le track en cours. Relancer la même commande reprend exactement là où ça s'est arrêté grâce au champ `embedded_methods`.

**Audio en RAM** — l'audio téléchargé via `ingest` n'est jamais écrit sur disque. Il transite directement en mémoire via pipe ffmpeg → librosa. Seuls les embeddings, fingerprints et index sont persistés.

**Multi-méthode** — plusieurs méthodes peuvent coexister dans la base. Modifier `EMBEDDING_METHOD` dans `config.py` et relancer `ingest` : les tracks déjà traités pour cette méthode sont ignorés, les autres sont complétés.

**MuQ sur Apple Silicon** — MuQ ne supporte pas Float16 sur CPU ni sur MPS (opérations ComplexFloat non supportées). Il fonctionne exclusivement sur CUDA. Sur Mac sans GPU NVIDIA, utiliser `clap` ou `mfcc`.

**SQLite et accès concurrents** — si le projet réside dans un dossier synchronisé par iCloud, les uploads en arrière-plan peuvent verrouiller `fingerprints.db` et provoquer des erreurs `database is locked`. Le dossier `data/` est exclu d'iCloud via attribut `xattr`. Le mode WAL et un mécanisme de retry sont activés pour gérer les contentions résiduelles.

**Après un `check --purge`** — l'index FAISS est supprimé pendant la purge. Relancer `python manage.py build-index` est obligatoire avant toute identification.

---

## Technologies

| Couche | Outils et modèles |
|--------|-------------------|
| **Embeddings audio** | `laion/clap-htsat-unfused`, `laion/larger_clap_music`, `OpenMuQ/MuQ-large-msd-iter`, `m-a-p/MERT-v1-95M`, librosa (MFCC) |
| **Recherche vectorielle** | FAISS (Flat / HNSW / IVF), ChromaDB |
| **Fingerprinting** | Constellation spectrale Shazam — librosa, scipy (FFT, corrélation) |
| **Deep learning** | PyTorch (CUDA / MPS / CPU), Transformers (HuggingFace) |
| **Audio** | librosa, soundfile, torchaudio, yt-dlp, ffmpeg |
| **Backend** | FastAPI, uvicorn, pandas, SQLite (WAL), Apache Parquet |
| **Frontend** | React 18, Vite, JavaScript ES6+ |
| **Évaluation** | matplotlib, numpy, scipy |

---

## Améliorations possibles

Les pistes suivantes ont été identifiées mais non implémentées dans le cadre du projet.

**Pipeline d'identification**
- Normalisation croisée des scores FAISS et fingerprint pour rendre le poids relatif indépendant de la taille de la base
- Fenêtrage adaptatif des segments : fenêtres plus courtes pour les morceaux à forte variation temporelle, plus longues pour les morceaux stables
- Mise en cache de l'index FAISS en mémoire pour éliminer le temps de chargement lors d'identifications successives
- Support du streaming audio en temps réel (WebSocket) plutôt qu'un enregistrement de durée fixe

**Embeddings**
- Fine-tuning de CLAP ou MuQ sur un corpus de morceaux dégradés (bruit, reverb) pour améliorer la robustesse en conditions difficiles
- Fusion tardive de plusieurs méthodes d'embedding (score ensemble) pour combiner leurs forces respectives
- Quantification des vecteurs (PQ — Product Quantization) pour réduire l'empreinte mémoire de l'index FAISS

**Augmentation**
- Ajout de conditions de dégradation supplémentaires lors de l'augmentation : compression MP3 artefacts, variations de tempo, changements de tonalité
- Utilisation de vrais enregistrements de bruit ambiant (cafés, transports) plutôt que du bruit blanc gaussien

**Infrastructure**
- Remplacement de ChromaDB par un vrai serveur vectoriel (Qdrant, Weaviate, Milvus) pour passer à une base de plusieurs millions de titres
- Indexation incrémentale FAISS sans reconstruction complète après chaque ingestion
- API d'ingestion exposée via REST pour alimenter la base sans accès direct au serveur

**Interface web**
- Historique des identifications côté client (localStorage)
- Partage du résultat via lien court
- Affichage des paroles synchronisées (via API externe)

---

## Références

Les articles de référence ayant guidé les choix techniques sont disponibles dans le dossier `research_paper/`.

| Auteur(s) | Titre | Pertinence |
|-----------|-------|------------|
| Wang, A. (2003) | *An Industrial Strength Audio Search Algorithm* | Fondement du fingerprinting Shazam — constellation spectrale et alignement temporel par histogramme d'offsets |
| Wu et al. (2023) | *Large-Scale Contrastive Language-Audio Pretraining* (CLAP) | Modèle d'embedding audio-texte multimodal utilisé en Stage 1 |
| Zhu et al. (2024) | *MuQ: Self-Supervised Music Representation* | Modèle de représentation musicale auto-supervisé, meilleure précision sur corpus musical |
| Castellon et al. | *Music2Latent2: Audio Embedding via Diffusion* | Méthode alternative d'embedding par modèle de diffusion latent |
| Défossez et al. | *A Fast Audio Similarity Retrieval Method* | Approche de recherche par similarité audio à grande échelle |
| Microsoft Research | *Audio Search and Retrieval* | Techniques industrielles de recherche audio à grande échelle |
| — | *Fast Music Identification* | Comparaison d'approches de fingerprinting pour l'identification rapide |
| — | *Audio Fingerprinting* | Revue des méthodes de fingerprinting spectral |
| — | *Predicting Song Title from Audio* | Approches de reconnaissance musicale par apprentissage supervisé |

---

## Équipe

Projet réalisé dans le cadre du Master Intelligence Artificielle et Data Science (IAD), Semestre 2 — parcours Big Data.

| Étudiant | GitHub |
|----------|--------|
| AIT MOKHTAR Clara | [@claraait123](https://github.com/claraait123) |
| AYDIN Maria | [@Mmajora53](https://github.com/Mmajora53) |
| GOURMELEN Thomas | [@thmsgo18](https://github.com/thmsgo18) |
| TAN Vincent | [@20centan](https://github.com/20centan) |
