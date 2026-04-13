# Shazam Maison — Reconnaissance Audio par Embeddings + Fingerprinting

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Méthodes](https://img.shields.io/badge/Embeddings-MFCC%20%7C%20CLAP%20%7C%20MuQ-green)
![Index](https://img.shields.io/badge/Index-ChromaDB%20%2B%20FAISS-orange)
![Stockage](https://img.shields.io/badge/Fingerprints-SQLite-lightgrey)
![Interface](https://img.shields.io/badge/Interface-React%20%2B%20FastAPI-blueviolet)

Système de reconnaissance musicale inspiré de Shazam. À partir d'un extrait audio (micro ou fichier), retrouve le morceau correspondant dans une base de données vectorielle et affiche les liens de streaming ainsi que des recommandations similaires.

---

## Cheat Sheet

```bash
# 1. Alimenter la base (télécharge + calcule embeddings + fingerprints + construit l'index)
python manage.py ingest --csv data/kaggle/data/spotify-streaming-top-50-world.csv

# 2. Enrichir les métadonnées (album, genre, date, pochette) via Deezer
python manage.py enrich

# 3. Lancer l'interface web
python manage.py start-webapp              # mode dev  (hot-reload, frontend :5173)
python manage.py start-webapp --prod       # mode prod (build + tout sur :8000)

# 4. Identifier un morceau en ligne de commande
python manage.py identify data/raw/mon_audio.mp3
python manage.py identify data/raw/mon_audio.mp3 --method clap --top 10 --detailed

# 5. Télécharger un audio de test
python manage.py download-audio "Miley Cyrus Flowers" --duration 30 --position middle

# 6. Vérifier l'intégrité des données (vue résumé)
python manage.py check

# 7. Voir le détail des problèmes d'embeddings/fingerprints
python manage.py check --details

# 8. Voir les tracks avec métadonnées manquantes
python manage.py check --metadata

# 9. Supprimer les tracks problématiques (purge chirurgicale)
python manage.py check --purge

# 10. Reconstruire l'index FAISS (si nécessaire)
python manage.py build-index
```

---

## Point d'entrée unique : `manage.py`

Toutes les opérations du projet passent par `manage.py` à la racine. La logique métier vit dans `src/` ; `manage.py` est une interface CLI légère.

```
python manage.py <commande> [options]
```

### Vue d'ensemble des commandes

| Groupe | Commande | Rôle |
|--------|----------|------|
| **Ingestion** | `ingest` | Télécharge + embeddings + fingerprints + index |
| | `augment` | Ajoute des vecteurs RIR (Room Impulse Response) aux tracks existants |
| | `rebuild-fingerprints` | Recalcule les fingerprints sans re-télécharger |
| | `build-index` | Reconstruit l'index FAISS depuis ChromaDB |
| **Maintenance** | `check` | Vérifie l'intégrité des données + purge |
| | `enrich` | Enrichit les métadonnées (Deezer + MusicBrainz) |
| | `clean` | Supprime toutes les données (ChromaDB, FAISS, fingerprints, metadata) |
| | `delete-rir` | Supprime les segments RIR d'une méthode dans ChromaDB |
| **Évaluation** | `identify` | Identifie un morceau audio (usage normal) |
| | `find-track` | Teste la reconnaissance avec track_id cible (évaluation) |
| | `benchmark` | Benchmark robustesse sur un seul morceau (Flowers) |
| | `evaluate` | Évaluation multi-tracks multi-méthodes (Top-1, Top-5, MRR, latence) |
| | `rir-evaluate` | Compare Stage 1 avec vs sans RIR sur plusieurs morceaux |
| | `plots` | Génère les 7 graphiques PNG pour le rapport (G1,G2,G4,G6,G9,G11,G12) |
| | `rir-impact` | Analyse RIR détaillée sur un seul fichier (affichage riche) |
| **Utilitaires** | `download-audio` | Télécharge un audio de test dans `data/raw/` |
| | `start-webapp` | Lance backend FastAPI + frontend React |

---

## Commandes détaillées

### `ingest` — Alimenter la base

Télécharge l'audio en RAM via yt-dlp, calcule embeddings + fingerprints, stocke dans ChromaDB + SQLite, puis reconstruit l'index FAISS. **Aucun MP3 n'est stocké sur disque.**

```bash
python manage.py ingest [OPTIONS]
```

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--csv PATH` | Path (multiple) | tous les CSV Kaggle | Fichier(s) CSV ou dossier de CSV Kaggle Spotify |
| `--method TEXT` | mfcc/clap/muq | `config.EMBEDDING_METHOD` | Méthode d'embedding à utiliser |

```bash
# Un seul CSV
python manage.py ingest --csv data/kaggle/data/spotify-streaming-top-50-world.csv

# Plusieurs CSV
python manage.py ingest \
  --csv data/kaggle/data/spotify-streaming-top-50-france.csv \
  --csv data/kaggle/data/spotify-streaming-top-50-usa.csv

# Tous les CSV d'un dossier
python manage.py ingest --csv data/kaggle/data/

# Sans --csv : utilise tous les CSV disponibles
python manage.py ingest

# Avec une méthode spécifique
python manage.py ingest --csv data/kaggle/data/ --method mfcc
```

> **Reprise automatique :** les tracks déjà traités pour la méthode active sont ignorés (champ `embedded_methods` dans `metadata.parquet`). En cas de crash, relancer suffit.

---

### `augment` — Ajouter des tracks via YouTube

Recherche un morceau sur YouTube, télécharge son audio en RAM et l'ajoute à la base (embeddings + fingerprint + index). Utile pour enrichir la base sans CSV.

```bash
python manage.py augment [OPTIONS]
```

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--query TEXT` | string | — | Requête YouTube (ex : `"Daft Punk Get Lucky"`) |
| `--method TEXT` | mfcc/clap/muq | `config.EMBEDDING_METHOD` | Méthode d'embedding |

```bash
python manage.py augment --query "Daft Punk Get Lucky"
python manage.py augment --query "Daft Punk Get Lucky" --method clap
```

---

### `rebuild-fingerprints` — Recalculer les fingerprints

Recalcule les fingerprints de tous les tracks (ou ceux manquants seulement) sans re-télécharger l'audio. Utile après un `--purge-missing-fp` ou si le fichier SQLite a été corrompu.

```bash
python manage.py rebuild-fingerprints [OPTIONS]
```

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--force` | flag | False | Recalcule tous les tracks, même ceux déjà dans SQLite |

```bash
# Recalcule uniquement les fingerprints manquants
python manage.py rebuild-fingerprints

# Tout recalculer
python manage.py rebuild-fingerprints --force
```

---

### `build-index` — Construire l'index FAISS

Reconstruit l'index FAISS depuis les embeddings ChromaDB. Appelé automatiquement par `ingest`, mais à relancer manuellement après un `check --purge`.

```bash
python manage.py build-index [OPTIONS]
```

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--method TEXT` | mfcc/clap/muq | toutes les méthodes | Méthode à indexer |
| `--index-type TEXT` | flat/hnsw/ivf | `config.INDEX_TYPE` | Type d'index FAISS |

```bash
# Toutes les méthodes, type défini dans config.py
python manage.py build-index

# Méthode spécifique
python manage.py build-index --method clap

# Type d'index différent
python manage.py build-index --index-type hnsw
```

---

### `check` — Vérifier l'intégrité des données

Vérifie la cohérence de ChromaDB, FAISS, SQLite et `metadata.parquet`. Peut supprimer chirurgicalement les tracks problématiques.

```bash
python manage.py check [OPTIONS]
```

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--method TEXT` | mfcc/clap/muq | toutes | Filtrer sur une méthode |
| `--details` | flag | False | Détail des warnings par catégorie |
| `--metadata` | flag | False | Tracks avec métadonnées manquantes |
| `--purge` | flag | False | Supprimer les tracks problématiques |
| `--purge-missing-fp` | flag | False | Supprimer uniquement les tracks sans fingerprint |
| `--yes` | flag | False | Ne pas demander confirmation avant la purge |

```bash
# Vue résumé
python manage.py check

# Filtrer sur clap uniquement
python manage.py check --method clap

# Détail des warnings
python manage.py check --details

# Tracks sans métadonnées complètes
python manage.py check --metadata

# Purge avec confirmation
python manage.py check --purge

# Purge silencieuse (CI/CD)
python manage.py check --purge --yes

# Purge seulement les fingerprints manquants
python manage.py check --purge-missing-fp
```

#### Checks effectués (`--details`)

| Code | Type | Description |
|------|------|-------------|
| C1 | Critique | Dimension des embeddings inattendue |
| C2 | Critique | NaN ou Inf dans les embeddings |
| C3 | Critique | Désynchronisation ChromaDB ↔ order parquet |
| C5 | Critique | FAISS index désynchronisé avec ChromaDB |
| C6 | Critique | Segments orphelins (ChromaDB sans metadata) |
| C6b | Critique | Track marqué traité dans metadata mais sans segments dans ChromaDB |
| C7 | Critique | Embedding incomplet (< 80 % des segments attendus) |
| Q1 | Qualité | Durée aberrante (≤ 0s ou > 10min) |
| Q2 | Qualité | Segment `start_s` dépassant la durée du track de plus de 5s |
| Q3 | Qualité | Fingerprint vide (0 hash) |
| Q4 | Qualité | Fingerprint anormalement pauvre (outlier IQR) |
| FP | Qualité | Tracks sans fingerprint (Stage 2 inopérant) |

#### Que fait `--purge` ?

La purge est **par méthode**, pas par track entier :
1. Suppression des segments de la méthode dans **ChromaDB**
2. Retrait de la méthode dans `embedded_methods` (**metadata.parquet**)
3. Si `embedded_methods` devient vide → ligne supprimée + fingerprint retiré de **SQLite**
4. Index FAISS de la méthode supprimé (relancer `build-index` après)

---

### `enrich` — Enrichir les métadonnées

Complète `metadata.parquet` avec `album`, `genre`, `release_date`, `cover_url` via Deezer puis MusicBrainz en fallback.

```bash
python manage.py enrich [OPTIONS]
```

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--force` | flag | False | Ré-enrichir même les tracks déjà complets |

```bash
# Enrichir les tracks avec au moins un champ vide (défaut)
python manage.py enrich

# Forcer la mise à jour de tous les tracks
python manage.py enrich --force
```

---

### `clean` — Supprimer un track

Supprime proprement un track de tous les stores (ChromaDB, SQLite, metadata.parquet, index FAISS).

```bash
python manage.py clean [OPTIONS]
```

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--track-id TEXT` | string | — | ID exact du track à supprimer |
| `--name TEXT` | string | — | Recherche par nom (artiste ou titre) |
| `--yes` | flag | False | Ne pas demander confirmation |

```bash
python manage.py clean --track-id f01ab00f1fdc5a57fd2676f4d68631a8
python manage.py clean --name "Miley Cyrus"
python manage.py clean --name "Flowers" --yes
```

---

### `delete-rir` — Supprimer les segments RIR

Supprime les segments augmentés par Room Impulse Response (RIR) d'un ou de tous les tracks dans ChromaDB.

```bash
python manage.py delete-rir [OPTIONS]
```

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--track-id TEXT` | string | tous | Track spécifique (sinon tous) |
| `--method TEXT` | mfcc/clap/muq | `config.EMBEDDING_METHOD` | Méthode concernée |
| `--yes` | flag | False | Ne pas demander confirmation |

```bash
# Supprimer les RIR de tous les tracks
python manage.py delete-rir

# Supprimer les RIR d'un seul track
python manage.py delete-rir --track-id f01ab00f1fdc5a57fd2676f4d68631a8

# Sans confirmation
python manage.py delete-rir --yes
```

---

### `evaluate` — Évaluation multi-tracks multi-méthodes

Évalue chaque méthode sur un ensemble de fichiers de test avec plusieurs conditions de dégradation. Produit des métriques Top-1, Top-5, MRR et latence, sauvegardées en JSON **et** en graphiques.

```bash
python manage.py evaluate [OPTIONS]
```

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--methods` | multiple | mfcc, clap | Méthodes à évaluer (répétable) |
| `--conditions` | multiple | toutes (5) | Conditions de dégradation (répétable) |
| `--n-tracks` | int | 0 (tous) | Limiter à N tracks du manifest |
| `--no-plot` | flag | False | Ne pas générer les graphiques |

**Conditions disponibles :** `clean` · `snr_20` · `snr_10` · `reverb` · `combo`

```bash
# Évaluation par défaut (mfcc + clap, 5 conditions, tous les tracks du manifest)
python manage.py evaluate

# Méthodes spécifiques
python manage.py evaluate --methods mfcc --methods clap --methods mert

# Conditions réduites (rapide)
python manage.py evaluate --conditions clean --conditions snr_20

# Sans graphiques automatiques
python manage.py evaluate --no-plot
```

> **Prérequis :** le manifest doit contenir des fichiers de test. Il est alimenté automatiquement par `download-audio` pour les tracks déjà dans la base.

**Produit :**
- `results/eval/eval_TIMESTAMP.json` — métriques complètes (Top-1/Top-5/MRR/latence + per_track)
- `results/plots/method_accuracy.png` — G6 : accuracy par méthode × condition (avec écart-type)
- `results/plots/stage_comparison.png` — G9 : Stage 1 (FAISS) vs Stage 2 (+ fingerprint)
- `results/plots/duration_impact.png` — G11 : accuracy vs durée d'extrait
- `results/plots/heatmap_accuracy.png` — G12 : heatmap méthodes × conditions

---

### `rir-evaluate` — Comparer Stage 1 avec vs sans RIR

Compare la précision Stage 1 (FAISS uniquement) avec et sans les vecteurs RIR dans l'index, sur l'ensemble des morceaux du manifest et plusieurs conditions de dégradation. Construit l'index sans RIR en mémoire — ne modifie pas la base.

```bash
python manage.py rir-evaluate [OPTIONS]
```

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--methods` | multiple | `config.EMBEDDING_METHOD` | Méthodes à évaluer (répétable) |
| `--conditions` | multiple | toutes (5) | Conditions de dégradation (répétable) |
| `--n-tracks` | int | 0 (tous) | Limiter à N tracks du manifest |
| `--no-plot` | flag | False | Ne pas générer les graphiques |

```bash
# Méthode par défaut, toutes les conditions
python manage.py rir-evaluate

# CLAP uniquement, conditions spécifiques
python manage.py rir-evaluate --methods clap --conditions clean --conditions reverb

# Limiter à 5 morceaux
python manage.py rir-evaluate --methods mfcc --methods clap --n-tracks 5
```

**Produit :**
- `results/eval/rir_eval_TIMESTAMP.json` — ranks et scores FAISS avec/sans RIR par track × condition
- `results/plots/rir_paired_bar_*.png` — G1 : accuracy avec vs sans RIR
- `results/plots/rir_delta_*.png` — G2 : gain Δ apporté par les RIR
- `results/plots/rir_faiss_scores_*.png` — G4 : score FAISS par morceau avec/sans RIR

---

### `plots` — Générer les graphiques du rapport

Lit des JSON d'évaluation et/ou d'évaluation RIR et produit **7 graphiques PNG** dans `results/plots/`.

```bash
python manage.py plots [OPTIONS]
```

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--eval JSON` | multiple | — | JSON d'évaluation multi-tracks — `results/eval/eval_*.json` (répétable) |
| `--rir-eval JSON` | multiple | — | JSON d'évaluation RIR — `results/eval/rir_eval_*.json` (répétable) |
| `--out-dir PATH` | Path | `results/plots/` | Dossier de sortie |

```bash
# Graphiques pipeline uniquement (G6, G9, G11, G12)
python manage.py plots --eval results/eval/eval_*.json

# Graphiques RIR uniquement (G1, G2, G4)
python manage.py plots --rir-eval results/eval/rir_eval_*.json

# Tous les graphiques
python manage.py plots \
  --eval     results/eval/eval_*.json \
  --rir-eval results/eval/rir_eval_*.json
```

**Graphiques produits depuis `--rir-eval` (comparaison RIR) :**

| Fichier | Description |
|---------|-------------|
| `rir_paired_bar_*.png` | G1 — Précision avec vs sans RIR par condition (barres groupées) |
| `rir_delta_*.png` | G2 — Gain Δ apporté par les RIR en points de pourcentage |
| `rir_faiss_scores_*.png` | G4 — Score FAISS du bon morceau par track avec/sans RIR |

**Graphiques produits depuis `--eval` (pipeline complet) :**

| Fichier | Description |
|---------|-------------|
| `method_accuracy.png` | G6 — Précision Top-1 par méthode × condition (avec écart-type) |
| `stage_comparison.png` | G9 — Stage 1 (FAISS seul) vs Stage 2 (+ fingerprint) |
| `duration_impact.png` | G11 — Précision vs durée de l'extrait (5 s / 10 s / 15 s / 30 s) |
| `heatmap_accuracy.png` | G12 — Heatmap méthodes × conditions (% accuracy, colormap continue) |

---

### `identify` — Identifier un morceau

Identifie un fichier audio et affiche les morceaux les plus probables. C'est la commande à utiliser au quotidien — l'équivalent CLI de l'interface web.

```bash
python manage.py identify AUDIO [OPTIONS]
```

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `AUDIO` | Path | — | Fichier audio à identifier (MP3, WAV, FLAC…) |
| `--method TEXT` | mfcc/clap/muq | `config.EMBEDDING_METHOD` | Méthode d'embedding |
| `--top INTEGER` | int | 5 | Nombre de résultats à afficher |
| `--detailed` | flag | False | Afficher les scores FAISS et fingerprint séparément |

```bash
# Identification simple
python manage.py identify data/raw/mon_audio.mp3

# Afficher le top 10 avec détail des scores
python manage.py identify data/raw/mon_audio.mp3 --top 10 --detailed

# Forcer une méthode spécifique
python manage.py identify data/raw/mon_audio.mp3 --method clap
```

---

### `find-track` — Tester la reconnaissance

Teste la reconnaissance d'un fichier audio et affiche si le bon track est trouvé, avec son rang et ses scores.

```bash
python manage.py find-track [OPTIONS]
```

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--audio PATH` | Path | `data/raw/flowers_middle_30s.mp3` | Fichier audio à tester |
| `--target TEXT` | string | ID de Flowers (Miley Cyrus) | Track_id attendu en réponse |
| `--top INTEGER` | int | 20 | Nombre de candidats à afficher |
| `--method TEXT` | mfcc/clap/muq | `config.EMBEDDING_METHOD` | Méthode d'embedding |

```bash
# Test avec les valeurs par défaut (Flowers, 30s, méthode config)
python manage.py find-track

# Test sur un fichier spécifique
python manage.py find-track \
  --audio data/raw/mon_audio.mp3 \
  --target <track_id> \
  --top 10 \
  --method clap
```

---

### `benchmark` — Benchmark de robustesse

Évalue la robustesse du système sur 5 cas de dégradation : bruit blanc, reverb, coupure basse fréquence, compression opus, et extrait court.

```bash
python manage.py benchmark [OPTIONS]
```

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--label TEXT` | string | horodatage | Nom du run (pour comparer plusieurs runs) |
| `--full` | flag | False | Suite complète (plus longue) |
| `--compare PATH` | Path (multiple) | — | Comparer plusieurs fichiers JSON de résultats |

```bash
# Lancer un benchmark
python manage.py benchmark

# Benchmark complet avec label
python manage.py benchmark --label "clap-v2" --full

# Comparer deux runs
python manage.py benchmark \
  --compare results/benchmark/clap-v1.json \
  --compare results/benchmark/clap-v2.json
```

#### Suite de tests

| # | Dégradation | Paramètre |
|---|-------------|-----------|
| 1 | Bruit blanc | SNR = 20 dB |
| 2 | Reverb légère | RIR simulé |
| 3 | Passe-haut 300 Hz | (simule téléphone) |
| 4 | Compression Opus | 64 kbps |
| 5 | Extrait court | 5 s (1 seul segment) |

Les résultats sont sauvegardés en JSON dans `results/benchmark/`.

---

### `rir-impact` — Impact de l'augmentation RIR

Compare les résultats Stage 1 avec et sans les segments augmentés RIR pour un fichier audio donné.

```bash
python manage.py rir-impact [OPTIONS]
```

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--audio PATH` | Path | `data/raw/flowers_middle_30s.mp3` | Fichier audio à tester |
| `--target TEXT` | string | ID de Flowers | Track_id attendu |
| `--top INTEGER` | int | 20 | Nombre de candidats |
| `--method TEXT` | mfcc/clap/muq | `config.EMBEDDING_METHOD` | Méthode |

```bash
python manage.py rir-impact

python manage.py rir-impact \
  --audio data/raw/mon_audio.mp3 \
  --target <track_id> \
  --method clap
```

---

### `download-audio` — Télécharger un audio de test

Télécharge un morceau depuis YouTube dans `data/raw/` pour tester la reconnaissance. Contrairement à `ingest`, **le fichier MP3 est stocké sur disque**.

```bash
python manage.py download-audio [OPTIONS]
```

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `QUERY` | string | — | Requête YouTube (ex : `"Miley Cyrus Flowers"`) |
| `--duration INTEGER` | 5/10/15/30 | morceau entier | Durée de l'extrait en secondes |
| `--position TEXT` | voir ci-dessous | `middle` | Position dans le morceau |

Positions disponibles : `start` · `first-quarter` · `middle` · `third-quarter` · `end`

```bash
# Morceau entier
python manage.py download-audio "Miley Cyrus Flowers"

# Extrait de 30s au milieu
python manage.py download-audio "Miley Cyrus Flowers" --duration 30 --position middle

# Extrait de 15s au premier quart
python manage.py download-audio "Travis Scott PARASAIL" --duration 15 --position first-quarter
```

Le fichier est nommé automatiquement : `Titre__position_durées.mp3`
Ex : `Miley Cyrus - Flowers (Official Video)__middle_30s.mp3`

> **Conseil :** `--position middle` donne les meilleurs résultats — le refrain est acoustiquement plus distinctif que l'intro.

---

### `start-webapp` — Lancer l'interface web

Lance le backend FastAPI et le frontend React.

```bash
python manage.py start-webapp [OPTIONS]
```

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--prod` | flag | False | Mode production (build statique + port unique) |
| `--port INTEGER` | int | 8000 | Port du backend FastAPI |

```bash
# Mode développement (hot-reload)
python manage.py start-webapp

# Mode production
python manage.py start-webapp --prod

# Port personnalisé
python manage.py start-webapp --port 8080
```

| Mode | Frontend | Backend | URL principale |
|------|----------|---------|----------------|
| Dev  | Vite hot-reload `:5173` | uvicorn --reload `:8000` | http://localhost:5173 |
| Prod | Build statique dans `dist/` | uvicorn `:8000` | http://localhost:8000 |

---

## Workflows typiques

### Générer les graphiques pour le rapport

```bash
# Étape 1 — Télécharger 10-15 clips de test (30s, refrain)
python manage.py download-audio "Miley Cyrus Flowers"        --duration 30 --position middle
python manage.py download-audio "Travis Scott PARASAIL"      --duration 30 --position middle
python manage.py download-audio "The Weeknd Blinding Lights" --duration 30 --position middle
# ... répéter pour chaque track dans la base

# Étape 2 — Évaluation multi-tracks (G6, G9, G11, G12)
python manage.py evaluate --methods mfcc --methods clap

# Étape 3 — Comparaison RIR (G1, G2, G4)
python manage.py rir-evaluate --methods clap

# Étape 4 — Générer tous les graphiques
python manage.py plots \
  --eval     results/eval/eval_*.json \
  --rir-eval results/eval/rir_eval_*.json

# → results/plots/ contient 7 graphiques PNG prêts pour le rapport
```

> **Optionnel :** `benchmark` reste disponible pour une analyse approfondie sur un seul morceau (14 cas de dégradation, affichage riche).

**Structure de `results/` :**
```
results/
├── EXPERIMENTS.md          # Journal des expériences (manuel)
├── benchmark/              # JSON de benchmark (un par run)
│   └── benchmark_TIMESTAMP_LABEL.json
├── eval/                   # JSON d'évaluation multi-tracks
│   ├── eval_TIMESTAMP.json
│   └── rir_eval_TIMESTAMP.json
└── plots/                  # Graphiques PNG générés
    ├── rir_paired_bar_clap.png    # G1  — accuracy avec vs sans RIR
    ├── rir_delta_clap.png         # G2  — gain Δ RIR
    ├── rir_faiss_scores_clap.png  # G4  — score FAISS par morceau
    ├── method_accuracy.png        # G6  — accuracy méthode × condition
    ├── stage_comparison.png       # G9  — Stage 1 vs Stage 2
    ├── duration_impact.png        # G11 — accuracy vs durée
    └── heatmap_accuracy.png       # G12 — heatmap méthodes × conditions
```

---

### Démarrage complet (première fois)

```bash
source venv/bin/activate
python manage.py ingest --csv data/kaggle/data/
python manage.py enrich
python manage.py start-webapp
```

### Ajouter des morceaux

```bash
python manage.py ingest --csv data/kaggle/data/nouveau_chart.csv
# L'index est reconstruit automatiquement à la fin de l'ingestion
```

### Changer de méthode d'embedding

```bash
# 1. Modifier EMBEDDING_METHOD dans src/config.py (ex: "mfcc" → "clap")
# 2. Relancer l'ingestion — les tracks déjà traités pour clap sont ignorés
python manage.py ingest
# 3. L'index est reconstruit automatiquement
```

### Après un check --purge

```bash
python manage.py check --purge --yes
python manage.py build-index   # Obligatoire — l'index a été supprimé
```

### Tester la reconnaissance

```bash
python manage.py download-audio "Daft Punk Get Lucky" --duration 30 --position middle
python manage.py find-track --audio "data/raw/Daft Punk - Get Lucky...middle_30s.mp3" --target <track_id>
```

---

## Identifier un morceau (CLI)

```bash
# Identification simple (méthode par défaut)
python manage.py identify data/raw/mon_audio.mp3

# Méthode spécifique
python manage.py identify data/raw/mon_audio.mp3 --method clap
python manage.py identify data/raw/mon_audio.mp3 --method mfcc

# Top 10 résultats
python manage.py identify data/raw/mon_audio.mp3 --top 10

# Scores détaillés (FP + FAISS séparément)
python manage.py identify data/raw/mon_audio.mp3 --detailed
```

Le score FP (fingerprint) du rang 1 doit être significativement plus élevé que le rang 2 pour un résultat certain. Si tous les scores FP sont à 0 (audio très dégradé), le classement retombe sur les scores FAISS.

### Depuis Python

```python
from src.retrieval.query_pipeline import identify_track

results = identify_track("data/raw/mon_audio.mp3")
results = identify_track("data/raw/mon_audio.mp3", method="clap")
results = identify_track("data/raw/mon_audio.mp3", top_n=10)

# results = [(track_id, score_final), ...]
for rank, (track_id, score) in enumerate(results, 1):
    print(f"{rank}. {track_id} — {score:.4f}")
```

---

## Interface Web

### Routes API

| Méthode | Route | Description |
|---------|-------|-------------|
| `POST` | `/api/identify` | Identification d'un fichier audio → résultats + recommandations |
| `GET`  | `/api/config`   | Paramètres UI (`listen_duration`, `confidence_ratio`, `embedding_method`) |
| `GET`  | `/api/health`   | Liveness check |

### Fonctionnalités

- **Enregistrement micro** — durée configurable via `UI_LISTEN_DURATION` dans `config.py`
- **Dépôt de fichier** — glisser-déposer ou sélection (MP3, WAV, FLAC, WebM…)
- **Résultat** — pochette d'album, titre, artiste, liens de streaming (YouTube, Spotify, Deezer, Apple Music)
- **Recommandations** — 4 morceaux du même genre avec modal de détail
- **Scores de similarité** — bouton `</>` dans le header pour afficher le top 10 des candidats avec leurs scores
- **Thème sombre / clair**
- **Bilingue** — français / anglais

### Architecture frontend

```
webapp/
├── backend/
│   └── server.py          # FastAPI — routes /api/identify, /api/config, /api/health
└── frontend/
    ├── src/
    │   ├── App.jsx                    # État global, routing entre vues
    │   ├── components/
    │   │   ├── Header.jsx             # Logo, toggle thème/langue, bouton debug
    │   │   ├── ListenButton.jsx       # Bouton micro animé (idle / recording / analyzing)
    │   │   ├── DropZone.jsx           # Zone de dépôt de fichier
    │   │   ├── ResultView.jsx         # Vue résultat — layout 2 colonnes
    │   │   ├── AlbumCover.jsx         # Pochette avec placeholder
    │   │   ├── StreamingLinks.jsx     # Boutons plateformes de streaming
    │   │   ├── Recommendations.jsx    # Grille de recommandations
    │   │   ├── RecModal.jsx           # Modal détail d'une recommandation
    │   │   ├── StatusPhrase.jsx       # Phrase d'état animée
    │   │   ├── LightWaves.jsx         # Vagues SVG animées (fond accueil)
    │   │   ├── BgWaves.jsx            # Anneaux de fond
    │   │   └── Footer.jsx             # Lien GitHub
    │   ├── hooks/
    │   │   └── useRecorder.js         # Hook MediaRecorder + countdown
    │   ├── i18n.js                    # Traductions FR/EN
    │   └── index.css                  # Thèmes, animations, layout complet
    └── package.json
```

---

## Installation

### Prérequis système

- **Python 3.10**
- **Node.js 18+** (pour le frontend React) : [nodejs.org](https://nodejs.org/)
- **ffmpeg** : `brew install ffmpeg` (macOS) / `apt install ffmpeg` (Linux) / [ffmpeg.org](https://ffmpeg.org/download.html) (Windows)
- **Clé API Kaggle** : créer un compte sur [kaggle.com](https://kaggle.com), télécharger `kaggle.json` et le placer dans `~/.kaggle/kaggle.json`

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

### Dépendances frontend

Le script `start-webapp` lance `npm install` automatiquement si besoin. Pour l'installer manuellement :

```bash
cd webapp/frontend
npm install
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
| `EMBEDDING_METHOD` | `"clap"` | Méthode active : `"mfcc"`, `"clap"` ou `"muq"` |
| `CLAP_MODEL_NAME` | `"laion/clap-htsat-unfused"` | Modèle CLAP (HuggingFace) |
| `MUQ_MODEL_NAME` | `"OpenMuQ/MuQ-large-msd-iter"` | Modèle MuQ (HuggingFace) |
| `MUQ_BATCH_SIZE` | `8` | Nombre de segments traités en batch par MuQ |
| `CLAP_BATCH_SIZE` | `10` | Nombre de segments traités en batch par CLAP |

### Recherche vectorielle

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `VECTOR_TOP_K_SEGMENTS` | `200` | Segments candidats récupérés depuis FAISS par segment requête |
| `VECTOR_TOP_N_TRACKS` | `20` | Tracks uniques envoyés en Stage 2 (fingerprinting) |
| `VECTOR_TOP_N_RESULTS` | `10` | Résultats finaux retournés à l'interface |

### Optimisations

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `OPT_FLOAT16` | `True` | CLAP/MuQ en demi-précision (réduit la RAM, CUDA uniquement) |
| `OPT_BATCH_EMBED` | `True` | Embedder les segments par batch |
| `OPT_SHORTCIRCUIT` | `True` | Sauter Stage 2 si le 1er candidat FAISS est largement devant |
| `OPT_SHORTCIRCUIT_RATIO` | `10.0` | Ratio score[0]/score[1] pour court-circuit |
| `OPT_FINGERPRINT_PARALLEL` | `True` | Charger les fingerprints candidats en parallèle |

### Fingerprinting

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `FP_SCORE_WEIGHT` | `10.0` | Poids du fingerprint dans le score final |

### Interface web

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `UI_LISTEN_DURATION` | `15` | Durée d'enregistrement micro en secondes |
| `UI_CONFIDENCE_RATIO` | `2.5` | Ratio score[0]/score[1] pour afficher un résultat comme certain |

---

## Méthodes d'embedding

| Méthode | Dim | Sample Rate | Précision | GPU | Temps CPU |
|---------|-----|-------------|-----------|-----|-----------|
| `mfcc` | 40 | 22 050 Hz | Bonne | Non | ~8 s |
| `clap` | 512 | 48 000 Hz | Bonne | CUDA / MPS | ~20 s |
| `muq` | 1024 | 24 000 Hz | Excellente | CUDA uniquement | ~2 min 30 |

> **MuQ sur Mac :** MuQ ne supporte pas Float16 sur CPU ni sur MPS (opérations ComplexFloat non supportées). Sur Apple Silicon sans GPU NVIDIA, utiliser `clap` ou `mfcc`.

---

## Pipeline de reconnaissance

```
Stage 1 — Recherche vectorielle (FAISS)
  ┌─────────────────────────────────────────────────────────┐
  │  Audio requête                                          │
  │    → resample (SR méthode)                             │
  │    → segmenter (fenêtres 5s, hop 3s)                   │
  │    → embed chaque segment (MFCC / CLAP / MuQ)          │
  │    → FAISS → top 200 segments par segment requête      │
  │    → agréger par track → top 20 tracks candidats       │
  └─────────────────────────────────────────────────────────┘
              ↓ (sauf court-circuit si ratio ≥ 10×)
Stage 2 — Re-ranking par fingerprinting
  ┌─────────────────────────────────────────────────────────┐
  │  Fingerprint requête (22 050 Hz fixe)                   │
  │    ↔ fingerprints des 20 candidats (SQLite)             │
  │    → score_final = score_faiss × (1 + score_fp × 10)   │
  │    → tri final → top 10 résultats                       │
  └─────────────────────────────────────────────────────────┘
```

---

## Architecture du projet

```
Projet/
├── manage.py                   # ← Point d'entrée unique (13 commandes)
├── src/
│   ├── config.py               # ← Tous les paramètres ici
│   ├── audio/
│   │   ├── loading.py          # Chargement audio (librosa)
│   │   └── preprocessing.py   # Segmentation + prétraitement
│   ├── features/
│   │   ├── embeddings_audio.py # MFCC, CLAP, MuQ
│   │   └── fingerprint.py     # Constellation map (Shazam)
│   ├── index/
│   │   └── build_index.py     # Construction index FAISS depuis ChromaDB
│   ├── retrieval/
│   │   ├── searcher.py        # Recherche dans FAISS
│   │   └── query_pipeline.py  # Pipeline complet (Stage 1 + Stage 2)
│   ├── ingestion/
│   │   └── ingest.py          # Logique d'ingestion CSV → DB
│   ├── maintenance/
│   │   ├── check.py           # Vérification intégrité + purge
│   │   ├── enrich.py          # Enrichissement métadonnées
│   │   ├── clean.py           # Suppression d'un track
│   │   └── delete_rir.py      # Suppression segments RIR
│   ├── evaluation/
│   │   ├── evaluate.py        # Métriques Top-1 / Top-5
│   │   ├── find_track.py      # Test reconnaissance d'un fichier
│   │   ├── benchmark.py       # Benchmark robustesse
│   │   └── rir_impact.py      # Impact RIR augmentation
│   ├── utils/
│   │   ├── metadata.py        # atomic_write_parquet, helpers
│   │   └── fingerprints_db.py # fp_load_stats, fp_delete, helpers SQLite
│   └── api/
│       └── app.py             # CLI Click (identify)
├── webapp/
│   ├── backend/
│   │   └── server.py          # FastAPI
│   └── frontend/              # React 18 + Vite
├── data/                      # Données persistantes (git-ignorées)
│   ├── chroma/                # ChromaDB (embeddings)
│   ├── features/fingerprints.db # SQLite (fingerprints)
│   ├── index/                 # FAISS (.faiss + segments .parquet)
│   └── processed/metadata.parquet
└── requirements.txt
```

---

## Stockage des données

### ChromaDB (`data/chroma/`)

Base de données vectorielle persistante. Stocke les embeddings de chaque segment audio avec leurs métadonnées (`track_id`, `start_s`). **Une collection par méthode** (`mfcc`, `clap`, `muq`).

- ID de chaque segment : `{track_id}_{i}`
- Permet de supprimer / réécrire proprement les segments d'un track sans décalage d'indices

### FAISS (`data/index/`)

Index vectoriel pour la recherche par similarité (Stage 1). Reconstruit depuis ChromaDB via `build-index`.

- `index_{method}_{type}.faiss` — l'index de recherche
- `segments_{method}.parquet` — mapping indice FAISS → `track_id`

### SQLite (`data/features/fingerprints.db`)

Empreintes audio pour le re-ranking (Stage 2). Une ligne par track, mise à jour atomiquement via `INSERT OR REPLACE`.

### Metadata (`data/processed/metadata.parquet`)

| Colonne | Description |
|---------|-------------|
| `track_id` | Identifiant MD5 unique |
| `title`, `artist` | Métadonnées de base |
| `duration_s` | Durée en secondes (mesurée à 22 050 Hz) |
| `album`, `genre`, `release_date`, `cover_url` | Enrichies via Deezer/MusicBrainz |
| `embedded_methods` | Liste des méthodes déjà calculées |

---

## Utilisation du GPU

| Méthode | GPU utilisé | Détail |
|---------|-------------|--------|
| MFCC | Non | 100 % CPU (numpy/librosa) |
| CLAP | Oui si disponible | CUDA → MPS (Apple Silicon) → CPU |
| MuQ  | Oui si disponible | CUDA → CPU (MPS exclu) |

> Le float16 (`OPT_FLOAT16`) n'est activé que sur CUDA. Sur CPU et MPS, float32 est utilisé.

---

## Dépannage

### `Collection 'clap' introuvable dans ChromaDB`
```bash
python manage.py ingest
```

### `FAISS index manquant`
L'index n'a pas encore été construit ou a été supprimé après une purge.
```bash
python manage.py build-index
```

### `npm: command not found`
Node.js n'est pas installé. Télécharger sur [nodejs.org](https://nodejs.org/) (LTS recommandé).

### `NotImplementedError: MPS device` (Mac Apple Silicon)
Normalement géré automatiquement. Si le problème persiste :
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python manage.py start-webapp
```

### Le processus freeze et ne répond plus (même à Ctrl+C)
Deadlock librosa/numpy. Tuer le processus depuis un autre terminal :
```bash
ps aux | grep manage.py   # trouver le PID
kill -9 <PID>
```
Les données déjà sauvegardées sont conservées. Relancer normalement — les tracks déjà traités seront ignorés.

### Manque de RAM
MuQ nécessite ~2 Go libres, CLAP ~1.5 Go. Fermer les autres applications et relancer.

---

## Git — Récupérer les modifications de l'équipe

```bash
git stash          # Mettre de côté tes modifications locales
git pull           # Récupérer les dernières modifications
git stash pop      # Réappliquer tes modifications
```
