# Référence des commandes — Shazam Maison

Toutes les opérations passent par `python manage.py <commande> [options]`.

> **Pré-requis :** activer l'environnement virtuel avant toute commande.
> ```bash
> source venv/bin/activate
> ```

---

## Table des matières

| Groupe | Commandes |
|--------|-----------|
| [Ingestion](#ingestion) | `ingest` · `augment` · `rebuild-fingerprints` · `build-index` |
| [Maintenance](#maintenance) | `check` · `enrich` · `clean` · `delete-rir` |
| [Identification](#identification) | `identify` · `find-track` · `download-audio` |
| [Évaluation](#évaluation) | `evaluate` · `rir-evaluate` · `rir-impact` · `benchmark` · `plots` |
| [Interface web](#interface-web) | `start-webapp` |

---

## Ingestion

### `ingest` — Alimenter la base

Télécharge l'audio en RAM via yt-dlp, calcule embeddings + fingerprints, stocke dans ChromaDB + SQLite, puis reconstruit l'index FAISS. Aucun MP3 n'est écrit sur disque.

```bash
python manage.py ingest [OPTIONS]
```

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--csv PATH` | Path (répétable) | tous les CSV dans `data/kaggle/` | Fichier CSV ou dossier Kaggle Spotify |
| `--method` | `mfcc`/`clap`/`muq`/`mert` | `config.EMBEDDING_METHOD` | Méthode d'embedding |

**Exemples :**

```bash
# Ingestion depuis un seul CSV
python manage.py ingest --csv data/kaggle/data/spotify-streaming-top-50-world.csv

# Plusieurs CSV en une commande
python manage.py ingest \
  --csv data/kaggle/data/spotify-streaming-top-50-france.csv \
  --csv data/kaggle/data/spotify-streaming-top-50-usa.csv

# Tous les CSV d'un dossier
python manage.py ingest --csv data/kaggle/data/

# Sans argument : tous les CSV Kaggle disponibles
python manage.py ingest

# Forcer une méthode
python manage.py ingest --csv data/kaggle/data/ --method mfcc
```

**Notes :**
- Reprise automatique : les tracks déjà traités pour la méthode active sont ignorés (champ `embedded_methods` dans `metadata.parquet`).
- Un crash n'est pas problématique — relancer la commande reprend exactement là où ça s'était arrêté.
- Les tracks présents dans plusieurs CSV (ex : hits mondiaux dans france + monde + usa) sont dédupliqués par `track_id`, pas par source CSV.

---

### `augment` — Augmenter les embeddings avec des RIR

Applique des Room Impulse Responses (réverbérations de salles) aux tracks existants pour enrichir la base vectorielle. Améliore la robustesse du Stage 1 (FAISS) face aux requêtes captées dans des environnements réverbérants.

```bash
python manage.py augment [OPTIONS]
```

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--method` | `mfcc`/`clap`/`muq`/`mert` | `config.EMBEDDING_METHOD` | Méthode d'embedding |
| `--tracks TEXT` | répétable | tous | `track_id(s)` spécifiques (sinon tous) |
| `--rir-source` | `synthetic`/`mit` | `config.RIR_SOURCE` | Source des RIRs |
| `--n-rir INTEGER` | int | `config.RIR_N` | Nombre de RIRs par track |
| `--rir-dir PATH` | Path | `config.RIR_MIT_DIR` | Dossier des WAV MIT (si `--rir-source mit`) |
| `--workers INTEGER` | int | 3 | Workers parallèles pour le téléchargement |
| `--device` | `cpu`/`cuda`/`mps` | auto | Device PyTorch |
| `--no-rebuild-index` | flag | False | Ne pas reconstruire l'index FAISS après |

**Exemples :**

```bash
# Augmentation avec RIRs synthétiques (défaut config.py)
python manage.py augment

# Choisir explicitement la source et le nombre de RIRs
python manage.py augment --rir-source synthetic --n-rir 5
python manage.py augment --rir-source mit --n-rir 7

# Augmenter un seul track
python manage.py augment --tracks f01ab00f1fdc5a57fd2676f4d68631a8

# Sans reconstruire l'index (si on enchaine plusieurs augmentations)
python manage.py augment --no-rebuild-index
python manage.py build-index   # reconstruire manuellement après
```

**Configuration dans `src/config.py` :**
```python
RIR_SOURCE  = "synthetic"   # "synthetic" | "mit"
RIR_N       = 5             # nombre de RIRs par track
RIR_MIT_DIR = "data/rir"    # dossier WAV MIT
```

**Source `synthetic` :** génère 10 RIRs mathématiques couvrant bathroom (RT60=0.15s) → warehouse (0.90s) → concert_hall (1.60s). Aucun téléchargement nécessaire.

**Source `mit` :** charge les fichiers WAV dans `data/rir/`, calcule le RT60 de chacun, sélectionne les N les plus diversifiés par échantillonnage uniforme sur la courbe RT60 triée. Nécessite d'avoir téléchargé les WAV MIT au préalable.

**Idempotent :** les RIRs déjà appliquées à un track sont ignorées. Seules les RIRs manquantes sont calculées.

---

### `rebuild-fingerprints` — Recalculer les fingerprints

Recalcule les fingerprints Shazam (constellation spectrale) de tous les tracks sans re-télécharger l'audio. Utile si `fingerprints.db` a été corrompu ou après une purge ciblée.

```bash
python manage.py rebuild-fingerprints [OPTIONS]
```

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--force` | flag | False | Recalcule tous les tracks, même ceux déjà dans SQLite |
| `--limit INTEGER` | int | tous | Limiter à N tracks (utile pour tester) |

**Exemples :**

```bash
# Recalcule uniquement les fingerprints manquants
python manage.py rebuild-fingerprints

# Tout recalculer (ex : après un changement de paramètres d'extraction)
python manage.py rebuild-fingerprints --force

# Tester sur 10 tracks
python manage.py rebuild-fingerprints --limit 10
```

---

### `build-index` — Construire l'index FAISS

Reconstruit l'index FAISS (et le fichier Parquet de mapping segments) depuis les embeddings ChromaDB. Appelé automatiquement par `ingest` et `augment`, mais à relancer manuellement après un `check --purge`.

```bash
python manage.py build-index [OPTIONS]
```

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--method` | `mfcc`/`clap`/`muq`/`mert` | toutes les méthodes présentes | Méthode à indexer |
| `--index-type` | `flat`/`hnsw`/`ivf` | `config.INDEX_TYPE` | Type d'index FAISS |

**Exemples :**

```bash
# Reconstruire tous les index
python manage.py build-index

# Une méthode spécifique
python manage.py build-index --method clap

# Type d'index différent (HNSW = plus rapide, légèrement moins précis)
python manage.py build-index --index-type hnsw

# Flat + méthode spécifique
python manage.py build-index --method mfcc --index-type flat
```

**Types d'index :**
| Type | Précision | Vitesse | RAM |
|------|-----------|---------|-----|
| `flat` | Exacte (bruteforce) | Lente | Faible |
| `hnsw` | Approchée (~99%) | Rapide | Modérée |
| `ivf` | Approchée | Rapide | Faible |

---

## Maintenance

### `check` — Vérifier l'intégrité des données

Vérifie la cohérence de ChromaDB, FAISS, SQLite et `metadata.parquet`. Peut supprimer chirurgicalement les tracks problématiques.

```bash
python manage.py check [OPTIONS]
```

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--method` | `mfcc`/`clap`/`muq`/`mert` | toutes | Filtrer sur une méthode |
| `--details` | flag | False | Afficher le détail des warnings par catégorie |
| `--metadata` | flag | False | Lister les tracks avec métadonnées manquantes |
| `--purge` | flag | False | Supprimer les tracks problématiques |
| `--purge-missing-fp` | flag | False | Supprimer uniquement les tracks sans fingerprint |
| `--yes` | flag | False | Ne pas demander confirmation avant la purge |

**Exemples :**

```bash
# Vue résumé (nombre de tracks OK / KO par catégorie)
python manage.py check

# Filtrer sur une méthode
python manage.py check --method clap

# Afficher le détail des warnings
python manage.py check --details

# Tracks avec métadonnées incomplètes (album, genre, cover_url…)
python manage.py check --metadata

# Purge interactive (demande confirmation)
python manage.py check --purge

# Purge silencieuse (CI/CD, scripts)
python manage.py check --purge --yes

# Purger uniquement les tracks sans fingerprint
python manage.py check --purge-missing-fp
```

**Codes de vérification (`--details`) :**

| Code | Niveau | Description |
|------|--------|-------------|
| `C1` | Critique | Dimension d'embedding inattendue |
| `C2` | Critique | NaN ou Inf dans les embeddings |
| `C3` | Critique | Désynchronisation ChromaDB ↔ metadata parquet |
| `C5` | Critique | Index FAISS désynchronisé avec ChromaDB |
| `C6` | Critique | Segments orphelins (ChromaDB sans metadata) |
| `C6b` | Critique | Track marqué traité mais sans segments dans ChromaDB |
| `C7` | Critique | Embedding incomplet (< 80 % des segments attendus) |
| `Q1` | Qualité | Durée aberrante (≤ 0s ou > 10min) |
| `Q2` | Qualité | Segment `start_s` dépassant la durée déclarée de plus de 5s |
| `Q3` | Qualité | Fingerprint vide (0 hash) |
| `Q4` | Qualité | Fingerprint anormalement pauvre (outlier IQR) |
| `FP` | Qualité | Track sans fingerprint — Stage 2 inopérant pour ce track |

**Ce que fait `--purge` (par méthode, pas par track entier) :**
1. Supprime les segments de la méthode dans ChromaDB
2. Retire la méthode de `embedded_methods` dans `metadata.parquet`
3. Si `embedded_methods` devient vide → supprime la ligne + le fingerprint SQLite
4. Supprime l'index FAISS de la méthode → **relancer `build-index` après**

---

### `enrich` — Enrichir les métadonnées

Complète `metadata.parquet` avec `album`, `genre`, `release_date`, `cover_url` via Deezer (puis MusicBrainz en fallback). Ne touche pas aux embeddings ni aux fingerprints.

```bash
python manage.py enrich [OPTIONS]
```

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--force` | flag | False | Ré-enrichir tous les tracks, même ceux déjà complets |
| `--only-missing` | flag | False | Traiter uniquement les tracks avec au moins un champ vide |

**Exemples :**

```bash
# Enrichir les tracks incomplets (défaut)
python manage.py enrich

# Forcer la mise à jour de tous les tracks
python manage.py enrich --force

# Traiter uniquement les tracks sans aucune métadonnée
python manage.py enrich --only-missing
```

---

### `clean` — Supprimer un track

Supprime proprement un track de **tous** les stores (ChromaDB, SQLite, `metadata.parquet`) et invalide l'index FAISS.

```bash
python manage.py clean [OPTIONS]
```

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--track-id TEXT` | string | — | ID exact du track |
| `--name TEXT` | string | — | Recherche floue par artiste ou titre |
| `--yes` | flag | False | Ne pas demander confirmation |

**Exemples :**

```bash
# Par track_id exact
python manage.py clean --track-id f01ab00f1fdc5a57fd2676f4d68631a8

# Par nom (artiste ou titre)
python manage.py clean --name "Miley Cyrus"
python manage.py clean --name "Flowers"

# Sans confirmation
python manage.py clean --name "Flowers" --yes
```

> Après un `clean`, relancer `python manage.py build-index` pour mettre à jour l'index FAISS.

---

### `delete-rir` — Supprimer les segments RIR

Supprime les segments augmentés par Room Impulse Response d'un ou de tous les tracks dans ChromaDB. Ne touche pas aux embeddings originaux ni aux fingerprints.

```bash
python manage.py delete-rir [OPTIONS]
```

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--track-id TEXT` | string | tous | Track spécifique |
| `--method` | `mfcc`/`clap`/`muq`/`mert` | `config.EMBEDDING_METHOD` | Méthode concernée |
| `--dry-run` | flag | False | Afficher ce qui serait supprimé sans supprimer |
| `--yes` | flag | False | Ne pas demander confirmation |

**Exemples :**

```bash
# Aperçu sans suppression
python manage.py delete-rir --dry-run

# Supprimer les RIR de tous les tracks (méthode active)
python manage.py delete-rir --yes

# Supprimer les RIR d'un seul track
python manage.py delete-rir --track-id f01ab00f1fdc5a57fd2676f4d68631a8

# Méthode spécifique
python manage.py delete-rir --method mfcc --yes
```

---

## Identification

### `identify` — Identifier un morceau

Identifie un fichier audio et retourne les morceaux les plus probables. C'est la commande principale — l'équivalent CLI de l'interface web.

**Pipeline :** Stage 1 (FAISS → top 20 candidats) → Stage 2 (re-ranking par fingerprinting Shazam). Classement final : score fingerprint en priorité, score FAISS en départage.

```bash
python manage.py identify AUDIO [OPTIONS]
```

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `AUDIO` | Path | — | Fichier audio à identifier (MP3, WAV, FLAC, OGG…) |
| `--method` | `mfcc`/`clap`/`muq`/`mert` | `config.EMBEDDING_METHOD` | Méthode d'embedding |
| `--top INTEGER` | int | 5 | Nombre de résultats à afficher |
| `--detailed` | flag | False | Afficher les scores FAISS et fingerprint séparément |

**Exemples :**

```bash
# Identification simple
python manage.py identify data/raw/mon_audio.mp3

# Top 10 résultats
python manage.py identify data/raw/mon_audio.mp3 --top 10

# Scores détaillés (FP + FAISS séparément)
python manage.py identify data/raw/mon_audio.mp3 --detailed

# Forcer une méthode
python manage.py identify data/raw/mon_audio.mp3 --method mfcc

# Combinaison
python manage.py identify data/raw/mon_audio.mp3 --method clap --top 10 --detailed
```

**Interprétation des scores :**
- **Score FP** (fingerprint) : cohérence temporelle entre les hashes de la requête et ceux de la base. Score > 0 indique un alignement temporel trouvé — plus c'est élevé, plus la correspondance est certaine.
- **Score FAISS** : similarité cosinus dans l'espace d'embeddings. Utilisé en tiebreaker quand tous les scores FP sont à 0 (audio très dégradé).
- Si le rang 1 a un score FP significativement supérieur au rang 2 → résultat fiable.

---

### `find-track` — Tester la reconnaissance (évaluation)

Teste la reconnaissance d'un fichier audio et indique si le bon track est trouvé, avec son rang et ses scores. Contrairement à `identify`, la réponse attendue est connue.

```bash
python manage.py find-track [OPTIONS]
```

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--audio PATH` | Path | `data/raw/flowers_middle_30s.mp3` | Fichier audio à tester |
| `--target TEXT` | string | ID de Flowers (Miley Cyrus) | `track_id` attendu en réponse |
| `--top INTEGER` | int | 20 | Nombre de candidats à afficher |
| `--method` | `mfcc`/`clap`/`muq`/`mert` | `config.EMBEDDING_METHOD` | Méthode d'embedding |

**Exemples :**

```bash
# Test par défaut (Flowers, valeurs par défaut)
python manage.py find-track

# Test sur un fichier et une cible spécifiques
python manage.py find-track \
  --audio  data/raw/mon_audio.mp3 \
  --target f01ab00f1fdc5a57fd2676f4d68631a8

# Avec une méthode et un top étendu
python manage.py find-track \
  --audio  data/raw/mon_audio.mp3 \
  --target f01ab00f1fdc5a57fd2676f4d68631a8 \
  --method clap \
  --top 10
```

---

### `download-audio` — Télécharger un audio de test

Télécharge un morceau depuis YouTube dans `data/raw/` pour tester la reconnaissance. Contrairement à `ingest`, **le fichier MP3 est stocké sur disque**.

```bash
python manage.py download-audio QUERY [OPTIONS]
```

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `QUERY` | string | — | Requête YouTube (ex : `"Miley Cyrus Flowers"`) |
| `--duration` | `5`/`10`/`15`/`30` | morceau entier | Durée de l'extrait en secondes |
| `--position` | voir ci-dessous | `middle` | Position dans le morceau |

**Positions disponibles :** `start` · `first-quarter` · `middle` · `third-quarter` · `end`

**Exemples :**

```bash
# Morceau entier
python manage.py download-audio "Miley Cyrus Flowers"

# Extrait 30s au milieu (recommandé pour les tests)
python manage.py download-audio "Miley Cyrus Flowers" --duration 30 --position middle

# Extrait court (cas difficile)
python manage.py download-audio "Daft Punk Get Lucky" --duration 5 --position middle

# Différentes positions
python manage.py download-audio "The Weeknd Blinding Lights" --duration 15 --position first-quarter
python manage.py download-audio "The Weeknd Blinding Lights" --duration 15 --position end
```

Le fichier est nommé automatiquement : `Artiste - Titre__position_Xs.mp3`
Ex : `Miley Cyrus - Flowers (Official Video)__middle_30s.mp3`

> **Conseil :** `--position middle` donne les meilleurs résultats — le refrain est acoustiquement plus distinctif que l'intro ou l'outro.

---

## Évaluation

### `evaluate` — Évaluation multi-tracks

Évalue le pipeline complet sur un ensemble de fichiers de test avec plusieurs conditions de dégradation. Calcule Top-1, Top-5, MRR et latence par méthode × condition. Produit des JSON et des graphiques.

```bash
python manage.py evaluate [OPTIONS]
```

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--methods` | répétable | `mfcc`, `clap` | Méthodes à évaluer |
| `--conditions` | répétable | toutes (5) | Conditions de dégradation |
| `--n-tracks INTEGER` | int | 0 (tous) | Limiter à N tracks du manifest |
| `--no-plot` | flag | False | Ne pas générer les graphiques |

**Conditions disponibles :**

| Condition | Description |
|-----------|-------------|
| `clean` | Audio sans dégradation |
| `snr_20` | Bruit blanc à 20 dB SNR |
| `snr_10` | Bruit blanc à 10 dB SNR |
| `reverb` | Réverbération simulée |
| `combo` | SNR 10 dB + reverb combinés |

**Exemples :**

```bash
# Évaluation complète (mfcc + clap, 5 conditions)
python manage.py evaluate

# Méthodes spécifiques
python manage.py evaluate --methods clap --methods mfcc --methods mert

# Conditions réduites (rapide, pour vérifier)
python manage.py evaluate --conditions clean --conditions snr_20

# 5 tracks seulement
python manage.py evaluate --n-tracks 5

# Sans graphiques
python manage.py evaluate --no-plot
```

**Prérequis :** un manifest de fichiers de test (alimenté par `download-audio`). Chaque track identifiable par `track_id` doit avoir son fichier de test dans `data/raw/`.

**Produit :**
- `results/eval/eval_TIMESTAMP.json` — métriques complètes
- `results/plots/method_accuracy.png` — G6 : accuracy par méthode × condition
- `results/plots/stage_comparison.png` — G9 : Stage 1 vs Stage 2
- `results/plots/duration_impact.png` — G11 : accuracy vs durée d'extrait
- `results/plots/heatmap_accuracy.png` — G12 : heatmap méthodes × conditions

---

### `rir-evaluate` — Comparer Stage 1 avec/sans RIR

Compare la précision Stage 1 (FAISS seul) avec et sans les vecteurs RIR dans l'index. Construit un index temporaire sans RIR en mémoire — **ne modifie pas la base**.

```bash
python manage.py rir-evaluate [OPTIONS]
```

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--methods` | répétable | `config.EMBEDDING_METHOD` | Méthodes à évaluer |
| `--conditions` | répétable | toutes (5) | Conditions de dégradation |
| `--n-tracks INTEGER` | int | 0 (tous) | Limiter à N tracks |
| `--no-plot` | flag | False | Ne pas générer les graphiques |

**Exemples :**

```bash
# Évaluation RIR par défaut
python manage.py rir-evaluate

# CLAP uniquement, conditions spécifiques
python manage.py rir-evaluate --methods clap --conditions clean --conditions reverb

# Plusieurs méthodes, limité à 5 tracks
python manage.py rir-evaluate --methods mfcc --methods clap --n-tracks 5
```

**Produit :**
- `results/eval/rir_eval_TIMESTAMP.json`
- `results/plots/rir_paired_bar_*.png` — G1 : accuracy avec vs sans RIR
- `results/plots/rir_delta_*.png` — G2 : gain Δ apporté par les RIR
- `results/plots/rir_faiss_scores_*.png` — G4 : score FAISS par morceau

---

### `rir-impact` — Analyse RIR sur un fichier

Analyse détaillée de l'impact des segments RIR sur un seul fichier audio — affichage riche en terminal.

```bash
python manage.py rir-impact [OPTIONS]
```

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--audio PATH` | Path | `data/raw/flowers_middle_30s.mp3` | Fichier audio à tester |
| `--target TEXT` | string | ID de Flowers | `track_id` attendu |
| `--top INTEGER` | int | 20 | Nombre de candidats à afficher |
| `--method` | `mfcc`/`clap`/`muq`/`mert` | `config.EMBEDDING_METHOD` | Méthode |

**Exemples :**

```bash
# Analyse par défaut (Flowers)
python manage.py rir-impact

# Fichier et cible spécifiques
python manage.py rir-impact \
  --audio  data/raw/mon_audio.mp3 \
  --target f01ab00f1fdc5a57fd2676f4d68631a8 \
  --method clap
```

---

### `benchmark` — Benchmark de robustesse (morceau unique)

Évalue la robustesse du système sur un morceau de référence (Flowers par défaut) à travers plusieurs conditions de dégradation. Produit un tableau de résultats riche et un JSON.

```bash
python manage.py benchmark [OPTIONS]
```

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--label TEXT` | string | horodatage | Nom du run (pour comparer) |
| `--full` | flag | False | Suite complète (plus longue mais plus exhaustive) |
| `--compare PATH` | Path (répétable) | — | Comparer plusieurs JSON de résultats |

**Exemples :**

```bash
# Benchmark rapide
python manage.py benchmark

# Avec un label pour traçabilité
python manage.py benchmark --label "clap-v2-large"

# Suite complète
python manage.py benchmark --label "clap-baseline" --full

# Comparer deux runs
python manage.py benchmark \
  --compare results/benchmark/clap-v1.json \
  --compare results/benchmark/clap-v2.json
```

**Conditions testées :**

| # | Dégradation | Paramètre |
|---|-------------|-----------|
| 1 | Audio propre | — |
| 2 | Bruit blanc | SNR = 20 dB |
| 3 | Bruit blanc fort | SNR = 10 dB |
| 4 | Réverbération légère | RT60 ≈ 0.4s |
| 5 | Passe-haut 300 Hz | (simule téléphone) |
| 6 | Compression Opus | 64 kbps |
| 7 | Extrait court | 5 s (1 seul segment) |

---

### `plots` — Générer les graphiques du rapport

Lit des JSON produits par `evaluate` et/ou `rir-evaluate` et génère les graphiques PNG pour le rapport.

```bash
python manage.py plots [OPTIONS]
```

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--eval JSON` | Path (répétable) | — | JSON d'évaluation (`results/eval/eval_*.json`) |
| `--rir-eval JSON` | Path (répétable) | — | JSON d'évaluation RIR (`results/eval/rir_eval_*.json`) |
| `--out-dir PATH` | Path | `results/plots/` | Dossier de sortie |

**Exemples :**

```bash
# Graphiques pipeline uniquement (G6, G9, G11, G12)
python manage.py plots --eval results/eval/eval_*.json

# Graphiques RIR uniquement (G1, G2, G4)
python manage.py plots --rir-eval results/eval/rir_eval_*.json

# Tous les graphiques (G1, G2, G4, G6, G9, G11, G12)
python manage.py plots \
  --eval     results/eval/eval_*.json \
  --rir-eval results/eval/rir_eval_*.json

# Sortie dans un dossier personnalisé
python manage.py plots --eval results/eval/eval_*.json --out-dir /tmp/graphs
```

**Graphiques produits :**

| Fichier | Graphique | Source |
|---------|-----------|--------|
| `rir_paired_bar_*.png` | G1 — Accuracy avec vs sans RIR par condition | `--rir-eval` |
| `rir_delta_*.png` | G2 — Gain Δ apporté par les RIR (points de %) | `--rir-eval` |
| `rir_faiss_scores_*.png` | G4 — Score FAISS par morceau avec/sans RIR | `--rir-eval` |
| `method_accuracy.png` | G6 — Accuracy Top-1 par méthode × condition | `--eval` |
| `stage_comparison.png` | G9 — Stage 1 (FAISS) vs Stage 2 (+ fingerprint) | `--eval` |
| `duration_impact.png` | G11 — Accuracy vs durée d'extrait | `--eval` |
| `heatmap_accuracy.png` | G12 — Heatmap méthodes × conditions | `--eval` |

---

## Interface web

### `start-webapp` — Lancer l'interface web

Lance simultanément le backend FastAPI et le frontend React.

```bash
python manage.py start-webapp [OPTIONS]
```

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--prod` | flag | False | Mode production (build statique, port unique) |
| `--port INTEGER` | int | 8000 | Port du backend FastAPI |

**Exemples :**

```bash
# Mode développement (hot-reload Vite)
python manage.py start-webapp

# Mode production
python manage.py start-webapp --prod

# Port personnalisé
python manage.py start-webapp --port 8080
python manage.py start-webapp --prod --port 9000
```

| Mode | Frontend | Backend | URL d'accès |
|------|----------|---------|-------------|
| Dev | Vite hot-reload `:5173` | uvicorn `--reload` `:8000` | http://localhost:5173 |
| Prod | Build statique dans `dist/` | uvicorn `:8000` | http://localhost:8000 |

---

## Workflows typiques

### Démarrage complet (première installation)

```bash
source venv/bin/activate

# 1. Alimenter la base (30-60 min selon le CSV et la méthode)
python manage.py ingest --csv data/kaggle/data/spotify-streaming-top-50-world.csv

# 2. Enrichir les métadonnées (pochettes, genres, dates)
python manage.py enrich

# 3. (Optionnel) Augmentation RIR pour améliorer la robustesse
python manage.py augment

# 4. Lancer l'interface web
python manage.py start-webapp
```

### Ajouter un nouveau CSV

```bash
python manage.py ingest --csv data/kaggle/data/nouveau_chart.csv
# L'index est reconstruit automatiquement à la fin
```

### Changer de méthode d'embedding

```bash
# 1. Modifier EMBEDDING_METHOD dans src/config.py
#    ex : "mfcc" → "clap"

# 2. Relancer l'ingestion (les tracks déjà traités pour clap sont ignorés)
python manage.py ingest

# 3. L'index FAISS est reconstruit automatiquement
```

### Après une purge

```bash
python manage.py check --purge --yes
python manage.py build-index    # obligatoire — l'index a été supprimé
```

### Générer les graphiques pour le rapport

```bash
# 1. Télécharger des clips de test pour les tracks dans la base
python manage.py download-audio "Miley Cyrus Flowers"        --duration 30 --position middle
python manage.py download-audio "Travis Scott PARASAIL"      --duration 30 --position middle
python manage.py download-audio "The Weeknd Blinding Lights" --duration 30 --position middle
# ... répéter pour chaque track

# 2. Évaluation pipeline complet (G6, G9, G11, G12)
python manage.py evaluate --methods mfcc --methods clap

# 3. Évaluation impact RIR (G1, G2, G4)
python manage.py rir-evaluate --methods clap

# 4. Générer tous les graphiques
python manage.py plots \
  --eval     results/eval/eval_*.json \
  --rir-eval results/eval/rir_eval_*.json
```

### Vérification rapide de santé

```bash
python manage.py check             # résumé
python manage.py check --details   # détail des warnings
python manage.py check --metadata  # métadonnées manquantes
```
