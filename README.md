# Projet-Big_Data

## Idée

Faire un shazam maison. Le but du projet est de créer un shazam, où a partir d'un fichier audio, on doit retrouver la musique la plus proche grace à une base de donnée vectorielle de musiques.

## Notes pratiques

1. Le fingerprinting audio est une technique qui permet d'identifier un son ou un morceau de musique à partir d'un extrait de mauvaise qualité. L'idée est que au lieu de comparer tout le fichier audio, on extrait une signature unique (comme une empreinte d'où le nom du principe).
2. MFCC = Mel-Frequency Cepstral Coefficients. Les MFCC décrivent le timbre du son, la forme globale du spectre, pas les notes exactes mais la texture sonore. Ici c'est une fonction d'embedding.
3. Pipeline d'une requette :
   1. On vectorise le fichier audio donné.
   2. On regarde dans la VDB.
   3. On prend les N plus proches voisins.
   4. On utilise la technique de fingerprinting sur ces N voisins.
   5. On donne les infos de la musique trouvée par cette recherche.

## À faire

1. trois articles de recherche à trouver et à lire par personne.
2. Faire le projet.
3. Faire un rapport.

## Articles de recherche

### Thomas

1. [An Industrial-Strength Audio Search Algorithm](research_paper/thomas/Wang03-shazam.pdf)

    Cet article présente un algorithme de reconnaissance audio, développé par Avery Li-Chun Wang (Shazam).

    L’objectif est d’identifier un morceau de musique à partir d’un court extrait sonore (quelques secondes), même lorsque celui-ci est fortement dégradé : bruit ambiant, voix superposées, distorsions, compression GSM ou coupures réseau.

    L’article propose une méthode de fingerprinting audio basée sur l’extraction de pics fréquentiels, appelés *constellations*. Ces points caractéristiques sont ensuite combinés par paires pour former des hashs temporels discriminants. L’algorithme identifie les morceaux en détectant des alignements temporels cohérents entre l’extrait audio et les pistes de la base de données.

2. [Cross modal audio search and retrieval with joint embeddings based on text and audio](research_paper/thomas/Audio%20search%20and%20retrieval%20-%20Microsoft.pdf)

    Cet article traite de la recherche et de la récupération audio multimodales, en combinant texte et audio.

    Les chercheurs s’attaquent ici à une des limites des moteurs de recherche audio : soit les moteurs comparent texte-texte (via des métadonnées), soit audio-audio (par similarité acoustique), sans interaction entre les deux.

    Les auteurs proposent un embedding conjoint audio-texte à l’aide d’un réseau de neurones siamois (Siamese Neural Network). Ce réseau projette des caractéristiques audio et textuelles dans un même espace, où la similarité sémantique et acoustique peut être mesurée directement.

3. [A fast audio similarity retrieval method for millions of music tracks](research_paper/thomas/A%20fast%20audio%20similarity%20retrieval%20method.pdf)

    Cet article parle du problème de la recherche rapide de similarité audio dans des bases de données musicales de très grande taille (plusieurs millions de morceaux).

    Le problème est que les méthodes de similarité audio les plus performantes reposent sur des modèles complexes (par exemple les modèles de timbre gaussiens), qui sont très coûteux à calculer et donc difficiles à appliquer sur de gros datasets.

    Les auteurs proposent une méthode filter-and-refine qui porte sur une adaptation de l’algorithme FastMap. L’idée est de projeter les morceaux de musique dans un espace vectoriel de plus faible dimension, afin d’effectuer une pré-sélection rapide des candidats les plus proches grâce à une distance euclidienne, puis d'améliorer les résultats avec la mesure exacte (divergence de Kullback–Leibler symétrisée).

### Clara

1. [Music2Latent2: Audio Compression with Summary Embeddings and Autoregressive Decoding](research_paper/clara/Music2Latent2_audio_embeding.pdf)

    Cet article présente Music2Latent2, un autoencodeur audio développé par Marco Pasini, Stefan Lattner et György Fazekas (Queen Mary University of London et Sony CSL Paris).

    L'objectif est de compresser efficacement des signaux audio dans un espace latent compact tout en préservant la qualité de reconstruction et l'utilité pour des tâches de music information retrieval (MIR), même à des ratios de compression élevés (64× à 128×).

    L'article propose une méthode basée sur des summary embeddings (embeddings non ordonnés), où chaque embedding capture des caractéristiques globales distinctes d'un segment audio (timbre, tempo), contrairement aux approches classiques qui répètent redondamment ces informations.

    Le décodage autorégressive réintroduit du bruit dans le chunk précédent pour éviter l'accumulation d'erreurs. Music2Latent2 surpasse les autoencodeurs existants sur les métriques de qualité audio (FAD/FADclap) et obtient des résultats supérieurs sur des tâches MIR comme l'autotagging ou la classification d'instruments.

2. [PDX: A Data Layout for Vector Similarity Search](research_paper/clara/PDX_vector_similarity.pdf)

    Cet article présente PDX (Partition Dimensions Across), un format de stockage pour les vecteurs (embeddings) développé par Leonardo Kuffo, Elena Krippner et Peter Boncz (CWI Amsterdam).

    L'objectif est d'accélérer la recherche de similarité vectorielle (Vector Similarity Search, VSS) en permettant un calcul de distance dimension par dimension, contrairement au format horizontal standard qui stocke les vecteurs bout-à-bout et nécessite d'accéder à toutes les dimensions même quand certaines ne sont jamais utilisées.

    L'article propose un format qui stocke les vecteurs par blocs verticaux (toutes les valeurs d'une même dimension ensemble dans un bloc), ce qui permet de traiter plusieurs vecteurs simultanément.

    - PDX-BOND est une stratégie de pruning flexible sans prétraitement qui fonctionne sur les vecteurs bruts et peut effectuer une recherche exacte. Les expériences montrent que PDX surpasse les systèmes vectoriels existants (FAISS, Milvus, USearch) de 2 à 7× en recherche exacte et approximative.

    - PDXearch, un framework qui adapte dynamiquement le nombre de dimensions explorées selon la requête, et avec des algorithmes de dimension-pruning qui approximent la distance en n'évaluant qu'un sous-ensemble de dimensions, permettant d'éliminer rapidement les vecteurs non-pertinents.

    PDX peut optimiser le calcul de distance lors de la recherche de l'extrait audio dans une base d'embeddings. PDX-BOND pourrait prioriser les dimensions les plus discriminantes pour éliminer rapidement les morceaux non-correspondants sans parcourir toutes les dimensions.

3. [MuQ: Self-Supervised Music Representation Learning with Mel Residual Vector Quantization](MuQ_music_vector.pdf)

    Cet article présente MuQ, un modèle d'apprentissage auto-supervisé pour la représentation musicale, développé par Haina Zhu, Yizhi Zhou et leurs collaborateurs (Shanghai Jiao Tong University, Nanjing University, Tencent AI Lab).

    L'objectif est de créer des représentations audio universelles capables de capturer simultanément les informations sémantiques (genre, émotion) et acoustiques (mélodie, tonalité, timbre) de la musique, pour améliorer les performances sur les tâches de music information retrieval (MIR) comme le tagging, la classification d'instruments ou la détection de tonalité.

    L'article propose Mel-RVQ (Mel Residual Vector Quantization), un tokenizer léger qui quantifie directement le spectrogramme Mel via une projection linéaire résiduelle. Le modèle MuQ utilise une architecture Conformer (12 couches, 310M paramètres) entraînée par masked language modeling à prédire ces tokens multi-résiduels. Un entraînement itératif raffine les représentations en réentraînant Mel-RVQ sur les latents de MuQ.

## Premiers pas

### Comment démarrer le projet

1. Importer le projet sur sa machine.
2. Créer un venv :

    ```bash
        python3 -m venv venv
        # Activation du venv
        source venv/bin/activate
        # Vérification de l'activation du venv
        which python
    ```

3. Installer les librairies necessaires pour l'éxécution du projet :

    ```bash
        pip install -r requirements.txt
    ```

### Ajout d'une librairie

1. On utilise pip pour installer la librairie qu'on souhaite (Il faut bien vérifier qu'on se situe dans le venv).
2. Puis on fait la commande :

    ```bash
        pip freeze > requirements.txt
        # Cette commande permet de mettre à jour les librairies nécessaires pour éxecuter le code du projet.
    ```

### J'ai des modifications en local et je suis sur une ancienne version du projet

```bash
    git stash
    git pull origin main
    git stash pop
```

## Structure du projet

```bash
shazam/
├── README.md                  # Présentation du projet, comment lancer
├── requirements.txt           # Dépendances Python
├── .gitignore

├── data/                      # Les données du projet
│   ├── raw/                   # Audios bruts (les datasets téléchargés)
│   ├── processed/             # Audios nettoyés / segments
│   ├── features/              # Descripteurs / empreintes / embeddings
│   └── index/                 # Fichiers de la base vectorielle

├── src/
│   ├── __init__.py
│   ├── config.py              # Chemins de fichiers, hyperparamètres simples

│   ├── data_utils/            # Gestion des données
│   │   ├── __init__.py

│   ├── audio/                 # Tout ce qui touche au signal audio
│   │   ├── __init__.py
│   │   ├── loading.py         # Chargement audio, resampling, mono, etc.
│   │   └── preprocessing.py   # Normalisation, découpe en fenêtres, etc.

│   ├── features/              # Représentations numériques de l’audio
│   │   ├── __init__.py

│   ├── index/                 # Base de données vectorielle
│   │   ├── __init__.py
│   │   └── build_index.py     # Script pour construire et mettre à jour l’index

│   ├── retrieval/             # Pipeline pour répondre à une requête
│   │   ├── __init__.py

│   └── api/                   # Interface simple pour tester le Shazam
│       ├── __init__.py
│       └── app.py             # Petitte API pour envoyer un audio et recevoir le résultat

├── scripts/                   # Scripts à lancer depuis le terminal

└── tests/                     # Pour les tests
    ├── __init__.py

```

## Lien intéressant

1. Kaggle: <https://www.kaggle.com>
2. Free Music Archive: <https://freemusicarchive.org>
3. GTZAN Genre Collection: <https://www.kaggle.com/datasets/carlthome/gtzan-genre-collection>
4. MusicGenres Dataset (HuggingFace): <https://huggingface.co/datasets/ccmusic-database/music_genre>
5. MusicCaps / AudioSet: <https://huggingface.co/datasets/google/MusicCaps>
6. Recherche: <https://dblp.org> & <https://dblp.uni-trier.de/search/publ>
7. Apache spark: <https://spark.apache.org>
