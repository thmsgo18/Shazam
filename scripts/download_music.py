"""
scripts/download_music.py

But : lire le CSV Kaggle Spotify, faire le matching YouTube, telecharger les MP3,
      puis lancer automatiquement le pipeline de build et verifier les chiffres.

Etapes :
    1. Lire le CSV Kaggle  -> extraire titre + artiste
    2. yt-dlp              -> recherche YouTube + telechargement MP3 (sans API YouTube)
    3. build_metadata.py   -> construction des metadonnees
    4. build_segment_embeddings.py -> calcul des embeddings
    5. build_index.py      -> construction de l'index FAISS
    6. Verification        -> coherence des chiffres entre chaque etape

Prerequis :
    pip install pandas yt-dlp
    brew install ffmpeg   

Telechargement du CSV Kaggle :
    pip install kaggle
    kaggle datasets download -d anxods/spotify-top-50-playlist-songs-anxods
    unzip spotify-top-50-playlist-songs-anxods.zip -d data/kaggle/

Usage :
    python scripts/download_music.py --csv data/kaggle/mon_fichier.csv --n 50 
    #Remplacez mon_fichier.csv par le top50 de votre choix
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

import click
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RAW_DIR = Path("data/raw")

# Noms de colonnes possibles selon la version du dataset
POSSIBLE_TITLE_COLS  = ["track_name", "name", "title", "song", "track"]
POSSIBLE_ARTIST_COLS = ["artist_name", "artists", "artist", "performer", "track_artist"]


# ===========================================================================
# ETAPE 1 — CSV KAGGLE : lire les metadonnees Spotify
# ===========================================================================

def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Retourne le premier nom de colonne present dans le DataFrame."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def load_tracks_from_csv(csv_path: Path, n: int) -> list[dict]:
    """
    Lit le CSV Kaggle et extrait les n premiers titres uniques.
    Retourne une liste de dicts avec : title, artist.
    """
    print(f"Lecture du CSV : {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"  {len(df)} lignes trouvees. Colonnes : {list(df.columns)}")

    # Trouver les bonnes colonnes automatiquement
    title_col  = find_column(df, POSSIBLE_TITLE_COLS)
    artist_col = find_column(df, POSSIBLE_ARTIST_COLS)

    if title_col is None:
        print(f"  Colonne titre introuvable. Colonnes disponibles : {list(df.columns)}")
        print(f"  Modifie POSSIBLE_TITLE_COLS dans le script.")
        sys.exit(1)

    if artist_col is None:
        print(f"  Colonne artiste introuvable. Colonnes disponibles : {list(df.columns)}")
        print(f"  Modifie POSSIBLE_ARTIST_COLS dans le script.")
        sys.exit(1)

    print(f"  Colonnes detectees -> titre : '{title_col}', artiste : '{artist_col}'")

    # Nettoyer les lignes vides et doublons
    df = df[[title_col, artist_col]].dropna()
    df = df.drop_duplicates(subset=[title_col, artist_col])

    # Trier par popularite si la colonne existe
    if "popularity" in df.columns:
        df = df.sort_values("popularity", ascending=False)

    tracks = []
    for row in df.head(n).itertuples(index=False):
        title  = str(getattr(row, title_col)).strip()
        artist = str(getattr(row, artist_col)).strip()

        # Certains datasets mettent plusieurs artistes entre crochets ex: "['Artist1', 'Artist2']"
        # On garde uniquement le premier
        if artist.startswith("["):
            artist = re.sub(r"[\[\]'\"]", "", artist).split(",")[0].strip()

        tracks.append({"title": title, "artist": artist})

    print(f"{len(tracks)} titres charges depuis le CSV.\n")
    return tracks


# ===========================================================================
# ETAPE 2 — yt-dlp : recherche YouTube + telechargement MP3 (sans API YouTube)
# ===========================================================================

def download_audio_direct(artist: str, title: str, dest_dir: Path) -> bool:
    """
    Recherche et telecharge directement via yt-dlp sans API YouTube.
    yt-dlp fait lui-meme la recherche YouTube avec ytsearch: — aucune cle requise.
    Retourne True si succes.
    """
    # Nom de fichier propre sans caracteres speciaux
    safe_name = "".join(
        c if c.isalnum() or c in "-_ " else "_"
        for c in f"{artist}_{title}"
    ).strip()
    filename  = f"{safe_name[:60]}.mp3"
    dest_path = dest_dir / filename

    # Ne pas retelecharger si deja present
    if dest_path.exists():
        print(f"  [skip] {filename} deja present")
        return True

    # ytsearch1: prend le 1er resultat YouTube correspondant a la requete
    query = f"{artist} {title} official audio"
    cmd = [
        "yt-dlp",
        f"ytsearch1:{query}",        # recherche YouTube directe, sans API
        "--extract-audio",           # extraire uniquement l'audio
        "--audio-format", "mp3",     # format de sortie MP3
        "--audio-quality", "5",      # qualite ~128kbps
        "--output", str(dest_dir / f"{safe_name[:60]}.%(ext)s"),
        "--quiet",
        "--no-warnings",
        "--socket-timeout", "30",
    ]

    try:
        result = subprocess.run(cmd, timeout=120)
        if result.returncode == 0:
            print(f"  [ok]   {filename}")
            return True
        print(f"  [err]  {filename} (code : {result.returncode})")
        return False
    except subprocess.TimeoutExpired:
        print(f"  [err]  {filename} (timeout)")
        return False
    except FileNotFoundError:
        print("  [err]  yt-dlp introuvable — installe-le avec : pip install yt-dlp")
        sys.exit(1)


# ===========================================================================
# ETAPES 3/4/5 — pipeline de build
# ===========================================================================

def run_step(label: str, cmd: list[str]) -> None:
    """Lance une commande shell et arrete le script si elle echoue."""
    print(f"\n{'─' * 50}")
    print(f"[{label}]")
    print(f"cmd : {' '.join(cmd)}")
    print(f"{'─' * 50}")
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1])
    if result.returncode != 0:
        print(f"Echec a l'etape : {label} (code {result.returncode})")
        sys.exit(result.returncode)
    print(f"OK : {label}")


# ===========================================================================
# ETAPE 6 — verification des chiffres
# ===========================================================================

def verify_pipeline() -> None:
    """
    Verifie la coherence des chiffres entre chaque etape du pipeline.
    Assure que rien n'a ete perdu entre les fichiers audio, les embeddings et l'index.

    Ce qu'on verifie :
        data/raw/          -> X fichiers audio
        metadata.parquet   -> doit avoir X lignes
        segments.parquet   -> Y segments (X morceaux x ~6 segments chacun)
        embeddings.npy     -> doit avoir Y lignes
        index.faiss        -> doit avoir Y vecteurs (index.ntotal == Y)
    """
    import numpy as np
    import faiss
    from src import config

    method = config.EMBEDDING_METHOD
    print(f"\n{'─' * 50}")
    print(f"[VERIFICATION DES CHIFFRES] methode = {method}")
    print(f"{'─' * 50}")

    # Nombre de fichiers audio dans data/raw/
    n_audio = sum(
        1 for f in RAW_DIR.rglob("*")
        if f.suffix.lower() in {".mp3", ".wav", ".flac"}
    )
    print(f"Fichiers audio dans data/raw/      : {n_audio}")

    # Nombre de lignes dans metadata.parquet
    meta_path = Path("data/processed/metadata.parquet")
    if meta_path.exists():
        n_meta = len(pd.read_parquet(meta_path))
        status = "OK" if n_meta == n_audio else "ATTENTION : mismatch"
        print(f"Lignes dans metadata.parquet       : {n_meta}  [{status}]")
    else:
        print(f"metadata.parquet                   : INTROUVABLE")
        return

    # Nombre de segments et d'embeddings
    seg_path = Path(f"data/features/segments_{method}.parquet")
    emb_path = Path(f"data/features/embeddings_{method}.npy")

    if seg_path.exists() and emb_path.exists():
        n_segments   = len(pd.read_parquet(seg_path))
        n_embeddings = np.load(emb_path).shape[0]
        status = "OK" if n_segments == n_embeddings else "ATTENTION : mismatch"
        print(f"Segments dans segments.parquet     : {n_segments}")
        print(f"Vecteurs dans embeddings.npy       : {n_embeddings}  [{status}]")
    else:
        print(f"segments/embeddings ({method})     : INTROUVABLES")
        return

    # Nombre de vecteurs dans l'index FAISS
    index_path = Path(f"data/index/index_{method}.faiss")
    if index_path.exists():
        index  = faiss.read_index(str(index_path))
        status = "OK" if index.ntotal == n_embeddings else "ATTENTION : mismatch"
        print(f"Vecteurs dans index FAISS          : {index.ntotal}  [{status}]")
    else:
        print(f"index FAISS ({method})             : INTROUVABLE")

    print(f"{'─' * 50}\n")


# ===========================================================================
# CLI
# ===========================================================================

@click.command()
@click.option("--csv", "csv_path", default=None,
              help="Chemin vers le CSV Kaggle (si non fourni, cherche dans data/kaggle/)")
@click.option("--n", default=50, show_default=True, help="Nombre de morceaux a telecharger")
def main(csv_path: str | None, n: int) -> None:
    """
    Lit le CSV Kaggle Spotify, telecharge les MP3 via yt-dlp,
    puis lance le pipeline de build et verifie les chiffres.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Trouver le CSV automatiquement si non fourni
    if csv_path is None:
        kaggle_dir = Path("data/kaggle")
        csvs = list(kaggle_dir.glob("*.csv")) if kaggle_dir.exists() else []
        if not csvs:
            print("Aucun CSV trouve dans data/kaggle/")
            print("Telecharge le dataset avec :")
            print("  kaggle datasets download -d anxods/spotify-top-50-playlist-songs-anxods")
            print("  unzip spotify-top-50-playlist-songs-anxods.zip -d data/kaggle/")
            sys.exit(1)
        csv_path = str(csvs[0])
        print(f"CSV detecte automatiquement : {csv_path}")

    # --- Etape 1 : lire les metadonnees depuis le CSV ---
    tracks = load_tracks_from_csv(Path(csv_path), n)

    success = 0

    for i, track in enumerate(tracks, start=1):
        artist = track["artist"]
        title  = track["title"]
        print(f"[{i}/{len(tracks)}] {artist} - {title}")

        # --- Etape 2 : recherche YouTube + telechargement MP3 ---
        ok = download_audio_direct(artist, title, RAW_DIR)
        if ok:
            success += 1

        time.sleep(1.0)  # pause pour eviter le rate-limiting YouTube

    print(f"\nTermine : {success}/{len(tracks)} morceaux telecharges dans {RAW_DIR}/")

    if success == 0:
        print("Aucun morceau telecharge — pipeline annule.")
        sys.exit(1)

    # --- Etapes 3/4/5 : pipeline de build ---
    run_step(
        "1/3 - Construction des metadonnees",
        [sys.executable, "src/data_utils/build_metadata.py"],
    )
    run_step(
        "2/3 - Calcul des embeddings par segment",
        [sys.executable, "scripts/build_segment_embeddings.py"],
    )
    run_step(
        "3/3 - Construction de l'index FAISS",
        [sys.executable, "src/index/build_index.py"],
    )

    # --- Etape 6 : verification des chiffres ---
    verify_pipeline()

    print(f"Pipeline termine — BDD prete avec {success} morceaux.")
    print("Pour identifier un morceau :")
    print("  python src/api/app.py data/raw/mon_fichier.mp3")


if __name__ == "__main__":
    main()


