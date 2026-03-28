"""
scripts/enrich_metadata.py

Enrichit metadata.parquet avec les données musicales (album, genre, release_date, cover_url).

Sources utilisées en cascade :
    1. Deezer API   — gratuit, sans clé, excellent pour la musique internationale
    2. MusicBrainz  — fallback pour les tracks introuvables sur Deezer (1 req/sec)

Usage :
    # Enrichir uniquement les tracks sans métadonnées
    python scripts/enrich_metadata.py

    # Forcer la mise à jour de tous les tracks (même ceux déjà enrichis)
    python scripts/enrich_metadata.py --force
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import time
from pathlib import Path

import click
import pandas as pd
import requests
from rich.console import Console
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress,
    SpinnerColumn, TextColumn, TimeElapsedColumn,
)

METADATA_PATH  = Path("data/processed/metadata.parquet")
ITUNES_FIELDS  = ["album", "genre", "release_date", "cover_url"]
DEEZER_DELAY   = 0.1   # secondes entre requêtes Deezer
MUSICBRAINZ_DELAY = 1.1  # secondes entre requêtes MusicBrainz (limite : 1 req/sec)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "ShazamMaison/1.0 (university-project)"})

console = Console()


# ===========================================================================
# Nettoyage du nom d'artiste
# ===========================================================================

def _clean_artist(artist: str) -> str:
    """
    Simplifie un nom d'artiste complexe pour améliorer le matching.
    Ex : "¥$ & Kanye West & Ty Dolla $ign" → "Kanye West"
    """
    artist = re.split(r"\s*[&,]\s*", artist)[0].strip()
    artist = re.sub(r"[¥$]", "", artist).strip()
    return artist


def _clean_title(title: str) -> str:
    """
    Simplifie un titre complexe pour améliorer le matching.
    Ex : "Calling (Spider-Man: Across the Spider-Verse) (feat. A Boogie...)" → "Calling"
    """
    # Supprimer tout ce qui est entre parenthèses
    title = re.sub(r"\(.*?\)", "", title).strip()
    # Supprimer les mentions feat. / ft. résiduelles
    title = re.sub(r"\s*(feat\.|ft\.)\s+.*", "", title, flags=re.IGNORECASE).strip()
    return title or title  # retourner l'original si vide après nettoyage


# ===========================================================================
# Deezer API
# ===========================================================================

def _deezer_artist_genre(artist_id: int) -> str | None:
    """
    Récupère le genre principal d'un artiste via l'endpoint Deezer /artist.
    Utilisé en fallback quand l'album n'a pas de genres.
    """
    try:
        resp = SESSION.get(f"https://api.deezer.com/artist/{artist_id}/top?limit=1", timeout=8)
        resp.raise_for_status()
        # Récupérer les infos de l'artiste directement
        resp2 = SESSION.get(f"https://api.deezer.com/artist/{artist_id}", timeout=8)
        resp2.raise_for_status()
        # L'artiste Deezer ne retourne pas de genre directement,
        # on tente via le genre de son album le plus populaire
        resp3 = SESSION.get(f"https://api.deezer.com/artist/{artist_id}/albums?limit=1", timeout=8)
        resp3.raise_for_status()
        albums = resp3.json().get("data", [])
        if not albums:
            return None
        album_id = albums[0]["id"]
        resp4 = SESSION.get(f"https://api.deezer.com/album/{album_id}", timeout=8)
        resp4.raise_for_status()
        genres = [g["name"] for g in resp4.json().get("genres", {}).get("data", [])]
        return genres[0] if genres else None
    except Exception:
        return None


def _deezer_search(artist: str, title: str) -> dict | None:
    """
    Cherche un morceau sur Deezer et retourne ses métadonnées enrichies.
    Effectue 2 appels : search → album (pour le genre et la date complète).
    Si l'album n'a pas de genre, tente via l'artiste.
    Retourne None si rien n'est trouvé.
    """
    try:
        # Appel 1 : recherche du morceau
        resp = SESSION.get(
            "https://api.deezer.com/search",
            params={"q": f"{artist} {title}", "limit": 5},
            timeout=8,
        )
        resp.raise_for_status()
        tracks = resp.json().get("data", [])
        if not tracks:
            return None

        track     = tracks[0]
        album_id  = track["album"]["id"]
        artist_id = track["artist"]["id"]

        # Appel 2 : détails de l'album (genre + date complète)
        resp2 = SESSION.get(f"https://api.deezer.com/album/{album_id}", timeout=8)
        resp2.raise_for_status()
        album = resp2.json()

        genres = [g["name"] for g in album.get("genres", {}).get("data", [])]

        # Si l'album n'a pas de genre → fallback via l'artiste
        genre = genres[0] if genres else _deezer_artist_genre(artist_id)

        return {
            "album":        track["album"]["title"],
            "genre":        genre,
            "release_date": album.get("release_date"),
            "cover_url":    (
                track["album"].get("cover_xl")
                or track["album"].get("cover_big")
                or track["album"].get("cover_medium")
            ),
        }
    except Exception:
        return None


# ===========================================================================
# MusicBrainz API (fallback)
# ===========================================================================

def _musicbrainz_search(artist: str, title: str) -> dict | None:
    """
    Cherche un morceau sur MusicBrainz (fallback si Deezer ne trouve rien).
    Limite stricte : 1 requête/seconde.
    Retourne None si rien n'est trouvé.
    """
    try:
        resp = SESSION.get(
            "https://musicbrainz.org/ws/2/recording/",
            params={
                "query": f'artist:"{artist}" AND recording:"{title}"',
                "fmt":   "json",
                "limit": 1,
            },
            timeout=10,
        )
        resp.raise_for_status()
        recordings = resp.json().get("recordings", [])
        if not recordings:
            return None

        rec      = recordings[0]
        releases = rec.get("releases", [])
        if not releases:
            return None

        release      = releases[0]
        release_date = release.get("date")
        album        = release.get("title")

        # Genres via tags (crowd-sourcés, triés par score)
        tags  = sorted(rec.get("tags", []), key=lambda t: t.get("count", 0), reverse=True)
        genre = tags[0]["name"].title() if tags else None

        return {
            "album":        album,
            "genre":        genre,
            "release_date": release_date,
            "cover_url":    None,  # Cover Art Archive non utilisé (appel supplémentaire)
        }
    except Exception:
        return None


# ===========================================================================
# Cascade Deezer → MusicBrainz
# ===========================================================================

def _fetch_metadata(artist: str, title: str) -> dict:
    """
    Récupère les métadonnées en cascade :
    1. Deezer — artiste complet + titre complet
    2. Deezer — artiste simplifié + titre complet
    3. Deezer — artiste simplifié + titre nettoyé (sans parenthèses)
    4. MusicBrainz — fallback final
    """
    fallback     = {"album": None, "genre": None, "release_date": None, "cover_url": None}
    artist_clean = _clean_artist(artist)
    title_clean  = _clean_title(title)

    queries = [(artist, title)]
    if artist_clean != artist:
        queries.append((artist_clean, title))
    if title_clean != title:
        queries.append((artist_clean, title_clean))

    for a, t in queries:
        result = _deezer_search(a, t)
        time.sleep(DEEZER_DELAY)
        if result:
            return result

    # MusicBrainz en dernier recours
    result = _musicbrainz_search(artist_clean, title_clean)
    time.sleep(MUSICBRAINZ_DELAY)
    return result or fallback


# ===========================================================================
# Écriture atomique
# ===========================================================================

def _atomic_write_parquet(path: Path, df: pd.DataFrame) -> None:
    """Écrit un fichier parquet de manière atomique (temp file + rename)."""
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(tmp_fd)
    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


# ===========================================================================
# CLI
# ===========================================================================

@click.command()
@click.option("--force", is_flag=True, default=False,
              help="Met à jour tous les tracks, même ceux déjà enrichis.")
@click.option("--only-missing", is_flag=True, default=False,
              help="Ne traite que les tracks avec au moins un champ vide (défaut sans --force).")
def main(force: bool, only_missing: bool) -> None:
    """
    Enrichit metadata.parquet avec les données Deezer (+ MusicBrainz en fallback).
    Par défaut, ne traite que les tracks dont au moins un champ est vide.
    """
    if not METADATA_PATH.exists():
        console.print("[red]metadata.parquet introuvable. Lance d'abord download_music.py.[/red]")
        sys.exit(1)

    df = pd.read_parquet(METADATA_PATH)

    for col in ITUNES_FIELDS:
        if col not in df.columns:
            df[col] = None

    if force:
        to_enrich = df
        console.print(f"[yellow]Mode --force : {len(to_enrich)} tracks à mettre à jour.[/yellow]")
    else:
        mask      = df[ITUNES_FIELDS].isnull().any(axis=1)
        to_enrich = df[mask]
        console.print(f"[cyan]{len(to_enrich)} track(s) avec au moins un champ vide sur {len(df)} total.[/cyan]")

    if to_enrich.empty:
        console.print("[green]Tous les tracks sont déjà enrichis.[/green]")
        return

    updated   = 0
    not_found = 0

    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    ]

    with Progress(*progress_columns, console=console) as progress:
        task = progress.add_task("[cyan]Enrichissement...", total=len(to_enrich))

        for row in to_enrich.itertuples():
            artist = str(row.artist)
            title  = str(row.title)

            progress.update(task, description=f"[cyan]{artist[:28]} — {title[:23]}")

            metadata = _fetch_metadata(artist, title)

            if all(v is None for v in metadata.values()):
                not_found += 1
            else:
                for field in ITUNES_FIELDS:
                    if force or pd.isnull(df.at[row.Index, field]):
                        df.at[row.Index, field] = metadata[field]
                updated += 1

            progress.advance(task)

    _atomic_write_parquet(METADATA_PATH, df)

    console.print(
        f"\n[green]✓ {updated} track(s) enrichi(s)[/green]  •  "
        f"[yellow]{not_found} introuvable(s) sur Deezer/MusicBrainz[/yellow]"
    )


if __name__ == "__main__":
    main()
