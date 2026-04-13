"""
src/maintenance/enrich.py

Enrichissement des métadonnées depuis Deezer + MusicBrainz (fallback).

Sources en cascade :
    1. Deezer API   — gratuit, sans clé, excellent pour la musique internationale
    2. MusicBrainz  — fallback pour les tracks introuvables sur Deezer (1 req/sec)

Point d'entrée public : run_enrich(force, only_missing)
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from rich.console import Console
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress,
    SpinnerColumn, TextColumn, TimeElapsedColumn,
)

from src.utils.metadata import atomic_write_parquet

ROOT          = Path(__file__).resolve().parents[2]
METADATA_PATH = ROOT / "data" / "processed" / "metadata.parquet"
ITUNES_FIELDS = ["album", "genre", "release_date", "cover_url"]

DEEZER_DELAY      = 0.1
MUSICBRAINZ_DELAY = 1.1

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "ShazamMaison/1.0 (university-project)"})

console = Console()


# ---------------------------------------------------------------------------
# Nettoyage des noms
# ---------------------------------------------------------------------------

def _clean_artist(artist: str) -> str:
    artist = re.split(r"\s*[&,]\s*", artist)[0].strip()
    artist = re.sub(r"[¥$]", "", artist).strip()
    return artist


def _clean_title(title: str) -> str:
    title = re.sub(r"\(.*?\)", "", title).strip()
    title = re.sub(r"\s*(feat\.|ft\.)\s+.*", "", title, flags=re.IGNORECASE).strip()
    return title or title


# ---------------------------------------------------------------------------
# Deezer API
# ---------------------------------------------------------------------------

def _deezer_artist_genre(artist_id: int) -> str | None:
    try:
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
    try:
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

        resp2 = SESSION.get(f"https://api.deezer.com/album/{album_id}", timeout=8)
        resp2.raise_for_status()
        album  = resp2.json()
        genres = [g["name"] for g in album.get("genres", {}).get("data", [])]
        genre  = genres[0] if genres else _deezer_artist_genre(artist_id)

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


# ---------------------------------------------------------------------------
# MusicBrainz API (fallback)
# ---------------------------------------------------------------------------

def _musicbrainz_search(artist: str, title: str) -> dict | None:
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
        tags         = sorted(rec.get("tags", []), key=lambda t: t.get("count", 0), reverse=True)
        genre        = tags[0]["name"].title() if tags else None

        return {
            "album":        album,
            "genre":        genre,
            "release_date": release_date,
            "cover_url":    None,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Cascade Deezer → MusicBrainz
# ---------------------------------------------------------------------------

def _fetch_metadata(artist: str, title: str) -> dict:
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

    result = _musicbrainz_search(artist_clean, title_clean)
    time.sleep(MUSICBRAINZ_DELAY)
    return result or fallback


# ---------------------------------------------------------------------------
# Point d'entrée public
# ---------------------------------------------------------------------------

def run_enrich(force: bool = False, only_missing: bool = False) -> None:
    """
    Enrichit metadata.parquet avec les données Deezer (+ MusicBrainz en fallback).

    Args:
        force:        si True, met à jour tous les tracks, même ceux déjà enrichis.
        only_missing: ne traite que les tracks avec au moins un champ vide (défaut sans --force).
    """
    if not METADATA_PATH.exists():
        console.print("[red]metadata.parquet introuvable. Lance d'abord l'ingestion.[/red]")
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
        console.print(
            f"[cyan]{len(to_enrich)} track(s) avec au moins un champ vide sur {len(df)} total.[/cyan]"
        )

    if to_enrich.empty:
        console.print("[green]Tous les tracks sont déjà enrichis.[/green]")
        return

    updated   = 0
    not_found = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
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

    atomic_write_parquet(METADATA_PATH, df)

    console.print(
        f"\n[green]✓ {updated} track(s) enrichi(s)[/green]  •  "
        f"[yellow]{not_found} introuvable(s) sur Deezer/MusicBrainz[/yellow]"
    )
