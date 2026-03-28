"""
scripts/download_test_audio.py

Script de test : télécharge un morceau depuis YouTube en MP3 dans data/raw/.

Usage :
    # Morceau complet
    python scripts/download_test_audio.py "Miley Cyrus Flowers"

    # Extrait de 30s au début
    python scripts/download_test_audio.py "Miley Cyrus Flowers" --duration 30

    # Extrait de 15s au milieu
    python scripts/download_test_audio.py "Miley Cyrus Flowers" --duration 15 --position middle

    # Extrait de 10s dans le 3ème quart
    python scripts/download_test_audio.py "Miley Cyrus Flowers" --duration 10 --position third-quarter

Options :
    --duration   Durée de l'extrait en secondes : 5, 10, 15, 30 (défaut : morceau entier)
    --position   Position de départ de l'extrait :
                   start         → début (0%)
                   first-quarter → 1er quart (25%)
                   middle        → milieu (50%)
                   third-quarter → 3ème quart (75%)
                   end           → fin (dernières N secondes)
                 Défaut : start
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import click

RAW_DIR = Path("data/raw")

POSITIONS = ["start", "first-quarter", "middle", "third-quarter", "end"]


def get_duration_seconds(video_url: str) -> float | None:
    """Récupère la durée d'une vidéo YouTube via yt-dlp (JSON)."""
    result = subprocess.run(
        ["yt-dlp", "--dump-json", "--no-playlist", video_url],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    try:
        info = json.loads(result.stdout)
        return float(info.get("duration", 0)) or None
    except (json.JSONDecodeError, ValueError):
        return None


def compute_start(total_duration: float, duration: int, position: str) -> float:
    """Calcule le timestamp de début selon la position choisie."""
    if position == "start":
        return 0.0
    elif position == "first-quarter":
        return total_duration * 0.25
    elif position == "middle":
        return max(0.0, total_duration / 2 - duration / 2)
    elif position == "third-quarter":
        return total_duration * 0.75
    elif position == "end":
        return max(0.0, total_duration - duration)
    return 0.0


@click.command()
@click.argument("query", nargs=-1, required=True)
@click.option(
    "--duration", type=click.Choice(["5", "10", "15", "30"]), default=None,
    help="Durée de l'extrait en secondes. Si absent, télécharge le morceau entier."
)
@click.option(
    "--position", type=click.Choice(POSITIONS), default="start", show_default=True,
    help="Position de départ de l'extrait dans le morceau."
)
def main(query: tuple[str], duration: str | None, position: str) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    search_query = " ".join(query)
    duration_s = int(duration) if duration else None

    click.echo(f"Recherche : {search_query}")

    # --- Résolution de l'URL via ytsearch ---
    resolve = subprocess.run(
        ["yt-dlp", "--get-id", "--no-playlist", f"ytsearch1:{search_query}"],
        capture_output=True, text=True
    )
    if resolve.returncode != 0 or not resolve.stdout.strip():
        click.echo("Erreur : aucune vidéo trouvée sur YouTube.", err=True)
        sys.exit(1)

    video_id  = resolve.stdout.strip()
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    click.echo(f"Vidéo    : {video_url}")

    # --- Téléchargement complet ---
    if duration_s is None:
        cmd = [
            "yt-dlp", video_url,
            "--extract-audio", "--audio-format", "mp3",
            "--audio-quality", "5",
            "--output", str(RAW_DIR / "%(title)s.%(ext)s"),
            "--socket-timeout", "30",
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            click.echo(f"Erreur yt-dlp (code {result.returncode})", err=True)
            sys.exit(result.returncode)

    # --- Téléchargement + découpe d'extrait ---
    else:
        # Récupérer la durée totale pour calculer le point de départ
        click.echo("Récupération des métadonnées...")
        total_duration = get_duration_seconds(video_url)
        if total_duration is None:
            click.echo("Impossible de récupérer la durée — position ignorée, départ à 0s.", err=True)
            start_s = 0.0
        else:
            start_s = compute_start(total_duration, duration_s, position)
            # S'assurer que l'extrait ne dépasse pas la fin du morceau
            max_start = max(0.0, total_duration - duration_s)
            if start_s > max_start:
                click.echo(
                    f"Avertissement : départ à {start_s:.1f}s + {duration_s}s dépasse "
                    f"la durée totale ({total_duration:.0f}s) — départ ramené à {max_start:.1f}s.",
                    err=True
                )
                start_s = max_start
            click.echo(
                f"Durée totale : {total_duration:.0f}s  |  "
                f"Extrait : {duration_s}s  à  {position}  (départ à {start_s:.1f}s)"
            )

        # Télécharger en entier dans un dossier temporaire puis découper avec ffmpeg
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # 1. Télécharger l'audio brut
            dl = subprocess.run([
                "yt-dlp", video_url,
                "--extract-audio", "--audio-format", "mp3",
                "--audio-quality", "5",
                "--output", str(tmp_path / "%(title)s.%(ext)s"),
                "--socket-timeout", "30",
            ])
            if dl.returncode != 0:
                click.echo(f"Erreur yt-dlp (code {dl.returncode})", err=True)
                sys.exit(dl.returncode)

            src_files = list(tmp_path.glob("*.mp3"))
            if not src_files:
                click.echo("Erreur : fichier MP3 introuvable après téléchargement.", err=True)
                sys.exit(1)

            src      = src_files[0]
            stem     = src.stem
            out_name = f"{stem}__{position}_{duration_s}s.mp3"
            out_path = RAW_DIR / out_name

            # 2. Découper avec ffmpeg
            cut = subprocess.run([
                "ffmpeg", "-y",
                "-ss", str(start_s),
                "-i", str(src),
                "-t", str(duration_s),
                "-acodec", "copy",
                str(out_path),
            ], capture_output=True)

            if cut.returncode != 0:
                click.echo("Erreur ffmpeg lors de la découpe.", err=True)
                click.echo(cut.stderr.decode(errors="ignore"), err=True)
                sys.exit(cut.returncode)

    files = sorted(RAW_DIR.glob("*.mp3"), key=lambda f: f.stat().st_mtime, reverse=True)
    if files:
        click.echo(f"\nFichier téléchargé : {files[0]}")


if __name__ == "__main__":
    main()
