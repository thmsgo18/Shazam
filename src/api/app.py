"""
src/api/app.py

Interface en ligne de commande pour identifier un morceau audio.
Responsable : Personne D

Usage :
    python src/api/app.py data/raw/music-fma-0000.wav
    python src/api/app.py data/raw/music-fma-0000.wav --method clap --top 3
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@click.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option("--method", default=None, help="mfcc / clap / muq (défaut : config.py)")
@click.option("--top", default=5, show_default=True, help="Nombre de résultats à afficher")
def identify(audio_file: str, method: str | None, top: int) -> None:
    """
    Identifie le morceau correspondant à AUDIO_FILE.

    Affiche le classement des morceaux les plus probables avec leur score.

    Dépendances :
        - src.retrieval.query_pipeline.identify_track()
        - data/processed/metadata.parquet  (pour afficher le nom de fichier)
    """
    ...


if __name__ == "__main__":
    identify()
