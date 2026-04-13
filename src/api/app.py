"""
src/api/app.py

Interface en ligne de commande pour identifier un morceau audio.
Responsable : Personne D

Usage :
    python src/api/app.py data/raw/music-fma-0000.wav
    python src/api/app.py data/raw/music-fma-0000.wav --method clap --top 3
"""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*upsample_bicubic2d.*", category=UserWarning)

import click
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.retrieval.query_pipeline import identify_track

ROOT          = Path(__file__).resolve().parents[2]
METADATA_PATH = ROOT / "data" / "processed" / "metadata.parquet"
console       = Console()


def _load_metadata() -> dict[str, dict]:
    """Retourne {track_id: {title, artist}} depuis metadata.parquet."""
    if not METADATA_PATH.exists():
        return {}
    df = pd.read_parquet(METADATA_PATH, columns=["track_id", "title", "artist"])
    return {row.track_id: {"title": row.title, "artist": row.artist} for row in df.itertuples()}


@click.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option("--method",   default=None,  help="mfcc / clap / muq (défaut : config.py)")
@click.option("--top",      default=5,     show_default=True, help="Nombre de résultats à afficher")
@click.option("--detailed", is_flag=True,  default=False, help="Afficher le détail des scores FAISS et fingerprint")
def identify(audio_file: str, method: str | None, top: int, detailed: bool) -> None:
    """
    Identifie le morceau correspondant à AUDIO_FILE.

    Affiche le classement des morceaux les plus probables avec leur score.
    Avec --detailed : affiche aussi les scores FAISS et fingerprint séparément.
    """
    console.print(f"\n[bold cyan]Identification de :[/bold cyan] {audio_file}")
    if method:
        console.print(f"[dim]Méthode : {method}[/dim]\n")

    results = identify_track(audio_file, method=method, top_n=top, detailed=detailed)

    if not results:
        console.print("[red]Aucun résultat trouvé.[/red]")
        return

    metadata = _load_metadata()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#",           style="dim", width=3)
    table.add_column("Artiste",                  width=24)
    table.add_column("Titre",                    width=34)
    table.add_column("Score final", justify="right")
    if detailed:
        table.add_column("Score FAISS", justify="right")
        table.add_column("Score FP",    justify="right")

    for rank, row in enumerate(results, start=1):
        track_id    = row[0]
        score_final = row[1]
        info   = metadata.get(track_id, {})
        artist = info.get("artist", track_id)[:24]
        title  = info.get("title",  "—")[:34]

        if detailed:
            score_faiss = row[2]
            score_fp    = row[3]
            table.add_row(str(rank), artist, title,
                          f"{score_final:.4f}",
                          f"{score_faiss:.4f}",
                          f"{score_fp:.4f}")
        else:
            table.add_row(str(rank), artist, title, f"{score_final:.4f}")

    console.print(table)


if __name__ == "__main__":
    identify()
