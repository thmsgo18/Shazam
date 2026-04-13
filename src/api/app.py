"""
src/api/app.py

Point d'entrée canonique pour l'identification audio du projet.

Ce module sert à la fois :
  - d'interface CLI (`python src/api/app.py ...`)
  - de couche réutilisable pour `manage.py identify`
  - de couche réutilisable pour le backend web FastAPI
"""

from __future__ import annotations

import math
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*upsample_bicubic2d.*", category=UserWarning)

try:
    import click
except ModuleNotFoundError:  # pragma: no cover - dépendance CLI optionnelle
    click = None

ROOT          = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.config as config

METADATA_PATH = ROOT / "data" / "processed" / "metadata.parquet"


def ensure_project_root_context() -> None:
    """Force un contexte d'exécution cohérent pour tous les points d'entrée."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)


def _clean(val, default=None):
    """Retourne default pour NaN / chaînes vides, sinon la valeur."""
    if val is None:
        return default
    if isinstance(val, float) and math.isnan(val):
        return default
    return val if val != "" else default


def _load_metadata() -> dict[str, dict]:
    """Retourne {track_id: {...}} depuis metadata.parquet."""
    import pandas as pd

    if not METADATA_PATH.exists():
        return {}
    df = pd.read_parquet(METADATA_PATH)
    metadata: dict[str, dict] = {}
    for row in df.itertuples(index=False):
        track_id = str(row.track_id)
        metadata[track_id] = {
            "title":      _clean(getattr(row, "title", None), track_id),
            "artist":     _clean(getattr(row, "artist", None), "Unknown"),
            "album":      _clean(getattr(row, "album", None)),
            "genre":      _clean(getattr(row, "genre", None)),
            "duration_s": _clean(getattr(row, "duration_s", None)),
            "cover_url":  _clean(getattr(row, "cover_url", None)),
        }
    return metadata


def _streaming_links(artist: str, title: str) -> dict[str, str]:
    """Construit des liens de recherche vers les plateformes de streaming."""
    query = f"{artist} {title}".replace(" ", "+")
    return {
        "youtube": f"https://www.youtube.com/results?search_query={query}",
        "spotify": f"https://open.spotify.com/search/{query}",
        "deezer":  f"https://www.deezer.com/search/{query}",
        "apple":   f"https://music.apple.com/search?term={query}",
    }


def _recommendations(track_id: str, genre: str | None, metadata: dict[str, dict], top: int = 4) -> list[dict]:
    """Retourne des recommandations simples basées sur le genre."""
    if not genre:
        return []

    recommendations = []
    for candidate_id, info in metadata.items():
        if candidate_id == track_id or info.get("genre") != genre:
            continue
        recommendations.append({
            "track_id":  candidate_id,
            "title":     info["title"],
            "artist":    info["artist"],
            "album":     info.get("album"),
            "cover_url": info.get("cover_url"),
            "streaming": _streaming_links(info["artist"], info["title"]),
        })
        if len(recommendations) >= top:
            break
    return recommendations


def get_ui_config() -> dict:
    """Expose la configuration UI lue depuis src/config.py."""
    return {
        "listen_duration": config.UI_LISTEN_DURATION,
        "confidence_ratio": config.UI_CONFIDENCE_RATIO,
        "embedding_method": config.EMBEDDING_METHOD,
    }


def identify_audio(
    audio_file: str,
    method: str | None = None,
    top: int | None = None,
    detailed: bool = False,
    metadata: dict[str, dict] | None = None,
) -> list[dict]:
    """Retourne les résultats d'identification enrichis avec les métadonnées."""
    ensure_project_root_context()

    top_n = top if top is not None else config.VECTOR_TOP_N_RESULTS
    from src.retrieval.query_pipeline import identify_track

    raw_results = identify_track(audio_file, method=method, top_n=top_n, detailed=detailed)
    if not raw_results:
        return []

    metadata = metadata if metadata is not None else _load_metadata()
    results = []
    for rank, row in enumerate(raw_results, start=1):
        track_id = row[0]
        score = row[1]
        score_faiss = row[2] if detailed else None
        score_fp = row[3] if detailed else score
        info = metadata.get(track_id, {})
        title = info.get("title", track_id)
        artist = info.get("artist", "Unknown")
        results.append({
            "rank":       rank,
            "track_id":   track_id,
            "title":      title,
            "artist":     artist,
            "album":      info.get("album"),
            "genre":      info.get("genre"),
            "duration_s": info.get("duration_s"),
            "cover_url":  info.get("cover_url"),
            "score":      round(score, 4),
            "score_faiss": round(score_faiss, 4) if score_faiss is not None else None,
            "score_fp":   round(score_fp, 4) if score_fp is not None else None,
            "streaming":  _streaming_links(artist, title),
        })
    return results


def build_identification_response(
    audio_file: str,
    method: str | None = None,
    top: int | None = None,
    detailed: bool = False,
) -> dict:
    """Construit la réponse canonique utilisée par l'interface web."""
    metadata = _load_metadata()
    results = identify_audio(audio_file, method=method, top=top, detailed=detailed, metadata=metadata)
    if not results:
        return {"results": [], "confident": False, "recommendations": []}

    confident = False
    if len(results) >= 2 and results[1]["score"] > 0:
        confident = (results[0]["score"] / results[1]["score"]) >= config.UI_CONFIDENCE_RATIO
    elif len(results) == 1:
        confident = True

    best = results[0]
    recommendations = _recommendations(best["track_id"], best.get("genre"), metadata)
    return {
        "results": results,
        "confident": confident,
        "recommendations": recommendations,
    }


def render_identification_table(results: list[dict], detailed: bool = False) -> None:
    """Affiche les résultats dans un tableau Rich."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=3)
    table.add_column("Artiste", width=24)
    table.add_column("Titre", width=34)
    table.add_column("Score final", justify="right")
    if detailed:
        table.add_column("Score FAISS", justify="right")
        table.add_column("Score FP", justify="right")

    for row in results:
        artist = row["artist"][:24]
        title = row["title"][:34]
        if detailed:
            table.add_row(
                str(row["rank"]),
                artist,
                title,
                f"{row['score']:.4f}",
                f"{row['score_faiss']:.4f}",
                f"{row['score_fp']:.4f}",
            )
        else:
            table.add_row(str(row["rank"]), artist, title, f"{row['score']:.4f}")

    console.print(table)


def run_identify_cli(audio_file: str, method: str | None, top: int, detailed: bool) -> None:
    """Exécute l'identification canonique et l'affiche pour le CLI."""
    from rich.console import Console

    console = Console()
    console.print(f"\n[bold cyan]Identification de :[/bold cyan] {audio_file}")
    if method:
        console.print(f"[dim]Méthode : {method}[/dim]\n")

    results = identify_audio(audio_file, method=method, top=top, detailed=detailed)
    if not results:
        console.print("[red]Aucun résultat trouvé.[/red]")
        return

    render_identification_table(results, detailed=detailed)


if click is not None:
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
        run_identify_cli(audio_file, method=method, top=top, detailed=detailed)
else:
    def identify(*_args, **_kwargs) -> None:
        raise RuntimeError("click is required to run src/api/app.py as a CLI entrypoint.")


if __name__ == "__main__":
    identify()
