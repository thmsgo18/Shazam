"""
src/maintenance/clean.py

Suppression complète de la base de données :
embeddings ChromaDB, index FAISS, fingerprints SQLite et metadata.parquet.

Point d'entrée public : run_clean(yes)
"""

from __future__ import annotations

import shutil
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src import config

ROOT    = Path(__file__).resolve().parents[2]
console = Console()

TARGETS = [
    (ROOT / config.CHROMA_DIR,      "Embeddings ChromaDB", "dossier"),
    (ROOT / config.INDEX_DIR,       "Index FAISS",         "dossier"),
    (ROOT / config.FINGERPRINTS_DB, "Fingerprints SQLite", "fichier"),
    (ROOT / config.METADATA_PATH,   "Metadata parquet",    "fichier"),
]


def _get_size(path: Path) -> str:
    """Retourne la taille lisible d'un fichier ou dossier."""
    if not path.exists():
        return "—"
    if path.is_file():
        size = path.stat().st_size
    else:
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    for unit in ("o", "Ko", "Mo", "Go"):
        if size < 1024:
            return f"{size:.0f} {unit}"
        size /= 1024
    return f"{size:.1f} To"


def run_clean(yes: bool = False) -> None:
    """
    Supprime tous les embeddings, fingerprints et métadonnées.

    Args:
        yes: si True, supprime sans demander de confirmation.
    """
    console.print("\n[bold red]Nettoyage de la base de données[/bold red]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Contenu",  width=25)
    table.add_column("Chemin",   width=40)
    table.add_column("Type",     width=8)
    table.add_column("Taille",   justify="right", width=10)
    table.add_column("Statut",   width=12)

    any_exists = False
    for path, label, kind in TARGETS:
        exists = path.exists()
        if exists:
            any_exists = True
        table.add_row(
            label,
            str(path.relative_to(ROOT)),
            kind,
            _get_size(path),
            "[green]présent[/green]" if exists else "[dim]absent[/dim]",
        )

    console.print(table)

    if not any_exists:
        console.print("\n[green]Base déjà vide. Rien à supprimer.[/green]\n")
        return

    if not yes:
        console.print("\n[bold yellow]Cette action est irréversible.[/bold yellow]")
        confirm = input("Confirmer la suppression ? [o/N] : ").strip().lower()
        if confirm not in ("o", "oui", "y", "yes"):
            console.print("[dim]Annulé.[/dim]\n")
            return

    console.print()
    for path, label, kind in TARGETS:
        if not path.exists():
            console.print(f"  [dim]— {label} déjà absent[/dim]")
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            console.print(f"  [green]✓ {label} supprimé[/green]")
        except Exception as exc:
            console.print(f"  [red]✗ {label} — erreur : {exc}[/red]")

    console.print("\n[bold green]Base de données nettoyée.[/bold green]\n")
