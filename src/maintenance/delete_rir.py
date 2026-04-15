"""
src/maintenance/delete_rir.py

Suppression des vecteurs RIR d'une méthode d'embedding :
  1. Scan ChromaDB → collecte tous les IDs contenant "_rir_"
  2. Suppression par batches dans ChromaDB
  3. Réinitialise la colonne rir_augmented dans metadata.parquet

Ne touche PAS aux vecteurs originaux, ni aux fingerprints, ni aux autres méthodes.

Point d'entrée public : run_delete_rir(method, dry_run, yes)
"""

from __future__ import annotations

import sys
from pathlib import Path

import chromadb
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress,
    SpinnerColumn, TaskProgressColumn, TimeElapsedColumn,
)
from rich.table import Table
from rich import box

import src.config as config
from src.utils.metadata import atomic_write_parquet

ROOT    = Path(__file__).resolve().parents[2]
console = Console()

BATCH_DELETE = 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scan_rir_ids(collection) -> list[str]:
    """Retourne tous les IDs RIR présents dans la collection."""
    PAGE   = 1000
    offset = 0
    rir_ids: list[str] = []

    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as prog:
        task = prog.add_task("Scan ChromaDB…", total=None)
        while True:
            page = collection.get(limit=PAGE, offset=offset, include=[])
            ids  = page["ids"]
            if not ids:
                break
            for id_ in ids:
                if "_rir_" in id_:
                    rir_ids.append(id_)
            prog.advance(task, len(ids))
            if len(ids) < PAGE:
                break
            offset += PAGE

    return rir_ids


def _delete_from_chroma(collection, ids: list[str], dry_run: bool) -> int:
    """Supprime les IDs par batches. Retourne le nombre supprimé."""
    if not ids:
        return 0
    if dry_run:
        return len(ids)

    deleted = 0
    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("Suppression ChromaDB…", total=len(ids))
        for i in range(0, len(ids), BATCH_DELETE):
            batch = ids[i: i + BATCH_DELETE]
            collection.delete(ids=batch)
            deleted += len(batch)
            prog.advance(task, len(batch))

    return deleted


def _clear_metadata(meta_path: Path, collection_key: str, dry_run: bool) -> int:
    """Supprime collection_key de rir_augmented. Retourne le nombre de tracks mis à jour."""
    if not meta_path.exists():
        return 0

    df = pd.read_parquet(meta_path)
    if "rir_augmented" not in df.columns:
        return 0

    updated = 0
    for i, row in df.iterrows():
        val = row["rir_augmented"]
        if not isinstance(val, dict):
            continue
        if collection_key not in val:
            continue
        if not dry_run:
            new_val = {k: v for k, v in val.items() if k != collection_key}
            df.at[i, "rir_augmented"] = new_val if new_val else None
        updated += 1

    if not dry_run and updated > 0:
        col       = df["rir_augmented"]
        all_empty = col.apply(
            lambda x: x is None or x != x or (isinstance(x, dict) and not x)
        ).all()
        if all_empty:
            df = df.drop(columns=["rir_augmented"])

        try:
            atomic_write_parquet(meta_path, df)
        except Exception as exc:
            console.print(f"[red]Erreur écriture metadata : {exc}[/red]")

    return updated


# ---------------------------------------------------------------------------
# Point d'entrée public
# ---------------------------------------------------------------------------

def run_delete_rir(
    method: str,
    dry_run: bool = False,
    yes: bool = False,
) -> None:
    """
    Supprime tous les vecteurs RIR d'une méthode dans ChromaDB + metadata.

    Args:
        method:  méthode d'embedding (mfcc / clap / muq / mert).
        dry_run: simule l'opération sans rien supprimer.
        yes:     confirme sans demander.
    """
    import click

    collection_key = config.get_collection_key(method)
    meta_path      = ROOT / config.METADATA_PATH

    console.print(Panel(
        f"[bold]Méthode     :[/bold] [cyan]{method}[/cyan]\n"
        f"[bold]Collection  :[/bold] [cyan]{collection_key}[/cyan]\n"
        f"[bold]Dry-run     :[/bold] [cyan]{dry_run}[/cyan]",
        title="[bold red]Suppression RIR[/bold red]",
        expand=False,
    ))

    chroma_client = chromadb.PersistentClient(path=str(ROOT / config.CHROMA_DIR))
    try:
        collection = chroma_client.get_collection(name=collection_key)
    except Exception:
        console.print(f"[red]Collection '{collection_key}' introuvable.[/red]")
        sys.exit(1)

    total_before = collection.count()
    console.print(
        f"  Collection [cyan]{collection_key}[/cyan] : "
        f"[white]{total_before:,}[/white] vecteurs au total\n"
    )

    console.print("[yellow]Scan des IDs RIR…[/yellow]")
    rir_ids = _scan_rir_ids(collection)

    if not rir_ids:
        console.print("[green]Aucun vecteur RIR trouvé dans ChromaDB.[/green]")
        console.print("[yellow]Vérification metadata.parquet…[/yellow]")
        updated_tracks = _clear_metadata(meta_path, collection_key, dry_run=False)
        if updated_tracks:
            console.print(
                f"[green]✓ metadata.parquet nettoyé ({updated_tracks} tracks).[/green]"
            )
        else:
            console.print("[dim]metadata.parquet déjà propre.[/dim]")
        return

    # Stats par RIR name
    rir_name_counts: dict[str, int] = {}
    for id_ in rir_ids:
        parts = id_.split("_rir_", 1)
        if len(parts) == 2:
            rir_name = "_".join(parts[1].split("_")[:-1])
            rir_name_counts[rir_name] = rir_name_counts.get(rir_name, 0) + 1

    t = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
    t.add_column("RIR name", width=40)
    t.add_column("Segments", justify="right", width=10)
    for name, cnt in sorted(rir_name_counts.items()):
        t.add_row(name, f"{cnt:,}")
    t.add_row("[bold]TOTAL[/bold]", f"[bold]{len(rir_ids):,}[/bold]")
    console.print(t)

    pct = len(rir_ids) / total_before * 100 if total_before else 0
    console.print(
        f"  → [red]{len(rir_ids):,}[/red] vecteurs RIR "
        f"([red]{pct:.1f}%[/red] de la collection)\n"
    )

    if dry_run:
        console.print("[dim]Mode dry-run : aucune modification.[/dim]")
        return

    if not yes:
        click.confirm(
            f"Supprimer {len(rir_ids):,} vecteurs RIR de '{collection_key}' ?",
            abort=True,
        )
        console.print()

    console.print("[yellow]Suppression dans ChromaDB…[/yellow]")
    deleted     = _delete_from_chroma(collection, rir_ids, dry_run=False)
    total_after = collection.count()
    console.print(
        f"[green]✓ {deleted:,} vecteurs supprimés.[/green] "
        f"Collection : {total_after:,} vecteurs restants.\n"
    )

    console.print("[yellow]Mise à jour metadata.parquet…[/yellow]")
    updated_tracks = _clear_metadata(meta_path, collection_key, dry_run=False)
    if updated_tracks:
        console.print(
            f"[green]✓ rir_augmented réinitialisé pour {updated_tracks} track(s).[/green]\n"
        )
    else:
        console.print("[dim]Aucune entrée rir_augmented à nettoyer.[/dim]\n")

    console.print(Panel(
        f"Vecteurs RIR supprimés  : [red]{deleted:,}[/red]\n"
        f"Vecteurs restants       : [green]{total_after:,}[/green]\n"
        f"Tracks metadata nettoyés: [green]{updated_tracks}[/green]\n\n"
        f"[dim]Pense à reconstruire l'index FAISS :[/dim]\n"
        f"[dim]python manage.py rebuild --what index[/dim]",
        title="[bold green]Terminé[/bold green]",
        expand=False,
    ))
