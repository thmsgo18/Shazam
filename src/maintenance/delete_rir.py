"""
src/maintenance/delete_rir.py

Deletion of RIR vectors from an embedding method:
  1. Scan ChromaDB → collect all IDs containing "_rir_"
  2. Batch deletion in ChromaDB
  3. Reset the rir_augmented column in metadata.parquet

Does NOT touch the original vectors, nor the fingerprints, nor other methods.

Public entry point: run_delete_rir(method, dry_run, yes)
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
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
    """Returns all RIR IDs present in the collection.

    Fetches all IDs in a single query (O(N)) instead of paginating with
    LIMIT/OFFSET which is O(N²) due to SQLite's offset scan behaviour.
    """
    n = collection.count()
    if n == 0:
        return []

    console.print(f"[dim]Fetching {n:,} IDs in a single pass…[/dim]")
    page    = collection.get(limit=n, offset=0, include=[])
    rir_ids = [id_ for id_ in page["ids"] if "_rir_" in id_]
    console.print(f"[dim]→ {len(rir_ids):,} RIR segment(s) identified[/dim]")
    return rir_ids


def _delete_from_chroma(collection, ids: list[str], dry_run: bool) -> int:
    """Deletes IDs in batches. Returns the number deleted."""
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
        task = prog.add_task("Deleting from ChromaDB…", total=len(ids))
        for i in range(0, len(ids), BATCH_DELETE):
            batch = ids[i: i + BATCH_DELETE]
            collection.delete(ids=batch)
            deleted += len(batch)
            prog.advance(task, len(batch))

    return deleted


def _clear_metadata(meta_path: Path, collection_key: str, dry_run: bool) -> int:
    """Removes collection_key from rir_augmented. Returns the number of updated tracks."""
    if not meta_path.exists():
        return 0

    df = pd.read_parquet(meta_path)
    if "rir_augmented" not in df.columns:
        return 0

    mask    = df["rir_augmented"].apply(
        lambda x: isinstance(x, dict) and collection_key in x
    )
    updated = int(mask.sum())

    if not dry_run and updated > 0:
        df.loc[mask, "rir_augmented"] = df.loc[mask, "rir_augmented"].apply(
            lambda x: ({k: v for k, v in x.items() if k != collection_key} or None)
        )

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
            console.print(f"[red]Error writing metadata: {exc}[/red]")

    return updated


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_delete_rir(
    method: str,
    dry_run: bool = False,
    yes: bool = False,
) -> None:
    """
    Deletes all RIR vectors for a method in ChromaDB + metadata.

    Args:
        method:  embedding method (mfcc / clap / muq / mert).
        dry_run: simulates the operation without deleting anything.
        yes:     confirms without asking.
    """

    collection_key = config.get_collection_key(method)
    meta_path      = ROOT / config.METADATA_PATH

    console.print(Panel(
        f"[bold]Method      :[/bold] [cyan]{method}[/cyan]\n"
        f"[bold]Collection  :[/bold] [cyan]{collection_key}[/cyan]\n"
        f"[bold]Dry-run     :[/bold] [cyan]{dry_run}[/cyan]",
        title="[bold red]RIR Deletion[/bold red]",
        expand=False,
    ))

    chroma_client = chromadb.PersistentClient(path=str(ROOT / config.CHROMA_DIR))
    try:
        collection = chroma_client.get_collection(name=collection_key)
    except Exception:
        console.print(f"[red]Collection '{collection_key}' not found.[/red]")
        sys.exit(1)

    total_before = collection.count()
    console.print(
        f"  Collection [cyan]{collection_key}[/cyan]: "
        f"[white]{total_before:,}[/white] total vectors\n"
    )

    console.print("[yellow]Scanning RIR IDs…[/yellow]")
    rir_ids = _scan_rir_ids(collection)

    if not rir_ids:
        console.print("[green]No RIR vector found in ChromaDB.[/green]")
        console.print("[yellow]Checking metadata.parquet…[/yellow]")
        updated_tracks = _clear_metadata(meta_path, collection_key, dry_run=False)
        if updated_tracks:
            console.print(
                f"[green]✓ metadata.parquet cleaned ({updated_tracks} tracks).[/green]"
            )
        else:
            console.print("[dim]metadata.parquet is already clean.[/dim]")
        return

    # Stats per RIR name
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
        f"  → [red]{len(rir_ids):,}[/red] RIR vectors "
        f"([red]{pct:.1f}%[/red] of the collection)\n"
    )

    if dry_run:
        console.print("[dim]Dry-run mode: no changes made.[/dim]")
        return

    if not yes:
        click.confirm(
            f"Delete {len(rir_ids):,} RIR vectors from '{collection_key}'?",
            abort=True,
        )
        console.print()

    console.print("[yellow]Deleting in ChromaDB…[/yellow]")
    deleted     = _delete_from_chroma(collection, rir_ids, dry_run=False)
    total_after = collection.count()
    console.print(
        f"[green]✓ {deleted:,} vectors deleted.[/green] "
        f"Collection: {total_after:,} vectors remaining.\n"
    )

    console.print("[yellow]Updating metadata.parquet…[/yellow]")
    updated_tracks = _clear_metadata(meta_path, collection_key, dry_run=False)
    if updated_tracks:
        console.print(
            f"[green]✓ rir_augmented reset for {updated_tracks} track(s).[/green]\n"
        )
    else:
        console.print("[dim]No rir_augmented entry to clean.[/dim]\n")

    console.print(Panel(
        f"RIR vectors deleted     : [red]{deleted:,}[/red]\n"
        f"Vectors remaining       : [green]{total_after:,}[/green]\n"
        f"Metadata tracks cleaned : [green]{updated_tracks}[/green]\n\n"
        f"[dim]Remember to rebuild the FAISS index:[/dim]\n"
        f"[dim]python manage.py rebuild --what index[/dim]",
        title="[bold green]Finished[/bold green]",
        expand=False,
    ))