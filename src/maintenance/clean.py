"""
src/maintenance/clean.py

Complete or partial deletion of the database:
ChromaDB embeddings, FAISS index, SQLite fingerprints, and metadata.parquet.

Public entry points:
  run_clean(yes)              — deletes everything
  run_clean_track(track_id)   — deletes a single track
"""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from src import config

ROOT    = Path(__file__).resolve().parents[2]
console = Console()

TARGETS = [
    (ROOT / config.CHROMA_DIR,      "ChromaDB Embeddings", "folder"),
    (ROOT / config.INDEX_DIR,       "FAISS Index",         "folder"),
    (ROOT / config.FINGERPRINTS_DB, "SQLite Fingerprints", "file"),
    (ROOT / config.METADATA_PATH,   "Parquet Metadata",    "file"),
]


def _get_size(path: Path) -> str:
    """Returns the human-readable size of a file or folder."""
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
    Deletes all embeddings, fingerprints, and metadata.

    Args:
        yes: if True, deletes without asking for confirmation.
    """
    console.print("\n[bold red]Database cleanup[/bold red]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Content",  width=25)
    table.add_column("Path",     width=40)
    table.add_column("Type",     width=8)
    table.add_column("Size",     justify="right", width=10)
    table.add_column("Status",   width=12)

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
            "[green]present[/green]" if exists else "[dim]absent[/dim]",
        )

    console.print(table)

    if not any_exists:
        console.print("\n[green]Database is already empty. Nothing to delete.[/green]\n")
        return

    if not yes:
        console.print("\n[bold yellow]This action is irreversible.[/bold yellow]")
        confirm = input("Confirm deletion? [y/N]: ").strip().lower()
        if confirm not in ("y", "yes"):
            console.print("[dim]Cancelled.[/dim]\n")
            return

    console.print()
    for path, label, kind in TARGETS:
        if not path.exists():
            console.print(f"  [dim]— {label} already absent[/dim]")
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            console.print(f"  [green]✓ {label} deleted[/green]")
        except Exception as exc:
            console.print(f"  [red]✗ {label} — error : {exc}[/red]")

    console.print("\n[bold green]Database cleaned.[/bold green]\n")


# ---------------------------------------------------------------------------
# Deletion of a single track
# ---------------------------------------------------------------------------

def run_clean_track(track_id: str, yes: bool = False) -> None:
    """
    Deletes a track from all storage locations:
      - Segments in all ChromaDB collections
      - Fingerprint in SQLite
      - Row in metadata.parquet

    Prints a reminder to rebuild the FAISS index after deletion.
    
    Args:
        track_id: identifier of the track to delete.
        yes:      if True, deletes without asking for confirmation.
    """
    metadata_path = ROOT / config.METADATA_PATH
    fp_db_path    = ROOT / config.FINGERPRINTS_DB

    # --- Check that the track exists ---
    title = track_id
    if metadata_path.exists():
        df = pd.read_parquet(metadata_path)
        row = df[df["track_id"] == track_id]
        if row.empty:
            console.print(f"\n[yellow]Track not found in metadata: {track_id}[/yellow]")
            console.print("[dim]Cleanup in ChromaDB and SQLite will still be attempted.[/dim]\n")
        else:
            t = row.iloc[0]
            title = f"{t.get('title', '?')} — {t.get('artist', '?')}"
    else:
        df = None

    console.print(f"\n[bold red]Deleting track:[/bold red] {title}")
    console.print(f"[dim]track_id: {track_id}[/dim]\n")


    if not yes:
        console.print("[bold yellow]This action is irreversible.[/bold yellow]")
        confirm = input("Confirm deletion? [y/N]: ").strip().lower()
        if confirm not in ("y", "yes"):
            console.print("[dim]Cancelled.[/dim]\n")
            return

    # --- ChromaDB: all collections ---
    chroma_dir = ROOT / config.CHROMA_DIR
    if chroma_dir.exists():
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(chroma_dir))
            collections = client.list_collections()
            for col in collections:
                results = col.get(where={"track_id": track_id}, include=[])
                ids_to_delete = results.get("ids", [])
                if ids_to_delete:
                    col.delete(ids=ids_to_delete)
                    console.print(
                        f"  [green]✓ ChromaDB [{col.name}][/green] "
                        f"— {len(ids_to_delete)} segment(s) deleted"
                    )
                else:
                    console.print(f"  [dim]— ChromaDB [{col.name}]: no segment found[/dim]")
        except Exception as exc:
            console.print(f"  [red]✗ ChromaDB — error: {exc}[/red]")
    else:
        console.print("  [dim]— ChromaDB: folder missing[/dim]")

    # --- SQLite fingerprints ---
    if fp_db_path.exists():
        try:
            conn = sqlite3.connect(str(fp_db_path))
            cur  = conn.execute(
                "DELETE FROM fingerprints WHERE track_id = ?", (track_id,)
            )
            conn.commit()
            conn.close()
            if cur.rowcount:
                console.print(f"  [green]✓ SQLite Fingerprint[/green] — deleted")
            else:
                console.print("  [dim]— SQLite Fingerprint: no entry found[/dim]")
        except Exception as exc:
            console.print(f"  [red]✗ SQLite Fingerprint — error: {exc}[/red]")
    else:
        console.print("  [dim]— SQLite Fingerprint: database missing[/dim]")

    # --- metadata.parquet ---
    if df is not None and not df[df["track_id"] == track_id].empty:
        try:
            df_new = df[df["track_id"] != track_id]
            tmp = metadata_path.with_suffix(".tmp.parquet")
            df_new.to_parquet(tmp, index=False)
            tmp.replace(metadata_path)
            console.print(f"  [green]✓ Metadata[/green] — row deleted")
        except Exception as exc:
            console.print(f"  [red]✗ Metadata — error: {exc}[/red]")
    else:
        console.print("  [dim]— Metadata: track missing or already deleted[/dim]")


    console.print(
        "\n[bold yellow]FAISS index not updated.[/bold yellow]\n"
        "Run [cyan]python manage.py rebuild --what index[/cyan] "
        "to rebuild the index.\n"
    )