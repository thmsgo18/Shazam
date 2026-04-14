"""
src/maintenance/clean.py

Suppression complète ou partielle de la base de données :
embeddings ChromaDB, index FAISS, fingerprints SQLite et metadata.parquet.

Points d'entrée publics :
  run_clean(yes)              — supprime tout
  run_clean_track(track_id)   — supprime un seul track
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


# ---------------------------------------------------------------------------
# Suppression d'un seul track
# ---------------------------------------------------------------------------

def run_clean_track(track_id: str, yes: bool = False) -> None:
    """
    Supprime un track de tous les stockages :
      - Segments dans toutes les collections ChromaDB
      - Fingerprint dans SQLite
      - Ligne dans metadata.parquet

    Imprime un rappel de reconstruire l'index FAISS après la suppression.

    Args:
        track_id: identifiant du track à supprimer.
        yes:      si True, supprime sans demander de confirmation.
    """
    metadata_path = ROOT / config.METADATA_PATH
    fp_db_path    = ROOT / config.FINGERPRINTS_DB

    # --- Vérifier que le track existe ---
    title = track_id
    if metadata_path.exists():
        df = pd.read_parquet(metadata_path)
        row = df[df["track_id"] == track_id]
        if row.empty:
            console.print(f"\n[yellow]Track introuvable dans metadata : {track_id}[/yellow]")
            console.print("[dim]Le nettoyage dans ChromaDB et SQLite sera quand même tenté.[/dim]\n")
        else:
            t = row.iloc[0]
            title = f"{t.get('title', '?')} — {t.get('artist', '?')}"
    else:
        df = None

    console.print(f"\n[bold red]Suppression du track :[/bold red] {title}")
    console.print(f"[dim]track_id : {track_id}[/dim]\n")

    if not yes:
        console.print("[bold yellow]Cette action est irréversible.[/bold yellow]")
        confirm = input("Confirmer la suppression ? [o/N] : ").strip().lower()
        if confirm not in ("o", "oui", "y", "yes"):
            console.print("[dim]Annulé.[/dim]\n")
            return

    # --- ChromaDB : toutes les collections ---
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
                        f"— {len(ids_to_delete)} segment(s) supprimé(s)"
                    )
                else:
                    console.print(f"  [dim]— ChromaDB [{col.name}] : aucun segment trouvé[/dim]")
        except Exception as exc:
            console.print(f"  [red]✗ ChromaDB — erreur : {exc}[/red]")
    else:
        console.print("  [dim]— ChromaDB : dossier absent[/dim]")

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
                console.print(f"  [green]✓ Fingerprint SQLite[/green] — supprimé")
            else:
                console.print("  [dim]— Fingerprint SQLite : aucune entrée trouvée[/dim]")
        except Exception as exc:
            console.print(f"  [red]✗ Fingerprint SQLite — erreur : {exc}[/red]")
    else:
        console.print("  [dim]— Fingerprint SQLite : base absente[/dim]")

    # --- metadata.parquet ---
    if df is not None and not df[df["track_id"] == track_id].empty:
        try:
            df_new = df[df["track_id"] != track_id]
            tmp = metadata_path.with_suffix(".tmp.parquet")
            df_new.to_parquet(tmp, index=False)
            tmp.replace(metadata_path)
            console.print(f"  [green]✓ Metadata[/green] — ligne supprimée")
        except Exception as exc:
            console.print(f"  [red]✗ Metadata — erreur : {exc}[/red]")
    else:
        console.print("  [dim]— Metadata : track absent ou déjà supprimé[/dim]")

    console.print(
        "\n[bold yellow]Index FAISS non mis à jour.[/bold yellow]\n"
        "Lancez [cyan]python manage.py rebuild --what index[/cyan] "
        "pour reconstruire l'index.\n"
    )
