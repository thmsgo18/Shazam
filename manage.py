#!/usr/bin/env python3
"""
manage.py — Point d'entrée unique du projet Shazam Maison.

Usage :
    python manage.py <commande> [options]

Commandes disponibles :
    ── Ingestion ──────────────────────────────────────────────────────────────
    ingest                 Télécharge et ingère des tracks depuis des CSV Kaggle
    augment                Augmente les embeddings avec des RIR (Room Impulse Response)
    rebuild-fingerprints   Recalcule les fingerprints Shazam (SQLite)
    build-index            (Re)construit l'index FAISS depuis ChromaDB

    ── Maintenance ────────────────────────────────────────────────────────────
    check                  Vérifie la cohérence des données (ChromaDB / FAISS / fingerprints)
    enrich                 Enrichit metadata.parquet via Deezer + MusicBrainz
    clean                  Supprime toutes les données (ChromaDB, FAISS, fingerprints, metadata)
    delete-rir             Supprime les vecteurs RIR d'une méthode dans ChromaDB

    ── Évaluation ─────────────────────────────────────────────────────────────
    find-track             Pipeline complet sur un fichier audio (Stage 1 + Stage 2)
    benchmark              Benchmark de robustesse (Flowers - Miley Cyrus, une méthode)
    evaluate               Évaluation multi-tracks multi-méthodes avec graphiques
    rir-evaluate           Compare Stage 1 avec vs sans RIR sur plusieurs morceaux
    plots                  Génère tous les graphiques pour le rapport (G1,G2,G4,G6,G9,G11,G12)
    rir-impact             Analyse RIR détaillée sur un seul fichier (affichage riche)

    ── Interface web ───────────────────────────────────────────────────────────
    start-webapp           Démarre backend FastAPI + frontend React/Vite
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Racine du projet dans sys.path pour que tous les imports src.* fonctionnent
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import click


# ═══════════════════════════════════════════════════════════════════════════════
# Groupe principal
# ═══════════════════════════════════════════════════════════════════════════════

@click.group()
def cli():
    """Shazam Maison — système de reconnaissance musicale (Master IAD S2)."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# INGESTION
# ═══════════════════════════════════════════════════════════════════════════════

@cli.command()
@click.option("--csv", "csv_paths", multiple=True,
              help="Chemin(s) vers un CSV Kaggle (répétable). "
                   "Défaut : data/kaggle/data/spotify-streaming-top-50-world.csv")
def ingest(csv_paths: tuple[str, ...]) -> None:
    """Télécharge et ingère des tracks depuis des CSV Kaggle."""
    from src.ingestion.ingest import run_ingest
    run_ingest(list(csv_paths) if csv_paths else None)


@cli.command()
@click.option("--method",
              type=click.Choice(["mfcc", "clap", "muq", "mert"]),
              default=None, help="Méthode d'embedding (défaut : config.py)")
@click.option("--tracks", multiple=True,
              help="track_id(s) à traiter (défaut : tous les tracks présents)")
@click.option("--rir-source", default=None,
              type=click.Choice(["synthetic", "mit"]),
              help="Source des RIRs : 'synthetic' (générées) ou 'mit' (WAV réels). Défaut : config.RIR_SOURCE")
@click.option("--n-rir", default=None, type=int,
              help="Nombre de RIRs par track (défaut : config.RIR_N)")
@click.option("--rir-dir", default=None,
              help="Dossier WAV MIT (défaut : config.RIR_MIT_DIR, uniquement si --rir-source=mit)")
@click.option("--workers", default=3, show_default=True,
              help="Workers parallèles pour le téléchargement")
@click.option("--device", default=None,
              help="Device PyTorch : cpu / cuda / mps (défaut : auto)")
@click.option("--no-rebuild-index", is_flag=True, default=False,
              help="Ne pas reconstruire l'index FAISS après l'augmentation")
def augment(
    method: str | None,
    tracks: tuple[str, ...],
    rir_source: str | None,
    n_rir: int | None,
    rir_dir: str | None,
    workers: int,
    device: str | None,
    no_rebuild_index: bool,
) -> None:
    """Augmente les embeddings avec des RIR (Room Impulse Response)."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK",      "1")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    from src.ingestion.augment_rir import run_augment
    run_augment(
        method=method,
        tracks=list(tracks) if tracks else "all",
        n_rir=n_rir,
        rir_dir=rir_dir,
        source=rir_source,
        workers=workers,
        device=device,
        rebuild_index=not no_rebuild_index,
    )


@cli.command("rebuild-fingerprints")
@click.option("--force-all", is_flag=True, default=False,
              help="Recalcule même les tracks déjà dans la DB")
@click.option("--limit", default=None, type=int,
              help="Limite le nombre de tracks à traiter")
@click.option("--workers", default=4, show_default=True,
              help="Threads parallèles")
@click.option("--dry-run", is_flag=True, default=False,
              help="Simule sans écrire")
def rebuild_fingerprints(
    force_all: bool,
    limit: int | None,
    workers: int,
    dry_run: bool,
) -> None:
    """Recalcule les fingerprints Shazam (SQLite) pour tous les tracks."""
    from src.ingestion.fingerprints import run_rebuild_fingerprints
    run_rebuild_fingerprints(
        force_all=force_all,
        limit=limit,
        workers=workers,
        dry_run=dry_run,
    )


@cli.command("build-index")
@click.option("--method", default=None,
              help="mfcc / clap / muq / mert — si omis, toutes les collections")
def build_index(method: str | None) -> None:
    """(Re)construit l'index FAISS depuis ChromaDB."""
    import chromadb
    from src import config
    from src.index.build_index import _build_for_method

    index_type    = getattr(config, "INDEX_TYPE", "flat")
    chroma_client = chromadb.PersistentClient(path=str(ROOT / config.CHROMA_DIR))

    if method:
        keys = [config.get_collection_key(method)]
        click.echo(f"[build-index] Clé collection résolue : {keys[0]}")
    else:
        keys = [c.name for c in chroma_client.list_collections()]
        if not keys:
            click.echo("[build-index] Aucune collection trouvée dans ChromaDB.")
            click.echo("[build-index] Lance d'abord : python manage.py ingest")
            sys.exit(1)
        click.echo(f"[build-index] Collections disponibles : {keys}")

    for key in keys:
        _build_for_method(key, index_type, chroma_client)

    click.echo("\n[build-index] Terminé.")


# ═══════════════════════════════════════════════════════════════════════════════
# MAINTENANCE
# ═══════════════════════════════════════════════════════════════════════════════

@cli.command()
@click.option("--method", default=None,
              help="Méthode à vérifier (défaut : toutes)")
@click.option("--details", is_flag=True, default=False,
              help="Détail de chaque problème d'embedding / fingerprint")
@click.option("--metadata", is_flag=True, default=False,
              help="Tracks avec métadonnées manquantes ou partielles")
@click.option("--purge", is_flag=True, default=False,
              help="Supprime les tracks problématiques")
@click.option("--purge-missing-fp", is_flag=True, default=False,
              help="Purge uniquement les tracks sans fingerprint")
@click.option("--yes", "-y", is_flag=True, default=False,
              help="Pas de confirmation avant de purger")
def check(
    method: str | None,
    details: bool,
    metadata: bool,
    purge: bool,
    purge_missing_fp: bool,
    yes: bool,
) -> None:
    """Vérifie la cohérence des données (ChromaDB / FAISS / fingerprints / metadata)."""
    from src.maintenance.check import run_check
    run_check(
        method=method,
        details=details,
        metadata=metadata,
        purge=purge,
        purge_missing_fp=purge_missing_fp,
        yes=yes,
    )


@cli.command()
@click.option("--force", is_flag=True, default=False,
              help="Met à jour tous les tracks, même ceux déjà enrichis")
@click.option("--only-missing", is_flag=True, default=False,
              help="Ne traite que les tracks avec au moins un champ vide")
def enrich(force: bool, only_missing: bool) -> None:
    """Enrichit metadata.parquet via Deezer + MusicBrainz."""
    from src.maintenance.enrich import run_enrich
    run_enrich(force=force, only_missing=only_missing)


@cli.command()
@click.option("--yes", "-y", is_flag=True, default=False,
              help="Supprimer sans confirmation")
def clean(yes: bool) -> None:
    """Supprime toutes les données (ChromaDB, FAISS, fingerprints, metadata)."""
    from src.maintenance.clean import run_clean
    run_clean(yes=yes)


@cli.command("delete-rir")
@click.option("--method", required=True,
              type=click.Choice(["mfcc", "clap", "muq", "mert"]),
              help="Méthode d'embedding dont supprimer les RIRs")
@click.option("--dry-run", is_flag=True, default=False,
              help="Simule sans rien supprimer")
@click.option("--yes", "-y", is_flag=True, default=False,
              help="Confirme sans demander")
def delete_rir(method: str, dry_run: bool, yes: bool) -> None:
    """Supprime tous les vecteurs RIR d'une méthode dans ChromaDB + metadata."""
    from src.maintenance.delete_rir import run_delete_rir
    run_delete_rir(method=method, dry_run=dry_run, yes=yes)


# ═══════════════════════════════════════════════════════════════════════════════
# ÉVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

@cli.command("identify")
@click.argument("audio", type=click.Path(exists=True))
@click.option("--method", default=None,
              help="mfcc / clap / muq / mert (défaut : config.py)")
@click.option("--top", default=5, show_default=True,
              help="Nombre de résultats à afficher")
@click.option("--detailed", is_flag=True, default=False,
              help="Afficher les scores FAISS et fingerprint séparément")
def identify(audio: str, method: str | None, top: int, detailed: bool) -> None:
    """Identifie le morceau correspondant à un fichier audio.

    \b
    Exemples :
      python manage.py identify data/raw/mon_audio.mp3
      python manage.py identify data/raw/mon_audio.mp3 --method clap --top 10
      python manage.py identify data/raw/mon_audio.mp3 --detailed
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK",      "1")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    from src.api.app import run_identify_cli

    run_identify_cli(audio, method=method, top=top, detailed=detailed)


@cli.command("find-track")
@click.argument("audio")
@click.option("--target", "target_track_id", default=None,
              help="track_id cible à suivre (défaut : Flowers - Miley Cyrus)")
@click.option("--top", default=20, show_default=True,
              help="Nb de résultats affichés")
@click.option("--method", default=None,
              help="mfcc / clap / muq / mert (défaut : config.py)")
def find_track(audio: str, target_track_id: str | None, top: int, method: str | None) -> None:
    """Pipeline complet sur un fichier audio — position du track cible (Stage 1 + Stage 2)."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK",      "1")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    from src.evaluation.find_track import run_find_track, FLOWERS_ID
    run_find_track(
        audio=audio,
        target_track_id=target_track_id or FLOWERS_ID,
        top=top,
        method=method,
    )


@cli.command()
@click.option("--label", default="baseline", show_default=True,
              help="Nom du run (ex : mfcc_v2, clap_baseline)")
@click.option("--full", is_flag=True, default=False,
              help="14 cas complets (défaut : 5 cas rapides)")
@click.option("--method", default=None,
              type=click.Choice(["mfcc", "clap", "muq", "mert"]),
              help="Méthode d'embedding (défaut : config.EMBEDDING_METHOD)")
@click.option("--compare", "json_paths", multiple=True, metavar="JSON",
              help="Comparer plusieurs fichiers JSON de résultats (répétable)")
def benchmark(label: str, full: bool, method: str | None,
              json_paths: tuple[str, ...]) -> None:
    """Benchmark de robustesse sur Flowers - Miley Cyrus (une méthode à la fois).

    \b
    Workflow comparaison multi-méthodes :
      python manage.py benchmark --method mfcc --full --label mfcc
      python manage.py benchmark --method clap --full --label clap
      python manage.py plots --benchmark results/benchmark/*_mfcc.json \\
                             --benchmark results/benchmark/*_clap.json
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    from src.evaluation.benchmark import run_benchmark, run_compare
    if json_paths:
        run_compare(list(json_paths))
    else:
        run_benchmark(label_run=label, full=full, method=method)


@cli.command()
@click.option("--methods", multiple=True, default=["mfcc", "clap"],
              type=click.Choice(["mfcc", "clap", "muq", "mert"]),
              show_default=True,
              help="Méthodes à évaluer (répétable)")
@click.option("--conditions", multiple=True,
              default=["clean", "snr_20", "snr_10", "reverb", "combo"],
              show_default=True,
              help="Conditions de dégradation à tester (répétable)")
@click.option("--n-tracks", default=0, show_default=True,
              help="Limiter à N tracks du manifest (0 = tous)")
@click.option("--no-plot", is_flag=True, default=False,
              help="Ne pas générer les graphiques automatiquement")
def evaluate(methods: tuple, conditions: tuple, n_tracks: int,
             no_plot: bool) -> None:
    """Évaluation comparative multi-tracks multi-méthodes avec graphiques.

    \b
    Prérequis : alimenter le manifest avec des clips de test :
      python manage.py download-audio "Artiste Titre" --duration 30 --position middle
      python manage.py download-audio "Autre Artiste" --duration 30 --position middle

    \b
    Génère :
      results/eval/eval_TIMESTAMP.json  — métriques Top-1/Top-5/MRR/latence
      results/plots/top1_top5.png       — graphique de comparaison
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    from src.evaluation.evaluate import run_evaluate
    run_evaluate(
        methods=list(methods),
        conditions=list(conditions),
        n_tracks=n_tracks,
        plot=not no_plot,
    )


@cli.command()
@click.option("--eval", "eval_paths", multiple=True, metavar="JSON",
              help="JSON(s) d'évaluation multi-tracks — results/eval/eval_*.json")
@click.option("--rir-eval", "rir_eval_paths", multiple=True, metavar="JSON",
              help="JSON(s) d'évaluation RIR — results/eval/rir_eval_*.json")
@click.option("--out-dir", default=None,
              help="Dossier de sortie (défaut : results/plots/)")
def plots(eval_paths: tuple, rir_eval_paths: tuple,
          out_dir: str | None) -> None:
    """Génère tous les graphiques pour le rapport depuis des JSON existants.

    \b
    Graphiques depuis --rir-eval (comparaison RIR) :
      rir_paired_bar_*.png    G1 — accuracy avec vs sans RIR par condition
      rir_delta_*.png         G2 — gain Δ apporté par les RIR
      rir_faiss_scores_*.png  G4 — score FAISS par morceau avec/sans RIR

    \b
    Graphiques depuis --eval (pipeline complet) :
      method_accuracy.png     G6  — accuracy méthode × condition (avec écart-type)
      stage_comparison.png    G9  — Stage 1 (FAISS) vs Stage 2 (+ fingerprint)
      duration_impact.png     G11 — accuracy vs durée de l'extrait
      heatmap_accuracy.png    G12 — heatmap méthodes × conditions (% colormap)

    \b
    Workflow complet :
      python manage.py evaluate     --methods mfcc clap
      python manage.py rir-evaluate --methods clap
      python manage.py plots \\
          --eval     results/eval/eval_*.json \\
          --rir-eval results/eval/rir_eval_*.json
    """
    from src.evaluation.plots import run_plots
    run_plots(
        eval_jsons=list(eval_paths)     or None,
        rir_eval_jsons=list(rir_eval_paths) or None,
        out_dir=ROOT / out_dir if out_dir else None,
    )


@cli.command("rir-evaluate")
@click.option("--methods", multiple=True, default=None,
              type=click.Choice(["mfcc", "clap", "muq", "mert"]),
              help="Méthodes à évaluer (défaut : config.EMBEDDING_METHOD)")
@click.option("--conditions", multiple=True,
              default=["clean", "snr_20", "snr_10", "reverb", "combo"],
              show_default=True,
              help="Conditions de dégradation (répétable)")
@click.option("--n-tracks", default=0, show_default=True,
              help="Limiter à N tracks du manifest (0 = tous)")
@click.option("--no-plot", is_flag=True, default=False,
              help="Ne pas générer les graphiques automatiquement")
def rir_evaluate(methods: tuple, conditions: tuple, n_tracks: int,
                 no_plot: bool) -> None:
    """Compare Stage 1 FAISS avec vs sans vecteurs RIR sur plusieurs morceaux.

    \b
    Construit un index temporaire sans RIR en mémoire (ne modifie pas la base).
    Pour chaque track du manifest × condition × méthode :
      - Score et rang AVEC les vecteurs RIR (index normal)
      - Score et rang SANS les vecteurs RIR (index temporaire)

    \b
    Produit :
      results/eval/rir_eval_TIMESTAMP.json
      results/plots/rir_paired_bar_*.png  (G1)
      results/plots/rir_delta_*.png       (G2)
      results/plots/rir_faiss_scores_*.png (G4)

    \b
    Prérequis : clips dans manifest (manage.py download-audio "..." --duration 30)
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    from src.evaluation.evaluate import run_rir_evaluate
    run_rir_evaluate(
        methods=list(methods) if methods else None,
        conditions=list(conditions),
        n_tracks=n_tracks,
        plot=not no_plot,
    )


@cli.command("rir-impact")
@click.argument("audio", default=None, required=False)
@click.option("--method", default=None,
              help="mfcc / clap / muq / mert (défaut : config.py)")
@click.option("--top", default=20, show_default=True,
              help="Nb de résultats dans les tableaux")
def rir_impact(audio: str | None, method: str | None, top: int) -> None:
    """Mesure l'impact des RIR sur le score FAISS (sans modifier la base)."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK",      "1")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    from src.evaluation.rir_impact import run_rir_impact, AUDIO_DEFAULT, FLOWERS_ID
    run_rir_impact(
        audio=audio or AUDIO_DEFAULT,
        target_track_id=FLOWERS_ID,
        top=top,
        method=method,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════════════

_POSITIONS = ["start", "first-quarter", "middle", "third-quarter", "end"]


@cli.command("download-audio")
@click.argument("query", nargs=-1, required=True)
@click.option("--duration",
              type=click.Choice(["5", "10", "15", "30"]), default=None,
              help="Durée de l'extrait en secondes. Absent = morceau entier.")
@click.option("--position",
              type=click.Choice(_POSITIONS), default="start", show_default=True,
              help="Position de départ : start / first-quarter / middle / third-quarter / end")
def download_audio(query: tuple[str, ...], duration: str | None, position: str) -> None:
    """Télécharge un morceau depuis YouTube dans data/raw/ (optionnellement découpé)."""
    import json
    import subprocess
    import tempfile

    raw_dir    = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    search_query = " ".join(query)
    duration_s   = int(duration) if duration else None

    click.echo(f"Recherche : {search_query}")

    # Résolution de l'URL via ytsearch
    resolve = subprocess.run(
        ["yt-dlp", "--get-id", "--no-playlist", f"ytsearch1:{search_query}"],
        capture_output=True, text=True,
    )
    if resolve.returncode != 0 or not resolve.stdout.strip():
        click.echo("Erreur : aucune vidéo trouvée sur YouTube.", err=True)
        sys.exit(1)

    video_id  = resolve.stdout.strip()
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    click.echo(f"Vidéo    : {video_url}")

    # Téléchargement complet (sans découpe)
    if duration_s is None:
        result = subprocess.run([
            "yt-dlp", video_url,
            "--extract-audio", "--audio-format", "mp3", "--audio-quality", "5",
            "--output", str(raw_dir / "%(title)s.%(ext)s"),
            "--socket-timeout", "30",
        ])
        if result.returncode != 0:
            click.echo(f"Erreur yt-dlp (code {result.returncode})", err=True)
            sys.exit(result.returncode)

    # Téléchargement + découpe d'extrait
    else:
        click.echo("Récupération des métadonnées...")
        meta = subprocess.run(
            ["yt-dlp", "--dump-json", "--no-playlist", video_url],
            capture_output=True, text=True,
        )
        total_duration: float | None = None
        if meta.returncode == 0:
            try:
                total_duration = float(json.loads(meta.stdout).get("duration", 0)) or None
            except (json.JSONDecodeError, ValueError):
                pass

        if total_duration is None:
            click.echo("Impossible de récupérer la durée — départ à 0s.", err=True)
            start_s = 0.0
        else:
            if position == "start":
                start_s = 0.0
            elif position == "first-quarter":
                start_s = total_duration * 0.25
            elif position == "middle":
                start_s = max(0.0, total_duration / 2 - duration_s / 2)
            elif position == "third-quarter":
                start_s = total_duration * 0.75
            else:  # end
                start_s = max(0.0, total_duration - duration_s)
            max_start = max(0.0, total_duration - duration_s)
            start_s   = min(start_s, max_start)
            click.echo(
                f"Durée totale : {total_duration:.0f}s  |  "
                f"Extrait : {duration_s}s à {position} (départ à {start_s:.1f}s)"
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dl = subprocess.run([
                "yt-dlp", video_url,
                "--extract-audio", "--audio-format", "mp3", "--audio-quality", "5",
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
            out_path = raw_dir / f"{src.stem}__{position}_{duration_s}s.mp3"
            cut      = subprocess.run([
                "ffmpeg", "-y", "-ss", str(start_s), "-i", str(src),
                "-t", str(duration_s), "-acodec", "copy", str(out_path),
            ], capture_output=True)
            if cut.returncode != 0:
                click.echo("Erreur ffmpeg lors de la découpe.", err=True)
                click.echo(cut.stderr.decode(errors="ignore"), err=True)
                sys.exit(cut.returncode)

    files = sorted(raw_dir.glob("*.mp3"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        return
    downloaded_file = files[0]
    click.echo(f"\n✅  Fichier téléchargé : {downloaded_file}")

    # ── Mise à jour du manifest (ground truth pour evaluate) ──
    try:
        import pandas as pd
        from src.evaluation.evaluate import find_track_id_by_query

        track_id = find_track_id_by_query(search_query)
        if track_id:
            # Lire le nom artiste/titre depuis metadata
            meta_path = ROOT / "data" / "processed" / "metadata.parquet"
            artist, title = "", ""
            if meta_path.exists():
                df = pd.read_parquet(meta_path, columns=["track_id", "artist", "title"])
                row = df[df["track_id"] == track_id]
                if not row.empty:
                    artist = row.iloc[0]["artist"]
                    title  = row.iloc[0]["title"]

            manifest_path = raw_dir / "manifest.json"
            existing: list = []
            if manifest_path.exists():
                with open(manifest_path, encoding="utf-8") as mf:
                    existing = json.load(mf)

            # Éviter les doublons (même filename)
            entry = {
                "filename":   downloaded_file.name,
                "track_id":   track_id,
                "artist":     artist,
                "title":      title,
                "position":   position,
                "duration_s": int(duration) if duration else None,
            }
            existing = [e for e in existing if e.get("filename") != entry["filename"]]
            existing.append(entry)

            with open(manifest_path, "w", encoding="utf-8") as mf:
                json.dump(existing, mf, ensure_ascii=False, indent=2)

            click.echo(f"📋  Manifest mis à jour : {artist} — {title} ({track_id[:8]}...)")
        else:
            click.echo(
                "⚠   Track non trouvé dans la base — manifest non mis à jour.\n"
                "    Ingérez ce morceau d'abord : python manage.py ingest"
            )
    except Exception as e:
        click.echo(f"⚠   Manifest non mis à jour : {e}", err=True)


# ═══════════════════════════════════════════════════════════════════════════════
# INTERFACE WEB
# ═══════════════════════════════════════════════════════════════════════════════

@cli.command("start-webapp")
@click.option("--prod", is_flag=True, default=False,
              help="Mode production : build le frontend, tout via FastAPI sur :8000")
@click.option("--port", default=8000, show_default=True,
              help="Port du backend FastAPI")
def start_webapp(prod: bool, port: int) -> None:
    """Démarre backend FastAPI + frontend React/Vite."""
    import signal
    import subprocess
    import time

    frontend_dir = ROOT / "webapp" / "frontend"

    # Vérifier npm
    try:
        subprocess.run(["npm", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        click.echo("❌  npm introuvable. Installe Node.js : https://nodejs.org/")
        sys.exit(1)

    # npm install si besoin
    if not (frontend_dir / "node_modules").exists():
        click.echo("📦  Installation des dépendances frontend (npm install)...")
        result = subprocess.run(["npm", "install"], cwd=frontend_dir)
        if result.returncode != 0:
            click.echo("❌  Échec de npm install.")
            sys.exit(1)
        click.echo("✅  Dépendances installées.\n")

    processes: list[subprocess.Popen] = []

    def _cleanup(*_):
        click.echo("\n⏹   Arrêt des serveurs...")
        for p in processes:
            try:
                p.terminate()
            except Exception:
                pass
        time.sleep(0.5)
        for p in processes:
            try:
                if p.poll() is None:
                    p.kill()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT,  _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    if prod:
        click.echo("🔨  Build du frontend React...")
        result = subprocess.run(["npm", "run", "build"], cwd=frontend_dir)
        if result.returncode != 0:
            click.echo("❌  Échec du build frontend.")
            sys.exit(1)
        click.echo("✅  Build terminé.\n")
        click.echo(f"🚀  Démarrage du serveur sur http://localhost:{port} …")
        backend = subprocess.Popen(
            [sys.executable, "-m", "uvicorn",
             "webapp.backend.server:app",
             "--host", "0.0.0.0", "--port", str(port)],
            cwd=ROOT,
        )
        processes.append(backend)
        click.echo(f"\n✅  Interface disponible → http://localhost:{port}")
    else:
        click.echo(f"🚀  Démarrage du backend FastAPI sur http://localhost:{port} …")
        backend = subprocess.Popen(
            [sys.executable, "-m", "uvicorn",
             "webapp.backend.server:app",
             "--host", "0.0.0.0", "--port", str(port), "--reload"],
            cwd=ROOT,
        )
        processes.append(backend)
        time.sleep(1)
        click.echo("⚡  Démarrage du frontend Vite (hot-reload)…")
        frontend = subprocess.Popen(["npm", "run", "dev"], cwd=frontend_dir)
        processes.append(frontend)
        click.echo(f"\n✅  Interface disponible → http://localhost:5173")
        click.echo(f"   Backend API         → http://localhost:{port}")

    click.echo("   Ctrl+C pour arrêter\n")

    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        _cleanup()


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cli()
