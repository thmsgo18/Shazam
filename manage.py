#!/usr/bin/env python3
"""
manage.py — Point d'entrée unique du projet Shazam Maison.

Usage :
    python manage.py <commande> [options]

Commandes disponibles :
    ── Construction ───────────────────────────────────────────────────────────
    build                  Pipeline complet : ingest → augment → enrich
    ingest                 Télécharge et ingère des tracks depuis des CSV Kaggle
    augment                Augmente les embeddings avec des RIR
    enrich                 Enrichit metadata.parquet via Deezer + MusicBrainz

    ── Maintenance ────────────────────────────────────────────────────────────
    check                  Vérifie la cohérence des données
    rebuild                Recalcule les fingerprints et/ou l'index FAISS
    clean                  Supprime des données (track, RIR, ou tout)

    ── Utilisation ────────────────────────────────────────────────────────────
    config                 Affiche la configuration active (src/config.py)
    identify               Identifie un fichier audio
    download-test          Télécharge un clip audio de test depuis YouTube

    ── Évaluation ─────────────────────────────────────────────────────────────
    eval benchmark         Benchmark de robustesse sur un fichier audio
    eval multi             Évaluation multi-tracks multi-conditions
    eval rir               Analyse l'impact des RIR (single-track ou multi)
    eval plots             Génère les graphiques depuis des JSON existants

    ── Interface web ───────────────────────────────────────────────────────────
    webapp                 Démarre backend FastAPI + frontend React/Vite
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

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
# CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

@cli.command()
@click.option("--csv", "csv_paths", multiple=True,
              help="Chemin(s) vers un CSV Kaggle (répétable). Défaut : data/kaggle/")
@click.option("--skip-rir", is_flag=True, default=False,
              help="Ne pas lancer l'augmentation RIR après l'ingestion")
@click.option("--skip-enrich", is_flag=True, default=False,
              help="Ne pas lancer l'enrichissement des métadonnées")
def build(csv_paths: tuple[str, ...], skip_rir: bool, skip_enrich: bool) -> None:
    """Pipeline complet : ingest → augment → enrich (point d'entrée recommandé).

    \b
    Tous les paramètres (méthode, RIR, workers…) sont lus depuis src/config.py.

    \b
    Exemples :
      python manage.py build --csv data/kaggle/data/spotify-streaming-top-50-world.csv
      python manage.py build --csv data/kaggle/data/spotify-streaming-top-50-world.csv --skip-rir
      python manage.py build  # tous les CSV dans data/kaggle/
    """
    from src.ingestion.ingest import run_ingest

    click.echo("\n[build] ── Étape 1/3 : Ingestion ─────────────────────────────")
    run_ingest(list(csv_paths) if csv_paths else None)

    if not skip_rir:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK",      "1")
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
        from src.ingestion.augment_rir import run_augment
        click.echo("\n[build] ── Étape 2/3 : Augmentation RIR ──────────────────────")
        run_augment()
    else:
        click.echo("\n[build] ── Étape 2/3 : Augmentation RIR (ignorée) ────────────")

    if not skip_enrich:
        from src.maintenance.enrich import run_enrich
        click.echo("\n[build] ── Étape 3/3 : Enrichissement métadonnées ────────────")
        run_enrich()
    else:
        click.echo("\n[build] ── Étape 3/3 : Enrichissement (ignoré) ───────────────")

    click.echo("\n[build] Terminé.")


@cli.command()
@click.option("--csv", "csv_paths", multiple=True,
              help="Chemin(s) vers un CSV Kaggle (répétable). Défaut : data/kaggle/")
def ingest(csv_paths: tuple[str, ...]) -> None:
    """Télécharge et ingère des tracks depuis des CSV Kaggle.

    \b
    Reprise automatique : les tracks déjà traités sont ignorés.
    La méthode d'embedding est lue depuis src/config.py (EMBEDDING_METHOD).
    """
    from src.ingestion.ingest import run_ingest
    run_ingest(list(csv_paths) if csv_paths else None)


@cli.command()
def augment() -> None:
    """Augmente les embeddings avec des RIR (Room Impulse Response).

    \b
    Tous les paramètres sont lus depuis src/config.py :
      RIR_SOURCE  — synthetic (générées) ou mit (WAV réels)
      RIR_N       — nombre de RIRs par track
      RIR_MIT_DIR — dossier WAV MIT (si RIR_SOURCE=mit)
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK",      "1")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    from src.ingestion.augment_rir import run_augment
    run_augment()


@cli.command()
@click.option("--force", is_flag=True, default=False,
              help="Met à jour tous les tracks, même ceux déjà enrichis")
@click.option("--only-missing", is_flag=True, default=False,
              help="Ne traite que les tracks avec au moins un champ vide")
def enrich(force: bool, only_missing: bool) -> None:
    """Enrichit metadata.parquet via Deezer + MusicBrainz.

    \b
    Champs enrichis : album, genre, release_date, cover_url.
    Sans option : enrichit uniquement les tracks non encore traités.
    """
    from src.maintenance.enrich import run_enrich
    run_enrich(force=force, only_missing=only_missing)


# ═══════════════════════════════════════════════════════════════════════════════
# MAINTENANCE
# ═══════════════════════════════════════════════════════════════════════════════

@cli.command()
@click.option("--details", is_flag=True, default=False,
              help="Détail de chaque problème par catégorie (codes C1-C7, Q1-Q4, FP)")
@click.option("--metadata", is_flag=True, default=False,
              help="Liste les tracks avec métadonnées manquantes ou partielles")
@click.option("--purge", is_flag=True, default=False,
              help="Supprime les tracks problématiques de tous les stores")
@click.option("--purge-missing-fp", is_flag=True, default=False,
              help="Purge uniquement les tracks sans fingerprint")
@click.option("--yes", "-y", is_flag=True, default=False,
              help="Pas de confirmation avant de purger")
def check(details: bool, metadata: bool, purge: bool, purge_missing_fp: bool, yes: bool) -> None:
    """Vérifie la cohérence des données (ChromaDB / FAISS / fingerprints / metadata).

    \b
    Codes de vérification (--details) :
      C1  Dimension d'embedding inattendue
      C2  NaN ou Inf dans les embeddings
      C3  Désynchronisation ChromaDB ↔ metadata
      C5  Index FAISS désynchronisé
      C6  Segments orphelins
      C7  Embedding incomplet (< 80 % des segments attendus)
      Q3  Fingerprint vide
      FP  Track sans fingerprint
    """
    from src.maintenance.check import run_check
    run_check(
        method=None,
        details=details,
        metadata=metadata,
        purge=purge,
        purge_missing_fp=purge_missing_fp,
        yes=yes,
    )


@cli.command()
@click.option("--what", type=click.Choice(["index", "fps", "all"]), default="all",
              show_default=True,
              help="index = FAISS uniquement · fps = fingerprints uniquement · all = les deux")
@click.option("--force", is_flag=True, default=False,
              help="Recalcule même les tracks déjà présents dans SQLite (fingerprints uniquement)")
def rebuild(what: str, force: bool) -> None:
    """Recalcule les fingerprints et/ou reconstruit l'index FAISS.

    \b
    Exemples :
      python manage.py rebuild               # recalcule tout
      python manage.py rebuild --what index  # FAISS seulement (après check --purge)
      python manage.py rebuild --what fps --force  # fingerprints, tout recalculer
    """
    if what in ("fps", "all"):
        from src.ingestion.fingerprints import run_rebuild_fingerprints
        click.echo("[rebuild] Recalcul des fingerprints...")
        run_rebuild_fingerprints(force_all=force, limit=None, workers=4, dry_run=False)
        click.echo("[rebuild] Fingerprints terminés.")

    if what in ("index", "all"):
        import chromadb
        from src import config
        from src.index.build_index import _build_for_method
        click.echo("[rebuild] Reconstruction de l'index FAISS...")
        index_type    = getattr(config, "INDEX_TYPE", "flat")
        chroma_client = chromadb.PersistentClient(path=str(ROOT / config.CHROMA_DIR))
        keys = [c.name for c in chroma_client.list_collections()]
        if not keys:
            click.echo("[rebuild] Aucune collection ChromaDB trouvée.")
            click.echo("[rebuild] Lance d'abord : python manage.py ingest")
            sys.exit(1)
        for key in keys:
            _build_for_method(key, index_type, chroma_client)
        click.echo("[rebuild] Index FAISS terminé.")


@cli.command()
@click.option("--track", "track_id", default=None, metavar="TRACK_ID",
              help="Supprime un track spécifique de tous les stores")
@click.option("--rir", is_flag=True, default=False,
              help="Supprime les segments RIR de la méthode active (EMBEDDING_METHOD dans config.py)")
@click.option("--all", "all_data", is_flag=True, default=False,
              help="Réinitialisation complète — supprime tout (ChromaDB, FAISS, SQLite, metadata)")
@click.option("--yes", "-y", is_flag=True, default=False,
              help="Pas de confirmation")
def clean(track_id: str | None, rir: bool, all_data: bool, yes: bool) -> None:
    """Supprime des données : un track, les segments RIR, ou toute la base.

    \b
    Exemples :
      python manage.py clean --track f01ab00f1fdc5a57fd2676f4d68631a8
      python manage.py clean --rir
      python manage.py clean --all --yes
    """
    if not track_id and not rir and not all_data:
        click.echo("Précise ce que tu veux supprimer :")
        click.echo("  --track TRACK_ID   Supprimer un track spécifique")
        click.echo("  --rir              Supprimer les segments RIR (méthode active)")
        click.echo("  --all              Tout supprimer (réinitialisation complète)")
        sys.exit(0)

    if track_id:
        from src.maintenance.clean import run_clean_track
        run_clean_track(track_id=track_id, yes=yes)

    if rir:
        from src.maintenance.delete_rir import run_delete_rir
        from src import config
        run_delete_rir(method=config.EMBEDDING_METHOD, dry_run=False, yes=yes)

    if all_data:
        from src.maintenance.clean import run_clean
        run_clean(yes=yes)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILISATION
# ═══════════════════════════════════════════════════════════════════════════════

@cli.command("config")
def show_config() -> None:
    """Affiche la configuration active (src/config.py)."""
    from src import config

    click.echo()
    click.echo("── Embedding ───────────────────────────────────────────────────")
    click.echo(f"  EMBEDDING_METHOD         : {config.EMBEDDING_METHOD}")
    if config.EMBEDDING_METHOD == "clap":
        click.echo(f"  CLAP_MODEL_NAME          : {config.CLAP_MODEL_NAME}")
        click.echo(f"  CLAP_SAMPLE_RATE         : {config.CLAP_SAMPLE_RATE} Hz")
    elif config.EMBEDDING_METHOD == "muq":
        click.echo(f"  MUQ_MODEL_NAME           : {config.MUQ_MODEL_NAME}")
        click.echo(f"  MUQ_SAMPLE_RATE          : {config.MUQ_SAMPLE_RATE} Hz")
    elif config.EMBEDDING_METHOD == "mert":
        click.echo(f"  MERT_MODEL_NAME          : {config.MERT_MODEL_NAME}")
        click.echo(f"  MERT_SAMPLE_RATE         : {config.MERT_SAMPLE_RATE} Hz")
    else:
        click.echo(f"  SAMPLE_RATE              : {config.SAMPLE_RATE} Hz")

    click.echo()
    click.echo("── Index & Recherche vectorielle ───────────────────────────────")
    click.echo(f"  INDEX_TYPE               : {config.INDEX_TYPE}")
    click.echo(f"  VECTOR_TOP_K_SEGMENTS    : {config.VECTOR_TOP_K_SEGMENTS}  (voisins FAISS par segment requête)")
    click.echo(f"  VECTOR_TOP_N_TRACKS      : {config.VECTOR_TOP_N_TRACKS}   (candidats Stage 1 → Stage 2)")
    click.echo(f"  VECTOR_TOP_N_RESULTS     : {config.VECTOR_TOP_N_RESULTS}  (résultats finaux retournés)")

    click.echo()
    click.echo("── Segmentation ────────────────────────────────────────────────")
    click.echo(f"  SEGMENT_WIN_S            : {config.SEGMENT_WIN_S} s")
    click.echo(f"  SEGMENT_HOP_S            : {config.SEGMENT_HOP_S} s")

    click.echo()
    click.echo("── Augmentation RIR ────────────────────────────────────────────")
    click.echo(f"  RIR_SOURCE               : {config.RIR_SOURCE}")
    click.echo(f"  RIR_N                    : {config.RIR_N}")
    if config.RIR_SOURCE == "mit":
        click.echo(f"  RIR_MIT_DIR              : {config.RIR_MIT_DIR}")

    click.echo()
    click.echo("── Interface web ───────────────────────────────────────────────")
    click.echo(f"  UI_LISTEN_DURATION       : {config.UI_LISTEN_DURATION} s")

    click.echo()
    click.echo("── Optimisations ───────────────────────────────────────────────")
    click.echo(f"  OPT_FLOAT16              : {config.OPT_FLOAT16}")
    click.echo(f"  OPT_BATCH_EMBED          : {config.OPT_BATCH_EMBED}")
    click.echo(f"  OPT_QUERY_DENOISE        : {config.OPT_QUERY_DENOISE}")
    click.echo(f"  OPT_FINGERPRINT_PARALLEL : {config.OPT_FINGERPRINT_PARALLEL}")
    click.echo()


@cli.command()
@click.argument("audio", type=click.Path(exists=True))
@click.option("--top", default=5, show_default=True,
              help="Nombre de résultats à afficher")
@click.option("--detailed", is_flag=True, default=False,
              help="Afficher les scores FAISS et fingerprint séparément")
@click.option("--target", "target_track_id", default=None, metavar="TRACK_ID",
              help="track_id attendu — active le mode évaluation et affiche le rang du track cible")
def identify(audio: str, top: int, detailed: bool, target_track_id: str | None) -> None:
    """Identifie le morceau correspondant à un fichier audio.

    \b
    Sans --target : retourne les N meilleurs résultats.
    Avec --target : mode évaluation — affiche aussi le rang du track attendu.

    \b
    Exemples :
      python manage.py identify data/raw/mon_audio.mp3
      python manage.py identify data/raw/mon_audio.mp3 --top 10 --detailed
      python manage.py identify data/raw/mon_audio.mp3 --target f01ab00f
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK",      "1")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    if target_track_id:
        from src.evaluation.find_track import run_find_track
        run_find_track(audio=audio, target_track_id=target_track_id, top=top, method=None)
    else:
        from src.api.app import run_identify_cli
        run_identify_cli(audio, method=None, top=top, detailed=detailed)


_POSITIONS = ["start", "first-quarter", "middle", "third-quarter", "end"]


@cli.command("download-test")
@click.argument("query", nargs=-1, required=True)
@click.option("--duration", type=click.Choice(["5", "10", "15", "30"]), default=None,
              help="Durée de l'extrait en secondes. Absent = morceau entier.")
@click.option("--position", type=click.Choice(_POSITIONS), default="start", show_default=True,
              help="Position dans le morceau : start / first-quarter / middle / third-quarter / end")
def download_test(query: tuple[str, ...], duration: str | None, position: str) -> None:
    """Télécharge un clip audio de test depuis YouTube dans data/raw/.

    \b
    Le clip est automatiquement ajouté au manifest de test
    (data/raw/manifest.json) utilisé par eval benchmark, eval multi et eval rir.

    \b
    Exemples :
      python manage.py download-test "Miley Cyrus Flowers" --duration 30 --position middle
      python manage.py download-test "Daft Punk Get Lucky" --duration 15 --position middle
      python manage.py download-test "The Weeknd Blinding Lights" --duration 5 --position middle
    """
    import json
    import subprocess
    import tempfile

    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    search_query = " ".join(query)
    duration_s   = int(duration) if duration else None

    click.echo(f"Recherche : {search_query}")

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
            else:
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
    click.echo(f"\n  Fichier téléchargé : {downloaded_file.name}")

    # Mise à jour du manifest (ground truth pour eval)
    try:
        import pandas as pd
        from src.evaluation.evaluate import find_track_id_by_query

        track_id = find_track_id_by_query(search_query)
        if track_id:
            meta_path = ROOT / "data" / "processed" / "metadata.parquet"
            artist, title = "", ""
            if meta_path.exists():
                df  = pd.read_parquet(meta_path, columns=["track_id", "artist", "title"])
                row = df[df["track_id"] == track_id]
                if not row.empty:
                    artist = row.iloc[0]["artist"]
                    title  = row.iloc[0]["title"]

            manifest_path = raw_dir / "manifest.json"
            existing: list = []
            if manifest_path.exists():
                with open(manifest_path, encoding="utf-8") as mf:
                    existing = json.load(mf)

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

            click.echo(f"  Manifest mis à jour : {artist} — {title} ({track_id[:8]}...)")
        else:
            click.echo(
                "  Track non trouvé dans la base — manifest non mis à jour.\n"
                "  Ingérez ce morceau d'abord : python manage.py ingest"
            )
    except Exception as e:
        click.echo(f"  Manifest non mis à jour : {e}", err=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ÉVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

@cli.group("eval")
def eval_group() -> None:
    """Évaluation du pipeline : benchmark, multi-tracks, impact RIR, graphiques."""
    pass


@eval_group.command("benchmark")
@click.argument("audio", required=False, default=None, type=click.Path())
@click.option("--label", default=None,
              help="Nom du run pour la traçabilité (défaut : horodatage)")
@click.option("--full", is_flag=True, default=False,
              help="Suite complète (~10 cas) — défaut : 4 cas rapides")
@click.option("--compare", "json_paths", multiple=True, metavar="JSON",
              help="Compare des runs précédents sans relancer le benchmark (répétable)")
def eval_benchmark(audio: str | None, label: str | None, full: bool,
                   json_paths: tuple) -> None:
    """Benchmark de robustesse sur un fichier audio de test.

    \b
    Le track cible est auto-détecté depuis data/raw/manifest.json.
    Utilise download-test pour enregistrer un clip dans le manifest.

    \b
    Exemples :
      python manage.py eval benchmark data/raw/mon_clip__middle_30s.mp3
      python manage.py eval benchmark data/raw/clip.mp3 --full --label "clap-v2"
      python manage.py eval benchmark --compare results/benchmark/run1.json \\
                                      --compare results/benchmark/run2.json
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    from src.evaluation.benchmark import run_benchmark, run_compare
    from datetime import datetime

    if json_paths:
        run_compare(list(json_paths))
    elif audio:
        effective_label = label or datetime.now().strftime("%Y%m%d_%H%M%S")
        run_benchmark(audio=audio, label_run=effective_label, full=full)
    else:
        click.echo("Erreur : fournir un fichier AUDIO ou des fichiers --compare.")
        click.echo("  python manage.py eval benchmark data/raw/mon_clip.mp3")
        click.echo("  python manage.py eval benchmark --compare results/benchmark/run1.json")
        sys.exit(1)


@eval_group.command("multi")
@click.option("--n-tracks", default=0, show_default=True,
              help="Limiter à N tracks du manifest (0 = tous)")
@click.option("--no-plot", is_flag=True, default=False,
              help="Ne pas générer les graphiques automatiquement")
def eval_multi(n_tracks: int, no_plot: bool) -> None:
    """Évaluation multi-tracks sur toutes les conditions de dégradation.

    \b
    Méthode active : lue depuis src/config.py (EMBEDDING_METHOD).
    Conditions testées : clean, snr_20, snr_10, reverb, combo.
    Prérequis : clips de test enregistrés via download-test.
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    from src import config
    from src.evaluation.evaluate import run_evaluate
    run_evaluate(
        methods=[config.EMBEDDING_METHOD],
        conditions=["clean", "snr_20", "snr_10", "reverb", "combo"],
        n_tracks=n_tracks,
        plot=not no_plot,
    )


@eval_group.command("rir")
@click.argument("audio", required=False, default=None)
@click.option("--target", "target_track_id", default=None, metavar="TRACK_ID",
              help="track_id cible (single-track). Auto-détecté depuis le manifest si absent.")
@click.option("--n-tracks", default=0, show_default=True,
              help="Limiter à N tracks du manifest (mode multi-tracks uniquement)")
@click.option("--no-plot", is_flag=True, default=False,
              help="Ne pas générer les graphiques (mode multi-tracks uniquement)")
def eval_rir(audio: str | None, target_track_id: str | None,
             n_tracks: int, no_plot: bool) -> None:
    """Analyse l'impact des RIR sur les scores FAISS.

    \b
    Avec AUDIO : analyse détaillée sur un seul fichier.
    Sans AUDIO : évaluation comparative multi-tracks depuis le manifest.

    \b
    Exemples :
      python manage.py eval rir data/raw/mon_clip.mp3
      python manage.py eval rir data/raw/mon_clip.mp3 --target f01ab00f
      python manage.py eval rir --n-tracks 10
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK",      "1")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

    if audio:
        from src.evaluation.rir_impact import run_rir_impact
        resolved_target = target_track_id or _lookup_track_id_from_manifest(audio)
        run_rir_impact(audio=audio, target_track_id=resolved_target, top=20, method=None)
    else:
        from src import config
        from src.evaluation.evaluate import run_rir_evaluate
        run_rir_evaluate(
            methods=[config.EMBEDDING_METHOD],
            conditions=["clean", "snr_20", "snr_10", "reverb", "combo"],
            n_tracks=n_tracks,
            plot=not no_plot,
        )


@eval_group.command("plots")
@click.option("--eval", "eval_paths", multiple=True, metavar="JSON",
              help="JSON(s) d'évaluation multi-tracks (results/eval/eval_*.json)")
@click.option("--rir-eval", "rir_eval_paths", multiple=True, metavar="JSON",
              help="JSON(s) d'évaluation RIR (results/eval/rir_eval_*.json)")
@click.option("--out-dir", default=None, metavar="PATH",
              help="Dossier de sortie (défaut : results/plots/)")
def eval_plots(eval_paths: tuple, rir_eval_paths: tuple, out_dir: str | None) -> None:
    """Génère les graphiques du rapport depuis des JSON existants.

    \b
    --rir-eval → G1 (paired bar), G2 (delta RIR), G4 (scores FAISS)
    --eval     → G6 (accuracy méthodes), G9 (Stage 1 vs 2), G11 (durée), G12 (heatmap)

    \b
    Exemple :
      python manage.py eval plots \\
        --eval     results/eval/eval_*.json \\
        --rir-eval results/eval/rir_eval_*.json
    """
    from src.evaluation.plots import run_plots
    run_plots(
        eval_jsons=list(eval_paths)         or None,
        rir_eval_jsons=list(rir_eval_paths) or None,
        out_dir=ROOT / out_dir if out_dir else None,
    )


def _lookup_track_id_from_manifest(audio_path: str) -> str | None:
    """Cherche le track_id d'un fichier audio dans data/raw/manifest.json."""
    import json
    manifest = ROOT / "data" / "raw" / "manifest.json"
    if not manifest.exists():
        return None
    filename = Path(audio_path).name
    try:
        with open(manifest, encoding="utf-8") as f:
            entries = json.load(f)
        for entry in entries:
            if entry.get("filename") == filename:
                return entry.get("track_id")
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# WEB
# ═══════════════════════════════════════════════════════════════════════════════

@cli.command()
@click.option("--prod", is_flag=True, default=False,
              help="Mode production : build frontend, tout via FastAPI sur le port défini")
@click.option("--port", default=8000, show_default=True,
              help="Port du backend FastAPI")
def webapp(prod: bool, port: int) -> None:
    """Démarre backend FastAPI + frontend React/Vite.

    \b
    Dev  : backend :8000 (reload) + frontend Vite :5173 (hot-reload)
    Prod : build statique + backend seul sur le port défini
    """
    import signal
    import subprocess
    import time

    frontend_dir = ROOT / "webapp" / "frontend"

    try:
        subprocess.run(["npm", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        click.echo("npm introuvable. Installe Node.js : https://nodejs.org/")
        sys.exit(1)

    if not (frontend_dir / "node_modules").exists():
        click.echo("Installation des dépendances frontend (npm install)...")
        result = subprocess.run(["npm", "install"], cwd=frontend_dir)
        if result.returncode != 0:
            click.echo("Echec de npm install.")
            sys.exit(1)
        click.echo("Dépendances installées.\n")

    processes: list[subprocess.Popen] = []

    def _cleanup(*_):
        click.echo("\nArrêt des serveurs...")
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
        click.echo("Build du frontend React...")
        result = subprocess.run(["npm", "run", "build"], cwd=frontend_dir)
        if result.returncode != 0:
            click.echo("Echec du build frontend.")
            sys.exit(1)
        click.echo(f"Démarrage sur http://localhost:{port} ...")
        backend = subprocess.Popen(
            [sys.executable, "-m", "uvicorn",
             "webapp.backend.server:app",
             "--host", "0.0.0.0", "--port", str(port)],
            cwd=ROOT,
        )
        processes.append(backend)
        click.echo(f"Interface disponible → http://localhost:{port}")
    else:
        click.echo(f"Démarrage du backend FastAPI sur http://localhost:{port} ...")
        backend = subprocess.Popen(
            [sys.executable, "-m", "uvicorn",
             "webapp.backend.server:app",
             "--host", "0.0.0.0", "--port", str(port), "--reload"],
            cwd=ROOT,
        )
        processes.append(backend)
        time.sleep(1)
        click.echo("Démarrage du frontend Vite (hot-reload)...")
        frontend = subprocess.Popen(["npm", "run", "dev"], cwd=frontend_dir)
        processes.append(frontend)
        click.echo(f"Interface disponible → http://localhost:5173")
        click.echo(f"Backend API         → http://localhost:{port}")

    click.echo("Ctrl+C pour arrêter\n")
    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        _cleanup()


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cli()
