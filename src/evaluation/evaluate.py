"""
src/evaluation/evaluate.py

Évaluation comparative du pipeline sur un ensemble de morceaux de test.

Principe :
  1. Lire data/raw/manifest.json — liste des fichiers de test avec leur track_id attendu.
     (Le manifest est alimenté automatiquement par `manage.py download-audio`.)
  2. Pour chaque fichier × méthode × condition de dégradation :
       appliquer la dégradation → fichier temporaire → identify_track() → métriques.
  3. Calculer Top-1, Top-5, MRR (Mean Reciprocal Rank) et latence moyenne.
  4. Sauvegarder un JSON dans results/eval/ + générer les graphiques.

Conditions de dégradation évaluées :
  clean   — fichier original sans modification
  snr_20  — bruit blanc gaussien à SNR 20 dB  (dégradation légère)
  snr_10  — bruit blanc gaussien à SNR 10 dB  (dégradation sévère)
  reverb  — reverb synthétique (convolution RIR)
  combo   — SNR 15 dB + reverb + passe-bande 300–7 kHz

Point d'entrée public : run_evaluate(methods, conditions, n_tracks, out_dir)
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
import time
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

ROOT         = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "data" / "raw" / "manifest.json"
METADATA_PATH = ROOT / "data" / "processed" / "metadata.parquet"
RESULTS_DIR   = ROOT / "results" / "eval"

# Réutilise les fonctions de dégradation déjà présentes dans benchmark.py
from src.evaluation.benchmark import (
    add_noise_at_snr,
    add_reverb,
    apply_bandpass,
    save_temp_wav,
)

# ─── Conditions de dégradation ──────────────────────────────────────────────

ALL_CONDITIONS = ["clean", "snr_20", "snr_10", "reverb", "combo"]

CONDITION_LABELS = {
    "clean":  "Clean",
    "snr_20": "Bruit SNR 20 dB",
    "snr_10": "Bruit SNR 10 dB",
    "reverb": "Reverb",
    "combo":  "Combo (15 dB+Rev+BP)",
}


def _apply_condition(waveform: np.ndarray, sr: int, condition: str) -> np.ndarray:
    """Applique une dégradation à un waveform selon la condition spécifiée."""
    np.random.seed(42)
    if condition == "clean":
        return waveform
    if condition == "snr_20":
        return add_noise_at_snr(waveform, 20)
    if condition == "snr_10":
        return add_noise_at_snr(waveform, 10)
    if condition == "reverb":
        return add_reverb(waveform, sr, decay=0.3)
    if condition == "combo":
        w = add_noise_at_snr(waveform, 15)
        w = add_reverb(w, sr, decay=0.3)
        w = apply_bandpass(w, sr, 300, 7000)
        return w
    raise ValueError(f"Condition inconnue : {condition}")


# ─── Manifest ───────────────────────────────────────────────────────────────

def load_manifest(path: Path = MANIFEST_PATH) -> list[dict]:
    """
    Charge le fichier manifest.json qui mappe filename → track_id.

    Format attendu :
    [
      {
        "filename":  "Miley Cyrus - Flowers__middle_30s.mp3",
        "track_id":  "f01ab00f1fdc5a57fd2676f4d68631a8",
        "artist":    "Miley Cyrus",
        "title":     "Flowers",
        "position":  "middle",
        "duration_s": 30
      },
      ...
    ]
    """
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Filtrer les entrées dont le fichier existe encore
    raw_dir = ROOT / "data" / "raw"
    return [e for e in data if (raw_dir / e["filename"]).exists()]


def find_track_id_by_query(query: str) -> str | None:
    """
    Cherche un track_id dans metadata.parquet par correspondance textuelle.

    Utilise la similarité de Jaccard sur les mots entre la requête
    et le champ 'artist + title' de chaque track.
    Retourne None si aucun match avec un score > 0.25.
    """
    if not METADATA_PATH.exists():
        return None
    df = pd.read_parquet(METADATA_PATH, columns=["track_id", "artist", "title"])

    def _words(text: str) -> set[str]:
        return set(re.sub(r"[^a-z0-9\s]", " ", text.lower()).split())

    query_words = _words(query)
    if not query_words:
        return None

    best_score, best_id = 0.0, None
    for row in df.itertuples():
        candidate = _words(f"{row.artist} {row.title}")
        if not candidate:
            continue
        jaccard = len(query_words & candidate) / len(query_words | candidate)
        if jaccard > best_score:
            best_score, best_id = jaccard, row.track_id

    return best_id if best_score > 0.25 else None


# ─── Évaluation d'un seul fichier × méthode × condition ─────────────────────

def _evaluate_one(
    audio_path: Path,
    track_id: str,
    method: str | None,
    condition: str,
) -> dict:
    """
    Évalue un fichier audio pour une méthode et une condition donnée.

    Returns:
        {top1, top5, rank, score_final, score_faiss, score_fp, latency_s, error}
    """
    from src.retrieval.query_pipeline import identify_track

    base_sr = 22050
    try:
        waveform, sr = librosa.load(str(audio_path), sr=base_sr, mono=True)
    except Exception as e:
        return {"top1": False, "top5": False, "rank": None,
                "score_final": None, "score_faiss": None, "score_fp": None,
                "latency_s": 0.0, "error": f"load: {e}"}

    degraded = _apply_condition(waveform, sr, condition)

    tmp_path = None
    try:
        tmp_path = save_temp_wav(degraded, sr)
        t0 = time.time()
        results = identify_track(tmp_path, method=method, detailed=True)
        latency = round(time.time() - t0, 2)
    except Exception as e:
        return {"top1": False, "top5": False, "rank": None,
                "score_final": None, "score_faiss": None, "score_fp": None,
                "latency_s": 0.0, "error": f"identify: {e}"}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if not results:
        return {"top1": False, "top5": False, "rank": None,
                "stage1_rank": None, "stage1_top1": False, "stage1_top5": False,
                "score_final": None, "score_faiss": None, "score_fp": None,
                "latency_s": latency, "error": "no results"}

    rank = next((i + 1 for i, r in enumerate(results) if r[0] == track_id), None)
    top1 = rank == 1
    top5 = rank is not None and rank <= 5

    # Rang Stage 1 : tri par score_faiss uniquement (avant re-ranking fingerprint)
    sorted_s1   = sorted(results, key=lambda r: r[2] if len(r) > 2 else 0.0, reverse=True)
    stage1_rank = next((i + 1 for i, r in enumerate(sorted_s1) if r[0] == track_id), None)

    score_final = score_faiss = score_fp = None
    if rank is not None:
        r = results[rank - 1]
        score_final = round(r[1], 4)
        score_faiss = round(r[2], 4) if len(r) > 2 else None
        score_fp    = round(r[3], 6) if len(r) > 3 else None

    return {
        "top1":         top1,
        "top5":         top5,
        "rank":         rank,
        "stage1_rank":  stage1_rank,
        "stage1_top1":  stage1_rank == 1,
        "stage1_top5":  stage1_rank is not None and stage1_rank <= 5,
        "score_final":  score_final,
        "score_faiss":  score_faiss,
        "score_fp":     score_fp,
        "latency_s":    latency,
        "error":        None,
    }


# ─── Point d'entrée public ──────────────────────────────────────────────────

def run_evaluate(
    methods:    list[str] | None = None,
    conditions: list[str] | None = None,
    n_tracks:   int = 0,
    out_dir:    Path | None = None,
    plot:       bool = True,
) -> dict:
    """
    Lance l'évaluation comparative multi-tracks multi-méthodes.

    Args:
        methods:    liste de méthodes à évaluer (défaut : ["mfcc", "clap"]).
        conditions: liste de conditions (défaut : toutes les 5 conditions).
        n_tracks:   limiter à N tracks du manifest (0 = tous).
        out_dir:    dossier de sortie pour le JSON (défaut : results/eval/).
        plot:       si True, génère les graphiques après l'évaluation.

    Returns:
        Dict résultats complet (aussi sauvegardé en JSON).
    """
    if methods is None:
        methods = ["mfcc", "clap"]
    if conditions is None:
        conditions = ALL_CONDITIONS

    out_dir = Path(out_dir) if out_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Chargement du manifest ──
    manifest = load_manifest()
    if not manifest:
        print(
            "[evaluate] ⚠  Aucun fichier de test dans data/raw/manifest.json.\n"
            "           Téléchargez des clips de test avec :\n"
            "             python manage.py download-audio \"Artiste Titre\" --duration 30\n"
            "           Les tracks doivent être dans la base (manage.py ingest) pour que\n"
            "           le manifest soit renseigné automatiquement."
        )
        return {}

    if n_tracks > 0:
        manifest = manifest[:n_tracks]

    raw_dir = ROOT / "data" / "raw"

    print(f"\n[evaluate] {len(manifest)} track(s) de test | "
          f"méthodes : {methods} | conditions : {conditions}\n")
    print(f"{'─'*70}")

    # ── Évaluation ──
    # Structure : results[method][condition] = liste de résultats par track
    per_method_condition: dict[str, dict[str, list[dict]]] = {
        m: {c: [] for c in conditions} for m in methods
    }

    for entry in manifest:
        filename  = entry["filename"]
        track_id  = entry["track_id"]
        artist    = entry.get("artist", "")
        title     = entry.get("title", "")
        audio_path = raw_dir / filename

        print(f"\n  [{artist} — {title}]  ({filename})")

        for method in methods:
            for condition in conditions:
                label = CONDITION_LABELS.get(condition, condition)
                res = _evaluate_one(audio_path, track_id, method, condition)

                if res["error"]:
                    rank_str = f"ERREUR ({res['error']})"
                elif res["rank"] is None:
                    rank_str = "NF"
                else:
                    rank_str = f"#{res['rank']}"

                ok = "✅" if res["top1"] else ("🔶" if res["top5"] else "❌")
                print(f"    {method.upper():5s}  {label:25s}  {rank_str:>4}  "
                      f"{res['latency_s']:>5.1f}s  {ok}")

                per_method_condition[method][condition].append({
                    "track_id":   track_id,
                    "filename":   filename,
                    "artist":     artist,
                    "title":      title,
                    "duration_s": entry.get("duration_s"),
                    **res,
                })

    print(f"\n{'─'*70}")

    # ── Agrégation des métriques ──
    agg_results: dict[str, dict[str, dict]] = {}
    summary:     dict[str, dict]             = {}

    for method in methods:
        agg_results[method] = {}
        all_top1, all_top5, all_latency = [], [], []

        for condition in conditions:
            rows = per_method_condition[method][condition]
            n    = len(rows)
            if n == 0:
                agg_results[method][condition] = {
                    "top1_accuracy": 0.0, "top5_accuracy": 0.0,
                    "mrr": 0.0, "mean_latency_s": 0.0, "n_tracks": 0,
                    "per_track": [],
                }
                continue

            top1_acc   = sum(1 for r in rows if r["top1"])          / n
            top5_acc   = sum(1 for r in rows if r["top5"])          / n
            top1_s1    = sum(1 for r in rows if r.get("stage1_top1")) / n
            top5_s1    = sum(1 for r in rows if r.get("stage1_top5")) / n
            mrr        = sum(
                1.0 / r["rank"] for r in rows if r["rank"] is not None
            ) / n
            mean_lat   = float(np.mean([r["latency_s"] for r in rows]))

            agg_results[method][condition] = {
                "top1_accuracy":        round(top1_acc, 3),
                "top5_accuracy":        round(top5_acc, 3),
                "top1_stage1_accuracy": round(top1_s1,  3),
                "top5_stage1_accuracy": round(top5_s1,  3),
                "mrr":                  round(mrr,       3),
                "mean_latency_s":       round(mean_lat,  2),
                "n_tracks":             n,
                "per_track":            rows,
            }

            all_top1.extend([r["top1"] for r in rows])
            all_top5.extend([r["top5"] for r in rows])
            all_latency.extend([r["latency_s"] for r in rows])

        summary[method] = {
            "overall_top1_pct": round(sum(all_top1) / len(all_top1) * 100, 1) if all_top1 else 0,
            "overall_top5_pct": round(sum(all_top5) / len(all_top5) * 100, 1) if all_top5 else 0,
            "mean_latency_s":   round(float(np.mean(all_latency)), 2) if all_latency else 0,
            "clean_top1_pct":   round(
                agg_results[method].get("clean", {}).get("top1_accuracy", 0) * 100, 1
            ),
        }

    # ── Affichage du récapitulatif ──
    print("\n[evaluate] Récapitulatif :\n")
    col_w = 14
    header = f"  {'Méthode':8s}" + "".join(
        f"  {CONDITION_LABELS.get(c, c)[:col_w].center(col_w)}" for c in conditions
    )
    print(header)
    print(f"  {'─'*8}" + "".join(f"  {'─'*col_w}" for _ in conditions))

    for method in methods:
        row = f"  {method.upper():8s}"
        for condition in conditions:
            agg = agg_results[method].get(condition, {})
            t1  = agg.get("top1_accuracy", 0)
            row += f"  {t1*100:>4.0f}% Top-1".center(col_w + 2)
        print(row)

    print()
    for method in methods:
        s = summary[method]
        print(f"  {method.upper():5s} — Clean Top-1: {s['clean_top1_pct']:>5.1f}%  "
              f"| Global Top-1: {s['overall_top1_pct']:>5.1f}%  "
              f"| Latence moy: {s['mean_latency_s']:.1f} s")

    # ── Sauvegarde JSON ──
    out_data = {
        "timestamp":  datetime.now().isoformat(),
        "methods":    methods,
        "conditions": conditions,
        "n_tracks":   len(manifest),
        "results":    agg_results,
        "summary":    summary,
    }
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"eval_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"\n  JSON → {json_path}")

    # ── Graphiques ──
    if plot:
        from src.evaluation.plots import run_plots
        plots_dir = ROOT / "results" / "plots"
        run_plots(eval_jsons=[json_path], out_dir=plots_dir)

    return out_data


# ─── Évaluation RIR multi-tracks ─────────────────────────────────────────────

def run_rir_evaluate(
    methods:    list[str] | None = None,
    conditions: list[str] | None = None,
    n_tracks:   int = 0,
    out_dir:    Path | None = None,
    plot:       bool = True,
) -> dict:
    """
    Compare les performances Stage 1 (FAISS) avec et sans vecteurs RIR,
    sur plusieurs morceaux et plusieurs conditions de dégradation.

    L'index sans RIR est construit en mémoire une seule fois par méthode
    (pas de modification de la base).

    Produit :
      results/eval/rir_eval_TIMESTAMP.json
      results/plots/rir_*.png (si plot=True)

    Args:
        methods:    méthodes à évaluer (défaut : [config.EMBEDDING_METHOD]).
        conditions: conditions de dégradation (défaut : toutes les 5).
        n_tracks:   limiter à N tracks du manifest (0 = tous).
        out_dir:    dossier de sortie JSON.
        plot:       générer les graphiques après.
    """
    import src.config as _cfg
    from src.evaluation.rir_impact import rir_impact_scores, _load_no_rir_index

    if methods is None:
        methods = [_cfg.EMBEDDING_METHOD]
    if conditions is None:
        conditions = ALL_CONDITIONS

    out_dir = Path(out_dir) if out_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    if not manifest:
        print(
            "[rir-evaluate] ⚠  Aucun fichier de test dans data/raw/manifest.json.\n"
            "               Téléchargez des clips avec : manage.py download-audio"
        )
        return {}

    if n_tracks > 0:
        manifest = manifest[:n_tracks]

    raw_dir = ROOT / "data" / "raw"
    print(f"\n[rir-evaluate] {len(manifest)} track(s) | méthodes : {methods} | "
          f"conditions : {conditions}\n{'─'*70}")

    # Structure : results[method][condition] = {with_rir: [...], without_rir: [...]}
    per_method: dict = {
        m: {c: {"with_rir": [], "without_rir": []} for c in conditions}
        for m in methods
    }

    for method in methods:
        collection_key = _cfg.get_collection_key(method)
        print(f"\n[{method.upper()}] Construction de l'index sans RIR…")

        # Construire l'index no-RIR UNE seule fois par méthode
        try:
            no_rir_index = _load_no_rir_index(collection_key)
        except Exception as e:
            print(f"  ⚠  Impossible de construire l'index sans RIR : {e}")
            continue

        for entry in manifest:
            filename   = entry["filename"]
            track_id   = entry["track_id"]
            artist     = entry.get("artist", "")
            title      = entry.get("title", "")
            audio_path = raw_dir / filename

            print(f"\n  [{artist} — {title}]")

            for condition in conditions:
                label    = CONDITION_LABELS.get(condition, condition)
                waveform_base, base_sr = librosa.load(str(audio_path), sr=22050, mono=True)
                degraded = _apply_condition(waveform_base, base_sr, condition)

                tmp_path = None
                try:
                    tmp_path = save_temp_wav(degraded, base_sr)
                    res = rir_impact_scores(
                        audio_path=tmp_path,
                        track_id=track_id,
                        method=method,
                        prebuilt_no_rir=no_rir_index,
                    )
                except Exception as e:
                    print(f"    ⚠  {label}: {e}")
                    continue
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)

                row = {
                    "track_id":    track_id,
                    "filename":    filename,
                    "artist":      artist,
                    "title":       title,
                    "duration_s":  entry.get("duration_s"),
                    **res["with_rir"],
                }
                per_method[method][condition]["with_rir"].append(row)

                row_no = {
                    "track_id":    track_id,
                    "filename":    filename,
                    "artist":      artist,
                    "title":       title,
                    "duration_s":  entry.get("duration_s"),
                    **res["without_rir"],
                }
                per_method[method][condition]["without_rir"].append(row_no)

                rw = res["with_rir"]["rank"]
                rn = res["without_rir"]["rank"]
                rw_s = f"#{rw}" if rw else "NF"
                rn_s = f"#{rn}" if rn else "NF"
                delta = (rn - rw) if (rw and rn) else None
                ok = ("↑ +" + str(delta) if delta and delta > 0
                      else ("→ =" if delta == 0 else ("↓ " + str(delta) if delta else "?")))
                print(f"    {label:25s}  sans={rn_s:>4}  avec={rw_s:>4}  {ok}")

    # ── Agrégation ──
    agg: dict = {}
    for method in methods:
        agg[method] = {}
        for condition in conditions:
            for variant in ("with_rir", "without_rir"):
                rows = per_method[method][condition][variant]
                n    = len(rows)
                if n == 0:
                    agg.setdefault(method, {})[f"{condition}_{variant}"] = {}
                    continue
                top1  = sum(1 for r in rows if r.get("rank") == 1) / n
                top5  = sum(1 for r in rows if r.get("rank") is not None and r["rank"] <= 5) / n
                scores = [r.get("faiss_score", 0) for r in rows]
                agg[method][f"{condition}_{variant}"] = {
                    "top1_accuracy":    round(top1,            3),
                    "top5_accuracy":    round(top5,            3),
                    "mean_faiss_score": round(float(np.mean(scores)), 4),
                    "n_tracks":         n,
                    "per_track":        rows,
                }

    # ── Sauvegarde JSON ──
    out_data = {
        "timestamp":  datetime.now().isoformat(),
        "type":       "rir_evaluation",
        "methods":    methods,
        "conditions": conditions,
        "n_tracks":   len(manifest),
        "results":    agg,
    }
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"rir_eval_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"\n  JSON → {json_path}")

    # ── Graphiques ──
    if plot:
        from src.evaluation.plots import run_plots
        plots_dir = ROOT / "results" / "plots"
        run_plots(rir_eval_jsons=[json_path], out_dir=plots_dir)

    return out_data
