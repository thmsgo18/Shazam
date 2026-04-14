"""
src/evaluation/benchmark.py

Benchmark de robustesse pour la reconnaissance musicale.

Méthodologie :
  - Fichier audio fourni en paramètre (enregistré dans le manifest via download-test)
  - Le track cible est auto-détecté depuis data/raw/manifest.json
  - 4 cas de test en mode rapide, ~10 en mode complet :
      1. CLEAN  | original      — fichier fourni tel quel
      2. SIM    | bruit SNR 20dB
      3. SIM    | reverb léger
      4. SIM    | combo bruit+reverb+filtre (cas difficile)
      ... (mode --full : +6 dégradations supplémentaires)

  - Les résultats sont loggés dans results/benchmark/benchmark_TIMESTAMP.json

Points d'entrée publics :
  run_benchmark(audio, label_run, full)  — lance un benchmark sur un fichier audio
  run_compare(json_paths)               — compare plusieurs runs JSON
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT        = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "benchmark"

# Couleurs terminal
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"


# ---------------------------------------------------------------------------
# Lookup manifest
# ---------------------------------------------------------------------------

def _lookup_target(audio_path: str) -> tuple[str | None, str]:
    """
    Cherche le track_id et le label lisible d'un fichier audio dans le manifest.

    Returns:
        (track_id, label) — track_id peut être None si non trouvé.
    """
    manifest = ROOT / "data" / "raw" / "manifest.json"
    filename = Path(audio_path).name

    if manifest.exists():
        try:
            with open(manifest, encoding="utf-8") as f:
                entries = json.load(f)
            for entry in entries:
                if entry.get("filename") == filename:
                    artist = entry.get("artist", "")
                    title  = entry.get("title", "")
                    label  = f"{artist} — {title}".strip(" —") or filename
                    return entry.get("track_id"), label
        except Exception:
            pass

    # Non trouvé dans le manifest — on continue sans target connu
    return None, Path(audio_path).stem


# ---------------------------------------------------------------------------
# Dégradations simulées
# ---------------------------------------------------------------------------

def add_noise_at_snr(waveform: np.ndarray, snr_db: float) -> np.ndarray:
    """Ajoute du bruit blanc gaussien à un SNR cible (en dB)."""
    signal_power = np.mean(waveform ** 2)
    if signal_power == 0:
        return waveform
    noise_power_target = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(waveform)) * np.sqrt(noise_power_target)
    return (waveform + noise).astype(np.float32)


def add_reverb(waveform: np.ndarray, sr: int, decay: float = 0.3) -> np.ndarray:
    """Reverb simple par convolution avec une RIR synthétique."""
    from scipy.signal import fftconvolve
    rir_len     = int(0.5 * sr)
    rir         = np.zeros(rir_len)
    reflections = [0, int(0.02 * sr), int(0.05 * sr), int(0.1 * sr), int(0.2 * sr)]
    gains       = [1.0, 0.6, 0.4, 0.25, 0.15]
    for t, g in zip(reflections, gains):
        if t < rir_len:
            rir[t] = g * (decay ** (t / sr))
    reverbed = fftconvolve(waveform, rir)[:len(waveform)]
    peak     = np.max(np.abs(reverbed))
    if peak > 0:
        reverbed = reverbed / peak * np.max(np.abs(waveform))
    return reverbed.astype(np.float32)


def apply_bandpass(waveform: np.ndarray, sr: int, low_hz: float = 300, high_hz: float = 7000) -> np.ndarray:
    """Filtre passe-bande pour simuler la réponse d'un micro bas de gamme."""
    from scipy.signal import butter, sosfilt
    sos = butter(4, [low_hz, high_hz], btype="band", fs=sr, output="sos")
    return sosfilt(sos, waveform).astype(np.float32)


def simulate_opus_codec(waveform: np.ndarray, sr: int, bitrate: int = 32000) -> np.ndarray:
    """Simule la dégradation du codec Opus via encodage/décodage."""
    try:
        import subprocess
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in:
            sf.write(f_in.name, waveform, sr)
            in_path = f_in.name
        out_opus = in_path.replace(".wav", ".opus")
        out_wav  = in_path.replace(".wav", "_decoded.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", in_path, "-c:a", "libopus", "-b:a", str(bitrate), out_opus],
            capture_output=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", out_opus, "-ar", str(sr), out_wav],
            capture_output=True,
        )
        result, _ = librosa.load(out_wav, sr=sr, mono=True)
        os.unlink(in_path)
        os.unlink(out_opus)
        os.unlink(out_wav)
        return result.astype(np.float32)
    except Exception:
        return waveform


def save_temp_wav(waveform: np.ndarray, sr: int) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, waveform, sr)
    tmp.close()
    return tmp.name


def compute_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    min_len = min(len(clean), len(noisy))
    clean, noisy = clean[:min_len], noisy[:min_len]
    noise        = noisy - clean
    signal_power = np.mean(clean ** 2)
    noise_power  = np.mean(noise ** 2)
    if noise_power == 0:
        return float("inf")
    return 10 * np.log10(signal_power / noise_power)


# ---------------------------------------------------------------------------
# Suite de tests
# ---------------------------------------------------------------------------

def build_test_suite(audio_path: str, full: bool = False) -> list[dict]:
    """
    Construit la liste des cas de test à partir du fichier audio fourni.

    Args:
        audio_path: chemin vers le fichier audio de référence (clean).
        full:       si True, génère ~10 cas (défaut : 4 cas rapides).

    Returns:
        Liste de dicts {label, path, type, tmp?}.
    """
    tests   = []
    base_sr = 22050

    # Cas 1 : fichier original fourni tel quel
    if os.path.exists(audio_path):
        tests.append({"label": "CLEAN  | original", "path": audio_path, "type": "real"})
    else:
        print(f"{RED}  [ERREUR] Fichier introuvable : {audio_path}{RESET}")
        return tests

    # Chargement pour générer les dégradations simulées
    try:
        waveform, sr = librosa.load(audio_path, sr=base_sr, mono=True)
    except Exception as e:
        print(f"{RED}  [ERREUR] Impossible de charger l'audio : {e}{RESET}")
        return tests

    np.random.seed(42)

    if full:
        sim_cases = [
            ("SIM    | bruit SNR 20dB",    add_noise_at_snr(waveform, 20)),
            ("SIM    | bruit SNR 10dB",    add_noise_at_snr(waveform, 10)),
            ("SIM    | bruit SNR  5dB",    add_noise_at_snr(waveform,  5)),
            ("SIM    | reverb léger",       add_reverb(waveform, sr, decay=0.3)),
            ("SIM    | reverb fort",        add_reverb(waveform, sr, decay=0.6)),
            ("SIM    | reverb+bruit 20dB",  add_reverb(add_noise_at_snr(waveform, 20), sr)),
            ("SIM    | reverb+bruit 10dB",  add_reverb(add_noise_at_snr(waveform, 10), sr)),
            ("SIM    | bandpass 300-7kHz",  apply_bandpass(waveform, sr, 300, 7000)),
            ("SIM    | combo 15dB+rev+bp",  apply_bandpass(add_reverb(add_noise_at_snr(waveform, 15), sr), sr)),
            ("SIM    | codec Opus 32kbps",  simulate_opus_codec(waveform, sr, 32000)),
        ]
    else:
        sim_cases = [
            ("SIM    | bruit SNR 20dB",   add_noise_at_snr(waveform, 20)),
            ("SIM    | reverb léger",      add_reverb(waveform, sr, decay=0.3)),
            ("SIM    | combo 15dB+rev+bp", apply_bandpass(add_reverb(add_noise_at_snr(waveform, 15), sr), sr)),
        ]

    for label, degraded in sim_cases:
        tmp_path = save_temp_wav(degraded, sr)
        real_snr = compute_snr(waveform, degraded)
        tests.append({
            "label": f"{label} (SNR={real_snr:.1f}dB)",
            "path":  tmp_path,
            "type":  "simulated",
            "tmp":   True,
        })

    return tests


# ---------------------------------------------------------------------------
# Identification
# ---------------------------------------------------------------------------

def run_identification(audio_path: str, target_track_id: str | None,
                       method: str | None = None) -> dict:
    """Lance identify_track et extrait les métriques clés."""
    from src.retrieval.query_pipeline import identify_track

    t0 = time.time()
    try:
        results = identify_track(audio_path, method=method, detailed=True)
    except Exception as e:
        return {
            "success": False, "error": str(e),
            "rank": None, "score_top1": None, "ratio": None,
            "score_target_faiss": None, "score_target_fp": None, "score_target_final": None,
            "elapsed_s": time.time() - t0,
        }
    elapsed = time.time() - t0

    if not results:
        return {
            "success": False, "error": "no results",
            "rank": None, "score_top1": None, "ratio": None,
            "score_target_faiss": None, "score_target_fp": None, "score_target_final": None,
            "elapsed_s": elapsed,
        }

    rank = target_faiss = target_fp = target_final = None
    if target_track_id:
        for i, r in enumerate(results):
            if r[0] == target_track_id:
                rank         = i + 1
                target_final = r[1]
                target_faiss = r[2]
                target_fp    = r[3]
                break

    score0 = results[0][1]
    score1 = results[1][1] if len(results) > 1 else 0
    ratio  = score0 / score1 if score1 > 0 else float("inf")

    return {
        "success":             True,
        "error":               None,
        "rank":                rank,
        "top1_is_target":      (results[0][0] == target_track_id) if target_track_id else None,
        "score_top1":          round(score0,        4),
        "ratio":               round(ratio,         3),
        "score_target_faiss":  round(target_faiss,  4) if target_faiss  is not None else None,
        "score_target_fp":     round(target_fp,     6) if target_fp     is not None else None,
        "score_target_final":  round(target_final,  4) if target_final  is not None else None,
        "elapsed_s":           round(elapsed,       1),
        "all_results": [
            {"rank": i+1, "track_id": r[0], "score": round(r[1], 4),
             "faiss": round(r[2], 4), "fp": round(r[3], 6)}
            for i, r in enumerate(results)
        ],
    }


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------

def rank_color(rank):
    if rank == 1:  return GREEN
    if rank <= 3:  return YELLOW
    return RED


def print_result_row(label: str, res: dict, width: int = 42):
    label_str = label[:width].ljust(width)
    if not res["success"]:
        print(f"  {label_str}  {RED}ERREUR : {res['error']}{RESET}")
        return
    rank     = res["rank"]
    r_col    = rank_color(rank) if rank else RED
    rank_str = f"#{rank}" if rank else "NF"
    ok = "✅" if rank == 1 else ("⚠️ " if rank and rank <= 3 else "❌")
    print(
        f"  {label_str}  "
        f"{r_col}{rank_str:>4}{RESET}  "
        f"score={res['score_target_final'] or 0:>9.4f}  "
        f"faiss={res['score_target_faiss'] or 0:>9.4f}  "
        f"fp={res['score_target_fp'] or 0:.4f}  "
        f"ratio={res['ratio']:>5.2f}  "
        f"{res['elapsed_s']:>5.1f}s  {ok}"
    )


def print_header():
    w = 42
    print(
        f"\n  {'LABEL'.ljust(w)}  {'Rank':>4}  "
        f"{'score_final':>11}  {'score_faiss':>11}  {'score_fp':>8}  "
        f"{'ratio':>7}  {'time':>6}  OK?"
    )
    print(f"  {'-'*w}  {'-'*4}  {'-'*11}  {'-'*11}  {'-'*8}  {'-'*7}  {'-'*6}  ---")


# ---------------------------------------------------------------------------
# Points d'entrée publics
# ---------------------------------------------------------------------------

def run_benchmark(
    audio: str,
    label_run: str = "baseline",
    full: bool = False,
) -> dict:
    """
    Lance le benchmark de robustesse sur un fichier audio et sauvegarde les résultats.

    Le track cible est auto-détecté depuis data/raw/manifest.json.
    La méthode d'embedding est lue depuis src/config.py (EMBEDDING_METHOD).

    Args:
        audio:     chemin vers le fichier audio de test (doit être dans le manifest).
        label_run: nom du run pour la traçabilité.
        full:      si True, exécute ~10 cas (défaut : 4 cas).

    Returns:
        Dict avec les résultats complets du run.
    """
    import src.config as config

    target_track_id, target_label = _lookup_target(audio)
    method = config.EMBEDDING_METHOD
    mode_str = "MODE COMPLET (~10 cas)" if full else "MODE RAPIDE (4 cas)"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{BOLD}{CYAN}{'═'*90}{RESET}")
    print(f"{BOLD}{CYAN}  BENCHMARK — {label_run}  [{mode_str}]{RESET}")
    print(f"{BOLD}{CYAN}  Méthode      : {method.upper()}{RESET}")
    print(f"{BOLD}{CYAN}  Fichier test : {Path(audio).name}{RESET}")
    if target_track_id:
        print(f"{BOLD}{CYAN}  Track cible  : {target_label} ({target_track_id}){RESET}")
    else:
        print(f"{YELLOW}  Track cible  : non trouvé dans le manifest — rang non calculé{RESET}")
        print(f"{YELLOW}  Lance d'abord : python manage.py download-test \"<artiste titre>\"{RESET}")
    print(f"{BOLD}{CYAN}  Date         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}")
    print(f"{BOLD}{CYAN}{'═'*90}{RESET}")

    tests = build_test_suite(audio_path=audio, full=full)
    if not tests:
        print(f"{RED}  Aucun fichier de test généré.{RESET}")
        return {}

    print(f"\n  {len(tests)} cas de test à exécuter...")
    print_header()

    all_results = {}
    tmp_files   = []

    for test in tests:
        if test.get("tmp"):
            tmp_files.append(test["path"])
        res = run_identification(test["path"], target_track_id=target_track_id, method=None)
        all_results[test["label"]] = {**res, "type": test["type"]}
        print_result_row(test["label"], res)

    for f in tmp_files:
        try:
            os.unlink(f)
        except Exception:
            pass

    print(f"\n{BOLD}  Récapitulatif :{RESET}")
    success = [r for r in all_results.values() if r.get("top1_is_target")]
    total   = len(all_results)
    if target_track_id:
        print(f"  Top-1 correct : {len(success)}/{total} ({100*len(success)/total:.0f}%)")
        real_tests = {k: v for k, v in all_results.items() if v["type"] == "real"}
        sim_tests  = {k: v for k, v in all_results.items() if v["type"] == "simulated"}
        if real_tests:
            ranks = [v["rank"] for v in real_tests.values() if v.get("rank")]
            if ranks:
                print(f"  Fichiers réels  — Rank moyen : {np.mean(ranks):.1f}")
        if sim_tests:
            ranks = [v["rank"] for v in sim_tests.values() if v.get("rank")]
            if ranks:
                print(f"  Simulations     — Rank moyen : {np.mean(ranks):.1f}")
    else:
        print(f"  {total} cas exécutés (pas de target connu — Top-1 non calculé)")

    out = {
        "run_label": label_run,
        "method":    method,
        "timestamp": datetime.now().isoformat(),
        "audio":     str(audio),
        "target":    {"track_id": target_track_id, "label": target_label},
        "results":   all_results,
        "summary": {
            "total_tests":       total,
            "top1_correct":      len(success) if target_track_id else None,
            "top1_accuracy_pct": round(100 * len(success) / total, 1) if target_track_id else None,
        },
    }

    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = label_run.replace(" ", "_").replace("/", "-")
    json_path  = RESULTS_DIR / f"benchmark_{ts}_{safe_label}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  Résultats sauvegardés → {json_path}")

    return out


def run_compare(json_paths: list[str]) -> None:
    """
    Affiche un tableau comparatif de plusieurs runs.

    Args:
        json_paths: liste de chemins vers des fichiers JSON de résultats.
    """
    runs = []
    for p in json_paths:
        with open(p) as f:
            runs.append(json.load(f))

    all_labels = []
    for run in runs:
        for k in run["results"].keys():
            if k not in all_labels:
                all_labels.append(k)

    run_names = [r["run_label"] for r in runs]
    col_w     = 14
    lbl_w     = 48

    print(f"\n{'─'*90}")
    print(f"  COMPARAISON ENTRE RUNS")
    print(f"{'─'*90}")
    header = f"  {'LABEL'.ljust(lbl_w)}" + "".join(f"  {n[:col_w].center(col_w)}" for n in run_names)
    print(header)
    print(f"  {'-'*lbl_w}  " + "  ".join(["-"*col_w]*len(runs)))

    for lbl in all_labels:
        row = f"  {lbl[:lbl_w].ljust(lbl_w)}"
        for run in runs:
            res = run["results"].get(lbl)
            if res is None:
                row += f"  {'—'.center(col_w)}"
            elif res.get("rank") == 1:
                row += f"  {GREEN}{'#1 ✅'.center(col_w)}{RESET}"
            elif res.get("rank"):
                row += f"  {YELLOW if res['rank'] <= 3 else RED}{'#'+str(res['rank'])+' ❌'.center(col_w)}{RESET}"
            else:
                row += f"  {RED}{'NF ❌'.center(col_w)}{RESET}"
        print(row)

    print(f"\n  Top-1 accuracy :")
    for run in runs:
        pct = run["summary"].get("top1_accuracy_pct")
        if pct is None:
            print(f"    {run['run_label']:30s} : (pas de target connu)")
            continue
        col = GREEN if pct >= 80 else (YELLOW if pct >= 50 else RED)
        print(
            f"    {run['run_label']:30s} : "
            f"{col}{pct:.0f}%{RESET} "
            f"({run['summary']['top1_correct']}/{run['summary']['total_tests']})"
        )
