"""
scripts/benchmark_noise_robustness.py

Benchmark de robustesse pour la reconnaissance musicale.

Méthodologie :
  - Ground truth : Flowers - Miley Cyrus (track_id connu)
  - 5 cas de test focalisés sur les conditions réelles :
      1. CLEAN 5s   — cas le plus court, minimal
      2. CLEAN 15s  — cas nominal Shazam
      3. CLEAN 30s  — cas long, limite haute
      4. MIC réel   — enregistrement micro Mac (93-Rue-Belliard.mp3) ← cas cible
      5. SIM combo  — pire dégradation simulée (bruit+reverb+filtre)

  - Option --full pour relancer les 14 cas complets (pour l'historique)
  - Les résultats sont loggés dans results/benchmark/benchmark_TIMESTAMP.json

Usage :
    python scripts/benchmark_noise_robustness.py --label "A6_ma_modif"
    python scripts/benchmark_noise_robustness.py --full --label "full_run"
    python scripts/benchmark_noise_robustness.py --compare results/benchmark/A4.json results/benchmark/A5.json
"""

import sys
import os
import time
import json
import argparse
import tempfile
import numpy as np
import soundfile as sf
import librosa
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── Ground truth ────────────────────────────────────────────────────────────
TARGET_TRACK_ID = "f01ab00f1fdc5a57fd2676f4d68631a8"  # Flowers - Miley Cyrus
TARGET_TITLE    = "Flowers - Miley Cyrus"

RESULTS_DIR = Path("results/benchmark")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Couleurs terminal ────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"


# ═════════════════════════════════════════════════════════════════════════════
# GÉNÉRATION DES DÉGRADATIONS SIMULÉES
# ═════════════════════════════════════════════════════════════════════════════

def add_noise_at_snr(waveform: np.ndarray, snr_db: float) -> np.ndarray:
    """Ajoute du bruit blanc gaussien à un SNR cible (en dB)."""
    signal_power = np.mean(waveform ** 2)
    if signal_power == 0:
        return waveform
    noise_power_target = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(waveform)) * np.sqrt(noise_power_target)
    return (waveform + noise).astype(np.float32)


def add_reverb(waveform: np.ndarray, sr: int, decay: float = 0.3) -> np.ndarray:
    """Reverb simple par convolution avec une RIR (Room Impulse Response) synthétique."""
    # RIR synthétique : quelques réflexions exponentiellement décroissantes
    rir_len = int(0.5 * sr)  # 500ms
    rir = np.zeros(rir_len)
    reflections = [0, int(0.02 * sr), int(0.05 * sr), int(0.1 * sr), int(0.2 * sr)]
    gains = [1.0, 0.6, 0.4, 0.25, 0.15]
    for t, g in zip(reflections, gains):
        if t < rir_len:
            rir[t] = g * (decay ** (t / sr))
    from scipy.signal import fftconvolve
    reverbed = fftconvolve(waveform, rir)[:len(waveform)]
    # Normaliser pour éviter le clipping
    peak = np.max(np.abs(reverbed))
    if peak > 0:
        reverbed = reverbed / peak * np.max(np.abs(waveform))
    return reverbed.astype(np.float32)


def apply_bandpass(waveform: np.ndarray, sr: int, low_hz: float = 200, high_hz: float = 7000) -> np.ndarray:
    """Filtre passe-bande pour simuler la réponse d'un micro bas de gamme."""
    from scipy.signal import butter, sosfilt
    sos = butter(4, [low_hz, high_hz], btype='band', fs=sr, output='sos')
    return sosfilt(sos, waveform).astype(np.float32)


def simulate_opus_codec(waveform: np.ndarray, sr: int, bitrate: int = 32000) -> np.ndarray:
    """Simule la dégradation du codec Opus (WebM browser) via encodage/décodage."""
    try:
        import subprocess
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in:
            sf.write(f_in.name, waveform, sr)
            in_path = f_in.name
        out_opus = in_path.replace(".wav", ".opus")
        out_wav  = in_path.replace(".wav", "_decoded.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", in_path, "-c:a", "libopus",
             "-b:a", str(bitrate), out_opus],
            capture_output=True
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", out_opus, "-ar", str(sr), out_wav],
            capture_output=True
        )
        result, _ = librosa.load(out_wav, sr=sr, mono=True)
        os.unlink(in_path); os.unlink(out_opus); os.unlink(out_wav)
        return result.astype(np.float32)
    except Exception:
        return waveform  # fallback sans dégradation codec


def save_temp_wav(waveform: np.ndarray, sr: int) -> str:
    """Sauvegarde un waveform dans un fichier WAV temporaire. Retourne le chemin."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, waveform, sr)
    tmp.close()
    return tmp.name


def compute_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Calcule le SNR réel entre un signal propre et sa version bruitée (en dB)."""
    min_len = min(len(clean), len(noisy))
    clean, noisy = clean[:min_len], noisy[:min_len]
    noise = noisy - clean
    signal_power = np.mean(clean ** 2)
    noise_power  = np.mean(noise ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)


# ═════════════════════════════════════════════════════════════════════════════
# RUNNER D'IDENTIFICATION
# ═════════════════════════════════════════════════════════════════════════════

def run_identification(audio_path: str) -> dict:
    """Lance identify_track et extrait les métriques clés."""
    from src.retrieval.query_pipeline import identify_track

    t0 = time.time()
    try:
        results = identify_track(audio_path)
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "rank": None,
            "score_top1": None,
            "ratio": None,
            "score_flowers_faiss": None,
            "score_flowers_fp": None,
            "score_flowers_final": None,
            "elapsed_s": time.time() - t0,
        }
    elapsed = time.time() - t0

    if not results:
        return {
            "success": False, "error": "no results",
            "rank": None, "score_top1": None, "ratio": None,
            "score_flowers_faiss": None, "score_flowers_fp": None,
            "score_flowers_final": None, "elapsed_s": elapsed,
        }

    # Chercher Flowers dans les résultats
    rank = None
    flowers_faiss = flowers_fp = flowers_final = None
    for i, r in enumerate(results):
        if r[0] == TARGET_TRACK_ID:
            rank = i + 1
            flowers_final = r[1]
            flowers_faiss = r[2]
            flowers_fp    = r[3]
            break

    score0 = results[0][1]
    score1 = results[1][1] if len(results) > 1 else 0
    ratio  = score0 / score1 if score1 > 0 else float('inf')

    return {
        "success": True,
        "error": None,
        "rank": rank,
        "top1_is_flowers": results[0][0] == TARGET_TRACK_ID,
        "score_top1": round(score0, 4),
        "ratio": round(ratio, 3),
        "score_flowers_faiss": round(flowers_faiss, 4) if flowers_faiss is not None else None,
        "score_flowers_fp":    round(flowers_fp,    6) if flowers_fp    is not None else None,
        "score_flowers_final": round(flowers_final, 4) if flowers_final is not None else None,
        "elapsed_s": round(elapsed, 1),
        "all_results": [
            {"rank": i+1, "track_id": r[0], "score": round(r[1],4),
             "faiss": round(r[2],4), "fp": round(r[3],6)}
            for i, r in enumerate(results)
        ],
    }


# ═════════════════════════════════════════════════════════════════════════════
# AFFICHAGE DES RÉSULTATS
# ═════════════════════════════════════════════════════════════════════════════

def rank_color(rank):
    if rank == 1:    return GREEN
    if rank <= 3:    return YELLOW
    return RED


def print_result_row(label: str, res: dict, width: int = 42):
    label_str = label[:width].ljust(width)
    if not res["success"]:
        print(f"  {label_str}  {RED}ERREUR : {res['error']}{RESET}")
        return

    rank = res["rank"]
    r_col = rank_color(rank) if rank else RED
    rank_str = f"#{rank}" if rank else "NF"

    ok = "✅" if rank == 1 else ("⚠️ " if rank and rank <= 3 else "❌")
    print(
        f"  {label_str}  "
        f"{r_col}{rank_str:>4}{RESET}  "
        f"score={res['score_flowers_final'] or 0:>9.4f}  "
        f"faiss={res['score_flowers_faiss'] or 0:>9.4f}  "
        f"fp={res['score_flowers_fp'] or 0:.4f}  "
        f"ratio={res['ratio']:>5.2f}  "
        f"{res['elapsed_s']:>5.1f}s  {ok}"
    )


def print_header():
    w = 42
    print(f"\n  {'LABEL'.ljust(w)}  {'Rank':>4}  {'score_final':>11}  {'score_faiss':>11}  {'score_fp':>8}  {'ratio':>7}  {'time':>6}  OK?")
    print(f"  {'-'*w}  {'-'*4}  {'-'*11}  {'-'*11}  {'-'*8}  {'-'*7}  {'-'*6}  ---")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def build_test_suite(full: bool = False):
    """
    Construit la liste des cas de test.

    Mode par défaut (full=False) : 5 cas ciblés
      - 3 cas propres (vérification de non-régression)
      - 1 cas micro réel (objectif principal)
      - 1 cas simulé combo (représentant des dégradations génériques)

    Mode full (full=True) : 14 cas complets pour l'historique
    """
    tests = []
    base_sr = 22050

    # ── Fichiers réels (toujours inclus) ─────────────────────────────────
    real_files = [
        ("CLEAN  | middle  5s", "data/raw/Miley Cyrus - Flowers (Official Video)__middle_5s.mp3"),
        ("CLEAN  | middle 15s", "data/raw/Miley Cyrus - Flowers (Official Video)__middle_15s.mp3"),
        ("CLEAN  | middle 30s", "data/raw/Miley Cyrus - Flowers (Official Video)__middle_30s.mp3"),
        ("MIC    | reel   24s", "data/raw/93-Rue-Belliard.mp3"),
    ]
    if full:
        real_files.insert(3, ("CLEAN  | start  30s", "data/raw/Miley Cyrus - Flowers (Official Video)__start_30s.mp3"))

    for label, path in real_files:
        if os.path.exists(path):
            tests.append({"label": label, "path": path, "type": "real"})
        else:
            print(f"{YELLOW}  [SKIP] Fichier manquant : {path}{RESET}")

    # ── Dégradations simulées ─────────────────────────────────────────────
    clean_path = "data/raw/Miley Cyrus - Flowers (Official Video)__middle_15s.mp3"
    if not os.path.exists(clean_path):
        print(f"{YELLOW}  [SKIP] Fichier de référence manquant pour simulations{RESET}")
        return tests

    waveform, sr = librosa.load(clean_path, sr=base_sr, mono=True)
    np.random.seed(42)

    if full:
        sim_cases = [
            ("SIM    | bruit SNR 30dB",    add_noise_at_snr(waveform, 30)),
            ("SIM    | bruit SNR 20dB",    add_noise_at_snr(waveform, 20)),
            ("SIM    | bruit SNR 10dB",    add_noise_at_snr(waveform, 10)),
            ("SIM    | bruit SNR  5dB",    add_noise_at_snr(waveform,  5)),
            ("SIM    | reverb 30dB+reverb", add_reverb(add_noise_at_snr(waveform, 30), sr, decay=0.3)),
            ("SIM    | reverb 10dB+reverb", add_reverb(add_noise_at_snr(waveform, 10), sr, decay=0.3)),
            ("SIM    | bandpass 300-7kHz",  apply_bandpass(waveform, sr, 300, 7000)),
            ("SIM    | combo 15dB+rev+bp",  apply_bandpass(add_reverb(add_noise_at_snr(waveform, 15), sr), sr)),
            ("SIM    | codec Opus 32kbps",  simulate_opus_codec(waveform, sr, 32000)),
        ]
    else:
        # Un seul cas simulé : le combo (pire cas, représente les dégradations génériques)
        sim_cases = [
            ("SIM    | combo 15dB+rev+bp", apply_bandpass(add_reverb(add_noise_at_snr(waveform, 15), sr), sr)),
        ]

    for label, degraded in sim_cases:
        tmp_path = save_temp_wav(degraded, sr)
        real_snr = compute_snr(waveform, degraded)
        full_label = f"{label} (SNR={real_snr:.1f}dB)"
        tests.append({"label": full_label, "path": tmp_path, "type": "simulated", "tmp": True})

    return tests


def run_benchmark(label_run: str, full: bool = False) -> dict:
    mode_str = "MODE COMPLET (14 cas)" if full else "MODE RAPIDE (5 cas)"
    print(f"\n{BOLD}{CYAN}{'═'*90}{RESET}")
    print(f"{BOLD}{CYAN}  BENCHMARK — {label_run}  [{mode_str}]{RESET}")
    print(f"{BOLD}{CYAN}  Ground truth : {TARGET_TITLE} ({TARGET_TRACK_ID}){RESET}")
    print(f"{BOLD}{CYAN}  Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}")
    print(f"{BOLD}{CYAN}{'═'*90}{RESET}")

    tests = build_test_suite(full=full)
    if not tests:
        print(f"{RED}  Aucun fichier de test trouvé.{RESET}")
        return {}

    print(f"\n  {len(tests)} cas de test à exécuter...")

    print_header()

    all_results = {}
    tmp_files = []

    for test in tests:
        if test.get("tmp"):
            tmp_files.append(test["path"])
        res = run_identification(test["path"])
        all_results[test["label"]] = {**res, "type": test["type"]}
        print_result_row(test["label"], res)

    # Nettoyage des fichiers temporaires
    for f in tmp_files:
        try:
            os.unlink(f)
        except Exception:
            pass

    # ── Récapitulatif ──────────────────────────────────────────────────────
    print(f"\n{BOLD}  Récapitulatif :{RESET}")
    success = [r for r in all_results.values() if r.get("top1_is_flowers")]
    total   = len(all_results)
    print(f"  Top-1 correct : {len(success)}/{total} ({100*len(success)/total:.0f}%)")

    real_tests = {k: v for k, v in all_results.items() if v["type"] == "real"}
    sim_tests  = {k: v for k, v in all_results.items() if v["type"] == "simulated"}
    if real_tests:
        print(f"  Fichiers réels  — Rank moyen Flowers : "
              f"{np.mean([v['rank'] for v in real_tests.values() if v['rank']]):.1f}")
    if sim_tests:
        ranks = [v['rank'] for v in sim_tests.values() if v.get('rank')]
        if ranks:
            print(f"  Simulations     — Rank moyen Flowers : {np.mean(ranks):.1f}")

    # ── Sauvegarde JSON ────────────────────────────────────────────────────
    out = {
        "run_label":  label_run,
        "timestamp":  datetime.now().isoformat(),
        "target":     {"track_id": TARGET_TRACK_ID, "title": TARGET_TITLE},
        "results":    all_results,
        "summary": {
            "total_tests": total,
            "top1_correct": len(success),
            "top1_accuracy_pct": round(100 * len(success) / total, 1),
        }
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = label_run.replace(" ", "_").replace("/", "-")
    json_path = RESULTS_DIR / f"benchmark_{ts}_{safe_label}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  Résultats sauvegardés → {json_path}")

    return out


# ═════════════════════════════════════════════════════════════════════════════

def compare_runs(json_paths: list[str]):
    """Affiche un tableau comparatif de plusieurs runs."""
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
    col_w = 14
    lbl_w = 48

    print(f"\n{'─'*90}")
    print(f"  COMPARAISON ENTRE RUNS")
    print(f"{'─'*90}")
    header = f"  {'LABEL'.ljust(lbl_w)}" + "".join(f"  {n[:col_w].center(col_w)}" for n in run_names)
    print(header)
    print(f"  {'-'*lbl_w}" + "  " + "  ".join(["-"*col_w]*len(runs)))

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
        pct = run["summary"]["top1_accuracy_pct"]
        col = GREEN if pct >= 80 else (YELLOW if pct >= 50 else RED)
        print(f"    {run['run_label']:30s} : {col}{pct:.0f}%{RESET} ({run['summary']['top1_correct']}/{run['summary']['total_tests']})")


# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark robustesse au bruit — Shazam Maison")
    parser.add_argument("--label",   default="baseline", help="Nom de ce run (ex: 'A6_ma_modif')")
    parser.add_argument("--full",    action="store_true", help="Mode complet : 14 cas (au lieu de 5 par défaut)")
    parser.add_argument("--compare", nargs="+", metavar="JSON", help="Comparer plusieurs fichiers JSON de résultats")
    args = parser.parse_args()

    if args.compare:
        compare_runs(args.compare)
    else:
        run_benchmark(label_run=args.label, full=args.full)
