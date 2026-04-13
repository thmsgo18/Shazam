"""
src/evaluation/plots.py

Génération de graphiques pour le rapport à partir des résultats d'évaluation.

Graphiques disponibles :

  ── Comparaison RIR (depuis rir_eval_*.json) ─────────────────────────────
  G1  rir_paired_bar.png       Accuracy avec vs sans RIR par condition
  G2  rir_delta.png            Gain Δ accuracy apporté par les RIR
  G4  rir_faiss_scores.png     Score FAISS du bon morceau par track

  ── Évaluation pipeline (depuis eval_*.json) ─────────────────────────────
  G6  method_accuracy.png      Accuracy par méthode × condition (avec écart-type)
  G9  stage_comparison.png     Stage 1 (FAISS seul) vs Stage 2 (+ fingerprint)
  G11 duration_impact.png      Accuracy en fonction de la durée de l'extrait
  G12 heatmap_accuracy.png     Heatmap méthodes × conditions (% accuracy, colormap continue)

Workflow :
  python manage.py benchmark    --method mfcc --full --label mfcc
  python manage.py benchmark    --method clap --full --label clap
  python manage.py evaluate     --methods mfcc clap
  python manage.py rir-evaluate --methods clap
  python manage.py plots \\
      --benchmark results/benchmark/*_mfcc.json \\
      --benchmark results/benchmark/*_clap.json \\
      --eval      results/eval/eval_*.json \\
      --rir-eval  results/eval/rir_eval_*.json

Point d'entrée : run_plots(benchmark_jsons, eval_jsons, rir_eval_jsons, out_dir)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np

ROOT        = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"

# ─── Palette cohérente ───────────────────────────────────────────────────────
METHOD_COLORS = {"mfcc": "#4878d0", "clap": "#ee854a", "muq": "#6acc65",
                 "mert": "#d65f5f", "unknown": "#8c8c8c"}
METHOD_LABELS = {"mfcc": "MFCC", "clap": "CLAP", "muq": "MuQ",
                 "mert": "MERT", "unknown": "Inconnu"}

COND_LABELS = {
    "clean":  "Clean",
    "snr_20": "SNR 20 dB",
    "snr_10": "SNR 10 dB",
    "reverb": "Reverb",
    "combo":  "Combo",
}

DPI      = 150
FIG_WIDE = (13, 5)
FIG_STD  = (9, 5)
FIG_SQ   = (7, 6)

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        DPI,
})

COLOR_WITH    = "#27ae60"   # vert  — avec RIR
COLOR_WITHOUT = "#e74c3c"   # rouge — sans RIR
COLOR_S1      = "#95a5a6"   # gris  — Stage 1 seul
COLOR_S2      = "#2980b9"   # bleu  — Stage 2 final


# ─── Utilitaires ────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plots] ✓ {path.name}")


def _cond_order(conds: list[str]) -> list[str]:
    """Réordonne les conditions dans l'ordre logique."""
    order = list(COND_LABELS.keys())
    return sorted(conds, key=lambda c: order.index(c) if c in order else 99)


def _bar_label(ax, bars, fmt="{:.0f}%", offset=1.5):
    """Annote les barres avec leur valeur."""
    for bar in bars:
        h = bar.get_height()
        if h > 3:
            ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                    fmt.format(h), ha="center", va="bottom", fontsize=8)


# ─── Chargement ─────────────────────────────────────────────────────────────

def load_benchmarks(json_paths: list[str | Path]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in json_paths:
        p = Path(p)
        if not p.exists():
            print(f"  [plots] ⚠  Introuvable : {p}")
            continue
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        method = data.get("method") or data.get("run_label", "unknown")
        for known in ("mfcc", "clap", "muq", "mert"):
            if known in method.lower():
                method = known
                break
        out[method] = data
    return out


def load_evaluations(json_paths: list[str | Path]) -> list[dict]:
    out = []
    for p in json_paths:
        p = Path(p)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                out.append(json.load(f))
    return out


# ─── G1 — Paired bar : accuracy avec vs sans RIR ────────────────────────────

def plot_rir_paired_bar(rir_evals: list[dict], out_dir: Path) -> None:
    """
    G1 — Pour chaque condition : 2 barres côte à côte (sans RIR vs avec RIR).
    Une figure par méthode si plusieurs méthodes sont présentes.
    """
    for ev in rir_evals:
        methods    = ev.get("methods", [])
        conditions = _cond_order(ev.get("conditions", []))
        results    = ev.get("results", {})

        for method in methods:
            method_data = results.get(method, {})
            cond_labels = [COND_LABELS.get(c, c) for c in conditions]

            acc_with    = [method_data.get(f"{c}_with_rir",    {}).get("top1_accuracy", 0) * 100
                           for c in conditions]
            acc_without = [method_data.get(f"{c}_without_rir", {}).get("top1_accuracy", 0) * 100
                           for c in conditions]

            x     = np.arange(len(conditions))
            w     = 0.35
            fig, ax = plt.subplots(figsize=FIG_WIDE)

            bars1 = ax.bar(x - w/2, acc_without, w, label="Sans RIR",
                           color=COLOR_WITHOUT, alpha=0.85, edgecolor="white")
            bars2 = ax.bar(x + w/2, acc_with,    w, label="Avec RIR",
                           color=COLOR_WITH,    alpha=0.85, edgecolor="white")

            _bar_label(ax, bars1)
            _bar_label(ax, bars2)

            ax.set_xticks(x)
            ax.set_xticklabels(cond_labels, fontsize=11)
            ax.set_ylabel("Précision Top-1 (%)")
            ax.set_ylim(0, 120)
            ax.set_title(f"Impact de l'augmentation RIR sur la précision — "
                         f"{METHOD_LABELS.get(method, method.upper())}")
            ax.legend(loc="upper right")
            ax.axhline(100, color="gray", linewidth=0.7, linestyle="--", alpha=0.4)

            n = ev.get("n_tracks", "?")
            ax.text(0.01, 0.97, f"n = {n} morceaux de test",
                    transform=ax.transAxes, fontsize=9, color="gray", va="top")

            fig.tight_layout()
            _save(fig, out_dir / f"rir_paired_bar_{method}.png")


# ─── G2 — Delta RIR ─────────────────────────────────────────────────────────

def plot_rir_delta(rir_evals: list[dict], out_dir: Path) -> None:
    """
    G2 — Gain Δ Top-1 accuracy = avec_RIR − sans_RIR, par condition.
    Barres vertes = amélioration, rouges = régression.
    Une figure par méthode.
    """
    for ev in rir_evals:
        methods    = ev.get("methods", [])
        conditions = _cond_order(ev.get("conditions", []))
        results    = ev.get("results", {})

        for method in methods:
            method_data = results.get(method, {})
            cond_labels = [COND_LABELS.get(c, c) for c in conditions]

            deltas = []
            for c in conditions:
                with_acc    = method_data.get(f"{c}_with_rir",    {}).get("top1_accuracy", None)
                without_acc = method_data.get(f"{c}_without_rir", {}).get("top1_accuracy", None)
                if with_acc is not None and without_acc is not None:
                    deltas.append((with_acc - without_acc) * 100)
                else:
                    deltas.append(0.0)

            x      = np.arange(len(conditions))
            colors = [COLOR_WITH if d >= 0 else COLOR_WITHOUT for d in deltas]

            fig, ax = plt.subplots(figsize=FIG_STD)
            bars = ax.bar(x, deltas, color=colors, alpha=0.88, edgecolor="white", width=0.55)

            for bar, d in zip(bars, deltas):
                if abs(d) > 1:
                    va  = "bottom" if d >= 0 else "top"
                    off = 0.5 if d >= 0 else -0.5
                    ax.text(bar.get_x() + bar.get_width() / 2, d + off,
                            f"{d:+.0f} pp", ha="center", va=va, fontsize=9,
                            fontweight="bold",
                            color=COLOR_WITH if d >= 0 else COLOR_WITHOUT)

            ax.axhline(0, color="black", linewidth=1.2)
            ax.set_xticks(x)
            ax.set_xticklabels(cond_labels, fontsize=11)
            ax.set_ylabel("Δ Précision Top-1 (points de pourcentage)")
            ax.set_title(f"Gain apporté par l'augmentation RIR — "
                         f"{METHOD_LABELS.get(method, method.upper())}\n"
                         f"(vert = amélioration, rouge = régression)")

            patches = [
                mpatches.Patch(color=COLOR_WITH,    label="Amélioration"),
                mpatches.Patch(color=COLOR_WITHOUT, label="Régression"),
            ]
            ax.legend(handles=patches, fontsize=9)

            n = ev.get("n_tracks", "?")
            ax.text(0.01, 0.97, f"n = {n} morceaux de test",
                    transform=ax.transAxes, fontsize=9, color="gray", va="top")

            fig.tight_layout()
            _save(fig, out_dir / f"rir_delta_{method}.png")


# ─── G4 — Score FAISS par morceau avec/sans RIR ─────────────────────────────

def plot_rir_faiss_scores(rir_evals: list[dict], out_dir: Path) -> None:
    """
    G4 — Score FAISS du bon morceau pour chaque morceau de test.
    Condition clean uniquement (condition de référence).
    Lignes connectant sans RIR → avec RIR pour chaque morceau.
    """
    for ev in rir_evals:
        methods = ev.get("methods", [])
        results = ev.get("results", {})

        for method in methods:
            method_data = results.get(method, {})
            tracks_with    = method_data.get("clean_with_rir",    {}).get("per_track", [])
            tracks_without = method_data.get("clean_without_rir", {}).get("per_track", [])

            if not tracks_with or not tracks_without:
                print(f"  [plots] ⚠  G4 : pas de données clean pour {method}")
                continue

            # Aligner par track_id
            by_id_with    = {r["track_id"]: r for r in tracks_with}
            by_id_without = {r["track_id"]: r for r in tracks_without}
            common_ids    = [tid for tid in by_id_with if tid in by_id_without]

            if not common_ids:
                continue

            labels      = [f"{by_id_with[tid].get('artist', '')} — "
                           f"{by_id_with[tid].get('title', tid[:8])}"
                           for tid in common_ids]
            scores_with    = [by_id_with[tid].get("faiss_score", 0)    for tid in common_ids]
            scores_without = [by_id_without[tid].get("faiss_score", 0) for tid in common_ids]

            n   = len(common_ids)
            x   = np.arange(n)
            w   = 0.35

            fig, ax = plt.subplots(figsize=(max(8, n * 1.4 + 2), 5))

            bars1 = ax.bar(x - w/2, scores_without, w, label="Sans RIR",
                           color=COLOR_WITHOUT, alpha=0.85, edgecolor="white")
            bars2 = ax.bar(x + w/2, scores_with,    w, label="Avec RIR",
                           color=COLOR_WITH,    alpha=0.85, edgecolor="white")

            # Lignes de connexion (flèche du gain)
            for i, (sw, sn) in enumerate(zip(scores_with, scores_without)):
                if sw > sn:
                    ax.annotate("", xy=(x[i] + w/2, sw), xytext=(x[i] - w/2, sn),
                                arrowprops=dict(arrowstyle="-", color="gray",
                                                lw=1.2, linestyle="dashed"))

            for bar, v in zip(bars1, scores_without):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=8)
            for bar, v in zip(bars2, scores_with):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=8)

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
            ax.set_ylabel("Score FAISS (Stage 1)")
            ax.set_title(f"Score FAISS du bon morceau avec et sans RIR — "
                         f"{METHOD_LABELS.get(method, method.upper())}"
                         f"\n(condition clean, plus haut = meilleure identification)")
            ax.legend()

            fig.tight_layout()
            _save(fig, out_dir / f"rir_faiss_scores_{method}.png")


# ─── G6 — Accuracy par méthode × condition (multi-tracks) ───────────────────

def plot_method_accuracy(evaluations: list[dict], out_dir: Path) -> None:
    """
    G6 — Grouped bar chart : Top-1 accuracy par méthode × condition.
    Barres groupées par méthode, avec écart-type sur les N morceaux.
    """
    # Agréger tous les JSON d'évaluation
    methods_seen: list[str] = []
    conds_seen:   list[str] = []

    # {method: {condition: [top1_per_track]}}
    raw: dict[str, dict[str, list[float]]] = {}

    for ev in evaluations:
        for method, cond_dict in ev.get("results", {}).items():
            if method not in methods_seen:
                methods_seen.append(method)
            if method not in raw:
                raw[method] = {}
            for cond, metrics in cond_dict.items():
                if cond not in conds_seen:
                    conds_seen.append(cond)
                per = [float(r.get("top1", False)) for r in metrics.get("per_track", [])]
                raw.setdefault(method, {}).setdefault(cond, []).extend(per)

    if not methods_seen or not conds_seen:
        return

    conds_sorted = _cond_order(conds_seen)
    cond_labels  = [COND_LABELS.get(c, c) for c in conds_sorted]
    nc           = len(conds_sorted)
    nm           = len(methods_seen)
    bar_w        = 0.7 / nm
    x            = np.arange(nc)

    fig, ax = plt.subplots(figsize=(max(10, nc * 1.6 + 2), 5))

    for mi, method in enumerate(methods_seen):
        color  = METHOD_COLORS.get(method, "#333")
        label  = METHOD_LABELS.get(method, method.upper())
        means, errs = [], []
        for cond in conds_sorted:
            vals = raw.get(method, {}).get(cond, [])
            if vals:
                means.append(float(np.mean(vals)) * 100)
                errs.append(float(np.std(vals))  * 100)
            else:
                means.append(0.0)
                errs.append(0.0)

        offset = (mi - nm / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, means, bar_w, label=label, color=color,
                      alpha=0.88, edgecolor="white",
                      yerr=errs, capsize=4,
                      error_kw={"linewidth": 1.5, "color": "dimgray"})
        _bar_label(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontsize=11)
    ax.set_ylabel("Précision Top-1 (%)")
    ax.set_ylim(0, 120)
    ax.set_title("Précision Top-1 par méthode et condition de dégradation\n"
                 "(barres d'erreur = écart-type sur les morceaux de test)")
    ax.legend(loc="upper right")
    ax.axhline(100, color="gray", linewidth=0.7, linestyle="--", alpha=0.4)

    n_tracks = max(
        (ev.get("n_tracks", 0) for ev in evaluations), default=0
    )
    ax.text(0.01, 0.97, f"n = {n_tracks} morceaux de test",
            transform=ax.transAxes, fontsize=9, color="gray", va="top")

    fig.tight_layout()
    _save(fig, out_dir / "method_accuracy.png")


# ─── G9 — Stage 1 vs Stage 2 ────────────────────────────────────────────────

def plot_stage_comparison(evaluations: list[dict], out_dir: Path) -> None:
    """
    G9 — Grouped bar : Top-1 accuracy Stage 1 (FAISS seul) vs Stage 2 (+ fingerprint).
    Montre le gain apporté par le re-ranking fingerprint.
    """
    methods_seen: list[str] = []
    conds_seen:   list[str] = []

    # {method: {condition: (s1_acc, s2_acc)}}
    data: dict[str, dict[str, tuple[float, float]]] = {}

    for ev in evaluations:
        for method, cond_dict in ev.get("results", {}).items():
            if method not in methods_seen:
                methods_seen.append(method)
            for cond, metrics in cond_dict.items():
                if cond not in conds_seen:
                    conds_seen.append(cond)
                s1 = metrics.get("top1_stage1_accuracy", 0) * 100
                s2 = metrics.get("top1_accuracy",        0) * 100
                data.setdefault(method, {})[cond] = (s1, s2)

    if not methods_seen or not conds_seen:
        return

    conds_sorted = _cond_order(conds_seen)
    cond_labels  = [COND_LABELS.get(c, c) for c in conds_sorted]
    nc           = len(conds_sorted)
    nm           = len(methods_seen)
    group_w      = 0.7
    bar_w        = group_w / (nm * 2)
    x            = np.arange(nc)

    fig, ax = plt.subplots(figsize=(max(10, nc * 1.8 + 2), 5))

    for mi, method in enumerate(methods_seen):
        color = METHOD_COLORS.get(method, "#333")
        label = METHOD_LABELS.get(method, method.upper())
        s1s   = [data.get(method, {}).get(c, (0, 0))[0] for c in conds_sorted]
        s2s   = [data.get(method, {}).get(c, (0, 0))[1] for c in conds_sorted]

        group_offset = (mi - nm / 2 + 0.5) * (bar_w * 2 + 0.02)
        bars1 = ax.bar(x + group_offset - bar_w/2, s1s, bar_w,
                       color=color, alpha=0.45, edgecolor="white",
                       hatch="//", label=f"{label} Stage 1")
        bars2 = ax.bar(x + group_offset + bar_w/2, s2s, bar_w,
                       color=color, alpha=0.88, edgecolor="white",
                       label=f"{label} Stage 2")

        for b1, b2, s1v, s2v in zip(bars1, bars2, s1s, s2s):
            if s2v > s1v + 2:
                ax.annotate(
                    "", xy=(b2.get_x() + b2.get_width()/2, s2v),
                    xytext=(b1.get_x() + b1.get_width()/2, s1v),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=1.5, mutation_scale=10),
                )

    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontsize=11)
    ax.set_ylabel("Précision Top-1 (%)")
    ax.set_ylim(0, 120)
    ax.set_title("Impact du re-ranking par fingerprinting\n"
                 "(hachuré = Stage 1 FAISS seul | plein = Stage 1 + Stage 2)")
    ax.legend(loc="upper right", ncol=nm, fontsize=9)

    fig.tight_layout()
    _save(fig, out_dir / "stage_comparison.png")


# ─── G11 — Impact de la durée de l'extrait ──────────────────────────────────

def plot_duration_impact(evaluations: list[dict], out_dir: Path) -> None:
    """
    G11 — Précision Top-1 en fonction de la durée de l'extrait (5s, 10s, 15s, 30s).
    Condition 'clean' uniquement.
    Une ligne par méthode.
    """
    # {method: {duration_s: [top1]}}
    raw: dict[str, dict[int, list[float]]] = {}

    for ev in evaluations:
        for method, cond_dict in ev.get("results", {}).items():
            tracks = cond_dict.get("clean", {}).get("per_track", [])
            if not tracks:
                continue
            for t in tracks:
                dur = t.get("duration_s")
                if dur is None:
                    continue
                dur = int(dur)
                top1 = float(t.get("top1", False))
                raw.setdefault(method, {}).setdefault(dur, []).append(top1)

    if not raw:
        print("  [plots] ⚠  G11 : pas de données duration_s dans le manifest.\n"
              "           Téléchargez des clips avec --duration 5 / 10 / 15 / 30.")
        return

    durations = sorted({d for m in raw.values() for d in m})
    if len(durations) < 2:
        print("  [plots] ⚠  G11 : moins de 2 durées différentes — "
              "téléchargez des clips avec --duration 5, 10, 15, 30")
        return

    fig, ax = plt.subplots(figsize=FIG_STD)
    plotted = False

    for method, dur_dict in raw.items():
        xs, ys, errs = [], [], []
        for d in durations:
            vals = dur_dict.get(d, [])
            if vals:
                xs.append(d)
                ys.append(float(np.mean(vals)) * 100)
                errs.append(float(np.std(vals))  * 100)
        if len(xs) < 2:
            continue
        color = METHOD_COLORS.get(method, "#333")
        label = METHOD_LABELS.get(method, method.upper())
        ax.errorbar(xs, ys, yerr=errs, marker="o", linewidth=2.5, markersize=9,
                    capsize=5, color=color, label=label)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Durée de l'extrait (secondes)")
    ax.set_ylabel("Précision Top-1 (%)")
    ax.set_ylim(-5, 110)
    ax.set_xticks(durations)
    ax.set_xticklabels([f"{d} s" for d in durations])
    ax.set_title("Précision en fonction de la durée de l'extrait\n"
                 "(condition clean — barres d'erreur = écart-type)")
    ax.axhline(50, color="gray", linewidth=0.8, linestyle="--", alpha=0.5,
               label="_seuil 50 %")
    ax.legend()

    fig.tight_layout()
    _save(fig, out_dir / "duration_impact.png")


# ─── G12 — Heatmap accuracy (%) ─────────────────────────────────────────────

def plot_heatmap_accuracy(evaluations: list[dict], out_dir: Path) -> None:
    """
    G12 — Heatmap méthodes × conditions.
    Valeurs : Top-1 accuracy en % (colormap continue vert = bon, rouge = mauvais).
    """
    methods_seen: list[str] = []
    conds_seen:   list[str] = []
    acc_matrix: dict[str, dict[str, float]] = {}

    for ev in evaluations:
        for method, cond_dict in ev.get("results", {}).items():
            if method not in methods_seen:
                methods_seen.append(method)
            for cond, metrics in cond_dict.items():
                if cond not in conds_seen:
                    conds_seen.append(cond)
                acc_matrix.setdefault(method, {})[cond] = \
                    metrics.get("top1_accuracy", 0) * 100

    if not methods_seen or not conds_seen:
        return

    conds_sorted = _cond_order(conds_seen)
    cond_labels  = [COND_LABELS.get(c, c) for c in conds_sorted]
    nr, nc       = len(methods_seen), len(conds_sorted)

    matrix = np.array([
        [acc_matrix.get(m, {}).get(c, 0.0) for c in conds_sorted]
        for m in methods_seen
    ], dtype=float)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rg", ["#e74c3c", "#f39c12", "#27ae60"]
    )

    fig_w = max(6, nc * 1.5 + 2.5)
    fig_h = max(3, nr * 1.2 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    plt.colorbar(im, ax=ax, label="Précision Top-1 (%)", shrink=0.85)

    for i in range(nr):
        for j in range(nc):
            v    = matrix[i, j]
            text_c = "white" if v < 35 or v > 75 else "black"
            ax.text(j, i, f"{v:.0f} %", ha="center", va="center",
                    fontsize=11, color=text_c, fontweight="bold")

    ax.set_xticks(range(nc))
    ax.set_xticklabels(cond_labels, fontsize=11)
    ax.set_yticks(range(nr))
    ax.set_yticklabels([METHOD_LABELS.get(m, m.upper()) for m in methods_seen],
                       fontsize=12, fontweight="bold")
    ax.tick_params(left=False, bottom=False)
    ax.set_title("Précision Top-1 (%) par méthode et condition", pad=12)

    n_tracks = max((ev.get("n_tracks", 0) for ev in evaluations), default=0)
    ax.set_xlabel(f"Condition de dégradation  (n = {n_tracks} morceaux de test)", fontsize=10)

    fig.tight_layout()
    _save(fig, out_dir / "heatmap_accuracy.png")


# ─── Point d'entrée public ──────────────────────────────────────────────────

def run_plots(
    benchmark_jsons:  list[str | Path] | None = None,
    eval_jsons:       list[str | Path] | None = None,
    rir_eval_jsons:   list[str | Path] | None = None,
    out_dir:          Path | None = None,
) -> None:
    """
    Génère tous les graphiques disponibles selon les données fournies.

    Args:
        benchmark_jsons  : JSON(s) de benchmark (un par méthode).
        eval_jsons       : JSON(s) d'évaluation multi-tracks (evaluate.py).
        rir_eval_jsons   : JSON(s) d'évaluation RIR (rir_evaluate).
        out_dir          : dossier de sortie (défaut : results/plots/).
    """
    out_dir = Path(out_dir) if out_dir else PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[plots] Génération → {out_dir}\n")

    evaluations: list[dict] = []
    if eval_jsons:
        evaluations = load_evaluations(eval_jsons)
        print(f"[plots] {len(evaluations)} évaluation(s) chargée(s)\n")

    rir_evals: list[dict] = []
    if rir_eval_jsons:
        rir_evals = load_evaluations(rir_eval_jsons)
        print(f"[plots] {len(rir_evals)} évaluation(s) RIR chargée(s)\n")

    # ── Graphiques RIR ──
    if rir_evals:
        print("── Comparaison RIR ──")
        plot_rir_paired_bar(rir_evals, out_dir)    # G1
        plot_rir_delta(rir_evals, out_dir)         # G2
        plot_rir_faiss_scores(rir_evals, out_dir)  # G4
        print()

    # ── Graphiques évaluation pipeline ──
    if evaluations:
        print("── Pipeline ──")
        plot_method_accuracy(evaluations, out_dir)    # G6
        plot_stage_comparison(evaluations, out_dir)   # G9
        plot_duration_impact(evaluations, out_dir)    # G11
        plot_heatmap_accuracy(evaluations, out_dir)   # G12
        print()

    if not rir_evals and not evaluations:
        print("[plots] ⚠  Aucune donnée. Utilisez --eval et/ou --rir-eval.\n")
        _print_workflow()
        return

    total = sum([bool(rir_evals) * 3, bool(evaluations) * 4])
    print(f"[plots] {total} graphique(s) généré(s)")
    _print_workflow()


def _print_workflow():
    print("\nWorkflow complet :")
    print("  python manage.py evaluate     --methods mfcc clap")
    print("  python manage.py rir-evaluate --methods clap")
    print("  python manage.py plots \\")
    print("      --eval     results/eval/eval_*.json \\")
    print("      --rir-eval results/eval/rir_eval_*.json")
