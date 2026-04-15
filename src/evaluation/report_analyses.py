"""
src/evaluation/report_analyses.py

Report-oriented analyses:
  - studio-mic
  - duration
  - stage12
  - rir
  - mic-conditions

Each analysis produces:
  - a JSON
  - a Markdown summary
  - readable graphs for the report
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from src import config
from src.evaluation.evaluate import CONDITION_LABELS, _evaluate_one, load_manifest, run_rir_evaluate

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "eval"
PLOTS_DIR = ROOT / "results" / "plots"
CACHE_DIR = RESULTS_DIR / "cache"
console = Console()


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dirs(out_dir: Path | None = None) -> Path:
    target = Path(out_dir) if out_dir else RESULTS_DIR
    target.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return target


def _save_fig(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  [eval] ✓ plot {path.name}")
    return str(path)


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in columns})
    print(f"[eval] CSV      → {path}")
    return path


def _cleanup_legacy_outputs(out_dir: Path, stem: str, suffixes: tuple[str, ...] = (".json", ".md")) -> None:
    for suffix in suffixes:
        for old_path in out_dir.glob(f"{stem}_*{suffix}"):
            if old_path.is_file():
                old_path.unlink(missing_ok=True)


def _format_table_value(key: str, value) -> str:
    if value is None:
        return "NF"
    if key == "n_queries":
        return str(value)
    if isinstance(value, float):
        return f"{value:.1f}" if "pct" in key else f"{value:.2f}"
    return str(value)


def _save_table_plot(
    title: str,
    rows: list[dict],
    columns: list[tuple[str, str]],
    path: Path,
    subtitle: str | None = None,
) -> str | None:
    if not rows:
        return None

    cell_text = [[_format_table_value(key, row.get(key)) for key, _ in columns] for row in rows]
    col_labels = [label for _, label in columns]

    n_rows = len(rows)
    fig_h = max(2.8, 1.85 + 0.5 * (n_rows + 1))
    fig, ax = plt.subplots(figsize=(13.8, fig_h), facecolor="#f8fafc")
    ax.axis("off")
    fig.text(0.015, 0.965, title, fontsize=18, fontweight="bold", color="#0f172a", va="top")
    if subtitle:
        fig.text(0.015, 0.925, subtitle, fontsize=10.5, color="#64748b", va="top")

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="upper left",
        cellLoc="center",
        colLoc="center",
        bbox=[0.015, 0.0, 0.97, 0.84],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5 if n_rows >= 8 else 10.5)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#d9e2ef")
        cell.set_linewidth(0.8)
        if row_idx == 0:
            cell.set_facecolor("#dbeafe")
            cell.set_text_props(weight="bold", color="#0f172a")
            cell.set_height(0.078)
        else:
            cell.set_facecolor("#ffffff" if row_idx % 2 == 1 else "#f8fbff")
            if col_idx == 0:
                cell.set_text_props(weight="bold", color="#1e3a8a")
            elif col_idx == 1:
                cell.set_text_props(color="#334155")
            else:
                cell.set_text_props(color="#0f172a")
            cell.set_height(0.074)

    return _save_fig(fig, path)


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_pct(values: list[bool]) -> float:
    return round(100 * _safe_mean([float(v) for v in values]), 1) if values else 0.0


def _build_pipeline_overview(rows: list[dict]) -> list[dict]:
    scenario_specs = [
        {
            "id": "studio",
            "label": "File excerpt",
            "description": "Direct digital excerpt. The query remains very close to the reference track.",
            "technical": "Stage 1 already retrieves the correct song in many cases; fingerprinting mostly acts as a final validation step.",
            "filter": lambda r: r.get("query_kind") == "studio",
        },
        {
            "id": "micro_clean",
            "label": "Microphone (music only)",
            "description": "Microphone capture without overlapping speech. Room acoustics and hardware still degrade the signal.",
            "technical": "Stage 1 drops sharply on real microphone captures, then fingerprint reranking recovers many of the correct candidates.",
            "filter": lambda r: r.get("query_kind") == "micro" and r.get("speech") == "clean",
        },
        {
            "id": "micro_speech",
            "label": "Microphone + speech",
            "description": "Hardest case: someone speaks while the music is playing.",
            "technical": "Speech perturbs the global embeddings; fingerprinting stays more robust because it relies on local musical structures.",
            "filter": lambda r: r.get("query_kind") == "micro" and r.get("speech") == "speech",
        },
    ]

    scenarios: list[dict] = []
    for spec in scenario_specs:
        subset = [r for r in rows if spec["filter"](r)]
        stage1_pct = _safe_pct([bool(r.get("stage1_top1")) for r in subset])
        stage2_pct = _safe_pct([bool(r.get("top1")) for r in subset])
        stage1_ranks = [r["stage1_rank"] for r in subset if r.get("stage1_rank") is not None]
        final_ranks = [r["final_rank"] for r in subset if r.get("final_rank") is not None]
        scenarios.append({
            "id": spec["id"],
            "label": spec["label"],
            "description": spec["description"],
            "technical": spec["technical"],
            "n_queries": len(subset),
            "stage1_pct": stage1_pct,
            "stage2_pct": stage2_pct,
            "gain_pct": round(stage2_pct - stage1_pct, 1),
            "mean_stage1_rank": round(_safe_mean(stage1_ranks), 2) if stage1_ranks else None,
            "mean_final_rank": round(_safe_mean(final_ranks), 2) if final_ranks else None,
        })
    return scenarios


def _cleanup_base_suite_plots() -> None:
    for name in (
        "scatter_studio_micro_stage1_vs_final_rank.png",
        "scatter_duration_vs_final_rank.png",
        "scatter_stage1_vs_stage2_rank.png",
        "scatter_studio_vs_micro_rank.png",
        "scatter_micro_clean_vs_speech_rank.png",
    ):
        (PLOTS_DIR / name).unlink(missing_ok=True)


def _cleanup_rir_plots() -> None:
    for pattern in (
        "rir_paired_bar_*.png",
        "rir_delta_*.png",
        "rir_faiss_scores_*.png",
    ):
        for path in PLOTS_DIR.glob(pattern):
            path.unlink(missing_ok=True)


def _rir_metrics_from_rows(rows: list[dict]) -> dict:
    ranks = [r.get("rank") for r in rows]
    valid_ranks = [r for r in ranks if r is not None]
    return {
        "top1_pct": _safe_pct([r is not None and r <= 1 for r in ranks]),
        "top5_pct": _safe_pct([r is not None and r <= 5 for r in ranks]),
        "top10_pct": _safe_pct([r is not None and r <= 10 for r in ranks]),
        "mean_rank": round(_safe_mean(valid_ranks), 2) if valid_ranks else None,
    }


def _rir_pair_key(row: dict) -> str:
    return f"{row.get('track_id')}::{row.get('filename')}"


def _rir_pairs_for_condition(result: dict, method: str, condition: str) -> list[dict]:
    method_data = result.get("results", {}).get(method, {})
    without_rows = method_data.get(f"{condition}_without_rir", {}).get("per_track", [])
    with_rows = method_data.get(f"{condition}_with_rir", {}).get("per_track", [])
    with_map = {_rir_pair_key(row): row for row in with_rows}

    pairs: list[dict] = []
    for row_no in without_rows:
        key = _rir_pair_key(row_no)
        row_with = with_map.get(key)
        attrs = _parse_query_attrs({
            "filename": row_no.get("filename", ""),
            "position": row_no.get("position", ""),
            "duration_s": row_no.get("duration_s"),
        })
        pairs.append({
            "method": method,
            "condition": condition,
            "condition_label": CONDITION_LABELS.get(condition, condition),
            "track_id": row_no.get("track_id"),
            "artist": row_no.get("artist", ""),
            "title": row_no.get("title", ""),
            "filename": row_no.get("filename", ""),
            "duration_s": float(row_no.get("duration_s") or 0.0),
            **attrs,
            "without_rir_rank": row_no.get("rank"),
            "without_rir_faiss_score": row_no.get("faiss_score", 0.0),
            "with_rir_rank": row_with.get("rank") if row_with else None,
            "with_rir_faiss_score": row_with.get("faiss_score", 0.0) if row_with else 0.0,
        })
    return pairs


def _build_rir_overview_rows(clean_pairs: list[dict]) -> list[dict]:
    scenario_specs = [
        ("File excerpt", lambda r: r.get("query_kind") == "studio"),
        ("Microphone (music only)", lambda r: r.get("query_kind") == "micro" and r.get("speech") == "clean"),
        ("Microphone + speech", lambda r: r.get("query_kind") == "micro" and r.get("speech") == "speech"),
    ]
    rows: list[dict] = []
    for label, pred in scenario_specs:
        subset = [r for r in clean_pairs if pred(r)]
        without_metrics = _rir_metrics_from_rows([{"rank": r.get("without_rir_rank")} for r in subset])
        with_metrics = _rir_metrics_from_rows([{"rank": r.get("with_rir_rank")} for r in subset])
        rows.append({
            "Scenario": label,
            "N queries": len(subset),
            "Without RIR Top-10 (%)": without_metrics["top10_pct"],
            "With RIR Top-10 (%)": with_metrics["top10_pct"],
            "Top-10 gain (pp)": round(with_metrics["top10_pct"] - without_metrics["top10_pct"], 1),
            "Without RIR mean rank": without_metrics["mean_rank"],
            "With RIR mean rank": with_metrics["mean_rank"],
        })
    return rows


def _plot_rir_pipeline_overview(method: str, clean_pairs: list[dict]) -> str | None:
    overview_rows = _build_rir_overview_rows(clean_pairs)
    if not any(row["N queries"] for row in overview_rows):
        return None

    fig, ax = plt.subplots(figsize=(10.8, 6.6), facecolor="#f8fafc")
    ax.set_facecolor("white")
    x = np.arange(len(overview_rows))
    width = 0.28
    without_color = "#cbd5e1"
    with_color = "#2563eb"
    without_vals = [row["Without RIR Top-10 (%)"] for row in overview_rows]
    with_vals = [row["With RIR Top-10 (%)"] for row in overview_rows]

    ax.bar(x - width / 2, without_vals, width=width, color=without_color, edgecolor="none", label="Without RIR")
    ax.bar(x + width / 2, with_vals, width=width, color=with_color, edgecolor="none", label="With RIR")

    for xpos, value in zip(x - width / 2, without_vals):
        ax.text(xpos, value + 2.2, f"{value:.1f}%", ha="center", va="bottom", fontsize=10, color="#64748b", fontweight="bold")
    for xpos, value in zip(x + width / 2, with_vals):
        ax.text(xpos, value + 2.2, f"{value:.1f}%", ha="center", va="bottom", fontsize=10, color="#0f172a", fontweight="bold")

    ax.set_ylim(0, 108)
    ax.set_xticks(x)
    ax.set_xticklabels([row["Scenario"] for row in overview_rows], fontsize=11)
    ax.set_ylabel("Top-10 accuracy (%)")
    ax.set_title(f"RIR augmentation impact — {method.upper()}")
    ax.grid(axis="y", color="#e2e8f0", linestyle="--", linewidth=1)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", length=0, pad=12)
    ax.tick_params(axis="y", colors="#64748b")
    legend = ax.legend(loc="upper left", frameon=False, ncol=2, bbox_to_anchor=(0.0, 1.02))
    for text in legend.get_texts():
        text.set_color("#475569")

    fig.text(
        0.5,
        0.04,
        "Measure: Top-10 accuracy before and after adding RIR-augmented vectors to the reference index.",
        color="#64748b",
        fontsize=10,
        ha="center",
    )

    suffix = "" if method == config.EMBEDDING_METHOD else f"_{method}"
    return _save_fig(fig, PLOTS_DIR / f"rir_pipeline_overview{suffix}.png")


def _build_rir_topk_summary_rows(clean_pairs: list[dict]) -> list[dict]:
    summary_rows: list[dict] = []

    def add_group(category: str, subcategory: str, subset: list[dict]) -> None:
        if not subset:
            return
        without = _rir_metrics_from_rows([{"rank": r.get("without_rir_rank")} for r in subset])
        with_r = _rir_metrics_from_rows([{"rank": r.get("with_rir_rank")} for r in subset])
        summary_rows.append({
            "Category": category,
            "Subcategory": subcategory,
            "N queries": len(subset),
            "Without RIR Top-1 (%)": without["top1_pct"],
            "Without RIR Top-5 (%)": without["top5_pct"],
            "Without RIR Top-10 (%)": without["top10_pct"],
            "With RIR Top-1 (%)": with_r["top1_pct"],
            "With RIR Top-5 (%)": with_r["top5_pct"],
            "With RIR Top-10 (%)": with_r["top10_pct"],
            "Top-1 gain (pp)": round(with_r["top1_pct"] - without["top1_pct"], 1),
            "Top-5 gain (pp)": round(with_r["top5_pct"] - without["top5_pct"], 1),
            "Top-10 gain (pp)": round(with_r["top10_pct"] - without["top10_pct"], 1),
            "Without RIR mean rank": without["mean_rank"],
            "With RIR mean rank": with_r["mean_rank"],
        })

    add_group("Overall", "All", clean_pairs)
    add_group("Query type", "Studio", [r for r in clean_pairs if r.get("query_kind") == "studio"])
    add_group("Query type", "Microphone", [r for r in clean_pairs if r.get("query_kind") == "micro"])

    for duration in (5, 15, 30):
        add_group("Duration", f"{duration} s", [r for r in clean_pairs if r.get("query_kind") == "studio" and r.get("duration_bucket") == duration])

    mic_rows = [r for r in clean_pairs if r.get("query_kind") == "micro"]
    add_group("Microphone speech", "No speech", [r for r in mic_rows if r.get("speech") == "clean"])
    add_group("Microphone speech", "With speech", [r for r in mic_rows if r.get("speech") == "speech"])

    for distance, label in (("close", "Close"), ("normal", "Normal"), ("far", "Far")):
        add_group("Microphone distance", label, [r for r in mic_rows if r.get("distance") == distance])

    for distance, dist_label in (("close", "Close"), ("normal", "Normal"), ("far", "Far")):
        for speech, speech_label in (("clean", "no speech"), ("speech", "with speech")):
            add_group(
                "Microphone condition",
                f"{dist_label} + {speech_label}",
                [r for r in mic_rows if r.get("distance") == distance and r.get("speech") == speech],
            )

    add_group("Scenario", "File excerpt", [r for r in clean_pairs if r.get("query_kind") == "studio"])
    add_group("Scenario", "Microphone (music only)", [r for r in mic_rows if r.get("speech") == "clean"])
    add_group("Scenario", "Microphone + speech", [r for r in mic_rows if r.get("speech") == "speech"])
    return summary_rows


def _build_rir_condition_summary_rows(result: dict, method: str) -> list[dict]:
    rows: list[dict] = []
    for condition in result.get("conditions", []):
        pairs = _rir_pairs_for_condition(result, method, condition)
        without = _rir_metrics_from_rows([{"rank": r.get("without_rir_rank")} for r in pairs])
        with_r = _rir_metrics_from_rows([{"rank": r.get("with_rir_rank")} for r in pairs])
        rows.append({
            "Method": method,
            "Condition": CONDITION_LABELS.get(condition, condition),
            "N queries": len(pairs),
            "Without RIR Top-1 (%)": without["top1_pct"],
            "Without RIR Top-5 (%)": without["top5_pct"],
            "Without RIR Top-10 (%)": without["top10_pct"],
            "With RIR Top-1 (%)": with_r["top1_pct"],
            "With RIR Top-5 (%)": with_r["top5_pct"],
            "With RIR Top-10 (%)": with_r["top10_pct"],
            "Top-1 gain (pp)": round(with_r["top1_pct"] - without["top1_pct"], 1),
            "Top-5 gain (pp)": round(with_r["top5_pct"] - without["top5_pct"], 1),
            "Top-10 gain (pp)": round(with_r["top10_pct"] - without["top10_pct"], 1),
            "Without RIR mean rank": without["mean_rank"],
            "With RIR mean rank": with_r["mean_rank"],
        })
    return rows


def _topk_metrics(rows: list[dict]) -> dict:
    final_ranks = [r.get("final_rank") for r in rows]
    stage1_ranks = [r.get("stage1_rank") for r in rows]
    return {
        "n_queries": len(rows),
        "final_top1_pct": _safe_pct([r is not None and r <= 1 for r in final_ranks]),
        "final_top5_pct": _safe_pct([r is not None and r <= 5 for r in final_ranks]),
        "final_top10_pct": _safe_pct([r is not None and r <= 10 for r in final_ranks]),
        "stage1_top1_pct": _safe_pct([r is not None and r <= 1 for r in stage1_ranks]),
        "stage1_top5_pct": _safe_pct([r is not None and r <= 5 for r in stage1_ranks]),
        "stage1_top10_pct": _safe_pct([r is not None and r <= 10 for r in stage1_ranks]),
        "mean_stage1_rank": round(_safe_mean([r for r in stage1_ranks if r is not None]), 2) if any(r is not None for r in stage1_ranks) else None,
        "mean_final_rank": round(_safe_mean([r for r in final_ranks if r is not None]), 2) if any(r is not None for r in final_ranks) else None,
    }


def _build_topk_summary_rows(rows: list[dict]) -> list[dict]:
    summary_rows: list[dict] = []
    category_labels = {
        "overall": "Overall",
        "query_kind": "Query type",
        "duration": "Duration",
        "micro_speech": "Microphone speech",
        "micro_distance": "Microphone distance",
        "micro_condition": "Microphone condition",
        "pipeline_overview": "Scenario",
    }
    subcategory_labels = {
        "all": "All",
        "studio": "Studio",
        "micro": "Microphone",
        "clean": "No speech",
        "speech": "With speech",
        "close": "Close",
        "normal": "Normal",
        "far": "Far",
        "close_clean": "Close + no speech",
        "close_speech": "Close + with speech",
        "normal_clean": "Normal + no speech",
        "normal_speech": "Normal + with speech",
        "far_clean": "Far + no speech",
        "5s": "5 s",
        "15s": "15 s",
        "30s": "30 s",
        "Extrait (fichier)": "File excerpt",
        "Micro (musique seule)": "Microphone (music only)",
        "Micro + voix": "Microphone + speech",
    }

    def add_group(category: str, subcategory: str, subset: list[dict]) -> None:
        if not subset:
            return
        metrics = _topk_metrics(subset)
        summary_rows.append({
            "category": category,
            "subcategory": subcategory,
            "category_label": category_labels.get(category, category),
            "subcategory_label": subcategory_labels.get(subcategory, subcategory),
            **metrics,
        })

    add_group("overall", "all", rows)

    for kind in ("studio", "micro"):
        add_group("query_kind", kind, [r for r in rows if r.get("query_kind") == kind])

    for duration in (5, 15, 30):
        add_group("duration", f"{duration}s", [r for r in rows if r.get("query_kind") == "studio" and r.get("duration_bucket") == duration])

    mic_rows = [r for r in rows if r.get("query_kind") == "micro"]
    for speech in ("clean", "speech"):
        add_group("micro_speech", speech, [r for r in mic_rows if r.get("speech") == speech])

    for distance in ("close", "normal", "far"):
        add_group("micro_distance", distance, [r for r in mic_rows if r.get("distance") == distance])

    for distance in ("close", "normal", "far"):
        for speech in ("clean", "speech"):
            add_group(
                "micro_condition",
                f"{distance}_{speech}",
                [r for r in mic_rows if r.get("distance") == distance and r.get("speech") == speech],
            )

    overview_map = {
        "Extrait (fichier)": [r for r in rows if r.get("query_kind") == "studio"],
        "Micro (musique seule)": [r for r in mic_rows if r.get("speech") == "clean"],
        "Micro + voix": [r for r in mic_rows if r.get("speech") == "speech"],
    }
    for label, subset in overview_map.items():
        add_group("pipeline_overview", label, subset)

    return summary_rows


def _topk_summary_csv_rows(summary_rows: list[dict]) -> list[dict]:
    return [
        {
            "Category": row["category_label"],
            "Subcategory": row["subcategory_label"],
            "N queries": row["n_queries"],
            "Stage 1 Top-1 (%)": row["stage1_top1_pct"],
            "Stage 1 Top-5 (%)": row["stage1_top5_pct"],
            "Stage 1 Top-10 (%)": row["stage1_top10_pct"],
            "Final Top-1 (%)": row["final_top1_pct"],
            "Final Top-5 (%)": row["final_top5_pct"],
            "Final Top-10 (%)": row["final_top10_pct"],
            "Mean Stage 1 rank": row["mean_stage1_rank"],
            "Mean Final rank": row["mean_final_rank"],
        }
        for row in summary_rows
    ]


def _plot_pipeline_overview(rows: list[dict]) -> tuple[list[dict], str | None]:
    scenarios = _build_pipeline_overview(rows)
    if not any(s["n_queries"] for s in scenarios):
        return scenarios, None

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 18,
        "axes.labelsize": 11,
    })

    fig, ax = plt.subplots(figsize=(10.8, 6.6), facecolor="#f8fafc")
    ax.set_facecolor("white")

    x = np.arange(len(scenarios))
    width = 0.28
    stage1_color = "#93c5fd"
    stage2_color = "#1d4ed8"

    stage1_vals = [s["stage1_pct"] for s in scenarios]
    stage2_vals = [s["stage2_pct"] for s in scenarios]

    ax.bar(x - width / 2, stage1_vals, width=width, color=stage1_color, edgecolor="none", label="Stage 1 (FAISS)")
    ax.bar(x + width / 2, stage2_vals, width=width, color=stage2_color, edgecolor="none", label="Stage 2 (Fingerprinting)")

    for xpos, value in zip(x - width / 2, stage1_vals):
        ax.text(xpos, value + 2.2, f"{value:.1f}%", ha="center", va="bottom", fontsize=10, color="#64748b", fontweight="bold")
    for xpos, value in zip(x + width / 2, stage2_vals):
        ax.text(xpos, value + 2.2, f"{value:.1f}%", ha="center", va="bottom", fontsize=10, color="#0f172a", fontweight="bold")

    ax.set_ylim(0, 108)
    ax.set_xticks(x)
    ax.set_xticklabels([s["label"] for s in scenarios], fontsize=11)
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("Pipeline resilience overview")
    ax.grid(axis="y", color="#e2e8f0", linestyle="--", linewidth=1)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", length=0, pad=12)
    ax.tick_params(axis="y", colors="#64748b")
    legend = ax.legend(loc="upper left", frameon=False, ncol=2, bbox_to_anchor=(0.0, 1.02))
    for text in legend.get_texts():
        text.set_color("#475569")

    fig.text(
        0.5,
        0.04,
        "Measure: Top-1 accuracy before reranking (Stage 1) and after fingerprint-based reranking (Stage 2).",
        color="#64748b",
        fontsize=10,
        ha="center",
    )

    plot_path = _save_fig(fig, PLOTS_DIR / "pipeline_resilience_overview.png")
    return scenarios, plot_path


def _limit_manifest(manifest: list[dict], n_tracks: int) -> list[dict]:
    if n_tracks <= 0:
        return manifest
    kept = []
    seen: set[str] = set()
    for entry in manifest:
        tid = entry["track_id"]
        if tid not in seen and len(seen) >= n_tracks:
            continue
        seen.add(tid)
        kept.append(entry)
    return kept


def _manifest_eval_order(manifest: list[dict]) -> list[dict]:
    """
    Reorders the manifest to process queries by track.

    Objective:
    - keep all variants of the same track side by side;
    - improve fingerprint cache locality;
    - avoid going through all reference_clips first then all micros.

    Intra-track order:
    1. studio then micro
    2. increasing duration
    3. filename for stability
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    first_seen_order: list[str] = []

    for entry in manifest:
        tid = entry["track_id"]
        if tid not in grouped:
            first_seen_order.append(tid)
        grouped[tid].append(entry)

    ordered: list[dict] = []
    for tid in first_seen_order:
        entries = grouped[tid]
        ordered.extend(sorted(
            entries,
            key=lambda e: (
                0 if _parse_query_attrs(e)["query_kind"] == "studio" else 1,
                float(e.get("duration_s") or 0.0),
                e.get("filename", ""),
            ),
        ))
    return ordered


def _parse_query_attrs(entry: dict) -> dict:
    pos = str(entry.get("position", "")).lower()
    filename = Path(entry["filename"]).stem
    query_kind = "micro" if pos.startswith("mic_") or "-mic-" in filename else "studio"
    distance = None
    speech = None
    if pos.startswith("mic_"):
        parts = pos.split("_")
        if len(parts) >= 3:
            _, distance, speech = parts[:3]
    elif "-mic-" in filename:
        parts = filename.split("-mic-")[-1].split("-")
        if len(parts) >= 2:
            distance, speech = parts[0], parts[1]

    duration = float(entry.get("duration_s") or 0.0)
    duration_bucket = int(round(duration)) if duration else None
    return {
        "query_kind": query_kind,
        "distance": distance,
        "speech": speech,
        "duration_bucket": duration_bucket,
    }


def _manifest_signature(manifest: list[dict], methods: list[str]) -> str:
    payload = {
        "methods": methods,
        "entries": [
            {
                "filename": e.get("filename"),
                "track_id": e.get("track_id"),
                "position": e.get("position"),
                "duration_s": e.get("duration_s"),
            }
            for e in manifest
        ],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:12]


def _base_cache_path(manifest: list[dict], methods: list[str]) -> Path:
    methods_slug = "_".join(sorted(methods)).replace("/", "_").replace("-", "_")
    sig = _manifest_signature(manifest, methods)
    return CACHE_DIR / f"base_eval_resume_{methods_slug}_{sig}.jsonl"


def _row_cache_key(method: str, entry: dict) -> str:
    return f"{method}::{entry['track_id']}::{entry['filename']}"


def _load_resume_cache(cache_path: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if not cache_path.exists():
        return cache
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row.get("_cache_key")
            if key:
                cache[key] = row
    return cache


def _append_resume_row(cache_path: Path, row: dict) -> None:
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def _release_eval_memory() -> None:
    """Releases temporary memory as much as possible between two queries without emptying useful large caches."""
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def _progress_label(entry: dict, method: str) -> str:
    label = f"{entry.get('artist', '?')} — {entry.get('title', '?')} | {Path(entry['filename']).name}"
    label = label.replace("\n", " ")
    if len(label) > 96:
        return label[:93] + "..."
    return f"[{method}] {label}"


def _evaluate_manifest_entries(
    methods: list[str] | None = None,
    n_tracks: int = 0,
    manifest: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    if methods is None:
        methods = [config.EMBEDDING_METHOD]
    if manifest is None:
        manifest = load_manifest()
    manifest = _limit_manifest(manifest, n_tracks)
    manifest = _manifest_eval_order(manifest)
    _ensure_dirs()
    raw_dir = ROOT / "data" / "raw"
    cache_path = _base_cache_path(manifest, methods)
    resume_cache = _load_resume_cache(cache_path)

    rows: list[dict] = []
    grouped: dict[str, list[dict]] = defaultdict(list)
    for entry in manifest:
        grouped[entry["track_id"]].append(entry)

    coverage = {
        "n_queries": len(manifest),
        "n_tracks": len(grouped),
        "tracks_with_studio": 0,
        "tracks_with_micro": 0,
        "tracks_with_both": 0,
        "missing_pairs": [],
    }

    for entries in grouped.values():
        has_studio = any(_parse_query_attrs(e)["query_kind"] == "studio" for e in entries)
        has_micro = any(_parse_query_attrs(e)["query_kind"] == "micro" for e in entries)
        coverage["tracks_with_studio"] += int(has_studio)
        coverage["tracks_with_micro"] += int(has_micro)
        coverage["tracks_with_both"] += int(has_studio and has_micro)
        artist = entries[0].get("artist", "?")
        title = entries[0].get("title", "?")
        if not has_studio:
            coverage["missing_pairs"].append(f"{artist} — {title}: missing studio excerpt")
        if not has_micro:
            coverage["missing_pairs"].append(f"{artist} — {title}: missing microphone recording")

    expected_keys = {
        _row_cache_key(method, entry)
        for method in methods
        for entry in manifest
    }
    resumed_rows = [resume_cache[k] for k in expected_keys if k in resume_cache]
    if resumed_rows:
        console.print(f"[eval] Resume found: {len(resumed_rows)}/{len(expected_keys)} query(ies) already computed")
    console.print(f"[dim][eval] Resume cache: {cache_path}[/dim]")

    total_queries = len(manifest) * len(methods)
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("{task.fields[current]}"),
        console=console,
        transient=False,
    ) as progress:
        task_id = progress.add_task(
            "Base eval",
            total=total_queries,
            current="initializing...",
        )

        for method in methods:
            for entry in manifest:
                progress.update(task_id, current=_progress_label(entry, method))
                cache_key = _row_cache_key(method, entry)
                if cache_key in resume_cache:
                    row = dict(resume_cache[cache_key])
                    row.pop("_cache_key", None)
                    rows.append(row)
                    progress.advance(task_id)
                    continue

                audio_path = raw_dir / entry["filename"]
                res = _evaluate_one(audio_path, entry["track_id"], method, "clean")
                attrs = _parse_query_attrs(entry)
                row = {
                    "method": method,
                    "track_id": entry["track_id"],
                    "artist": entry.get("artist", ""),
                    "title": entry.get("title", ""),
                    "filename": entry["filename"],
                    "position": entry.get("position", ""),
                    "duration_s": float(entry.get("duration_s") or 0.0),
                    "condition": "clean",
                    **attrs,
                    "stage1_rank": res.get("stage1_rank"),
                    "final_rank": res.get("rank"),
                    "score_faiss": float(res.get("score_faiss") or 0.0),
                    "score_fp": float(res.get("score_fp") or 0.0),
                    "top1": bool(res.get("top1", False)),
                    "top5": bool(res.get("top5", False)),
                    "stage1_top1": bool(res.get("stage1_top1", False)),
                    "stage1_top5": bool(res.get("stage1_top5", False)),
                    "latency_s": float(res.get("latency_s") or 0.0),
                    "error": res.get("error"),
                }
                rows.append(row)
                cached_row = dict(row)
                cached_row["_cache_key"] = cache_key
                _append_resume_row(cache_path, cached_row)
                resume_cache[cache_key] = cached_row
                progress.advance(task_id)
                _release_eval_memory()

    rows.sort(key=lambda r: (r["artist"], r["title"], r["query_kind"], r["duration_s"], r["filename"]))
    return rows, coverage


def _resolve_base_rows(
    methods: list[str] | None = None,
    n_tracks: int = 0,
    rows: list[dict] | None = None,
    coverage: dict | None = None,
) -> tuple[list[dict], dict]:
    """Returns an already computed base table or launches the single evaluation pass."""
    if rows is not None and coverage is not None:
        return rows, coverage
    return _evaluate_manifest_entries(methods=methods, n_tracks=n_tracks)


def _summary_for_rows(rows: list[dict], group_key: str) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)

    summary = []
    for key, vals in sorted(grouped.items()):
        stage1_ranks = [v["stage1_rank"] for v in vals if v["stage1_rank"] is not None]
        final_ranks = [v["final_rank"] for v in vals if v["final_rank"] is not None]
        summary.append({
            group_key: key,
            "n_queries": len(vals),
            "top1_pct": round(100 * _safe_mean([float(v["top1"]) for v in vals]), 1),
            "top5_pct": round(100 * _safe_mean([float(v["top5"]) for v in vals]), 1),
            "stage1_top1_pct": round(100 * _safe_mean([float(v["stage1_top1"]) for v in vals]), 1),
            "stage1_top5_pct": round(100 * _safe_mean([float(v["stage1_top5"]) for v in vals]), 1),
            "mean_stage1_rank": round(_safe_mean(stage1_ranks), 2) if stage1_ranks else None,
            "mean_final_rank": round(_safe_mean(final_ranks), 2) if final_ranks else None,
            "mean_faiss": round(_safe_mean([v["score_faiss"] for v in vals]), 4),
            "mean_fp": round(_safe_mean([v["score_fp"] for v in vals]), 4),
            "mean_latency_s": round(_safe_mean([v["latency_s"] for v in vals]), 2),
        })
    return summary


def _write_json_md(
    analysis_name: str,
    payload: dict,
    markdown: str,
    out_dir: Path,
) -> tuple[Path, Path]:
    _cleanup_legacy_outputs(out_dir, analysis_name)
    json_path = out_dir / f"{analysis_name}.json"
    md_path = out_dir / f"{analysis_name}.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    md_path.write_text(markdown, encoding="utf-8")
    print(f"[eval] JSON     → {json_path}")
    print(f"[eval] Markdown → {md_path}")
    return json_path, md_path


def _markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    lines = []
    headers = [label for _, label in columns]
    keys = [key for key, _ in columns]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        vals = []
        for key in keys:
            val = row.get(key)
            if val is None:
                vals.append("NF")
            elif isinstance(val, float):
                vals.append(f"{val:.4f}" if abs(val) < 100 else f"{val:.2f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _query_kind_color(query_kind: str) -> str:
    return "#2980b9" if query_kind == "studio" else "#e74c3c"


def _categorical_jitter(n: int, center: float, spread: float = 0.08) -> np.ndarray:
    if n <= 0:
        return np.array([])
    if n == 1:
        return np.array([center])
    return np.linspace(center - spread, center + spread, n)


def run_studio_mic_analysis(methods: list[str] | None = None, n_tracks: int = 0,
                            out_dir: Path | None = None, plot: bool = True,
                            rows: list[dict] | None = None, coverage: dict | None = None) -> dict:
    out_dir = _ensure_dirs(out_dir)
    base_rows, coverage = _resolve_base_rows(methods=methods, n_tracks=n_tracks, rows=rows, coverage=coverage)
    rows = [r for r in base_rows if r["query_kind"] in {"studio", "micro"}]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["track_id"]].append(row)
    tracks_with_both = {
        tid for tid, vals in grouped.items()
        if {v["query_kind"] for v in vals} == {"studio", "micro"}
    }
    rows = [r for r in rows if r["track_id"] in tracks_with_both]

    summary = _summary_for_rows(rows, "query_kind")
    plot_paths: list[str] = []

    if plot and summary:
        valid_rows = [r for r in rows if r["final_rank"] is not None]
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for kind, label in [("studio", "Studio"), ("micro", "Micro")]:
            subset = [r for r in valid_rows if r["query_kind"] == kind]
            ax.scatter(
                [r["stage1_rank"] for r in subset],
                [r["final_rank"] for r in subset],
                color=_query_kind_color(kind),
                label=label,
                alpha=0.8,
                s=45,
            )
        max_rank = max(
            max(r["stage1_rank"], r["final_rank"])
            for r in valid_rows
            if r["stage1_rank"] is not None and r["final_rank"] is not None
        ) if valid_rows else 1
        ax.plot([1, max_rank], [1, max_rank], linestyle="--", color="#7f8c8d", linewidth=1)
        ax.set_xlabel("Stage 1 rank")
        ax.set_ylabel("Final rank")
        ax.set_title("Studio vs micro: Stage 1 rank vs final rank")
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.legend()
        plot_paths.append(_save_fig(fig, PLOTS_DIR / "scatter_studio_micro_stage1_vs_final_rank.png"))

    md = [
        "# Studio vs Microphone Analysis",
        "",
        "## Summary by query type",
        _markdown_table(summary, [
            ("query_kind", "Query type"),
            ("n_queries", "N"),
            ("top1_pct", "Top-1 (%)"),
            ("top5_pct", "Top-5 (%)"),
            ("mean_stage1_rank", "Mean Stage 1 rank"),
            ("mean_final_rank", "Mean final rank"),
            ("mean_faiss", "Mean FAISS"),
            ("mean_fp", "Mean FP"),
        ]),
        "",
        "## Per-query results",
        _markdown_table(rows, [
            ("artist", "Artist"),
            ("title", "Title"),
            ("filename", "File"),
            ("query_kind", "Type"),
            ("duration_s", "Duration (s)"),
            ("stage1_rank", "Stage 1 rank"),
            ("final_rank", "Final rank"),
            ("score_faiss", "FAISS"),
            ("score_fp", "FP"),
        ]),
    ]

    payload = {"analysis": "studio-mic", "coverage": coverage, "summary": summary, "rows": rows, "plots": plot_paths}
    _write_json_md("studio_mic", payload, "\n".join(md) + "\n", out_dir)
    return payload


def run_duration_analysis(methods: list[str] | None = None, n_tracks: int = 0,
                          out_dir: Path | None = None, plot: bool = True,
                          rows: list[dict] | None = None, coverage: dict | None = None) -> dict:
    out_dir = _ensure_dirs(out_dir)
    base_rows, coverage = _resolve_base_rows(methods=methods, n_tracks=n_tracks, rows=rows, coverage=coverage)
    rows = [r for r in base_rows if r["query_kind"] == "studio" and r["duration_bucket"] in {5, 15, 30}]
    summary = _summary_for_rows(rows, "duration_bucket")
    plot_paths: list[str] = []

    if plot and summary:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        subset = [r for r in rows if r["final_rank"] is not None]
        ax.scatter(
            [r["duration_s"] for r in subset],
            [r["final_rank"] for r in subset],
            color="#2980b9",
            alpha=0.8,
            s=45,
        )
        ax.set_xlabel("Query duration (s)")
        ax.set_ylabel("Final rank")
        ax.set_title("Duration vs final rank")
        ax.invert_yaxis()
        plot_paths.append(_save_fig(fig, PLOTS_DIR / "scatter_duration_vs_final_rank.png"))

    md = [
        "# Duration Analysis",
        "",
        "## Summary by duration",
        _markdown_table(summary, [
            ("duration_bucket", "Duration"),
            ("n_queries", "N"),
            ("top1_pct", "Top-1 (%)"),
            ("top5_pct", "Top-5 (%)"),
            ("mean_stage1_rank", "Mean Stage 1 rank"),
            ("mean_final_rank", "Mean final rank"),
            ("mean_faiss", "Mean FAISS"),
            ("mean_fp", "Mean FP"),
        ]),
        "",
        "## Per-query results",
        _markdown_table(rows, [
            ("artist", "Artist"),
            ("title", "Title"),
            ("filename", "File"),
            ("duration_bucket", "Duration"),
            ("stage1_rank", "Stage 1 rank"),
            ("final_rank", "Final rank"),
            ("score_faiss", "FAISS"),
            ("score_fp", "FP"),
        ]),
    ]

    payload = {"analysis": "duration", "coverage": coverage, "summary": summary, "rows": rows, "plots": plot_paths}
    _write_json_md("duration", payload, "\n".join(md) + "\n", out_dir)
    return payload


def run_stage12_analysis(methods: list[str] | None = None, n_tracks: int = 0,
                         out_dir: Path | None = None, plot: bool = True,
                         rows: list[dict] | None = None, coverage: dict | None = None) -> dict:
    out_dir = _ensure_dirs(out_dir)
    base_rows, coverage = _resolve_base_rows(methods=methods, n_tracks=n_tracks, rows=rows, coverage=coverage)
    rows = [dict(r) for r in base_rows]
    for row in rows:
        if row["stage1_rank"] is not None and row["final_rank"] is not None:
            row["rank_gain"] = row["stage1_rank"] - row["final_rank"]
        else:
            row["rank_gain"] = None

    summary = []
    groups = {
        "all": rows,
        "studio": [r for r in rows if r["query_kind"] == "studio"],
        "micro": [r for r in rows if r["query_kind"] == "micro"],
    }
    for label, vals in groups.items():
        gains = [v["rank_gain"] for v in vals if v["rank_gain"] is not None]
        improved = sum(1 for g in gains if g > 0)
        unchanged = sum(1 for g in gains if g == 0)
        degraded = sum(1 for g in gains if g < 0)
        summary.append({
            "group": label,
            "n_queries": len(vals),
            "stage1_top1_pct": round(100 * _safe_mean([float(v["stage1_top1"]) for v in vals]), 1),
            "final_top1_pct": round(100 * _safe_mean([float(v["top1"]) for v in vals]), 1),
            "stage1_top5_pct": round(100 * _safe_mean([float(v["stage1_top5"]) for v in vals]), 1),
            "final_top5_pct": round(100 * _safe_mean([float(v["top5"]) for v in vals]), 1),
            "mean_rank_gain": round(_safe_mean(gains), 2) if gains else 0.0,
            "improved_pct": round(100 * improved / len(gains), 1) if gains else 0.0,
            "unchanged_pct": round(100 * unchanged / len(gains), 1) if gains else 0.0,
            "degraded_pct": round(100 * degraded / len(gains), 1) if gains else 0.0,
        })

    plot_paths: list[str] = []
    if plot and summary:
        ranked_rows = [r for r in rows if r["stage1_rank"] is not None and r["final_rank"] is not None]
        fig, ax = plt.subplots(figsize=(8, 5.5))
        max_rank = max(max(r["stage1_rank"], r["final_rank"]) for r in ranked_rows) if ranked_rows else 1
        ax.scatter(
            [r["stage1_rank"] for r in ranked_rows],
            [r["final_rank"] for r in ranked_rows],
            color="#34495e",
            alpha=0.8,
            s=45,
        )
        ax.plot([1, max_rank], [1, max_rank], linestyle="--", color="#7f8c8d", linewidth=1)
        ax.set_xlabel("Stage 1 rank (FAISS)")
        ax.set_ylabel("Final rank (after fingerprinting)")
        ax.set_title("Stage 1 vs Stage 2 ranking")
        ax.invert_xaxis()
        ax.invert_yaxis()
        plot_paths.append(_save_fig(fig, PLOTS_DIR / "scatter_stage1_vs_stage2_rank.png"))

    md = [
        "# Stage 1 / Stage 2 Analysis",
        "",
        "## Summary",
        _markdown_table(summary, [
            ("group", "Group"),
            ("n_queries", "N"),
            ("stage1_top1_pct", "Stage1 Top-1 (%)"),
            ("final_top1_pct", "Final Top-1 (%)"),
            ("stage1_top5_pct", "Stage1 Top-5 (%)"),
            ("final_top5_pct", "Final Top-5 (%)"),
            ("mean_rank_gain", "Mean rank gain"),
            ("improved_pct", "Improved (%)"),
            ("unchanged_pct", "Unchanged (%)"),
            ("degraded_pct", "Degraded (%)"),
        ]),
        "",
        "## Per-query results",
        _markdown_table(rows, [
            ("artist", "Artist"),
            ("title", "Title"),
            ("filename", "File"),
            ("query_kind", "Type"),
            ("stage1_rank", "Stage 1 rank"),
            ("final_rank", "Final rank"),
            ("rank_gain", "Gain"),
            ("score_faiss", "FAISS"),
            ("score_fp", "FP"),
        ]),
    ]

    payload = {"analysis": "stage12", "coverage": coverage, "summary": summary, "rows": rows, "plots": plot_paths}
    _write_json_md("stage12", payload, "\n".join(md) + "\n", out_dir)
    return payload


def run_mic_conditions_analysis(methods: list[str] | None = None, n_tracks: int = 0,
                                out_dir: Path | None = None, plot: bool = True,
                                rows: list[dict] | None = None, coverage: dict | None = None) -> dict:
    out_dir = _ensure_dirs(out_dir)
    base_rows, coverage = _resolve_base_rows(methods=methods, n_tracks=n_tracks, rows=rows, coverage=coverage)
    rows = [r for r in base_rows if r["query_kind"] == "micro" and r["distance"] and r["speech"]]

    condition_rows = []
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["distance"], row["speech"])].append(row)

    for (distance, speech), vals in sorted(grouped.items()):
        stage1_ranks = [v["stage1_rank"] for v in vals if v["stage1_rank"] is not None]
        final_ranks = [v["final_rank"] for v in vals if v["final_rank"] is not None]
        condition_rows.append({
            "distance": distance,
            "speech": speech,
            "n_queries": len(vals),
            "top1_pct": round(100 * _safe_mean([float(v["top1"]) for v in vals]), 1),
            "top5_pct": round(100 * _safe_mean([float(v["top5"]) for v in vals]), 1),
            "mean_stage1_rank": round(_safe_mean(stage1_ranks), 2) if stage1_ranks else None,
            "mean_final_rank": round(_safe_mean(final_ranks), 2) if final_ranks else None,
            "mean_faiss": round(_safe_mean([v["score_faiss"] for v in vals]), 4),
            "mean_fp": round(_safe_mean([v["score_fp"] for v in vals]), 4),
        })

    plot_paths: list[str] = []
    if plot and rows:
        # Studio vs micro final rank strip plot
        fig, ax = plt.subplots(figsize=(8, 5.5))
        studio_rows = [r for r in rows if r["query_kind"] == "studio" and r["final_rank"] is not None]
        micro_rows = [r for r in rows if r["query_kind"] == "micro" and r["final_rank"] is not None]
        xs_studio = _categorical_jitter(len(studio_rows), 0.0)
        xs_micro = _categorical_jitter(len(micro_rows), 1.0)
        ax.scatter(xs_studio, [r["final_rank"] for r in studio_rows], color="#2980b9", alpha=0.8, s=45, label="Studio")
        ax.scatter(xs_micro, [r["final_rank"] for r in micro_rows], color="#e74c3c", alpha=0.8, s=45, label="Micro")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["studio", "micro"])
        ax.set_ylabel("Final rank")
        ax.set_title("Final rank: studio vs micro")
        ax.invert_yaxis()
        ax.legend()
        plot_paths.append(_save_fig(fig, PLOTS_DIR / "scatter_studio_vs_micro_rank.png"))

        # Micro clean vs speech final rank strip plot
        fig, ax = plt.subplots(figsize=(8, 5.5))
        clean_rows = [r for r in rows if r["query_kind"] == "micro" and r["speech"] == "clean" and r["final_rank"] is not None]
        speech_rows = [r for r in rows if r["query_kind"] == "micro" and r["speech"] == "speech" and r["final_rank"] is not None]
        xs_clean = _categorical_jitter(len(clean_rows), 0.0)
        xs_speech = _categorical_jitter(len(speech_rows), 1.0)
        ax.scatter(xs_clean, [r["final_rank"] for r in clean_rows], color="#2980b9", alpha=0.8, s=45, label="Micro clean")
        ax.scatter(xs_speech, [r["final_rank"] for r in speech_rows], color="#e74c3c", alpha=0.8, s=45, label="Micro speech")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["clean", "speech"])
        ax.set_ylabel("Final rank")
        ax.set_title("Final rank: micro clean vs micro speech")
        ax.invert_yaxis()
        ax.legend()
        plot_paths.append(_save_fig(fig, PLOTS_DIR / "scatter_micro_clean_vs_speech_rank.png"))

    md = [
        "# Microphone Conditions Analysis",
        "",
        "## Summary by condition",
        _markdown_table(condition_rows, [
            ("distance", "Distance"),
            ("speech", "Speech"),
            ("n_queries", "N"),
            ("top1_pct", "Top-1 (%)"),
            ("top5_pct", "Top-5 (%)"),
            ("mean_stage1_rank", "Mean Stage 1 rank"),
            ("mean_final_rank", "Mean final rank"),
            ("mean_faiss", "Mean FAISS"),
            ("mean_fp", "Mean FP"),
        ]),
        "",
        "## Per-query results",
        _markdown_table(rows, [
            ("artist", "Artist"),
            ("title", "Title"),
            ("filename", "File"),
            ("distance", "Distance"),
            ("speech", "Speech"),
            ("stage1_rank", "Stage 1 rank"),
            ("final_rank", "Final rank"),
            ("score_faiss", "FAISS"),
            ("score_fp", "FP"),
        ]),
    ]

    payload = {"analysis": "mic-conditions", "coverage": coverage, "summary": condition_rows, "rows": rows, "plots": plot_paths}
    _write_json_md("mic_conditions", payload, "\n".join(md) + "\n", out_dir)
    return payload


def run_rir_analysis(methods: list[str] | None = None, n_tracks: int = 0,
                     out_dir: Path | None = None, plot: bool = True) -> dict:
    out_dir = _ensure_dirs(out_dir)
    if methods is None:
        methods = [config.EMBEDDING_METHOD]
    _cleanup_rir_plots()
    # Align the RIR analysis to the same test basis as `eval`/`eval base`:
    # the actual requests of the manifest, taken as is.
    result = run_rir_evaluate(methods=methods, conditions=["clean"],
                              n_tracks=n_tracks, out_dir=out_dir, plot=False)

    rows = []
    category_csvs: list[str] = []
    condition_csvs: list[str] = []
    overview_plots: list[str] = []
    for method in result.get("methods", []):
        method_data = result.get("results", {}).get(method, {})
        clean_pairs = _rir_pairs_for_condition(result, method, "clean")
        category_rows = _build_rir_topk_summary_rows(clean_pairs)
        condition_rows_csv = _build_rir_condition_summary_rows(result, method)

        suffix = "" if method == config.EMBEDDING_METHOD else f"_{method}"
        category_csv = out_dir / f"rir_topk_summary_by_category{suffix}.csv"
        condition_csv = out_dir / f"rir_topk_summary_by_condition{suffix}.csv"
        category_csvs.append(str(_write_csv(category_csv, category_rows, [
            "Category",
            "Subcategory",
            "N queries",
            "Without RIR Top-1 (%)",
            "Without RIR Top-5 (%)",
            "Without RIR Top-10 (%)",
            "With RIR Top-1 (%)",
            "With RIR Top-5 (%)",
            "With RIR Top-10 (%)",
            "Top-1 gain (pp)",
            "Top-5 gain (pp)",
            "Top-10 gain (pp)",
            "Without RIR mean rank",
            "With RIR mean rank",
        ])))
        condition_csvs.append(str(_write_csv(condition_csv, condition_rows_csv, [
            "Method",
            "Condition",
            "N queries",
            "Without RIR Top-1 (%)",
            "Without RIR Top-5 (%)",
            "Without RIR Top-10 (%)",
            "With RIR Top-1 (%)",
            "With RIR Top-5 (%)",
            "With RIR Top-10 (%)",
            "Top-1 gain (pp)",
            "Top-5 gain (pp)",
            "Top-10 gain (pp)",
            "Without RIR mean rank",
            "With RIR mean rank",
        ])))
        if plot:
            plot_path = _plot_rir_pipeline_overview(method, clean_pairs)
            if plot_path:
                overview_plots.append(plot_path)

        for condition in result.get("conditions", []):
            without = method_data.get(f"{condition}_without_rir", {})
            with_r = method_data.get(f"{condition}_with_rir", {})
            without_rows = without.get("per_track", [])
            with_rows = with_r.get("per_track", [])
            without_top10 = _rir_metrics_from_rows(without_rows)["top10_pct"]
            with_top10 = _rir_metrics_from_rows(with_rows)["top10_pct"]
            rows.append({
                "method": method,
                "condition": condition,
                "top1_without": round(100 * without.get("top1_accuracy", 0), 1),
                "top1_with": round(100 * with_r.get("top1_accuracy", 0), 1),
                "delta_top1": round(100 * (with_r.get("top1_accuracy", 0) - without.get("top1_accuracy", 0)), 1),
                "top5_without": round(100 * without.get("top5_accuracy", 0), 1),
                "top5_with": round(100 * with_r.get("top5_accuracy", 0), 1),
                "top10_without": without_top10,
                "top10_with": with_top10,
                "faiss_without": without.get("mean_faiss_score", 0),
                "faiss_with": with_r.get("mean_faiss_score", 0),
            })

    md = [
        "# RIR Analysis",
        "",
        "## Summary by condition",
        _markdown_table(rows, [
            ("method", "Method"),
            ("condition", "Condition"),
            ("top1_without", "Top-1 w/o RIR (%)"),
            ("top1_with", "Top-1 with RIR (%)"),
            ("delta_top1", "Delta Top-1 (pp)"),
            ("top5_without", "Top-5 w/o RIR (%)"),
            ("top5_with", "Top-5 with RIR (%)"),
            ("top10_without", "Top-10 w/o RIR (%)"),
            ("top10_with", "Top-10 with RIR (%)"),
            ("faiss_without", "Mean FAISS w/o"),
            ("faiss_with", "Mean FAISS with"),
        ]),
    ]

    payload = {
        "analysis": "rir",
        "result": result,
        "summary": rows,
        "overview_plots": overview_plots,
        "category_csvs": category_csvs,
        "condition_csvs": condition_csvs,
    }
    _write_json_md("rir_analysis", payload, "\n".join(md) + "\n", out_dir)
    return payload


def run_base_eval_suite(methods: list[str] | None = None, n_tracks: int = 0,
                        out_dir: Path | None = None, plot: bool = True) -> dict:
    out_dir = _ensure_dirs(out_dir)
    if methods is None:
        methods = [config.EMBEDDING_METHOD]

    _cleanup_base_suite_plots()

    print("\n[eval] ── Base evaluation pass ───────────────────────────────────────")
    base_rows, coverage = _evaluate_manifest_entries(methods=methods, n_tracks=n_tracks)
    overview_rows, overview_plot = _plot_pipeline_overview(base_rows) if plot else (_build_pipeline_overview(base_rows), None)
    topk_summary_rows = _build_topk_summary_rows(base_rows)
    _cleanup_legacy_outputs(out_dir, "eval_topk_summary_by_category", suffixes=(".csv",))
    csv_rows = _topk_summary_csv_rows(topk_summary_rows)
    csv_path = _write_csv(
        out_dir / "eval_topk_summary_by_category.csv",
        csv_rows,
        [
            "Category",
            "Subcategory",
            "N queries",
            "Stage 1 Top-1 (%)",
            "Stage 1 Top-5 (%)",
            "Stage 1 Top-10 (%)",
            "Final Top-1 (%)",
            "Final Top-5 (%)",
            "Final Top-10 (%)",
            "Mean Stage 1 rank",
            "Mean Final rank",
        ],
    )

    base_payload = {
        "analysis": "base-eval-rows",
        "timestamp": datetime.now().isoformat(),
        "methods": methods,
        "n_tracks": n_tracks,
        "coverage": coverage,
        "rows": base_rows,
        "pipeline_overview": overview_rows,
        "plots": ([overview_plot] if overview_plot else []),
        "topk_summary_csv": str(csv_path),
        "topk_summary_rows": topk_summary_rows,
    }
    base_md = [
        "# Base Evaluation Rows",
        "",
        f"- Generated: {base_payload['timestamp']}",
        f"- Methods: {', '.join(methods)}",
        f"- Track limit: {n_tracks if n_tracks else 'all'}",
        "",
        "## Coverage",
        f"- Queries in manifest: {coverage.get('n_queries', 0)}",
        f"- Tracks in manifest: {coverage.get('n_tracks', 0)}",
        f"- Tracks with both studio and mic: {coverage.get('tracks_with_both', 0)}",
        "",
        "## Pipeline overview",
        _markdown_table(overview_rows, [
            ("label", "Scenario"),
            ("n_queries", "N"),
            ("stage1_pct", "Stage 1 Top-1 (%)"),
            ("stage2_pct", "Stage 2 Top-1 (%)"),
            ("gain_pct", "Gain (pp)"),
            ("mean_stage1_rank", "Mean Stage 1 rank"),
            ("mean_final_rank", "Mean final rank"),
        ]),
        "",
        "## Top-k summary by category",
        _markdown_table(topk_summary_rows, [
            ("category_label", "Category"),
            ("subcategory_label", "Subcategory"),
            ("n_queries", "N"),
            ("stage1_top1_pct", "Stage 1 Top-1 (%)"),
            ("stage1_top5_pct", "Stage 1 Top-5 (%)"),
            ("stage1_top10_pct", "Stage 1 Top-10 (%)"),
            ("final_top1_pct", "Final Top-1 (%)"),
            ("final_top5_pct", "Final Top-5 (%)"),
            ("final_top10_pct", "Final Top-10 (%)"),
        ]),
    ]
    _write_json_md("base_eval_rows", base_payload, "\n".join(base_md) + "\n", out_dir)

    print("\n[eval] ── Studio vs mic ───────────────────────────────────────────────")
    studio_mic = run_studio_mic_analysis(
        methods=methods, n_tracks=n_tracks, out_dir=out_dir, plot=False, rows=base_rows, coverage=coverage
    )

    print("\n[eval] ── Duration ───────────────────────────────────────────────────")
    duration = run_duration_analysis(
        methods=methods, n_tracks=n_tracks, out_dir=out_dir, plot=False, rows=base_rows, coverage=coverage
    )

    print("\n[eval] ── Stage 1 vs Stage 2 ─────────────────────────────────────────")
    stage12 = run_stage12_analysis(
        methods=methods, n_tracks=n_tracks, out_dir=out_dir, plot=False, rows=base_rows, coverage=coverage
    )

    print("\n[eval] ── Microphone conditions ─────────────────────────────────────")
    mic_conditions = run_mic_conditions_analysis(
        methods=methods, n_tracks=n_tracks, out_dir=out_dir, plot=False, rows=base_rows, coverage=coverage
    )

    payload = {
        "analysis": "eval-base-suite",
        "timestamp": datetime.now().isoformat(),
        "methods": methods,
        "n_tracks": n_tracks,
        "coverage": coverage,
        "base_eval_rows": base_payload,
        "pipeline_overview": {
            "summary": overview_rows,
            "plot": overview_plot,
        },
        "topk_summary": {
            "csv": str(csv_path),
            "rows": topk_summary_rows,
            "csv_rows": csv_rows,
        },
        "studio_mic": studio_mic,
        "duration": duration,
        "stage12": stage12,
        "mic_conditions": mic_conditions,
    }

    summary_md = [
        "# Full Evaluation Summary",
        "",
        f"- Generated: {payload['timestamp']}",
        f"- Methods: {', '.join(methods)}",
        f"- Track limit: {n_tracks if n_tracks else 'all'}",
        "",
        "## Coverage",
        f"- Queries in manifest: {studio_mic.get('coverage', {}).get('n_queries', 0)}",
        f"- Tracks in manifest: {studio_mic.get('coverage', {}).get('n_tracks', 0)}",
        f"- Tracks with both studio and mic: {studio_mic.get('coverage', {}).get('tracks_with_both', 0)}",
        "",
        "## Generated analyses",
        "- pipeline overview plot",
        f"- top-k summary CSV: {csv_path.name}",
        "- base-eval-rows",
        "- studio-mic",
        "- duration",
        "- stage12",
        "- mic-conditions",
    ]

    _write_json_md("eval_base_summary", payload, "\n".join(summary_md) + "\n", out_dir)
    return payload


def run_full_eval(methods: list[str] | None = None, n_tracks: int = 0,
                  out_dir: Path | None = None, plot: bool = True) -> dict:
    """Backward compatibility: alias to the base evaluation suite."""
    return run_base_eval_suite(methods=methods, n_tracks=n_tracks, out_dir=out_dir, plot=plot)
