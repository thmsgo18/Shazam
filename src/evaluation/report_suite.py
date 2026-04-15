"""
src/evaluation/report_suite.py

Report-oriented evaluation suite:
  - analysis of real files present in data/raw/manifest.json
  - comparison studio / micro / duration on real queries
  - detection of gaps in the test set
  - generation of a JSON, a Markdown summary and graphs

Public entry point:
  run_report_suite(methods=None, n_tracks=0, out_dir=None, plot=True)
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.evaluation.evaluate import load_manifest, _evaluate_one

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "eval"
PLOTS_DIR = ROOT / "results" / "plots"


def _query_kind(entry: dict) -> str:
    pos = str(entry.get("position", "")).lower()
    name = str(entry.get("filename", "")).lower()
    if pos == "micro" or "micro" in name:
        return "micro"
    if pos in {"start", "middle", "clean"}:
        return "studio"
    return "other"


def _query_label(entry: dict) -> str:
    kind = _query_kind(entry)
    pos = entry.get("position", "unknown")
    dur = entry.get("duration_s")
    dur_str = f"{int(round(float(dur)))}s" if dur is not None else "?"
    short = Path(entry["filename"]).stem
    short = short.replace("__", " / ")
    return f"{short}\n({kind}, {pos}, {dur_str})"


def _safe_rank(rank: int | None, fallback: int) -> int:
    return rank if rank is not None else fallback


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [report-suite] ✓ {path.name}")


def _plot_real_query_ranks(rows: list[dict], method: str, out_dir: Path) -> Path | None:
    if not rows:
        return None

    labels = [row["query_label"] for row in rows]
    max_rank = max(
        [_safe_rank(row.get("stage1_rank"), 0) for row in rows]
        + [_safe_rank(row.get("final_rank"), 0) for row in rows]
        + [10]
    )
    missing_rank = max_rank + 2

    stage1 = [_safe_rank(row.get("stage1_rank"), missing_rank) for row in rows]
    final = [_safe_rank(row.get("final_rank"), missing_rank) for row in rows]

    x = np.arange(len(rows))
    w = 0.36

    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 1.4), 5.5))
    bars1 = ax.bar(x - w / 2, stage1, w, label="Stage 1 rank", color="#95a5a6", alpha=0.9)
    bars2 = ax.bar(x + w / 2, final, w, label="Final rank", color="#2980b9", alpha=0.9)

    for bars, values in [(bars1, stage1), (bars2, final)]:
        for bar, v in zip(bars, values):
            text = "NF" if v == missing_rank else f"#{v}"
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.25, text,
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Rank (lower is better)")
    ax.set_ylim(0, missing_rank + 3)
    ax.invert_yaxis()
    ax.set_title(f"Ranks on real queries — {method.upper()}\nStage 1 vs final pipeline")
    ax.legend()

    out = out_dir / f"report_real_ranks_{method}.png"
    _save(fig, out)
    return out


def _plot_real_query_scores(rows: list[dict], method: str, out_dir: Path) -> Path | None:
    if not rows:
        return None

    labels = [row["query_label"] for row in rows]
    faiss = [row.get("score_faiss") or 0.0 for row in rows]
    fp = [row.get("score_fp") or 0.0 for row in rows]

    x = np.arange(len(rows))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, len(rows) * 1.4), 7.0), sharex=True)

    bars1 = ax1.bar(x, faiss, color="#ee854a", edgecolor="white")
    for bar, v in zip(bars1, faiss):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + max(0.05, 0.01 * max(faiss + [1])),
                 f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax1.set_ylabel("FAISS score")
    ax1.set_title(f"Scores on real queries — {method.upper()}")

    bars2 = ax2.bar(x, fp, color="#27ae60", edgecolor="white")
    for bar, v in zip(bars2, fp):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + max(0.001, 0.05 * max(fp + [0.01])),
                 f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax2.set_ylabel("Fingerprint score")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)

    fig.tight_layout()
    out = out_dir / f"report_real_scores_{method}.png"
    _save(fig, out)
    return out


def _write_markdown(
    rows: list[dict],
    coverage: dict,
    out_path: Path,
    plot_paths: list[Path],
    methods: list[str],
) -> None:
    lines: list[str] = []
    lines.append("# Report Evaluation Suite\n")
    lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Methods: {', '.join(methods)}")
    lines.append(f"- Query files in manifest: {coverage['n_queries']}")
    lines.append(f"- Distinct tracks in manifest: {coverage['n_tracks']}")
    lines.append(f"- Tracks with at least one studio query: {coverage['tracks_with_studio']}")
    lines.append(f"- Tracks with at least one micro query: {coverage['tracks_with_micro']}")
    lines.append(f"- Tracks with both studio and micro queries: {coverage['tracks_with_both']}\n")

    if coverage["missing_pairs"]:
        lines.append("## Coverage Gaps\n")
        for msg in coverage["missing_pairs"]:
            lines.append(f"- {msg}")
        lines.append("")

    if plot_paths:
        lines.append("## Generated Figures\n")
        for p in plot_paths:
            lines.append(f"- `{p.relative_to(ROOT)}`")
        lines.append("")

    lines.append("## Real Query Comparison\n")
    lines.append("| Method | Artist | Title | File | Kind | Duration (s) | Stage 1 rank | Final rank | FAISS score | FP score | Latency (s) |")
    lines.append("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|")

    for row in rows:
        lines.append(
            f"| {row['method'].upper()} | {row['artist']} | {row['title']} | `{row['filename']}` | "
            f"{row['query_kind']} | {row['duration_s']:.2f} | "
            f"{row['stage1_rank'] if row['stage1_rank'] is not None else 'NF'} | "
            f"{row['final_rank'] if row['final_rank'] is not None else 'NF'} | "
            f"{(row['score_faiss'] or 0.0):.4f} | {(row['score_fp'] or 0.0):.4f} | {row['latency_s']:.2f} |"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_report_suite(
    methods: list[str] | None = None,
    n_tracks: int = 0,
    out_dir: Path | None = None,
    plot: bool = True,
) -> dict:
    """
    Launches the report-oriented evaluation suite on the real files of the manifest.

    Args:
        methods: methods to evaluate (default: [config.EMBEDDING_METHOD]).
        n_tracks: limits the number of distinct tracks from the manifest (0 = all).
        out_dir: output folder for JSON/Markdown.
        plot: generates the associated graphs.

    Returns:
        Serializable dict with detailed results.
    """
    if methods is None:
        methods = [config.EMBEDDING_METHOD]

    out_dir = Path(out_dir) if out_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    if not manifest:
        print("[report-suite] ⚠ No manifest found or no files listed.")
        return {}

    if n_tracks > 0:
        kept = []
        seen: set[str] = set()
        for entry in manifest:
            tid = entry["track_id"]
            if tid not in seen and len(seen) >= n_tracks:
                continue
            seen.add(tid)
            kept.append(entry)
        manifest = kept

    grouped: dict[str, list[dict]] = defaultdict(list)
    for entry in manifest:
        grouped[entry["track_id"]].append(entry)

    missing_pairs: list[str] = []
    tracks_with_studio = tracks_with_micro = tracks_with_both = 0
    for entries in grouped.values():
        artist = entries[0].get("artist", "?")
        title = entries[0].get("title", "?")
        has_studio = any(_query_kind(e) == "studio" for e in entries)
        has_micro = any(_query_kind(e) == "micro" for e in entries)
        tracks_with_studio += int(has_studio)
        tracks_with_micro += int(has_micro)
        tracks_with_both += int(has_studio and has_micro)
        if not has_studio:
            missing_pairs.append(f"{artist} — {title}: no studio excerpt in `data/raw`.")
        if not has_micro:
            missing_pairs.append(f"{artist} — {title}: no microphone recording in `data/raw`.")

    raw_dir = ROOT / "data" / "raw"
    rows: list[dict] = []
    for method in methods:
        print(f"\n[report-suite] Method: {method}")
        for entry in manifest:
            audio_path = raw_dir / entry["filename"]
            res = _evaluate_one(audio_path, entry["track_id"], method, "clean")
            row = {
                "method": method,
                "track_id": entry["track_id"],
                "artist": entry.get("artist", ""),
                "title": entry.get("title", ""),
                "filename": entry["filename"],
                "position": entry.get("position", "unknown"),
                "query_kind": _query_kind(entry),
                "query_label": _query_label(entry),
                "duration_s": float(entry.get("duration_s") or 0.0),
                "final_rank": res.get("rank"),
                "stage1_rank": res.get("stage1_rank"),
                "score_faiss": res.get("score_faiss"),
                "score_fp": res.get("score_fp"),
                "latency_s": float(res.get("latency_s") or 0.0),
                "top1": bool(res.get("top1", False)),
                "top5": bool(res.get("top5", False)),
                "stage1_top1": bool(res.get("stage1_top1", False)),
                "stage1_top5": bool(res.get("stage1_top5", False)),
                "error": res.get("error"),
            }
            rows.append(row)
            print(
                f"  - {entry['filename']}: "
                f"stage1={row['stage1_rank'] if row['stage1_rank'] is not None else 'NF'} | "
                f"final={row['final_rank'] if row['final_rank'] is not None else 'NF'} | "
                f"faiss={(row['score_faiss'] or 0.0):.4f} | fp={(row['score_fp'] or 0.0):.4f}"
            )

    rows.sort(key=lambda r: (r["artist"], r["title"], r["query_kind"], r["duration_s"], r["filename"]))

    coverage = {
        "n_queries": len(manifest),
        "n_tracks": len(grouped),
        "tracks_with_studio": tracks_with_studio,
        "tracks_with_micro": tracks_with_micro,
        "tracks_with_both": tracks_with_both,
        "missing_pairs": missing_pairs,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = out_dir / f"report_suite_{ts}.json"
    out_md = out_dir / f"report_suite_{ts}.md"

    plot_paths: list[Path] = []
    if plot:
        for method in methods:
            rows_m = [r for r in rows if r["method"] == method]
            p1 = _plot_real_query_ranks(rows_m, method, PLOTS_DIR)
            p2 = _plot_real_query_scores(rows_m, method, PLOTS_DIR)
            if p1:
                plot_paths.append(p1)
            if p2:
                plot_paths.append(p2)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "type": "report_suite",
        "methods": methods,
        "coverage": coverage,
        "rows": rows,
        "plots": [str(p) for p in plot_paths],
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    _write_markdown(rows, coverage, out_md, plot_paths, methods)

    print(f"\n[report-suite] JSON     → {out_json}")
    print(f"[report-suite] Markdown → {out_md}")
    if missing_pairs:
        print("[report-suite] Gaps detected:")
        for msg in missing_pairs:
            print(f"  • {msg}")

    return payload
