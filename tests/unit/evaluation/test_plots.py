from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation import plots


class PlotsTests(unittest.TestCase):
    def _evaluation_payload(self) -> dict:
        return {
            "n_tracks": 2,
            "results": {
                "mfcc": {
                    "clean": {
                        "top1_accuracy": 0.5,
                        "per_track": [
                            {"top1": True, "stage1_top1": False, "track_id": "track_1", "duration_s": 5},
                            {"top1": False, "stage1_top1": False, "track_id": "track_2", "duration_s": 15},
                        ],
                    },
                    "reverb": {
                        "top1_accuracy": 1.0,
                        "per_track": [
                            {"top1": True, "stage1_top1": True, "track_id": "track_1", "duration_s": 5},
                        ],
                    },
                }
            },
        }

    def _rir_payload(self) -> dict:
        return {
            "methods": ["mfcc"],
            "conditions": ["clean", "reverb"],
            "n_tracks": 2,
            "results": {
                "mfcc": {
                    "clean_with_rir": {
                        "top1_accuracy": 1.0,
                        "per_track": [{"track_id": "track_1", "artist": "Artist", "title": "Song", "faiss_score": 10.0}],
                    },
                    "clean_without_rir": {
                        "top1_accuracy": 0.5,
                        "per_track": [{"track_id": "track_1", "artist": "Artist", "title": "Song", "faiss_score": 8.0}],
                    },
                    "reverb_with_rir": {"top1_accuracy": 1.0, "per_track": []},
                    "reverb_without_rir": {"top1_accuracy": 0.0, "per_track": []},
                }
            },
        }

    def test_load_helpers_and_condition_order(self) -> None:
        self.assertEqual(plots._cond_order(["combo", "clean"]), ["clean", "combo"])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "benchmark.json"
            path.write_text(json.dumps({"method": "clap"}), encoding="utf-8")
            benchmarks = plots.load_benchmarks([path])
            evaluations = plots.load_evaluations([path])
        self.assertIn("clap", benchmarks)
        self.assertEqual(len(evaluations), 1)

    def test_run_plots_generates_expected_png_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "plots"
            eval_path = Path(tmpdir) / "eval.json"
            rir_path = Path(tmpdir) / "rir.json"
            eval_path.write_text(json.dumps(self._evaluation_payload()), encoding="utf-8")
            rir_path.write_text(json.dumps(self._rir_payload()), encoding="utf-8")

            plots.run_plots(eval_jsons=[eval_path], rir_eval_jsons=[rir_path], out_dir=out_dir)

            expected = [
                out_dir / "method_accuracy.png",
                out_dir / "stage_comparison.png",
                out_dir / "duration_impact.png",
                out_dir / "heatmap_accuracy.png",
                out_dir / "rir_paired_bar_mfcc.png",
                out_dir / "rir_delta_mfcc.png",
                out_dir / "rir_faiss_scores_mfcc.png",
            ]
            for path in expected:
                self.assertTrue(path.exists(), msg=f"missing plot: {path.name}")


if __name__ == "__main__":
    unittest.main()
