from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.evaluation import report_suite


class ReportSuiteTests(unittest.TestCase):
    def test_query_helpers(self) -> None:
        micro_entry = {"position": "micro", "filename": "mic_recordings/clip.wav", "duration_s": 15}
        studio_entry = {"position": "middle", "filename": "reference_clips/clip.wav", "duration_s": 5}
        self.assertEqual(report_suite._query_kind(micro_entry), "micro")
        self.assertEqual(report_suite._query_kind(studio_entry), "studio")
        self.assertIn("(studio, middle, 5s)", report_suite._query_label(studio_entry))
        self.assertEqual(report_suite._safe_rank(None, 42), 42)

    def test_plot_helpers_and_markdown_writer(self) -> None:
        rows = [
            {
                "method": "mfcc",
                "artist": "Artist",
                "title": "Song",
                "filename": "reference_clips/sample.wav",
                "query_kind": "studio",
                "query_label": "sample",
                "duration_s": 15.0,
                "stage1_rank": 2,
                "final_rank": 1,
                "score_faiss": 0.7,
                "score_fp": 0.8,
                "latency_s": 0.1,
            }
        ]
        coverage = {
            "n_queries": 1,
            "n_tracks": 1,
            "tracks_with_studio": 1,
            "tracks_with_micro": 0,
            "tracks_with_both": 0,
            "missing_pairs": [],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            with patch.object(report_suite, "ROOT", out_dir):
                rank_plot = report_suite._plot_real_query_ranks(rows, "mfcc", out_dir)
                score_plot = report_suite._plot_real_query_scores(rows, "mfcc", out_dir)
                md_path = out_dir / "report.md"
                report_suite._write_markdown(rows, coverage, md_path, [rank_plot, score_plot], ["mfcc"])
                content = md_path.read_text(encoding="utf-8")
                self.assertTrue(rank_plot.exists())
                self.assertTrue(score_plot.exists())
                self.assertIn("Report Evaluation Suite", content)
                self.assertIn("Artist", content)

    def test_run_report_suite_returns_payload(self) -> None:
        manifest = [
            {
                "filename": "reference_clips/sample.wav",
                "track_id": "track_1",
                "artist": "Artist",
                "title": "Song",
                "position": "middle",
                "duration_s": 15,
            }
        ]
        fake_eval = {
            "rank": 1,
            "stage1_rank": 2,
            "score_faiss": 0.7,
            "score_fp": 0.8,
            "latency_s": 0.2,
            "top1": True,
            "top5": True,
            "stage1_top1": False,
            "stage1_top5": True,
            "error": None,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            with patch("src.evaluation.report_suite.load_manifest", return_value=manifest), \
                 patch("src.evaluation.report_suite._evaluate_one", return_value=fake_eval):
                payload = report_suite.run_report_suite(methods=["mfcc"], out_dir=out_dir, plot=False)

        self.assertEqual(payload["coverage"]["n_queries"], 1)
        self.assertEqual(payload["rows"][0]["final_rank"], 1)


if __name__ == "__main__":
    unittest.main()
