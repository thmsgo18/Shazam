from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation import report_analyses


class ReportAnalysesTests(unittest.TestCase):
    def test_basic_helpers(self) -> None:
        self.assertEqual(report_analyses._format_table_value("score_pct", 12.345), "12.3")
        self.assertEqual(report_analyses._format_table_value("value", None), "NF")
        self.assertEqual(report_analyses._safe_mean([1, 2, 3]), 2.0)
        self.assertEqual(report_analyses._safe_pct([True, False]), 50.0)

    def test_manifest_helpers(self) -> None:
        manifest = [
            {"track_id": "a", "filename": "reference_clips/a.wav", "position": "middle", "duration_s": 15},
            {"track_id": "a", "filename": "ref-mic-close-clean.wav", "position": "mic_close_clean", "duration_s": 15},
            {"track_id": "b", "filename": "reference_clips/b.wav", "position": "middle", "duration_s": 5},
        ]
        limited = report_analyses._limit_manifest(manifest, 1)
        ordered = report_analyses._manifest_eval_order(manifest)
        attrs = report_analyses._parse_query_attrs({"filename": "song-mic-far-speech.wav", "position": "mic_far_speech", "duration_s": 30})
        sig = report_analyses._manifest_signature(manifest, ["mfcc"])
        self.assertEqual(len(limited), 2)
        self.assertEqual(ordered[0]["track_id"], "a")
        self.assertEqual(attrs["query_kind"], "micro")
        self.assertEqual(attrs["distance"], "far")
        self.assertEqual(attrs["speech"], "speech")
        self.assertEqual(attrs["duration_bucket"], 30)
        self.assertEqual(len(sig), 12)

    def test_cache_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "resume.jsonl"
            report_analyses._append_resume_row(cache_path, {"_cache_key": "one", "value": 1})
            cache_path.write_text(cache_path.read_text(encoding="utf-8") + "not-json\n", encoding="utf-8")
            cache = report_analyses._load_resume_cache(cache_path)
        self.assertEqual(cache["one"]["value"], 1)

    def test_rir_and_topk_summary_helpers(self) -> None:
        pairs = [
            {"track_id": "t1", "filename": "a", "query_kind": "studio", "duration_bucket": 15, "without_rir_rank": 4, "with_rir_rank": 2},
            {"track_id": "t2", "filename": "b", "query_kind": "micro", "speech": "clean", "distance": "close", "without_rir_rank": None, "with_rir_rank": 5},
            {"track_id": "t3", "filename": "c", "query_kind": "micro", "speech": "speech", "distance": "far", "without_rir_rank": 8, "with_rir_rank": 3},
        ]
        rir_metrics = report_analyses._rir_metrics_from_rows([{"rank": 1}, {"rank": 7}, {"rank": None}])
        rir_overview = report_analyses._build_rir_overview_rows(pairs)
        rir_topk = report_analyses._build_rir_topk_summary_rows(pairs)

        rows = [
            {"final_rank": 1, "stage1_rank": 4, "query_kind": "studio", "duration_bucket": 15},
            {"final_rank": 5, "stage1_rank": 8, "query_kind": "micro", "speech": "clean", "distance": "close"},
            {"final_rank": None, "stage1_rank": None, "query_kind": "micro", "speech": "speech", "distance": "far"},
        ]
        topk_metrics = report_analyses._topk_metrics(rows)
        topk_summary = report_analyses._build_topk_summary_rows(rows)

        self.assertEqual(rir_metrics["top1_pct"], 33.3)
        self.assertTrue(any(row["Scenario"] == "File excerpt" for row in rir_overview))
        self.assertTrue(any(row["Category"] == "Overall" for row in rir_topk))
        self.assertEqual(topk_metrics["n_queries"], 3)
        self.assertTrue(any(row["category"] == "overall" for row in topk_summary))

    def test_rir_pair_helpers(self) -> None:
        result = {
            "conditions": ["clean"],
            "results": {
                "mfcc": {
                    "clean_without_rir": {
                        "per_track": [
                            {"track_id": "t1", "filename": "song-mic-close-clean.wav", "position": "mic_close_clean", "duration_s": 15, "artist": "Artist", "title": "Song", "rank": 4, "faiss_score": 2.0}
                        ]
                    },
                    "clean_with_rir": {
                        "per_track": [
                            {"track_id": "t1", "filename": "song-mic-close-clean.wav", "position": "mic_close_clean", "duration_s": 15, "artist": "Artist", "title": "Song", "rank": 2, "faiss_score": 3.0}
                        ]
                    },
                }
            },
        }
        pairs = report_analyses._rir_pairs_for_condition(result, "mfcc", "clean")
        condition_rows = report_analyses._build_rir_condition_summary_rows(result, "mfcc")
        self.assertEqual(pairs[0]["with_rir_rank"], 2)
        self.assertEqual(condition_rows[0]["Method"], "mfcc")


if __name__ == "__main__":
    unittest.main()
