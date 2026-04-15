from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.evaluation import benchmark
from tests.helpers.audio_factory import mixed_wave, write_wav


class BenchmarkTests(unittest.TestCase):
    def test_lookup_target_reads_manifest_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "data" / "raw" / "manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    [
                        {
                            "filename": "clip.wav",
                            "track_id": "track_1",
                            "artist": "Artist",
                            "title": "Song",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with patch.object(benchmark, "ROOT", root):
                track_id, label = benchmark._lookup_target("clip.wav")

        self.assertEqual(track_id, "track_1")
        self.assertEqual(label, "Artist — Song")

    def test_add_noise_at_snr_preserves_zero_signal(self) -> None:
        waveform = np.zeros(100, dtype=np.float32)
        np.testing.assert_array_equal(benchmark.add_noise_at_snr(waveform, 20), waveform)

    def test_compute_snr_returns_infinity_for_identical_signals(self) -> None:
        waveform = np.ones(16, dtype=np.float32)
        self.assertEqual(benchmark.compute_snr(waveform, waveform), float("inf"))

    def test_build_test_suite_fast_mode_creates_real_and_simulated_cases(self) -> None:
        path = None
        tests = []
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = write_wav(Path(tmpdir) / "input.wav", mixed_wave(duration_s=1.0))
                tests = benchmark.build_test_suite(str(path), full=False)
        finally:
            for test in tests:
                if test.get("tmp") and os.path.exists(test["path"]):
                    os.unlink(test["path"])

        self.assertEqual(len(tests), 4)
        self.assertEqual(tests[0]["type"], "real")
        self.assertTrue(all("label" in test for test in tests))

    def test_run_identification_extracts_rank_and_scores(self) -> None:
        fake_results = [
            ("track_x", 0.9, 0.9, 0.0),
            ("track_1", 0.4, 0.4, 0.0),
        ]

        with patch("src.retrieval.query_pipeline.identify_track", return_value=fake_results):
            result = benchmark.run_identification("audio.wav", target_track_id="track_1", method="mfcc")

        self.assertTrue(result["success"])
        self.assertEqual(result["rank"], 2)
        self.assertEqual(result["score_target_final"], 0.4)
        self.assertEqual(result["ratio"], 2.25)


if __name__ == "__main__":
    unittest.main()
