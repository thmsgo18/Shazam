from __future__ import annotations

import unittest

import numpy as np

from src.features.fingerprint import extract_fingerprint, fingerprint_similarity
from tests.helpers.audio_factory import mixed_wave


class FingerprintTests(unittest.TestCase):
    def test_extract_fingerprint_returns_v2_hashes(self) -> None:
        waveform = mixed_wave(duration_s=3.0)

        hashes = extract_fingerprint(waveform, sr=22050)

        self.assertIsInstance(hashes, set)
        self.assertGreater(len(hashes), 0)
        self.assertTrue(all(len(item) == 4 for item in hashes))

    def test_fingerprint_similarity_returns_zero_for_empty_inputs(self) -> None:
        self.assertEqual(fingerprint_similarity(set(), {(1, 2, 3)}), 0.0)
        self.assertEqual(fingerprint_similarity({(1, 2, 3)}, set()), 0.0)

    def test_fingerprint_similarity_supports_v1_format(self) -> None:
        query = {(1, 2, 3), (4, 5, 6)}
        candidate = {(1, 2, 3), (9, 9, 9)}

        score = fingerprint_similarity(query, candidate)

        self.assertEqual(score, 0.5)

    def test_fingerprint_similarity_uses_temporal_alignment_for_v2(self) -> None:
        query = {(10, 20, 4, 100), (11, 21, 5, 110)}
        candidate = {(10, 20, 4, 150), (11, 21, 5, 160), (99, 99, 9, 999)}

        score = fingerprint_similarity(query, candidate)

        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
