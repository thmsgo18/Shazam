from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from src.audio.preprocessing import iter_segments, preprocess_query
from tests.helpers.audio_factory import mixed_wave


class IterSegmentsTests(unittest.TestCase):
    def test_iter_segments_returns_full_and_padded_last_segment(self) -> None:
        waveform = np.arange(95, dtype=np.float32)

        segments = list(iter_segments(waveform, sr=10, win_s=4.0, hop_s=3.0, min_win=0.5))

        self.assertEqual([start for start, _ in segments], [0.0, 3.0, 6.0])
        self.assertEqual([len(seg) for _, seg in segments], [40, 40, 40])
        np.testing.assert_array_equal(segments[-1][1][:35], waveform[60:95])
        np.testing.assert_array_equal(segments[-1][1][-5:], np.zeros(5, dtype=np.float32))

    def test_iter_segments_validates_parameters(self) -> None:
        waveform = np.ones(10, dtype=np.float32)
        with self.assertRaises(ValueError):
            list(iter_segments(waveform, sr=0))
        with self.assertRaises(ValueError):
            list(iter_segments(waveform, sr=10, win_s=0))
        with self.assertRaises(ValueError):
            list(iter_segments(waveform, sr=10, hop_s=0))


class PreprocessQueryTests(unittest.TestCase):
    def test_preprocess_query_keeps_shape_dtype_and_peak_bound(self) -> None:
        waveform = mixed_wave(duration_s=1.0)

        with patch("src.config.OPT_QUERY_DENOISE", False):
            processed = preprocess_query(waveform, sr=22050)

        self.assertEqual(processed.shape, waveform.shape)
        self.assertEqual(processed.dtype, np.float32)
        self.assertFalse(np.isnan(processed).any())
        self.assertLessEqual(float(np.abs(processed).max()), 0.951)


if __name__ == "__main__":
    unittest.main()
