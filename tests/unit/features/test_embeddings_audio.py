from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from src.features.embeddings_audio import embed_segment, mfcc_stats_embedding
from tests.helpers.audio_factory import mixed_wave


class EmbeddingsAudioTests(unittest.TestCase):
    def test_mfcc_stats_embedding_returns_zero_vector_for_empty_input(self) -> None:
        embedding = mfcc_stats_embedding(np.array([], dtype=np.float32), sr=22050, n_mfcc=4)

        self.assertEqual(embedding.shape, (8,))
        np.testing.assert_array_equal(embedding, np.zeros((8,), dtype=np.float32))

    def test_mfcc_stats_embedding_returns_normalized_embedding(self) -> None:
        embedding = mfcc_stats_embedding(mixed_wave(duration_s=1.0), sr=22050, n_mfcc=8)

        self.assertEqual(embedding.shape, (16,))
        self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0, places=5)

    def test_embed_segment_dispatches_to_selected_backend(self) -> None:
        waveform = np.ones(32, dtype=np.float32)

        with patch("src.features.embeddings_audio.mfcc_stats_embedding", return_value=np.array([1.0])) as mfcc:
            result = embed_segment(waveform, 22050, method="mfcc")
        mfcc.assert_called_once()
        np.testing.assert_array_equal(result, np.array([1.0]))

        with patch("src.features.embeddings_audio.clap_embedding", return_value=np.array([2.0])) as clap:
            result = embed_segment(waveform, 22050, method="clap", clap_model_name="demo")
        clap.assert_called_once()
        np.testing.assert_array_equal(result, np.array([2.0]))

        with patch("src.features.embeddings_audio.muq_embedding", return_value=np.array([3.0])) as muq:
            result = embed_segment(waveform, 22050, method="muq", muq_model_name="demo")
        muq.assert_called_once()
        np.testing.assert_array_equal(result, np.array([3.0]))

        with patch("src.features.embeddings_audio.mert_embedding", return_value=np.array([4.0])) as mert:
            result = embed_segment(waveform, 22050, method="mert", mert_model_name="demo")
        mert.assert_called_once()
        np.testing.assert_array_equal(result, np.array([4.0]))

    def test_embed_segment_requires_model_name_for_model_backends(self) -> None:
        waveform = np.ones(16, dtype=np.float32)

        with self.assertRaises(ValueError):
            embed_segment(waveform, 22050, method="clap")
        with self.assertRaises(ValueError):
            embed_segment(waveform, 22050, method="muq")
        with self.assertRaises(ValueError):
            embed_segment(waveform, 22050, method="mert")
        with self.assertRaises(ValueError):
            embed_segment(waveform, 22050, method="unknown")


if __name__ == "__main__":
    unittest.main()
