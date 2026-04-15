from __future__ import annotations

import tempfile
import unittest

import numpy as np

from src.audio.loading import load_audio
from tests.helpers.audio_factory import sine_wave, write_wav


class LoadAudioTests(unittest.TestCase):
    def test_load_audio_resamples_to_target_rate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_wav(f"{tmpdir}/tone.wav", sine_wave(sr=16000, duration_s=1.0), sr=16000)

            waveform, sr = load_audio(str(path), target_sr=22050)

        self.assertEqual(sr, 22050)
        self.assertIsInstance(waveform, np.ndarray)
        self.assertEqual(waveform.ndim, 1)
        self.assertGreater(len(waveform), 0)

    def test_load_audio_can_keep_stereo(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            left = sine_wave(frequency=220.0, duration_s=0.5)
            right = sine_wave(frequency=440.0, duration_s=0.5)
            stereo = np.stack([left, right], axis=1)
            path = write_wav(f"{tmpdir}/stereo.wav", stereo, sr=22050)

            waveform, sr = load_audio(str(path), target_sr=22050, mono=False)

        self.assertEqual(sr, 22050)
        self.assertEqual(waveform.ndim, 2)
        self.assertEqual(waveform.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
