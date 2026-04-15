from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.utils import youtube


class YoutubeUtilsExtendedTests(unittest.TestCase):
    def test_normalize_ffmpeg_path_accepts_file_and_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ffmpeg_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
            ffmpeg_path = Path(tmpdir) / ffmpeg_name
            ffmpeg_path.write_text("", encoding="utf-8")

            self.assertEqual(youtube._normalize_ffmpeg_path(ffmpeg_path), str(ffmpeg_path))
            self.assertEqual(youtube._normalize_ffmpeg_path(tmpdir), str(ffmpeg_path))

    def test_build_download_command_contains_expected_flags(self) -> None:
        with patch("src.utils.youtube._resolve_yt_dlp_command", return_value=["python", "-m", "yt_dlp"]):
            command = youtube._build_download_command("ytsearch1:test", "%(id)s.%(ext)s", "/usr/bin/ffmpeg")

        self.assertIn("--extract-audio", command)
        self.assertIn("--ffmpeg-location", command)
        self.assertEqual(command[-1], "ytsearch1:test")

    def test_error_classification_helpers(self) -> None:
        self.assertTrue(youtube._is_auth_error("Please sign in to confirm your age"))
        self.assertTrue(youtube._is_retryable_error("HTTP Error 429: Too Many Requests"))
        self.assertEqual(youtube._map_download_error("ERROR: Requested format is not available"), "no compatible audio format found")

    def test_run_download_fails_fast_when_ffmpeg_is_missing(self) -> None:
        with patch("src.utils.youtube._resolve_ffmpeg_binary", return_value=None):
            tmpdir, audio_path, reason = youtube._run_download("ytsearch1:test", retries=1, timeout=1)

        self.assertIsNone(tmpdir)
        self.assertIsNone(audio_path)
        self.assertIn("ffmpeg", reason)

    def test_run_download_returns_audio_file_on_success(self) -> None:
        fake_proc = MagicMock()
        fake_proc.communicate.return_value = ("", "")
        fake_proc.returncode = 0
        fake_proc.pid = 123

        with patch("src.utils.youtube._resolve_ffmpeg_binary", return_value="/usr/bin/ffmpeg"), \
             patch("src.utils.youtube._build_download_command", return_value=["yt-dlp", "target"]), \
             patch("src.utils.youtube._cookie_argument_sets", return_value=[]), \
             patch("src.utils.youtube.tempfile.mkdtemp", return_value="/tmp/test-ytdlp"), \
             patch("src.utils.youtube.subprocess.Popen", return_value=fake_proc), \
             patch("src.utils.youtube._pick_downloaded_audio_file", return_value=Path("/tmp/test-ytdlp/audio.mp3")):
            tmpdir, audio_path, reason = youtube._run_download("ytsearch1:test", retries=1, timeout=10)

        self.assertEqual(tmpdir, "/tmp/test-ytdlp")
        self.assertEqual(audio_path, "/tmp/test-ytdlp/audio.mp3")
        self.assertEqual(reason, "")


if __name__ == "__main__":
    unittest.main()
