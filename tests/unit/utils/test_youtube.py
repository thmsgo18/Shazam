from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.utils import youtube


class YoutubeUtilsTests(unittest.TestCase):
    def test_resolve_yt_dlp_command_prefers_current_python_module(self):
        with patch("src.utils.youtube.importlib.util.find_spec", return_value=object()):
            command = youtube._resolve_yt_dlp_command()
        self.assertEqual(command, [youtube.sys.executable, "-m", "yt_dlp"])

    def test_resolve_yt_dlp_command_falls_back_to_binary(self):
        with (
            patch("src.utils.youtube.importlib.util.find_spec", return_value=None),
            patch("src.utils.youtube.shutil.which", return_value="/usr/local/bin/yt-dlp"),
        ):
            command = youtube._resolve_yt_dlp_command()
        self.assertEqual(command, ["/usr/local/bin/yt-dlp"])

    def test_resolve_ffmpeg_binary_accepts_directory_env_var(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ffmpeg_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
            ffmpeg_path = Path(tmpdir) / ffmpeg_name
            ffmpeg_path.write_text("", encoding="utf-8")

            with (
                patch.dict(os.environ, {"FFMPEG_BINARY": tmpdir}, clear=False),
                patch("src.utils.youtube.shutil.which", return_value=None),
            ):
                resolved = youtube._resolve_ffmpeg_binary()

        self.assertEqual(resolved, str(ffmpeg_path))

    def test_cookie_argument_sets_are_opt_in(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(youtube._cookie_argument_sets(), [])

    def test_cookie_argument_sets_support_browser_list(self):
        with patch.dict(os.environ, {"YT_DLP_BROWSER": "firefox, chrome"}, clear=True):
            cookie_sets = youtube._cookie_argument_sets()

        self.assertEqual(
            cookie_sets,
            [
                ["--cookies-from-browser", "firefox"],
                ["--cookies-from-browser", "chrome"],
            ],
        )

    def test_pick_downloaded_audio_file_prefers_mp3(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            webm_path = Path(tmpdir) / "track.webm"
            mp3_path = Path(tmpdir) / "track.mp3"
            webm_path.write_text("", encoding="utf-8")
            mp3_path.write_text("", encoding="utf-8")

            picked = youtube._pick_downloaded_audio_file(tmpdir)

        self.assertEqual(picked, mp3_path)

    def test_map_download_error_for_authentication(self):
        error = "ERROR: Sign in to confirm your age"
        self.assertIn("YT_DLP_BROWSER", youtube._map_download_error(error))


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
