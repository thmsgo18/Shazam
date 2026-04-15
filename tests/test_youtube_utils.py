import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
