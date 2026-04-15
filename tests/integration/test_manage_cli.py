from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

import manage
from tests.helpers.audio_factory import mixed_wave, write_wav


class ManageCliIntegrationTests(unittest.TestCase):
    def test_identify_command_routes_to_api_cli_without_target(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = write_wav(Path(tmpdir) / "query.wav", mixed_wave(duration_s=0.5))
            with patch("src.api.app.run_identify_cli") as run_identify_cli:
                result = runner.invoke(manage.cli, ["identify", str(audio_path), "--top", "3"])

        self.assertEqual(result.exit_code, 0, msg=result.output)
        run_identify_cli.assert_called_once()

    def test_identify_command_routes_to_find_track_with_target(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = write_wav(Path(tmpdir) / "query.wav", mixed_wave(duration_s=0.5))
            with patch("src.evaluation.find_track.run_find_track") as run_find_track:
                result = runner.invoke(manage.cli, ["identify", str(audio_path), "--target", "track_1"])

        self.assertEqual(result.exit_code, 0, msg=result.output)
        run_find_track.assert_called_once()


if __name__ == "__main__":
    unittest.main()
