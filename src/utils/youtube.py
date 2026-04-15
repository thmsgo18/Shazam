"""
src/utils/youtube.py

Portable audio download utilities via yt-dlp.
Shared between ingestion, fingerprint reconstruction and RIR augmentation.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


_AUDIO_SUFFIX_PRIORITY = {
    ".mp3": 0,
    ".m4a": 1,
    ".opus": 2,
    ".aac": 3,
    ".wav": 4,
    ".flac": 5,
    ".mp4": 6,
    ".mka": 7,
    ".webm": 8,
}


def _normalize_ffmpeg_path(path_value: str | Path | None) -> str | None:
    """Accepts either an ffmpeg executable path or a directory containing it."""
    if not path_value:
        return None

    candidate = Path(path_value).expanduser()
    if candidate.is_file():
        return str(candidate)

    if candidate.is_dir():
        binary_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
        nested = candidate / binary_name
        if nested.exists():
            return str(nested)

    return None


# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------

def kill_proc(proc: subprocess.Popen) -> None:
    """Kills a subprocess cleanly (Unix process group, Windows kill)."""
    if sys.platform == "win32":
        proc.kill()
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            proc.kill()


def _resolve_yt_dlp_command() -> list[str] | None:
    """
    Resolves a portable yt-dlp runner.

    Prefers the current Python environment (`python -m yt_dlp`) so venv installs
    work reliably, then falls back to a `yt-dlp` executable in PATH.
    """
    if importlib.util.find_spec("yt_dlp") is not None:
        return [sys.executable, "-m", "yt_dlp"]

    binary = shutil.which("yt-dlp")
    if binary:
        return [binary]

    return None


def _resolve_ffmpeg_binary() -> str | None:
    """
    Resolves ffmpeg from an explicit env var, PATH, or optional imageio-ffmpeg.
    """
    env_value = os.getenv("FFMPEG_BINARY")
    if env_value:
        ffmpeg_path = _normalize_ffmpeg_path(env_value)
        if ffmpeg_path:
            return ffmpeg_path

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except Exception:
        return None

    try:
        ffmpeg_path = get_ffmpeg_exe()
    except Exception:
        return None

    return ffmpeg_path if ffmpeg_path and Path(ffmpeg_path).exists() else None


def _cookie_argument_sets() -> list[list[str]]:
    """
    Returns optional cookie arguments.

    Nothing is forced by default: browser cookies are machine-specific and were
    a common source of failures. They are now opt-in through environment
    variables only.
    """
    cookiefile = os.getenv("YT_DLP_COOKIEFILE")
    if cookiefile:
        cookie_path = Path(cookiefile).expanduser()
        if cookie_path.exists():
            return [["--cookies", str(cookie_path)]]

    browsers = os.getenv("YT_DLP_BROWSER", "").strip()
    if not browsers:
        return []

    cookie_sets: list[list[str]] = []
    for browser in (item.strip() for item in browsers.split(",")):
        if not browser:
            continue
        if browser.lower() == "none":
            return []
        cookie_sets.append(["--cookies-from-browser", browser])
    return cookie_sets


def _pick_downloaded_audio_file(tmpdir: str) -> Path | None:
    """Finds the best audio file produced by yt-dlp in the temp directory."""
    directory = Path(tmpdir)
    if not directory.exists():
        return None

    files = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in _AUDIO_SUFFIX_PRIORITY
    ]
    if not files:
        return None

    return min(
        files,
        key=lambda path: (
            _AUDIO_SUFFIX_PRIORITY.get(path.suffix.lower(), 99),
            -path.stat().st_mtime,
        ),
    )


def _build_download_command(target: str, output_template: str, ffmpeg_binary: str) -> list[str]:
    yt_dlp_command = _resolve_yt_dlp_command()
    if yt_dlp_command is None:
        raise FileNotFoundError("yt-dlp is not installed or not found in PATH")

    return [
        *yt_dlp_command,
        "--ignore-config",
        "--extract-audio",
        "--audio-format",
        "mp3",
        "--audio-quality",
        "5",
        "--format",
        "bestaudio/best",
        "--output",
        output_template,
        "--ffmpeg-location",
        ffmpeg_binary,
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "--no-progress",
        "--restrict-filenames",
        "--no-part",
        "--socket-timeout",
        "30",
        "--retries",
        "2",
        "--fragment-retries",
        "2",
        target,
    ]


def _is_auth_error(message: str) -> bool:
    lower = message.lower()
    return any(
        token in lower
        for token in (
            "sign in to confirm",
            "cookies-from-browser",
            "age-restricted",
            "age restriction",
            "this video may be inappropriate",
            "login required",
            "authentication required",
        )
    )


def _is_retryable_error(message: str) -> bool:
    lower = message.lower()
    return any(
        token in lower
        for token in (
            "timed out",
            "timeout",
            "connection reset",
            "temporarily unavailable",
            "temporary failure",
            "429",
            "too many requests",
            "http error 5",
            "network is unreachable",
            "ssl",
        )
    )


def _map_download_error(message: str) -> str:
    lower = message.lower()

    if _is_auth_error(message):
        return (
            "video requires authentication/cookies "
            "(configure YT_DLP_BROWSER or YT_DLP_COOKIEFILE if needed)"
        )
    if "video unavailable" in lower or "this video is unavailable" in lower:
        return "video unavailable"
    if "not available in your country" in lower or "not available in your region" in lower:
        return "video unavailable in this region"
    if "unsupported url" in lower or "invalid url" in lower:
        return "invalid YouTube URL"
    if "ffmpeg" in lower and ("not found" in lower or "missing" in lower or "not installed" in lower):
        return "ffmpeg is not installed or not found"
    if "requested format is not available" in lower:
        return "no compatible audio format found"
    if "unable to extract" in lower:
        return "yt-dlp could not extract the video metadata"
    if "http error 429" in lower or "too many requests" in lower:
        return "YouTube rate limit reached"

    lines = [line.strip() for line in message.splitlines() if line.strip()]
    return lines[-1] if lines else "yt-dlp error"


def _run_download(
    target: str,
    retries: int,
    timeout: int,
    stop_event=None,
) -> tuple[str | None, str | None, str]:
    """
    Downloads a single target (query or direct URL) to a temp directory.
    """
    ffmpeg_binary = _resolve_ffmpeg_binary()
    if ffmpeg_binary is None:
        return None, None, "ffmpeg is not installed or not found"

    try:
        _build_download_command(target, "%(id)s.%(ext)s", ffmpeg_binary)
    except FileNotFoundError as exc:
        return None, None, str(exc)

    cookie_sets = _cookie_argument_sets()
    last_reason = "yt-dlp error"

    for attempt in range(retries):
        if stop_event and stop_event.is_set():
            return None, None, "stop requested"

        should_retry = False
        auth_failed = False

        for cookie_args in [[]] + cookie_sets:
            if cookie_args and not auth_failed:
                break

            tmpdir = tempfile.mkdtemp()
            proc = None
            try:
                output_template = str(Path(tmpdir) / "%(id)s.%(ext)s")
                cmd = _build_download_command(target, output_template, ffmpeg_binary)
                cmd = [*cmd[:-1], *cookie_args, cmd[-1]]

                popen_kw: dict = {
                    "stderr": subprocess.PIPE,
                    "stdout": subprocess.DEVNULL,
                    "text": True,
                }
                if sys.platform == "win32":
                    popen_kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
                else:
                    popen_kw["start_new_session"] = True

                proc = subprocess.Popen(cmd, **popen_kw)
                try:
                    _, stderr = proc.communicate(timeout=timeout)
                except subprocess.TimeoutExpired:
                    kill_proc(proc)
                    proc.wait()
                    shutil.rmtree(tmpdir, ignore_errors=True)
                    last_reason = "timeout"
                    should_retry = True
                    break

                if proc.returncode != 0:
                    auth_failed = _is_auth_error(stderr)
                    last_reason = _map_download_error(stderr)
                    shutil.rmtree(tmpdir, ignore_errors=True)
                    if auth_failed:
                        continue
                    should_retry = _is_retryable_error(stderr)
                    break

                audio_file = _pick_downloaded_audio_file(tmpdir)
                if audio_file is None:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                    last_reason = "download finished but no audio file was created"
                    should_retry = True
                    break

                return tmpdir, str(audio_file), ""

            except Exception as exc:
                if proc is not None:
                    try:
                        kill_proc(proc)
                    except Exception:
                        pass
                shutil.rmtree(tmpdir, ignore_errors=True)
                last_reason = str(exc)
                break

        if not should_retry or attempt >= retries - 1:
            break
        time.sleep(2)

    return None, None, last_reason


# ---------------------------------------------------------------------------
# Download by search (yt-dlp ytsearch)
# ---------------------------------------------------------------------------

def download_audio_search(
    artist: str,
    title: str,
    retries: int = 3,
    stop_event=None,
) -> tuple[str, str, str, None] | tuple[None, None, None, str]:
    """
    Downloads audio by searching "{artist} {title}" on YouTube.

    Returns:
        (tmpdir, audio_path, youtube_url, None)  — success
        (None,   None,      None,        reason) — failure
    """
    queries = [
        f"ytsearch1:{artist} {title} official audio",
        f"ytsearch1:{artist} {title}",
    ]
    last_reason = "not found on YouTube"

    for query in queries:
        if stop_event and stop_event.is_set():
            return None, None, None, "stop requested"

        tmpdir, audio_path, reason = _run_download(
            target=query,
            retries=retries,
            timeout=45,
            stop_event=stop_event,
        )
        if audio_path is None:
            last_reason = reason
            continue

        youtube_url = f"https://www.youtube.com/watch?v={Path(audio_path).stem}"
        return tmpdir, audio_path, youtube_url, None

    return None, None, None, last_reason


# ---------------------------------------------------------------------------
# Download from direct YouTube URL
# ---------------------------------------------------------------------------

def download_audio_from_url(
    url: str,
    retries: int = 3,
    stop_event=None,
) -> tuple[str, str] | tuple[None, None]:
    """
    Downloads audio from a direct YouTube URL.

    Returns:
        (tmpdir, audio_path) — success
        (None, None)         — failure
    """
    tmpdir, audio_path, _ = _run_download(
        target=url,
        retries=retries,
        timeout=60,
        stop_event=stop_event,
    )
    if audio_path is None:
        return None, None
    return tmpdir, audio_path


# ---------------------------------------------------------------------------
# Audio loading without deadlock (macOS)
# ---------------------------------------------------------------------------

def load_audio_safe(mp3_path: str, sr: int):
    """
    Loads an audio file without macOS deadlock:
      ffmpeg converts it to mono WAV (clean subprocess, no internal threads)
      soundfile reads the WAV (thread-safe, no OpenBLAS)

    Returns float32 np.ndarray or None on error.
    """
    import soundfile as sf

    ffmpeg_binary = _resolve_ffmpeg_binary()
    if ffmpeg_binary is None:
        return None

    wav_path = mp3_path + "_tmp.wav"
    try:
        result = subprocess.run(
            [
                ffmpeg_binary,
                "-y",
                "-i",
                mp3_path,
                "-ar",
                str(sr),
                "-ac",
                "1",
                "-sample_fmt",
                "s16",
                wav_path,
            ],
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0:
            return None
        data, _ = sf.read(wav_path, dtype="float32")
        return data
    except Exception:
        return None
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass
