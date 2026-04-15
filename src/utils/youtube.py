"""
src/utils/youtube.py

Utilitaires de téléchargement audio via yt-dlp.
Partagés entre l'ingestion, la reconstruction des fingerprints et l'augmentation RIR.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers processus
# ---------------------------------------------------------------------------

def kill_proc(proc: subprocess.Popen) -> None:
    """Tue un subprocess proprement (groupe de processus Unix, kill Windows)."""
    if sys.platform == "win32":
        proc.kill()
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            proc.kill()


# ---------------------------------------------------------------------------
# Téléchargement par recherche (yt-dlp ytsearch)
# ---------------------------------------------------------------------------

def download_audio_search(
    artist: str,
    title: str,
    retries: int = 3,
    stop_event=None,
) -> tuple[str, str, str, None] | tuple[None, None, None, str]:
    """
    Télécharge l'audio en cherchant "{artist} {title}" sur YouTube.

    Retourne :
        (tmpdir, mp3_path, youtube_url, None)  — succès
        (None,   None,     None,        raison) — échec

    Args:
        artist:      artiste du morceau.
        title:       titre du morceau.
        retries:     nombre de tentatives par requête.
        stop_event:  threading.Event facultatif (arrêt propre Ctrl+C).
    """
    queries = [
        f"{artist} {title} official audio",
        f"{artist} {title}",
    ]
    base_cmd = [
        "yt-dlp",
        "--extract-audio", "--audio-format", "mp3",
        "--audio-quality", "5",
        "--output", "%(id)s.%(ext)s",
        "--quiet", "--no-warnings",
        "--socket-timeout", "30",
        "--cookies-from-browser", "chrome",
        "--remote-components", "ejs:github",
    ]
    last_reason = "introuvable sur YouTube"

    for query in queries:
        if stop_event and stop_event.is_set():
            return None, None, None, "arrêt demandé"
        cmd = [base_cmd[0], f"ytsearch1:{query}"] + base_cmd[1:]
        for attempt in range(retries):
            tmpdir = tempfile.mkdtemp()
            proc = None
            try:
                popen_kw: dict = {"cwd": tmpdir, "stderr": subprocess.PIPE}
                if sys.platform == "win32":
                    popen_kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
                else:
                    popen_kw["start_new_session"] = True
                proc = subprocess.Popen(cmd, **popen_kw)
                try:
                    _, stderr_bytes = proc.communicate(timeout=45)
                except subprocess.TimeoutExpired:
                    kill_proc(proc)
                    proc.wait()
                    shutil.rmtree(tmpdir, ignore_errors=True)
                    last_reason = "timeout"
                    if attempt < retries - 1:
                        time.sleep(3)
                        continue
                    break

                if proc.returncode != 0:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                    stderr = stderr_bytes.decode(errors="ignore")
                    if "Sign in" in stderr or "age" in stderr:
                        last_reason = "restriction d'âge YouTube"
                        break
                    if "unavailable" in stderr or "not available" in stderr:
                        last_reason = "vidéo indisponible dans cette région"
                        break
                    last_reason = "erreur yt-dlp"
                    if attempt < retries - 1:
                        time.sleep(3)
                        continue
                    break

                files = list(Path(tmpdir).glob("*.mp3"))
                if not files:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                    last_reason = "introuvable sur YouTube"
                    break

                youtube_url = f"https://www.youtube.com/watch?v={files[0].stem}"
                return tmpdir, str(files[0]), youtube_url, None

            except Exception as exc:
                if proc is not None:
                    try:
                        kill_proc(proc)
                    except Exception:
                        pass
                shutil.rmtree(tmpdir, ignore_errors=True)
                last_reason = str(exc)
                break

    return None, None, None, last_reason


# ---------------------------------------------------------------------------
# Téléchargement depuis une URL YouTube directe
# ---------------------------------------------------------------------------

def download_audio_from_url(
    url: str,
    retries: int = 3,
    stop_event=None,
) -> tuple[str, str] | tuple[None, None]:
    """
    Télécharge l'audio depuis une URL YouTube directe.

    Retourne :
        (tmpdir, mp3_path) — succès
        (None, None)       — échec

    Args:
        url:        URL YouTube directe.
        retries:    nombre de tentatives.
        stop_event: threading.Event facultatif (arrêt propre Ctrl+C).
    """
    cmd = [
        "yt-dlp", url,
        "--extract-audio", "--audio-format", "mp3",
        "--audio-quality", "5",
        "--output", "%(id)s.%(ext)s",
        "--quiet", "--no-warnings",
        "--socket-timeout", "30",
        "--cookies-from-browser", "chrome",
        "--remote-components", "ejs:github",
    ]
    for attempt in range(retries):
        if stop_event and stop_event.is_set():
            return None, None
        tmpdir = tempfile.mkdtemp()
        try:
            kwargs: dict = {"cwd": tmpdir, "stderr": subprocess.PIPE}
            if sys.platform != "win32":
                kwargs["start_new_session"] = True
            proc = subprocess.Popen(cmd, **kwargs)
            try:
                proc.communicate(timeout=60)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception:
                    proc.kill()
                proc.wait()
                shutil.rmtree(tmpdir, ignore_errors=True)
                if attempt < retries - 1:
                    time.sleep(2)
                continue
            if proc.returncode != 0:
                shutil.rmtree(tmpdir, ignore_errors=True)
                if attempt < retries - 1:
                    time.sleep(2)
                continue
            files = list(Path(tmpdir).glob("*.mp3"))
            if files:
                return tmpdir, str(files[0])
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            shutil.rmtree(tmpdir, ignore_errors=True)
    return None, None


# ---------------------------------------------------------------------------
# Chargement audio sans deadlock (macOS)
# ---------------------------------------------------------------------------

def load_audio_safe(mp3_path: str, sr: int):
    """
    Charge un MP3 sans deadlock macOS :
      ffmpeg convertit MP3 → WAV mono (subprocess propre, pas de threads internes)
      soundfile lit le WAV (thread-safe, pas d'OpenBLAS)

    Retourne np.ndarray float32 ou None si erreur.
    """
    import numpy as np
    import soundfile as sf

    wav_path = mp3_path + "_tmp.wav"
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path,
             "-ar", str(sr), "-ac", "1", "-sample_fmt", "s16", wav_path],
            capture_output=True, timeout=60,
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
