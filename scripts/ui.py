"""
scripts/ui.py

Interface graphique Streamlit — Shazam Maison.

Lancement :
    streamlit run scripts/ui.py

Prérequis :
    pip install streamlit>=1.31.0
"""
from __future__ import annotations

import os
import random
import subprocess
import sys
import tempfile
import urllib.parse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*upsample_bicubic2d.*", category=UserWarning)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.retrieval.query_pipeline import identify_track


# ──────────────────────────────────────────────────────────────────────────────
# Page config — doit être le premier appel Streamlit
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Shazam Maison",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ──────────────────────────────────────────────────────────────────────────────
# Traductions
# ──────────────────────────────────────────────────────────────────────────────

TRANSLATIONS: dict[str, dict[str, str]] = {
    "fr": {
        "title":           "Shazam Maison",
        "subtitle":        "Reconnaissance musicale par IA",
        "listen_title":    "Identifier un morceau",
        "listen_hint":     f"Enregistrez ~{config.UI_LISTEN_DURATION}s puis arrêtez",
        "or":              "ou déposez un fichier",
        "upload_hint":     "MP3 · WAV · FLAC · M4A · MP4 · MOV…",
        "identifying":     "Identification en cours…",
        "found":           "Morceau identifié",
        "uncertain":       "Résultats possibles",
        "uncertain_hint":  "Le système hésite — voici les candidats les plus probables",
        "confident_hint":  "Identifié avec confiance",
        "no_result":       "Morceau non trouvé",
        "try_again":       "Essayez avec un extrait plus long ou plus clair",
        "recommendations": "Dans le même genre",
        "listen_on":       "Écouter sur",
        "album":           "Album",
        "genre":           "Genre",
        "year":            "Année",
        "github":          "Code source",
        "project":         "Projet Big Data · Master IAD",
        "lang_switch":     "English",
    },
    "en": {
        "title":           "Shazam Maison",
        "subtitle":        "AI-powered music recognition",
        "listen_title":    "Identify a track",
        "listen_hint":     f"Record ~{config.UI_LISTEN_DURATION}s then stop",
        "or":              "or drop a file",
        "upload_hint":     "MP3 · WAV · FLAC · M4A · MP4 · MOV…",
        "identifying":     "Identifying…",
        "found":           "Track identified",
        "uncertain":       "Possible matches",
        "uncertain_hint":  "The system is unsure — here are the most likely candidates",
        "confident_hint":  "Identified with confidence",
        "no_result":       "Track not found",
        "try_again":       "Try with a longer or clearer excerpt",
        "recommendations": "In the same genre",
        "listen_on":       "Listen on",
        "album":           "Album",
        "genre":           "Genre",
        "year":            "Year",
        "github":          "Source code",
        "project":         "Big Data project · Master IAD",
        "lang_switch":     "Français",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────────────────────

CSS = """
<style>
/* ── Masquer le chrome Streamlit ───────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Base ──────────────────────────────────────────────────────────────────── */
.stApp {
    background: #09090E;
    color: #E2E2EA;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 5rem;
    max-width: 700px !important;
}

/* ── Header ────────────────────────────────────────────────────────────────── */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 1.5rem;
    margin-bottom: 2rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.header-left {
    display: flex;
    align-items: center;
    gap: 12px;
}
.header-logo {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, #6C63FF, #3EC6C6);
    border-radius: 11px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
}
.header-title {
    font-size: 20px;
    font-weight: 800;
    letter-spacing: -0.4px;
    background: linear-gradient(90deg, #A09BFF, #3EC6C6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.header-right {
    display: flex;
    align-items: center;
    gap: 16px;
}
.github-link {
    display: flex;
    align-items: center;
    gap: 6px;
    color: #555;
    text-decoration: none;
    font-size: 12px;
    font-weight: 500;
    transition: color 0.2s;
    padding: 6px 12px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.07);
}
.github-link:hover { color: #A09BFF; border-color: rgba(160,155,255,0.3); }

/* ── Section écoute ────────────────────────────────────────────────────────── */
.listen-hero {
    text-align: center;
    padding: 2rem 0 1.5rem;
}
.listen-hero-title {
    font-size: 30px;
    font-weight: 800;
    color: #F0F0FA;
    margin-bottom: 8px;
    letter-spacing: -0.5px;
}
.listen-hero-hint {
    font-size: 14px;
    color: #555;
    margin-bottom: 2rem;
}

/* Bouton micro — agrandir et arrondir le widget natif de Streamlit */
[data-testid="stAudioInput"] {
    display: flex !important;
    justify-content: center !important;
}
[data-testid="stAudioInput"] > div:first-child {
    background: rgba(108,99,255,0.08) !important;
    border: 2px solid rgba(108,99,255,0.35) !important;
    border-radius: 50% !important;
    width: 110px !important;
    height: 110px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 0 0 0 rgba(108,99,255,0.4);
    animation: microPulse 3s ease-in-out infinite;
}
[data-testid="stAudioInput"] > div:first-child:hover {
    border-color: rgba(108,99,255,0.8) !important;
    background: rgba(108,99,255,0.15) !important;
    transform: scale(1.06);
    box-shadow: 0 0 30px rgba(108,99,255,0.2);
}
@keyframes microPulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(108,99,255,0.25); }
    50%       { box-shadow: 0 0 0 14px rgba(108,99,255,0); }
}
[data-testid="stAudioInput"] label { display: none !important; }

/* ── Séparateur ────────────────────────────────────────────────────────────── */
.separator {
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 1.75rem 0 1.25rem;
    color: #333;
    font-size: 12px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.separator::before, .separator::after {
    content: "";
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.05);
}

/* ── Zone upload ───────────────────────────────────────────────────────────── */
[data-testid="stFileUploadDropzone"] {
    background: rgba(255,255,255,0.015) !important;
    border: 1.5px dashed rgba(255,255,255,0.1) !important;
    border-radius: 18px !important;
    transition: all 0.2s;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: rgba(108,99,255,0.4) !important;
    background: rgba(108,99,255,0.03) !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] small { color: #444 !important; font-size: 12px !important; }

/* ── Carte résultat ────────────────────────────────────────────────────────── */
.result-wrap {
    margin-top: 2rem;
    animation: slideUp 0.45s cubic-bezier(0.16, 1, 0.3, 1);
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 20px;
}
.badge-ok   { background: rgba(62,198,198,0.12); color: #3EC6C6; border: 1px solid rgba(62,198,198,0.25); }
.badge-warn { background: rgba(255,193,7,0.12);  color: #FFB800; border: 1px solid rgba(255,193,7,0.25); }

/* Pochette */
.cover-img {
    width: 100%;
    aspect-ratio: 1;
    object-fit: cover;
    border-radius: 18px;
    box-shadow: 0 24px 64px rgba(0,0,0,0.7);
    display: block;
}
.cover-placeholder {
    width: 100%;
    aspect-ratio: 1;
    background: linear-gradient(135deg, #16162A, #0D0D18);
    border-radius: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 64px;
    box-shadow: 0 24px 64px rgba(0,0,0,0.7);
}

/* Infos titre / artiste */
.track-title {
    font-size: 28px;
    font-weight: 800;
    color: #F4F4FC;
    line-height: 1.15;
    letter-spacing: -0.5px;
    margin-bottom: 6px;
}
.track-artist {
    font-size: 16px;
    font-weight: 500;
    color: #8080A0;
    margin-bottom: 14px;
}
.track-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    font-size: 12px;
    color: #505068;
    margin-bottom: 20px;
}
.meta-item { display: flex; align-items: center; gap: 5px; }

/* Boutons streaming */
.streaming-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 4px;
}
.s-btn {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 7px 15px;
    border-radius: 22px;
    font-size: 12px;
    font-weight: 700;
    text-decoration: none;
    letter-spacing: 0.2px;
    transition: all 0.2s;
}
.s-btn:hover { transform: translateY(-2px); filter: brightness(1.25); }
.s-yt  { background: rgba(255,68,68,0.1);   color: #FF4444; border: 1px solid rgba(255,68,68,0.22); }
.s-sp  { background: rgba(29,185,84,0.1);   color: #1DB954; border: 1px solid rgba(29,185,84,0.22); }
.s-dz  { background: rgba(168,86,240,0.1);  color: #A856F0; border: 1px solid rgba(168,86,240,0.22); }
.s-am  { background: rgba(252,60,68,0.1);   color: #FC3C44; border: 1px solid rgba(252,60,68,0.22); }

/* ── Recommandations ───────────────────────────────────────────────────────── */
.reco-section-title {
    font-size: 11px;
    font-weight: 700;
    color: #383848;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 2.5rem 0 1rem;
}
.reco-item {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 12px 14px;
    border-radius: 14px;
    background: rgba(255,255,255,0.018);
    border: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 8px;
    text-decoration: none;
    transition: all 0.2s;
}
.reco-item:hover {
    background: rgba(108,99,255,0.06);
    border-color: rgba(108,99,255,0.2);
    transform: translateX(3px);
}
.reco-cover {
    width: 46px; height: 46px;
    border-radius: 9px;
    object-fit: cover;
    background: #16162A;
    flex-shrink: 0;
}
.reco-texts { flex: 1; min-width: 0; }
.reco-title { font-size: 14px; font-weight: 600; color: #E2E2EA; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.reco-artist { font-size: 12px; color: #505068; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-top: 2px; }
.reco-arrow { color: #333; font-size: 14px; flex-shrink: 0; }

/* ── Candidats incertains ──────────────────────────────────────────────────── */
.candidate {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 16px;
    border-radius: 16px;
    background: rgba(255,193,7,0.03);
    border: 1px solid rgba(255,193,7,0.12);
    margin-bottom: 10px;
    animation: slideUp 0.4s ease;
}
.candidate-rank {
    font-size: 22px;
    font-weight: 900;
    color: rgba(255,184,0,0.25);
    min-width: 28px;
    text-align: center;
}
.candidate-cover {
    width: 52px; height: 52px;
    border-radius: 10px;
    object-fit: cover;
    flex-shrink: 0;
}
.candidate-texts { flex: 1; min-width: 0; }
.candidate-title  { font-size: 15px; font-weight: 700; color: #F0F0FA; margin-bottom: 3px; }
.candidate-artist { font-size: 12px; color: #666; }
.candidate-score {
    font-size: 11px;
    font-weight: 700;
    color: #FFB800;
    background: rgba(255,184,0,0.1);
    border: 1px solid rgba(255,184,0,0.2);
    padding: 3px 10px;
    border-radius: 12px;
    flex-shrink: 0;
}

/* ── Pas de résultat ───────────────────────────────────────────────────────── */
.no-result {
    text-align: center;
    padding: 3.5rem 1rem;
    animation: slideUp 0.4s ease;
}
.no-result-icon  { font-size: 52px; margin-bottom: 16px; line-height: 1; }
.no-result-title { font-size: 20px; font-weight: 700; color: #383848; margin-bottom: 8px; }
.no-result-hint  { font-size: 13px; color: #2E2E40; }

/* ── Footer ────────────────────────────────────────────────────────────────── */
.app-footer {
    text-align: center;
    margin-top: 4rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(255,255,255,0.04);
    color: #2A2A38;
    font-size: 12px;
}
.app-footer a { color: #383848; text-decoration: none; transition: color 0.2s; }
.app-footer a:hover { color: #6C63FF; }

/* ── Toggle langue (petit bouton en haut) ──────────────────────────────────── */
[data-testid="stButton"] button {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: #555 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    padding: 4px 14px !important;
    border-radius: 20px !important;
    transition: all 0.2s !important;
}
[data-testid="stButton"] button:hover {
    border-color: rgba(160,155,255,0.4) !important;
    color: #A09BFF !important;
}
</style>
"""


# ──────────────────────────────────────────────────────────────────────────────
# Icônes SVG plateformes
# ──────────────────────────────────────────────────────────────────────────────

SVG_GITHUB = (
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">'
    '<path d="M12 0C5.37 0 0 5.37 0 12c0 5.3 3.44 9.8 8.2 11.4.6.1.82-.26.82-.57v-2c-3.34.72'
    '-4.04-1.6-4.04-1.6-.54-1.38-1.33-1.75-1.33-1.75-1.08-.74.08-.72.08-.72 1.2.08 1.83 1.23'
    ' 1.83 1.23 1.06 1.82 2.8 1.3 3.47.99.1-.77.42-1.3.76-1.6-2.67-.3-5.47-1.33-5.47-5.93 0'
    '-1.31.47-2.38 1.24-3.22-.14-.3-.54-1.52.1-3.18 0 0 1-.32 3.3 1.23a11.5 11.5 0 0 1 3-.4c'
    '1.02 0 2.04.13 3 .4 2.28-1.55 3.3-1.23 3.3-1.23.64 1.66.24 2.88.12 3.18.77.84 1.23 1.9'
    ' 1.23 3.22 0 4.61-2.8 5.63-5.48 5.92.43.37.81 1.1.81 2.22v3.3c0 .32.22.68.82.56C20.56'
    ' 21.8 24 17.3 24 12c0-6.63-5.37-12-12-12z"/></svg>'
)

SVG_YT = (
    '<svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">'
    '<path d="M23.5 6.2a3 3 0 0 0-2.1-2.1C19.5 3.5 12 3.5 12 3.5s-7.5 0-9.4.6A3 3 0 0 0 .5 '
    '6.2 31 31 0 0 0 0 12a31 31 0 0 0 .5 5.8 3 3 0 0 0 2.1 2.1c1.9.6 9.4.6 9.4.6s7.5 0 '
    '9.4-.6a3 3 0 0 0 2.1-2.1A31 31 0 0 0 24 12a31 31 0 0 0-.5-5.8zM9.7 15.5V8.5l6.3 3.5'
    '-6.3 3.5z"/></svg>'
)

SVG_SP = (
    '<svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">'
    '<path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.52 17.28c-.24'
    '.36-.66.48-1.02.24-2.82-1.74-6.36-2.1-10.56-1.14-.42.12-.78-.18-.9-.54-.12-.42.18-.78'
    '.54-.9 4.56-1.02 8.52-.6 11.64 1.32.42.18.48.66.3 1.02zm1.44-3.3c-.3.42-.84.6-1.26.3'
    '-3.24-1.98-8.16-2.58-11.94-1.38-.48.12-.99-.18-1.11-.66-.12-.48.18-.99.66-1.11 4.38'
    '-1.32 9.78-.66 13.5 1.62.36.24.54.78.15 1.23zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.3c'
    '-.57.18-1.17-.15-1.35-.72-.18-.57.15-1.17.72-1.35 4.26-1.26 11.28-1.02 15.72 1.62.54'
    '.3.72 1.02.42 1.56-.3.42-1.02.6-1.56.3z"/></svg>'
)

SVG_DZ = (
    '<svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">'
    '<path d="M18.9 13.1h2.4v1.4h-2.4zm0-2.3h2.4v1.4h-2.4zm0-2.3h2.4v1.4h-2.4zM2.7 17.7H5v'
    '1.4H2.7zm4 0h2.4v1.4H6.7zm4.1 0h2.4v1.4h-2.4zm4 0h2.4v1.4h-2.4zm4.1 0h2.4v1.4h-2.4z'
    'M10.8 15.4h2.4v1.4h-2.4zm4.1 0h2.4v1.4h-2.4zm4 0H21.3v1.4h-2.4zm-12.2 0h2.4v1.4H6.7z'
    'm-4 0H5v1.4H2.7zm8.1-2.3h2.4v1.4h-2.4zm4.1 0h2.4v1.4h-2.4zm-8.2 0h2.4v1.4H6.7zm8.2'
    '-2.3h2.4v1.4h-2.4zm4.1-4.6h2.4v1.4h-2.4z"/></svg>'
)

SVG_AM = (
    '<svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">'
    '<path d="M23.994 6.124a9.23 9.23 0 0 0-.24-2.19c-.317-1.31-1.062-2.31-2.18-3.043a5.022'
    ' 5.022 0 0 0-1.877-.726A10.496 10.496 0 0 0 18.133.015L18 0H6l-.14.016a10.496 10.496 0 '
    '0 0-1.562.149 5.022 5.022 0 0 0-1.877.726C1.301 1.624.557 2.625.24 3.934A9.16 9.16 0 0 '
    '0 .006 6.124L0 6.25v11.5l.006.126a9.16 9.16 0 0 0 .234 2.19c.317 1.31 1.062 2.31 2.181'
    ' 3.043a5.022 5.022 0 0 0 1.877.726c.528.1 1.06.149 1.562.149L6 24h12l.133-.016a10.496 '
    '10.496 0 0 0 1.562-.149 5.022 5.022 0 0 0 1.877-.726c1.119-.733 1.864-1.733 2.181-3.043'
    'a9.16 9.16 0 0 0 .234-2.19L24 17.75V6.25l-.006-.126zM16.172 7.01l.234.02a1.933 1.933 0 0'
    ' 1 .505.103l-.648 2.378a1.41 1.41 0 0 0-.39-.094l-.195-.014-4.685 1.12v5.538c0 .638-.27'
    ' 1.157-.805 1.55a2.517 2.517 0 0 1-.869.35 2.622 2.622 0 0 1-1.354-.073 2.05 2.05 0 0 '
    '1-.928-.613 1.922 1.922 0 0 1-.427-1.239c0-.61.233-1.11.698-1.507.342-.292.758-.47 1.22'
    '-.54.387-.059.77-.035 1.147.073.178.053.353.12.517.206V9.244c0-.476.33-.892.797-.99l'
    '5.182-1.244z"/></svg>'
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

GITHUB_URL = "https://github.com/thmsgo18/Shazam"


@st.cache_data
def load_metadata() -> dict[str, dict]:
    meta_path = Path("data/processed/metadata.parquet")
    if not meta_path.exists():
        return {}
    df = pd.read_parquet(meta_path)
    result: dict[str, dict] = {}
    for row in df.itertuples():
        result[str(row.track_id)] = {
            "title":        str(getattr(row, "title",        None) or "—"),
            "artist":       str(getattr(row, "artist",       None) or "—"),
            "album":        str(getattr(row, "album",        None) or ""),
            "genre":        str(getattr(row, "genre",        None) or ""),
            "release_date": str(getattr(row, "release_date", None) or ""),
            "cover_url":    str(getattr(row, "cover_url",    None) or ""),
        }
    return result


def _valid(val: str) -> bool:
    return bool(val) and val not in ("None", "nan", "—", "")


def get_recommendations(
    metadata: dict[str, dict],
    exclude_id: str,
    genre: str,
    n: int = 3,
) -> list[dict]:
    if not _valid(genre):
        return []
    pool = [
        info for tid, info in metadata.items()
        if tid != exclude_id and info.get("genre") == genre
    ]
    return random.sample(pool, min(n, len(pool)))


def streaming_links_html(artist: str, title: str) -> str:
    q = urllib.parse.quote(f"{artist} {title}")
    platforms = [
        ("YouTube",     f"https://www.youtube.com/results?search_query={q}", "s-yt", SVG_YT),
        ("Spotify",     f"https://open.spotify.com/search/{q}",              "s-sp", SVG_SP),
        ("Deezer",      f"https://www.deezer.com/search/{q}",                "s-dz", SVG_DZ),
        ("Apple Music", f"https://music.apple.com/search?term={q}",          "s-am", SVG_AM),
    ]
    html = '<div class="streaming-row">'
    for name, url, cls, svg in platforms:
        html += f'<a href="{url}" target="_blank" class="s-btn {cls}">{svg} {name}</a>'
    html += "</div>"
    return html


def cover_html(url: str, css_class: str = "cover-img", placeholder_size: str = "64px") -> str:
    if _valid(url):
        return f'<img src="{url}" class="{css_class}" alt="cover"/>'
    return (
        f'<div class="cover-placeholder" style="font-size:{placeholder_size}">🎵</div>'
    )


def process_audio(audio_bytes: bytes, suffix: str = ".wav") -> list[tuple[str, float]] | None:
    """Sauvegarde en temp, identifie, nettoie. Aucun fichier permanent."""
    tmp_input = wav_path = None
    try:
        # Écriture du fichier source
        fd, tmp_input = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(audio_bytes)

        # Si format vidéo ou format non supporté → extraction audio via ffmpeg
        ext = suffix.lower()
        if ext not in (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"):
            fd2, wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd2)
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", tmp_input,
                    "-vn", "-ar", "22050", "-ac", "1",
                    wav_path,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            source = wav_path
        else:
            source = tmp_input

        return identify_track(
            source,
            method=config.EMBEDDING_METHOD,
            top_n=config.VECTOR_TOP_N_RESULTS,
        )

    except Exception as e:
        st.error(f"Erreur lors de l'identification : {e}")
        return None
    finally:
        for p in [tmp_input, wav_path]:
            if p and Path(p).exists():
                try:
                    os.unlink(p)
                except Exception:
                    pass


# ──────────────────────────────────────────────────────────────────────────────
# Rendu des résultats
# ──────────────────────────────────────────────────────────────────────────────

def render_confident(
    track_id: str,
    info: dict,
    t: dict,
    metadata: dict[str, dict],
) -> None:
    title  = info.get("title",  "—")
    artist = info.get("artist", "—")
    album  = info.get("album",  "")
    genre  = info.get("genre",  "")
    year   = info.get("release_date", "")[:4] if _valid(info.get("release_date", "")) else ""
    cover  = info.get("cover_url", "")

    st.markdown(
        f'<div class="badge badge-ok">✦ {t["confident_hint"]}</div>',
        unsafe_allow_html=True,
    )

    col_cover, col_info = st.columns([1, 1.7], gap="large")

    with col_cover:
        st.markdown(cover_html(cover, "cover-img", "64px"), unsafe_allow_html=True)

    with col_info:
        st.markdown(f'<div class="track-title">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="track-artist">{artist}</div>', unsafe_allow_html=True)

        meta_parts = []
        if _valid(album):
            meta_parts.append(f'<span class="meta-item">💿 {album}</span>')
        if _valid(genre):
            meta_parts.append(f'<span class="meta-item">🎸 {genre}</span>')
        if year:
            meta_parts.append(f'<span class="meta-item">📅 {year}</span>')
        if meta_parts:
            st.markdown(
                f'<div class="track-meta">{"".join(meta_parts)}</div>',
                unsafe_allow_html=True,
            )

        st.markdown(streaming_links_html(artist, title), unsafe_allow_html=True)

    # Recommandations
    recos = get_recommendations(metadata, track_id, genre, n=3)
    if recos:
        st.markdown(
            f'<div class="reco-section-title">♫ {t["recommendations"]}</div>',
            unsafe_allow_html=True,
        )
        for reco in recos:
            r_title  = reco.get("title",  "—")
            r_artist = reco.get("artist", "—")
            r_cover  = reco.get("cover_url", "")
            r_query  = urllib.parse.quote(f"{r_artist} {r_title}")
            img_html = (
                f'<img src="{r_cover}" class="reco-cover" alt="cover"/>'
                if _valid(r_cover)
                else '<div class="reco-cover" style="display:flex;align-items:center;'
                     'justify-content:center;font-size:20px;background:#16162A;">🎵</div>'
            )
            st.markdown(
                f'<a href="https://www.youtube.com/results?search_query={r_query}" '
                f'target="_blank" class="reco-item">'
                f'{img_html}'
                f'<div class="reco-texts">'
                f'<div class="reco-title">{r_title}</div>'
                f'<div class="reco-artist">{r_artist}</div>'
                f'</div>'
                f'<div class="reco-arrow">›</div>'
                f'</a>',
                unsafe_allow_html=True,
            )


def render_uncertain(
    results: list[tuple[str, float]],
    metadata: dict[str, dict],
    t: dict,
) -> None:
    st.markdown(
        f'<div class="badge badge-warn">⚠ {t["uncertain"]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="font-size:13px;color:#444;margin-bottom:1.2rem;">{t["uncertain_hint"]}</p>',
        unsafe_allow_html=True,
    )
    for rank, (track_id, score) in enumerate(results[:3], start=1):
        info   = metadata.get(track_id, {})
        title  = info.get("title",  track_id[:14] + "…")
        artist = info.get("artist", "—")
        cover  = info.get("cover_url", "")
        img_html = (
            f'<img src="{cover}" class="candidate-cover" alt="cover"/>'
            if _valid(cover)
            else '<div class="candidate-cover" style="background:#16162A;display:flex;'
                 'align-items:center;justify-content:center;font-size:22px;">🎵</div>'
        )
        st.markdown(
            f'<div class="candidate">'
            f'<div class="candidate-rank">#{rank}</div>'
            f'{img_html}'
            f'<div class="candidate-texts">'
            f'<div class="candidate-title">{title}</div>'
            f'<div class="candidate-artist">{artist}</div>'
            f'</div>'
            f'<div class="candidate-score">{score:.1f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_no_result(t: dict) -> None:
    st.markdown(
        f'<div class="no-result">'
        f'<div class="no-result-icon">🔇</div>'
        f'<div class="no-result-title">{t["no_result"]}</div>'
        f'<div class="no-result-hint">{t["try_again"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if "lang" not in st.session_state:
        st.session_state.lang = "fr"

    t = TRANSLATIONS[st.session_state.lang]

    # CSS
    st.markdown(CSS, unsafe_allow_html=True)

    # ── Header ─────────────────────────────────────────────────────────────────
    col_header, col_lang = st.columns([5, 1])
    with col_header:
        st.markdown(
            f'<div class="app-header">'
            f'  <div class="header-left">'
            f'    <div class="header-logo">🎵</div>'
            f'    <div class="header-title">{t["title"]}</div>'
            f'  </div>'
            f'  <div class="header-right">'
            f'    <a href="{GITHUB_URL}" target="_blank" class="github-link">'
            f'      {SVG_GITHUB} {t["github"]}'
            f'    </a>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_lang:
        st.markdown("<div style='padding-top:6px'></div>", unsafe_allow_html=True)
        if st.button(t["lang_switch"], key="lang_btn"):
            st.session_state.lang = "en" if st.session_state.lang == "fr" else "fr"
            st.rerun()

    # Metadata
    metadata = load_metadata()

    # ── Zone écoute ─────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="listen-hero">'
        f'  <div class="listen-hero-title">{t["listen_title"]}</div>'
        f'  <div class="listen-hero-hint">{t["listen_hint"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    audio_data = st.audio_input(" ", label_visibility="collapsed", key="mic")

    # ── Séparateur + Upload ──────────────────────────────────────────────────────
    st.markdown(f'<div class="separator">{t["or"]}</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        t["upload_hint"],
        type=["mp3", "wav", "flac", "ogg", "m4a", "aac", "mp4", "mov", "avi", "mkv", "webm"],
        label_visibility="collapsed",
        key="upload",
    )

    # ── Identification ──────────────────────────────────────────────────────────
    source_bytes: bytes | None = None
    source_suffix = ".wav"

    if audio_data is not None:
        source_bytes  = audio_data.read()
        source_suffix = ".wav"
    elif uploaded is not None:
        source_bytes  = uploaded.read()
        source_suffix = Path(uploaded.name).suffix or ".mp3"

    if source_bytes:
        with st.spinner(t["identifying"]):
            results = process_audio(source_bytes, suffix=source_suffix)

        st.markdown('<div class="result-wrap">', unsafe_allow_html=True)

        if not results:
            render_no_result(t)
        else:
            # Seuil de confiance
            confident = True
            if len(results) >= 2 and results[1][1] > 0:
                confident = (results[0][1] / results[1][1]) >= config.UI_CONFIDENCE_RATIO

            if confident:
                render_confident(results[0][0], metadata.get(results[0][0], {}), t, metadata)
            else:
                render_uncertain(results, metadata, t)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Footer ──────────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="app-footer">'
        f'  <a href="{GITHUB_URL}" target="_blank">github.com/thmsgo18/Shazam</a>'
        f'  &nbsp;·&nbsp; {t["project"]}'
        f'</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
