#!/usr/bin/env python3
"""
scripts/start_webapp.py

Démarre l'interface web Shazam (backend FastAPI + frontend React/Vite).

Modes :
    python scripts/start_webapp.py           # mode dev  — hot-reload Vite sur :5173
    python scripts/start_webapp.py --prod    # mode prod — build frontend, tout via FastAPI sur :8000

Options :
    --port PORT    Port du backend FastAPI (défaut : 8000)
    --prod         Build le frontend puis sert tout via FastAPI (mode production)
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT         = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "webapp" / "frontend"
DIST_DIR     = FRONTEND_DIR / "dist"


# ── Helpers ────────────────────────────────────────────────────────────────

def _check_npm() -> None:
    """Vérifie que npm est disponible."""
    try:
        subprocess.run(["npm", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("❌  npm introuvable. Installe Node.js : https://nodejs.org/")
        sys.exit(1)


def _npm_install_if_needed() -> None:
    """Lance `npm install` si node_modules est absent."""
    if not (FRONTEND_DIR / "node_modules").exists():
        print("📦  Installation des dépendances frontend (npm install)...")
        result = subprocess.run(["npm", "install"], cwd=FRONTEND_DIR)
        if result.returncode != 0:
            print("❌  Échec de npm install.")
            sys.exit(1)
        print("✅  Dépendances installées.\n")


def _cleanup(processes: list[subprocess.Popen]) -> None:
    print("\n⏹   Arrêt des serveurs...")
    for p in processes:
        try:
            p.terminate()
        except Exception:
            pass
    time.sleep(0.5)
    for p in processes:
        try:
            if p.poll() is None:
                p.kill()
        except Exception:
            pass


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lance l'interface web Shazam (backend + frontend)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--prod", action="store_true",
        help="Mode production : build le frontend et sert tout via FastAPI",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port du backend FastAPI (défaut : 8000)",
    )
    args = parser.parse_args()

    _check_npm()
    _npm_install_if_needed()

    processes: list[subprocess.Popen] = []

    # ── Gestionnaire de signal (Ctrl+C) ──────────────────────────────────
    def _signal_handler(*_):
        _cleanup(processes)
        sys.exit(0)

    signal.signal(signal.SIGINT,  _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # ── Mode production ──────────────────────────────────────────────────
    if args.prod:
        print("🔨  Build du frontend React...")
        result = subprocess.run(["npm", "run", "build"], cwd=FRONTEND_DIR)
        if result.returncode != 0:
            print("❌  Échec du build frontend.")
            sys.exit(1)
        print("✅  Build terminé.\n")

        print(f"🚀  Démarrage du serveur sur http://localhost:{args.port} …")
        backend = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "webapp.backend.server:app",
                "--host", "0.0.0.0",
                "--port", str(args.port),
            ],
            cwd=ROOT,
        )
        processes.append(backend)

        print(f"\n✅  Interface disponible → http://localhost:{args.port}")
        print("   Ctrl+C pour arrêter\n")

    # ── Mode développement ───────────────────────────────────────────────
    else:
        print(f"🚀  Démarrage du backend FastAPI sur http://localhost:{args.port} …")
        backend = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "webapp.backend.server:app",
                "--host", "0.0.0.0",
                "--port", str(args.port),
                "--reload",
            ],
            cwd=ROOT,
        )
        processes.append(backend)
        time.sleep(1)  # laisser le backend démarrer avant Vite

        print("⚡  Démarrage du frontend Vite (hot-reload)…")
        frontend = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=FRONTEND_DIR,
        )
        processes.append(frontend)

        print(f"\n✅  Interface disponible → http://localhost:5173")
        print(f"   Backend API         → http://localhost:{args.port}")
        print("   Ctrl+C pour arrêter\n")

    # ── Attente des processus ────────────────────────────────────────────
    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        _cleanup(processes)


if __name__ == "__main__":
    main()
