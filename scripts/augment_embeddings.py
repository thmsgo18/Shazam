"""
scripts/augment_embeddings.py

Data augmentation à l'indexation : génère des versions dégradées (bruit, reverb,
filtre) d'un morceau et les ajoute dans ChromaDB sous le même track_id.

Principe : CLAP a été entraîné sur de l'audio propre. Quand une requête micro
arrive (bruitée), elle ne ressemble à rien dans la base → FAISS ne trouve pas.
En ajoutant des versions dégradées du morceau dans la base, la requête micro
trouve un vecteur "bruité" proche d'elle → Stage 1 (FAISS) fonctionne.

Les embeddings augmentés partagent le même track_id que l'original → ils
s'agrègent correctement dans aggregate_by_track() et font monter le score du
bon morceau.

Usage :
    python scripts/augment_embeddings.py \\
        --audio "data/raw/Miley Cyrus - Flowers (Official Video).mp3" \\
        --track-id f01ab00f1fdc5a57fd2676f4d68631a8

    # Après : reconstruire l'index FAISS
    python src/index/build_index.py

Options :
    --audio     Chemin vers le fichier audio propre du morceau
    --track-id  track_id du morceau dans la base
    --method    Méthode d'embedding (défaut : valeur dans config.py)
    --augs      Types d'augmentation : noise30 noise20 noise10 reverb combo
    --dry-run   Affiche ce qui serait fait sans modifier ChromaDB
    --force     Réécrit les embeddings augmentés même s'ils existent déjà
"""

from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import librosa

from pathlib import Path
from scipy.signal import butter, sosfilt, fftconvolve

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ⚠️ Sur Mac : charger CLAP avant FAISS/ChromaDB (conflit Accelerate framework)
from src.features.embeddings_audio import embed_segment, clap_batch_embeddings, _load_clap
import chromadb
from src import config
from src.audio.loading import load_audio
from src.audio.preprocessing import iter_segments

# ─── Couleurs terminal ────────────────────────────────────────────────────────
G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; C = "\033[96m"; B = "\033[1m"; Z = "\033[0m"


# ═════════════════════════════════════════════════════════════════════════════
# TRANSFORMATIONS D'AUGMENTATION
# ═════════════════════════════════════════════════════════════════════════════

def add_noise(waveform: np.ndarray, snr_db: float, seed: int = 42) -> np.ndarray:
    """Bruit blanc gaussien à un SNR cible (dB)."""
    rng = np.random.default_rng(seed)
    signal_power = np.mean(waveform ** 2)
    if signal_power == 0:
        return waveform
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.standard_normal(len(waveform)) * np.sqrt(noise_power)
    return np.clip(waveform + noise, -1.0, 1.0).astype(np.float32)


def add_reverb(waveform: np.ndarray, sr: int, decay: float = 0.3) -> np.ndarray:
    """Réverbération synthétique (quelques réflexions exponentielles)."""
    rir_len = int(0.5 * sr)
    rir = np.zeros(rir_len)
    for t, g in zip([0, int(0.02*sr), int(0.05*sr), int(0.1*sr), int(0.2*sr)],
                    [1.0, 0.6, 0.4, 0.25, 0.15]):
        if t < rir_len:
            rir[t] = g * (decay ** (t / sr))
    out = fftconvolve(waveform, rir)[:len(waveform)]
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak * np.max(np.abs(waveform))
    return out.astype(np.float32)


def apply_bandpass(waveform: np.ndarray, sr: int,
                   low_hz: float = 300, high_hz: float = 7000) -> np.ndarray:
    """Filtre passe-bande simulant la réponse fréquentielle d'un micro bas de gamme."""
    sos = butter(4, [low_hz, high_hz], btype='band', fs=sr, output='sos')
    return sosfilt(sos, waveform).astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# Catalogue des augmentations disponibles
# Chaque valeur est un callable (waveform, sr) → waveform_augmenté
AUGMENTATIONS: dict[str, tuple[str, callable]] = {
    "noise30": (
        "Bruit blanc SNR=30dB (léger)",
        lambda w, sr: add_noise(w, snr_db=30, seed=30),
    ),
    "noise20": (
        "Bruit blanc SNR=20dB (modéré)",
        lambda w, sr: add_noise(w, snr_db=20, seed=20),
    ),
    "noise10": (
        "Bruit blanc SNR=10dB (fort — seuil de rupture identifié)",
        lambda w, sr: add_noise(w, snr_db=10, seed=10),
    ),
    "reverb": (
        "Réverbération de salle (decay=0.3)",
        lambda w, sr: add_reverb(w, sr, decay=0.3),
    ),
    "combo": (
        "Combo réaliste : bruit SNR=15dB + reverb + filtre micro 300-7kHz",
        lambda w, sr: apply_bandpass(add_reverb(add_noise(w, snr_db=15, seed=15), sr), sr),
    ),
}

ALL_AUG_KEYS = list(AUGMENTATIONS.keys())


# ═════════════════════════════════════════════════════════════════════════════
# CORE
# ═════════════════════════════════════════════════════════════════════════════

def get_collection(method: str):
    """Retourne la collection ChromaDB pour la méthode donnée."""
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    return client.get_or_create_collection(name=method)


def already_augmented(collection, track_id: str, aug_key: str) -> bool:
    """Vérifie si des embeddings augmentés existent déjà pour ce track + aug_key."""
    prefix = f"{track_id}_0_aug_{aug_key}"  # On vérifie juste le premier segment
    result = collection.get(ids=[prefix])
    return len(result["ids"]) > 0


def augment_track(
    audio_path: str,
    track_id: str,
    method: str,
    aug_keys: list[str],
    dry_run: bool = False,
    force: bool = False,
) -> None:
    """
    Charge l'audio, génère les versions augmentées, embed chacune, stocke dans ChromaDB.
    """
    print(f"\n{B}Track{Z}    : {track_id}")
    print(f"{B}Audio{Z}    : {audio_path}")
    print(f"{B}Méthode{Z}  : {method}")
    print(f"{B}Augs{Z}     : {', '.join(aug_keys)}")
    if dry_run:
        print(f"{Y}  [DRY RUN — aucune écriture]{Z}")

    # ── Chargement du sample rate cible ──────────────────────────────────────
    if method == "clap":
        targ_sr = config.CLAP_SAMPLE_RATE
    elif method == "muq":
        targ_sr = config.MUQ_SAMPLE_RATE
    else:
        targ_sr = config.SAMPLE_RATE

    print(f"\n{C}[1/3] Chargement audio → {targ_sr} Hz...{Z}")
    waveform, sr = load_audio(audio_path, target_sr=targ_sr)
    duration = len(waveform) / sr
    print(f"  Durée : {duration:.1f}s  |  Sample rate : {sr} Hz")

    # ── Segmentation ──────────────────────────────────────────────────────────
    segments = [(start_s, seg) for start_s, seg in iter_segments(waveform, sr)]
    n_segs = len(segments)
    print(f"  Segments : {n_segs} × {config.SEGMENT_WIN_S}s (hop={config.SEGMENT_HOP_S}s)")

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    collection = get_collection(method)
    print(f"\n{C}[2/3] Génération et stockage des embeddings augmentés...{Z}")

    total_added = 0

    for aug_key in aug_keys:
        desc, transform = AUGMENTATIONS[aug_key]
        print(f"\n  {B}→ {aug_key}{Z} : {desc}")

        # Vérifier si déjà présent
        if not force and already_augmented(collection, track_id, aug_key):
            print(f"    {Y}Déjà présent dans ChromaDB — skip (utilise --force pour réécrire){Z}")
            continue

        if dry_run:
            print(f"    {Y}[DRY RUN] Générerait {n_segs} segments augmentés{Z}")
            continue

        # Appliquer la transformation sur le waveform entier
        wf_aug = transform(waveform, sr)

        # Re-segmenter le waveform augmenté (même découpage que l'original)
        segs_aug = [seg for _, seg in iter_segments(wf_aug, sr)]

        # Embedder par batch
        if method == "clap" and len(segs_aug) > 1:
            embeddings = clap_batch_embeddings(
                segs_aug, sr,
                model_name=config.CLAP_MODEL_NAME,
            )
        else:
            embeddings = np.vstack([
                embed_segment(seg, sr, method,
                              clap_model_name=config.CLAP_MODEL_NAME,
                              muq_model_name=config.MUQ_MODEL_NAME)
                for seg in segs_aug
            ])

        # IDs uniques : {track_id}_{seg_idx}_aug_{aug_key}
        aug_ids = [f"{track_id}_{i}_aug_{aug_key}" for i in range(len(embeddings))]

        # Supprimer les anciens si --force
        if force:
            try:
                collection.delete(ids=aug_ids)
            except Exception:
                pass

        # Ajouter dans ChromaDB — même track_id dans les metadata → agrégation correcte
        collection.add(
            embeddings=embeddings.tolist(),
            ids=aug_ids,
            metadatas=[
                {"track_id": track_id, "start_s": float(segments[i][0])}
                for i in range(len(embeddings))
            ],
        )

        n = len(embeddings)
        total_added += n
        print(f"    {G}✓ {n} embeddings ajoutés dans ChromaDB (collection '{method}'){Z}")

    # ── Récapitulatif ─────────────────────────────────────────────────────────
    print(f"\n{C}[3/3] Terminé.{Z}")
    if not dry_run:
        print(f"  {G}{total_added} embeddings augmentés ajoutés au total.{Z}")
        if total_added > 0:
            print(f"\n  {B}⚠  N'oublie pas de reconstruire l'index FAISS :{Z}")
            print(f"     python src/index/build_index.py\n")
    else:
        print(f"  {Y}Dry-run : rien n'a été écrit.{Z}")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Data augmentation à l'indexation pour robustesse au bruit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--audio", required=True,
        help="Chemin vers le fichier audio propre du morceau"
    )
    parser.add_argument(
        "--track-id", required=True,
        help="track_id du morceau dans la base (ex: f01ab00f1fdc5a57fd2676f4d68631a8)"
    )
    parser.add_argument(
        "--method", default=config.EMBEDDING_METHOD,
        choices=["mfcc", "clap", "muq"],
        help=f"Méthode d'embedding (défaut : '{config.EMBEDDING_METHOD}' depuis config.py)"
    )
    parser.add_argument(
        "--augs", nargs="+", default=ALL_AUG_KEYS,
        choices=ALL_AUG_KEYS,
        metavar="AUG",
        help=f"Types d'augmentation à appliquer. Choix : {', '.join(ALL_AUG_KEYS)} (défaut : tous)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Affiche ce qui serait fait sans écrire dans ChromaDB"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Réécrit les embeddings augmentés même s'ils existent déjà"
    )

    args = parser.parse_args()

    if not Path(args.audio).exists():
        print(f"{R}Erreur : fichier audio introuvable : {args.audio}{Z}")
        sys.exit(1)

    print(f"\n{B}{C}{'═'*60}{Z}")
    print(f"{B}{C}  DATA AUGMENTATION — Shazam Maison{Z}")
    print(f"{B}{C}{'═'*60}{Z}")
    print(f"\n  Augmentations disponibles :")
    for k, (desc, _) in AUGMENTATIONS.items():
        marker = "→" if k in args.augs else " "
        active = G if k in args.augs else Z
        print(f"  {marker} {active}{k:<10}{Z} : {desc}")

    augment_track(
        audio_path=args.audio,
        track_id=args.track_id,
        method=args.method,
        aug_keys=args.augs,
        dry_run=args.dry_run,
        force=args.force,
    )


if __name__ == "__main__":
    main()
