from __future__ import annotations

import os
import shutil
from pathlib import Path
import whisper
from transformers import pipeline
import asyncio
import pysubs2

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def get_downloads_dir() -> Path:
    """Return the user's Downloads folder, falling back to creating it."""
    home = Path(os.path.expanduser("~"))
    for name in ("Downloads", "T\xe9l\xe9chargements"):
        candidate = home / name
        if candidate.is_dir():
            return candidate
    downloads = home / "Downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    return downloads


def safe_rmtree(path: str | Path) -> None:
    """Remove a directory tree without raising if it fails."""
    try:
        shutil.rmtree(path)
    except OSError:
        pass

def format_time(t: float) -> str:
    """
    Formate un temps en secondes au format SRT (HH:MM:SS,mmm).
    """
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    ms = int((s - int(s)) * 1000)
    return f"{int(h):02}:{int(m):02}:{int(s):02},{ms:03}"

async def generate_styled_subtitles(audio_path: str) -> str:
    """
    Transcrit un fichier audio avec Whisper et génère des sous-titres stylisés au format ASS.

    :param audio_path: Chemin vers le fichier audio.
    :return: Chemin vers le fichier ASS généré.
    """
    # Charger le modèle Whisper (base = rapide)
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, fp16=False)

    # Créer les sous-titres à partir du résultat Whisper
    subs = pysubs2.load_from_whisper(result)

    # Appliquer le style
    subs.styles["Default"].fontname = "Arial"
    subs.styles["Default"].fontsize = 10
    subs.styles["Default"].primarycolor = pysubs2.Color(255, 255, 255, 0)
    subs.styles["Default"].outlinecolor = pysubs2.Color(0, 0, 0, 0)
    subs.styles["Default"].alignment = pysubs2.Alignment.BOTTOM_CENTER
    subs.styles["Default"].margin_v = 50


    # Ajouter un effet d'apparition mot par mot avec les balises \K (sans disparition) et \1c&H808080& pour mots déjà affichés
    for line in subs:
        mots = line.text.split()
        nb_mots = len(mots)
        duree = int((line.end - line.start) / max(nb_mots, 1) / 10)

        texte_k = r"{\1c&H808080&}"  # Couleur après affichage (gris clair)
        for mot in mots:
            texte_k += f"{{\\K{duree}}}{mot} "
        line.text = texte_k.strip()

    # Chemin du fichier ASS
    ass_path = os.path.join(Path(audio_path).parent, 'subtitles.ass')
    subs.save(ass_path)

    print(ass_path)
    return ass_path
