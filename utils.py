from __future__ import annotations

import os
import shutil
from pathlib import Path
import whisper


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

def generate_srt_file(audio_path : str) -> str:
    # Charger le modèle Whisper (base = rapide)
    model = whisper.load_model("base")
    # Transcrire la vidéo (audio en texte)
    result = model.transcribe(audio_path, fp16=False)

    file_path = os.path.join(Path(audio_path).parent, 'subtitles.srt')
    # Sauvegarder en SRT
    with open(file_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result["segments"]):
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()

            def format_time(t):
                h, rem = divmod(t, 3600)
                m, s = divmod(rem, 60)
                ms = int((s - int(s)) * 1000)
                return f"{int(h):02}:{int(m):02}:{int(s):02},{ms:03}"

            f.write(f"{i+1}\n{format_time(start)} --> {format_time(end)}\n{text}\n\n")
    print(file_path)
    return file_path

if __name__ == "__main__":
    generate_srt_file('/Users/matys/Downloads/test.mp4')