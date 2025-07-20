from __future__ import annotations

import os
import shutil
from pathlib import Path


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
