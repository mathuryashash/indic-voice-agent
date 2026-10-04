"""Load API keys from local .env files into os.environ without printing them."""
from __future__ import annotations

import os
from pathlib import Path

KEY_FILES = [
    Path(__file__).resolve().parent.parent / ".env",  # project-local (gitignored)
    Path("D:/aihackthonforbuilder/backend/.env"),
    Path("D:/ascent_hackathon/.env"),
    Path.home() / "AppData/Local/hermes/.env",
]
WANTED = ("GROQ_API_KEY", "NVIDIA_API_KEY", "NVIDIA_BASE_URL")


def load_keys(files: list[Path] | None = None) -> list[str]:
    """Return the names (never values) of variables that were loaded."""
    loaded = []
    for f in files or KEY_FILES:
        try:
            lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for line in lines:
            k, sep, v = line.strip().partition("=")
            k = k.strip().removeprefix("export ").strip()
            v = v.strip().strip('"').strip("'")
            if sep and k in WANTED and v and not os.environ.get(k):
                os.environ[k] = v
                loaded.append(k)
    return loaded
