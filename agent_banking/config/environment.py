"""Load project-local secrets once; process environment always takes precedence."""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


def load_environment():
    path = ROOT_DIR / ".env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


load_environment()
